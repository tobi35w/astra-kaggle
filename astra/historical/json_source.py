"""Process Dukascopy-node M1 JSON exports as a local historical dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..data import validate
from ..history import TIMEFRAMES
from ..models import Candle, MarketData
from .pipeline import _OUTPUTS, _write_candles
from .resampler import resample
from .validation import _bar_crosses_closed_market, inspect_csv


def process_dukascopy_json(
    source_path: str | Path | Iterable[str | Path],
    symbol: str,
    root: str | Path,
    *,
    max_gap_minutes: int = 1,
    expected_start: datetime | None = None,
    expected_end: datetime | None = None,
    clip_to_range: bool = False,
) -> dict:
    """Convert a Dukascopy-node JSON export into validated Astra CSVs and reports.

    Dukascopy-node timestamps are Unix milliseconds at the M1 candle open;
    Astra stores UTC candle close/availability timestamps. Flat records are
    preserved, including zero-volume records, so callers should download with
    the CLI ``--flats`` option.
    """
    symbol = symbol.upper()
    if len(symbol) != 6 or not symbol.isalpha():
        raise ValueError("symbol must be a six-letter FX pair")
    if max_gap_minutes < 0:
        raise ValueError("max_gap_minutes cannot be negative")
    if clip_to_range and (
        expected_start is None
        or expected_end is None
        or expected_start.tzinfo is None
        or expected_end.tzinfo is None
        or expected_start >= expected_end
    ):
        raise ValueError("clipping requires an explicit aware start and end")

    source_paths = (
        (Path(source_path),)
        if isinstance(source_path, (str, Path))
        else tuple(Path(path) for path in source_path)
    )
    if not source_paths:
        raise ValueError("at least one Dukascopy JSON source file is required")

    unique: dict[datetime, Candle] = {}
    duplicate_count = 0
    source_rows = 0
    closed_session_rows = 0
    source_hashes = {}
    for path in source_paths:
        raw = path.read_bytes()
        source_hashes[path.as_posix()] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw.decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Dukascopy JSON must contain a nonempty candle array")
        source_rows += len(rows)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"invalid candle row {index + 1}")
            try:
                raw_stamp = row["timestamp"]
                if not isinstance(raw_stamp, int) or isinstance(raw_stamp, bool):
                    raise TypeError("timestamp must be integer Unix milliseconds")
                stamp = datetime.fromtimestamp(raw_stamp / 1000, UTC)
                if stamp.second or stamp.microsecond:
                    raise ValueError("timestamp is not aligned to a minute")
                if _bar_crosses_closed_market(stamp, timedelta(minutes=1)):
                    closed_session_rows += 1
                    continue
                stamp += timedelta(minutes=1)
                candle = Candle(
                    stamp,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                    "provider",
                )
            except (KeyError, OverflowError, OSError, TypeError, ValueError) as error:
                raise ValueError(f"invalid Dukascopy candle row {index + 1}") from error
            if clip_to_range:
                assert expected_start is not None and expected_end is not None
                if not expected_start < stamp <= expected_end:
                    continue
            previous = unique.get(stamp)
            if previous is not None:
                if previous != candle:
                    raise ValueError(
                        f"conflicting duplicate candle at {stamp.isoformat()}"
                    )
                duplicate_count += 1
            else:
                unique[stamp] = candle

    candles = tuple(unique[stamp] for stamp in sorted(unique))
    validate(MarketData(symbol, "M1", candles), min_candles=1)
    root = Path(root)
    m1_path = root / "normalized" / "m1" / f"{symbol}.csv"
    _write_candles(m1_path, candles, volume_unit="millions")

    reports = {}
    outputs = {}
    for timeframe in _OUTPUTS:
        if timeframe == "M1":
            path = m1_path
        else:
            path = root / "processed" / timeframe.lower() / f"{symbol}.csv"
            _write_candles(path, resample(candles, timeframe), volume_unit="millions")
        result = inspect_csv(
            path,
            symbol,
            timeframe,
            max_gap_minutes=max_gap_minutes,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        reports[timeframe] = result
        outputs[timeframe] = {
            "path": path.as_posix(),
            "rows": result.rows,
            "status": result.status,
            "expected_gaps": result.expected_gap_count,
            "suspicious_gaps": [
                {
                    "previous_close_utc": gap.previous.isoformat(),
                    "next_close_utc": gap.next.isoformat(),
                    "missing_minutes": gap.missing_minutes,
                    "missing_candles": gap.missing_minutes
                    // int(TIMEFRAMES[timeframe].total_seconds() // 60),
                }
                for gap in result.suspicious_gaps
            ],
            "checksum_sha256": result.checksum,
        }

    report = {
        "status": "PASS"
        if all(r.status == "PASS" for r in reports.values())
        else "DATA QUALITY INSUFFICIENT",
        "source": "Dukascopy-node M1 JSON, BID",
        "symbol": symbol,
        "requested_start": expected_start.isoformat() if expected_start else None,
        "requested_end_exclusive": expected_end.isoformat() if expected_end else None,
        "source_paths": [path.as_posix() for path in source_paths],
        "source_sha256": source_hashes,
        "source_rows": source_rows,
        "excluded_closed_session_rows": closed_session_rows,
        "unique_m1_rows": len(candles),
        "duplicate_rows": duplicate_count,
        "zero_volume_rows": sum(candle.volume == 0 for candle in candles),
        "volume_type": "provider",
        "volume_unit": "millions (dukascopy-node 1.50.0 default)",
        "timestamp_conversion": "UTC Unix-millisecond candle-open + 1 minute = candle close/availability",
        "outputs": outputs,
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / f"{symbol.lower()}-dukascopy-json.json"
    report_text = report_dir / f"{symbol.lower()}-dukascopy-json.txt"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_text.write_text(
        "\n".join(
            f"{timeframe}: {item.status}; {item.rows} rows; "
            f"{len(item.suspicious_gaps)} suspicious gaps; "
            f"{item.expected_gap_count} expected gaps"
            for timeframe, item in reports.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "symbol": symbol,
        "source_rows": source_rows,
        "unique_m1_rows": len(candles),
        "excluded_closed_session_rows": closed_session_rows,
        "duplicate_rows": duplicate_count,
        "zero_volume_rows": report["zero_volume_rows"],
        "timeframes": {key: value.status for key, value in reports.items()},
        "report_json": report_json.as_posix(),
        "report_text": report_text.as_posix(),
    }
