"""Convert the qleap FX OHLCV parquet snapshot into Astra's managed CSV format.

The snapshot is a source representation, not verified training data: its prices
differ from Astra's curated Dukascopy-node extraction, so converted output is
kept in its own dataset root and never stitched across sources. Timestamps are
candle-close UTC, matching Astra's semantics.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..data import validate
from ..history import TIMEFRAMES
from ..models import Candle, MarketData
from .pipeline import _OUTPUTS, _write_candles
from .resampler import resample
from .validation import _bar_crosses_closed_market, inspect_csv

REPORT_STEM = "qleap-parquet"


def _read_parquet_candles(path: Path, symbol: str) -> tuple[Candle, ...]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    stamps = table.column("timestamp").to_pylist()
    candles = []
    for stamp, opening, closing, low, high, volume in zip(
        stamps,
        table.column("Open").to_pylist(),
        table.column("Close").to_pylist(),
        table.column("Low").to_pylist(),
        table.column("High").to_pylist(),
        table.column("Volume").to_pylist(),
    ):
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        else:
            stamp = stamp.astimezone(UTC)
        candles.append(
            Candle(
                stamp,
                float(opening),
                float(high),
                float(low),
                float(closing),
                float(volume),
                "provider",
            )
        )
    return tuple(candles)


def process_qleap_parquet(
    source_dir: str | Path,
    symbol: str,
    root: str | Path,
    *,
    max_gap_minutes: int = 1,
    expected_start: datetime | None = None,
    expected_end: datetime | None = None,
    clip_to_range: bool = False,
) -> dict:
    """Convert one pair's qleap 1min parquet files into validated Astra CSVs.

    Dukascopy-node/qleap timestamps are naive UTC candle-close minutes; Astra
    stores the same semantics. Closed-session bars (FX weekend) are excluded and
    counted, flat zero-volume bars are preserved.
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

    source_dir = Path(source_dir)
    paths = sorted(source_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no parquet files found under {source_dir}")

    unique: dict[datetime, Candle] = {}
    duplicate_count = 0
    source_rows = 0
    closed_session_rows = 0
    source_hashes = {}
    for path in paths:
        raw = path.read_bytes()
        source_hashes[path.as_posix()] = hashlib.sha256(raw).hexdigest()
        candles = _read_parquet_candles(path, symbol)
        source_rows += len(candles)
        for candle in candles:
            if _bar_crosses_closed_market(
                candle.timestamp - timedelta(minutes=1), timedelta(minutes=1)
            ):
                closed_session_rows += 1
                continue
            if clip_to_range:
                assert expected_start is not None and expected_end is not None
                if not expected_start < candle.timestamp <= expected_end:
                    continue
            previous = unique.get(candle.timestamp)
            if previous is not None:
                if previous != candle:
                    raise ValueError(
                        f"conflicting duplicate candle at {candle.timestamp.isoformat()}"
                    )
                duplicate_count += 1
            else:
                unique[candle.timestamp] = candle

    ordered = tuple(unique[stamp] for stamp in sorted(unique))
    validate(MarketData(symbol, "M1", ordered), min_candles=1)
    root = Path(root)
    m1_path = root / "normalized" / "m1" / f"{symbol}.csv"
    _write_candles(m1_path, ordered, volume_unit="source_units")

    reports = {}
    outputs: dict[str, dict] = {}
    for timeframe in _OUTPUTS:
        if timeframe == "M1":
            path = m1_path
        else:
            path = root / "processed" / timeframe.lower() / f"{symbol}.csv"
            _write_candles(
                path, resample(ordered, timeframe), volume_unit="source_units"
            )
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
        "source": "qleap/fx_majors_tick_ohlcv (Dukascopy-attributed, independently unverified)",
        "symbol": symbol,
        "requested_start": expected_start.isoformat() if expected_start else None,
        "requested_end_exclusive": expected_end.isoformat() if expected_end else None,
        "source_paths": [path.as_posix() for path in paths],
        "source_sha256": source_hashes,
        "source_rows": source_rows,
        "excluded_closed_session_rows": closed_session_rows,
        "unique_m1_rows": len(ordered),
        "duplicate_rows": duplicate_count,
        "zero_volume_rows": sum(candle.volume == 0 for candle in ordered),
        "volume_type": "provider",
        "volume_unit": "source_units (source-declared; not independently verified)",
        "timestamp_conversion": "naive UTC candle-close minute, interpreted as UTC",
        "provenance": "converted qleap parquet snapshot; not independently authenticated",
        "outputs": outputs,
        "limitations": [
            "Prices and volumes differ from Astra's curated Dukascopy-node extraction; do not mix sources within one dataset.",
            "Source provenance is dataset-card attribution only, not independent price authentication.",
        ],
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / f"{symbol.lower()}-{REPORT_STEM}.json"
    report_text = report_dir / f"{symbol.lower()}-{REPORT_STEM}.txt"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_text.write_text(
        "\n".join(
            f"{timeframe}: {item['status']}; {item['rows']} rows; "
            f"{len(item['suspicious_gaps'])} suspicious gaps; "
            f"{item['expected_gaps']} expected gaps"
            for timeframe, item in outputs.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "symbol": symbol,
        "source_rows": source_rows,
        "unique_m1_rows": len(ordered),
        "excluded_closed_session_rows": closed_session_rows,
        "duplicate_rows": duplicate_count,
        "timeframes": {key: value["status"] for key, value in outputs.items()},
        "report_json": report_json.as_posix(),
        "report_text": report_text.as_posix(),
    }
