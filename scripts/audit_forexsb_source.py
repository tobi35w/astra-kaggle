"""Audit saved ForexSB spread-record M1 archives without admitting a dataset.

This is an offline research tool. It never downloads, fills missing prices,
changes source files, or writes a managed dataset manifest. The 28-byte layout
is taken from the publisher's DataFeed.parseResponse JavaScript. Older 24-byte
archives require a separate reviewed decoder; no heuristic fallback is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import struct
import sys
import zlib
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.adapters import SQLiteRepository
from astra.historical.downloader import _file, iter_days
from astra.historical.dukascopy import decode_m1_bi5, price_scale_for
from astra.historical.resampler import resample
from astra.historical.validation import _market_open, inspect_candles
from astra.models import Candle
from astra.repositories import Entity, Journal, json_value

PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4")
EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
RECORD = struct.Struct("<7i")
MAX_COMPRESSED = 16 * 1024 * 1024
MAX_DECOMPRESSED = 128 * 1024 * 1024
PRICE_FIELDS = ("open", "high", "low", "close")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def decode_archive(payload: bytes, symbol: str, metadata: dict) -> tuple[Candle, ...]:
    """Keep raw provider volume and convert UTC minute opens to closes once."""
    scale = metadata.get("priceScale")
    if (
        symbol not in PAIRS
        or metadata.get("symbol") != symbol
        or type(scale) is not int
        or scale != price_scale_for(symbol)
    ):
        raise ValueError("unsupported symbol or inconsistent source price scale")
    if not payload or len(payload) > MAX_COMPRESSED:
        raise ValueError("archive is empty or exceeds the compressed size limit")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as archive:
            raw = archive.read(MAX_DECOMPRESSED + 1)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("archive failed gzip integrity verification") from error
    if not raw or len(raw) > MAX_DECOMPRESSED or len(raw) % RECORD.size:
        raise ValueError("archive has an unsupported size or record layout")
    candles = []
    previous = None
    for minutes, opening, high, low, closing, volume, spread in RECORD.iter_unpack(raw):
        if previous is not None and minutes <= previous:
            raise ValueError("source timestamps must be ordered and unique")
        if not 0 < low <= min(opening, closing) <= max(opening, closing) <= high:
            raise ValueError("source contains invalid OHLC prices")
        if volume < 0 or spread < 0:
            raise ValueError("source contains negative volume or spread")
        previous = minutes
        candles.append(
            Candle(
                EPOCH + MINUTE * (minutes + 1),
                opening / scale,
                high / scale,
                low / scale,
                closing / scale,
                volume,
                "provider",
            )
        )
    return tuple(candles)


def expected_closes(start: datetime, end: datetime) -> set[datetime]:
    """Use Astra's existing DST-aware FX calendar and source-open [start, end)."""
    return {
        opening + MINUTE
        for i in range(int((end - start) / MINUTE))
        if _market_open(opening := start + MINUTE * i)
    }


def compare_reference(candles, symbol, reference_root, manifest, start, end) -> dict:
    """Compare only checksum-verified original BID files; never merge records."""
    observed = {c.timestamp: c for c in candles}
    matched = exact_ohlc = exact_close = omitted_flat = omitted_other = 0
    close_diff_total = close_diff_max = 0.0
    files, differences = [], []
    pip = 0.01 if symbol.endswith("JPY") else 0.0001
    for day in iter_days(start.date(), (end - MINUTE).date()):
        key = f"{symbol}:{day.isoformat()}"
        entry = manifest.get("downloads", {}).get(key, {})
        if entry.get("status") != "downloaded":
            continue
        path = _file(reference_root, symbol, day)
        path.resolve().relative_to(reference_root.resolve())
        if Path(entry.get("path", "")) != path.relative_to(reference_root):
            raise ValueError("reference manifest path mismatch")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry.get("checksum"):
            raise ValueError("reference checksum mismatch")
        original = decode_m1_bi5(payload, symbol, day)
        if len(original) != entry.get("rows"):
            raise ValueError("reference row count mismatch")
        files.append({"unit": key, "path": str(path), "sha256": digest})
        for candle in original:
            if not start <= candle.timestamp - MINUTE < end:
                continue
            if not _market_open(candle.timestamp - MINUTE):
                continue
            alternate = observed.get(candle.timestamp)
            if alternate is None:
                if (
                    candle.volume == 0
                    and len({getattr(candle, field) for field in PRICE_FIELDS}) == 1
                ):
                    omitted_flat += 1
                else:
                    omitted_other += 1
                continue
            matched += 1
            old_prices = [getattr(candle, field) for field in PRICE_FIELDS]
            new_prices = [getattr(alternate, field) for field in PRICE_FIELDS]
            exact_ohlc += old_prices == new_prices
            exact_close += candle.close == alternate.close
            delta = abs(candle.close - alternate.close) / pip
            close_diff_total += delta
            close_diff_max = max(close_diff_max, delta)
            if old_prices != new_prices and len(differences) < 20:
                differences.append(
                    {
                        "close_utc": candle.timestamp,
                        "reference_bid_ohlc": old_prices,
                        "forexsb_ohlc": new_prices,
                    }
                )
    return {
        "matched_minutes": matched,
        "exact_ohlc_rows": exact_ohlc,
        "exact_close_rows": exact_close,
        "close_mae_pips": close_diff_total / matched if matched else None,
        "close_max_abs_pips": close_diff_max if matched else None,
        "omitted_reference_flat_zero_volume_minutes": omitted_flat,
        "omitted_reference_other_minutes": omitted_other,
        "difference_examples": differences,
        "verified_reference_files": files,
    }


def audit_pair(source_root, reference_root, symbol, metadata, manifest, start, end):
    path = source_root / f"forexsb-{symbol}1.lb.gz"
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError("archive exceeds the compressed size limit")
    payload = path.read_bytes()
    all_candles = decode_archive(payload, symbol, metadata[symbol])
    candles = tuple(c for c in all_candles if start <= c.timestamp - MINUTE < end)
    expected = expected_closes(start, end)
    actual = {c.timestamp for c in candles}
    missing = sorted(expected - actual)
    required_by_day = Counter((stamp - MINUTE).date() for stamp in expected)
    present_by_day = Counter((stamp - MINUTE).date() for stamp in expected & actual)
    daily = [
        {
            "day": day.isoformat(),
            "required_minutes": count,
            "observed_minutes": present_by_day[day],
            "missing_minutes": count - present_by_day[day],
        }
        for day, count in sorted(required_by_day.items())
    ]
    frames = {}
    for timeframe in TIMEFRAMES:
        bars = candles if timeframe == "M1" else resample(candles, timeframe)
        validation = inspect_candles(
            bars, symbol, timeframe, expected_start=start, expected_end=end
        )
        frames[timeframe] = json_value(validation)
        frames[timeframe]["missing_open_minutes"] = sum(
            gap.missing_minutes for gap in validation.suspicious_gaps
        )
    return {
        "symbol": symbol,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload),
        "source_rows": len(all_candles),
        "first_source_open_utc": all_candles[0].timestamp - MINUTE,
        "last_source_open_utc": all_candles[-1].timestamp - MINUTE,
        "scope_rows": len(candles),
        "required_open_minutes": len(expected),
        "missing_open_minutes": len(missing),
        "outside_session_rows": len(actual - expected),
        "required_open_days": len(daily),
        "observed_open_days": sum(row["observed_minutes"] > 0 for row in daily),
        "fully_complete_open_days": sum(row["missing_minutes"] == 0 for row in daily),
        "daily_coverage": daily,
        "missing_open_utc": [stamp - MINUTE for stamp in missing],
        "timeframes": frames,
        "reference_overlap": compare_reference(
            candles, symbol, reference_root, manifest, start, end
        ),
    }


def run_audit(
    source_root, reference_root, output_root, start_date, end_date, *, at=None
):
    """Write a unique audit run and append its evidence through Journal."""
    source_root, reference_root, output_root = (
        Path(p).resolve() for p in (source_root, reference_root, output_root)
    )
    if output_root == reference_root or reference_root in output_root.parents:
        raise ValueError("audit output must be separate from the reference dataset")
    if end_date < start_date:
        raise ValueError("end date must not precede start date")
    if (end_date - start_date).days >= 366:
        raise ValueError("one audit is limited to 366 source dates")
    at = at or datetime.now(UTC)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("audit time must be timezone-aware")
    start = datetime.combine(start_date, datetime.min.time(), UTC)
    end = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), UTC)
    if end > at:
        raise ValueError("audit requires a completed source date range")
    manifest_path = reference_root / "manifests" / "dukascopy_manifest.json"
    info_path = source_root / "forexsb-info.json"
    inputs = {str(p): sha256(p) for p in (manifest_path, info_path)}
    for optional in (
        reference_root / "acquisition-summary.json",
        source_root / "forexsb-app.js",
        source_root / "forexsb-info-response.bin",
    ):
        if optional.is_file():
            inputs[str(optional)] = sha256(optional)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    pairs = [
        audit_pair(source_root, reference_root, p, metadata, manifest, start, end)
        for p in PAIRS
    ]
    for item in pairs:
        inputs[item["source_path"]] = item["source_sha256"]
        for reference in item["reference_overlap"]["verified_reference_files"]:
            inputs[reference["path"]] = reference["sha256"]
    if any(sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("an input changed during the audit")
    run_id = at.astimezone(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    report = json_value(
        {
            "run_id": run_id,
            "audit_at": at,
            "evidence_kind": "OFFLINE_AUDIT_OF_SUPPLIED_ARCHIVES",
            "source": "ForexSB public Dukascopy-derived distribution",
            "source_url": "https://data.forexsb.com/",
            "scope": {"start_open_utc": start, "end_open_exclusive_utc": end},
            "record_layout": "seven little-endian int32 fields; 28 bytes per row",
            "timestamp_conversion": "UTC minutes since 2000-01-01; add one minute for close/availability",
            "price_basis": "not asserted by decoder; compared with original Dukascopy BID",
            "volume_semantics": "raw provider integer retained; units unverified; no rounding or zero substitution",
            "spread_semantics": "nonnegative raw points validated; not an execution-cost model",
            "replacement_approved": False,
            "managed_dataset_import_performed": False,
            "training_or_evaluation_performed": False,
            "input_sha256": inputs,
            "inputs_unchanged_after_audit": True,
            "pairs": pairs,
            "limitations": [
                "An audit does not authenticate a publisher or admit data to evaluation.",
                "Date presence does not prove complete open-session minute coverage.",
                "Source representations remain separate; overlapping matches do not authorize splicing.",
                "Missing observations are never filled; incomplete derived bars are omitted.",
                "Only the requested window is used for coverage, resampling and overlap comparison.",
                "No trading outcomes, holdout reservation or model promotion were assessed.",
            ],
        }
    )
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    repository = SQLiteRepository(str(output / "audit.sqlite3"))
    try:
        record = Journal(repository).record(
            Entity.AUDIT, at, {"event": "forexsb_source_audit", **report}
        )
        if not any(
            r.record_id == record.record_id for r in repository.query(Entity.AUDIT)
        ):
            raise RuntimeError("audit journal readback failed")
        report["journal_record_id"] = record.record_id
    finally:
        repository.close()
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    args = parser.parse_args(argv)
    output, report = run_audit(
        args.source_root, args.reference_root, args.output_root, args.start, args.end
    )
    print(
        json.dumps(
            {
                "report": str(output / "audit.json"),
                "replacement_approved": report["replacement_approved"],
                "missing_open_minutes": {
                    pair["symbol"]: pair["missing_open_minutes"]
                    for pair in report["pairs"]
                },
            }
        )
    )
    return 0  # Successful research audit, not successful dataset admission.


if __name__ == "__main__":
    raise SystemExit(main())
