"""Run offline integrity and training-readiness checks on the staged FX data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
REQUIRED = ("Time", "Open", "High", "Low", "Close", "Volume", "timestamp_ms")


def scalar_sum(values: pa.Array) -> int:
    value = pc.sum(values).as_py()
    return int(value or 0)


def audit_file(item: dict) -> dict:
    source_path = DATA_ROOT / "four-pairs-source-parquet" / item["source_file"]
    labeled_path = DATA_ROOT / item["labeled_file"]
    source = pq.read_table(source_path)
    labeled = pq.read_table(labeled_path)
    issues = []

    if not set(REQUIRED).issubset(source.column_names):
        issues.append("missing required source columns")
    if source.num_rows != item["rows"] or labeled.num_rows != source.num_rows:
        issues.append("row count differs from curation manifest")
    if source.column_names != [
        name for name in labeled.column_names if name not in ("Symbol", "Timeframe")
    ]:
        issues.append("labeled schema differs from source schema")
    elif not source.equals(labeled.select(source.column_names)):
        issues.append("labeled parquet changed source values")
    if set(labeled["Symbol"].to_pylist()) != {item["symbol"]}:
        issues.append("symbol provenance label mismatch")
    if set(labeled["Timeframe"].to_pylist()) != {item["timeframe"]}:
        issues.append("timeframe provenance label mismatch")

    rows = source.num_rows
    nulls = {
        name: source[name].null_count
        for name in REQUIRED
        if name in source.column_names
    }
    if any(nulls.values()):
        issues.append("nulls present in required columns")
    timestamp_type = source["Time"].type
    if not pa.types.is_timestamp(timestamp_type) or timestamp_type.tz != "UTC":
        issues.append(f"Time is not a UTC timestamp ({timestamp_type})")

    prices = {
        name: source[name].to_numpy(zero_copy_only=False)
        for name in ("Open", "High", "Low", "Close")
    }
    finite = np.logical_and.reduce([np.isfinite(array) for array in prices.values()])
    invalid_ohlc = int(
        np.count_nonzero(
            finite
            & (
                (prices["High"] < prices["Low"])
                | (prices["Open"] < prices["Low"])
                | (prices["Open"] > prices["High"])
                | (prices["Close"] < prices["Low"])
                | (prices["Close"] > prices["High"])
            )
        )
    )
    nonfinite_price_rows = int(rows - np.count_nonzero(finite))
    nonpositive_price_rows = int(
        np.count_nonzero(
            finite & np.logical_or.reduce([array <= 0 for array in prices.values()])
        )
    )
    if invalid_ohlc:
        issues.append("OHLC bounds violated")
    if nonfinite_price_rows:
        issues.append("nonfinite OHLC values")
    if nonpositive_price_rows:
        issues.append("nonpositive OHLC values")

    volume = source["Volume"].to_numpy(zero_copy_only=False)
    negative_volume = int(np.count_nonzero(volume < 0))
    if negative_volume:
        issues.append("negative volume")

    times_ms = source["timestamp_ms"].to_numpy(zero_copy_only=False)
    deltas = np.diff(times_ms)
    timeframe = item["timeframe"]
    expected_ms = TIMEFRAME_MINUTES[timeframe] * 60_000
    timestamp_ms_match = np.array_equal(
        source["Time"].cast(pa.timestamp("ms", tz="UTC")).cast(pa.int64()).to_numpy(),
        times_ms,
    )
    if not timestamp_ms_match:
        issues.append("Time and timestamp_ms disagree")
    duplicate_count = int(np.count_nonzero(deltas == 0))
    out_of_order = int(np.count_nonzero(deltas < 0))
    if duplicate_count:
        issues.append("duplicate timestamps")
    if out_of_order:
        issues.append("timestamps out of order")
    large_gaps = deltas[deltas > expected_ms * 1.5]
    estimated_missing_intervals = int(
        np.maximum(np.ceil(deltas / expected_ms) - 1, 0).sum()
    )

    return {
        "source_file": item["source_file"],
        "symbol": item["symbol"],
        "timeframe": timeframe,
        "rows": rows,
        "first_time_utc": str(source["Time"][0].as_py()),
        "last_time_utc": str(source["Time"][-1].as_py()),
        "null_counts": nulls,
        "nonfinite_price_rows": nonfinite_price_rows,
        "nonpositive_price_rows": nonpositive_price_rows,
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "duplicate_timestamps": duplicate_count,
        "out_of_order_timestamps": out_of_order,
        "time_matches_timestamp_ms": timestamp_ms_match,
        "gaps_over_1_5_expected_intervals": len(large_gaps),
        "estimated_missing_intervals_before_calendar_review": estimated_missing_intervals,
        "largest_gap_minutes": round(int(deltas.max()) / 60_000, 3)
        if len(deltas)
        else 0,
        "issues": issues,
        "structural_pass": not issues,
    }


def main() -> None:
    manifest_path = DATA_ROOT / "four-pairs-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [audit_file(item) for item in manifest["files"]]
    dataset = load_from_disk(str(DATA_ROOT / "four-pairs-dataset"))
    expected_rows = sum(item["rows"] for item in entries)
    dataset_check = {
        "rows_match_manifest": len(dataset) == expected_rows == manifest["rows"],
        "has_symbol_and_timeframe": {"Symbol", "Timeframe"}.issubset(
            dataset.column_names
        ),
        "symbols": sorted(set(dataset.unique("Symbol"))),
        "timeframes": sorted(set(dataset.unique("Timeframe"))),
        "columns": dataset.column_names,
        "sample_rows": dataset.select([0, len(dataset) - 1]).to_list(),
    }
    if (
        not dataset_check["rows_match_manifest"]
        or not dataset_check["has_symbol_and_timeframe"]
    ):
        raise ValueError("Saved combined dataset fails manifest/schema validation")

    full_manifest = json.loads(
        (DATA_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    overlap_path = DATA_ROOT / "overlap-comparison.json"
    overlap_comparison = json.loads(overlap_path.read_text(encoding="utf-8"))
    overlap_rows = sum(
        item["matched_timestamps"] for item in overlap_comparison["comparisons"]
    )
    exact_rows = sum(
        window["exact_ohlc_rows"]
        for item in overlap_comparison["comparisons"]
        for window in item["overlapping_segments"]
    )
    report = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "source": "Yllvar/fx-historical-data; public Hub split and original pair/timeframe Parquet files",
        "four_pair_scope": manifest["scope"],
        "combined_dataset": dataset_check,
        "file_count": len(entries),
        "structural_pass_files": sum(item["structural_pass"] for item in entries),
        "structural_fail_files": [
            item for item in entries if not item["structural_pass"]
        ],
        "files": entries,
        "full_hub_split": {
            "rows": full_manifest["rows"],
            "columns": full_manifest["columns"],
            "symbol_timeframe_identity_available": False,
            "training_ready_as_flat_split": False,
        },
        "cross_source_consistency": {
            "comparison_report": str(overlap_path.relative_to(ROOT)),
            "pair_timeframe_combinations_with_overlap": sum(
                bool(item["overlapping_segments"])
                for item in overlap_comparison["comparisons"]
            ),
            "matched_timestamps": overlap_rows,
            "exact_ohlc_rows": exact_rows,
            "exact_ohlc_fraction": exact_rows / overlap_rows if overlap_rows else None,
            "combinations_without_overlap": [
                f"{item['symbol']} {item['timeframe']}"
                for item in overlap_comparison["comparisons"]
                if not item["overlapping_segments"]
            ],
        },
        "research_readiness": {
            "four_pair_subset_can_be_loaded_with_pair_and_timeframe_identity": True,
            "historically_broad_enough_to_replace_existing_curated_history": False,
            "requires_followup": [
                "Quantify missing-bar/session gaps against an explicit FX calendar.",
                "Review the rare OHLC mismatches in overlap-comparison.json before research use.",
                "Measure spread/cost assumptions; this source has OHLCV but no bid/ask quotes.",
                "Keep bar-open timestamps distinct from close/availability times in joins and labels.",
                "Run chronological train/validation splits with overlapping label purging before model use.",
                "Verify source authenticity independently; structural checks cannot authenticate a publisher.",
            ],
        },
    }
    output = DATA_ROOT / "quality-report.json"
    output.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "file_count",
                    "structural_pass_files",
                    "structural_fail_files",
                    "combined_dataset",
                    "research_readiness",
                )
            },
            indent=2,
            default=str,
        )
    )
    print(f"Full quality report: {output}")


if __name__ == "__main__":
    main()
