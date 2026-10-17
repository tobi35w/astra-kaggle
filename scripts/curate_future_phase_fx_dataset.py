"""Create a provenance-labeled four-pair subset of the Hugging Face FX data."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
SOURCE_ROOT = DATA_ROOT / "four-pairs-source-parquet"
LABELED_ROOT = DATA_ROOT / "four-pairs-labeled-parquet"
DATASET_PATH = DATA_ROOT / "four-pairs-dataset"
STAGING_PATH = DATA_ROOT / "four-pairs-dataset.incomplete"
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4")
REQUIRED_COLUMNS = {"Time", "Open", "High", "Low", "Close", "Volume"}


def main() -> None:
    if DATASET_PATH.exists() or STAGING_PATH.exists():
        raise FileExistsError(
            "Curated dataset output already exists; inspect it before rerunning"
        )
    entries = []
    source_files = []
    total_rows = 0

    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            filename = f"{pair}/{pair}_{timeframe}.parquet"
            local_source = Path(
                hf_hub_download(
                    repo_id="Yllvar/fx-historical-data",
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(SOURCE_ROOT),
                )
            )
            table = pq.read_table(local_source)
            missing = REQUIRED_COLUMNS - set(table.column_names)
            if missing:
                raise ValueError(
                    f"{filename} is missing required columns: {sorted(missing)}"
                )
            if table.num_rows == 0:
                raise ValueError(f"{filename} is empty")

            timestamps = table["Time"]
            if not pa.types.is_timestamp(timestamps.type):
                raise ValueError(
                    f"{filename} Time column is not a timestamp: {timestamps.type}"
                )
            monotonic = pc.all(
                pc.greater_equal(
                    timestamps.slice(1), timestamps.slice(0, table.num_rows - 1)
                )
            ).as_py()
            if not monotonic:
                raise ValueError(f"{filename} timestamps are not sorted")

            labeled = table.append_column("Symbol", pa.array([pair] * table.num_rows))
            labeled = labeled.append_column(
                "Timeframe", pa.array([timeframe] * table.num_rows)
            )
            target = LABELED_ROOT / pair / f"{pair}_{timeframe}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(labeled, target, compression="zstd")
            check = pq.read_table(target)
            if (
                check.num_rows != table.num_rows
                or check["Symbol"][0].as_py() != pair
                or check["Timeframe"][0].as_py() != timeframe
            ):
                raise ValueError(
                    f"{filename} failed labeled parquet round-trip verification"
                )

            row = {
                "source_file": filename,
                "labeled_file": str(target.relative_to(DATA_ROOT)),
                "symbol": pair,
                "timeframe": timeframe,
                "rows": table.num_rows,
                "first_time_utc": str(timestamps[0].as_py()),
                "last_time_utc": str(timestamps[-1].as_py()),
                "columns": check.column_names,
                "monotonic_timestamps": True,
            }
            entries.append(row)
            source_files.append(str(target))
            total_rows += table.num_rows
            print(f"Verified {pair} {timeframe}: {table.num_rows:,} rows")

    DATASET_ROOT = DATA_ROOT / "labeled-cache"
    labeled_ds = load_dataset(
        "parquet",
        data_files={"train": source_files},
        split="train",
        cache_dir=str(DATASET_ROOT),
    )
    if len(labeled_ds) != total_rows or not {"Symbol", "Timeframe"}.issubset(
        labeled_ds.column_names
    ):
        raise ValueError(
            "Combined labeled dataset failed row-count or provenance-column checks"
        )
    labeled_ds.save_to_disk(str(STAGING_PATH))
    from datasets import load_from_disk

    reloaded = load_from_disk(str(STAGING_PATH))
    if len(reloaded) != total_rows or reloaded.column_names != labeled_ds.column_names:
        raise ValueError("Curated dataset failed the saved round-trip check")
    STAGING_PATH.replace(DATASET_PATH)

    manifest = {
        "dataset_id": "Yllvar/fx-historical-data",
        "scope": {"symbols": PAIRS, "timeframes": TIMEFRAMES},
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": total_rows,
        "columns": reloaded.column_names,
        "files": entries,
        "verification": "each source parquet checked; timestamps sorted; labeled parquet round-tripped; combined dataset saved and reloaded",
        "storage": str(DATASET_PATH.relative_to(ROOT)),
        "note": "Source bar times represent bar open times per dataset card; convert to availability/close times causally before joins or evaluation.",
    }
    (DATA_ROOT / "four-pairs-manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Curated labeled dataset verified: {len(reloaded):,} rows")
    print(f"Saved to: {DATASET_PATH}")


if __name__ == "__main__":
    main()
