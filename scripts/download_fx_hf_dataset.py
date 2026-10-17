"""Download and verify the published Hugging Face FX dataset for a later phase."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
CACHE = DATA_ROOT / "hf-cache"
SAVED = DATA_ROOT / "train-dataset"
STAGING = DATA_ROOT / "train-dataset.incomplete"


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if SAVED.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset at {SAVED}")

    print("Connecting to Astra's Forex dataset...")
    ds = load_dataset(
        "Yllvar/fx-historical-data",
        split="train",
        cache_dir=str(CACHE),
    )

    print("\nDataset loaded successfully!")
    print(ds)
    print("\nColumn names:", ds.column_names)
    preview = ds.select(range(min(5, len(ds)))).to_pandas()
    print("\nFirst 5 rows:")
    print(preview.to_string(index=False))

    if len(ds) == 0 or not {"Open", "High", "Low", "Close"}.issubset(ds.column_names):
        raise ValueError("Loaded dataset is empty or missing required OHLC columns")
    if STAGING.exists():
        raise FileExistsError(
            f"Incomplete prior save exists at {STAGING}; inspect it first"
        )

    ds.save_to_disk(str(STAGING))
    saved = load_from_disk(str(STAGING))
    if len(saved) != len(ds) or saved.column_names != ds.column_names:
        raise ValueError("Saved dataset failed row-count or schema verification")
    if saved[0] != ds[0] or saved[-1] != ds[-1]:
        raise ValueError("Saved dataset failed first/last record verification")
    STAGING.replace(SAVED)

    manifest = {
        "dataset_id": "Yllvar/fx-historical-data",
        "split": "train",
        "loaded_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(saved),
        "columns": saved.column_names,
        "features": {name: str(feature) for name, feature in saved.features.items()},
        "first_record": saved[0],
        "last_record": saved[-1],
        "storage": str(SAVED.relative_to(ROOT)),
        "hub_cache": str(CACHE.relative_to(ROOT)),
        "verification": "reloaded from disk; row count, columns, first and last records match",
        "usage_note": (
            "Inspect columns before training. The published preview documents OHLCV/time "
            "but does not show pair or timeframe identifiers; do not mix rows until "
            "instrument/timeframe provenance is confirmed."
        ),
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (DATA_ROOT / "README.md").write_text(
        "# Hugging Face FX dataset snapshot\n\n"
        "Source: https://huggingface.co/datasets/Yllvar/fx-historical-data\n"
        "License declared by the dataset card: CC-BY-4.0.\n\n"
        "The complete `train` split was saved in Hugging Face Arrow format under "
        "`train-dataset/`; the Hub download cache is under `hf-cache/`. See "
        "`manifest.json` for row count, schema, and local round-trip verification.\n\n"
        "This is a future-phase source snapshot. It has not been used in Astra's "
        "current historical evaluation. The published full split covers 30 symbols "
        "and 7 timeframes. Confirm symbol/timeframe provenance and curate the four "
        "approved pairs before using any records for model training or evaluation.\n",
        encoding="utf-8",
    )
    print(f"\nVerified dataset saved to: {SAVED}")
    print(f"Manifest: {DATA_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
