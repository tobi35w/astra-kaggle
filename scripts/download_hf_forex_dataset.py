"""Download, cache, and record the public Hugging Face Forex OHLCV split."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_hf_forex_experiment import CACHE, DATA_ROOT, DATASET_ID

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "ticker")


def main() -> int:
    from datasets import load_dataset

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, split="train", cache_dir=str(CACHE))
    if tuple(dataset.column_names) != REQUIRED_COLUMNS:
        raise ValueError(f"unexpected dataset schema: {dataset.column_names}")
    if not len(dataset):
        raise ValueError("downloaded dataset is empty")
    tickers = sorted(dataset.unique("ticker"))
    first = dataset[0]
    last = dataset[-1]
    manifest = {
        "dataset_id": DATASET_ID,
        "split": "train",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "features": {key: str(value) for key, value in dataset.features.items()},
        "tickers": tickers,
        "first_record": first,
        "last_record": last,
        "license": "not declared in cached Hugging Face dataset metadata",
        "timestamp_convention": (
            "not declared; Astra treats timestamp as candle close/availability time"
        ),
        "verification": "train split loaded; schema and nonempty row count verified",
        "storage": str(CACHE.relative_to(REPO)),
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (DATA_ROOT / "README.md").write_text(
        "# OHLCV-1m-Forex snapshot\n\n"
        f"Source: https://huggingface.co/datasets/{DATASET_ID}\n\n"
        f"The `train` split is cached under `{CACHE.relative_to(REPO)}`. "
        f"See `manifest.json` for the loaded row count ({len(dataset):,}), schema, "
        "ticker list, and local verification. The cached dataset has no README or "
        "declared license metadata, and its timestamp convention is unspecified. "
        "Astra treats source timestamps as candle close/availability times.\n\n"
        "Use `scripts/train_hf_forex_experiment.py` to run a chronological, "
        "research-only experiment for a single supported ticker. It checks OHLC, "
        "finite values, timestamp ordering, and duplicates before training. "
        "The external data is not independently authenticated, does not identify "
        "bid/ask prices, and does not activate a model for execution.\n",
        encoding="utf-8",
    )
    print(f"Verified {len(dataset):,} rows across {len(tickers)} tickers")
    print(f"Manifest: {DATA_ROOT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
