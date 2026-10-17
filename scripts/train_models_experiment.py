"""Train chronological logistic models on managed real-history windows.

The ML gate in astra/ml.py requires an explicit `real_data` provenance flag.
This runner refuses to set it for anything except managed datasets that passed
the loader's quality gate, so synthetic fixtures can never reach training.
Results are journaled; validation metrics are reported as measured, with no
profitability claim.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.historical.dataset import load_dataset
from astra.ml import train
from astra.repositories import Entity, Journal, MemoryRepository

MANAGED_PROVENANCE_PREFIXES = ("managed dukascopy", "converted qleap parquet")
CAVEAT = (
    "Chronological split with labels purged at the boundary. Validation metrics "
    "describe this window only and do not establish a deployable edge."
)


def _is_managed_real_data(data) -> bool:
    """Only the managed loader's verified provenance may unlock ML training."""
    metadata = data.metadata
    if metadata.get("quality_status") != "PASS":
        return False
    if "synthetic" in str(metadata.get("source", "")).lower():
        return False
    provenance = str(metadata.get("provenance", "")).strip().lower()
    return provenance.startswith(MANAGED_PROVENANCE_PREFIXES)


def run(root, *, pair, timeframe, journal=None) -> dict:
    """Train one model per window for a pair and return a measured report."""
    journal = journal or Journal(MemoryRepository())
    data = load_dataset(root, pair, timeframe)
    if not _is_managed_real_data(data):
        return {
            "pair": pair,
            "timeframe": timeframe,
            "status": "REFUSED",
            "reason": "managed, quality-PASS real history is required for training",
            "caveat": CAVEAT,
        }
    training_data = replace(data, metadata=data.metadata | {"real_data": True})
    result = train(training_data, enabled=True, journal=journal)
    report = {
        "pair": pair,
        "timeframe": timeframe,
        "status": result.status,
        "reason": result.reason,
        "caveat": CAVEAT,
    }
    if result.model is not None:
        model = result.model
        report["model"] = {
            "version": model.version,
            "trained_at": model.trained_at.isoformat(),
            "training_end": model.training_end.isoformat(),
            "validation_start": model.validation_start.isoformat(),
            "validation_accuracy": model.validation_accuracy,
            "validation_brier": model.validation_brier,
            "sample_count": model.sample_count,
        }
    journal.record(
        Entity.MODEL if result.model is not None else Entity.AUDIT,
        data.candles[-1].timestamp,
        {"kind": "ml_training_experiment", **report},
        pair=pair,
        timeframe=timeframe,
    )
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", default="H4", choices=("M5", "M15", "H1", "H4"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        report = run(args.dataset_root, pair=args.pair, timeframe=args.timeframe)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__}))
        return 2
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "TRAINED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
