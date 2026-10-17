"""Run a finite, locally journaled comparison of the four HF Forex models."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.adapters import SQLiteRepository
from astra.repositories import Entity, Journal
from scripts.train_hf_forex_experiment import (
    CAVEAT,
    DATA_ROOT,
    DATASET_ID,
    PAIRS,
    run,
    source_inventory,
)


def _write_json(path: Path, value) -> None:
    # Each campaign owns a new directory; never overwrite the previous experiment.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _implementation_inventory() -> dict:
    names = (
        "scripts/compare_hf_forex_models.py",
        "scripts/train_hf_forex_experiment.py",
        "astra/ml.py",
        "astra/data.py",
        "astra/models.py",
        "astra/indicators.py",
        "astra/historical/validation.py",
    )
    return {name: sha256((REPO / name).read_bytes()).hexdigest() for name in names}


def _markdown(summary: dict) -> str:
    lines = [
        "# Hugging Face Forex model comparison",
        "",
        f"Status: **{summary['status']}**. Completed at {summary['completed_at']}.",
        "",
        (
            "One fixed logistic model per pair, latest "
            f"{summary['maximum_rows_per_pair']:,} rows at most, chronological 70/30 "
            "split, overlapping training labels purged. Baselines are fitted only "
            "to the training labels."
        ),
        "",
        "| Pair | Validation samples | Model accuracy | Majority accuracy | Improvement (pp) | Model Brier | Constant Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in summary["reports"]:
        values = report.get("validation", {}).get("overall")
        if not values:
            lines.append(
                f"| {report['pair']} | {report['status']} | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {report['pair']} | {values['sample_count']:,} | "
            f"{values['accuracy']:.4%} | {values['majority_baseline_accuracy']:.4%} | "
            f"{values['accuracy_improvement_percentage_points']:+.4f} | "
            f"{values['brier']:.8f} | {values['constant_probability_brier']:.8f} |"
        )
    lines.extend(
        [
            "",
            (
                "Positive Brier improvement means lower probability error than the "
                "training-frequency constant. Accuracy alone does not establish useful skill."
            ),
            "",
            "## Data and interpretation",
            "",
            CAVEAT,
            "",
            (
                "Each pair has its own date range; these scores are not a same-period "
                "contest or a portfolio return. Zero price changes count as the non-up "
                "class. Overlapping validation labels are dependent observations."
            ),
            "",
            "| Pair | Selected first close (UTC) | Selected last close (UTC) | Missing open-session minutes | Validation windows excluded from continuous subset |",
            "|---|---|---|---:|---:|",
        ]
    )
    for report in summary["reports"]:
        lines.append(
            f"| {report['pair']} | {report['first_available_at']} | "
            f"{report['last_available_at']} | "
            f"{report['data_quality']['missing_open_session_minutes']:,} | "
            f"{report.get('validation', {}).get('excluded_from_continuous_subset', '—')} |"
        )
    lines.extend(
        [
            "",
            (
                "The per-pair JSON files include calendar-month results, calibration bins, "
                "predicted class frequencies, and scores restricted to complete, consecutive "
                "15-bar feature windows plus 3-bar labels inside the assumed FX session. "
                "The subset uses the same fitted model and training baseline; it is a "
                "diagnostic, not a second training run. Missing bars are never filled."
            ),
            "",
            (
                "No cost-aware trading evaluation, model promotion, or execution was performed. "
                "This archive cannot replace the missing September–October 2026 data. "
                "A later, separately verified holdout and risk-gated evaluation with costs "
                "remain necessary before using a candidate for paper decisions."
            ),
            "",
            (
                "`plan.json` records configuration, source and code hashes before fitting. "
                "`experiment.sqlite3` stores standard model records and separate evaluation "
                "and audit records through Astra's Journal. The earlier EURUSD report is preserved."
            ),
            "",
            (
                f"Source identity recheck passed: {summary['source_identity_rechecked']}. "
                f"Implementation identity recheck passed: {summary['implementation_identity_rechecked']}."
            ),
            "",
        ]
    )
    if summary.get("error"):
        lines.extend([f"Failure: {summary['error']}", ""])
    return "\n".join(lines)


def run_comparison(output_root, *, pairs=PAIRS, max_rows=400_000) -> dict:
    pairs = tuple(pairs)
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(pair not in PAIRS for pair in pairs)
    ):
        raise ValueError("select unique pairs from AUDUSD, EURUSD, GBPUSD, USDJPY")
    if max_rows < 15:
        raise ValueError("max_rows must be at least 15")
    inputs = source_inventory()
    implementation = _implementation_inventory()
    started_at = datetime.now(UTC)
    identity = f"{started_at:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    output = Path(output_root).resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "run_id": identity,
        "recorded_at": started_at.isoformat(),
        "dataset_id": DATASET_ID,
        "pairs": list(pairs),
        "maximum_rows_per_pair": max_rows,
        "selection": "last rows per ticker in source order; never sort or fill gaps",
        "timestamp_policy": "unverified source labels assumed close/availability times",
        "training_fraction": 0.7,
        "label_horizon_observed_bars": 3,
        "label": "1 when future close is higher; 0 otherwise, including ties",
        "purge": "training label availability must be strictly before validation start",
        "estimator": "StandardScaler fitted on training only; LogisticRegression(random_state=0, max_iter=1000)",
        "threshold": 0.5,
        "baselines": "training majority class and training up-frequency probability",
        "tuning": "none",
        "holdout_status": "historical diagnostic; freshness not certified; prior EURUSD period reused",
        "promotion_eligible": False,
        "source": inputs,
        "implementation_sha256": implementation,
        "caveat": CAVEAT,
    }
    _write_json(output / "plan.json", plan)
    repository = SQLiteRepository(str(output / "experiment.sqlite3"))
    journal = Journal(repository)
    summary = {
        "run_id": identity,
        "status": "RUNNING",
        "output_directory": str(output),
        "maximum_rows_per_pair": max_rows,
        "reports": [],
        "source_identity_rechecked": False,
        "implementation_identity_rechecked": False,
        "promotion_eligible": False,
        "profitability_evaluated": False,
    }
    try:
        journal.record(
            Entity.AUDIT, started_at, {"event": "hf_comparison_planned", **plan}
        )
        for pair in pairs:
            print(f"Training and diagnosing {pair} ...", flush=True)
            report = run(pair, max_rows=max_rows, journal=journal)
            _write_json(output / f"{pair}-M1-report.json", report)
            summary["reports"].append(report)
            print(f"{pair}: {report['status']}", flush=True)
        if source_inventory() != inputs:
            raise ValueError("source snapshot changed during the experiment")
        summary["source_identity_rechecked"] = True
        if _implementation_inventory() != implementation:
            raise ValueError("experiment implementation changed during the run")
        summary["implementation_identity_rechecked"] = True
        summary["status"] = (
            "COMPLETED"
            if all(report["status"] == "TRAINED" for report in summary["reports"])
            else "INCOMPLETE"
        )
        summary["journal_model_records"] = len(repository.query(Entity.MODEL))
        summary["journal_evaluation_records"] = len(repository.query(Entity.EVALUATION))
        if summary["journal_model_records"] != sum(
            report["status"] == "TRAINED" for report in summary["reports"]
        ) or summary["journal_evaluation_records"] != len(pairs):
            raise ValueError("journal read-back counts differ from experiment results")
    except KeyboardInterrupt:
        summary["status"] = "INTERRUPTED"
        summary["error"] = "comparison interrupted; completed pair reports are retained"
    except Exception as error:  # noqa: BLE001 - retain completed evidence after failure.
        summary["status"] = "FAILED"
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        summary["completed_at"] = datetime.now(UTC).isoformat()
        try:
            journal.record(
                Entity.AUDIT,
                datetime.fromisoformat(summary["completed_at"]),
                {"event": "hf_comparison_finished", **summary},
            )
        finally:
            repository.close()
        _write_json(output / "comparison.json", summary)
        (output / "comparison.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", choices=PAIRS, default=PAIRS)
    parser.add_argument("--max-rows", type=int, default=400_000)
    parser.add_argument(
        "--output-root", type=Path, default=DATA_ROOT / "training" / "comparisons"
    )
    args = parser.parse_args(argv)
    result = run_comparison(args.output_root, pairs=args.pairs, max_rows=args.max_rows)
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(Path(result["output_directory"]) / "comparison.md"),
            }
        )
    )
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
