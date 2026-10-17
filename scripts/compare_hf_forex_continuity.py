"""Compare gap-safe models using the saved HF snapshot and fixed historical splits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.adapters import SQLiteRepository
from astra.ml import load_model
from astra.repositories import Entity, Journal, json_value
from scripts.hf_forex_continuity import FITTING_SETTINGS, run
from scripts.train_hf_forex_experiment import (
    DATA_ROOT,
    DATASET_ID,
    PAIRS,
    source_inventory,
)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _implementation_inventory():
    names = (
        "scripts/compare_hf_forex_continuity.py",
        "scripts/hf_forex_continuity.py",
        "scripts/hf_source_clock.py",
        "scripts/train_hf_forex_experiment.py",
        "astra/ml.py",
        "astra/data.py",
        "astra/models.py",
        "astra/indicators.py",
        "astra/historical/validation.py",
        "astra/repositories.py",
        "astra/adapters.py",
    )
    return {name: sha256((REPO / name).read_bytes()).hexdigest() for name in names}


def _references(root, pairs, max_rows, inputs, timestamp_shift_minutes=0):
    if root is None:
        return {}, {}
    root = Path(root).resolve()
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["source"] != inputs or plan["maximum_rows_per_pair"] != max_rows:
        raise ValueError("reference source or row limit differs from this comparison")
    if plan.get("source_timestamp_shift_minutes", 0) != timestamp_shift_minutes:
        raise ValueError("reference timestamp convention differs from this comparison")
    for name in (
        "scripts/train_hf_forex_experiment.py",
        "astra/ml.py",
        "astra/data.py",
        "astra/models.py",
        "astra/indicators.py",
        "astra/historical/validation.py",
    ):
        if (
            plan.get("implementation_sha256", {}).get(name)
            != sha256((REPO / name).read_bytes()).hexdigest()
        ):
            raise ValueError(
                "reference continuity implementation is not the verified original"
            )
    summary_path = root / "comparison.json"
    previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if previous_summary["status"] != "COMPLETED" or not all(
        previous_summary.get(key)
        for key in ("source_identity_rechecked", "implementation_identity_rechecked")
    ):
        raise ValueError(
            "reference comparison did not complete with verified identities"
        )
    paths = [plan_path, summary_path]
    reports = {}
    for pair in pairs:
        path = root / f"{pair}-M1-report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["pair"] != pair or report["status"] != "TRAINED":
            raise ValueError("reference must contain a trained report for every pair")
        reports[pair] = report
        paths.append(path)
    return reports, {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}


def _reference_comparison(report, reference):
    current = report["validation"]
    previous = reference["validation"]
    baseline = previous["continuous_open_session_windows"]
    if (
        report["dataset_version"] != reference["dataset_version"]
        or current["fixed_validation_start"] != previous["validation_start"]
        or current["validation_samples"] != baseline["sample_count"]
    ):
        raise ValueError(
            "reference and clean model do not share the same validation subset"
        )
    return {
        "previous_model_version": reference["model"]["version"],
        "same_dataset_boundary_and_subset_count": True,
        "previous_model_on_continuous_validation": baseline,
        "accuracy_change_percentage_points": (
            current["overall"]["accuracy"] - baseline["accuracy"]
        )
        * 100,
        "brier_reduction": baseline["brier"] - current["overall"]["brier"],
        "interpretation": "Same historical subset; this comparison consumes reused development data.",
    }


def _markdown(summary):
    timestamp_note = (
        "Source opening timestamps are shifted one minute to availability; sampled BID/open provenance supports this convention but does not authenticate the whole snapshot."
        if summary.get("source_timestamp_shift_minutes") == 1
        else "Source timestamps are assumed close/availability times; bid/ask basis and volume semantics remain unverified."
    )
    lines = [
        "# Hugging Face gap-safe model comparison",
        "",
        f"Status: **{summary['status']}**. Recorded at {summary['completed_at']}.",
        "",
        "The original 70/30 chronological boundary is fixed before filtering. Both fitting and scoring require 15 consecutive feature bars and 3 consecutive label bars inside the assumed FX session. No gaps are filled. All selected candidates use the same fixed model settings; no threshold tuning is performed.",
        "",
        "| Pair | Clean training | Clean validation | Accuracy | Training-majority baseline | Gain (pp) | Brier | Constant Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in summary["reports"]:
        if report["status"] != "TRAINED":
            lines.append(f"| {report['pair']} | {report['status']} | | | | | | |")
            continue
        v = report["validation"]
        s = v["overall"]
        lines.append(
            f"| {report['pair']} | {v['training_samples']:,} | {v['validation_samples']:,} | "
            f"{s['accuracy']:.4%} | {s['majority_baseline_accuracy']:.4%} | "
            f"{s['accuracy_improvement_percentage_points']:+.4f} | {s['brier']:.8f} | "
            f"{s['constant_probability_brier']:.8f} |"
        )
    if any("reference_comparison" in r for r in summary["reports"]):
        lines += [
            "",
            "| Pair | Accuracy change from original model on the same clean subset (pp) | Brier reduction |",
            "|---|---:|---:|",
        ]
        for report in summary["reports"]:
            if comparison := report.get("reference_comparison"):
                lines.append(
                    f"| {report['pair']} | {comparison['accuracy_change_percentage_points']:+.4f} | "
                    f"{comparison['brier_reduction']:+.8f} |"
                )
    lines += [
        "",
        "These are retrospective classification diagnostics on previously consumed history. Future bar availability is used to select complete label windows; coverage selection is not a historical trading decision. Overlapping labels are dependent observations. "
        + timestamp_note
        + " Accuracy does not establish net profitability. No model is promoted or used for execution.",
        "",
        "The frozen plan records source, reference, implementation and dependency identities before fitting. Per-pair reports include exclusion counts, fixed and retained timestamps, calendar-month scores and calibration bins. Standard models, reports and audit records are saved through Journal/SQLiteRepository. Model and core evaluation payloads are read back and compared before completion; reference comparisons are added as run-level metadata.",
        "",
        f"Source identity verified: {summary['source_identity_rechecked']}. Implementation identity verified: {summary['implementation_identity_rechecked']}. Journal payloads verified: {summary['journal_payloads_verified']}.",
        "",
    ]
    if summary.get("error"):
        lines += [f"Failure: {summary['error']}", ""]
    return "\n".join(lines)


def run_comparison(
    output_root,
    *,
    pairs=PAIRS,
    max_rows=400_000,
    reference_root=None,
    timestamp_shift_minutes=0,
):
    pairs = tuple(pairs)
    if not pairs or len(set(pairs)) != len(pairs) or any(p not in PAIRS for p in pairs):
        raise ValueError("select unique pairs from AUDUSD, EURUSD, GBPUSD, USDJPY")
    if max_rows < 15:
        raise ValueError("max_rows must be at least 15")
    if timestamp_shift_minutes not in (0, 1):
        raise ValueError("supported timestamp shifts are zero or one minute")
    inputs = source_inventory()
    implementation = _implementation_inventory()
    references, reference_hashes = _references(
        reference_root, pairs, max_rows, inputs, timestamp_shift_minutes
    )
    started = datetime.now(UTC)
    identity = f"{started:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    output = Path(output_root).resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "run_id": identity,
        "recorded_at": started.isoformat(),
        "dataset_id": DATASET_ID,
        "pairs": list(pairs),
        "maximum_rows_per_pair": max_rows,
        "selection": "latest source rows per ticker; no sorting, filling or merging",
        "partition": "original examples' chronological 70/30 boundary fixed before filtering",
        "purge": "training label availability strictly before original validation boundary",
        "continuity": "all 15 feature bars plus 3 label bars consecutive M1 and inside assumed FX session",
        "estimator": "training-only StandardScaler; LogisticRegression(random_state=0, max_iter=1000)",
        "fitting_settings": {
            **FITTING_SETTINGS,
            **(
                {"source_timestamp_shift_minutes": 1} if timestamp_shift_minutes else {}
            ),
        },
        "threshold": 0.5,
        "baselines": "clean training majority label and clean training up-frequency",
        "comparison_criteria": "report accuracy and Brier against controls, original clean-subset scores where supplied, monthly variation and exclusions; no tuning, selection or promotion",
        "holdout_status": "reused historical development data; no untouched holdout certified",
        "source_timestamp_shift_minutes": timestamp_shift_minutes,
        "timestamp_policy": (
            "source opening time +1 minute = close/availability; sampled provenance only"
            if timestamp_shift_minutes
            else "unverified source timestamps assumed close/availability times"
        ),
        "price_basis": "unknown",
        "promotion_eligible": False,
        "profitability_evaluated": False,
        "source": inputs,
        "reference_sha256": reference_hashes,
        "implementation_sha256": implementation,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "pyarrow")
        },
    }
    _write_json(output / "plan.json", plan)
    repository = SQLiteRepository(str(output / "experiment.sqlite3"))
    journal = Journal(repository)
    summary = {
        "run_id": identity,
        "status": "RUNNING",
        "output_directory": str(output),
        "source_timestamp_shift_minutes": timestamp_shift_minutes,
        "reports": [],
        "source_identity_rechecked": False,
        "implementation_identity_rechecked": False,
        "reference_identity_rechecked": False,
        "journal_payloads_verified": False,
        "promotion_eligible": False,
        "profitability_evaluated": False,
    }
    try:
        journal.record(
            Entity.AUDIT, started, {"event": "hf_continuity_planned", **plan}
        )
        for pair in pairs:
            print(f"Training on continuous windows: {pair} ...", flush=True)
            report = run(
                pair,
                max_rows=max_rows,
                journal=journal,
                timestamp_shift_minutes=timestamp_shift_minutes,
            )
            # The core report is checked before adding run-level comparison metadata.
            rows = repository.query(Entity.EVALUATION, pair=pair, timeframe="M1")
            if len(rows) != 1 or any(
                rows[0].payload.get(k) != v for k, v in report.items()
            ):
                raise ValueError("journal evaluation payload differs from the report")
            if report["status"] == "TRAINED":
                model = load_model(
                    journal,
                    pair=pair,
                    timeframe="M1",
                    as_of=datetime.fromisoformat(report["model"]["trained_at"]),
                )
                if json_value(model) != report["model"]:
                    raise ValueError(
                        "journal model payload differs from the fitted model"
                    )
                if pair in references:
                    report["reference_comparison"] = _reference_comparison(
                        report, references[pair]
                    )
            _write_json(output / f"{pair}-M1-report.json", report)
            summary["reports"].append(report)
            print(f"{pair}: {report['status']}", flush=True)
        if source_inventory() != inputs:
            raise ValueError("source snapshot changed during the experiment")
        summary["source_identity_rechecked"] = True
        if _implementation_inventory() != implementation:
            raise ValueError("implementation changed during the experiment")
        summary["implementation_identity_rechecked"] = True
        if any(
            sha256(Path(p).read_bytes()).hexdigest() != h
            for p, h in reference_hashes.items()
        ):
            raise ValueError("reference evidence changed during the experiment")
        summary["reference_identity_rechecked"] = True
        if len(repository.query(Entity.MODEL)) != sum(
            r["status"] == "TRAINED" for r in summary["reports"]
        ):
            raise ValueError("journal model count differs from the comparison")
        if len(repository.query(Entity.EVALUATION)) != len(pairs):
            raise ValueError("journal evaluation count differs from the comparison")
        summary["journal_payloads_verified"] = True
        summary["status"] = (
            "COMPLETED"
            if all(r["status"] == "TRAINED" for r in summary["reports"])
            else "INCOMPLETE"
        )
    except KeyboardInterrupt:
        summary["status"] = "INTERRUPTED"
        summary["error"] = "comparison interrupted; completed reports retained"
    except Exception as error:  # noqa: BLE001 - preserve finite run evidence on failure.
        summary["status"] = "FAILED"
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        summary["completed_at"] = max(
            [datetime.now(UTC)]
            + [datetime.fromisoformat(r["completed_at"]) for r in summary["reports"]]
        ).isoformat()
        try:
            journal.record(
                Entity.AUDIT,
                datetime.fromisoformat(summary["completed_at"]),
                {"event": "hf_continuity_finished", **summary},
            )
        except Exception as error:  # noqa: BLE001 - preserve a failed receipt on persistence errors.
            summary["status"] = "FAILED"
            summary["error"] = (
                f"final audit persistence failed: {type(error).__name__}: {error}"
            )
        finally:
            try:
                repository.close()
            except Exception as error:  # noqa: BLE001 - still save the failure receipt.
                summary["status"] = "FAILED"
                summary["error"] = (
                    f"repository close failed: {type(error).__name__}: {error}"
                )
        _write_json(output / "comparison.json", summary)
        (output / "comparison.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", choices=PAIRS, default=PAIRS)
    parser.add_argument("--max-rows", type=int, default=400_000)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--timestamp-shift-minutes", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--output-root", type=Path, default=DATA_ROOT / "training" / "continuity"
    )
    args = parser.parse_args(argv)
    result = run_comparison(
        args.output_root,
        pairs=args.pairs,
        max_rows=args.max_rows,
        reference_root=args.reference_root,
        timestamp_shift_minutes=args.timestamp_shift_minutes,
    )
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
