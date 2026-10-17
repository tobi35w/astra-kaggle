"""Portable cloud launcher; keeps the original research/source interfaces intact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_ID = "keithtwesigye/OHLCV-1m-Forex"
DATASET_REVISION = "f38b5658b02d1a081f9c2abb314d8a3972caae1e"
KAGGLE_WORKING = Path("/kaggle/working")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_hashes(root, names):
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def configure_source(cache):
    """Inject a mounted/downloaded cache into the unchanged local Arrow adapter."""
    from scripts import train_hf_forex_experiment as source

    source.CACHE = Path(cache).resolve()
    return source


def verify_training(summary):
    from astra.adapters import SQLiteRepository
    from astra.ml import load_model
    from astra.repositories import Entity, Journal, json_value

    if summary["status"] != "COMPLETED" or not all(
        summary.get(key)
        for key in (
            "journal_payloads_verified",
            "source_identity_rechecked",
            "implementation_identity_rechecked",
        )
    ):
        raise ValueError("training did not complete with verified payloads")
    root = Path(summary["output_directory"])
    if not (root / "experiment.sqlite3").is_file():
        raise ValueError("training journal is missing")
    names = ["plan.json", "comparison.json"] + [
        f"{report['pair']}-M1-report.json" for report in summary["reports"]
    ]
    if not summary["reports"] or len(names) != len(set(names)):
        raise ValueError("training reports are empty or contain duplicate pairs")
    if _read_json(root / "comparison.json") != summary:
        raise ValueError("saved comparison differs from the training result")
    if (root / "verification.json").exists():
        raise FileExistsError("training already has a verification receipt")
    repo = SQLiteRepository(str(root / "experiment.sqlite3"))
    try:
        journal = Journal(repo)
        models = repo.query(Entity.MODEL)
        evaluations = repo.query(Entity.EVALUATION)
        if len(models) != len(summary["reports"]) or len(evaluations) != len(models):
            raise ValueError("model/evaluation counts differ from training results")
        for report in summary["reports"]:
            if _read_json(root / f"{report['pair']}-M1-report.json") != report:
                raise ValueError("saved pair report differs from the training result")
            saved = [row for row in evaluations if row.pair == report["pair"]]
            core = {k: v for k, v in report.items() if k != "reference_comparison"}
            if len(saved) != 1 or saved[0].payload != {
                "kind": "hf_forex_continuity_experiment",
                **core,
            }:
                raise ValueError("saved evaluation differs from the training report")
            model = load_model(
                journal,
                pair=report["pair"],
                timeframe="M1",
                as_of=datetime.fromisoformat(report["model"]["trained_at"]),
            )
            if json_value(model) != report["model"]:
                raise ValueError("saved model differs from the training report")
        receipt = {
            "status": "VERIFIED",
            "verified_at": max(
                [datetime.now(UTC)] + [r.occurred_at for r in (*models, *evaluations)]
            ).isoformat(),
            "run_id": summary["run_id"],
            "models_verified": len(models),
            "evaluations_verified": len(evaluations),
            "reference_files_sha256": _file_hashes(root, names),
            "profitability_proven": False,
        }
        record = journal.record(
            Entity.AUDIT,
            datetime.fromisoformat(receipt["verified_at"]),
            {"event": "kaggle_training_verified", **receipt},
        )
        if not any(
            saved.record_id == record.record_id and saved.payload == record.payload
            for saved in repo.query(Entity.AUDIT)
        ):
            raise ValueError("training verification audit failed read-back")
    finally:
        repo.close()
    with (root / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return root


def _validate_reference(root):
    """Read small receipts before an expensive download; full source checks follow."""
    root = Path(root).resolve()
    names = (
        "plan.json",
        "comparison.json",
        "USDJPY-M1-report.json",
        "verification.json",
    )
    plan, summary, report, receipt = [_read_json(root / name) for name in names]
    if not all(isinstance(value, dict) for value in (plan, summary, report, receipt)):
        raise TypeError("reference receipts must contain JSON objects")
    if not isinstance(plan.get("source"), dict) or not isinstance(
        summary.get("reports"), list
    ):
        raise TypeError("reference source and report list are malformed")
    if any(not isinstance(value, dict) for value in summary["reports"]):
        raise TypeError("reference report list is malformed")
    if (
        summary.get("status") != "COMPLETED"
        or receipt.get("status") != "VERIFIED"
        or report.get("pair") != "USDJPY"
        or report.get("status") != "TRAINED"
        or plan.get("source", {}).get("dataset_id") != DATASET_ID
        or plan.get("source", {}).get("cache_snapshot") != DATASET_REVISION
        or not isinstance(plan.get("maximum_rows_per_pair"), int)
        or plan["maximum_rows_per_pair"] < 15
    ):
        raise ValueError(
            "reference must be a verified USDJPY run from the pinned HF snapshot"
        )
    models = [r for r in summary.get("reports", []) if r.get("pair") == "USDJPY"]
    if len(models) != 1 or models[0].get("model") != report.get("model"):
        raise ValueError("reference model differs between saved reports")
    if receipt.get("run_id") is not None and receipt["run_id"] != summary.get("run_id"):
        raise ValueError("verification receipt belongs to a different training run")
    # Older local receipts did not have portable relative-path hashes. Their
    # source and selected-row identities are still checked by the evaluator.
    hashes = receipt.get("reference_files_sha256")
    if hashes is not None:
        expected_names = {"plan.json", "comparison.json"} | {
            f"{r['pair']}-M1-report.json" for r in summary["reports"]
        }
        if not isinstance(hashes, dict) or set(hashes) != expected_names:
            raise ValueError("reference receipt has an incomplete file inventory")
        # Every name is derived from a supported pair; never follow paths from
        # an uploaded JSON receipt when checking its hashes.
        if any(
            r.get("pair") not in ("USDJPY", "AUDUSD", "EURUSD", "GBPUSD")
            for r in summary["reports"]
        ):
            raise ValueError("reference contains an unsupported pair")
        if _file_hashes(root, sorted(expected_names)) != hashes:
            raise ValueError("reference files changed after training verification")
    return root


def _latest_reference(workspace):
    paths = sorted((workspace / "training").glob("*/verification.json"), reverse=True)
    for path in paths:
        try:
            return _validate_reference(path.parent)
        except (OSError, ValueError, KeyError, TypeError):
            continue  # An interrupted/malformed newer run must not hide valid history.
    raise ValueError(
        "no verified USDJPY reference; run --mode train/all or supply --reference-root"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "train", "profitability", "all"), default="all"
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path("/kaggle/working/astra-results")
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=("USDJPY", "AUDUSD", "EURUSD", "GBPUSD"),
        default=["USDJPY"],
    )
    parser.add_argument("--max-rows", type=int, default=400000)
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="explicitly allow heavy work outside Kaggle",
    )
    args = parser.parse_args(argv)
    if args.mode != "check" and not args.allow_local and not KAGGLE_WORKING.is_dir():
        parser.error(
            "heavy work is disabled outside Kaggle; --allow-local explicitly overrides this guard"
        )
    if args.max_rows < 300 or len(set(args.pairs)) != len(args.pairs):
        parser.error("use at least 300 rows and unique pairs")
    if args.mode == "all" and "USDJPY" not in args.pairs:
        parser.error("the profitability experiment requires USDJPY in --pairs")
    if args.reference_root and args.mode not in ("profitability", "check"):
        parser.error(
            "--reference-root is for profitability-only mode; train/all create a new reference"
        )
    if sys.version_info < (3, 12):  # noqa: UP036 - runs directly from an uninstalled source ZIP.
        parser.error("Python 3.12+ required; use the notebook's isolated runtime setup")
    packages = {}
    for name in ("numpy", "scikit-learn", "pyarrow", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            parser.error(
                f"missing {name}; install kaggle/requirements.txt and project [dev,ml] extras"
            )
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "dataset_id": DATASET_ID,
        "pinned_revision": DATASET_REVISION,
    }
    if args.mode == "check":
        print(
            json.dumps(
                {
                    "status": "ENVIRONMENT_READY",
                    **runtime,
                    "data_checked": False,
                    "training_started": False,
                },
                indent=2,
            )
        )
        return 0
    try:
        return _run(args, runtime)
    except KeyboardInterrupt:
        print(
            "Run interrupted; completed result directories were retained.",
            file=sys.stderr,
        )
        return 130
    except Exception as error:  # noqa: BLE001 - notebook gets a readable failure and nonzero exit.
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                    "results": str(args.workspace.resolve()),
                    "next_step": "Inspect preserved reports. Enable Kaggle Internet for downloads; use an unchanged cache/reference or rerun train/all for a new reference.",
                    "profitability_proven": False,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


def _run(args, runtime):
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    workspace = args.workspace.resolve()
    reference = None
    if args.mode == "profitability":
        reference = (
            _validate_reference(args.reference_root)
            if args.reference_root
            else _latest_reference(workspace)
        )
    # Reject missing input mounts before creating output or downloading anything.
    if args.input_cache and not args.input_cache.is_dir():
        raise ValueError("--input-cache must be the existing hf-cache directory")
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / f"runtime-{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(runtime, handle, indent=2)
    if args.input_cache:
        cache = args.input_cache.resolve()
    else:
        # Kaggle saves all of /kaggle/working as outputs, not just our results.
        # Keep multi-gigabyte downloads in session-temporary storage instead.
        cache_home = Path(tempfile.gettempdir()) / "astra-hf-cache"
        cache = cache_home / "datasets"
        os.environ["HF_HOME"] = str(cache_home / "hub-home")
        from datasets import load_dataset

        print(
            f"Loading pinned public dataset {DATASET_ID}@{DATASET_REVISION}", flush=True
        )
        dataset = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split="train",
            cache_dir=str(cache),
            token=False,
        )
        if not len(dataset) or set(dataset.column_names) != {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ticker",
        }:
            raise ValueError("downloaded dataset schema is unexpected")
        del dataset
    source = configure_source(cache)
    # Read the small manifest/file listing now; full identities are frozen by each experiment.
    paths, _ = source._cached_source_files()
    from scripts.evaluate_hf_profitability import validate_provenance

    provenance = REPO / "kaggle" / "provenance" / "audit.json"
    validate_provenance(
        _read_json(provenance),
        {"dataset_id": DATASET_ID, "cache_snapshot": paths[0].parent.name},
    )
    if args.mode in ("train", "all"):
        from scripts.compare_hf_forex_continuity import run_comparison

        summary = run_comparison(
            workspace / "training",
            pairs=args.pairs,
            max_rows=args.max_rows,
            timestamp_shift_minutes=1,
        )
        if summary["status"] != "COMPLETED":
            print(
                json.dumps(
                    {
                        "status": summary["status"],
                        "error": summary.get("error"),
                        "report": str(
                            Path(summary["output_directory"]) / "comparison.md"
                        ),
                    }
                ),
                flush=True,
            )
            return 1
        reference = verify_training(summary)
        print(f"Verified training report: {reference / 'comparison.md'}", flush=True)
    if args.mode in ("profitability", "all"):
        from scripts.evaluate_hf_profitability import run_experiment

        summary = run_experiment(
            workspace / "profitability",
            reference_root=reference,
            provenance_path=REPO / "kaggle" / "provenance" / "audit.json",
        )
        print(
            f"Profitability report: {Path(summary['output_directory']) / 'report.md'}",
            flush=True,
        )
        if summary["status"] != "COMPLETED":
            print(
                json.dumps({"status": summary["status"], "error": summary.get("error")})
            )
            return 1
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "results": str(workspace),
                "profitability_proven": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
