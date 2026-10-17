"""Synthetic launcher checks: no downloads, real-data scans, or model fitting."""

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from astra.adapters import SQLiteRepository
from astra.ml import Model
from astra.repositories import Entity, Journal, json_value
from scripts import kaggle_run as runner


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_training(tmp_path):
    """A tiny Journal fixture, not a claimed historical experiment."""
    root = tmp_path / "training" / "20260101T000000-fixture"
    root.mkdir(parents=True)
    at = datetime(2026, 1, 1, tzinfo=UTC)
    model = Model(
        "synthetic-model",
        at,
        at,
        at,
        (0.0,) * 4,
        (1.0,) * 4,
        (0.0,) * 4,
        -0.1,
        0.5,
        0.25,
        300,
    )
    report = {
        "purpose": "synthetic test only",
        "pair": "USDJPY",
        "status": "TRAINED",
        "model": json_value(model),
    }
    summary = {
        "status": "COMPLETED",
        "run_id": root.name,
        "output_directory": str(root),
        "reports": [report],
        "journal_payloads_verified": True,
        "source_identity_rechecked": True,
        "implementation_identity_rechecked": True,
    }
    _write_json(
        root / "plan.json",
        {
            "purpose": "synthetic test only",
            "maximum_rows_per_pair": 300,
            "source": {
                "dataset_id": runner.DATASET_ID,
                "cache_snapshot": runner.DATASET_REVISION,
            },
        },
    )
    _write_json(root / "comparison.json", summary)
    _write_json(root / "USDJPY-M1-report.json", report)
    repository = SQLiteRepository(str(root / "experiment.sqlite3"))
    try:
        journal = Journal(repository)
        journal.record(Entity.MODEL, at, model, pair="USDJPY", timeframe="M1")
        journal.record(
            Entity.EVALUATION,
            at,
            {"kind": "hf_forex_continuity_experiment", **report},
            pair="USDJPY",
            timeframe="M1",
        )
    finally:
        repository.close()
    return root, summary


def test_local_heavy_guard_precedes_dependencies_and_creates_nothing(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(runner, "KAGGLE_WORKING", tmp_path / "missing-kaggle")

    def forbidden(*args, **kwargs):
        pytest.fail("local heavy guard must run before dependency checks or work")

    monkeypatch.setattr(runner.importlib.metadata, "version", forbidden)
    monkeypatch.setattr(runner, "_run", forbidden)
    workspace = tmp_path / "results"
    with pytest.raises(SystemExit) as caught:
        runner.main(["--workspace", str(workspace)])
    assert caught.value.code == 2
    assert "heavy work is disabled outside Kaggle" in capsys.readouterr().err
    assert not workspace.exists()


def test_check_mode_is_local_and_does_not_load_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "KAGGLE_WORKING", tmp_path / "missing-kaggle")
    monkeypatch.setattr(runner.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda name: "test-version"
    )
    monkeypatch.setattr(
        runner, "_run", lambda *args: pytest.fail("check must be read-only")
    )
    assert (
        runner.main(["--mode", "check", "--workspace", str(tmp_path / "results")]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ENVIRONMENT_READY"
    assert report["pinned_revision"] == runner.DATASET_REVISION
    assert report["data_checked"] is False
    assert report["training_started"] is False
    assert not (tmp_path / "results").exists()


def test_default_cloud_workflow_is_usdjpy_all(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "KAGGLE_WORKING", tmp_path)
    monkeypatch.setattr(runner.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda name: "test-version"
    )
    calls = []
    monkeypatch.setattr(runner, "_run", lambda *args: calls.append(args) or 0)
    assert runner.main([]) == 0
    args, runtime = calls[0]
    assert (args.mode, args.pairs, args.max_rows) == ("all", ["USDJPY"], 400000)
    assert runtime["pinned_revision"] == runner.DATASET_REVISION


def test_training_receipt_checks_reopened_payloads_and_binds_reports(
    synthetic_training,
):
    root, summary = synthetic_training
    assert runner.verify_training(summary) == root
    receipt = runner._read_json(root / "verification.json")
    assert receipt["models_verified"] == receipt["evaluations_verified"] == 1
    assert set(receipt["reference_files_sha256"]) == {
        "plan.json",
        "comparison.json",
        "USDJPY-M1-report.json",
    }
    assert runner._validate_reference(root) == root
    repository = SQLiteRepository(str(root / "experiment.sqlite3"))
    try:
        audits = repository.query(Entity.AUDIT)
        assert audits[-1].payload == {"event": "kaggle_training_verified", **receipt}
    finally:
        repository.close()
    report = runner._read_json(root / "USDJPY-M1-report.json")
    report["purpose"] = "mutated after verification"
    _write_json(root / "USDJPY-M1-report.json", report)
    with pytest.raises(ValueError, match="changed after training verification"):
        runner._validate_reference(root)


def test_training_receipt_rejects_mismatched_saved_evaluation(synthetic_training):
    root, summary = synthetic_training
    summary["reports"][0]["purpose"] = "different from durable evaluation"
    _write_json(root / "comparison.json", summary)
    _write_json(root / "USDJPY-M1-report.json", summary["reports"][0])
    with pytest.raises(ValueError, match="saved evaluation differs"):
        runner.verify_training(summary)
    assert not (root / "verification.json").exists()


def test_latest_reference_ignores_a_newer_incomplete_run(synthetic_training, tmp_path):
    root, summary = synthetic_training
    runner.verify_training(summary)
    partial = tmp_path / "training" / "20990101T000000-incomplete"
    partial.mkdir()
    (partial / "verification.json").write_text("{broken", encoding="utf-8")
    assert runner._latest_reference(tmp_path) == root


def test_reference_failure_happens_before_outputs_or_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "test-original")
    args = SimpleNamespace(
        mode="profitability",
        workspace=tmp_path / "results",
        reference_root=tmp_path / "missing-reference",
        input_cache=None,
    )
    monkeypatch.setattr(
        runner, "configure_source", lambda *args: pytest.fail("no source scan")
    )
    with pytest.raises(FileNotFoundError):
        runner._run(args, {})
    assert not args.workspace.exists()


def test_cloud_download_is_pinned_anonymous_and_training_uses_availability(
    tmp_path,
    monkeypatch,
):
    from scripts import compare_hf_forex_continuity, evaluate_hf_profitability

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "test-original")
    monkeypatch.setenv("HF_HOME", "test-original")
    monkeypatch.setattr(
        runner.tempfile, "gettempdir", lambda: str(tmp_path / "session-tmp")
    )
    repo = tmp_path / "project"
    provenance = repo / "kaggle" / "provenance"
    provenance.mkdir(parents=True)
    _write_json(provenance / "audit.json", {"purpose": "synthetic fixture"})
    monkeypatch.setattr(runner, "REPO", repo)
    calls = []

    class FakeDataset:
        column_names = ("timestamp", "open", "high", "low", "close", "volume", "ticker")

        def __len__(self):
            return 1

    def fake_load(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return FakeDataset()

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load)
    )
    cache_calls = []

    def configure(cache):
        cache_calls.append(cache)
        return SimpleNamespace(
            _cached_source_files=lambda: (
                (tmp_path / runner.DATASET_REVISION / "synthetic.arrow",),
                {},
            )
        )

    monkeypatch.setattr(runner, "configure_source", configure)
    checked = []
    monkeypatch.setattr(
        evaluate_hf_profitability,
        "validate_provenance",
        lambda *args: checked.append(args),
    )
    fitted = []

    def train(output_root, **kwargs):
        fitted.append(kwargs)
        return {
            "status": "FAILED",
            "error": "synthetic stop",
            "output_directory": str(output_root),
        }

    monkeypatch.setattr(compare_hf_forex_continuity, "run_comparison", train)
    monkeypatch.setattr(
        runner, "verify_training", lambda *args: pytest.fail("failed run cannot verify")
    )
    args = SimpleNamespace(
        mode="all",
        workspace=tmp_path / "results",
        reference_root=None,
        input_cache=None,
        pairs=["USDJPY"],
        max_rows=400000,
    )
    assert runner._run(args, {}) == 1
    assert calls[0][0] == runner.DATASET_ID
    assert calls[0][1]["revision"] == runner.DATASET_REVISION
    assert calls[0][1]["token"] is False
    assert cache_calls == [tmp_path / "session-tmp" / "astra-hf-cache" / "datasets"]
    assert checked[0][1]["cache_snapshot"] == runner.DATASET_REVISION
    assert fitted == [
        {"pairs": ["USDJPY"], "max_rows": 400000, "timestamp_shift_minutes": 1}
    ]
