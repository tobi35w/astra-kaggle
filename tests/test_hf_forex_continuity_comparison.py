"""Synthetic runner fixtures; no real cache or historical evidence is modified."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from astra.adapters import SQLiteRepository
from astra.repositories import Entity
from scripts import compare_hf_forex_continuity as comparison
from scripts import hf_forex_continuity as core
from scripts import train_hf_forex_experiment as original


@pytest.fixture
def source(monkeypatch):
    def rows(pair, limit):
        result = []
        for i in range(600):
            opening = 1.1 + math.sin((i - 1) * 0.3) * 0.001
            close = 1.1 + math.sin(i * 0.3) * 0.001
            result.append(
                {
                    "timestamp": datetime(2025, 1, 6, tzinfo=UTC)
                    + timedelta(minutes=i + (60 if i >= 200 else 0)),
                    "open": opening,
                    "high": max(opening, close) + 0.0001,
                    "low": min(opening, close) - 0.0001,
                    "close": close,
                    "volume": 2.0,
                    "ticker": pair,
                }
            )
        return result

    monkeypatch.setattr(core, "_load_ticker_rows", rows)
    monkeypatch.setattr(original, "_load_ticker_rows", rows)
    monkeypatch.setattr(comparison, "source_inventory", lambda: {"synthetic": True})
    return rows


def _reference(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps(
            {
                "source": {"synthetic": True},
                "maximum_rows_per_pair": 600,
                "implementation_sha256": comparison._implementation_inventory(),
            }
        )
    )
    report = original.run("EURUSD", max_rows=600)
    (root / "EURUSD-M1-report.json").write_text(json.dumps(report))
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "source_identity_rechecked": True,
                "implementation_identity_rechecked": True,
            }
        )
    )
    return root


def test_plan_precedes_fitting_and_models_roundtrip_without_overwriting_reference(
    tmp_path, monkeypatch, source
):
    reference = _reference(tmp_path)
    before = {p: p.read_bytes() for p in reference.iterdir()}
    output_root = tmp_path / "runs"

    def observed_run(*args, **kwargs):
        plans = list(output_root.glob("*/plan.json"))
        plan = json.loads(plans[-1].read_text())
        assert plan["pairs"] == ["EURUSD"]
        assert plan["reference_sha256"]
        assert plan["fitting_settings"]["label_horizon_minutes"] == 3
        assert "before filtering" in plan["partition"]
        return core.run(*args, **kwargs)

    monkeypatch.setattr(comparison, "run", observed_run)
    results = [
        comparison.run_comparison(
            output_root, pairs=("EURUSD",), max_rows=600, reference_root=reference
        )
        for _ in range(2)
    ]
    assert results[0]["output_directory"] != results[1]["output_directory"]
    for summary in results:
        assert summary["status"] == "COMPLETED"
        assert summary["journal_payloads_verified"]
        report = summary["reports"][0]
        assert report["reference_comparison"]["same_dataset_boundary_and_subset_count"]
        assert report["validation"]["excluded_training_samples"] == 17
        output = Path(summary["output_directory"])
        assert json.loads((output / "comparison.json").read_text()) == summary
        repository = SQLiteRepository(str(output / "experiment.sqlite3"))
        try:
            assert len(repository.query(Entity.MODEL)) == 1
            assert len(repository.query(Entity.EVALUATION)) == 1
            assert len(repository.query(Entity.AUDIT)) == 2
        finally:
            repository.close()
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("kind", ["source", "implementation", "reference"])
def test_changed_inputs_prevent_completed_status(tmp_path, monkeypatch, source, kind):
    reference = _reference(tmp_path)
    if kind == "source":
        values = iter([{"synthetic": True}, {"synthetic": "changed"}])
        monkeypatch.setattr(comparison, "source_inventory", lambda: next(values))
    elif kind == "implementation":
        values = iter([{"code": "before"}, {"code": "after"}])
        monkeypatch.setattr(
            comparison, "_implementation_inventory", lambda: next(values)
        )
    else:

        def changed(*args, **kwargs):
            report = core.run(*args, **kwargs)
            path = reference / "EURUSD-M1-report.json"
            path.write_text(path.read_text() + "\n")
            return report

        monkeypatch.setattr(comparison, "run", changed)
    summary = comparison.run_comparison(
        tmp_path / "runs", pairs=("EURUSD",), max_rows=600, reference_root=reference
    )
    assert summary["status"] == "FAILED"
    assert "changed during" in summary["error"]
    assert not summary["promotion_eligible"]


def test_report_journal_mismatch_prevents_completion(tmp_path, monkeypatch, source):
    def wrong_report(*args, **kwargs):
        report = core.run(*args, **kwargs)
        report["source_rows"] = 123456
        return report

    monkeypatch.setattr(comparison, "run", wrong_report)
    summary = comparison.run_comparison(tmp_path, pairs=("EURUSD",), max_rows=600)
    assert summary["status"] == "FAILED"
    assert "evaluation payload" in summary["error"]
    assert not summary["journal_payloads_verified"]


def test_reference_requires_same_source_and_row_limit_before_start(tmp_path, source):
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="row limit"):
        comparison.run_comparison(
            tmp_path / "runs", pairs=("EURUSD",), max_rows=500, reference_root=reference
        )
    assert not (tmp_path / "runs").exists()


def test_shifted_run_rejects_legacy_reference_before_fitting(tmp_path, source):
    reference = _reference(tmp_path)
    with pytest.raises(ValueError, match="timestamp convention"):
        comparison.run_comparison(
            tmp_path / "runs",
            pairs=("EURUSD",),
            max_rows=600,
            reference_root=reference,
            timestamp_shift_minutes=1,
        )
    assert not (tmp_path / "runs").exists()


def test_shifted_comparison_records_availability_clock(tmp_path, source):
    summary = comparison.run_comparison(
        tmp_path / "runs", pairs=("EURUSD",), max_rows=600, timestamp_shift_minutes=1
    )
    assert summary["status"] == "COMPLETED", summary.get("error")
    report = summary["reports"][0]
    assert report["source_timestamp_shift_minutes"] == 1
    assert report["fitting_settings"]["source_timestamp_shift_minutes"] == 1
    assert report["first_available_at"] == "2025-01-06T00:01:00+00:00"
    assert report["raw_dataset_version"] != report["dataset_version"]
    plan = json.loads((Path(summary["output_directory"]) / "plan.json").read_text())
    assert plan["source_timestamp_shift_minutes"] == 1


def test_incompatible_validation_subset_prevents_comparison(tmp_path, source):
    reference = _reference(tmp_path)
    path = reference / "EURUSD-M1-report.json"
    report = json.loads(path.read_text())
    report["validation"]["continuous_open_session_windows"]["sample_count"] -= 1
    path.write_text(json.dumps(report))
    summary = comparison.run_comparison(
        tmp_path / "runs", pairs=("EURUSD",), max_rows=600, reference_root=reference
    )
    assert summary["status"] == "FAILED"
    assert "validation subset" in summary["error"]


def test_interrupted_run_preserves_frozen_plan_and_failure_receipt(
    tmp_path, monkeypatch, source
):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(comparison, "run", interrupted)
    summary = comparison.run_comparison(tmp_path, pairs=("EURUSD",), max_rows=600)
    assert summary["status"] == "INTERRUPTED"
    assert not summary["reports"]
    output = Path(summary["output_directory"])
    assert (output / "plan.json").exists()
    assert (
        json.loads((output / "comparison.json").read_text())["status"] == "INTERRUPTED"
    )


def test_old_and_new_continuity_masks_select_identical_validation_timestamps(
    source, monkeypatch
):
    from dataclasses import replace

    rows = source("EURUSD", 600)
    for row in rows[500:]:
        row["timestamp"] += timedelta(minutes=90)
    data = original.to_market_data(rows, "EURUSD")
    _, validation, _, _ = core.continuous_split(data)
    examples = tuple(
        replace(e, values=(e.timestamp.timestamp(),)) for e in original.dataset(data)
    )
    monkeypatch.setattr(original, "dataset", lambda data: examples)
    monkeypatch.setattr(
        original,
        "score_predictions",
        lambda labels, probs, **kwargs: {"timestamps": probs},
    )

    class TimestampProbe:
        trained_at = datetime(2026, 10, 16, tzinfo=UTC)

        def predict(self, values, at):
            return values[0]

    previous = original.validation_diagnostics(data, TimestampProbe())
    assert previous["continuous_open_session_windows"]["timestamps"] == [
        e.timestamp.timestamp() for e in validation
    ]


def test_final_audit_failure_is_saved_as_failed_receipt(tmp_path, monkeypatch, source):
    record = comparison.Journal.record

    def unavailable_final_audit(self, entity, at, value, **kwargs):
        if entity == Entity.AUDIT and value.get("event") == "hf_continuity_finished":
            raise OSError("synthetic unavailable journal")
        return record(self, entity, at, value, **kwargs)

    monkeypatch.setattr(comparison.Journal, "record", unavailable_final_audit)
    summary = comparison.run_comparison(tmp_path, pairs=("EURUSD",), max_rows=600)
    assert summary["status"] == "FAILED"
    assert "final audit persistence failed" in summary["error"]
    saved = json.loads(
        (Path(summary["output_directory"]) / "comparison.json").read_text()
    )
    assert saved == summary
