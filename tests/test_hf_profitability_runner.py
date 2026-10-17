"""Synthetic frozen-run orchestration; never uses or changes real HF evidence."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from astra.adapters import SQLiteRepository
from astra.ml import Model
from astra.repositories import Entity, json_value
from scripts import evaluate_hf_profitability as runner
from scripts.train_hf_forex_experiment import to_market_data


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rows = [
        {
            "timestamp": datetime(2025, 1, 6, 12, tzinfo=UTC) + timedelta(minutes=i),
            "open": 150.0,
            "high": 150.05,
            "low": 149.95,
            "close": 150.0,
            "volume": 2,
            "ticker": "USDJPY",
        }
        for i in range(80)
    ]
    source = {
        "purpose": "synthetic test only",
        "dataset_id": "synthetic",
        "cache_snapshot": "fixture",
    }
    monkeypatch.setattr(runner, "source_inventory", lambda: source)
    monkeypatch.setattr(runner, "_load_ticker_rows", lambda pair, limit: rows)
    raw = to_market_data(rows, "USDJPY", max_rows=80)
    model = Model(
        "synthetic-model",
        datetime(2026, 10, 16, tzinfo=UTC),
        rows[20]["timestamp"],
        rows[25]["timestamp"],
        (0.0,) * 4,
        (1.0,) * 4,
        (0.0,) * 4,
        -0.1,
        0.5,
        0.25,
        80,
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    values = {
        "plan.json": {"source": source, "maximum_rows_per_pair": 80},
        "comparison.json": {
            "status": "COMPLETED",
            "reports": [{"pair": "USDJPY", "model": json_value(model)}],
        },
        "USDJPY-M1-report.json": {
            "pair": "USDJPY",
            "status": "TRAINED",
            "model": json_value(model),
            "dataset_version": raw.metadata["dataset_version"],
            "validation": {"overall": {"constant_probability_baseline": 0.48}},
        },
        "verification.json": {"status": "VERIFIED"},
        "provenance.json": {
            "purpose": "synthetic fixture only",
            "schema": "astra.hf.sampled_provenance.v1",
            "status": "completed_sampled_comparison",
            "dataset_id": "synthetic",
            "dataset_revision": "fixture",
            "price_basis": "bid",
            "timestamp_convention": "candle_open",
            "availability_shift_seconds": 60,
            "whole_cache_authenticated": False,
            "samples": [
                {
                    "pair": "USDJPY",
                    "count": 2,
                    "zero_offset_ohlc_exact_matches": 2,
                    "minus_one_minute_ohlc_matches": 0,
                    "plus_one_minute_ohlc_matches": 0,
                }
            ],
        },
    }
    for name, value in values.items():
        (reference / name).write_text(json.dumps(value))
    return reference, rows, raw, model


def test_frozen_plan_cost_matrix_and_durable_real_engine_payloads(
    tmp_path, monkeypatch, setup
):
    reference, rows, raw, model = setup
    before = {p: p.read_bytes() for p in reference.iterdir()}
    output = tmp_path / "runs"
    original = runner.simulate

    def observed(data, frozen, **kwargs):
        plan = json.loads(next(output.glob("*/plan.json")).read_text())
        assert plan["model"]["trained_at"] == model.trained_at.isoformat()
        assert plan["costs_spread_slippage_pips"]["double"] == [2, 0.4]
        assert plan["source_timestamp_shift_minutes"] == 1
        assert data.candles[0].timestamp == raw.candles[0].timestamp + timedelta(
            minutes=1
        )
        assert frozen == model
        return original(data, frozen, **kwargs)

    monkeypatch.setattr(runner, "simulate", observed)
    summary = runner.run_experiment(
        output,
        reference_root=reference,
        provenance_path=reference / "provenance.json",
        bootstrap_replicates=100,
    )
    assert summary["status"] == "COMPLETED", summary.get("error")
    assert (
        len(summary["cases"]) == 24
    )  # continuous + independently restarted partial month
    assert summary["journal_payloads_verified"]
    assert not summary["assessment"]["profitability_proven"]
    repo = SQLiteRepository(
        str(Path(summary["output_directory"]) / "experiment.sqlite3")
    )
    try:
        assert len(repo.query(Entity.EVALUATION)) == 24
        assert len(repo.query(Entity.MODEL)) == 1
        assert len(repo.query(Entity.AUDIT)) == 2
        assert all(
            record.occurred_at >= model.trained_at
            for record in repo.query(Entity.EVALUATION)
        )
    finally:
        repo.close()
    assert all(p.read_bytes() == value for p, value in before.items())
    assert raw.metadata["price_basis"] == "unknown"
    assert rows[0]["timestamp"] == raw.candles[0].timestamp


def test_missing_month_endpoint_is_reported_not_retroactively_selected(setup):
    _, _, raw, model = setup
    from dataclasses import replace

    last = replace(raw.candles[-1], timestamp=datetime(2025, 2, 1, 12, tzinfo=UTC))
    data = replace(raw, candles=raw.candles + (last,))
    windows, omitted = runner._windows(data, model.validation_start)
    assert any(
        m["month"] == "2025-01" and m["required_end"] == "2025-01-31T22:00:00+00:00"
        for m in omitted
    )
    assert all(w["name"] != "2025-01" for w in windows)


def test_availability_reference_is_not_shifted_twice(tmp_path, monkeypatch, setup):
    reference, _, raw, model = setup
    from dataclasses import replace

    shifted = runner._availability_view(raw)
    shifted_model = replace(
        model,
        training_end=model.training_end + timedelta(minutes=1),
        validation_start=model.validation_start + timedelta(minutes=1),
    )
    plan_path = reference / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["source_timestamp_shift_minutes"] = 1
    plan_path.write_text(json.dumps(plan))
    report_path = reference / "USDJPY-M1-report.json"
    report = json.loads(report_path.read_text())
    report.update(
        source_timestamp_shift_minutes=1,
        raw_dataset_version=raw.metadata["dataset_version"],
        dataset_version=shifted.metadata["dataset_version"],
        model=json_value(shifted_model),
    )
    report_path.write_text(json.dumps(report))
    comparison_path = reference / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["reports"][0]["model"] = json_value(shifted_model)
    comparison_path.write_text(json.dumps(comparison))
    seen = []

    def inspect_start(data, start):
        assert data.candles == shifted.candles
        seen.append(start)
        return [], []

    monkeypatch.setattr(runner, "_windows", inspect_start)
    summary = runner.run_experiment(
        tmp_path / "runs",
        reference_root=reference,
        provenance_path=reference / "provenance.json",
    )
    assert summary["status"] == "COMPLETED", summary.get("error")
    assert seen == [shifted_model.validation_start]
    saved = json.loads((Path(summary["output_directory"]) / "plan.json").read_text())
    assert (
        saved["effective_training_label_end"] == shifted_model.training_end.isoformat()
    )
    assert saved["model"]["trained_at"] == model.trained_at.isoformat()
    assert saved["raw_dataset_version"] == raw.metadata["dataset_version"]


@pytest.mark.parametrize(
    "change",
    [{"dataset_revision": "wrong"}, {"availability_shift_seconds": 0}, {"samples": []}],
)
def test_unmatched_provenance_rejected_before_row_loading(
    tmp_path, monkeypatch, setup, change
):
    reference, _, _, _ = setup
    path = reference / "provenance.json"
    audit = json.loads(path.read_text())
    audit.update(change)
    path.write_text(json.dumps(audit))

    def forbidden(*args):
        raise AssertionError("provenance must pass before loading rows")

    monkeypatch.setattr(runner, "_load_ticker_rows", forbidden)
    with pytest.raises(ValueError, match="provenance|sample evidence"):
        runner.run_experiment(
            tmp_path / "runs", reference_root=reference, provenance_path=path
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("failure", ["source", "interrupt", "journal"])
def test_interrupted_changed_or_unpersisted_run_cannot_complete(
    tmp_path, monkeypatch, setup, failure
):
    reference, _, _, _ = setup
    if failure == "source":
        values = iter(
            [
                json.loads((reference / "plan.json").read_text())["source"],
                {"changed": True},
            ]
        )
        monkeypatch.setattr(runner, "source_inventory", lambda: next(values))
    elif failure == "interrupt":

        def stop(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(runner, "simulate", stop)
    else:
        original = runner.Journal.record

        def unavailable(self, entity, at, value, **kwargs):
            if (
                entity == Entity.AUDIT
                and value.get("event") == "hf_profitability_finished"
            ):
                raise OSError("synthetic journal failure")
            return original(self, entity, at, value, **kwargs)

        monkeypatch.setattr(runner.Journal, "record", unavailable)
    result = runner.run_experiment(
        tmp_path / "runs",
        reference_root=reference,
        provenance_path=reference / "provenance.json",
        bootstrap_replicates=100,
    )
    assert result["status"] == ("INTERRUPTED" if failure == "interrupt" else "FAILED")
    assert not result["profitability_proven"]
    assert (
        json.loads((Path(result["output_directory"]) / "summary.json").read_text())[
            "status"
        ]
        == result["status"]
    )
