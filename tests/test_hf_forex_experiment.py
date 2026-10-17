"""Deterministic synthetic fixtures for the HF experiment; no market evidence."""

import json
import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from astra.data import DataValidationError
from scripts.train_hf_forex_experiment import to_market_data


def _rows(count=20):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 1.1 + index * 0.0001,
            "high": 1.101 + index * 0.0001,
            "low": 1.099 + index * 0.0001,
            "close": 1.1005 + index * 0.0001,
            "volume": 2.0,
            "ticker": "EURUSD",
        }
        for index in range(count)
    ]


def test_external_rows_become_availability_timestamped_m1_candles():
    data = to_market_data(_rows(), "EURUSD", max_rows=15)

    assert data.timeframe == "M1"
    assert len(data.candles) == 15
    assert data.candles[0].timestamp == datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
    assert data.candles[0].volume_type == "provider"
    assert "unspecified" in data.metadata["timestamp_convention"]
    assert data.metadata["real_data"] is True


def test_external_rows_reject_duplicate_timestamps():
    rows = _rows()
    rows[5]["timestamp"] = rows[4]["timestamp"]

    with pytest.raises(DataValidationError, match="strictly ordered"):
        to_market_data(rows, "EURUSD")


def test_external_rows_reject_nan_prices():
    rows = _rows()
    rows[10]["close"] = float("nan")

    with pytest.raises(DataValidationError, match="non-finite"):
        to_market_data(rows, "EURUSD")


def test_naive_timestamps_are_rejected_before_host_timezone_conversion():
    rows = _rows()
    rows[3]["timestamp"] = rows[3]["timestamp"].replace(tzinfo=None)

    with pytest.raises(DataValidationError, match="timezone-aware"):
        to_market_data(rows, "EURUSD")


def test_aware_source_offsets_preserve_the_same_instant_and_identity():
    rows = _rows()
    original = to_market_data(rows, "EURUSD")
    for row in rows:
        row["timestamp"] = row["timestamp"].astimezone(timezone(timedelta(hours=5)))
    converted = to_market_data(rows, "EURUSD")
    assert converted.candles == original.candles
    assert converted.metadata["dataset_version"] == original.metadata["dataset_version"]
    rows[-1]["close"] += 0.00001
    changed = to_market_data(rows, "EURUSD")
    assert changed.metadata["dataset_version"] != original.metadata["dataset_version"]


def _arrow_cache(tmp_path, monkeypatch, shards):
    import pyarrow as pa
    from pyarrow import ipc

    import scripts.train_hf_forex_experiment as runner

    cache = tmp_path / "cache"
    root = cache / "keithtwesigye___ohlcv-1m-forex" / "default" / "0.0.0" / "synthetic"
    root.mkdir(parents=True)
    for index, rows in enumerate(shards):
        table = pa.Table.from_pylist(rows)
        path = root / f"ohlcv-1m-forex-train-{index:05d}-of-{len(shards):05d}.arrow"
        with path.open("wb") as handle, ipc.new_stream(handle, table.schema) as writer:
            writer.write_table(table, max_chunksize=7)
    (root / "dataset_info.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "num_examples": sum(map(len, shards)),
                        "shard_lengths": list(map(len, shards)),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "CACHE", cache)
    return root


def test_local_arrow_reader_selects_tail_across_shards_and_keeps_tickers_separate(
    tmp_path, monkeypatch
):
    import scripts.train_hf_forex_experiment as runner

    rows = _rows(40)
    other = [dict(row, ticker="GBPUSD") for row in rows]
    _arrow_cache(tmp_path, monkeypatch, [rows[:20] + other, rows[20:]])
    actual = runner._load_ticker_rows("EURUSD", 25)

    assert actual == rows[-25:]
    inventory = runner.source_inventory()
    assert inventory["declared_rows"] == 80
    assert len(inventory["files"]) == 3
    assert all(len(item["sha256"]) == 64 for item in inventory["files"])


def test_local_arrow_reader_rejects_old_duplicates_even_when_outside_selected_tail(
    tmp_path, monkeypatch
):
    import scripts.train_hf_forex_experiment as runner

    rows = _rows(40)
    rows[2]["timestamp"] = rows[1]["timestamp"]
    _arrow_cache(tmp_path, monkeypatch, [rows[:20], rows[20:]])

    with pytest.raises(DataValidationError, match="strictly ordered"):
        runner._load_ticker_rows("EURUSD", 15)


def test_local_arrow_reader_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    import scripts.train_hf_forex_experiment as runner

    rows = _rows(40)
    root = _arrow_cache(tmp_path, monkeypatch, [rows[:20], rows[20:]])
    (root / "ohlcv-1m-forex-train-00001-of-00002.arrow").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        runner._load_ticker_rows("EURUSD", 15)


def test_majority_baseline_uses_training_distribution_despite_validation_class_reversal():
    from scripts.train_hf_forex_experiment import score_predictions

    result = score_predictions([1, 1], [0.8, 0.8], training_up_fraction=1 / 3)

    assert result["majority_baseline_class"] == 0
    assert result["majority_baseline_accuracy"] == 0
    assert result["accuracy"] == 1
    assert result["constant_probability_brier"] == pytest.approx(4 / 9)
    assert result["brier"] == pytest.approx(0.04)


def _oscillating_rows(pair="EURUSD", *, gap=False):
    rows = _rows(600)
    for index, row in enumerate(rows):
        opening = 1.1 + math.sin((index - 1) * 0.3) * 0.001
        closing = 1.1 + math.sin(index * 0.3) * 0.001
        row.update(
            ticker=pair,
            open=opening,
            close=closing,
            high=max(opening, closing) + 0.0001,
            low=min(opening, closing) - 0.0001,
        )
        if gap and index >= 500:
            row["timestamp"] += timedelta(hours=3)
    return rows


def test_experiment_preserves_loadable_model_schema_and_purges_labels(
    monkeypatch, journal
):
    import scripts.train_hf_forex_experiment as runner
    from astra.ml import load_model
    from astra.repositories import Entity

    monkeypatch.setattr(
        runner, "_load_ticker_rows", lambda pair, limit: _oscillating_rows(pair)
    )
    report = runner.run(journal=journal)

    assert report["status"] == "TRAINED"
    assert report["validation"]["purged_training_samples"] == 3
    assert (
        report["validation"]["training_label_end"]
        < report["validation"]["validation_start"]
    )
    assert report["validation"]["overall"]["accuracy"] == pytest.approx(
        report["model"]["validation_accuracy"]
    )
    assert report["validation"]["overall"]["brier"] == pytest.approx(
        report["model"]["validation_brier"]
    )
    available_at = datetime.fromisoformat(report["model"]["trained_at"])
    restored = load_model(journal, pair="EURUSD", timeframe="M1", as_of=available_at)
    assert restored.version == report["model"]["version"]
    assert len(journal.repository.query(Entity.MODEL)) == 1
    assert len(journal.repository.query(Entity.EVALUATION)) == 1
    assert report["promotion_eligible"] is False
    assert report["profitability_evaluated"] is False


def test_gap_diagnostics_exclude_the_entire_feature_and_label_window(monkeypatch):
    import scripts.train_hf_forex_experiment as runner

    monkeypatch.setattr(
        runner,
        "_load_ticker_rows",
        lambda pair, limit: _oscillating_rows(pair, gap=True),
    )
    report = runner.run()

    assert report["data_quality"]["missing_open_session_minutes"] == 180
    assert report["data_quality"]["status"] == "WARN"
    assert report["validation"]["labels_spanning_more_than_three_minutes"] == 3
    assert report["validation"]["excluded_from_continuous_subset"] == 17
    assert (
        report["validation"]["continuous_open_session_windows"]["sample_count"]
        == report["validation"]["validation_samples"] - 17
    )


def test_comparison_records_plan_before_training_and_roundtrips_local_journal(
    tmp_path, monkeypatch
):
    import scripts.compare_hf_forex_models as comparison
    import scripts.train_hf_forex_experiment as runner
    from astra.adapters import SQLiteRepository
    from astra.repositories import Entity

    monkeypatch.setattr(
        comparison, "source_inventory", lambda: {"purpose": "synthetic fixture only"}
    )
    monkeypatch.setattr(
        runner, "_load_ticker_rows", lambda pair, limit: _oscillating_rows(pair)
    )

    def observed_run(pair, **kwargs):
        plans = list(tmp_path.glob("*/plan.json"))
        assert len(plans) == 1
        assert json.loads(plans[0].read_text())["pairs"] == ["EURUSD", "USDJPY"]
        return runner.run(pair, **kwargs)

    monkeypatch.setattr(comparison, "run", observed_run)
    summary = comparison.run_comparison(
        tmp_path, pairs=("EURUSD", "USDJPY"), max_rows=600
    )

    assert summary["status"] == "COMPLETED"
    assert summary["source_identity_rechecked"] is True
    from pathlib import Path

    output = Path(summary["output_directory"])
    persisted = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert persisted == summary
    repository = SQLiteRepository(str(output / "experiment.sqlite3"))
    try:
        assert len(repository.query(Entity.MODEL)) == 2
        assert len(repository.query(Entity.EVALUATION)) == 2
        assert len(repository.query(Entity.AUDIT)) == 2
    finally:
        repository.close()


def test_comparison_source_change_prevents_completion(tmp_path, monkeypatch):
    import scripts.compare_hf_forex_models as comparison
    import scripts.train_hf_forex_experiment as runner

    inventories = iter([{"synthetic_version": 1}, {"synthetic_version": 2}])
    monkeypatch.setattr(comparison, "source_inventory", lambda: next(inventories))
    monkeypatch.setattr(
        runner, "_load_ticker_rows", lambda pair, limit: _oscillating_rows(pair)
    )
    summary = comparison.run_comparison(tmp_path, pairs=("EURUSD",), max_rows=600)

    assert summary["status"] == "FAILED"
    assert "snapshot changed" in summary["error"]
    assert summary["source_identity_rechecked"] is False
    assert summary["promotion_eligible"] is False


def test_comparison_interruption_is_not_reported_as_complete(tmp_path, monkeypatch):
    import scripts.compare_hf_forex_models as comparison

    monkeypatch.setattr(
        comparison, "source_inventory", lambda: {"purpose": "synthetic fixture only"}
    )

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(comparison, "run", interrupted)
    summary = comparison.run_comparison(tmp_path, pairs=("EURUSD",), max_rows=600)

    assert summary["status"] == "INTERRUPTED"
    assert not summary["reports"]
