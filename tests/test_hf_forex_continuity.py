"""Synthetic continuity/causality fixtures, never evidence of market performance."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from astra.ml import dataset, load_model
from astra.repositories import Entity
from scripts import hf_forex_continuity as experiment
from scripts.train_hf_forex_experiment import to_market_data

MINUTE = timedelta(minutes=1)
FROZEN_NOW = datetime(2026, 10, 16, 12, tzinfo=UTC)


def _rows(count=600, *, start=None, gaps=()):
    start = start or datetime(2025, 1, 6, 0, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        opening = 1.1 + math.sin((index - 1) * 0.3) * 0.001
        closing = 1.1 + math.sin(index * 0.3) * 0.001
        rows.append(
            {
                "timestamp": start
                + index * MINUTE
                + sum(
                    (duration for position, duration in gaps if index >= position),
                    timedelta(),
                ),
                "open": opening,
                "close": closing,
                "high": max(opening, closing) + 0.0001,
                "low": min(opening, closing) - 0.0001,
                "volume": 2.0,
                "ticker": "EURUSD",
            }
        )
    return rows


def _source(monkeypatch, rows):
    monkeypatch.setattr(experiment, "_load_ticker_rows", lambda pair, limit: rows)


def _freeze(monkeypatch):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FROZEN_NOW if tz is None else FROZEN_NOW.astimezone(tz)

    monkeypatch.setattr(experiment, "datetime", FrozenDateTime)


@pytest.mark.parametrize("gap_position", [200, 500])
def test_full_feature_and_label_gap_windows_excluded_from_each_partition(gap_position):
    rows = _rows(gaps=((gap_position, timedelta(hours=3)),))
    data = to_market_data(rows, "EURUSD")
    original = dataset(data)
    training, validation, counts, quality = experiment.continuous_split(data)
    retained = {example.timestamp for example in (*training, *validation)}
    expected_excluded = {
        rows[index]["timestamp"] for index in range(gap_position - 3, gap_position + 14)
    }

    assert not expected_excluded & retained
    assert rows[gap_position - 4]["timestamp"] in retained
    # Only ATR's predecessor close crosses this last excluded feature window.
    assert rows[gap_position + 13]["timestamp"] not in retained
    # A gap immediately before the first feature candle is outside the window.
    assert rows[gap_position + 14]["timestamp"] in retained
    assert counts["excluded_training_samples"] == (17 if gap_position == 200 else 0)
    assert counts["excluded_validation_samples"] == (17 if gap_position == 500 else 0)
    assert counts["purged_training_samples"] == 3
    assert quality["timestamp_gap_count"] == 1
    assert quality["largest_timestamp_gap_minutes"] == 181
    assert len(data.candles) == len(rows)
    assert counts["original_examples"] == len(original)
    assert all(
        example.label_available_at - example.timestamp == 3 * MINUTE
        for example in (*training, *validation)
    )


def test_weekend_gap_is_not_concatenated_into_training_features():
    # Friday's final close is 22:00 UTC; Sunday's first is 22:01 UTC (NY winter).
    rows = _rows(
        start=datetime(2025, 1, 3, 17, 1, tzinfo=UTC),
        gaps=((300, timedelta(days=2)),),
    )
    training, validation, counts, quality = experiment.continuous_split(
        to_market_data(rows, "EURUSD")
    )
    retained = {example.timestamp for example in (*training, *validation)}

    assert quality["closed_session_bar_count"] == 0
    assert quality["timestamp_gap_count"] == 1
    assert counts["excluded_training_samples"] == 17
    assert counts["excluded_validation_samples"] == 0
    assert rows[299]["timestamp"] not in retained
    assert rows[313]["timestamp"] not in retained
    assert rows[314]["timestamp"] in retained
    assert all(
        example.label_available_at - example.timestamp == 3 * MINUTE
        for example in (*training, *validation)
    )


def test_session_filter_rejects_weekend_bars_even_without_timestamp_gaps():
    rows = _rows(start=datetime(2025, 1, 3, 21, tzinfo=UTC))
    training, validation, counts, quality = experiment.continuous_split(
        to_market_data(rows, "EURUSD")
    )

    assert quality["timestamp_gap_count"] == 0
    assert quality["closed_session_bar_count"] == 539
    assert len(training) == 44
    assert not validation
    # Candle availability 22:00 covers the open 21:59-22:00 interval and is valid.
    assert training[-1].timestamp == datetime(2025, 1, 3, 21, 57, tzinfo=UTC)
    assert training[-1].label_available_at == datetime(2025, 1, 3, 22, tzinfo=UTC)
    assert (
        counts["excluded_validation_samples"] == counts["original_validation_samples"]
    )


def test_original_boundary_and_purge_stay_fixed_when_first_validation_windows_fail():
    rows = _rows(gaps=((425, timedelta(hours=3)),))
    data = to_market_data(rows, "EURUSD")
    original = dataset(data)
    split_index = int(len(original) * 0.7)
    fixed_boundary = original[split_index].timestamp
    training, validation, counts, _ = experiment.continuous_split(data)

    assert fixed_boundary == rows[422]["timestamp"]
    assert counts["fixed_validation_start"] == fixed_boundary.isoformat()
    assert counts["validation_start"] == fixed_boundary.isoformat()
    assert counts["retained_validation_start"] == rows[439]["timestamp"].isoformat()
    assert validation[0].timestamp > fixed_boundary
    assert counts["original_training_samples"] == split_index
    assert counts["original_validation_samples"] == len(original) - split_index
    assert counts["purged_training_samples"] == 3
    assert counts["training_samples"] == split_index - 3
    assert counts["excluded_validation_samples"] == 17
    assert all(example.label_available_at < fixed_boundary for example in training)
    assert not {
        example.timestamp for example in original[split_index - 3 : split_index]
    } & {example.timestamp for example in training}


def test_fit_uses_only_retained_training_and_standard_model_is_available_at_creation(
    monkeypatch, journal
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = _rows(gaps=((200, timedelta(hours=3)), (500, timedelta(hours=3))))
    _source(monkeypatch, rows)
    _freeze(monkeypatch)
    report = experiment.run(journal=journal)
    model = load_model(journal, pair="EURUSD", timeframe="M1", as_of=FROZEN_NOW)
    training, validation, _, _ = experiment.continuous_split(
        to_market_data(rows, "EURUSD")
    )
    expected_scaler = StandardScaler().fit([example.values for example in training])
    expected_estimator = LogisticRegression(random_state=0, max_iter=1000).fit(
        expected_scaler.transform([example.values for example in training]),
        [example.label for example in training],
    )

    assert report["status"] == "TRAINED"
    assert model is not None
    assert model.means == tuple(expected_scaler.mean_)
    assert model.scales == tuple(expected_scaler.scale_)
    assert model.coefficients == tuple(expected_estimator.coef_[0])
    assert model.trained_at == FROZEN_NOW
    assert model.training_end < model.validation_start
    assert model.sample_count == len(training) + len(validation)
    assert model.validation_accuracy == report["validation"]["overall"]["accuracy"]
    assert model.validation_brier == report["validation"]["overall"]["brier"]
    assert (
        load_model(journal, pair="EURUSD", timeframe="M1", as_of=model.training_end)
        is None
    )
    with pytest.raises(ValueError, match="did not exist"):
        model.predict(validation[0].values, validation[0].timestamp)
    records = journal.repository.query(Entity.MODEL)
    evaluations = journal.repository.query(Entity.EVALUATION)
    assert len(records) == len(evaluations) == 1
    assert records[0].occurred_at == evaluations[0].occurred_at == FROZEN_NOW
    assert evaluations[0].payload["kind"] == "hf_forex_continuity_experiment"
    assert report["promotion_eligible"] is False
    assert report["profitability_evaluated"] is False
    assert report["price_basis"] == "unknown"
    assert "previously consumed" in report["caveat"]
    assert sum(
        bucket["sample_count"] for bucket in report["validation"]["calibration"]
    ) == len(validation)
    assert sum(
        score["sample_count"]
        for score in report["validation"]["by_decision_month"].values()
    ) == len(validation)
    assert report["validation"]["overall"]["constant_probability_baseline"] == sum(
        example.label for example in training
    ) / len(training)


def test_future_validation_price_changes_do_not_change_scaling_fit_or_training_version(
    monkeypatch,
):
    rows = _rows(gaps=((425, timedelta(hours=3)),))
    _source(monkeypatch, rows)
    _freeze(monkeypatch)
    before = experiment.run()
    for index in range(422, len(rows)):
        # A different future price path changes source identity and validation;
        # the purged fit's latest training label uses the preceding candle 421.
        price = 1.11 + index * 0.00001
        rows[index].update(
            open=price, close=price + 0.00002, high=price + 0.0001, low=price - 0.0001
        )
    after = experiment.run()

    assert before["dataset_version"] != after["dataset_version"]
    assert before["training_content_version"] == after["training_content_version"]
    for field in (
        "version",
        "means",
        "scales",
        "coefficients",
        "intercept",
        "training_end",
    ):
        assert before["model"][field] == after["model"][field]
    assert before["validation"]["overall"] != after["validation"]["overall"]


def test_training_change_changes_version_and_same_inputs_are_deterministic(monkeypatch):
    rows = _rows()
    _source(monkeypatch, rows)
    _freeze(monkeypatch)
    first = experiment.run()
    assert experiment.run() == first
    rows[100]["high"] += 0.00001
    changed = experiment.run()

    assert changed["model"]["version"] != first["model"]["version"]
    assert changed["training_content_version"] != first["training_content_version"]


def test_model_and_report_never_precede_latest_source_availability(
    monkeypatch, journal
):
    rows = _rows(start=datetime(2030, 1, 7, 0, 1, tzinfo=UTC))
    _source(monkeypatch, rows)
    _freeze(monkeypatch)
    report = experiment.run(journal=journal)
    latest = rows[-1]["timestamp"]

    assert report["status"] == "TRAINED"
    assert datetime.fromisoformat(report["model"]["trained_at"]) == latest
    assert datetime.fromisoformat(report["completed_at"]) == latest
    assert journal.repository.query(Entity.MODEL)[0].occurred_at == latest
    assert journal.repository.query(Entity.EVALUATION)[0].occurred_at == latest


@pytest.mark.parametrize("count", [15, 20, 320])
def test_insufficient_clean_samples_record_evaluation_without_model(
    monkeypatch, journal, count
):
    # 320 source bars produce 303 examples; the fixed purge leaves only 300.
    # A gap removes 17 more, below the original experiment's 300 threshold.
    rows = _rows(count, gaps=((200, timedelta(hours=3)),))
    _source(monkeypatch, rows)
    report = experiment.run(journal=journal)

    assert report["status"] == "INSUFFICIENT_DATA"
    assert "model" not in report
    assert not journal.repository.query(Entity.MODEL)
    assert len(journal.repository.query(Entity.EVALUATION)) == 1
    counts = report["validation"]
    assert counts["training_samples"] + counts["validation_samples"] < 300
    assert counts["original_examples"] == (
        counts["purged_training_samples"]
        + counts["excluded_training_samples"]
        + counts["excluded_validation_samples"]
        + counts["training_samples"]
        + counts["validation_samples"]
    )


def test_retained_single_class_training_is_insufficient(monkeypatch, journal):
    rows = _rows()
    for index, row in enumerate(rows):
        price = 1.1 + index * 0.00001
        row.update(open=price, close=price, high=price + 0.0001, low=price - 0.0001)
    _source(monkeypatch, rows)
    report = experiment.run(journal=journal)

    assert report["status"] == "INSUFFICIENT_DATA"
    assert "both outcome classes" in report["reason"]
    assert not journal.repository.query(Entity.MODEL)
