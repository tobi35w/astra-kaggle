"""Synthetic clock conversions; never read a market cache or fit a model."""

from datetime import UTC, datetime, timedelta

import pytest

from scripts import hf_forex_continuity as continuity
from scripts.hf_source_clock import availability_view
from scripts.train_hf_forex_experiment import to_market_data


def source_data():
    return to_market_data(
        [
            {
                "timestamp": datetime(2025, 1, 3, 21, 45, tzinfo=UTC)
                + timedelta(minutes=i),
                "open": 150.0,
                "high": 150.1,
                "low": 149.9,
                "close": 150.0,
                "volume": 2,
                "ticker": "USDJPY",
            }
            for i in range(20)
        ],
        "USDJPY",
    )


def test_availability_shift_preserves_raw_prices_and_records_distinct_identity():
    raw = source_data()
    original_metadata = dict(raw.metadata)
    shifted = availability_view(raw)
    assert raw.metadata == original_metadata
    assert shifted.metadata["raw_dataset_version"] == raw.metadata["dataset_version"]
    assert shifted.metadata["dataset_version"] != raw.metadata["dataset_version"]
    for before, after in zip(raw.candles, shifted.candles):
        assert after.timestamp == before.timestamp + timedelta(minutes=1)
        assert (after.open, after.high, after.low, after.close, after.volume) == (
            before.open,
            before.high,
            before.low,
            before.close,
            before.volume,
        )
    # Source openings 22:00 onward are outside the NY Friday session.
    _, _, _, quality = continuity.continuous_split(shifted)
    assert quality["closed_session_bar_count"] == 5


def test_zero_shift_keeps_legacy_identity_and_double_shift_is_rejected():
    raw = source_data()
    assert availability_view(raw, 0) is raw
    with pytest.raises(ValueError, match="already been shifted"):
        availability_view(availability_view(raw))


@pytest.mark.parametrize("shift", [-1, 2, 60])
def test_unsupported_clock_is_rejected_before_loading(monkeypatch, shift):
    def forbidden(*args):
        raise AssertionError("source must not be read for an invalid clock")

    monkeypatch.setattr(continuity, "_load_ticker_rows", forbidden)
    with pytest.raises(ValueError, match="timestamp shifts"):
        continuity.run(timestamp_shift_minutes=shift)
