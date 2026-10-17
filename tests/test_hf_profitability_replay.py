"""Synthetic fixtures only: these tests establish behavior, not trading returns."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import log

import pytest

from astra.config import Settings
from astra.ml import Model, features
from astra.models import Candle, MarketData
from astra.risk import RiskEngine
from scripts.hf_profitability_replay import simulate

MINUTE = timedelta(minutes=1)
MONDAY = datetime(2025, 1, 6, 12, tzinfo=UTC)


def synthetic_data(count=40, *, first=MONDAY):
    return MarketData(
        "USDJPY",
        "M1",
        tuple(
            Candle(first + i * MINUTE, 100.0, 100.05, 99.95, 100.0)
            for i in range(count)
        ),
        {"price_basis": "unknown", "synthetic_fixture": True},
    )


def frozen_model(probability=0.7):
    return Model(
        "synthetic-frozen-model",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 1, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
        (0.0,) * 4,
        (1.0,) * 4,
        (0.0,) * 4,
        log(probability / (1 - probability)),
        0.0,
        0.0,
        1,
    )


def replay(data=None, model=None, **kwargs):
    data = synthetic_data() if data is None else data
    model = frozen_model() if model is None else model
    return simulate(data, model, start_at=data.candles[14].timestamp, **kwargs)


class SpyRisk:
    def __init__(self):
        self.calls = []
        self.engine = RiskEngine(
            Settings(
                trading_enabled=True,
                max_spread=1,
                max_slippage=1,
                max_position_size=100_000,
                max_pair_exposure=100_000,
                max_correlated_exposure=100_000,
            )
        )

    def approve(self, signal, state, spread=0, **kwargs):
        self.calls.append((signal, state, spread, kwargs))
        return self.engine.approve(signal, state, spread, **kwargs)


def test_uses_exact_past_features_and_real_model_creation_timestamp():
    data, original = synthetic_data(19), frozen_model()
    calls = []

    class SpyModel:
        version = original.version
        trained_at = original.trained_at
        training_end = original.training_end
        validation_start = original.validation_start

        def predict(self, values, at):
            calls.append((values, at))
            return original.predict(values, at)

    result = replay(data, SpyModel())
    assert calls == [
        (
            features(replace(data, candles=data.candles[:15]), MONDAY + 14 * MINUTE),
            original.trained_at,
        )
    ]
    assert result["model_trained_at"] == original.trained_at.isoformat()
    assert result["model_training_end"] == original.training_end.isoformat()
    assert result["source_metadata"] == data.metadata
    assert data.metadata["price_basis"] == "unknown"
    assert result["assumptions"]["future_continuity_filter"] is False
    assert not result["promotion_eligible"]


def test_next_bar_fill_and_elapsed_three_minute_exit():
    result = replay(synthetic_data(19))
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["decision_at"] == (MONDAY + 14 * MINUTE).isoformat()
    assert trade["opened_at"] == (MONDAY + 15 * MINUTE).isoformat()
    assert trade["closed_at"] == (MONDAY + 17 * MINUTE).isoformat()
    assert trade["reason"] == "time_exit"
    assert trade["planned_exit_at"] == trade["closed_at"]
    assert result["risk_counters"]["signal_approved"] == 1
    assert result["risk_counters"]["fill_approved"] == 1
    assert result["orders"][0]["final_available_at"] == trade["opened_at"]


def test_future_prices_do_not_change_earlier_decisions_or_trades():
    data = synthetic_data()
    changed = replace(
        data,
        candles=tuple(
            replace(
                c,
                open=c.open + 20,
                high=c.high + 20,
                low=c.low + 20,
                close=c.close + 20,
            )
            if i > 20
            else c
            for i, c in enumerate(data.candles)
        ),
    )
    before, after = replay(data), replay(changed)
    boundary = (MONDAY + 20 * MINUTE).isoformat()
    for key, field in (("risk_events", "decision_at"), ("trades", "closed_at")):
        assert [x for x in before[key] if x[field] <= boundary] == [
            x for x in after[key] if x[field] <= boundary
        ]


def test_future_gap_does_not_suppress_prior_pending_candidate():
    data = synthetic_data(24)
    changed = replace(data, candles=data.candles[:15] + data.candles[16:])
    result = replay(changed)
    assert result["risk_events"][0] == replay(data)["risk_events"][0]
    assert result["orders"][0]["status"] == "CANCELLED"
    assert (
        result["orders"][0]["final_available_at"] == (MONDAY + 16 * MINUTE).isoformat()
    )
    assert result["gaps"]["uncertain_pending_fills"] == 1
    assert result["gaps"]["uncertain_gap_exits"] == 0
    assert result["limits"]["data_gap_uncertainty"]
    assert not result["trades"]


def test_observed_gap_closes_at_first_observed_open_without_invented_stop_path():
    data = synthetic_data(24)
    later = replace(data.candles[17], open=100.3, high=100.6, low=99.0, close=100.4)
    data = replace(data, candles=data.candles[:16] + (later,) + data.candles[18:])
    result = replay(data)
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["reason"] == "data_gap"
    assert trade["exit"] == pytest.approx(100.3 - 0.005 - 0.002)
    assert trade["closed_at"] == later.timestamp.isoformat()
    assert result["gaps"]["uncertain_gap_exits"] == 1
    assert result["gaps"]["uncertain_pending_fills"] == 0
    assert result["limits"]["data_gap_uncertainty"]
    assert not result["promotion_eligible"]


def test_fill_risk_checks_actual_entry_without_moving_original_stop_or_target():
    data = synthetic_data(18)
    expensive_open = replace(
        data.candles[15], open=100.3, high=100.35, low=100.25, close=100.3
    )
    data = replace(
        data, candles=data.candles[:15] + (expensive_open,) + data.candles[16:]
    )
    spy = SpyRisk()
    result = replay(data, risk_engine=spy)
    assert len(spy.calls) == 2
    signal, fill = spy.calls[0][0], spy.calls[1][0]
    assert signal.entry == 100.0
    assert fill.entry == pytest.approx(100.307)
    assert fill.stop_loss == signal.stop_loss
    assert fill.take_profit == signal.take_profit
    assert result["risk_events"][0]["approval"]["approved"]
    assert not result["risk_events"][1]["approval"]["approved"]
    assert "expected risk/reward is below 1:1" in result["orders"][0]["reason"]
    assert not result["trades"]


@pytest.mark.parametrize("price_basis", ["mid", "bid"])
@pytest.mark.parametrize("probability", [0.3, 0.7])
def test_spread_both_slippages_commission_and_total_pnl_units_reconcile(
    price_basis, probability
):
    result = replay(
        synthetic_data(19),
        frozen_model(probability),
        price_basis=price_basis,
        commission_per_unit=0.006,
    )
    trade = result["trades"][0]
    assert trade["pnl_per_unit"] == pytest.approx(-0.01 - 2 * 0.002 - 0.006)
    assert trade["fees"] == pytest.approx(0.006 * trade["units"])
    assert trade["pnl"] == pytest.approx(trade["pnl_per_unit"] * trade["units"])
    assert trade["units"] == trade["size"] == 10_000
    assert result["initial_equity"] - result["final_equity"] == pytest.approx(
        -trade["pnl"]
    )
    assert result["risk_events"][0]["execution_cost_per_unit"] == pytest.approx(0.02)
    expected_exit_cost = 0.013 if price_basis == "mid" else 0.008
    assert result["risk_events"][1]["execution_cost_per_unit"] == pytest.approx(
        expected_exit_cost
    )
    assert result["reconciliation"]["passed"]
    assert sum(d["pnl"] for d in result["daily"]) == pytest.approx(trade["pnl"])
    assert result["account_currency"] == "JPY"


def test_explicit_units_cap_even_for_injected_risk_and_zero_costs():
    result = replay(
        synthetic_data(19), risk_engine=SpyRisk(), spread_pips=0, slippage_pips=0
    )
    assert result["trades"][0]["units"] == 10_000
    assert result["risk_events"][0]["explicit_size_cap_applied"]
    assert result["risk_events"][1]["explicit_size_cap_applied"]
    assert result["reconciliation"]["total_pnl"] == pytest.approx(0)


def test_consecutive_loss_guard_halts_permanently_and_flat_days_remain_in_curve():
    monday = synthetic_data(70)
    wednesday = synthetic_data(25, first=MONDAY + timedelta(days=2))
    data = replace(monday, candles=monday.candles + wednesday.candles)
    result = replay(data)
    assert len(result["trades"]) == 5
    assert result["rejection_reasons"]["consecutive-loss protection triggered"] == 1
    assert (
        result["limits"]["permanent_halt"]["available_at"]
        == (MONDAY + 29 * MINUTE).isoformat()
    )
    assert result["risk_counters"]["skipped_permanent_halt"] > 0
    assert result["risk_events"][-1]["state"]["consecutive_losses"] == 5
    assert len(result["daily"]) == 2
    assert result["daily"][1]["date"] == "2025-01-08"
    assert result["daily"][1]["observed_bars"] == 25
    assert result["daily"][1]["pnl"] == 0
    assert result["daily"][1]["end_equity"] == result["final_equity"]


def test_stop_first_when_fill_bar_touches_both_stop_and_target():
    data = synthetic_data(18)
    both = replace(data.candles[15], high=101.0, low=99.0)
    data = replace(data, candles=data.candles[:15] + (both,) + data.candles[16:])
    result = replay(data)
    trade = result["trades"][0]
    assert trade["reason"] == "stop_loss"
    assert trade["opened_at"] == trade["closed_at"] == both.timestamp.isoformat()
    assert trade["exit"] == pytest.approx(99.8 - 0.007)


def test_known_weekly_close_blocks_entries_before_three_minute_interval_reaches_it():
    first = datetime(2025, 1, 3, 21, 40, tzinfo=UTC)  # Friday, NY close 22:00 UTC.
    result = replay(synthetic_data(26, first=first))
    signals = [e for e in result["risk_events"] if e["phase"] == "signal"]
    assert len(signals) == 1
    assert signals[0]["decision_at"] == (first + 14 * MINUTE).isoformat()
    assert result["trades"][0]["closed_at"] == (first + 17 * MINUTE).isoformat()
    assert result["risk_counters"]["skipped_scheduled_closure"] > 0


def test_majority_baseline_uses_training_fraction_without_model_predictions():
    result = replay(
        synthetic_data(19),
        frozen_model(0.9),
        policy="training_majority",
        training_up_fraction=0.4,
    )
    assert result["trades"][0]["side"] == "SHORT"
    assert result["risk_events"][0]["signal"]["confidence"] == pytest.approx(0.6)


def test_timestamp_boundaries_and_missing_final_observation_fail_explicitly():
    data = synthetic_data(19)
    with pytest.raises(ValueError, match="after training"):
        replay(data, replace(frozen_model(), training_end=data.candles[14].timestamp))
    with pytest.raises(ValueError, match="at/after validation"):
        replay(
            data, replace(frozen_model(), validation_start=data.candles[15].timestamp)
        )
    with pytest.raises(ValueError, match="observed closing candle"):
        replay(data, end_at=data.candles[-1].timestamp + MINUTE)
    with pytest.raises(ValueError, match="timezone-aware"):
        replay(data, end_at=datetime(2025, 1, 6, 12))  # noqa: DTZ001
