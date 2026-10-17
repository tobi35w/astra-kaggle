from astra.backtest import BacktestCosts
from astra.models import Candle, MarketData
from scripts.run_curated_history_phase import (
    costs_for,
    experiment_id,
    starting_equity,
)


def test_costs_use_pair_pip_and_bid_basis():
    usd_costs = costs_for("EURUSD", 2.0)
    jpy_costs = costs_for("USDJPY", 2.0)

    assert usd_costs == BacktestCosts(0.0002, 0.00004, 0.0, 1.0, "bid")
    assert jpy_costs == BacktestCosts(0.02, 0.004, 0.0, 1.0, "bid")


def test_usdjpy_account_starts_in_quote_currency_at_training_cutoff():
    candles = tuple(
        Candle(
            timestamp,
            150.0,
            151.0,
            149.0,
            close,
            1.0,
            "provider",
        )
        for timestamp, close in enumerate((150.0, 151.0, 152.0), start=1)
    )
    data = MarketData("USDJPY", "H1", candles)

    assert starting_equity("USDJPY", data, 2) == 1_510_000.0
    assert starting_equity("EURUSD", data, 2) == 10_000.0


def test_experiment_identity_is_deterministic_and_scope_specific():
    first = experiment_id("EURUSD", "H1", "2022-01_to_2022-12", "base", "ema_trend")

    assert first == experiment_id(
        "EURUSD", "H1", "2022-01_to_2022-12", "base", "ema_trend"
    )
    assert first != experiment_id(
        "EURUSD", "H1", "2022-01_to_2022-12", "double_cost", "ema_trend"
    )
    assert first != experiment_id(
        "USDJPY", "H1", "2022-01_to_2022-12", "base", "ema_trend"
    )
