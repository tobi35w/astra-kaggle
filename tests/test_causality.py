from dataclasses import replace
from datetime import timedelta

import pytest

from astra.config import Settings
from astra.decision import decide
from astra.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    roc,
    rolling_volatility,
    rsi,
    sma,
)
from astra.intelligence import detect_regime, select
from astra.market import multi_timeframe, structure, swing_points
from astra.ml import chronological_split, dataset, features
from astra.models import RiskState, Side, StrategyPerformance
from astra.risk import RiskEngine
from astra.strategies import EmaTrend, StrategyRegistry


def test_future_candles_cannot_change_past_decision_or_features(market_data):
    data = market_data(90, oscillate=True)
    at = data.candles[59].timestamp
    prefix = replace(data, candles=data.candles[:60])
    poisoned = replace(
        data,
        candles=data.candles[:60]
        + tuple(replace(c, close=float("nan")) for c in data.candles[60:]),
    )
    risk = RiskEngine(Settings(trading_enabled=True))
    args = (list(StrategyRegistry().all()), risk, RiskState(1000, 1000, 1000))
    assert decide(poisoned, *args, decision_time=at) == decide(
        prefix, *args, decision_time=at
    )
    assert features(poisoned, at) == features(prefix, at)
    assert multi_timeframe({"M5": poisoned}, at, ("M5",)) == multi_timeframe(
        {"M5": prefix}, at, ("M5",)
    )


def test_future_fitness_and_degradation_never_affect_past_selection(market_data):
    data = market_data()
    context = detect_regime(data)
    future = tuple(
        StrategyPerformance(
            "ema_trend",
            context.regime.value,
            data.pair,
            data.timeframe,
            100,
            -1000,
            0.9,
            context.timestamp + timedelta(days=i + 1),
            fitness=0,
        )
        for i in range(30)
    )
    assert select(data, context, [EmaTrend()], future) == select(
        data, context, [EmaTrend()]
    )
    past = (replace(future[0], observed_at=context.timestamp),)
    assert select(data, context, [EmaTrend()], past) != select(
        data, context, [EmaTrend()]
    )


def test_all_indicators_are_prefix_invariant(market_data):
    data = market_data(100, oscillate=True)
    closes = [c.close for c in data.candles]
    highs = [c.high for c in data.candles]
    lows = [c.low for c in data.candles]
    for end in (30, 50, 75):
        for calculate in (
            lambda x: sma(x, 8),
            lambda x: ema(x, 8),
            rsi,
            roc,
            rolling_volatility,
        ):
            assert calculate(closes)[:end] == calculate(closes[:end])
        for calculate in (atr, adx):
            assert calculate(highs, lows, closes)[:end] == calculate(
                highs[:end], lows[:end], closes[:end]
            )
        for calculate in (macd, bollinger_bands):
            for full, prefix in zip(calculate(closes), calculate(closes[:end])):
                assert full[:end] == prefix


def test_confirmed_structure_never_backdates_pivots(market_data):
    data = market_data(80, oscillate=True)
    full = structure(data)
    for end in (20, 40, 60):
        partial = structure(replace(data, candles=data.candles[:end]))
        assert tuple(p for p in full.swings if p[0] < end) == partial.swings
        assert (
            tuple(p for p in full.classifications if p[0] < end)
            == partial.classifications
        )
    # A high-series trough must never become a swing low.
    expected_highs = [
        p for p in swing_points([c.high for c in data.candles]) if p[1] == "high"
    ]
    assert [p for p in full.swings if p[1] == "high"] == expected_highs


def test_timeframe_roles_alignment_conflict_and_staleness(market_data):
    at = market_data().candles[-1].timestamp
    steps = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}
    frames = {
        tf: market_data(50, tf, start=at - timedelta(minutes=minutes * 49))
        for tf, minutes in steps.items()
    }
    aligned = multi_timeframe(frames, at)
    assert aligned.aligned and aligned.agreement == 1
    assert aligned.execution_timeframe == "M5" and aligned.higher_trend == Side.LONG
    assert aligned.higher_regime == detect_regime(frames["H4"]).regime
    frames["H1"] = market_data(
        50, "H1", direction=-1, start=at - timedelta(minutes=60 * 49)
    )
    conflict = multi_timeframe(frames, at)
    assert not conflict.aligned and conflict.agreement < aligned.agreement
    assert not multi_timeframe(frames, at + timedelta(days=2)).aligned
    assert not multi_timeframe({}, at).aligned
    assert not multi_timeframe(frames, at.replace(tzinfo=None)).aligned


def test_all_strategies_see_identical_prefix_context(market_data):
    data = market_data(90, oscillate=True)
    at = data.candles[59].timestamp
    prefix = replace(data, candles=data.candles[:60])
    context = detect_regime(prefix)
    for strategy in StrategyRegistry().all():
        result = strategy.generate(prefix, context)
        future = replace(
            data,
            candles=data.candles[:60]
            + tuple(replace(c, close=999) for c in data.candles[60:]),
        )
        sliced = replace(
            future, candles=tuple(c for c in future.candles if c.timestamp <= at)
        )
        assert strategy.generate(sliced, detect_regime(sliced)) == result


def test_ml_labels_are_purged_at_chronological_boundary(market_data):
    data = market_data(100)
    rows = dataset(data, horizon=8)
    assert all(row.values == features(data, row.timestamp) for row in rows)
    train, test = chronological_split(rows)
    assert max(e.label_available_at for e in train) < test[0].timestamp
    assert all(e.timestamp < e.label_available_at for e in rows)
    with pytest.raises(ValueError):
        chronological_split(tuple(reversed(rows)))
    with pytest.raises(ValueError):
        dataset(market_data(), horizon=0)
