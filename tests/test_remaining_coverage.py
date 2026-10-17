"""Focused edge cases for causal utilities and fail-closed guards."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from astra.config import Settings
from astra.data import DataValidationError
from astra.history import load_csv
from astra.indicators import (
    adx,
    bollinger_bands,
    ema,
    macd,
    rolling_volatility,
    true_ranges,
)
from astra.learning import evaluate_degradation
from astra.market import session_context, swing_points
from astra.models import Side, utc_now
from astra.risk import RiskEngine
from astra.strategies import StrategyRegistry, valid_signal


def test_indicator_validation_and_short_history():
    assert ema([1, 2], 3) == [None, None]
    for call in (
        lambda: true_ranges([1], [], [1]),
        lambda: rolling_volatility([1, 2], 1),
        lambda: bollinger_bands([1], 1, -1),
        lambda: adx([1], [], [1]),
        lambda: macd([1, 2], 2, 1),
    ):
        with pytest.raises(ValueError):
            call()
    assert adx([1], [0], [0]) == [None]


def test_history_rejects_gaps(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2025-01-01T00:00:00+00:00,1,2,0.5,1,10\n"
        "2025-01-01T00:02:00+00:00,1,2,0.5,1,10\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="missing candles"):
        load_csv(path, "EURUSD", "M1", reject_gaps=True)
    from astra.history import missing_candles

    assert len(missing_candles(load_csv(path, "EURUSD", "M1"))) == 1
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("astra.history.missing_candles", lambda _data: ((1, 2),))
        with pytest.raises(DataValidationError, match="missing candles"):
            load_csv(path, "EURUSD", "M1", reject_gaps=True)


def test_miscellaneous_utc_sessions_and_degradation_guards():
    from astra.indicators import rsi

    assert (
        session_context(datetime(2025, 1, 1, 13, tzinfo=UTC))
        == "LONDON_NEW_YORK_OVERLAP"
    )
    assert session_context(datetime(2025, 1, 1, 18, tzinfo=UTC)) == "NEW_YORK"
    assert (
        session_context(datetime(2025, 1, 1, 22, tzinfo=UTC))
        == "ROLLOVER_LOW_LIQUIDITY"
    )
    with pytest.raises(ValueError):
        swing_points([1, 2], 0, 1)
    with pytest.raises(ValueError):
        evaluate_degradation((), recent_count=10, minimum_records=10)
    assert utc_now().tzinfo is UTC
    assert rsi([1, 1, 1], 1)[-1] == 50
    assert rsi([1], 1) == [None]


def test_risk_rejects_flat_and_too_close_stop(market_data):
    from astra.models import RiskState, Signal

    risk = RiskEngine(Settings(trading_enabled=True, minimum_stop_distance=0.01))
    state = RiskState(1000, 1000, 1000)
    flat = Signal("x", Side.FLAT, 1, 1, 0.9, 1.2, "test")
    assert not risk.approve(flat, state, 0).approved
    close = Signal("x", Side.LONG, 1, 1, 0.999999, 1.2, "test")
    assert any("stop distance" in x for x in risk.approve(close, state, 0).reasons)
    assert not valid_signal(replace(flat, side=Side.FLAT))


def test_strategy_registry_and_invalid_reward_reward():
    from astra.models import Signal

    class Duplicate:
        name = "same"

        def generate(self, *_):
            return None

    registry = StrategyRegistry([Duplicate()])
    with pytest.raises(ValueError):
        registry.register(Duplicate())
    assert not valid_signal(Signal("bad", Side.LONG, 0.5, 1, 1.1, 1.2, "invalid"))
    assert len(registry.all()) == 1


def test_strategy_no_signal_edges(market_data):
    from astra.models import MarketSnapshot
    from astra.strategies import (
        Breakout,
        MeanReversion,
        MomentumContinuation,
        PriceActionRejection,
        RangeFade,
        Regime,
        VolatilityExpansion,
    )

    data = market_data(40)
    base = MarketSnapshot(
        "EURUSD",
        "M5",
        data.candles[-1].timestamp,
        Side.LONG,
        0.2,
        0.001,
        Regime.RANGING,
        (),
    )
    neutral = replace(base, trend=Side.FLAT)
    assert RangeFade().generate(data, base) is None
    assert Breakout(lookback=20).generate(data, replace(base, volatility=0)) is None
    assert MomentumContinuation().generate(data, neutral) is None
    flat_data = replace(
        data,
        candles=tuple(
            replace(c, close=1.1, open=1.1, high=1.1005, low=1.0995)
            for c in data.candles
        ),
    )
    assert MeanReversion().generate(flat_data, base) is None
    assert PriceActionRejection().generate(data, replace(base, volatility=0)) is None
    assert VolatilityExpansion().generate(data, base) is None


def test_strategy_range_and_candle_neutral_edges(market_data):
    from astra.models import MarketData, MarketSnapshot
    from astra.strategies import (
        MeanReversion,
        PriceActionRejection,
        RangeFade,
        Regime,
        VolatilityExpansion,
    )

    data = market_data(21)
    ctx = MarketSnapshot(
        "EURUSD",
        "M5",
        data.candles[-1].timestamp,
        Side.LONG,
        0.2,
        0.001,
        Regime.RANGING,
        (),
    )
    flat_candles = tuple(
        replace(c, open=c.close, high=c.close, low=c.close) for c in data.candles
    )
    flat_data = MarketData(data.pair, data.timeframe, flat_candles)
    assert RangeFade().generate(flat_data, ctx) is None
    quiet_last = replace(
        data.candles[-1], high=data.candles[-1].close, low=data.candles[-1].close
    )
    quiet_data = MarketData(
        data.pair, data.timeframe, data.candles[:-1] + (quiet_last,)
    )
    assert PriceActionRejection().generate(quiet_data, ctx) is None
    neutral_last = replace(
        data.candles[-1],
        open=data.candles[-1].close,
        high=data.candles[-1].close + 0.01,
        low=data.candles[-1].close - 0.01,
    )
    neutral_data = MarketData(
        data.pair, data.timeframe, data.candles[:-1] + (neutral_last,)
    )
    assert VolatilityExpansion().generate(neutral_data, ctx) is None
    flat_context = replace(ctx, volatility=2.0)
    assert RangeFade().generate(data, flat_context) is None
    assert MeanReversion().generate(data, flat_context) is None


def test_range_fade_flat_range(market_data):
    from astra.models import Candle, MarketData, MarketSnapshot, Regime
    from astra.strategies import RangeFade

    data = market_data(10)
    same = tuple(
        replace(c, open=1.1, high=1.1, low=1.1, close=1.1) for c in data.candles
    )
    flat = MarketData(data.pair, data.timeframe, same)
    ctx = MarketSnapshot(
        data.pair,
        data.timeframe,
        same[-1].timestamp,
        Side.FLAT,
        0,
        0.01,
        Regime.RANGING,
        (),
    )
    assert RangeFade().generate(flat, ctx) is None
    assert RangeFade().generate(flat, replace(ctx, volatility=0)) is None
    from astra.strategies import MeanReversion

    above_mean = replace(
        flat,
        candles=tuple(
            replace(c, close=1.2, open=1.2, high=1.2001, low=1.1999) for c in same
        ),
    )
    assert MeanReversion().generate(above_mean, ctx) is None
    centered = tuple(
        Candle(
            datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * i),
            1.5,
            2.0,
            1.0,
            1.5,
            1,
        )
        for i in range(10)
    )
    center_data = MarketData("EURUSD", "M5", centered)
    center_context = MarketSnapshot(
        "EURUSD", "M5", centered[-1].timestamp, Side.FLAT, 0, 0.1, Regime.RANGING, ()
    )
    assert RangeFade().generate(center_data, center_context) is None


def test_intelligence_conflicts_and_degradation_rejection(market_data, monkeypatch):
    from astra.intelligence import analyze_timeframes, detect_regime, select
    from astra.market import MultiTimeframeContext
    from astra.models import MarketSnapshot, Regime, Signal, StrategyPerformance

    data = market_data(40)
    stamp = data.candles[-1].timestamp
    context = MultiTimeframeContext({}, False, ("H1 conflict",), evidence=("fixture",))
    regime = detect_regime(data, context)
    assert regime.contradictions == ("H1 conflict",)
    analyze_timeframes({"M5": data}, stamp, required=("M5",))
    baseline = detect_regime(data)
    monkeypatch.setattr(
        "astra.intelligence.analyze",
        lambda *_: replace(baseline, trend=Side.LONG, regime=Regime.TRENDING),
    )
    monkeypatch.setattr(
        "astra.intelligence.volatility",
        lambda *_: type("V", (), {"relative_atr": 0.01, "state": "normal"})(),
    )
    monkeypatch.setattr(
        "astra.intelligence.structure",
        lambda *_: type(
            "S",
            (),
            {
                "trend": Side.FLAT,
                "breakout": None,
                "failed_breakout": None,
                "pullback": None,
            },
        )(),
    )
    monkeypatch.setattr(
        "astra.intelligence.momentum",
        lambda *_: type("M", (), {"direction": Side.SHORT})(),
    )
    assert detect_regime(data).regime == Regime.UNCLEAR
    snapshot = MarketSnapshot(
        data.pair, data.timeframe, stamp, Side.LONG, 0.5, 0.001, Regime.TRENDING, ()
    )
    signal = Signal("fixture", Side.LONG, 0.8, 1.1, 1.09, 1.12, "test")
    monkeypatch.setattr("astra.intelligence.rank", lambda *_a, **_kw: [(0.9, signal)])
    history = tuple(
        StrategyPerformance(
            "fixture",
            "TRENDING",
            data.pair,
            data.timeframe,
            1,
            1.0 if i < 10 else -1.0,
            0.0,
            stamp - timedelta(days=20 - i),
            win_rate=0.8 if i < 10 else 0.1,
            cost_sensitivity=0.0 if i < 10 else 0.8,
        )
        for i in range(20)
    )
    result = select(data, snapshot, [object()], performance=history)
    assert not result.candidates
    assert any("degradation" in item for item in result.rejected)


def test_memory_summary_skips_empty_trade_groups(journal):
    from astra.memory import PerformanceMemory
    from astra.models import StrategyPerformance

    at = datetime(2025, 1, 1, tzinfo=UTC)
    PerformanceMemory(journal).record(
        StrategyPerformance(
            "fixture",
            "RANGING",
            "EURUSD",
            "M5",
            0,
            0,
            0,
            at,
        )
    )
    assert PerformanceMemory(journal).summarize("EURUSD", "M5", at) == ()


def test_regime_and_timeframe_rejection_paths(market_data, monkeypatch):
    from astra.intelligence import detect_regime
    from astra.market import multi_timeframe
    from astra.models import Regime

    data = market_data(35)
    conflict = multi_timeframe(
        {"M5": data},
        data.candles[-1].timestamp,
        required=("M5",),
        execution_timeframe="M5",
    )
    snap = detect_regime(data, conflict)
    assert snap.regime in (
        Regime.TRANSITION,
        Regime.UNCLEAR,
        Regime.TRENDING,
        Regime.BREAKOUT,
        Regime.RANGING,
        Regime.VOLATILE,
    )
    wrong = replace(data, timeframe="M15")
    result = multi_timeframe({"M5": wrong}, data.candles[-1].timestamp, ("M5",))
    assert any("label mismatch" in reason for reason in result.conflicts)
    other_pair = replace(data, pair="GBPUSD")
    result = multi_timeframe(
        {"M5": data, "H1": replace(other_pair, timeframe="H1")},
        data.candles[-1].timestamp,
        ("M5", "H1"),
    )
    assert any("pair mismatch" in reason for reason in result.conflicts)
    too_short = replace(data, candles=data.candles[:5])
    result = multi_timeframe(
        {"M5": too_short}, too_short.candles[-1].timestamp, ("M5",)
    )
    assert any("insufficient bars" in reason for reason in result.conflicts)
    malformed = replace(data, candles=(replace(data.candles[0], high=float("nan")),))
    result = multi_timeframe({"M5": malformed}, data.candles[-1].timestamp, ("M5",))
    assert any("invalid timeframe" in reason for reason in result.conflicts)
    from astra.market import Structure
    from astra.models import MarketSnapshot

    snapshot = MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        Side.LONG,
        0.5,
        0.001,
        Regime.TRENDING,
        (),
    )
    monkeypatch.setattr("astra.intelligence.detect_regime", lambda *_: snapshot)
    monkeypatch.setattr(
        "astra.market.structure",
        lambda *_: Structure(
            (), (), None, None, None, Side.SHORT, None, None, None, None, None
        ),
    )
    result = multi_timeframe({"M5": data}, data.candles[-1].timestamp, ("M5",))
    assert any("trend and structure conflict" in reason for reason in result.conflicts)


def test_robustness_holdout_validation(market_data):
    from astra.robustness import chronological_holdout

    data = market_data(40)
    for fraction in (0, 1):
        with pytest.raises(ValueError, match="train_fraction"):
            chronological_holdout(data, fraction)
    with pytest.raises(ValueError, match="at least two"):
        chronological_holdout(market_data(3), 0.5)


def test_market_transitional_regime(market_data, monkeypatch):
    from astra.market import analyze

    data = market_data(40)

    def averages(_values, period):
        return [1.10011 if period == 8 else 1.1] * len(data.candles)

    monkeypatch.setattr("astra.market.ema", averages)
    monkeypatch.setattr("astra.market.atr", lambda *_a: [0.001] * len(data.candles))
    assert analyze(data).regime.value == "TRANSITION"


def test_replay_cancels_pending_final_order(market_data, monkeypatch):
    from astra.backtest import BacktestCosts
    from astra.config import Settings
    from astra.execution import OrderStatus
    from astra.models import Action, Decision, Regime, RiskApproval, Signal
    from astra.pipeline import replay_paper

    data = market_data(2)
    stamp = data.candles[0].timestamp
    signal = Signal(
        "fixture",
        Side.LONG,
        0.9,
        0.5,
        0.4,
        0.8,
        "test",
        order_type="LIMIT",
        expires_after_bars=10,
    )
    decision = Decision(
        stamp,
        data.pair,
        data.timeframe,
        Action.TRADE,
        "trade",
        Regime.TRENDING,
        signal,
        RiskApproval(True, (), 1),
    )
    monkeypatch.setattr("astra.pipeline.decide", lambda *_a, **_kw: decision)
    result = replay_paper(data, [], Settings(), BacktestCosts())
    assert result["orders"]
    assert result["orders"][0].status == OrderStatus.CANCELLED


def test_ml_causal_split_and_optional_dependency(market_data, monkeypatch):
    import builtins

    from astra.ml import Example, chronological_split, train

    stamp = datetime(2025, 1, 1, tzinfo=UTC)
    examples = tuple(
        Example(
            stamp + timedelta(minutes=i),
            stamp + timedelta(minutes=i + 2),
            (0.0,) * 8,
            i % 2,
        )
        for i in range(4)
    )
    with pytest.raises(ValueError, match="no training labels"):
        chronological_split(examples, 0.5)
    data = replace(market_data(400), metadata={"real_data": True})
    large = tuple(
        Example(
            stamp + timedelta(minutes=i),
            stamp + timedelta(minutes=i + 1),
            (0.0,) * 8,
            i % 2,
        )
        for i in range(400)
    )
    monkeypatch.setattr("astra.ml.dataset", lambda _data: large)
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("test optional dependency path")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    assert train(data, enabled=True, minimum_samples=100).status == "DISABLED"


def test_ml_model_schema_mismatch_fails_closed(journal):
    from astra.ml import load_model
    from astra.repositories import Entity

    at = datetime(2025, 1, 1, tzinfo=UTC)
    journal.record(
        Entity.MODEL,
        at,
        {
            "trained_at": at.isoformat(),
            "training_end": at.isoformat(),
            "validation_start": at.isoformat(),
            "means": [],
            "scales": [],
            "coefficients": [],
            "feature_schema": ["unexpected"],
            "version": "fixture",
            "intercept": 0,
            "validation_accuracy": 0.5,
            "validation_brier": 0.25,
            "sample_count": 1,
        },
        pair="EURUSD",
        timeframe="M5",
    )
    with pytest.raises(ValueError, match="schema or availability"):
        load_model(journal, pair="EURUSD", timeframe="M5", as_of=at)


def test_pipeline_execution_failure_and_persistence_fail_closed(
    market_data, monkeypatch
):
    from astra.execution import PaperExecutionProvider, PaperPortfolio
    from astra.models import Action, Decision, Regime, RiskApproval, RiskState, Signal
    from astra.pipeline import AstraPaperPipeline

    data = market_data(40)
    stamp = data.candles[-1].timestamp
    signal = Signal("fixture", Side.LONG, 0.9, 1.1, 1.09, 1.12, "test")
    decision = Decision(
        stamp,
        data.pair,
        data.timeframe,
        Action.TRADE,
        "trade",
        Regime.TRENDING,
        signal,
        RiskApproval(True, (), 1),
    )
    monkeypatch.setattr("astra.pipeline.decide", lambda *_a, **_kw: decision)

    class FailingRepository:
        def append(self, _decision):
            raise RuntimeError("storage unavailable")

    class FailingJournal:
        repository = type("R", (), {"query": lambda *_a, **_kw: ()})()

        def record(self, *_a, **_kw):
            raise RuntimeError("audit unavailable")

        def decision(self, *_a, **_kw):
            raise RuntimeError("audit unavailable")

    portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
    portfolio.closed.extend(
        (
            type("T", (), {"pnl": -1})(),
            type("T", (), {"pnl": 1})(),
            type("T", (), {"pnl": -1})(),
        )
    )
    monkeypatch.setattr(
        portfolio, "open", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError())
    )
    pipeline = AstraPaperPipeline(
        [], RiskEngine(Settings()), portfolio, FailingRepository(), FailingJournal()
    )
    result = pipeline.evaluate(data, RiskState(1000, 1000, 1000), decision_time=stamp)
    assert result.paper_position is None
    assert "PERSISTENCE_FAILURE" in result.decision.reason_codes


def test_pipeline_recheck_persists_decision(market_data, monkeypatch):
    from astra.database import InMemoryDecisionRepository
    from astra.execution import PaperExecutionProvider, PaperPortfolio
    from astra.models import Action, Decision, Regime, RiskApproval, RiskState, Signal
    from astra.pipeline import AstraPaperPipeline

    data = market_data(40)
    stamp = data.candles[-1].timestamp
    signal = Signal("fixture", Side.LONG, 0.9, 1.1, 1.09, 1.12, "test")
    decision = Decision(
        stamp,
        data.pair,
        data.timeframe,
        Action.TRADE,
        "trade",
        Regime.TRENDING,
        signal,
        RiskApproval(True, (), 1),
    )
    monkeypatch.setattr("astra.pipeline.decide", lambda *_a, **_kw: decision)
    repository = InMemoryDecisionRepository()
    portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
    pipeline = AstraPaperPipeline(
        [],
        RiskEngine(Settings(trading_enabled=True, max_spread=0.001)),
        portfolio,
        repository,
    )
    result = pipeline.evaluate(
        data, RiskState(1000, 1000, 1000), spread=0.0001, decision_time=stamp
    )
    assert result.decision.action == Action.TRADE, result.decision.reason
    assert repository.all()


def test_strategy_registry_rejects_non_signal_and_reward_sides():
    from astra.models import Signal
    from astra.strategies import risk_reward

    assert risk_reward(Signal("flat", Side.FLAT, 0.5, 1, 0.9, 1.2, "test")) == 0
    assert not valid_signal(object())


def test_volatility_expansion_requires_twenty_one_bars(market_data):
    from astra.models import MarketSnapshot, Regime
    from astra.strategies import VolatilityExpansion

    data = market_data(20)
    context = MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        Side.LONG,
        0.5,
        0.001,
        Regime.TRENDING,
        (),
    )
    assert VolatilityExpansion().generate(data, context) is None
