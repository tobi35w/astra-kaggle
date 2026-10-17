from dataclasses import replace
from datetime import timedelta

import pytest

from astra.config import Settings
from astra.decision import decide
from astra.execution import PaperExecutionProvider, PaperPortfolio
from astra.health import check_health
from astra.intelligence import detect_regime, select
from astra.market import multi_timeframe
from astra.ml import FEATURE_NAMES, Model
from astra.models import Action, RiskState, StrategyPerformance
from astra.pipeline import AstraPaperPipeline
from astra.repositories import Entity, Journal, MemoryRepository, PersistenceError
from astra.risk import RiskEngine
from astra.strategies import EmaTrend


def test_pipeline_tracks_trade_context_and_memory(journal, market_data):
    data = market_data()
    at = data.candles[-1].timestamp
    book = PaperPortfolio(PaperExecutionProvider(), 1000)
    pipe = AstraPaperPipeline(
        [EmaTrend()], RiskEngine(Settings(trading_enabled=True)), book, journal=journal
    )
    result = pipe.evaluate(data, RiskState(1000, 1000, 1000), decision_time=at)
    assert result.paper_position is not None
    price = result.paper_position.signal.stop_loss
    closed = pipe.mark_candle(
        at + timedelta(minutes=5), price + 0.001, price - 0.001, price, pair=data.pair
    )
    assert closed[0].regime == "TRENDING"
    assert (
        journal.repository.query(Entity.PERFORMANCE)[0].payload["regime"] == "TRENDING"
    )
    assert book.risk_state().consecutive_losses == 1


def test_pipeline_execution_and_memory_failures_return_no_trade(market_data):
    class BrokenExecutor(PaperExecutionProvider):
        def execute(self, *args, **kwargs):
            raise RuntimeError("execution fixture failure")

    data = market_data()
    at = data.candles[-1].timestamp
    pipe = AstraPaperPipeline(
        [EmaTrend()],
        RiskEngine(Settings(trading_enabled=True)),
        PaperPortfolio(BrokenExecutor(), 1000),
    )
    result = pipe.evaluate(data, RiskState(1000, 1000, 1000), decision_time=at)
    assert result.decision.action == Action.NO_TRADE
    assert result.decision.reason_codes == ("EXECUTION_UNCERTAINTY",)
    mismatch = pipe.evaluate(
        data, RiskState(1000, 1000, 1000), decision_time=at, value_per_price_unit=2
    )
    assert mismatch.decision.action == Action.NO_TRADE

    class BrokenMemory(MemoryRepository):
        def query(self, *args, **kwargs):
            raise PersistenceError("offline")

    pipe.journal = Journal(BrokenMemory())
    assert (
        pipe.evaluate(
            data, RiskState(1000, 1000, 1000), decision_time=at
        ).decision.action
        == Action.NO_TRADE
    )


def test_runtime_dependency_faults_future_context_and_stale_data_fail_closed(
    market_data, journal
):
    data = market_data()
    at = data.candles[-1].timestamp
    risk = RiskEngine(Settings(trading_enabled=True))
    state = RiskState(1000, 1000, 1000)
    future_context = multi_timeframe({"M5": data}, at + timedelta(minutes=1), ("M5",))
    decision = decide(
        data,
        [EmaTrend()],
        risk,
        state,
        decision_time=at,
        timeframe_context=future_context,
    )
    assert (
        decision.action == Action.NO_TRADE and "INVALID_DATA" in decision.reason_codes
    )
    assert (
        "STALE_DATA"
        in decide(
            data, [EmaTrend()], risk, state, decision_time=at + timedelta(days=1)
        ).reason_codes
    )
    assert (
        decide(
            data, [EmaTrend()], risk, state, spread=float("nan"), decision_time=at
        ).action
        == Action.NO_TRADE
    )
    assert (
        decide(data, [EmaTrend()], risk, state, decision_time=at, min_score=2).action
        == Action.NO_TRADE
    )

    class BrokenHealth(MemoryRepository):
        def healthy(self):
            raise RuntimeError("offline")

    assert (
        decide(
            data,
            [EmaTrend()],
            risk,
            state,
            decision_time=at,
            journal=Journal(BrokenHealth()),
        ).action
        == Action.NO_TRADE
    )
    invalid = replace(
        data, candles=(replace(data.candles[0], volume_type="bad"),) + data.candles[1:]
    )
    assert not check_health(invalid, now=at).healthy


def test_available_model_adds_context_and_missing_model_vetoes(market_data, journal):
    data = market_data()
    at = data.candles[-1].timestamp
    risk = RiskEngine(Settings(trading_enabled=True, ml_enabled=True))
    state = RiskState(1000, 1000, 1000)
    assert (
        decide(
            data, [EmaTrend()], risk, state, decision_time=at, journal=journal
        ).action
        == Action.NO_TRADE
    )
    model = Model(
        "fixture",
        at,
        at - timedelta(days=2),
        at - timedelta(days=1),
        (0,) * 4,
        (1,) * 4,
        (0.1,) * 4,
        0,
        0.5,
        0.25,
        100,
        FEATURE_NAMES,
    )
    journal.record(Entity.MODEL, at, model, pair=data.pair, timeframe=data.timeframe)
    result = decide(data, [EmaTrend()], risk, state, decision_time=at, journal=journal)
    assert (
        result.action == Action.TRADE
        and result.context["model"]["version"] == "fixture"
    )


def test_session_specific_paper_outcomes_adjust_selection(market_data):
    data = market_data()
    at = data.candles[-1].timestamp
    context = detect_regime(data)
    records = tuple(
        StrategyPerformance(
            "ema_trend",
            context.regime.value,
            data.pair,
            data.timeframe,
            1,
            -1,
            0,
            at - timedelta(minutes=i),
            outcome=-1,
        )
        for i in range(12)
    )
    baseline = select(data, context, [EmaTrend()])
    adjusted = select(data, context, [EmaTrend()], records)
    assert adjusted.candidates[0][0] < baseline.candidates[0][0]
    incompatible = tuple(replace(r, session="DIFFERENT_SESSION") for r in records)
    assert select(data, context, [EmaTrend()], incompatible) == baseline


def test_unsafe_settings_are_rejected():
    for config in (
        Settings(max_trade_risk_pct=0.1),
        Settings(max_daily_loss_pct=0),
        Settings(minimum_confidence=2),
        Settings(persistence_backend="unknown"),
    ):
        with pytest.raises(ValueError):
            config.validate()


def test_immediate_paper_fill_rechecks_risk_at_cost_adjusted_price(
    market_data, journal
):
    data = market_data()
    at = data.candles[-1].timestamp
    book = PaperPortfolio(PaperExecutionProvider(slippage=0.00005), 1000)
    pipe = AstraPaperPipeline(
        [EmaTrend()],
        RiskEngine(
            Settings(trading_enabled=True, max_spread=0.001, max_slippage=0.0001)
        ),
        book,
        journal=journal,
    )
    result = pipe.evaluate(
        data, RiskState(1000, 1000, 1000), decision_time=at, spread=0.0002
    )
    assert result.paper_position is not None
    assert result.paper_position.entry == pytest.approx(
        result.decision.signal.entry + 0.00015
    )
    assert "fill risk recheck" in result.decision.reason
    assert len(journal.repository.query(Entity.DECISION)) == 2


def test_immediate_fill_costs_can_veto_previously_acceptable_reward_risk(
    market_data, journal
):
    from astra.models import Side, Signal

    class MarginalReward:
        name = "marginal"

        def generate(self, data, context):
            price = data.candles[-1].close
            return Signal(
                self.name, Side.LONG, 0.8, price, price - 0.1, price + 0.105, "fixture"
            )

    data = market_data()
    book = PaperPortfolio(PaperExecutionProvider(), 1000)
    pipe = AstraPaperPipeline(
        [MarginalReward()],
        RiskEngine(Settings(trading_enabled=True, max_spread=0.02)),
        book,
        journal=journal,
    )
    result = pipe.evaluate(
        data,
        RiskState(1000, 1000, 1000),
        decision_time=data.candles[-1].timestamp,
        spread=0.01,
    )
    assert result.decision.action == Action.NO_TRADE
    assert "risk/reward" in result.decision.reason
    assert not book.positions
