"""Fail-safe coordinator connecting analysis, risk, audit, and paper execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .database import DecisionRepository
from .decision import decide
from .execution import PaperPortfolio, PaperPosition
from .market import MultiTimeframeContext
from .models import (
    Action,
    Decision,
    HealthStatus,
    MarketData,
    RiskState,
    StrategyPerformance,
)
from .repositories import Entity, Journal
from .risk import RiskEngine
from .strategies import Strategy


@dataclass(frozen=True)
class PipelineResult:
    decision: Decision
    paper_position: PaperPosition | None


class AstraPaperPipeline:
    """Coordinates paper decisions only; it has no live execution implementation."""

    def __init__(
        self,
        strategies: list[Strategy],
        risk: RiskEngine,
        portfolio: PaperPortfolio,
        decision_repository: DecisionRepository | None = None,
        journal: Journal | None = None,
    ):
        self.strategies = tuple(strategies)
        self.risk = risk
        self.portfolio = portfolio
        self.decision_repository = decision_repository
        self.journal = journal
        if journal:
            self.portfolio.journal = journal

    def evaluate(
        self,
        data: MarketData,
        risk_state: RiskState,
        *,
        spread: float = 0.0,
        expected_slippage: float = 0.0,
        value_per_price_unit: float | None = None,
        health: HealthStatus | None = None,
        performance: tuple[StrategyPerformance, ...] = (),
        timeframe_context: MultiTimeframeContext | None = None,
        decision_time: datetime | None = None,
    ) -> PipelineResult:
        if (
            data.metadata.get("price_basis", "mid")
            != self.portfolio.executor.price_basis
        ):
            from .models import utc_now

            return PipelineResult(
                Decision(
                    decision_time or utc_now(),
                    data.pair,
                    data.timeframe,
                    Action.NO_TRADE,
                    "data price basis does not match paper execution",
                    reason_codes=("EXECUTION_UNCERTAINTY",),
                ),
                None,
            )
        if (
            value_per_price_unit is not None
            and value_per_price_unit != self.portfolio.executor.value_per_price_unit
        ):
            from .models import utc_now

            return PipelineResult(
                Decision(
                    decision_time or utc_now(),
                    data.pair,
                    data.timeframe,
                    Action.NO_TRADE,
                    "execution price value does not match portfolio configuration",
                    reason_codes=("EXECUTION_UNCERTAINTY",),
                ),
                None,
            )
        if decision_time is not None:
            self.portfolio.advance_day(decision_time)
        local = self.portfolio.risk_state(data.pair)
        local_positions = tuple(self.portfolio.positions.values())
        local_losses = 0
        for trade in reversed(self.portfolio.closed):
            if trade.pnl >= 0:
                break
            local_losses += 1
        state = replace(
            risk_state,
            equity=self.portfolio.equity,
            peak_equity=max(risk_state.peak_equity, local.peak_equity),
            day_start_equity=max(risk_state.day_start_equity, local.day_start_equity),
            open_risk=max(risk_state.open_risk, local.open_risk),
            correlated_exposure=max(
                risk_state.correlated_exposure, local.correlated_exposure
            ),
            open_positions=max(risk_state.open_positions, len(local_positions)),
            open_exposure=max(
                risk_state.open_exposure,
                sum(position.size for position in local_positions),
            ),
            pair_exposure=max(
                risk_state.pair_exposure,
                sum(
                    position.size
                    for position in local_positions
                    if position.pair == data.pair
                ),
            ),
            consecutive_losses=max(risk_state.consecutive_losses, local_losses),
        )
        if self.journal and not performance:
            from .memory import PerformanceMemory

            try:
                performance = PerformanceMemory(self.journal).history(
                    data.pair, data.timeframe, decision_time
                )
            except Exception:  # noqa: BLE001 - fail closed at an external boundary.
                from .models import utc_now

                return PipelineResult(
                    Decision(
                        decision_time or utc_now(),
                        data.pair,
                        data.timeframe,
                        Action.NO_TRADE,
                        "performance persistence unavailable",
                        reason_codes=("PERSISTENCE_FAILURE",),
                    ),
                    None,
                )
        decision = decide(
            data,
            list(self.strategies),
            self.risk,
            state,
            spread,
            health=health,
            repository=self.decision_repository,
            performance=performance,
            timeframe_context=timeframe_context,
            decision_time=decision_time,
            expected_slippage=max(expected_slippage, self.portfolio.executor.slippage),
            value_per_price_unit=self.portfolio.executor.value_per_price_unit,
            execution_cost_per_unit=self.portfolio.executor.commission_per_unit
            + (spread + 2 * self.portfolio.executor.slippage)
            * self.portfolio.executor.value_per_price_unit,
            journal=self.journal,
        )
        if (
            decision.action.value != "TRADE"
            or decision.signal is None
            or decision.risk is None
        ):
            return PipelineResult(decision, None)
        signal, approval = decision.signal, decision.risk
        try:
            adjustment = (
                self.portfolio.executor.fill_price(signal.entry, signal, spread)
                - signal.entry
            )
            if adjustment:
                effective = replace(
                    decision.signal,
                    entry=decision.signal.entry + adjustment,
                )
                approval = self.risk.approve(
                    effective,
                    state,
                    spread,
                    expected_slippage=max(
                        expected_slippage, self.portfolio.executor.slippage
                    ),
                    value_per_price_unit=self.portfolio.executor.value_per_price_unit,
                    execution_cost_per_unit=self.portfolio.executor.commission_per_unit
                    + self.portfolio.executor.exit_cost(spread)
                    * self.portfolio.executor.value_per_price_unit,
                )
                decision = replace(
                    decision,
                    risk=approval,
                    action=Action.TRADE if approval.approved else Action.NO_TRADE,
                    reason="paper fill risk recheck: " + "; ".join(approval.reasons),
                    reason_codes=()
                    if approval.approved
                    else ("EXECUTION_UNCERTAINTY",),
                )
                if self.decision_repository:
                    self.decision_repository.append(decision)
                if self.journal:
                    self.journal.decision(decision)
                if not approval.approved:
                    return PipelineResult(decision, None)
            position = self.portfolio.open(
                signal,
                approval,
                decision.timestamp,
                spread,
                data.pair,
                data.timeframe,
                decision.regime.value,
            )
        except Exception as error:  # noqa: BLE001 - an execution fault must fail safely
            failed = replace(
                decision,
                action=Action.NO_TRADE,
                reason=f"paper execution failed: {type(error).__name__}",
                reason_codes=("EXECUTION_UNCERTAINTY",),
            )
            if self.decision_repository is not None:
                try:
                    self.decision_repository.append(failed)
                except Exception:  # noqa: BLE001 - fail closed at an external boundary.
                    failed = replace(
                        failed,
                        reason_codes=("EXECUTION_UNCERTAINTY", "PERSISTENCE_FAILURE"),
                    )
            if self.journal:
                try:
                    self.journal.record(
                        Entity.AUDIT, failed.timestamp, failed, pair=data.pair
                    )
                except Exception:  # noqa: BLE001 - fail closed at an external boundary.
                    failed = replace(
                        failed,
                        reason_codes=("EXECUTION_UNCERTAINTY", "PERSISTENCE_FAILURE"),
                    )
            return PipelineResult(failed, None)
        return PipelineResult(decision, position)

    def mark_candle(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
        *,
        pair=None,
        opening=None,
    ):
        """Update open paper positions from a newly completed OHLC candle."""
        closed = self.portfolio.mark(
            timestamp, high, low, close, pair=pair, opening=opening
        )
        if self.journal:
            from .memory import PerformanceMemory

            for trade in closed:
                PerformanceMemory(self.journal).observe_trade(trade)
        return closed


def replay_paper(
    data, strategies, settings, costs, initial_equity=10000.0, *, journal=None
):
    """Replay supplied candles chronologically through adaptive decisions and pending orders."""
    from .data import validate
    from .execution import OrderStatus, PaperExecutionProvider
    from .memory import PerformanceMemory

    validate(data)
    if data.metadata.get("price_basis", "mid") != costs.price_basis:
        raise ValueError(
            "Paper replay requires midpoint prices by default; explicitly select the matching price basis"
        )
    risk = RiskEngine(settings)
    portfolio = PaperPortfolio(
        PaperExecutionProvider(
            costs.slippage,
            costs.commission_per_unit,
            costs.value_per_price_unit,
            price_basis=costs.price_basis,
        ),
        initial_equity,
        journal,
    )
    decisions = []
    for i, candle in enumerate(data.candles):
        portfolio.advance_day(candle.timestamp)
        for position in portfolio.positions.values():
            position.last_price = portfolio.executor.mark_price(
                candle.open, position.signal, costs.spread
            )
        portfolio.process_orders(candle, data.pair, risk, costs.spread, data.timeframe)
        closed = portfolio.mark(
            candle.timestamp,
            candle.high,
            candle.low,
            candle.close,
            pair=data.pair,
            opening=candle.open,
            include_new=True,
        )
        if journal:
            for trade in closed:
                PerformanceMemory(journal).observe_trade(trade)
        prefix = replace(data, candles=data.candles[: i + 1])
        performance = (
            PerformanceMemory(journal).history(
                data.pair, data.timeframe, candle.timestamp
            )
            if journal
            else ()
        )
        decision = decide(
            prefix,
            strategies,
            risk,
            portfolio.risk_state(data.pair),
            costs.spread,
            decision_time=candle.timestamp,
            performance=performance,
            expected_slippage=costs.slippage,
            value_per_price_unit=costs.value_per_price_unit,
            execution_cost_per_unit=costs.commission_per_unit
            + (costs.spread + 2 * costs.slippage) * costs.value_per_price_unit,
            journal=journal,
        )
        decisions.append(decision)
        if decision.action == Action.TRADE and i < len(data.candles) - 1:
            portfolio.submit(
                decision.signal,
                decision.risk,
                candle.timestamp,
                data.pair,
                decision.regime.value,
            )
    for order in portfolio.orders.values():
        if order.status == OrderStatus.PENDING:
            portfolio.cancel(order.order_id, data.candles[-1].timestamp)
    if journal:
        portfolio.checkpoint(data.candles[-1].timestamp)
    return {
        "mode": "paper replay",
        "portfolio_id": portfolio.portfolio_id,
        "equity": portfolio.equity,
        "balance": portfolio.balance,
        "decisions": decisions,
        "orders": tuple(portfolio.orders.values()),
        "positions": tuple(portfolio.positions.values()),
        "trades": portfolio.closed,
        "assumptions": costs,
    }
