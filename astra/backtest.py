"""Chronological next-open single-position backtester and research helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from math import isfinite, sqrt

from .data import validate
from .models import MarketData, Side
from .strategies import Strategy


@dataclass(frozen=True)
class BacktestCosts:
    spread: float = 0.0
    slippage: float = 0.0
    commission_per_unit: float = 0.0
    value_per_price_unit: float = 1.0
    price_basis: str = "mid"

    def __post_init__(self) -> None:
        if not all(
            isfinite(cost) and cost >= 0
            for cost in (self.spread, self.slippage, self.commission_per_unit)
        ):
            raise ValueError("backtest costs must be finite and non-negative")
        if not isfinite(self.value_per_price_unit) or self.value_per_price_unit <= 0:
            raise ValueError("price value conversion must be finite and positive")
        if self.price_basis not in ("mid", "bid"):
            raise ValueError("execution price basis must be mid or bid")


@dataclass(frozen=True)
class TradeResult:
    strategy: str
    side: Side
    entry: float
    exit: float
    pnl_per_unit: float
    reason: str
    entry_index: int = 0
    exit_index: int = 0
    size: float = 1.0
    risk_per_unit: float = 0.0
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    regime: str = "UNCLEAR"
    confidence: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None

    @property
    def pnl(self) -> float:
        return self.pnl_per_unit * self.size


@dataclass(frozen=True)
class BacktestReport:
    trades: tuple[TradeResult, ...]
    equity_curve: tuple[float, ...]
    metrics: dict[str, float]
    assumptions: BacktestCosts
    orders: tuple = ()
    marked_equity_curve: tuple[float, ...] = ()
    decisions: tuple[dict, ...] = ()
    execution_assumptions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionAssumptions:
    max_positions: int = 1
    warmup_bars: int = 30
    close_at_day_end: bool = False
    context_lookback_bars: int | None = None

    def __post_init__(self):
        if self.max_positions < 1 or self.warmup_bars < 1:
            raise ValueError("positions and warmup must be positive")
        if self.context_lookback_bars is not None and self.context_lookback_bars < 30:
            raise ValueError("context lookback must be at least 30 bars")


def run(
    data: MarketData, strategy: Strategy, costs: BacktestCosts = BacktestCosts()
) -> list[TradeResult]:
    """Compatibility unit-size report using the same risk-gated event simulator."""
    return list(
        _simulate(
            data, strategy, costs, 10_000, 0.005, ExecutionAssumptions(), unit_size=True
        ).trades
    )


def metrics(trades: list[TradeResult]) -> dict[str, float]:
    pnls = [t.pnl for t in trades]
    gains, losses = sum(x for x in pnls if x > 0), -sum(x for x in pnls if x < 0)
    wins = [x for x in pnls if x > 0]
    losing = [x for x in pnls if x < 0]
    equity, peak, max_dd = 0.0, 0.0, 0.0
    curve = []
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        curve.append(equity)
    mean = sum(pnls) / len(pnls) if pnls else 0.0
    stdev = (
        sqrt(sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1))
        if len(pnls) > 1
        else 0.0
    )
    return {
        "trade_count": float(len(pnls)),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "expectancy": mean,
        "profit_factor": gains / losses if losses else float("inf") if gains else 0.0,
        "average_win": sum(wins) / len(wins) if wins else 0.0,
        "average_loss": sum(losing) / len(losing) if losing else 0.0,
        "total_return": sum(pnls),
        "net_pnl": sum(pnls),
        "max_drawdown": max_dd,
        "risk_adjusted": mean / stdev if stdev else 0.0,
        "consistency": len(wins) / len(pnls) if pnls else 0.0,
    }


def run_report(
    data: MarketData,
    strategy: Strategy,
    costs: BacktestCosts = BacktestCosts(),
    initial_equity: float = 10_000.0,
    fixed_fraction: float = 0.005,
    *,
    execution: ExecutionAssumptions = ExecutionAssumptions(),
    risk_engine=None,
    journal=None,
    start_index: int | None = None,
    evidence_source: str = "BACKTEST",
    progress_callback=None,
) -> BacktestReport:
    if (
        not isfinite(initial_equity)
        or not isfinite(fixed_fraction)
        or initial_equity <= 0
        or not 0 < fixed_fraction <= 0.01
    ):
        raise ValueError(
            "initial equity must be positive and per-trade risk must be within (0, 1%]"
        )
    report = _simulate(
        data,
        strategy,
        costs,
        initial_equity,
        fixed_fraction,
        execution,
        risk_engine=risk_engine,
        start_index=start_index,
        progress_callback=progress_callback,
    )
    if journal:
        from .repositories import Entity

        keys = {
            "pair": data.pair,
            "timeframe": data.timeframe,
            "strategy": strategy.name,
        }
        record = journal.record(
            Entity.BACKTEST, data.candles[-1].timestamp, report, **keys
        )
        journal.record(
            Entity.METRICS,
            data.candles[-1].timestamp,
            report.metrics,
            correlation_id=record.record_id,
            **keys,
        )
        from .evidence import record_experiment

        record_experiment(
            journal,
            data,
            strategy,
            report,
            source=evidence_source,
            initial_equity=initial_equity,
            fixed_fraction=fixed_fraction,
            start_index=start_index,
            risk_engine=risk_engine,
        )
    return report


def walk_forward(
    data: MarketData, train_size: int, test_size: int
) -> list[tuple[MarketData, MarketData]]:
    if train_size < 1 or test_size < 1:
        raise ValueError("window sizes must be positive")
    folds, start = [], 0
    while start + train_size + test_size <= len(data.candles):
        folds.append(
            (
                MarketData(
                    data.pair,
                    data.timeframe,
                    data.candles[start : start + train_size],
                    data.metadata,
                ),
                MarketData(
                    data.pair,
                    data.timeframe,
                    data.candles[start + train_size : start + train_size + test_size],
                    data.metadata,
                ),
            )
        )
        start += test_size
    return folds


def cost_sensitivity(
    data: MarketData, strategy: Strategy, costs: list[BacktestCosts]
) -> list[dict[str, float]]:
    return [metrics(run(data, strategy, cost)) for cost in costs]


def _simulate(
    data,
    strategy,
    costs,
    initial_equity,
    fraction,
    execution,
    *,
    unit_size=False,
    risk_engine=None,
    start_index=None,
    progress_callback=None,
):
    from copy import deepcopy

    from .config import Settings
    from .execution import (
        OrderStatus,
        PaperExecutionProvider,
        PaperOrder,
        PaperPortfolio,
    )
    from .intelligence import detect_regime
    from .risk import RiskEngine

    validate(data, 2)
    if data.metadata.get("price_basis", "mid") != costs.price_basis:
        raise ValueError(
            "OHLC simulation requires midpoint prices by default; explicitly select the matching price basis"
        )
    strategy = deepcopy(
        strategy
    )  # Isolate mutable strategy state between arena/cost/fold runs.
    risk = risk_engine or RiskEngine(
        Settings(
            trading_enabled=True,
            max_trade_risk_pct=fraction,
            max_spread=costs.spread,
            max_slippage=costs.slippage,
            max_position_size=1e12,
            max_pair_exposure=1e12,
            max_correlated_exposure=1e12,
            max_open_positions=execution.max_positions,
        )
    )
    portfolio = PaperPortfolio(
        PaperExecutionProvider(
            costs.slippage,
            costs.commission_per_unit,
            costs.value_per_price_unit,
            price_basis=costs.price_basis,
        ),
        initial_equity,
    )
    trades, curve, marked, decisions = [], [initial_equity], [initial_equity], []
    entry_indices, stop_distances, signals = {}, {}, {}
    realized = initial_equity
    start = (
        execution.warmup_bars
        if start_index is None
        else max(execution.warmup_bars, start_index)
    )
    # ExecutionAssumptions enforces warmup_bars >= 1, and max() preserves it.
    # A signal at start is observable at that close; execution begins on the following bar.
    for i, candle in enumerate(data.candles):
        if i < start:
            continue
        portfolio.advance_day(candle.timestamp)
        for position in portfolio.positions.values():
            position.last_price = portfolio.executor.mark_price(
                candle.open, position.signal, costs.spread
            )  # Known opening gap, before evaluating pending entries.
        opened = portfolio.process_orders(
            candle, data.pair, risk, costs.spread, data.timeframe
        )
        for position in opened:
            if unit_size:
                # Unit research mode never increases the independently approved size.
                original = position.size
                position.size = min(1.0, original)
                portfolio.balance += (
                    (original - position.size) * costs.commission_per_unit / 2
                )
            entry_indices[position.position_id] = i
            signals[position.position_id] = position.signal
            stop_distances[position.position_id] = abs(
                position.entry - position.signal.stop_loss
            )
        closed = list(
            portfolio.mark(
                candle.timestamp,
                candle.high,
                candle.low,
                candle.close,
                pair=data.pair,
                opening=candle.open,
                include_new=True,
            )
        )
        from .history import TIMEFRAMES

        at_day_end = (
            candle.timestamp + TIMEFRAMES[data.timeframe]
        ).date() != candle.timestamp.date()
        if i == len(data.candles) - 1 or (execution.close_at_day_end and at_day_end):
            for position_id in tuple(portfolio.positions):
                closed.append(
                    portfolio.close(
                        position_id,
                        candle.close,
                        candle.timestamp,
                        "end_of_data" if i == len(data.candles) - 1 else "end_of_day",
                    )
                )
        for trade in closed:
            trades.append(
                TradeResult(
                    trade.strategy,
                    Side(trade.side),
                    trade.entry,
                    trade.exit,
                    trade.pnl / trade.size,
                    trade.reason,
                    entry_indices[trade.position_id],
                    i,
                    trade.size,
                    stop_distances[trade.position_id],
                    trade.opened_at,
                    trade.closed_at,
                    trade.regime,
                    trade.confidence,
                    signals[trade.position_id].stop_loss,
                    signals[trade.position_id].take_profit,
                )
            )
            realized += trade.pnl
            curve.append(realized)
        marked.append(portfolio.equity)
        if progress_callback is not None and (i + 1) % 5000 == 0:
            progress_callback(
                {
                    "bar": i + 1,
                    "total_bars": len(data.candles),
                    "closed_trades": len(trades),
                    "equity": portfolio.equity,
                    "balance": portfolio.balance,
                    "open_positions": [
                        {
                            "pair": position.pair,
                            "timeframe": position.timeframe,
                            "side": position.signal.side.value,
                            "entry": position.entry,
                            "mark": position.last_price,
                            "size": position.size,
                            "unrealized_pnl": (position.last_price - position.entry)
                            * (1 if position.signal.side.value == "LONG" else -1)
                            * position.size
                            * costs.value_per_price_unit,
                            "opened_at": position.opened_at,
                            "stop_loss": position.signal.stop_loss,
                            "take_profit": position.signal.take_profit,
                        }
                        for position in portfolio.positions.values()
                    ],
                    "last_closed_trade": trades[-1] if trades else None,
                }
            )
        if i == len(data.candles) - 1:
            break
        prefix_start = (
            0
            if execution.context_lookback_bars is None
            else max(0, i + 1 - execution.context_lookback_bars)
        )
        prefix = replace(data, candles=data.candles[prefix_start : i + 1])
        context = detect_regime(prefix)
        spec = getattr(strategy, "specification", None)
        if spec and context.regime not in spec.suitable_regimes:
            decisions.append(
                {"index": i, "action": "NO_TRADE", "reason": "unsuitable strategy"}
            )
            continue
        try:
            signal = strategy.generate(prefix, context)
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            decisions.append(
                {"index": i, "action": "NO_TRADE", "reason": "strategy failure"}
            )
            continue
        if signal is None:
            decisions.append({"index": i, "action": "NO_TRADE", "reason": "no setup"})
            continue
        approval = risk.approve(
            signal,
            portfolio.risk_state(data.pair),
            costs.spread,
            expected_slippage=costs.slippage,
            value_per_price_unit=costs.value_per_price_unit,
            execution_cost_per_unit=costs.commission_per_unit
            + (costs.spread + 2 * costs.slippage) * costs.value_per_price_unit,
        )
        decisions.append(
            {
                "index": i,
                "action": "TRADE" if approval.approved else "NO_TRADE",
                "signal": signal,
                "risk": approval,
                "regime": context.regime.value,
            }
        )
        if approval.approved:
            portfolio.submit(
                signal, approval, candle.timestamp, data.pair, context.regime.value
            )
        else:
            order = PaperOrder(
                len(portfolio.orders) + 1,
                signal,
                candle.timestamp,
                data.pair,
                OrderStatus.REJECTED,
                reason="; ".join(approval.reasons),
            )
            portfolio.orders[order.order_id] = order
    for order in portfolio.orders.values():
        if order.status == OrderStatus.PENDING:
            order.status, order.reason = OrderStatus.CANCELLED, "end of dataset"
    result_metrics = metrics(trades)
    peak, max_dd = initial_equity, 0.0
    for value in marked:
        peak = max(peak, value)
        max_dd = max(max_dd, (peak - value) / peak)
    result_metrics["total_return_pct"] = (portfolio.equity / initial_equity - 1) * 100
    result_metrics["max_drawdown_pct"] = max_dd * 100
    result_metrics["net_pnl"] = portfolio.equity - initial_equity
    # Chronological quartile profitability measures consistency across time, not win rate.
    windows = [
        sum(t.pnl for t in trades if min(3, 4 * t.exit_index // len(data.candles)) == j)
        for j in range(4)
    ]
    result_metrics["consistency"] = sum(p > 0 for p in windows) / 4
    from dataclasses import asdict

    return BacktestReport(
        tuple(trades),
        tuple(curve),
        result_metrics,
        costs,
        tuple(portfolio.orders.values()),
        tuple(marked),
        tuple(decisions),
        asdict(execution),
    )
