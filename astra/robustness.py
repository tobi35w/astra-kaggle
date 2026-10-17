"""Chronological holdout and parameter/cost robustness helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .backtest import (
    BacktestCosts,
    cost_sensitivity,
    metrics,
    run,
    run_report,
    walk_forward,
)
from .models import MarketData
from .strategies import Strategy


@dataclass(frozen=True)
class HoldoutResult:
    train: MarketData
    test: MarketData
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]


def chronological_holdout(
    data: MarketData, train_fraction: float = 0.7
) -> tuple[MarketData, MarketData]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    split = int(len(data.candles) * train_fraction)
    if split < 2 or len(data.candles) - split < 2:
        raise ValueError("holdout requires at least two candles per split")
    from .models import MarketData as MD

    return MD(data.pair, data.timeframe, data.candles[:split], data.metadata), MD(
        data.pair, data.timeframe, data.candles[split:], data.metadata
    )


def evaluate_holdout(
    data: MarketData,
    strategy: Strategy,
    costs: BacktestCosts = BacktestCosts(),
    train_fraction: float = 0.7,
) -> HoldoutResult:
    train, test = chronological_holdout(data, train_fraction)
    return HoldoutResult(
        train,
        test,
        metrics(run(train, strategy, costs)),
        run_report(data, strategy, costs, start_index=len(train.candles) - 1).metrics,
    )


def evaluate_walk_forward(
    data: MarketData,
    strategy: Strategy,
    train_size: int,
    test_size: int,
    costs: BacktestCosts | None = None,
    *,
    fit: Callable[[MarketData], Strategy] | None = None,
    journal=None,
) -> tuple[WalkForwardFold, ...]:
    assumptions = costs or BacktestCosts()
    folds = walk_forward(data, train_size, test_size)
    results = []
    for train, test in folds:
        # Fitting sees only the training window. The resulting strategy is frozen for test.
        candidate = fit(train) if fit else strategy
        combined = MarketData(
            data.pair, data.timeframe, train.candles + test.candles, data.metadata
        )
        results.append(
            WalkForwardFold(
                train.candles[0].timestamp,
                train.candles[-1].timestamp,
                test.candles[0].timestamp,
                test.candles[-1].timestamp,
                run_report(train, candidate, assumptions).metrics,
                run_report(
                    combined,
                    candidate,
                    assumptions,
                    start_index=len(train.candles) - 1,
                    journal=journal,
                    evidence_source="OOS",
                ).metrics,
            )
        )
    if journal:
        from .repositories import Entity

        journal.record(
            Entity.WALK_FORWARD,
            data.candles[-1].timestamp,
            {"folds": results},
            pair=data.pair,
            timeframe=data.timeframe,
            strategy=strategy.name,
        )
    return tuple(results)


def parameter_sensitivity(
    data: MarketData,
    factory: Callable[[float], Strategy],
    values: tuple[float, ...],
    costs: BacktestCosts = BacktestCosts(),
) -> tuple[tuple[float, dict[str, float]], ...]:
    """Evaluate explicit parameter perturbations; no parameter fitting is performed here."""
    return tuple((value, metrics(run(data, factory(value), costs))) for value in values)


def cost_grid(
    data: MarketData,
    strategy: Strategy,
    spreads: tuple[float, ...],
    slippages: tuple[float, ...],
    commissions: tuple[float, ...] = (0.0,),
) -> tuple[dict[str, float], ...]:
    return tuple(
        result
        for spread in spreads
        for commission in commissions
        for result in cost_sensitivity(
            data,
            strategy,
            [
                BacktestCosts(
                    spread=spread, slippage=slippage, commission_per_unit=commission
                )
                for slippage in slippages
            ],
        )
    )


@dataclass(frozen=True)
class RobustnessReport:
    state: str
    evidence: tuple[str, ...]
    baseline_metrics: dict[str, float]
    stressed_metrics: tuple[dict[str, float], ...]
    out_of_sample_metrics: dict[str, float]
    cost_sensitivity: float
    parameter_results: tuple = ()
    generalization_gap: float = 0.0


def generalization_gap(training, holdout):
    """Bounded deterioration in scale-normalized expectancy; a warning, not proof of overfit."""

    def normalized(m):
        return m["expectancy"] / max(
            abs(m["average_win"]), abs(m["average_loss"]), 1e-9
        )

    return max(0.0, min(1.0, normalized(training) - normalized(holdout)))


def robustness_report(
    data,
    strategy,
    costs=BacktestCosts(),
    *,
    parameter_factory=None,
    parameters=(),
    minimum_trades=30,
    journal=None,
):
    from dataclasses import replace

    from .repositories import Entity

    train, _test = chronological_holdout(data)
    # All perturbation selection/inspection is restricted to training history.
    baseline = run_report(train, strategy, costs).metrics
    floor = train.candles[-1].close * 0.0001
    variants = (
        replace(costs, spread=max(costs.spread * 2, floor)),
        replace(costs, slippage=max(costs.slippage * 2, floor / 2)),
        replace(
            costs,
            commission_per_unit=max(
                costs.commission_per_unit * 2, floor * costs.value_per_price_unit
            ),
        ),
    )
    stressed = tuple(run_report(train, strategy, c).metrics for c in variants)
    oos = run_report(data, strategy, costs, start_index=len(train.candles) - 1).metrics
    gap = generalization_gap(baseline, oos)
    sensitivity = max(
        0.0,
        min(
            1.0,
            max((baseline["net_pnl"] - m["net_pnl"] for m in stressed), default=0)
            / max(abs(baseline["net_pnl"]), abs(baseline["average_loss"]), 1e-9),
        ),
    )
    params = (
        parameter_sensitivity(train, parameter_factory, parameters, costs)
        if parameter_factory
        else ()
    )
    insufficient = (
        baseline["trade_count"] < minimum_trades or oos["trade_count"] < minimum_trades
    )
    fragile = (
        oos["expectancy"] <= 0
        or any(m["expectancy"] <= 0 for m in stressed)
        or sensitivity > 0.5
        or gap > 0.5
        or oos["max_drawdown_pct"] > 10
        or any(m["expectancy"] <= 0 for _, m in params)
    )
    state = "INSUFFICIENT_DATA" if insufficient else "FRAGILE" if fragile else "ROBUST"
    result = RobustnessReport(
        state,
        (
            "chronological holdout; no test-set parameter fitting",
            "cost perturbations evaluated on training data",
            "heuristic diagnostic, not proof of future performance",
        ),
        baseline,
        stressed,
        oos,
        sensitivity,
        params,
        gap,
    )
    if journal:
        journal.record(
            Entity.ROBUSTNESS,
            data.candles[-1].timestamp,
            result,
            pair=data.pair,
            timeframe=data.timeframe,
            strategy=strategy.name,
        )
    return result
