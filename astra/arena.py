"""Strategy candidate comparison with explicit, configurable evidence weights."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backtest import BacktestCosts, run_report
from .models import MarketData, MarketSnapshot, Signal
from .strategies import Strategy, risk_reward, valid_signal


@dataclass(frozen=True)
class FitnessWeights:
    signal_confidence: float = 0.5
    risk_reward: float = 0.3
    regime_match: float = 0.2


def rank(
    data: MarketData,
    context: MarketSnapshot,
    strategies: list[Strategy],
    weights: FitnessWeights = FitnessWeights(),
) -> list[tuple[float, Signal]]:
    found = []
    for strategy in strategies:
        spec = getattr(strategy, "specification", None)
        if spec and (
            context.regime not in spec.suitable_regimes
            or len(data.candles) < spec.required_bars
        ):
            continue
        try:
            signal = strategy.generate(data, context)
        except (ValueError, ArithmeticError):
            continue
        if signal is None or not valid_signal(signal):
            continue
        rr = risk_reward(signal)
        score = (
            weights.signal_confidence * signal.confidence
            + weights.risk_reward * min(rr / 3, 1)
            + weights.regime_match
            * (1.0 if spec and context.regime in spec.suitable_regimes else 0.25)
        )
        found.append((score, signal))
    return sorted(found, key=lambda item: (-item[0], item[1].strategy))


@dataclass(frozen=True)
class ArenaEvaluation:
    strategy: str
    pair: str
    timeframe: str
    metrics: dict[str, float]
    session_expectancy: dict[str, float]
    regime_expectancy: dict[str, float]
    fitness: float
    evidence: str
    cost_sensitivity: float = 0.0
    slippage_sensitivity: float = 0.0
    robustness: str = "INSUFFICIENT_DATA"
    out_of_sample_metrics: dict[str, float] = field(default_factory=dict)
    pair_expectancy: dict[str, float] = field(default_factory=dict)
    timeframe_expectancy: dict[str, float] = field(default_factory=dict)
    generalization_gap: float = 0.0


@dataclass(frozen=True)
class ArenaWeights:
    expectancy: float = 0.35
    drawdown: float = 0.30
    risk_adjusted: float = 0.20
    consistency: float = 0.15


def evaluate_strategy(
    data: MarketData,
    strategy: Strategy,
    costs: BacktestCosts = BacktestCosts(),
    weights: ArenaWeights = ArenaWeights(),
    *,
    journal=None,
) -> ArenaEvaluation:
    """Evaluate reproducibly on one pair/timeframe; no claim of out-of-sample edge."""
    report = run_report(data, strategy, costs, journal=journal)
    sessions: dict[str, list[float]] = {}
    regimes: dict[str, list[float]] = {}
    from .intelligence import detect_regime
    from .market import session_context
    from .models import MarketData

    for trade in report.trades:
        name = session_context(data.candles[trade.entry_index].timestamp)
        outcome = trade.pnl
        sessions.setdefault(name, []).append(outcome)
        prefix = MarketData(
            data.pair, data.timeframe, data.candles[: trade.entry_index], data.metadata
        )
        regime = detect_regime(prefix).regime.value
        regimes.setdefault(regime, []).append(outcome)
    session_expectancy = {
        name: sum(values) / len(values) for name, values in sessions.items()
    }
    regime_expectancy = {
        name: sum(values) / len(values) for name, values in regimes.items()
    }
    m = report.metrics
    # Bounded components avoid letting high raw return dominate. Thin evidence is penalized.
    exp_component = max(
        0.0, min(1.0, m["expectancy"] / (abs(m["average_loss"]) or 1.0) + 0.5)
    )
    dd_component = max(0.0, 1 - m["max_drawdown_pct"] / 5)
    risk_component = max(0.0, min(1.0, m["risk_adjusted"] / 2 + 0.5))
    evidence_component = min(1.0, m["trade_count"] / 30)
    from .robustness import robustness_report

    robustness = robustness_report(data, strategy, costs)
    fitness = (
        weights.expectancy * exp_component
        + weights.drawdown * dd_component
        + weights.risk_adjusted * risk_component
        + weights.consistency * m["consistency"]
    ) * evidence_component
    fitness *= 1 - 0.4 * robustness.cost_sensitivity
    fitness *= 1 - 0.25 * robustness.generalization_gap
    fitness *= (
        1.0
        if robustness.state == "ROBUST"
        else 0.65
        if robustness.state == "FRAGILE"
        else 0.75
    )
    evidence = (
        "insufficient trades"
        if m["trade_count"] < 10
        else "historical sample; validate out of sample"
    )
    evaluation = ArenaEvaluation(
        strategy.name,
        data.pair,
        data.timeframe,
        m,
        session_expectancy,
        regime_expectancy,
        fitness,
        evidence,
        robustness.cost_sensitivity,
        max(
            0.0,
            min(
                1.0,
                (
                    robustness.baseline_metrics["net_pnl"]
                    - robustness.stressed_metrics[1]["net_pnl"]
                )
                / max(abs(robustness.baseline_metrics["net_pnl"]), 1e-9),
            ),
        ),
        robustness.state,
        robustness.out_of_sample_metrics,
        {data.pair: m["expectancy"]},
        {data.timeframe: m["expectancy"]},
        robustness.generalization_gap,
    )
    if journal:
        from .repositories import Entity

        keys = {
            "pair": data.pair,
            "timeframe": data.timeframe,
            "strategy": strategy.name,
        }
        journal.record(
            Entity.EVALUATION, data.candles[-1].timestamp, evaluation, **keys
        )
        journal.record(
            Entity.FITNESS,
            data.candles[-1].timestamp,
            {"fitness": fitness, "evidence": evidence},
            **keys,
        )
    return evaluation


def compete(
    data: MarketData,
    strategies: list[Strategy],
    costs: BacktestCosts = BacktestCosts(),
    weights: ArenaWeights = ArenaWeights(),
    *,
    journal=None,
) -> tuple[ArenaEvaluation, ...]:
    return tuple(
        sorted(
            (
                evaluate_strategy(data, strategy, costs, weights, journal=journal)
                for strategy in strategies
            ),
            key=lambda item: (-item.fitness, item.strategy),
        )
    )
