"""Context-aware regime, confidence, and strategy selection services."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from itertools import pairwise

from .arena import FitnessWeights, rank
from .data import validate
from .market import (
    MultiTimeframeContext,
    analyze,
    momentum,
    multi_timeframe,
    session_context,
    structure,
    volatility,
)
from .models import (
    ConfidenceBreakdown,
    MarketData,
    MarketSnapshot,
    Regime,
    Side,
    Signal,
    StrategyPerformance,
)
from .strategies import Strategy


def detect_regime(
    data: MarketData, timeframe_context: MultiTimeframeContext | None = None
) -> MarketSnapshot:
    """Classify a regime from causal trend, path efficiency, and volatility evidence."""
    validate(data, min_candles=1)
    snapshot = analyze(data)
    if len(data.candles) < 30:
        return MarketSnapshot(
            snapshot.pair,
            snapshot.timeframe,
            snapshot.timestamp,
            snapshot.trend,
            snapshot.trend_strength,
            snapshot.volatility,
            Regime.INSUFFICIENT_DATA,
            ("fewer than 30 bars",),
            "insufficient",
            "valid",
        )
    closes = [c.close for c in data.candles[-20:]]
    net = abs(closes[-1] - closes[0])
    traveled = sum(abs(b - a) for a, b in pairwise(closes))
    efficiency = net / traveled if traveled else 0.0
    ranges = [c.high - c.low for c in data.candles[-20:]]
    baseline = sum(ranges[:-1]) / max(1, len(ranges) - 1)
    expansion = ranges[-1] / baseline if baseline else 0.0
    vol = volatility(data)
    price_structure = structure(data)
    mom = momentum(data)
    evidence = [
        f"path efficiency {efficiency:.2f}",
        f"relative ATR {vol.relative_atr:.4f}",
        f"structure {price_structure.trend.value}; break={price_structure.breakout}",
        f"momentum {mom.direction.value}",
        f"failed breakout={price_structure.failed_breakout}; pullback={price_structure.pullback}",
    ]
    if expansion >= 2 and price_structure.breakout:
        regime = Regime.BREAKOUT
        evidence.append("latest range exceeds twice recent average")
    elif vol.state == "high":
        regime = Regime.VOLATILE
    elif vol.state == "compressed":
        regime = Regime.QUIET
    elif efficiency > 0.55 and snapshot.trend != Side.FLAT:
        regime = Regime.TRENDING
    elif efficiency < 0.25:
        regime = Regime.RANGING
    else:
        regime = Regime.TRANSITIONAL
    uncertainty = (
        "weak"
        if efficiency < 0.15 or expansion == 0
        else "moderate"
        if regime == Regime.TRANSITIONAL
        else "strong"
    )
    if timeframe_context is not None:
        evidence.extend(timeframe_context.evidence)
        if timeframe_context.conflicts:
            regime, uncertainty = Regime.TRANSITION, "weak"
            evidence.extend(timeframe_context.conflicts)
    if mom.direction not in (snapshot.trend, Side.FLAT) and regime == Regime.TRENDING:
        regime, uncertainty = Regime.UNCLEAR, "weak"
    return MarketSnapshot(
        snapshot.pair,
        snapshot.timeframe,
        snapshot.timestamp,
        snapshot.trend,
        snapshot.trend_strength,
        snapshot.volatility,
        regime,
        tuple(evidence),
        "low"
        if uncertainty == "strong"
        else "moderate"
        if uncertainty == "moderate"
        else "high",
        "valid",
        0.8 if uncertainty == "strong" else 0.5 if uncertainty == "moderate" else 0.2,
        timeframe_context.conflicts if timeframe_context else (),
    )


def analyze_timeframes(
    data_by_timeframe: dict[str, MarketData],
    decision_time: datetime,
    required: tuple[str, ...] = ("M5", "M15", "H1", "H4"),
) -> MultiTimeframeContext:
    return multi_timeframe(data_by_timeframe, decision_time, required)


def confidence(
    market: float,
    regime: float,
    strategy: float,
    setup: float,
    execution: float,
    conflicts: tuple[str, ...] = (),
    timeframe_alignment: float = 0.5,
    supporting_evidence: tuple[str, ...] = (),
) -> ConfidenceBreakdown:
    parts = (market, regime, strategy, setup, execution)
    if any(not 0 <= p <= 1 for p in (*parts, timeframe_alignment)):
        raise ValueError("confidence inputs must be between zero and one")
    # Geometric aggregation penalizes one weak component. Conflicts reduce by 0.1 each.
    raw = (market * regime * strategy * setup * execution * timeframe_alignment) ** (
        1 / 6
    )
    overall = max(0.0, raw - 0.1 * len(conflicts))
    level = (
        "strong evidence"
        if overall >= 0.75
        else "moderate evidence"
        if overall >= 0.55
        else "weak evidence"
        if overall >= 0.35
        else "insufficient evidence"
    )
    if conflicts:
        level = "conflicting evidence"
    state = (
        "CONFLICTED"
        if conflicts
        else "STRONG"
        if overall >= 0.75
        else "MODERATE"
        if overall >= 0.55
        else "WEAK"
        if overall >= 0.35
        else "INSUFFICIENT"
    )
    return ConfidenceBreakdown(
        *parts,
        overall,
        level,
        conflicts,
        timeframe_alignment,
        state,
        supporting_evidence,
    )


@dataclass(frozen=True)
class Selection:
    candidates: tuple[tuple[float, Signal], ...]
    rejected: tuple[str, ...]
    reason: str


def select(
    data: MarketData,
    context: MarketSnapshot,
    strategies: list[Strategy],
    performance: tuple[StrategyPerformance, ...] = (),
    degradation_penalty: float = 0.15,
    weights: FitnessWeights = FitnessWeights(),
    min_fitness: float = 0.4,
) -> Selection:
    from .evidence import comparable_evidence

    performance = comparable_evidence(
        tuple(p for p in performance if p.observed_at <= context.timestamp)
    )
    historical = {}
    outcomes: dict[str, list[StrategyPerformance]] = {}
    session = session_context(context.timestamp)
    for item in sorted(performance, key=lambda record: record.observed_at):
        if item.observed_at > context.timestamp:
            continue
        if item.session in ("ALL", session) and (
            item.pair,
            item.timeframe,
            item.regime,
        ) == (
            data.pair,
            data.timeframe,
            context.regime.value,
        ):
            historical[item.strategy] = item
            if item.outcome is not None:
                outcomes.setdefault(item.strategy, []).append(item)
    for name, records in outcomes.items():
        count = sum(r.trades for r in records)
        if count:
            historical[name] = replace(
                historical[name],
                trades=count,
                expectancy=sum(r.expectancy * r.trades for r in records) / count,
                win_rate=sum(r.win_rate * r.trades for r in records) / count,
            )
    ranked = rank(data, context, strategies, weights)
    adjusted = []
    rejected = []
    for score, signal in ranked:
        history = historical.get(signal.strategy)
        if history and history.fitness is not None:
            score = 0.65 * score + 0.35 * history.fitness
        if history and history.trades >= 10 and history.expectancy < 0:
            score -= degradation_penalty
        from .learning import evaluate_degradation

        past = tuple(
            r
            for r in performance
            if r.strategy == signal.strategy
            and r.observed_at <= context.timestamp
            and r.session in ("ALL", session)
            and (r.pair, r.timeframe, r.regime)
            == (data.pair, data.timeframe, context.regime.value)
        )
        degradation = evaluate_degradation(past)
        if degradation.degraded:
            score -= degradation_penalty
            if degradation.severity == "DISABLED":
                rejected.append(signal.strategy + ": strategy degradation")
                continue
        if score < min_fitness:
            rejected.append(signal.strategy)
        else:
            adjusted.append((score, signal))
    adjusted.sort(key=lambda item: (-item[0], item[1].strategy))
    reason = (
        "no candidate met fitness threshold (weak setup or strategy degradation)"
        if not adjusted
        else "ranked by current setup with historical degradation penalty"
    )
    return Selection(tuple(adjusted), tuple(rejected), reason)
