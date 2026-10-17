"""Evidence-based performance memory and simple degradation monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .database import StrategyPerformanceRepository
from .models import StrategyPerformance


@dataclass(frozen=True)
class Degradation:
    degraded: bool
    baseline_expectancy: float
    recent_expectancy: float
    evidence_count: int
    reason: str
    severity: str = "HEALTHY"
    evidence: tuple[str, ...] = ()


def evaluate_degradation(
    records: tuple[StrategyPerformance, ...],
    recent_count: int = 10,
    minimum_records: int = 20,
    tolerance: float = 0.0,
    as_of=None,
) -> Degradation:
    if recent_count < 1 or minimum_records <= recent_count:
        raise ValueError("minimum_records must exceed positive recent_count")
    ordered = sorted(
        (r for r in records if as_of is None or r.observed_at <= as_of),
        key=lambda record: (record.observed_at, record.sequence),
    )
    if len(ordered) < minimum_records:
        return Degradation(
            False, 0.0, 0.0, len(ordered), "insufficient recorded performance history"
        )
    baseline = mean(record.expectancy for record in ordered[:-recent_count])
    recent = mean(record.expectancy for record in ordered[-recent_count:])
    degraded = recent < baseline - tolerance
    before, after = ordered[:-recent_count], ordered[-recent_count:]
    evidence = []
    if degraded:
        evidence.append("declining expectancy")
    if mean(r.win_rate for r in after) < mean(r.win_rate for r in before) - 0.1:
        evidence.append("declining win rate")
    same_drawdown_unit = len({r.drawdown_unit for r in ordered}) == 1
    drawdown_tolerance = 0.5 if ordered[-1].drawdown_unit == "R" else 0.02
    if same_drawdown_unit and (
        mean(r.max_drawdown for r in after)
        > mean(r.max_drawdown for r in before) + drawdown_tolerance
    ):
        evidence.append("increasing drawdown")
    if (
        mean(r.cost_sensitivity for r in after)
        > mean(r.cost_sensitivity for r in before) + 0.2
    ):
        evidence.append("increased cost sensitivity")
    degraded = bool(evidence)
    return Degradation(
        degraded,
        baseline,
        recent,
        len(ordered),
        "recent expectancy below baseline" if degraded else "no degradation detected",
        "DISABLED" if len(evidence) >= 3 else "PENALIZED" if degraded else "HEALTHY",
        tuple(evidence),
    )


def record_result(
    repository: StrategyPerformanceRepository, performance: StrategyPerformance
) -> None:
    repository.append(performance)
