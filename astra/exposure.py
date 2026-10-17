"""Pair correlation and portfolio concentration observations for independent risk inputs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import sqrt

from .models import Side


def returns(closes: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(b / a - 1 for a, b in pairwise(closes) if a != 0)


def correlation(first: tuple[float, ...], second: tuple[float, ...]) -> float | None:
    """Pearson correlation over paired simple returns; undefined for short/flat inputs."""
    n = min(len(first), len(second))
    if n < 2:
        return None
    a, b = first[-n:], second[-n:]
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    return cov / sqrt(var_a * var_b) if var_a and var_b else None


@dataclass(frozen=True)
class OpenExposure:
    pair: str
    side: Side
    notional: float


def correlated_exposure(
    target_pair: str,
    candidates: tuple[OpenExposure, ...],
    correlations: dict[str, float],
    threshold: float = 0.7,
) -> float:
    """Conservatively sum notionals of open pairs whose absolute correlation exceeds threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("correlation threshold must be within [0,1]")
    return sum(
        position.notional
        for position in candidates
        if position.pair != target_pair
        and abs(correlations.get(position.pair, 0.0)) >= threshold
    )
