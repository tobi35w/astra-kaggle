"""Causal, dependency-free indicator calculations."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def sma(values: Sequence[float], period: int) -> list[float | None]:
    _period(period)
    out: list[float | None] = []
    for i in range(len(values)):
        out.append(
            sum(values[i - period + 1 : i + 1]) / period if i + 1 >= period else None
        )
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    _period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2 / (period + 1)
    for i in range(period, len(values)):
        seed += alpha * (values[i] - seed)
        out[i] = seed
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    _period(period)
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = [max(values[i] - values[i - 1], 0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0) for i in range(1, len(values))]
    avg_g, avg_l = sum(gains[:period]) / period, sum(losses[:period]) / period
    out[period] = (
        100
        if avg_l == 0 and avg_g
        else 50
        if avg_l == avg_g == 0
        else 100 - 100 / (1 + avg_g / avg_l)
        if avg_l
        else 0
    )
    for i in range(period + 1, len(values)):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        out[i] = 100 - 100 / (1 + avg_g / avg_l) if avg_l else 100 if avg_g else 50
    return out


def true_ranges(
    high: Sequence[float], low: Sequence[float], close: Sequence[float]
) -> list[float]:
    if not (len(high) == len(low) == len(close)):
        raise ValueError("OHLC arrays must have equal lengths")
    return [
        high[i] - low[i]
        if i == 0
        else max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
        for i in range(len(high))
    ]


def atr(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    return sma(true_ranges(high, low, close), period)


def roc(values: Sequence[float], period: int = 1) -> list[float | None]:
    _period(period)
    return [
        ((values[i] / values[i - period]) - 1) * 100
        if i >= period and values[i - period] != 0
        else None
        for i in range(len(values))
    ]


def rolling_volatility(values: Sequence[float], period: int = 20) -> list[float | None]:
    _period(period)
    if period < 2:
        raise ValueError("sample volatility requires period >= 2")
    returns = [None] + [
        values[i] / values[i - 1] - 1 if values[i - 1] else None
        for i in range(1, len(values))
    ]
    out: list[float | None] = [None] * len(values)
    for i in range(period, len(values)):
        sample = returns[i - period + 1 : i + 1]
        valid = [x for x in sample if x is not None]
        if len(valid) == period:
            mean = sum(valid) / period
            out[i] = sqrt(sum((x - mean) ** 2 for x in valid) / (period - 1))
    return out


def bollinger_bands(values: Sequence[float], period: int = 20, deviations: float = 2.0):
    """Return rolling population-standard-deviation lower/middle/upper bands."""
    _period(period)
    if deviations < 0:
        raise ValueError("deviations cannot be negative")
    middle = sma(values, period)
    lower: list[float | None] = []
    upper: list[float | None] = []
    for i, center in enumerate(middle):
        if center is None:
            lower.append(None)
            upper.append(None)
            continue
        window = values[i - period + 1 : i + 1]
        variance = sum((x - center) ** 2 for x in window) / period
        width = sqrt(variance) * deviations
        lower.append(center - width)
        upper.append(center + width)
    return lower, middle, upper


def adx(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder-smoothed ADX; values are unavailable until enough directional history exists."""
    _period(period)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("OHLC arrays must have equal lengths")
    n = len(close)
    out: list[float | None] = [None] * n
    if n < 2 * period:
        return out
    plus_dm, minus_dm, ranges = [0.0], [0.0], true_ranges(high, low, close)
    for i in range(1, n):
        up, down = high[i] - high[i - 1], low[i - 1] - low[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    tr_s = sum(ranges[1 : period + 1])
    plus_s, minus_s = sum(plus_dm[1 : period + 1]), sum(minus_dm[1 : period + 1])
    dx: list[float | None] = [None] * n
    for i in range(period, n):
        if i > period:
            tr_s = tr_s - tr_s / period + ranges[i]
            plus_s = plus_s - plus_s / period + plus_dm[i]
            minus_s = minus_s - minus_s / period + minus_dm[i]
        plus_di = 100 * plus_s / tr_s if tr_s else 0.0
        minus_di = 100 * minus_s / tr_s if tr_s else 0.0
        total = plus_di + minus_di
        dx[i] = 100 * abs(plus_di - minus_di) / total if total else 0.0
    seed_end = 2 * period - 1
    current = sum(x for x in dx[period : 2 * period] if x is not None) / period
    out[seed_end] = current
    for i in range(seed_end + 1, n):
        current = (current * (period - 1) + (dx[i] or 0.0)) / period
        out[i] = current
    return out


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if fast >= slow:
        raise ValueError("fast period must be less than slow period")
    a, b = ema(values, fast), ema(values, slow)
    line = [x - y if x is not None and y is not None else None for x, y in zip(a, b)]
    valid = [x for x in line if x is not None]
    sig_valid = ema(valid, signal)
    sig_iter = iter(sig_valid)
    sig = [None if x is None else next(sig_iter) for x in line]
    hist = [
        x - y if x is not None and y is not None else None for x, y in zip(line, sig)
    ]
    return line, sig, hist


def _period(period: int) -> None:
    if period < 1:
        raise ValueError("period must be >= 1")
