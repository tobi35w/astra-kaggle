"""Small causal market-context and regime analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .data import validate
from .indicators import atr, ema, macd, roc, rsi
from .models import MarketData, MarketSnapshot, Regime, Side


def analyze(
    data: MarketData,
    fast_period: int = 8,
    slow_period: int = 21,
    volatility_period: int = 14,
) -> MarketSnapshot:
    closes = [c.close for c in data.candles]
    fast, slow = ema(closes, fast_period), ema(closes, slow_period)
    fast_now, slow_now = fast[-1], slow[-1]
    trend = (
        Side.LONG
        if fast_now is not None and slow_now is not None and fast_now > slow_now
        else Side.SHORT
        if fast_now is not None and slow_now is not None and fast_now < slow_now
        else Side.FLAT
    )
    strength = min(
        abs(fast_now - slow_now) / closes[-1] * 100
        if fast_now is not None and slow_now is not None and closes[-1]
        else 0,
        1,
    )
    atr_values = atr(
        [c.high for c in data.candles],
        [c.low for c in data.candles],
        closes,
        volatility_period,
    )
    vol = atr_values[-1] or 0
    relative = vol / closes[-1] if closes[-1] else 0
    evidence = []
    if trend != Side.FLAT:
        evidence.append(f"{trend.value.lower()} EMA alignment")
    evidence.append(f"ATR={vol:.8g}")
    if fast_now is None or slow_now is None or len(closes) < slow_period + 2:
        regime = Regime.UNCLEAR
    elif relative > 0.02:
        regime = Regime.VOLATILE
    elif strength > 0.03:
        regime = Regime.TRENDING
    elif strength < 0.005:
        regime = Regime.RANGING
    else:
        regime = Regime.TRANSITIONAL
    return MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        trend,
        strength,
        vol,
        regime,
        tuple(evidence),
    )


def swing_points(
    values: list[float], left: int = 2, right: int = 2
) -> list[tuple[int, str, float]]:
    """Confirmed pivots; a pivot at i is not available until i+right."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be positive")
    points = []
    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        if values[i] == max(window) and window.count(values[i]) == 1:
            points.append((i + right, "high", values[i]))
        if values[i] == min(window) and window.count(values[i]) == 1:
            points.append((i + right, "low", values[i]))
    return sorted(points)


@dataclass(frozen=True)
class Structure:
    swings: tuple[tuple[int, str, float], ...]
    classifications: tuple[tuple[int, str], ...]
    support: float | None
    resistance: float | None
    breakout: str | None
    trend: Side = Side.FLAT
    range_low: float | None = None
    range_high: float | None = None
    failed_breakout: str | None = None
    pullback: str | None = None
    structural_break: str | None = None


def support_resistance_zones(
    data: MarketData, tolerance: float, left: int = 2, right: int = 2
) -> tuple[tuple[float, float, int], ...]:
    """Cluster confirmed swing prices into transparent [low, high, touch_count] zones."""
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    pivots = sorted(
        price
        for price in (
            [
                p
                for _, kind, p in swing_points(
                    [c.high for c in data.candles], left, right
                )
                if kind == "high"
            ]
            + [
                p
                for _, kind, p in swing_points(
                    [c.low for c in data.candles], left, right
                )
                if kind == "low"
            ]
        )
    )
    zones: list[list[float]] = []
    for price in pivots:
        if zones and price - zones[-1][1] <= tolerance:
            zones[-1][1] = price
            zones[-1][2] += 1
        else:
            zones.append([price, price, 1])
    return tuple((z[0], z[1], int(z[2])) for z in zones)


def structure(
    data: MarketData, left: int = 2, right: int = 2, tolerance: float = 0.0
) -> Structure:
    highs = [
        p
        for p in swing_points([c.high for c in data.candles], left, right)
        if p[1] == "high"
    ]
    lows = [
        p
        for p in swing_points([c.low for c in data.candles], left, right)
        if p[1] == "low"
    ]
    all_swings = sorted(highs + lows)
    labels = []
    prior: dict[str, float] = {}
    for index, kind, price in all_swings:
        previous = prior.get(kind)
        label = (
            "EH"
            if kind == "high"
            and previous is not None
            and abs(price - previous) <= tolerance
            else "EL"
            if kind == "low"
            and previous is not None
            and abs(price - previous) <= tolerance
            else "HH"
            if kind == "high" and previous is not None and price > previous + tolerance
            else "LH"
            if kind == "high" and previous is not None
            else "HL"
            if kind == "low" and previous is not None and price > previous + tolerance
            else "LL"
            if kind == "low" and previous is not None
            else "H"
            if kind == "high"
            else "L"
        )
        labels.append((index, label))
        prior[kind] = price
    support = lows[-1][2] if lows else None
    resistance = highs[-1][2] if highs else None
    last = data.candles[-1].close
    broken = (
        "up"
        if resistance is not None and last > resistance + tolerance
        else "down"
        if support is not None and last < support - tolerance
        else None
    )
    latest_high = next(
        (label for _, label in reversed(labels) if label in {"HH", "LH", "EH"}), "H"
    )
    latest_low = next(
        (label for _, label in reversed(labels) if label in {"HL", "LL", "EL"}), "L"
    )
    trend = (
        Side.LONG
        if (latest_high, latest_low) == ("HH", "HL")
        else Side.SHORT
        if (latest_high, latest_low) == ("LH", "LL")
        else Side.FLAT
    )
    prior_bars = data.candles[-21:-1]
    lo = min((c.low for c in prior_bars), default=None)
    hi = max((c.high for c in prior_bars), default=None)
    candle = data.candles[-1]
    failed = (
        "up"
        if hi is not None and candle.high > hi and candle.close <= hi
        else "down"
        if lo is not None and candle.low < lo and candle.close >= lo
        else None
    )
    previous_close = data.candles[-2].close if len(data.candles) > 1 else last
    pullback = (
        "bullish"
        if trend == Side.LONG
        and support is not None
        and support < last < previous_close
        else "bearish"
        if trend == Side.SHORT
        and resistance is not None
        and resistance > last > previous_close
        else None
    )
    return Structure(
        tuple(all_swings),
        tuple(labels),
        support,
        resistance,
        broken,
        trend,
        lo,
        hi,
        failed,
        pullback,
        broken,
    )


@dataclass(frozen=True)
class Momentum:
    rsi: float | None
    roc: float | None
    macd_histogram: float | None
    direction: Side
    evidence: tuple[str, ...]


def momentum(data: MarketData, period: int = 14) -> Momentum:
    closes = [c.close for c in data.candles]
    r = rsi(closes, period)[-1]
    change = roc(closes, period)[-1]
    _, _, hist = macd(closes)
    h = hist[-1]
    positive = sum(x is not None and x > 0 for x in (change, h))
    negative = sum(x is not None and x < 0 for x in (change, h))
    direction = (
        Side.LONG
        if positive > negative
        else Side.SHORT
        if negative > positive
        else Side.FLAT
    )
    return Momentum(
        r,
        change,
        h,
        direction,
        tuple(
            x
            for x in (
                "positive ROC"
                if change is not None and change > 0
                else "negative ROC"
                if change is not None and change < 0
                else None,
                "positive MACD histogram"
                if h is not None and h > 0
                else "negative MACD histogram"
                if h is not None and h < 0
                else None,
            )
            if x
        ),
    )


@dataclass(frozen=True)
class Volatility:
    atr: float
    relative_atr: float
    state: str


def volatility(data: MarketData, period: int = 14) -> Volatility:
    closes = [c.close for c in data.candles]
    values = atr(
        [c.high for c in data.candles], [c.low for c in data.candles], closes, period
    )
    value = values[-1] or 0.0
    relative = value / closes[-1] if closes and closes[-1] else 0.0
    state = (
        "high" if relative > 0.02 else "compressed" if relative < 0.001 else "normal"
    )
    return Volatility(value, relative, state)


def session_context(timestamp: datetime) -> str:
    """Approximate UTC forex sessions; daylight-saving overlap is intentionally simplified."""
    hour = timestamp.utctimetuple().tm_hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 16:
        return "LONDON_NEW_YORK_OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "ROLLOVER_LOW_LIQUIDITY"


@dataclass(frozen=True)
class MultiTimeframeContext:
    snapshots: dict[str, MarketSnapshot]
    aligned: bool
    conflicts: tuple[str, ...]
    decision_time: datetime | None = None
    execution_timeframe: str = "M5"
    lower_setup: str = "unknown"
    higher_trend: Side = Side.FLAT
    higher_structure: Side = Side.FLAT
    higher_regime: Regime = Regime.INSUFFICIENT_DATA
    agreement: float = 0.0
    structures: dict[str, Structure] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()


def multi_timeframe(
    data_by_timeframe: dict[str, MarketData],
    decision_time: datetime,
    required: tuple[str, ...] = ("M5", "M15", "H1", "H4"),
    execution_timeframe: str = "M5",
) -> MultiTimeframeContext:
    """Analyze only bars timestamped at or before the decision time."""
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        return MultiTimeframeContext(
            {}, False, ("decision timestamp must be timezone-aware",)
        )
    snapshots = {}
    structures = {}
    conflicts = []
    pair = next(iter(data_by_timeframe.values())).pair if data_by_timeframe else None
    for timeframe in required:
        data = data_by_timeframe.get(timeframe)
        if data is None:
            conflicts.append(f"missing timeframe {timeframe}")
            continue
        if data.timeframe != timeframe:
            conflicts.append(f"data timeframe label mismatch for {timeframe}")
            continue
        if pair is not None and data.pair != pair:
            conflicts.append(f"pair mismatch for {timeframe}")
            continue
        try:
            # Slice before validation: even a malformed future price must not alter the past.
            eligible = tuple(c for c in data.candles if c.timestamp <= decision_time)
            available = MarketData(data.pair, timeframe, eligible, data.metadata)
            validate(available, min_candles=1)
        except (ValueError, TypeError, AttributeError) as error:
            conflicts.append(f"invalid timeframe data for {timeframe}: {error}")
            continue
        from .history import TIMEFRAMES
        from .intelligence import detect_regime

        if len(eligible) < 30:
            conflicts.append(f"insufficient bars for {timeframe}")
        if (
            timeframe in TIMEFRAMES
            and decision_time - eligible[-1].timestamp > TIMEFRAMES[timeframe]
        ):
            conflicts.append(f"stale timeframe {timeframe}")
        snapshots[timeframe] = detect_regime(available)
        structures[timeframe] = structure(available)
        if structures[timeframe].trend not in (Side.FLAT, snapshots[timeframe].trend):
            conflicts.append(f"{timeframe} trend and structure conflict")
    directions = {s.trend for s in snapshots.values() if s.trend != Side.FLAT}
    if len(directions) > 1:
        conflicts.append("timeframe trend directions conflict")
    from .history import TIMEFRAMES

    ordered = sorted(snapshots, key=lambda tf: TIMEFRAMES[tf])
    higher = snapshots[ordered[-1]] if ordered else None
    higher_structure = structures.get("H1") or (
        structures[ordered[-1]] if ordered else None
    )
    counts = [
        sum(s.trend == direction for s in snapshots.values())
        for direction in (Side.LONG, Side.SHORT)
    ]
    agreement = max(counts, default=0) / len(required) if required else 0.0
    lower = structures.get("M15") or structures.get(execution_timeframe)
    return MultiTimeframeContext(
        snapshots,
        not conflicts,
        tuple(conflicts),
        decision_time,
        execution_timeframe,
        (lower.pullback or lower.breakout or "no confirmed setup")
        if lower
        else "unknown",
        higher.trend if higher else Side.FLAT,
        higher_structure.trend if higher_structure else Side.FLAT,
        higher.regime if higher else Regime.INSUFFICIENT_DATA,
        agreement,
        structures,
        tuple(
            f"{tf}: {snapshots[tf].trend.value}/{snapshots[tf].regime.value}"
            for tf in ordered
        ),
    )
