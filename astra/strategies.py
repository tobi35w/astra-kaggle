"""Ten distinct causal strategy families sharing one signal contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from .market import structure
from .models import MarketData, MarketSnapshot, Regime, Side, Signal


class Strategy(Protocol):
    name: str

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None: ...


@dataclass(frozen=True)
class StrategySpec:
    identity: str
    family: str
    thesis: str
    suitable_regimes: tuple[Regime, ...]
    required_bars: int
    setup: str
    entry: str
    stop: str
    exit: str
    risk_assumptions: str = (
        "OHLC mid prices; stop may gap; independent risk approval required"
    )

    @property
    def unsuitable_regimes(self):
        return tuple(regime for regime in Regime if regime not in self.suitable_regimes)


SPECS = {
    "ema_trend": StrategySpec(
        "ema_trend",
        "trend",
        "Persistent EMA direction continues",
        (Regime.TRENDING,),
        30,
        "fast EMA agrees with trend",
        "next available open",
        "1.5 ATR",
        "2R target or stop",
    ),
    "breakout": StrategySpec(
        "breakout",
        "breakout",
        "Escape from prior range persists",
        (Regime.TRENDING, Regime.RANGING, Regime.BREAKOUT, Regime.TRANSITION),
        21,
        "close beyond prior range excluding current bar",
        "next open",
        "1.5 ATR",
        "2R or return into range",
    ),
    "pullback": StrategySpec(
        "pullback",
        "pullback",
        "Trend resumes after retracement",
        (Regime.TRENDING,),
        30,
        "rejection within 0.5 ATR of short mean",
        "next open",
        "1.25 ATR",
        "2R or structure break",
    ),
    "momentum": StrategySpec(
        "momentum",
        "momentum",
        "Directional displacement continues",
        (Regime.TRENDING, Regime.BREAKOUT),
        30,
        "eight-bar move exceeds ATR and agrees with trend",
        "next open",
        "1 ATR",
        "2R",
    ),
    "mean_reversion": StrategySpec(
        "mean_reversion",
        "mean reversion",
        "Range deviation returns to mean",
        (Regime.RANGING,),
        20,
        "20-bar mean deviation exceeds ATR",
        "next open",
        "1 ATR",
        "rolling mean",
    ),
    "range_fade": StrategySpec(
        "range_fade",
        "range",
        "Range edge rejects price",
        (Regime.RANGING,),
        10,
        "outer 15% of ten-bar range",
        "next open",
        "1 ATR",
        "range midpoint, at least 1R",
    ),
    "price_action": StrategySpec(
        "price_action",
        "price action",
        "Dominant wick indicates rejection",
        (Regime.RANGING, Regime.TRENDING, Regime.TRANSITION),
        10,
        "wick exceeds twice candle body",
        "next open",
        "max of ATR and candle range",
        "2R",
    ),
    "volatility": StrategySpec(
        "volatility",
        "volatility",
        "Directional range expansion continues",
        (Regime.BREAKOUT, Regime.VOLATILE, Regime.TRENDING),
        21,
        "range exceeds 1.5 times preceding mean",
        "next open",
        "1 ATR",
        "2R",
    ),
    "structure_continuation": StrategySpec(
        "structure_continuation",
        "market structure",
        "Trend resumes after a structural pullback to confirmed support",
        (Regime.TRENDING, Regime.TRANSITION),
        30,
        "HH/HL structure with price reclaimed above the confirmed swing low",
        "next open",
        "beyond the confirmed swing low",
        "2R or structural break",
    ),
    "session_opening_range": StrategySpec(
        "session_opening_range",
        "session",
        "Opening-range breakout persists during the high-liquidity overlap",
        (Regime.TRENDING, Regime.BREAKOUT, Regime.TRANSITION),
        30,
        "close breaks the first bars' range of the London/NY overlap session",
        "next open",
        "1.5 ATR",
        "2R or return into the opening range",
    ),
}


class DescribedStrategy:
    name: str

    @property
    def specification(self) -> StrategySpec:
        return SPECS[self.name]


class EmaTrend(DescribedStrategy):
    name = "ema_trend"

    def __init__(self, stop_atr: float = 1.5, reward_risk: float = 2.0):
        self.stop_atr, self.reward_risk = stop_atr, reward_risk

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if (
            context.regime != Regime.TRENDING
            or context.trend == Side.FLAT
            or context.volatility <= 0
        ):
            return None
        entry = data.candles[-1].close
        distance = context.volatility * self.stop_atr
        stop = entry - distance if context.trend == Side.LONG else entry + distance
        target = (
            entry + distance * self.reward_risk
            if context.trend == Side.LONG
            else entry - distance * self.reward_risk
        )
        return Signal(
            self.name,
            context.trend,
            min(0.75, 0.5 + context.trend_strength),
            entry,
            stop,
            target,
            "confirmed EMA trend context",
            self.reward_risk,
        )


class RangeFade(DescribedStrategy):
    name = "range_fade"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if (
            context.regime != Regime.RANGING
            or len(data.candles) < 10
            or context.volatility <= 0
        ):
            return None
        recent = data.candles[-10:]
        lo, hi, price = (
            min(c.low for c in recent),
            max(c.high for c in recent),
            recent[-1].close,
        )
        if hi == lo:
            return None
        side = (
            Side.LONG
            if price <= lo + 0.15 * (hi - lo)
            else Side.SHORT
            if price >= hi - 0.15 * (hi - lo)
            else Side.FLAT
        )
        if side == Side.FLAT:
            return None
        stop = (
            price - context.volatility
            if side == Side.LONG
            else price + context.volatility
        )
        target = (lo + hi) / 2
        rr = abs(target - price) / context.volatility
        if rr < 1:
            return None
        return Signal(
            self.name,
            side,
            0.5,
            price,
            stop,
            target,
            "range-edge reversion candidate",
            rr,
        )


class Breakout(DescribedStrategy):
    """Close through the preceding range, with the current candle excluded."""

    name = "breakout"

    def __init__(self, lookback: int = 20, stop_atr: float = 1.5):
        self.lookback, self.stop_atr = lookback, stop_atr

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if len(data.candles) <= self.lookback or context.volatility <= 0:
            return None
        prior = data.candles[-self.lookback - 1 : -1]
        close = data.candles[-1].close
        high, low = max(c.high for c in prior), min(c.low for c in prior)
        side = Side.LONG if close > high else Side.SHORT if close < low else Side.FLAT
        if side == Side.FLAT:
            return None
        distance = context.volatility * self.stop_atr
        stop = close - distance if side == Side.LONG else close + distance
        target = close + 2 * distance if side == Side.LONG else close - 2 * distance
        return Signal(
            self.name,
            side,
            0.65,
            close,
            stop,
            target,
            "close broke the prior range",
            2,
            "price returns inside broken range",
        )


class TrendPullback(DescribedStrategy):
    """Enter only when trend is intact and price has retraced toward its fast mean."""

    name = "pullback"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if (
            context.regime != Regime.TRENDING
            or context.trend == Side.FLAT
            or len(data.candles) < 10
            or context.volatility <= 0
        ):
            return None
        closes = [c.close for c in data.candles]
        mean = sum(closes[-8:]) / 8
        candle = data.candles[-1]
        near_mean = abs(candle.close - mean) <= 0.5 * context.volatility
        rejection = (
            candle.close > candle.open
            if context.trend == Side.LONG
            else candle.close < candle.open
        )
        if not near_mean or not rejection:
            return None
        risk = 1.25 * context.volatility
        stop = (
            candle.close - risk if context.trend == Side.LONG else candle.close + risk
        )
        target = (
            candle.close + 2 * risk
            if context.trend == Side.LONG
            else candle.close - 2 * risk
        )
        return Signal(
            self.name,
            context.trend,
            0.62,
            candle.close,
            stop,
            target,
            "trend pullback reclaimed short mean",
            2,
            "trend structure breaks",
        )


class MomentumContinuation(DescribedStrategy):
    name = "momentum"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if (
            context.trend == Side.FLAT
            or context.volatility <= 0
            or len(data.candles) < 16
        ):
            return None
        closes = [c.close for c in data.candles]
        move = closes[-1] - closes[-9]
        side = (
            Side.LONG
            if move > 0 and context.trend == Side.LONG
            else Side.SHORT
            if move < 0 and context.trend == Side.SHORT
            else Side.FLAT
        )
        if side == Side.FLAT or abs(move) < context.volatility:
            return None
        risk = context.volatility
        return Signal(
            self.name,
            side,
            0.6,
            closes[-1],
            closes[-1] - risk if side == Side.LONG else closes[-1] + risk,
            closes[-1] + 2 * risk if side == Side.LONG else closes[-1] - 2 * risk,
            "momentum agrees with trend",
            2,
        )


class MeanReversion(DescribedStrategy):
    name = "mean_reversion"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if (
            context.regime != Regime.RANGING
            or len(data.candles) < 20
            or context.volatility <= 0
        ):
            return None
        prices = [c.close for c in data.candles[-20:]]
        mean = sum(prices) / len(prices)
        distance = prices[-1] - mean
        if abs(distance) < context.volatility:
            return None
        side = Side.SHORT if distance > 0 else Side.LONG
        stop = (
            prices[-1] + context.volatility
            if side == Side.SHORT
            else prices[-1] - context.volatility
        )
        return Signal(
            self.name,
            side,
            0.55,
            prices[-1],
            stop,
            mean,
            "range deviation reverting toward rolling mean",
            abs(distance) / context.volatility,
        )


class PriceActionRejection(DescribedStrategy):
    name = "price_action"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if len(data.candles) < 10 or context.volatility <= 0:
            return None
        c = data.candles[-1]
        body = abs(c.close - c.open)
        lower_wick = min(c.open, c.close) - c.low
        upper_wick = c.high - max(c.open, c.close)
        side = (
            Side.LONG
            if lower_wick > 2 * max(body, 1e-12)
            else Side.SHORT
            if upper_wick > 2 * max(body, 1e-12)
            else Side.FLAT
        )
        if side == Side.FLAT:
            return None
        risk = max(context.volatility, c.high - c.low)
        return Signal(
            self.name,
            side,
            0.52,
            c.close,
            c.close - risk if side == Side.LONG else c.close + risk,
            c.close + 2 * risk if side == Side.LONG else c.close - 2 * risk,
            "rejection wick dominates candle body",
            2,
        )


class VolatilityExpansion(DescribedStrategy):
    name = "volatility"

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if len(data.candles) < 21 or context.volatility <= 0:
            return None
        previous_range = sum(c.high - c.low for c in data.candles[-21:-1]) / 20
        current = data.candles[-1]
        if current.high - current.low <= 1.5 * previous_range:
            return None
        side = (
            Side.LONG
            if current.close > current.open
            else Side.SHORT
            if current.close < current.open
            else Side.FLAT
        )
        if side == Side.FLAT:
            return None
        risk = context.volatility
        return Signal(
            self.name,
            side,
            0.58,
            current.close,
            current.close - risk if side == Side.LONG else current.close + risk,
            current.close + 2 * risk if side == Side.LONG else current.close - 2 * risk,
            "range expanded beyond preceding average",
            2,
        )


class StructureContinuation(DescribedStrategy):
    """Enter structural pullbacks: trend from confirmed swings, stop beyond the swing level."""

    name = "structure_continuation"

    def __init__(self, reward_risk: float = 2.0):
        self.reward_risk = reward_risk

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        if len(data.candles) < 21:
            return None
        analysis = structure(data)
        price = data.candles[-1].close
        if (
            analysis.trend == Side.LONG
            and analysis.pullback == "bullish"
            and analysis.support is not None
        ):
            stop = analysis.support
            risk = price - stop
            if risk <= 0 or not isfinite(risk):
                return None
            return Signal(
                self.name,
                Side.LONG,
                0.6,
                price,
                stop,
                price + risk * self.reward_risk,
                "pullback to confirmed support inside an HH/HL structure",
                self.reward_risk,
                "price closes back below the confirmed swing low",
            )
        if (
            analysis.trend == Side.SHORT
            and analysis.pullback == "bearish"
            and analysis.resistance is not None
        ):
            stop = analysis.resistance
            risk = stop - price
            if risk <= 0 or not isfinite(risk):
                return None
            return Signal(
                self.name,
                Side.SHORT,
                0.6,
                price,
                stop,
                price - risk * self.reward_risk,
                "pullback to confirmed resistance inside an LH/LL structure",
                self.reward_risk,
                "price closes back above the confirmed swing high",
            )
        return None


class SessionOpeningRange(DescribedStrategy):
    """Break out of the range formed by a session's first bars, in-session only."""

    name = "session_opening_range"

    def __init__(
        self,
        range_bars: int = 4,
        stop_atr: float = 1.5,
        reward_risk: float = 2.0,
        sessions: tuple[str, ...] = ("LONDON_NEW_YORK_OVERLAP",),
    ):
        if range_bars < 1:
            raise ValueError("opening range needs at least one bar")
        self.range_bars = range_bars
        self.stop_atr = stop_atr
        self.reward_risk = reward_risk
        self.sessions = sessions

    def generate(self, data: MarketData, context: MarketSnapshot) -> Signal | None:
        from .market import session_context

        if context.volatility <= 0 or len(data.candles) <= self.range_bars:
            return None
        session = session_context(data.candles[-1].timestamp)
        if session not in self.sessions:
            return None
        # Walk back over the contiguous bars of this session instance.
        run: list = []
        for candle in reversed(data.candles):
            if session_context(candle.timestamp) != session:
                break
            run.append(candle)
        if len(run) <= self.range_bars:
            return None  # the opening range is not complete yet
        run.reverse()
        opening = run[: self.range_bars]
        range_high = max(candle.high for candle in opening)
        range_low = min(candle.low for candle in opening)
        price = data.candles[-1].close
        if range_low <= price <= range_high:
            return None  # still trading inside the opening range
        side = Side.LONG if price > range_high else Side.SHORT
        distance = context.volatility * self.stop_atr
        stop = price - distance if side == Side.LONG else price + distance
        target = (
            price + distance * self.reward_risk
            if side == Side.LONG
            else price - distance * self.reward_risk
        )
        return Signal(
            self.name,
            side,
            0.6,
            price,
            stop,
            target,
            f"breakout of the {session.lower().replace('_', ' ')} opening range",
            self.reward_risk,
            "price returns inside the opening range",
        )


class StrategyRegistry:
    """Named discovery layer for the small set of distinct research strategies."""

    def __init__(self, strategies: list[Strategy] | None = None):
        self._strategies: dict[str, Strategy] = {}
        available: list[Strategy] = (
            strategies
            if strategies is not None
            else [
                EmaTrend(),
                Breakout(),
                TrendPullback(),
                MomentumContinuation(),
                MeanReversion(),
                RangeFade(),
                PriceActionRejection(),
                VolatilityExpansion(),
                StructureContinuation(),
                SessionOpeningRange(),
            ]
        )
        for strategy in available:
            self.register(strategy)

    def register(self, strategy: Strategy) -> None:
        if not strategy.name or strategy.name in self._strategies:
            raise ValueError(
                f"strategy name is empty or already registered: {strategy.name!r}"
            )
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> Strategy:
        return self._strategies[name]

    def all(self) -> tuple[Strategy, ...]:
        return tuple(self._strategies.values())


def risk_reward(signal: Signal) -> float:
    if signal.side == Side.LONG:
        risk, reward = (
            signal.entry - signal.stop_loss,
            signal.take_profit - signal.entry,
        )
    elif signal.side == Side.SHORT:
        risk, reward = (
            signal.stop_loss - signal.entry,
            signal.entry - signal.take_profit,
        )
    else:
        return 0.0
    return reward / risk if risk > 0 and reward > 0 else 0.0


def valid_signal(signal: Signal) -> bool:
    if not isinstance(signal, Signal):
        return False
    values = (signal.confidence, signal.entry, signal.stop_loss, signal.take_profit)
    return (
        signal.side in (Side.LONG, Side.SHORT)
        and all(isinstance(value, (int, float)) and isfinite(value) for value in values)
        and 0 <= signal.confidence <= 1
        and min(signal.entry, signal.stop_loss, signal.take_profit) > 0
        and signal.order_type in {"MARKET", "LIMIT", "STOP"}
        and isinstance(signal.expires_after_bars, int)
        and signal.expires_after_bars > 0
        and risk_reward(signal) > 0
    )
