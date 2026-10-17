"""Immutable domain types shared across Astra subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Regime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    QUIET = "LOW_VOLATILITY"  # noqa: PIE796 - backward-compatible Python alias.
    TRANSITION = "TRANSITION"
    TRANSITIONAL = "TRANSITION"  # noqa: PIE796 - backward-compatible Python alias.
    BREAKOUT = "BREAKOUT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNCLEAR = "UNCLEAR"


class Action(str, Enum):
    TRADE = "TRADE"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    volume_type: str = "tick"
    broker_timestamp: datetime | None = None


@dataclass(frozen=True)
class MarketData:
    pair: str
    timeframe: str
    candles: tuple[Candle, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketSnapshot:
    pair: str
    timeframe: str
    timestamp: datetime
    trend: Side
    trend_strength: float
    volatility: float
    regime: Regime
    evidence: tuple[str, ...]
    uncertainty: str = "moderate"
    data_quality: str = "valid"
    regime_confidence: float = 0.5
    contradictions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Signal:
    strategy: str
    side: Side
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reason: str
    risk_reward: float = 0.0
    invalidation: str = "stop loss or setup condition invalidated"
    order_type: str = "MARKET"
    expires_after_bars: int = 1


@dataclass(frozen=True)
class ConfidenceBreakdown:
    market: float
    regime: float
    strategy: float
    setup: float
    execution: float
    overall: float
    evidence_level: str
    conflicts: tuple[str, ...] = ()
    timeframe_alignment: float = 0.5
    state: str = "INSUFFICIENT"
    supporting_evidence: tuple[str, ...] = ()
    uncertainty: str = "uncalibrated heuristic; not a probability"


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    reasons: tuple[str, ...]
    checked_at: datetime


@dataclass(frozen=True)
class StrategyPerformance:
    strategy: str
    regime: str
    pair: str
    timeframe: str
    trades: int
    expectancy: float
    max_drawdown: float
    observed_at: datetime
    version: int = 1
    session: str = "ALL"
    setup_type: str = "ALL"
    confidence_bucket: str = "ALL"
    win_rate: float = 0.0
    cost_sensitivity: float = 0.0
    fitness: float | None = None
    outcome: float | None = None
    source: str = "LEGACY"
    metric_unit: str = "account_currency"
    drawdown_unit: str = "legacy"
    dataset_version: str = ""
    experiment_id: str = ""
    sequence: int = 0


@dataclass(frozen=True)
class RiskState:
    equity: float
    day_start_equity: float
    peak_equity: float
    consecutive_losses: int = 0
    open_exposure: float = 0.0
    halted: bool = False
    open_positions: int = 0
    correlated_exposure: float = 0.0
    pair_exposure: float = 0.0
    open_risk: float = 0.0


@dataclass(frozen=True)
class RiskApproval:
    approved: bool
    reasons: tuple[str, ...]
    position_size: float = 0.0


@dataclass(frozen=True)
class Decision:
    timestamp: datetime
    pair: str
    timeframe: str
    action: Action
    reason: str
    regime: Regime = Regime.UNCLEAR
    signal: Signal | None = None
    risk: RiskApproval | None = None
    confidence: ConfidenceBreakdown | None = None
    reason_codes: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(UTC)
