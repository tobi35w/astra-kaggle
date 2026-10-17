"""Fail-safe system health checks for decision-time dependencies."""

from __future__ import annotations

from datetime import UTC, datetime

from .data import validate
from .models import HealthStatus, MarketData


def check_health(
    data: MarketData | None,
    *,
    now: datetime | None = None,
    max_age_seconds: int = 3600,
    database_ok: bool = True,
    risk_engine_ok: bool = True,
    execution_ok: bool = True,
    strategy_count: int = 1,
    configuration_valid: bool = True,
    persistence_ok: bool = True,
    model_required: bool = False,
    model_available: bool = False,
) -> HealthStatus:
    now = now or datetime.now(UTC)
    reasons = []
    if now.tzinfo is None or now.utcoffset() is None:
        reasons.append("health check timestamp must be timezone-aware")
    if data is None:
        reasons.append("market data unavailable")
    else:
        try:
            validate(data)
            if now.tzinfo is not None and now.utcoffset() is not None:
                age = (now - data.candles[-1].timestamp).total_seconds()
                if age < 0 or age > max_age_seconds:
                    reasons.append("market data stale or timestamp is in the future")
        except (ValueError, TypeError, AttributeError) as error:
            reasons.append(f"market data invalid: {error}")
    if not database_ok:
        reasons.append("database unavailable")
    if not risk_engine_ok:
        reasons.append("risk engine unavailable")
    if not execution_ok:
        reasons.append("paper execution unavailable")
    if strategy_count <= 0:
        reasons.append("no strategies registered")
    if not configuration_valid:
        reasons.append("configuration invalid")
    if not persistence_ok:
        reasons.append("persistence unavailable")
    if model_required and not model_available:
        reasons.append("required model unavailable")
    return HealthStatus(
        not reasons, tuple(reasons) or ("all critical checks passed",), now
    )
