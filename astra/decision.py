"""Explainable end-to-end decision orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite

from .data import validate
from .database import DecisionRepository
from .health import check_health
from .intelligence import confidence as build_confidence
from .intelligence import detect_regime, select
from .market import MultiTimeframeContext
from .models import (
    Action,
    Decision,
    HealthStatus,
    Regime,
    RiskState,
    StrategyPerformance,
)
from .repositories import Journal, json_value
from .risk import RiskEngine
from .strategies import Strategy


def decide(
    data,
    strategies: list[Strategy],
    risk: RiskEngine,
    state: RiskState,
    spread: float = 0.0,
    min_score: float = 0.45,
    *,
    health: HealthStatus | None = None,
    repository: DecisionRepository | None = None,
    performance: tuple[StrategyPerformance, ...] = (),
    timeframe_context: MultiTimeframeContext | None = None,
    decision_time: datetime | None = None,
    expected_slippage: float = 0.0,
    value_per_price_unit: float = 1.0,
    journal: Journal | None = None,
    execution_cost_per_unit: float = 0.0,
) -> Decision:
    timestamp = decision_time or datetime.now(UTC)
    details = {
        "risk_state": json_value(state),
        "configuration": risk.settings.public(),
        "execution_assumptions": {
            "spread": spread,
            "slippage": expected_slippage,
            "value_per_price_unit": value_per_price_unit,
            "cost_per_unit": execution_cost_per_unit,
        },
        "strategies": [s.name for s in strategies],
        "dataset": json_value(data.metadata),
    }

    def finish(decision: Decision) -> Decision:
        decision = replace(
            decision,
            context=details,
            reason_codes=reason_codes(decision.reason)
            if decision.action != Action.TRADE
            else (),
        )
        try:
            if repository is not None:
                repository.append(decision)
            if journal is not None:
                journal.decision(decision, data)
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            return replace(
                decision,
                action=Action.NO_TRADE,
                reason="persistence failed; execution vetoed",
                reason_codes=("SYSTEM_HEALTH_FAILURE", "PERSISTENCE_FAILURE"),
            )
        return decision

    try:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("decision time must be timezone-aware")
        data = replace(
            data, candles=tuple(c for c in data.candles if c.timestamp <= timestamp)
        )
        validate(data)
    except (ValueError, TypeError, AttributeError) as error:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                f"data rejected: {error}",
            )
        )
    if len(data.candles) < 30:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "insufficient data: at least 30 available candles required",
                Regime.INSUFFICIENT_DATA,
            )
        )
    if (
        not all(isfinite(value) and value >= 0 for value in (spread, expected_slippage))
        or not isfinite(value_per_price_unit)
        or value_per_price_unit <= 0
    ):
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "spread or slippage input is invalid",
            )
        )
    model_available = False
    if risk.settings.ml_enabled and journal is not None:
        from .ml import features, load_model

        try:
            model = load_model(
                journal, pair=data.pair, timeframe=data.timeframe, as_of=timestamp
            )
            if model is not None:
                details["model"] = {
                    "version": model.version,
                    "estimate": model.predict(features(data, timestamp), timestamp),
                    "meaning": "model estimate; independent risk and heuristic confidence still apply",
                }
                model_available = True
        except (ValueError, RuntimeError):
            model_available = False
    system_health = check_health(
        data,
        now=decision_time or datetime.now(UTC),
        max_age_seconds=risk.settings.max_data_age_seconds,
        strategy_count=len(strategies),
        persistence_ok=journal.healthy() if journal is not None else True,
        model_required=risk.settings.ml_enabled,
        model_available=model_available,
    )
    details["health"] = json_value(system_health)
    health_reasons = list(system_health.reasons if not system_health.healthy else ())
    if health is not None and not health.healthy:
        health_reasons.extend(health.reasons)
    if health_reasons:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "; ".join(health_reasons),
            )
        )
    if timeframe_context is not None:
        details["timeframes"] = json_value(timeframe_context)
        if (
            timeframe_context.decision_time is not None
            and timeframe_context.decision_time > timestamp
        ) or any(
            s.timestamp > timestamp or s.pair != data.pair
            for s in timeframe_context.snapshots.values()
        ):
            return finish(
                Decision(
                    timestamp,
                    data.pair,
                    data.timeframe,
                    Action.NO_TRADE,
                    "invalid data: timeframe context contains unavailable or mismatched information",
                )
            )
    if timeframe_context is not None and not timeframe_context.aligned:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "; ".join(timeframe_context.conflicts),
            )
        )
    context = detect_regime(data, timeframe_context)
    details["market"] = json_value(context)
    from .market import session_context, structure

    details["structure"] = json_value(structure(data))
    details["session"] = session_context(timestamp)
    details["historical_performance"] = json_value(
        tuple(p for p in performance if p.observed_at <= timestamp)
    )
    if context.regime in {Regime.UNCLEAR, Regime.INSUFFICIENT_DATA}:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "unclear regime or insufficient evidence",
                context.regime,
            )
        )
    if (
        context.volatility / data.candles[-1].close
        > risk.settings.max_relative_volatility
    ):
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "excessive volatility",
                context.regime,
            )
        )
    try:
        selection = select(data, context, strategies, performance)
        details["selection"] = json_value(selection)
    except Exception as error:  # noqa: BLE001 - any strategy fault must fail to NO TRADE
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                f"strategy evaluation failed: {type(error).__name__}",
                context.regime,
            )
        )
    if not selection.candidates:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "unsuitable strategy or weak setup; " + selection.reason,
                context.regime,
            )
        )
    score, signal = selection.candidates[0]
    candidate_sides = {candidate.side for _, candidate in selection.candidates}
    strategy_conflicts = (
        ("candidate strategies disagree",) if len(candidate_sides) > 1 else ()
    )
    decision_confidence = build_confidence(
        market=0.8,  # Valid, sufficient candles; a range is not intrinsically low confidence.
        regime=context.regime_confidence,
        strategy=signal.confidence,
        setup=min(score, 1.0),
        execution=1.0
        if spread == 0
        else max(0.0, 1.0 - spread / max(risk.settings.max_spread, 1e-12)),
        conflicts=(
            (timeframe_context.conflicts if timeframe_context else ())
            + strategy_conflicts
        ),
        timeframe_alignment=timeframe_context.agreement if timeframe_context else 0.5,
        supporting_evidence=context.evidence + (signal.reason,),
    )
    if score < min_score:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "weak setup: best candidate score is below threshold",
                context.regime,
                signal,
                confidence=decision_confidence,
            )
        )
    if decision_confidence.overall < risk.settings.minimum_confidence or (
        strategy_conflicts and decision_confidence.overall < 0.55
    ):
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "conflicting strategy evidence"
                if strategy_conflicts
                else "insufficient combined evidence confidence",
                context.regime,
                signal,
                confidence=decision_confidence,
            )
        )
    approval = risk.approve(
        signal,
        state,
        spread,
        expected_slippage=expected_slippage,
        value_per_price_unit=value_per_price_unit,
        execution_cost_per_unit=execution_cost_per_unit,
    )
    if not approval.approved:
        return finish(
            Decision(
                timestamp,
                data.pair,
                data.timeframe,
                Action.NO_TRADE,
                "; ".join(approval.reasons),
                context.regime,
                signal,
                approval,
                decision_confidence,
            )
        )
    return finish(
        Decision(
            timestamp,
            data.pair,
            data.timeframe,
            Action.TRADE,
            "candidate passed configured risk checks (paper only)",
            context.regime,
            signal,
            approval,
            decision_confidence,
        )
    )


def reason_codes(reason: str) -> tuple[str, ...]:
    text = reason.lower()
    mapping = {
        "insufficient data": "INSUFFICIENT_DATA",
        "at least": "INSUFFICIENT_DATA",
        "stale": "STALE_DATA",
        "data rejected": "INVALID_DATA",
        "invalid data": "INVALID_DATA",
        "unclear regime": "UNCLEAR_REGIME",
        "timeframe": "CONFLICTING_TIMEFRAMES",
        "weak setup": "WEAK_SETUP",
        "confidence": "LOW_CONFIDENCE",
        "conflicting": "CONFLICTING_EVIDENCE",
        "unsuitable": "UNSUITABLE_STRATEGY",
        "degradation": "STRATEGY_DEGRADATION",
        "spread": "EXCESSIVE_SPREAD",
        "volatility": "EXCESSIVE_VOLATILITY",
        "risk/reward": "INSUFFICIENT_REWARD_RISK",
        "exposure": "EXCESSIVE_EXPOSURE",
        "correlated": "CORRELATED_EXPOSURE",
        "concentration": "PAIR_CONCENTRATION",
        "daily loss": "DAILY_LOSS_PROTECTION",
        "drawdown": "DRAWDOWN_PROTECTION",
        "consecutive-loss": "CONSECUTIVE_LOSS_PROTECTION",
        "unavailable": "SYSTEM_HEALTH_FAILURE",
        "failed": "SYSTEM_HEALTH_FAILURE",
        "slippage": "EXECUTION_UNCERTAINTY",
        "execution": "EXECUTION_UNCERTAINTY",
        "kill switch": "KILL_SWITCH_ACTIVE",
        "emergency": "EMERGENCY_HALT",
        "disabled": "PAPER_TRADING_DISABLED",
        "configuration": "CONFIGURATION_INVALID",
        "position size is zero": "RISK_BUDGET_EXHAUSTED",
        "open positions": "MAX_OPEN_POSITIONS",
    }
    return tuple(
        dict.fromkeys(code for phrase, code in mapping.items() if phrase in text)
    ) or ("NO_VALID_CANDIDATE",)
