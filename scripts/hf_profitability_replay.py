"""Conditional, retrospective paper replay of the previously inspected HF data.

This is a research simulation, not historical model availability or a fresh
holdout. Source candles and their unverified metadata are never rewritten.
Only elapsed clock time and already observed candles determine eligibility.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite

from astra.config import Settings
from astra.data import validate
from astra.execution import OrderStatus, PaperExecutionProvider, PaperPortfolio
from astra.historical.validation import _market_open
from astra.ml import features
from astra.models import Side, Signal
from astra.repositories import json_value
from astra.risk import RiskEngine

MINUTE = timedelta(minutes=1)
HOLDING_TIME = 3 * MINUTE
MAX_BASE_UNITS = 10_000.0


class _RecordedRisk:
    """Record unmodified engine inputs at signal and actual repriced fill."""

    def __init__(self, engine):
        self.engine = engine
        self.events = []
        self.phase = "signal"
        self.decision_at = None
        self.signal_decision_at = None
        self.candidate_id = None

    def approve(self, signal, state, spread=0.0, **kwargs):
        approval = self.engine.approve(signal, state, spread, **kwargs)
        original_size = approval.position_size
        if approval.approved:
            approval = replace(
                approval, position_size=min(approval.position_size, MAX_BASE_UNITS)
            )
        self.events.append(
            json_value(
                {
                    "phase": self.phase,
                    "candidate_id": self.candidate_id,
                    "decision_at": self.decision_at,
                    "signal_decision_at": self.signal_decision_at,
                    "signal": signal,
                    "state": state,
                    "spread": spread,
                    **kwargs,
                    "approval": approval,
                    "engine_position_size": original_size,
                    "explicit_size_cap_applied": original_size > MAX_BASE_UNITS,
                }
            )
        )
        return approval


def _aware(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _liquidation_equity(book):
    # PaperPortfolio mid marks exclude the liquidation half-spread. Include
    # every remaining exit cost in the reported research equity curve.
    return book.equity - sum(
        position.size
        * (
            book.executor.exit_cost(position.spread)
            * book.executor.value_per_price_unit
            + book.executor.commission_per_unit / 2
        )
        for position in book.positions.values()
    )


def _daily_rows(initial_equity, observations, trades):
    closed_counts = Counter(trade["closed_at"][:10] for trade in trades)
    rows = []
    previous_equity = initial_equity
    for key, sample in sorted(observations.items()):
        equity = sample.get("end_equity", previous_equity)
        pnl = equity - previous_equity
        rows.append(
            {
                "date": key,
                "start_equity": previous_equity,
                "end_equity": equity,
                "end_book_equity": sample.get("end_book_equity", equity),
                "pnl": pnl,
                "return": pnl / previous_equity if previous_equity else 0.0,
                "trade_count": closed_counts[key],
                "observed_bars": sample.get("observed_bars", 0),
                "min_equity": sample.get("min_equity", equity),
                "max_equity": sample.get("max_equity", equity),
            }
        )
        previous_equity = equity
    return rows


def simulate(
    data,
    model,
    *,
    start_at,
    end_at=None,
    price_basis="mid",
    spread_pips=1.0,
    slippage_pips=0.2,
    policy="model",
    training_up_fraction=0.5,
    initial_equity=1_000_000.0,
    risk_engine=None,
    commission_per_unit=0.0,
):
    """Replay a frozen policy with source-close event availability timestamps.

    ``commission_per_unit`` is the round-trip fee in the pair's quote currency
    per base unit. ``end_at`` is a predeclared experiment boundary; when omitted,
    the supplied series' last timestamp is declared before replay. Numerical
    scoring occurs at the model's actual creation timestamp, never backdated.
    """
    created_at = datetime.now(UTC)
    validate(data, 15)
    if data.timeframe != "M1":
        raise ValueError("HF profitability replay requires M1 candles")
    if any(c.timestamp.second or c.timestamp.microsecond for c in data.candles):
        raise ValueError("M1 source timestamps must be minute aligned")
    _aware(start_at, "start_at")
    end_at = data.candles[-1].timestamp if end_at is None else end_at
    _aware(end_at, "end_at")
    if end_at < start_at:
        raise ValueError("end_at must not precede start_at")
    if end_at not in {c.timestamp for c in data.candles}:
        raise ValueError("predeclared end_at requires an observed closing candle")
    if policy not in {"model", "training_majority"}:
        raise ValueError("policy must be model or training_majority")
    if not isfinite(training_up_fraction) or not 0 <= training_up_fraction <= 1:
        raise ValueError("training_up_fraction must be in [0, 1]")
    if not all(isfinite(x) and x >= 0 for x in (spread_pips, slippage_pips)):
        raise ValueError("spread and slippage must be finite and non-negative")
    for name in ("training_end", "validation_start", "trained_at"):
        _aware(getattr(model, name), f"model.{name}")
    if model.training_end >= start_at or model.validation_start > start_at:
        raise ValueError("replay must begin after training and at/after validation")
    pip = 0.01 if data.pair.endswith("JPY") else 0.0001
    spread, slippage = spread_pips * pip, slippage_pips * pip
    executor = PaperExecutionProvider(
        slippage=slippage,
        commission_per_unit=commission_per_unit,
        value_per_price_unit=1.0,
        price_basis=price_basis,
    )
    book = PaperPortfolio(executor, initial_equity)
    # Stable research identity has no persistence side effects; the parent
    # records the complete result using its actual evaluation creation time.
    identity = (
        data.metadata.get("dataset_version", "unversioned"),
        data.pair,
        model.version,
        start_at.isoformat(),
        end_at.isoformat(),
        policy,
        price_basis,
        spread_pips,
        slippage_pips,
        commission_per_unit,
        initial_equity,
    )
    book.portfolio_id = (
        "hf-research-" + sha256(repr(identity).encode()).hexdigest()[:24]
    )
    settings = Settings(
        trading_enabled=True,
        max_trade_risk_pct=0.005,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.20,
        max_consecutive_losses=5,
        minimum_confidence=0.5,
        max_open_positions=1,
        max_position_size=MAX_BASE_UNITS,
        max_pair_exposure=MAX_BASE_UNITS,
        max_correlated_exposure=MAX_BASE_UNITS,
        max_spread=spread,
        max_slippage=slippage,
    )
    risk = _RecordedRisk(
        risk_engine if risk_engine is not None else RiskEngine(settings)
    )
    window = deque(maxlen=15)
    counters = Counter()
    order_events, gap_events = [], []
    order_final_times, signal_times, exit_times = {}, {}, {}
    observations = {}
    previous = None
    pending_order = None
    permanent_halt = None
    peak, max_drawdown = initial_equity, 0.0

    def order_snapshot(order, stamp, event):
        order_events.append(
            {"event": event, "available_at": stamp.isoformat(), **json_value(order)}
        )
        if order.status != OrderStatus.PENDING:
            order_final_times[order.order_id] = stamp.isoformat()

    for candle in data.candles:
        stamp = candle.timestamp
        if stamp > end_at:
            break
        gap = previous is not None and stamp - previous != MINUTE
        if gap:
            window.clear()
        window.append(candle)
        previous_stamp, previous = previous, stamp
        if stamp < start_at:
            continue
        counters["observed_bars"] += 1
        book.advance_day(stamp)
        if gap:
            pending = [pending_order] if pending_order is not None else []
            positions = tuple(book.positions)
            gap_events.append(
                {
                    "previous_available_at": previous_stamp.isoformat(),
                    "observed_at": stamp.isoformat(),
                    "elapsed_minutes": (stamp - previous_stamp).total_seconds() / 60,
                    "uncertain_pending_fills": len(pending),
                    "uncertain_gap_exits": len(positions),
                }
            )
            for order in pending:
                book.cancel(order.order_id, stamp)
                order.reason = (
                    "data_gap: unobserved next-bar fill cannot be established"
                )
                order_snapshot(order, stamp, "cancelled_data_gap")
            pending_order = None
            for position_id in positions:
                book.close(position_id, candle.open, stamp, "data_gap")
            counters["uncertain_pending_fills"] += len(pending)
            counters["uncertain_gap_exits"] += len(positions)
        else:
            if pending_order is not None:
                order = pending_order  # One position or pending order at a time.
                risk.phase, risk.decision_at = "fill", stamp
                risk.signal_decision_at = order.created_at
                risk.candidate_id = order.order_id
                positions = book.process_orders(candle, data.pair, risk, spread, "M1")
                for position in positions:
                    signal_times[position.position_id] = order.created_at
                    exit_times[position.position_id] = order.created_at + HOLDING_TIME
                order_snapshot(order, stamp, "fill_decision")
                if order.status != OrderStatus.PENDING:
                    pending_order = None
        # A market order fills at this bar's open, so its stop may be touched
        # within the same bar. Both-hit ambiguity is resolved stop-first by the
        # existing paper ledger, with all information available at bar close.
        book.mark(
            stamp,
            candle.high,
            candle.low,
            candle.close,
            pair=data.pair,
            opening=candle.open,
            include_new=True,
        )
        for position_id in tuple(book.positions):
            if stamp >= exit_times[position_id]:
                book.close(position_id, candle.close, stamp, "time_exit")
            elif stamp == end_at or not _market_open(stamp):
                book.close(position_id, candle.close, stamp, "scheduled_close")

        equity = _liquidation_equity(book)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        day = stamp.astimezone(UTC).date().isoformat()
        observation = observations.setdefault(
            day, {"observed_bars": 0, "min_equity": equity, "max_equity": equity}
        )
        observation["observed_bars"] += 1
        observation.update(end_equity=equity, end_book_equity=book.equity)
        observation["min_equity"] = min(observation["min_equity"], equity)
        observation["max_equity"] = max(observation["max_equity"], equity)

        if stamp + HOLDING_TIME > end_at:
            counters["skipped_end_boundary"] += 1
            continue
        if not all(_market_open(stamp + step * MINUTE) for step in range(4)):
            counters["skipped_scheduled_closure"] += 1
            continue
        if len(window) != 15 or not all(
            _market_open(bar.timestamp - MINUTE) for bar in window
        ):
            counters["skipped_feature_continuity"] += 1
            continue
        if book.positions or pending_order is not None:
            counters["skipped_open_or_pending"] += 1
            continue
        if permanent_halt is not None:
            counters["skipped_permanent_halt"] += 1
            continue
        values = features(replace(data, candles=tuple(window)), stamp)
        probability = (
            float(model.predict(values, model.trained_at))
            if policy == "model"
            else training_up_fraction
        )
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("model probability must be finite and in [0, 1]")
        side = Side.LONG if probability >= 0.5 else Side.SHORT
        direction = 1 if side == Side.LONG else -1
        atr14 = values[2] * candle.close
        signal = Signal(
            "hf_frozen_3m_" + policy,
            side,
            max(probability, 1 - probability),
            candle.close,
            candle.close - direction * 2 * atr14,
            candle.close + direction * 4 * atr14,
            "fixed p>=0.5 direction; 2 ATR stop, 4 ATR target, three elapsed minutes",
        )
        counters["candidates"] += 1
        risk.phase, risk.decision_at, risk.signal_decision_at = "signal", stamp, stamp
        risk.candidate_id = len(book.orders) + 1
        approval = risk.approve(
            signal,
            book.risk_state(data.pair),
            spread,
            data_valid=True,
            expected_slippage=slippage,
            value_per_price_unit=1.0,
            execution_cost_per_unit=commission_per_unit + spread + 2 * slippage,
        )
        if approval.approved:
            order = book.submit(signal, approval, stamp, data.pair)
            pending_order = order
            order_snapshot(order, stamp, "created")
        elif any(
            fragment in reason
            for reason in approval.reasons
            for fragment in (
                "consecutive-loss",
                "drawdown limit",
                "kill switch",
                "emergency halt",
            )
        ):
            permanent_halt = {
                "available_at": stamp.isoformat(),
                "reasons": list(approval.reasons),
            }

    trades = []
    for trade in book.closed:
        trades.append(
            {
                **json_value(trade),
                "units": trade.size,
                "pnl_per_unit": trade.pnl / trade.size,
                "decision_at": signal_times[trade.position_id].isoformat(),
                "planned_exit_at": exit_times[trade.position_id].isoformat(),
            }
        )
    orders = [
        {
            **json_value(order),
            "final_available_at": order_final_times.get(order.order_id),
        }
        for order in book.orders.values()
    ]
    rejections = Counter(
        reason
        for event in risk.events
        if not event["approval"]["approved"]
        for reason in event["approval"]["reasons"]
    )
    counters.update(
        {
            "orders_" + status.value.lower(): sum(
                o.status == status for o in book.orders.values()
            )
            for status in OrderStatus
        }
    )
    for phase in ("signal", "fill"):
        for approved in (True, False):
            counters[f"{phase}_{'approved' if approved else 'rejected'}"] = sum(
                e["phase"] == phase and e["approval"]["approved"] == approved
                for e in risk.events
            )
    total_pnl = sum(t["pnl"] for t in trades)
    residual = book.balance - initial_equity - total_pnl
    uncertain = bool(
        counters["uncertain_pending_fills"] or counters["uncertain_gap_exits"]
    )
    return {
        "pair": data.pair,
        "account_currency": data.pair[-3:],
        "quote_currency": data.pair[-3:],
        "initial_equity": initial_equity,
        "final_equity": _liquidation_equity(book),
        "max_drawdown_pct": max_drawdown,
        "policy": policy,
        "assumed_price_basis": price_basis,
        "source_price_basis": data.metadata.get("price_basis", "unknown"),
        "source_metadata": json_value(data.metadata),
        "start_at": start_at.isoformat(),
        "end_at": end_at.isoformat(),
        "created_at": created_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "model_version": model.version,
        "model_trained_at": model.trained_at.isoformat(),
        "model_training_end": model.training_end.isoformat(),
        "model_validation_start": model.validation_start.isoformat(),
        "assumptions": {
            "price_basis": price_basis,
            "spread_pips": spread_pips,
            "slippage_pips_each_side": slippage_pips,
            "commission_per_unit_roundtrip": commission_per_unit,
            "pip": pip,
            "value_per_price_unit": 1.0,
            "account_currency": data.pair[-3:],
            "size_unit": "base currency units",
            "pnl_unit": "total quote currency",
            "pnl_per_unit_unit": "quote currency per base unit",
            "equity_curve": "marked liquidation equity after remaining exit costs",
            "timestamps": "source timestamps assumed to be candle close/availability",
            "session": "America/New_York Sunday 17:00 through Friday 17:00",
            "entry": "next observed contiguous minute open with repriced risk approval",
            "feature_window": "15 prior/including-current consecutive open-session M1 bars",
            "future_continuity_filter": False,
            "model_scoring": "retrospective numerical scoring at original model.trained_at",
            "decision_threshold": 0.5,
            "stop_atr": 2,
            "target_atr": 4,
            "time_exit_minutes_from_decision": 3,
            "training_up_fraction": training_up_fraction,
            "end_boundary": "predeclared; requires observed final candle; no entry with later time exit",
            "gap_policy": "cancel uncertain pending; exit existing at first observed open, available at close",
        },
        "limits": {
            "default_risk_settings": settings.public(),
            "injected_risk_engine": risk_engine is not None,
            "max_base_units": MAX_BASE_UNITS,
            "source_authentication": "UNVERIFIED",
            "fresh_holdout": False,
            "data_gap_uncertainty": uncertain,
            "permanent_halt": permanent_halt,
            "caveat": "Conditional retrospective research on previously consumed data; not proof of profitability or historical deployability.",
        },
        "trades": trades,
        "orders": orders,
        "order_events": order_events,
        "risk_events": risk.events,
        "daily": _daily_rows(initial_equity, observations, trades),
        "risk_counters": dict(counters),
        "rejection_reasons": dict(rejections),
        "gaps": {
            "events": gap_events,
            "uncertain_pending_fills": counters["uncertain_pending_fills"],
            "uncertain_gap_exits": counters["uncertain_gap_exits"],
        },
        "reconciliation": {
            "initial_equity": initial_equity,
            "final_balance": book.balance,
            "final_equity": _liquidation_equity(book),
            "total_pnl": total_pnl,
            "balance_change": book.balance - initial_equity,
            "residual": residual,
            "passed": abs(residual) <= max(1e-8, initial_equity * 1e-12)
            and not book.positions
            and not any(o.status == OrderStatus.PENDING for o in book.orders.values()),
            "open_positions": len(book.positions),
            "pending_orders": counters["orders_pending"],
            "max_drawdown_pct": max_drawdown,
        },
        "promotion_eligible": False,
    }
