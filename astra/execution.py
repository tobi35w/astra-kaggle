"""Paper-only execution boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from enum import Enum
from math import isfinite
from typing import Protocol
from uuid import uuid4

from .models import Candle, RiskApproval, RiskState, Side, Signal
from .repositories import Entity, Journal, RecordKeys
from .strategies import valid_signal


@dataclass(frozen=True)
class PaperFill:
    strategy: str
    side: str
    size: float
    requested_price: float
    fill_price: float
    slippage: float


@dataclass
class PaperPosition:
    position_id: int
    signal: Signal
    size: float
    entry: float
    opened_at: datetime
    pair: str = ""
    spread: float = 0.0
    last_price: float = 0.0
    timeframe: str = ""
    regime: str = "UNCLEAR"
    setup_type: str = "unknown"


@dataclass(frozen=True)
class ClosedPaperTrade:
    position_id: int
    strategy: str
    side: str
    size: float
    entry: float
    exit: float
    pnl: float
    reason: str
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    pair: str = ""
    timeframe: str = ""
    fees: float = 0.0
    regime: str = "UNCLEAR"
    confidence: float | None = None
    setup_type: str = "unknown"
    portfolio_id: str = ""


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class PaperOrder:
    order_id: int
    signal: Signal
    created_at: datetime
    pair: str
    status: OrderStatus = OrderStatus.PENDING
    bars_waited: int = 0
    reason: str = "awaiting an eligible candle"
    position_id: int | None = None
    regime: str = "UNCLEAR"
    last_processed_at: datetime | None = None


def order_price(signal: Signal, candle: Candle) -> float | None:
    """Mid-price trigger. Intrabar limit/stop touches fill only at the specified level."""
    long = signal.side.value == "LONG"
    if signal.order_type == "MARKET":
        return candle.open
    if signal.order_type == "LIMIT":
        return (
            min(candle.open, signal.entry)
            if long and candle.low <= signal.entry
            else max(candle.open, signal.entry)
            if not long and candle.high >= signal.entry
            else None
        )
    return (
        max(candle.open, signal.entry)
        if long and candle.high >= signal.entry
        else min(candle.open, signal.entry)
        if not long and candle.low <= signal.entry
        else None
    )


def exit_trigger(signal: Signal, high: float, low: float, opening: float | None = None):
    """Conservative OHLC: stop wins ambiguous collisions; targets get no gap improvement."""
    long = signal.side.value == "LONG"
    stop_hit = low <= signal.stop_loss if long else high >= signal.stop_loss
    target_hit = high >= signal.take_profit if long else low <= signal.take_profit
    if stop_hit:
        price = signal.stop_loss
        if opening is not None:
            price = min(price, opening) if long else max(price, opening)
        return price, "stop_loss"
    if target_hit:
        return signal.take_profit, "take_profit"
    return None


class ExecutionProvider(Protocol):
    def execute(
        self, signal: Signal, approval: RiskApproval, spread: float = 0.0
    ) -> PaperFill: ...


class PaperExecutionProvider:
    def __init__(
        self,
        slippage: float = 0.0,
        commission_per_unit: float = 0.0,
        value_per_price_unit: float = 1.0,
        *,
        price_basis: str = "mid",
    ):
        if not all(
            isfinite(value) and value >= 0 for value in (slippage, commission_per_unit)
        ):
            raise ValueError("execution costs must be finite and non-negative")
        if not isfinite(value_per_price_unit) or value_per_price_unit <= 0:
            raise ValueError("price value conversion must be finite and positive")
        self.slippage, self.commission_per_unit = slippage, commission_per_unit
        self.value_per_price_unit = value_per_price_unit
        if price_basis not in ("mid", "bid"):
            raise ValueError("execution price basis must be mid or bid")
        self.price_basis = price_basis

    def entry_offset(self, signal: Signal, spread: float) -> float:
        direction = 1 if signal.side.value == "LONG" else -1
        return (
            (spread if direction == 1 else 0.0)
            if self.price_basis == "bid"
            else direction * spread / 2
        )

    def exit_offset(self, signal: Signal, spread: float) -> float:
        direction = 1 if signal.side.value == "LONG" else -1
        return (
            (0.0 if direction == 1 else spread)
            if self.price_basis == "bid"
            else -direction * spread / 2
        )

    def mark_price(self, price: float, signal: Signal, spread: float) -> float:
        # Preserve legacy midpoint marks; BID books mark on the liquidation quote.
        return price + (
            self.exit_offset(signal, spread) if self.price_basis == "bid" else 0.0
        )

    def fill_price(self, price: float, signal: Signal, spread: float) -> float:
        direction = 1 if signal.side.value == "LONG" else -1
        return price + self.entry_offset(signal, spread) + direction * self.slippage

    def exit_cost(self, spread: float) -> float:
        # BID stops/targets are executable quote levels, so spread is already in geometry.
        return self.slippage + (spread / 2 if self.price_basis == "mid" else 0.0)

    def execute(
        self, signal: Signal, approval: RiskApproval, spread: float = 0.0
    ) -> PaperFill:
        if (
            not approval.approved
            or not isfinite(approval.position_size)
            or approval.position_size <= 0
        ):
            raise PermissionError("risk engine vetoed execution")
        if not isfinite(spread) or spread < 0:
            raise ValueError("spread must be finite and non-negative")
        if not isinstance(signal, Signal) or signal.side not in (Side.LONG, Side.SHORT):
            raise ValueError("paper executor rejected malformed strategy signal")
        values = (signal.confidence, signal.entry, signal.stop_loss, signal.take_profit)
        if (
            not all(
                isinstance(value, (int, float)) and isfinite(value) for value in values
            )
            or signal.order_type not in {"MARKET", "LIMIT", "STOP"}
            or not isinstance(signal.expires_after_bars, int)
            or signal.expires_after_bars <= 0
        ):
            raise ValueError("paper executor rejected malformed strategy signal")
        fill_price = self.fill_price(signal.entry, signal, spread)
        if not valid_signal(replace(signal, entry=fill_price)):
            raise ValueError("execution price invalidates stop/target geometry")
        direction = 1 if signal.side.value == "LONG" else -1
        price = fill_price
        cost = (price - signal.entry) * direction
        return PaperFill(
            signal.strategy,
            signal.side.value,
            approval.position_size,
            signal.entry,
            price,
            cost,
        )


class PaperPortfolio:
    """Deterministic paper position ledger; no network or broker behavior."""

    def __init__(
        self,
        executor: PaperExecutionProvider,
        initial_equity: float,
        journal: Journal | None = None,
    ):
        if not isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("initial equity must be positive")
        self.executor = executor
        self.balance = initial_equity
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.day_start_equity = initial_equity
        self.active_day: date | None = None
        self.journal = journal
        self.portfolio_id = str(uuid4())
        self.positions: dict[int, PaperPosition] = {}
        self.closed: list[ClosedPaperTrade] = []
        self._next_id = 1
        self.orders: dict[int, PaperOrder] = {}
        self._last_marks: dict[str, datetime] = {}
        self._revision = 0

    @property
    def equity(self):
        return self.balance + sum(
            (p.last_price - p.entry)
            * (1 if p.signal.side.value == "LONG" else -1)
            * p.size
            * self.executor.value_per_price_unit
            for p in self.positions.values()
        )

    def advance_day(self, timestamp):
        day = timestamp.astimezone(UTC).date()
        if self.active_day is not None and day < self.active_day:
            raise ValueError("portfolio clock cannot move backwards")
        if self.active_day != day:
            self.active_day = day
            self.day_start_equity = self.equity
        self.peak_equity = max(self.peak_equity, self.equity)

    def risk_state(self, pair="") -> RiskState:
        losses = 0
        for trade in reversed(self.closed):
            if trade.pnl >= 0:
                break
            losses += 1
        positions = tuple(self.positions.values())
        # Without a supplied correlation model, reserve all other pairs conservatively.
        return RiskState(
            self.equity,
            self.day_start_equity,
            self.peak_equity,
            losses,
            sum(p.size for p in positions),
            False,
            len(positions),
            sum(p.size for p in positions if p.pair != pair),
            sum(p.size for p in positions if p.pair == pair),
            sum(
                (
                    abs(p.last_price - p.signal.stop_loss)
                    * self.executor.value_per_price_unit
                    + self.executor.commission_per_unit / 2
                    + self.executor.exit_cost(p.spread)
                    * self.executor.value_per_price_unit
                )
                * p.size
                for p in positions
            ),
        )

    def open(
        self,
        signal: Signal,
        approval: RiskApproval,
        timestamp,
        spread: float = 0.0,
        pair: str = "",
        timeframe: str = "",
        regime: str = "UNCLEAR",
    ) -> PaperPosition:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("paper timestamp must be timezone-aware")
        self.advance_day(timestamp)
        fill = self.executor.execute(signal, approval, spread)
        if not valid_signal(replace(signal, entry=fill.fill_price)):
            raise ValueError("execution price invalidates stop/target geometry")
        position = PaperPosition(
            self._next_id,
            signal,
            fill.size,
            fill.fill_price,
            timestamp,
            pair,
            spread,
            self.executor.mark_price(signal.entry, signal, spread),
            timeframe,
            regime,
            signal.strategy,
        )
        if self.journal:
            from .repositories import Record, json_value

            keys: RecordKeys = {
                "pair": pair,
                "timeframe": timeframe,
                "strategy": signal.strategy,
                "correlation_id": self.portfolio_id + f":{position.position_id}",
            }
            self.journal.repository.append_batch(
                (
                    Record(
                        Entity.EXECUTION,
                        timestamp,
                        json_value({"fill": fill, "position": position}),
                        **keys,
                    ),
                    Record(Entity.POSITION, timestamp, json_value(position), **keys),
                )
            )
        self.positions[position.position_id] = position
        self.balance -= self.executor.commission_per_unit * position.size / 2
        self._next_id += 1
        return position

    def submit(
        self, signal, approval, timestamp, pair="", regime="UNCLEAR"
    ) -> PaperOrder:
        if not approval.approved or not valid_signal(signal):
            raise PermissionError("risk engine vetoed paper order")
        order = PaperOrder(len(self.orders) + 1, signal, timestamp, pair, regime=regime)
        self._record_order(order, timestamp)
        self.orders[order.order_id] = order
        return order

    def _record_order(self, order, timestamp):
        if self.journal:
            self.journal.record(
                Entity.ORDER,
                timestamp,
                order,
                pair=order.pair,
                strategy=order.signal.strategy,
                correlation_id=self.portfolio_id,
            )

    def cancel(self, order_id, timestamp):
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError("only pending orders can be cancelled")
        order.status, order.reason = OrderStatus.CANCELLED, "cancelled"
        self._record_order(order, timestamp)

    def process_orders(self, candle: Candle, pair: str, risk, spread=0.0, timeframe=""):
        from .data import validate
        from .models import MarketData

        validate(MarketData(pair or "paper", timeframe or "M5", (candle,)), 1)
        if not isfinite(spread) or spread < 0:
            raise ValueError("spread must be finite and non-negative")
        self.advance_day(candle.timestamp)
        positions = []
        for order in self.orders.values():
            if (
                order.status != OrderStatus.PENDING
                or order.pair != pair
                or candle.timestamp <= order.created_at
                or (
                    order.last_processed_at is not None
                    and candle.timestamp <= order.last_processed_at
                )
            ):
                continue
            order.bars_waited += 1
            order.last_processed_at = candle.timestamp
            offset = (
                self.executor.entry_offset(order.signal, spread)
                if self.executor.price_basis == "bid"
                else 0.0
            )
            quoted = replace(
                candle,
                open=candle.open + offset,
                high=candle.high + offset,
                low=candle.low + offset,
                close=candle.close + offset,
            )
            price = order_price(order.signal, quoted)
            if price is not None:
                price -= offset
                signal = replace(order.signal, entry=price)
                effective = replace(
                    signal,
                    entry=self.executor.fill_price(price, signal, spread),
                )
                approval = risk.approve(
                    effective,
                    self.risk_state(pair),
                    spread,
                    data_valid=self.journal.healthy() if self.journal else True,
                    expected_slippage=self.executor.slippage,
                    value_per_price_unit=self.executor.value_per_price_unit,
                    execution_cost_per_unit=self.executor.commission_per_unit
                    + self.executor.exit_cost(spread)
                    * self.executor.value_per_price_unit,
                )
                if approval.approved:
                    position = self.open(
                        signal,
                        approval,
                        candle.timestamp,
                        spread,
                        pair,
                        timeframe,
                        order.regime,
                    )
                    positions.append(position)
                    order.status, order.position_id, order.reason = (
                        OrderStatus.FILLED,
                        position.position_id,
                        "risk-approved fill",
                    )
                else:
                    order.status, order.reason = (
                        OrderStatus.REJECTED,
                        "; ".join(approval.reasons),
                    )
            elif order.bars_waited >= order.signal.expires_after_bars:
                order.status, order.reason = (
                    OrderStatus.EXPIRED,
                    "unfilled order expired",
                )
            self._record_order(order, candle.timestamp)
        return tuple(positions)

    def close(
        self, position_id, price, timestamp, reason="manual_close"
    ) -> ClosedPaperTrade:
        position = self.positions[position_id]
        if timestamp < position.opened_at or not isfinite(price) or price <= 0:
            raise ValueError("invalid paper close")
        direction = 1 if position.signal.side.value == "LONG" else -1
        exit_price = (
            price
            + self.executor.exit_offset(position.signal, position.spread)
            - direction * self.executor.slippage
        )
        gross = (
            (exit_price - position.entry)
            * direction
            * position.size
            * self.executor.value_per_price_unit
        )
        fees = self.executor.commission_per_unit * position.size
        trade = ClosedPaperTrade(
            position_id,
            position.signal.strategy,
            position.signal.side.value,
            position.size,
            position.entry,
            exit_price,
            gross - fees,
            reason,
            position.opened_at,
            timestamp,
            position.pair,
            position.timeframe,
            fees,
            position.regime,
            position.signal.confidence,
            position.setup_type,
            self.portfolio_id,
        )
        if self.journal:
            from .repositories import Record, json_value

            keys: RecordKeys = {
                "pair": position.pair,
                "timeframe": position.timeframe,
                "strategy": trade.strategy,
                "correlation_id": self.portfolio_id + f":{position_id}",
            }
            self.journal.repository.append_batch(
                tuple(
                    Record(entity, timestamp, json_value(value), **keys)
                    for entity, value in (
                        (Entity.TRADE, trade),
                        (Entity.PNL, {"pnl": trade.pnl, "fees": fees}),
                        (
                            Entity.POSITION,
                            {"position_id": position_id, "status": "CLOSED"},
                        ),
                    )
                )
            )
        self.balance += gross - fees / 2
        self.closed.append(trade)
        del self.positions[position_id]
        self.peak_equity = max(self.peak_equity, self.equity)
        return trade

    def mark(
        self,
        timestamp,
        high: float,
        low: float,
        close: float,
        *,
        pair: str | None = None,
        opening: float | None = None,
        include_new: bool = False,
    ) -> tuple[ClosedPaperTrade, ...]:
        if (
            not all(isfinite(v) and v > 0 for v in (high, low, close))
            or not low <= close <= high
        ):
            raise ValueError("invalid OHLC mark")
        if len({p.pair for p in self.positions.values()}) > 1 and pair is None:
            raise ValueError("pair is required when marking a multi-pair portfolio")
        key = pair or ""
        if key in self._last_marks and timestamp <= self._last_marks[key]:
            raise ValueError("marks must be strictly chronological")
        self._last_marks[key] = timestamp
        self.advance_day(timestamp)
        just_closed = []
        for position_id, position in tuple(self.positions.items()):
            if pair is not None and position.pair != pair:
                continue
            if timestamp < position.opened_at or (
                timestamp == position.opened_at and not include_new
            ):
                continue
            offset = (
                self.executor.exit_offset(position.signal, position.spread)
                if self.executor.price_basis == "bid"
                else 0.0
            )
            trigger = exit_trigger(
                position.signal,
                high + offset,
                low + offset,
                opening + offset if opening is not None else None,
            )
            if (
                trigger
                and trigger[1] == "take_profit"
                and timestamp == position.opened_at
                and position.signal.order_type != "MARKET"
            ):
                # Without intrabar sequencing, a touched target may predate a pending entry.
                trigger = None
            if trigger:
                just_closed.append(
                    self.close(position_id, trigger[0] - offset, timestamp, trigger[1])
                )
            else:
                position.last_price = self.executor.mark_price(
                    close, position.signal, position.spread
                )
        self.peak_equity = max(self.peak_equity, self.equity)
        return tuple(just_closed)

    def checkpoint(self, timestamp):
        """Persist a complete book at an explicit recovery boundary; never replay orders on restore."""
        if not self.journal:
            raise ValueError("checkpoint requires persistence")
        self._revision += 1
        return self.journal.record(
            Entity.POSITION,
            timestamp,
            {
                "snapshot": True,
                "portfolio_id": self.portfolio_id,
                "revision": self._revision,
                "initial_equity": self.initial_equity,
                "balance": self.balance,
                "peak_equity": self.peak_equity,
                "day_start_equity": self.day_start_equity,
                "active_day": self.active_day.isoformat() if self.active_day else None,
                "positions": tuple(self.positions.values()),
                "closed": self.closed,
                "orders": tuple(self.orders.values()),
                "next_id": self._next_id,
                "last_marks": self._last_marks,
                "execution": {
                    "slippage": self.executor.slippage,
                    "commission_per_unit": self.executor.commission_per_unit,
                    "value_per_price_unit": self.executor.value_per_price_unit,
                    "price_basis": self.executor.price_basis,
                },
            },
            correlation_id=self.portfolio_id,
        )

    @classmethod
    def restore(cls, journal, portfolio_id, as_of):
        rows = [
            r
            for r in journal.repository.query(Entity.POSITION, as_of=as_of)
            if r.payload.get("snapshot")
            and r.payload.get("portfolio_id") == portfolio_id
        ]
        if not rows:
            raise ValueError("paper checkpoint unavailable")
        payload = max(rows, key=lambda r: r.payload["revision"]).payload
        book = cls(
            PaperExecutionProvider(**payload["execution"]),
            payload["initial_equity"],
            journal,
        )
        book.portfolio_id = portfolio_id
        for name in ("balance", "peak_equity", "day_start_equity"):
            setattr(book, name, payload[name])
        book.active_day = (
            date.fromisoformat(payload["active_day"]) if payload["active_day"] else None
        )
        book._next_id, book._revision = payload["next_id"], payload["revision"]
        book._last_marks = {
            k: datetime.fromisoformat(v) for k, v in payload["last_marks"].items()
        }
        from .models import Side

        for item in payload["positions"]:
            value = dict(item)
            value["signal"] = Signal(
                **(value["signal"] | {"side": Side(value["signal"]["side"])})
            )
            value["opened_at"] = datetime.fromisoformat(value["opened_at"])
            position = PaperPosition(**value)
            book.positions[position.position_id] = position
        for item in payload["orders"]:
            value = dict(item)
            value["signal"] = Signal(
                **(value["signal"] | {"side": Side(value["signal"]["side"])})
            )
            value["created_at"] = datetime.fromisoformat(value["created_at"])
            value["last_processed_at"] = (
                datetime.fromisoformat(value["last_processed_at"])
                if value["last_processed_at"]
                else None
            )
            value["status"] = OrderStatus(value["status"])
            order = PaperOrder(**value)
            book.orders[order.order_id] = order
        for item in payload["closed"]:
            value = dict(item)
            for name in ("opened_at", "closed_at"):
                value[name] = (
                    datetime.fromisoformat(value[name]) if value[name] else None
                )
            book.closed.append(ClosedPaperTrade(**value))
        return book
