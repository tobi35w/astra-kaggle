"""Persistence boundary and deterministic in-memory audit repository."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Protocol

from .models import Decision, StrategyPerformance


class DecisionRepository(Protocol):
    def append(self, decision: Decision) -> None: ...
    def all(self) -> tuple[Decision, ...]: ...


class InMemoryDecisionRepository:
    def __init__(self):
        self._decisions: list[Decision] = []

    def append(self, decision: Decision) -> None:
        self._decisions.append(decision)

    def all(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)


class StrategyPerformanceRepository(Protocol):
    def append(self, performance: StrategyPerformance) -> None: ...
    def history(self, strategy: str) -> tuple[StrategyPerformance, ...]: ...


class InMemoryStrategyPerformanceRepository:
    """Append-only versioned results; historical observations are never overwritten."""

    def __init__(self):
        self._records: list[StrategyPerformance] = []

    def append(self, performance: StrategyPerformance) -> None:
        self._records.append(performance)

    def history(self, strategy: str) -> tuple[StrategyPerformance, ...]:
        return tuple(record for record in self._records if record.strategy == strategy)


class PostgresDecisionRepository:
    """Optional PostgreSQL adapter; requires psycopg and a caller-supplied connection string."""

    def __init__(self, connection_string: str):
        if not connection_string:
            raise ValueError("PostgreSQL connection string is required")
        self.connection_string = connection_string

    def append(self, decision: Decision) -> None:
        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as error:
            raise RuntimeError(
                "install Astra's 'postgres' extra to use PostgreSQL persistence"
            ) from error
        signal = _jsonable(decision.signal)
        risk = _jsonable(decision.risk)
        with (
            psycopg.connect(self.connection_string) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                "INSERT INTO astra_decisions (occurred_at,pair,timeframe,action,reason,regime,strategy,signal,risk) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    decision.timestamp,
                    decision.pair,
                    decision.timeframe,
                    decision.action.value,
                    decision.reason,
                    decision.regime.value,
                    decision.signal.strategy if decision.signal else None,
                    Jsonb(signal) if signal is not None else None,
                    Jsonb(risk) if risk is not None else None,
                ),
            )


def _jsonable(value):
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)

    def convert(item):
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError(f"cannot encode {type(item).__name__}")

    return json.loads(json.dumps(value, default=convert))
