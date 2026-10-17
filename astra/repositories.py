"""Versioned, append-only domain journal. No adapter knowledge in business logic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from math import isfinite
from typing import Any, Protocol, TypedDict
from uuid import uuid4


class Entity(str, Enum):
    OBSERVATION = "market_observation"
    ANALYSIS = "analysis_snapshot"
    REGIME = "market_regime"
    SIGNAL = "strategy_signal"
    EVALUATION = "strategy_evaluation"
    FITNESS = "strategy_fitness"
    DECISION = "decision"
    RISK = "risk_decision"
    ORDER = "paper_order"
    POSITION = "position"
    EXECUTION = "execution"
    TRADE = "paper_trade"
    PNL = "pnl"
    BACKTEST = "backtest_run"
    METRICS = "backtest_metrics"
    WALK_FORWARD = "walk_forward"
    ROBUSTNESS = "robustness"
    MODEL = "model_metadata"
    CONFIDENCE = "confidence"
    CALIBRATION = "calibration"
    HEALTH = "health"
    AUDIT = "audit"
    PERFORMANCE = "performance"


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("journal timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not isfinite(value):
        return None  # Undefined ratios (e.g. profit factor without losses).
    return value


@dataclass(frozen=True)
class Record:
    entity: Entity
    occurred_at: datetime  # Availability time, never the backdated start of an outcome.
    payload: dict
    pair: str = ""
    timeframe: str = ""
    strategy: str = ""
    correlation_id: str = ""
    record_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = 1

    def row(self) -> dict:
        row = json_value(self)
        json.dumps(row, allow_nan=False)
        if self.schema_version != 1 or not isinstance(self.payload, dict):
            raise ValueError("invalid journal record")
        return row

    @classmethod
    def from_row(cls, row: dict) -> Record:
        values = dict(row)
        values.pop("recorded_at", None)
        values["entity"] = Entity(values["entity"])
        stamp = values["occurred_at"]
        values["occurred_at"] = (
            datetime.fromisoformat(stamp) if isinstance(stamp, str) else stamp
        )
        return cls(**values)


class PersistenceError(RuntimeError):
    """Sanitized adapter failure; credentials and remote response bodies are excluded."""


class RecordKeys(TypedDict):
    pair: str
    timeframe: str
    strategy: str
    correlation_id: str


class Repository(Protocol):
    def append_batch(self, records: tuple[Record, ...]) -> None: ...
    def query(
        self,
        entity: Entity,
        *,
        as_of: datetime | None = None,
        pair: str = "",
        timeframe: str = "",
        strategy: str = "",
    ) -> tuple[Record, ...]: ...
    def healthy(self) -> bool: ...


class MemoryRepository:
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def append_batch(self, records: tuple[Record, ...]) -> None:
        rows = [r.row() for r in records]
        for row in rows:
            self._rows.setdefault(row["record_id"], json.loads(json.dumps(row)))

    def query(self, entity: Entity, *, as_of=None, pair="", timeframe="", strategy=""):
        return tuple(
            sorted(
                (
                    Record.from_row(json.loads(json.dumps(row)))
                    for row in self._rows.values()
                    if row["entity"] == entity.value
                    and (
                        as_of is None
                        or datetime.fromisoformat(row["occurred_at"]) <= as_of
                    )
                    and all(
                        not value or row[key] == value
                        for key, value in (
                            ("pair", pair),
                            ("timeframe", timeframe),
                            ("strategy", strategy),
                        )
                    )
                ),
                key=lambda r: (r.occurred_at, r.record_id),
            )
        )

    def healthy(self):
        return True


class Journal:
    """Application-facing recorder; a single batch forms one reconstructable operation."""

    def __init__(self, repository: Repository):
        self.repository = repository

    def healthy(self) -> bool:
        try:
            return self.repository.healthy()
        except Exception:  # noqa: BLE001 - injected dependencies must fail closed.
            return False

    def record(self, entity: Entity, at: datetime, value: Any, **keys) -> Record:
        payload = json_value(value)
        record = Record(
            entity,
            at,
            payload if isinstance(payload, dict) else {"value": payload},
            **keys,
        )
        self.repository.append_batch((record,))
        return record

    def decision(self, decision, data=None) -> None:
        keys = {
            "pair": decision.pair,
            "timeframe": decision.timeframe,
            "strategy": decision.signal.strategy if decision.signal else "",
            "correlation_id": str(uuid4()),
        }
        observation = data
        if data is not None and hasattr(data, "candles"):
            candles = data.candles
            observation = {
                "pair": data.pair,
                "timeframe": data.timeframe,
                "available_at": candles[-1].timestamp
                if candles
                else decision.timestamp,
                "bar_count": len(candles),
                "candle": candles[-1] if candles else None,
                "metadata": data.metadata,
            }
        pairs = [
            (Entity.DECISION, decision),
            (Entity.AUDIT, {"event": "decision", "decision": decision}),
        ]
        if observation is not None:
            pairs.append((Entity.OBSERVATION, observation))
        for entity, value in (
            (Entity.ANALYSIS, decision.context),
            (Entity.REGIME, {"regime": decision.regime}),
            (Entity.CONFIDENCE, decision.confidence),
            (Entity.SIGNAL, decision.signal),
            (Entity.RISK, decision.risk),
        ):
            if value is not None:
                pairs.append((entity, value))
        self.repository.append_batch(
            tuple(
                Record(entity, decision.timestamp, json_value(value), **keys)
                for entity, value in pairs
            )
        )
