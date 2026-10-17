"""Validated data and read-only advisory contracts for future external adapters.

No provider connection or order authority is implemented here.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .data import DataValidationError, validate
from .evidence import digest
from .historical.dataset import require_quality
from .history import PAIRS, TIMEFRAMES
from .models import Decision, MarketData
from .repositories import json_value


class MarketDataSource(Protocol):
    def closed_bars(self, pair: str, timeframe: str, as_of: datetime) -> MarketData: ...


class MarketDataGateway:
    """Validate an injected source before it reaches analysis or paper risk."""

    def __init__(self, source: MarketDataSource, *, max_age_seconds=3600):
        if max_age_seconds <= 0:
            raise ValueError("data freshness limit must be positive")
        self.source = source
        self.max_age_seconds = max_age_seconds

    def load(self, pair: str, timeframe: str, as_of: datetime) -> MarketData:
        if pair not in PAIRS or timeframe not in TIMEFRAMES or as_of.tzinfo is None:
            raise DataValidationError("invalid market-data request")
        data = self.source.closed_bars(pair, timeframe, as_of)
        if (data.pair, data.timeframe) != (pair, timeframe):
            raise DataValidationError(
                "source returned a different instrument or timeframe"
            )
        if data.metadata.get("timestamp_role") != "bar_close":
            raise DataValidationError(
                "adapter must supply close/availability timestamps"
            )
        if data.metadata.get("price_basis") not in ("bid", "mid"):
            raise DataValidationError("adapter must declare a supported price basis")
        validate(data, 2)
        age = (as_of - data.candles[-1].timestamp).total_seconds()
        if age < 0 or age > self.max_age_seconds:
            raise DataValidationError(
                "source data is unfinished, future-dated, or stale"
            )
        require_quality(data)
        return data


@dataclass(frozen=True)
class AdvisoryContext:
    schema_version: int
    context_id: str
    decision_time: datetime
    payload_json: str


def advisory_context(decision: Decision) -> AdvisoryContext:
    """Export quantitative facts only. No configuration, credentials or write handles."""
    dataset = decision.context.get("dataset", {})
    payload = {
        "authority": "research interpretation only; cannot approve or execute trades",
        "decision": {
            "action": decision.action.value,
            "reason": decision.reason,
            "reason_codes": decision.reason_codes,
            "pair": decision.pair,
            "timeframe": decision.timeframe,
            "regime": decision.regime.value,
            "timestamp": decision.timestamp,
        },
        "signal": decision.signal,
        "risk_approval": decision.risk,
        "confidence": decision.confidence,
        "market": decision.context.get("market"),
        "dataset": {
            key: dataset[key]
            for key in (
                "source",
                "price_basis",
                "dataset_version",
                "quality_status",
                "canonical_m1_sha256",
            )
            if key in dataset
        },
    }
    encoded = json.dumps(json_value(payload), sort_keys=True, allow_nan=False)
    return AdvisoryContext(1, digest(payload), decision.timestamp, encoded)
