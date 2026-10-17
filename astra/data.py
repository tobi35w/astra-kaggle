"""Candle validation and provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from math import isfinite
from typing import Protocol, runtime_checkable

from .models import MarketData


class DataValidationError(ValueError):
    """Raised when market data cannot safely be analyzed."""


def validate(data: MarketData, min_candles: int = 2) -> None:
    if data.metadata.get("quality_status", "PASS") != "PASS":
        raise DataValidationError("DATA QUALITY INSUFFICIENT")
    if min_candles < 1:
        raise ValueError("min_candles must be >= 1")
    if not data.pair.strip() or not data.timeframe.strip():
        raise DataValidationError("pair and timeframe are required")
    if len(data.candles) < min_candles:
        raise DataValidationError(f"at least {min_candles} candles are required")
    previous = None
    for index, candle in enumerate(data.candles):
        if candle.timestamp.tzinfo is None or candle.timestamp.utcoffset() is None:
            raise DataValidationError(
                f"candle {index} timestamp must be timezone-aware"
            )
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(map(_finite, values)):
            raise DataValidationError(f"candle {index} contains non-finite values")
        if min(candle.open, candle.high, candle.low, candle.close) <= 0:
            raise DataValidationError(f"candle {index} contains non-positive prices")
        if (
            candle.low > candle.high
            or not candle.low <= candle.open <= candle.high
            or not candle.low <= candle.close <= candle.high
        ):
            raise DataValidationError(f"candle {index} has impossible OHLC values")
        if candle.volume < 0:
            raise DataValidationError(f"candle {index} has negative volume")
        if candle.volume_type not in ("tick", "real", "provider"):
            raise DataValidationError(
                f"candle {index} volume_type must be 'tick', 'real', or 'provider'"
            )
        normalized = candle.timestamp.astimezone(UTC)
        if previous is not None and normalized <= previous:
            raise DataValidationError(
                "candles must be strictly ordered without duplicates"
            )
        previous = normalized


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and isfinite(value)


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_market_data(self, pair: str, timeframe: str, limit: int) -> MarketData: ...


@dataclass
class InMemoryProvider:
    data: MarketData

    def get_market_data(self, pair: str, timeframe: str, limit: int) -> MarketData:
        if pair != self.data.pair or timeframe != self.data.timeframe:
            raise KeyError("requested market series is unavailable")
        return MarketData(
            pair, timeframe, self.data.candles[-limit:], self.data.metadata
        )


if __name__ == "__main__":
    import sys

    from .historical.cli import historical_main

    arguments = sys.argv[1:]
    if arguments and arguments[0] == "historical":
        arguments = arguments[1:]
    raise SystemExit(historical_main(arguments))
