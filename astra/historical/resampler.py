"""Causal UTC-boundary resampling; incomplete target bars are omitted."""

from __future__ import annotations

from datetime import UTC, datetime

from ..history import TIMEFRAMES
from ..models import Candle


def resample(candles, target_timeframe: str) -> tuple[Candle, ...]:
    from ..data import validate
    from ..models import MarketData

    candles = tuple(candles)
    if candles:
        validate(MarketData("resample", "M1", candles), min_candles=1)
    if target_timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {target_timeframe}")
    target = TIMEFRAMES[target_timeframe]
    source = TIMEFRAMES["M1"]
    count = int(target.total_seconds() // source.total_seconds())
    # Every supported timeframe is M1 or greater, so count is always positive.
    buckets: dict[datetime, list[Candle]] = {}
    for candle in candles:
        # validate() above already enforces timezone-aware candle timestamps.
        close = candle.timestamp.astimezone(UTC)
        opened = close - source
        epoch = int(opened.timestamp())
        boundary = epoch - epoch % int(target.total_seconds())
        key = datetime.fromtimestamp(boundary, UTC)
        buckets.setdefault(key, []).append(candle)
    output = []
    for opening in sorted(buckets):
        group = sorted(buckets[opening], key=lambda c: c.timestamp)
        if len({c.volume_type for c in group}) != 1:
            raise ValueError("cannot resample a bar with mixed volume types")
        expected = tuple(opening + source * i + source for i in range(count))
        actual = tuple(c.timestamp.astimezone(UTC) for c in group)
        if actual != expected:
            continue
        output.append(
            Candle(
                expected[-1],
                group[0].open,
                max(c.high for c in group),
                min(c.low for c in group),
                group[-1].close,
                sum(c.volume for c in group),
                group[0].volume_type,
            )
        )
    return tuple(output)
