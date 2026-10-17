"""Idempotent batch importer through Astra's existing Journal abstraction."""

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..history import TIMEFRAMES, load_csv
from ..repositories import Entity, Journal, Record
from .validation import inspect_candles


def import_csv(
    path: str | Path, symbol: str, timeframe: str, journal: Journal, *, batch_size=1000
) -> int:
    if timeframe not in TIMEFRAMES or batch_size < 1:
        raise ValueError("invalid timeframe or batch size")
    count = 0
    batch = []
    # Validate the complete input before the first database mutation. Reuse the
    # validated snapshot so a changed CSV cannot slip between preflight and import.
    data = load_csv(path, symbol, timeframe, min_candles=1)
    quality = inspect_candles(data.candles, symbol, timeframe)
    if quality.status != "PASS":
        raise ValueError("DATA QUALITY INSUFFICIENT: historical import rejected")
    version = data.metadata["dataset_version"]
    for candle in data.candles:
        stamp = candle.timestamp
        identity = f"{version}:{symbol}:{timeframe}:{stamp.isoformat()}"
        batch.append(
            Record(
                Entity.OBSERVATION,
                stamp,
                {
                    "candle": candle,
                    "source": data.metadata["source"],
                    "dataset_version": version,
                    "volume_unit": data.metadata["volume_unit"],
                    "price_basis": data.metadata["price_basis"],
                },
                pair=symbol,
                timeframe=timeframe,
                record_id=str(uuid5(NAMESPACE_URL, identity)),
            )
        )
        if len(batch) == batch_size:
            journal.repository.append_batch(tuple(batch))
            count += len(batch)
            batch.clear()
    if batch:
        journal.repository.append_batch(tuple(batch))
        count += len(batch)
    return count
