"""Explicit source-open to availability conversion, without modifying cached rows."""

from dataclasses import replace
from datetime import timedelta
from hashlib import sha256


def availability_view(data, shift_minutes=1):
    if shift_minutes not in (0, 1):
        raise ValueError("supported timestamp shifts are zero or one minute")
    if not shift_minutes:
        return data
    if data.metadata.get("source_timestamp_shift_minutes", 0):
        raise ValueError("source timestamps have already been shifted")
    version = data.metadata["dataset_version"]
    return replace(
        data,
        candles=tuple(
            replace(c, timestamp=c.timestamp + timedelta(minutes=1))
            for c in data.candles
        ),
        metadata={
            **data.metadata,
            "raw_dataset_version": version,
            "dataset_version": "sha256:"
            + sha256((version + ":open-plus-one-minute").encode()).hexdigest(),
            "timestamp_convention": "raw source open time +1 minute = close/availability time",
            "source_timestamp_shift_minutes": 1,
            "sampled_price_basis": "bid; sampled Dukascopy matches, whole snapshot not authenticated",
        },
    )
