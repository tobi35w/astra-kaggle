"""Normalize verified raw daily files into sorted, close-timestamped CSV."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ..models import Candle
from .downloader import _file, iter_days
from .dukascopy import decode_m1_bi5
from .manifest import ManifestStore
from .validation import _market_open

FIELDS = ("timestamp", "open", "high", "low", "close", "volume", "volume_type")


def normalize_symbol(root, symbol: str, start: date, end: date) -> Path:
    root = Path(root)
    manifest = ManifestStore(root)
    unique: dict[datetime, Candle] = {}
    for day in iter_days(start, end):
        path = _file(root, symbol, day)
        entry = manifest.unit(symbol, day.isoformat())
        if not path.is_file():
            if (
                entry
                and entry.get("status") == "closed"
                and not any(
                    _market_open(
                        datetime.combine(day, time.min, UTC) + timedelta(minutes=minute)
                    )
                    for minute in range(1440)
                )
            ):
                continue
            raise ValueError(
                "verified raw M1 file is missing or the closure is unverified"
            )
        raw = path.read_bytes()
        if (
            not entry
            or entry.get("status") != "downloaded"
            or entry.get("checksum") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("raw M1 file does not match a verified manifest entry")
        for candle in decode_m1_bi5(raw, symbol, day):
            if not _market_open(candle.timestamp - timedelta(minutes=1)):
                if candle.volume != 0 or not (
                    candle.open == candle.high == candle.low == candle.close
                ):
                    raise ValueError(
                        "closed-session source candle is not flat zero-volume padding"
                    )
                continue
            old = unique.get(candle.timestamp)
            if old is not None and old != candle:
                raise ValueError(
                    f"conflicting duplicate candle at {candle.timestamp.isoformat()}"
                )
            unique[candle.timestamp] = candle
    if not unique:
        raise ValueError(f"no verified raw M1 files available for {symbol}")
    target = root / "normalized" / "m1" / f"{symbol}.csv"
    from .pipeline import _write_candles

    _write_candles(
        target,
        (candle for _, candle in sorted(unique.items())),
        volume_unit="provider_raw",
    )
    return target
