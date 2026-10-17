"""Strict CSV OHLCV ingestion for real historical data supplied by the user."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .data import DataValidationError, validate
from .models import Candle, MarketData

TIMEFRAMES = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
}
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")


def load_csv(
    path: str | Path,
    pair: str,
    timeframe: str,
    timestamp_column: str = "timestamp",
    *,
    allowed_pairs: tuple[str, ...] = PAIRS,
    reject_gaps: bool = False,
    min_candles: int = 2,
) -> MarketData:
    if pair not in allowed_pairs:
        raise DataValidationError("unsupported currency pair")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    candles = []
    volume_units = set()
    price_bases = set()
    contents = Path(path).read_bytes()
    with io.StringIO(contents.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        required = {timestamp_column, "open", "high", "low", "close"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise DataValidationError(
                f"CSV must contain columns: {', '.join(sorted(required))}"
            )
        volume_column = (
            "tick_volume"
            if "tick_volume" in reader.fieldnames
            else "volume"
            if "volume" in reader.fieldnames
            else None
        )
        if volume_column is None:
            raise DataValidationError(
                "CSV must contain either a volume or tick_volume column"
            )
        for line, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat(row[timestamp_column].strip())
                if timestamp.tzinfo is None:
                    raise ValueError("timestamp has no timezone offset")
                candles.append(
                    Candle(
                        timestamp.astimezone(UTC),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row[volume_column]),
                        (row.get("volume_type") or "tick").strip().lower(),
                    )
                )
                volume_units.add((row.get("volume_unit") or "").strip())
                price_bases.add((row.get("price_basis") or "mid").strip().lower())
            except (TypeError, ValueError) as error:
                raise DataValidationError(f"invalid CSV row {line}: {error}") from error
    if len({candle.volume_type for candle in candles}) > 1:
        raise DataValidationError("CSV mixes volume types")
    if len(volume_units - {""}) > 1:
        raise DataValidationError("CSV mixes volume units")
    if not price_bases.issubset({"mid", "bid", "ask"}):
        raise DataValidationError("CSV contains an unsupported price basis")
    if len(price_bases) > 1:
        raise DataValidationError("CSV mixes price bases")
    data = MarketData(
        pair,
        timeframe,
        tuple(candles),
        {
            "source": "csv",
            "path": str(path),
            "timestamp_role": "bar_close",
            "volume_type": candles[0].volume_type if candles else "tick",
            "volume_unit": next(iter(volume_units - {""}), None),
            "price_basis": next(iter(price_bases), "mid"),
            "dataset_version": "astra-sha256:" + hashlib.sha256(contents).hexdigest(),
        },
    )
    validate(data, min_candles=min_candles)
    interval = TIMEFRAMES[timeframe].total_seconds()
    if any(c.timestamp.timestamp() % interval for c in candles):
        raise DataValidationError(
            "timestamps are not aligned to UTC timeframe boundaries"
        )
    # Each candle was already checked against the UTC timeframe boundary above,
    # so every pairwise interval is necessarily an integer multiple too.
    if reject_gaps and missing_candles(data):
        raise DataValidationError(
            "missing candles; inspect market calendar before accepting gaps"
        )
    return data


def missing_candles(data: MarketData) -> tuple[tuple[datetime, datetime], ...]:
    interval = TIMEFRAMES[data.timeframe]
    return tuple(
        (a.timestamp, b.timestamp)
        for a, b in zip(data.candles, data.candles[1:])
        if b.timestamp - a.timestamp > interval * 1.5
    )
