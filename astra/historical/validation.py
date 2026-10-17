"""Data quality checks, with expected Forex closures separated from gaps."""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

from ..data import DataValidationError, validate
from ..history import TIMEFRAMES, load_csv
from ..models import Candle, MarketData
from .models import Gap, ValidationReport

_NY = ZoneInfo("America/New_York")


def _market_open(stamp: datetime) -> bool:
    local = stamp.astimezone(_NY)
    weekday = local.weekday()
    minute = local.hour * 60 + local.minute
    if weekday == 5 or (weekday == 6 and minute < 17 * 60):
        return False
    return not (weekday == 4 and minute >= 17 * 60)


def _bar_crosses_closed_market(opening: datetime, interval: timedelta) -> bool:
    """Return true when a target bar is not wholly inside the FX open session."""
    minutes = int(interval.total_seconds() // 60)
    return not all(
        _market_open(opening + timedelta(minutes=minute)) for minute in range(minutes)
    )


def _csv_failure_counts(path, symbol, timeframe) -> tuple[int, int, int, int, int]:
    """Preserve useful counts after strict ingestion rejects a malformed file."""
    rows = duplicates = conflicts = invalid = reversed_rows = 0
    seen: dict[datetime, Candle] = {}
    previous = None
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            try:
                stamp = datetime.fromisoformat(row["timestamp"])
                if stamp.tzinfo is None or stamp.utcoffset() is None:
                    raise ValueError("timestamp must be timezone-aware")
                stamp = stamp.astimezone(UTC)
                candle = Candle(
                    stamp,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(
                        row["tick_volume"] if "tick_volume" in row else row["volume"]
                    ),
                    (row.get("volume_type") or "tick").strip().lower(),
                )
                validate(MarketData(symbol, timeframe, (candle,)), min_candles=1)
                if stamp.timestamp() % TIMEFRAMES[timeframe].total_seconds():
                    raise ValueError("misaligned timestamp")
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            if stamp in seen:
                duplicates += 1
                conflicts += seen[stamp] != candle
            if previous is not None and stamp < previous:
                reversed_rows += 1
            previous = stamp
            seen[stamp] = candle
    return rows, duplicates, conflicts, invalid, reversed_rows


def inspect_csv(
    path: str | Path,
    symbol: str,
    timeframe: str,
    *,
    max_gap_minutes=15,
    expected_start: datetime | None = None,
    expected_end: datetime | None = None,
) -> ValidationReport:
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    try:
        data = load_csv(path, symbol, timeframe, allowed_pairs=(symbol,), min_candles=1)
    except (DataValidationError, ValueError) as error:
        try:
            counts = _csv_failure_counts(path, symbol, timeframe)
        except (UnicodeError, csv.Error):
            counts = (0, 0, 0, 1, 0)
        return ValidationReport(
            symbol,
            timeframe,
            *counts,
            0,
            (),
            None,
            None,
            digest,
            "FAIL",
            (f"{type(error).__name__}: {error}",),
        )
    return inspect_candles(
        data.candles,
        symbol,
        timeframe,
        max_gap_minutes=max_gap_minutes,
        checksum=digest,
        expected_start=expected_start,
        expected_end=expected_end,
    )


def inspect_candles(
    candles,
    symbol,
    timeframe,
    *,
    max_gap_minutes=15,
    checksum="",
    expected_start: datetime | None = None,
    expected_end: datetime | None = None,
) -> ValidationReport:
    """Check an observed window; optional bounds are UTC source-open [start, end).

    Every missing open-session bar is reported. The legacy gap threshold only
    annotates severity; it never turns incomplete data into PASS.
    """
    if max_gap_minutes < 0 or timeframe not in TIMEFRAMES:
        raise ValueError("invalid quality-check configuration")
    candles = tuple(candles)
    interval = TIMEFRAMES[timeframe]
    warnings = []
    try:
        validate(MarketData(symbol, timeframe, candles), min_candles=1)
        if any(c.timestamp.timestamp() % interval.total_seconds() for c in candles):
            raise DataValidationError(
                "candles are not aligned to UTC timeframe boundaries"
            )
    except (DataValidationError, ValueError) as error:
        return ValidationReport(
            symbol,
            timeframe,
            len(candles),
            0,
            0,
            1,
            0,
            0,
            (),
            None,
            None,
            checksum,
            "FAIL",
            (str(error),),
        )
    for bound in (expected_start, expected_end):
        if bound is not None and (bound.tzinfo is None or bound.utcoffset() is None):
            raise ValueError("expected range bounds must be timezone-aware")
    if expected_start and expected_end and expected_end <= expected_start:
        raise ValueError("expected range must be nonempty")
    openings = [c.timestamp.astimezone(UTC) - interval for c in candles]
    if (
        expected_start
        and openings[0] < expected_start
        or expected_end
        and candles[-1].timestamp > expected_end
    ):
        warnings.append("observations fall outside the requested range")
    gaps = []
    expected = 0
    boundaries = [c.timestamp for c in candles]
    if expected_start is not None:
        # First fully-contained target bar; edge fragments cannot be emitted.
        seconds = interval.total_seconds()
        stamp = expected_start.timestamp()
        first_open = datetime.fromtimestamp(-(-stamp // seconds) * seconds, UTC)
        boundaries.insert(0, first_open)
    if expected_end is not None:
        seconds = interval.total_seconds()
        last_close = datetime.fromtimestamp(
            expected_end.timestamp() // seconds * seconds, UTC
        )
        boundaries.append(last_close + interval)
    for previous, current in pairwise(boundaries):
        delta = current - previous
        if delta <= interval:
            continue
        missing = max(0, int(delta.total_seconds() // interval.total_seconds()) - 1)
        starts = (previous + interval * step for step in range(missing))
        open_missing = sum(
            not _bar_crosses_closed_market(opening, interval) for opening in starts
        )
        if open_missing == 0:
            expected += 1
            continue
        minutes = open_missing * int(interval.total_seconds() // 60)
        gaps.append(Gap(previous, current, minutes, "market_open_gap"))
        if minutes >= max_gap_minutes:
            warnings.append(f"open-session gap of {minutes} minutes")
    closed_rows = sum(
        _bar_crosses_closed_market(opening, interval) for opening in openings
    )
    if closed_rows:
        warnings.append(f"{closed_rows} candles overlap the assumed FX weekend closure")
    status = "WARN" if gaps or warnings else "PASS"
    return ValidationReport(
        symbol,
        timeframe,
        len(candles),
        0,
        0,
        0,
        0,
        expected,
        tuple(gaps),
        candles[0].timestamp,
        candles[-1].timestamp,
        checksum,
        status,
        tuple(warnings),
    )
