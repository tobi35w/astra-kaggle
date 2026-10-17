"""Environment-backed defaults for the local historical pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

ACTIVE_HISTORICAL_SYMBOLS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")


@dataclass(frozen=True)
class HistoricalSettings:
    root: Path = Path("data")
    symbols: tuple[str, ...] = ACTIVE_HISTORICAL_SYMBOLS
    start_date: date = date(2018, 1, 1)
    end_date: date | None = None
    workers: int = 2
    retries: int = 4
    timeout: float = 15.0
    read_timeout: float | None = None
    max_elapsed_seconds: float = 180.0
    retry_jitter_seconds: float = 0.5
    max_consecutive_failures: int = 3
    max_gap_minutes: int = 15
    batch_size: int = 1000
    base_url: str = "https://datafeed.dukascopy.com/datafeed"

    @classmethod
    def from_env(cls) -> HistoricalSettings:
        raw_symbols = os.getenv("HISTORICAL_DATA_SYMBOLS", "")
        symbols = (
            tuple(
                part.upper()
                for part in re.split(r"[\s,]+", raw_symbols.strip())
                if part
            )
            or ACTIVE_HISTORICAL_SYMBOLS
        )
        raw_start = os.getenv("HISTORICAL_DATA_START_DATE", "2018-01-01")
        raw_end = os.getenv("HISTORICAL_DATA_END_DATE", "").strip()
        try:
            start = date.fromisoformat(raw_start)
            end = date.fromisoformat(raw_end) if raw_end else None
        except ValueError:
            raise ValueError("invalid historical data date configuration") from None
        try:
            workers = int(os.getenv("HISTORICAL_DATA_WORKERS", "2"))
            retries = int(os.getenv("HISTORICAL_DATA_RETRIES", "4"))
            timeout = float(os.getenv("HISTORICAL_DATA_TIMEOUT", "15"))
            read_timeout = float(
                os.getenv("HISTORICAL_DATA_READ_TIMEOUT", str(timeout))
            )
            max_elapsed = float(os.getenv("HISTORICAL_DATA_MAX_ELAPSED_SECONDS", "180"))
            retry_jitter = float(
                os.getenv("HISTORICAL_DATA_RETRY_JITTER_SECONDS", "0.5")
            )
            max_consecutive = int(
                os.getenv("HISTORICAL_DATA_MAX_CONSECUTIVE_FAILURES", "3")
            )
            max_gap = int(os.getenv("HISTORICAL_DATA_MAX_GAP_MINUTES", "15"))
            batch_size = int(os.getenv("HISTORICAL_DATA_BATCH_SIZE", "1000"))
        except ValueError:
            raise ValueError("invalid numeric historical data configuration") from None
        root = Path(os.getenv("HISTORICAL_DATA_ROOT", "data"))
        settings = cls(
            root=root,
            symbols=symbols,
            start_date=start,
            end_date=end,
            workers=workers,
            retries=retries,
            timeout=timeout,
            read_timeout=read_timeout,
            max_elapsed_seconds=max_elapsed,
            retry_jitter_seconds=retry_jitter,
            max_consecutive_failures=max_consecutive,
            max_gap_minutes=max_gap,
            batch_size=batch_size,
            base_url=os.getenv(
                "HISTORICAL_DATA_BASE_URL", "https://datafeed.dukascopy.com/datafeed"
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        today = datetime.now(UTC).date()
        if self.start_date > today:
            raise ValueError("historical start date cannot be in the future")
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("historical end date must not precede start date")
        if self.end_date is not None and self.end_date > today:
            raise ValueError("historical end date cannot be in the future")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("historical symbols must be nonempty and unique")
        if any(not re.fullmatch(r"[A-Z]{6}", symbol) for symbol in self.symbols):
            raise ValueError("historical symbols must be six uppercase letters")
        if any(symbol not in ACTIVE_HISTORICAL_SYMBOLS for symbol in self.symbols):
            raise ValueError(
                "historical downloads are limited to AUDUSD, EURUSD, GBPUSD and USDJPY"
            )
        if not 1 <= self.workers <= 8:
            raise ValueError("historical workers must be between 1 and 8")
        if not 0 <= self.retries <= 10:
            raise ValueError("historical retries must be between 0 and 10")
        if not 1 <= self.timeout <= 120:
            raise ValueError("historical timeout must be between 1 and 120 seconds")
        read_timeout = self.timeout if self.read_timeout is None else self.read_timeout
        if not 1 <= read_timeout <= 120:
            raise ValueError(
                "historical read timeout must be between 1 and 120 seconds"
            )
        if not read_timeout <= self.max_elapsed_seconds <= 3600:
            raise ValueError(
                "historical elapsed budget must cover a read and be <= 3600 seconds"
            )
        if not 0 <= self.retry_jitter_seconds <= 10:
            raise ValueError("historical retry jitter must be between 0 and 10 seconds")
        if not 1 <= self.max_consecutive_failures <= 20:
            raise ValueError(
                "historical consecutive failure limit must be between 1 and 20"
            )
        if not 0 <= self.max_gap_minutes <= 1440:
            raise ValueError("maximum gap minutes must be between 0 and 1440")
        if not 1 <= self.batch_size <= 50000:
            raise ValueError("historical import batch size must be 1–50000")
        if not str(self.root).strip():
            raise ValueError("historical data root cannot be empty")
        if self.base_url.rstrip("/") not in {
            "https://datafeed.dukascopy.com/datafeed",
            "https://www.dukascopy.com/datafeed",
        }:
            raise ValueError("historical base URL must be an approved Dukascopy route")
