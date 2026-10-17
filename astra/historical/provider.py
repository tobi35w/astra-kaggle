"""Source-independent historical M1 provider boundary."""

from __future__ import annotations

from datetime import date
from typing import Protocol


class TransientSourceError(RuntimeError):
    """A source failure that is safe to retry."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        reason_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.reason_code = reason_code


class PermanentSourceError(RuntimeError):
    """A non-retryable source failure."""


class HistoricalDataProvider(Protocol):
    source_name: str

    def build_m1_url(self, symbol: str, day: date) -> str: ...

    def fetch_m1_day(self, symbol: str, day: date, timeout: float) -> bytes | None: ...
