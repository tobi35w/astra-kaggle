"""Resumable, bounded downloader for daily Dukascopy M1 files."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path

from ..data import DataValidationError
from .config import HistoricalSettings
from .dukascopy import DukascopyProvider, decode_m1_bi5
from .manifest import ManifestStore
from .models import DownloadSummary, DownloadUnitResult
from .provider import PermanentSourceError, TransientSourceError
from .validation import _market_open


def iter_days(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _file(root: Path, symbol: str, day: date) -> Path:
    return (
        root
        / "raw"
        / "dukascopy"
        / symbol
        / str(day.year)
        / f"{day.month - 1:02d}"
        / f"{day.day:02d}_BID_candles_min_1.bi5"
    )


class HistoricalDownloader:
    def __init__(
        self,
        provider=None,
        *,
        root="data",
        workers=2,
        retries=4,
        timeout=15,
        read_timeout=None,
        max_elapsed_seconds=180.0,
        retry_jitter_seconds=0.5,
        max_consecutive_failures=3,
        manifest=None,
        sleep=time.sleep,
        request_interval=0.5,
        progress=None,
    ):
        self.provider = provider or DukascopyProvider()
        self.root = Path(root)
        self.workers = workers
        self.retries = retries
        self.timeout = timeout
        self.read_timeout = timeout if read_timeout is None else read_timeout
        self.max_elapsed_seconds = max_elapsed_seconds
        self.retry_jitter_seconds = retry_jitter_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.manifest = manifest or ManifestStore(self.root)
        self.sleep = sleep
        if request_interval < 0:
            raise ValueError("request interval cannot be negative")
        self.request_interval = request_interval
        self._rate_lock = threading.Lock()
        self._last_request = 0.0
        self._paused = threading.Event()
        self._failure_lock = threading.Lock()
        self._consecutive_failures = 0
        self._pause_reason = "provider circuit is open"
        self.progress = progress

    def download(self, symbols, start: date, end: date, *, dry_run=False, force=False):
        HistoricalSettings(
            root=self.root,
            symbols=tuple(symbols),
            start_date=start,
            end_date=end,
            workers=self.workers,
            retries=self.retries,
            timeout=self.timeout,
            read_timeout=self.read_timeout,
            max_elapsed_seconds=self.max_elapsed_seconds,
            retry_jitter_seconds=self.retry_jitter_seconds,
            max_consecutive_failures=self.max_consecutive_failures,
        ).validate()
        units = [(symbol, day) for symbol in symbols for day in iter_days(start, end)]
        if dry_run:
            return DownloadSummary(
                len(units),
                0,
                0,
                0,
                0,
                0,
                tuple(
                    DownloadUnitResult(
                        symbol, day, "planned", str(_file(self.root, symbol, day))
                    )
                    for symbol, day in units
                ),
            )
        results = []
        if self.workers == 1:
            for index, (symbol, day) in enumerate(units):
                try:
                    result = self._download_one(symbol, day, force)
                except KeyboardInterrupt:
                    self._record_interrupted(symbol, day)
                    for pending_symbol, pending_day in units[index + 1 :]:
                        self._record_deferred(
                            pending_symbol, pending_day, "interrupted before request"
                        )
                    raise
                except Exception as error:  # noqa: BLE001 - isolate and sanitize source failures
                    result = self._record_unexpected_failure(symbol, day, error)
                results.append(result)
                self._report_progress(result, len(results), len(units))
        else:
            pool = ThreadPoolExecutor(max_workers=self.workers)
            pending_units = iter(units)
            futures: dict[Future, tuple[str, date]] = {}

            def submit_available():
                while len(futures) < self.workers and not self._paused.is_set():
                    try:
                        symbol, day = next(pending_units)
                    except StopIteration:
                        return
                    futures[pool.submit(self._download_one, symbol, day, force)] = (
                        symbol,
                        day,
                    )

            submit_available()
            try:
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        symbol, day = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as error:  # noqa: BLE001 - isolate source failures
                            result = self._record_unexpected_failure(symbol, day, error)
                        results.append(result)
                        self._report_progress(result, len(results), len(units))
                    if self._paused.is_set():
                        for symbol, day in pending_units:
                            result = self._record_deferred(
                                symbol, day, self._pause_reason
                            )
                            results.append(result)
                            self._report_progress(result, len(results), len(units))
                        break
                    submit_available()
            except KeyboardInterrupt:
                for future, (symbol, day) in tuple(futures.items()):
                    if future.cancel():
                        self._record_deferred(symbol, day, "interrupted before request")
                for symbol, day in pending_units:
                    self._record_deferred(symbol, day, "interrupted before request")
                pool.shutdown(wait=True, cancel_futures=True)
                # Running workers finish under bounded connection/read deadlines and
                # persist their own unit results before the interrupt is returned.
                raise
            else:
                pool.shutdown(wait=True, cancel_futures=True)
        results.sort(key=lambda result: (result.symbol, result.day))
        counts = {
            state: sum(r.status == state for r in results)
            for state in (
                "downloaded",
                "cached",
                "missing",
                "failed",
                "invalid",
                "closed",
                "deferred",
            )
        }
        return DownloadSummary(len(units), **counts, results=tuple(results))

    def _record_unexpected_failure(self, symbol, day, error):
        result = DownloadUnitResult(symbol, day, "failed", error=type(error).__name__)
        self.manifest.update_unit(
            symbol,
            day.isoformat(),
            {
                "status": "failed",
                "error": result.error,
                "failure_reason": result.error,
                "source": "dukascopy-bid-m1",
                "last_attempt_at": datetime.now(UTC).isoformat(),
            },
        )
        return result

    def _record_deferred(self, symbol, day, reason):
        result = DownloadUnitResult(symbol, day, "deferred", error=reason)
        self.manifest.update_unit(
            symbol,
            day.isoformat(),
            {
                "status": "deferred",
                "error": reason,
                "failure_reason": reason,
                "attempted": False,
                "source": "dukascopy-bid-m1",
                "last_attempt_at": datetime.now(UTC).isoformat(),
            },
        )
        return result

    def _record_interrupted(self, symbol, day):
        self.manifest.update_unit(
            symbol,
            day.isoformat(),
            {
                "status": "interrupted",
                "error": "KeyboardInterrupt",
                "failure_reason": "interrupted while active",
                "source": "dukascopy-bid-m1",
                "last_attempt_at": datetime.now(UTC).isoformat(),
            },
        )

    def _note_success(self):
        with self._failure_lock:
            self._consecutive_failures = 0

    def _note_failure(self, reason):
        with self._failure_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                self._pause_reason = (
                    f"provider circuit opened after {self._consecutive_failures} "
                    f"consecutive failures ({reason})"
                )
                self._paused.set()

    def _report_progress(self, result, completed, total):
        if self.progress is not None:
            self.progress(result, completed, total)

    def _download_one(self, symbol, day, force):
        target = _file(self.root, symbol, day)
        prior = self.manifest.unit(symbol, day.isoformat())
        # A day wholly inside the FX weekend has no tradeable candles. Leave
        # already downloaded raw files untouched for reproducibility, but do
        # not repeatedly request missing Saturday files from the public feed.
        opening = datetime.combine(day, clock_time.min, UTC)
        wholly_closed = not any(
            _market_open(opening + timedelta(minutes=minute)) for minute in range(1440)
        )
        if wholly_closed and not target.is_file():
            self.manifest.update_unit(
                symbol,
                day.isoformat(),
                {"status": "closed", "source": "fx-session-calendar"},
            )
            return DownloadUnitResult(symbol, day, "closed")
        if target.is_file() and not force:
            raw = target.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            try:
                rows = len(decode_m1_bi5(raw, symbol, day))
                if (
                    not rows
                    or not prior
                    or prior.get("checksum") != digest
                    or prior.get("status") != "downloaded"
                ):
                    raise DataValidationError("daily cache is empty or unverified")
            except (DataValidationError, ValueError):
                # A damaged or unverified cache must be fetched again below.
                rows = 0
            else:
                entry = {
                    "status": "downloaded",
                    "checksum": digest,
                    "rows": rows,
                    "path": str(target.relative_to(self.root)),
                    "source": "dukascopy-bid-m1",
                    "size_bytes": len(raw),
                    "validation_status": "PASS",
                    "validated_at": datetime.now(UTC).isoformat(),
                    "retrieved_at": prior.get("retrieved_at"),
                }
                self.manifest.update_unit(symbol, day.isoformat(), entry)
                return DownloadUnitResult(
                    symbol, day, "cached", str(target), rows, digest
                )
        if self._paused.is_set():
            return self._record_deferred(symbol, day, self._pause_reason)
        error = None
        started = time.monotonic()
        deadline = started + self.max_elapsed_seconds
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = "retry time budget exhausted"
                break
            try:
                with self._rate_lock:
                    if self._paused.is_set():
                        return self._record_deferred(symbol, day, self._pause_reason)
                    delay = self.request_interval - (
                        time.monotonic() - self._last_request
                    )
                    if delay > 0:
                        if delay >= remaining:
                            error = "retry time budget exhausted during request pacing"
                            break
                        self.sleep(delay)
                    self._last_request = time.monotonic()
                connect_timeout = min(
                    self.timeout, max(0.1, deadline - time.monotonic())
                )
                read_timeout = min(
                    self.read_timeout, max(0.1, deadline - time.monotonic())
                )
                if getattr(self.provider, "supports_deadline_controls", False):
                    payload = self.provider.fetch_m1_day(
                        symbol,
                        day,
                        connect_timeout,
                        read_timeout=read_timeout,
                        deadline=deadline,
                    )
                else:
                    payload = self.provider.fetch_m1_day(symbol, day, connect_timeout)
                if payload is None:
                    self.manifest.update_unit(
                        symbol,
                        day.isoformat(),
                        {
                            "status": "missing",
                            "source": "dukascopy-bid-m1",
                            "error": "HTTP 404 or empty source unit",
                            "failure_reason": "source unit not found",
                            "last_attempt_at": datetime.now(UTC).isoformat(),
                        },
                    )
                    return DownloadUnitResult(symbol, day, "missing")
                rows = len(decode_m1_bi5(payload, symbol, day))
                if rows == 0:
                    raise DataValidationError("empty daily candle file")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                try:
                    with temporary.open("wb") as handle:
                        handle.write(payload)
                        handle.flush()
                    digest = hashlib.sha256(payload).hexdigest()
                    # Re-read and decode the staged bytes before accepting them.
                    staged = temporary.read_bytes()
                    if hashlib.sha256(staged).hexdigest() != digest:
                        raise DataValidationError("staged file hash mismatch")
                    rows = len(decode_m1_bi5(staged, symbol, day))
                    if rows == 0:
                        raise DataValidationError("empty staged daily candle file")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                entry = {
                    "status": "downloaded",
                    "checksum": digest,
                    "rows": rows,
                    "path": str(target.relative_to(self.root)),
                    "source": "dukascopy-bid-m1",
                    "size_bytes": len(payload),
                    "validation_status": "PASS",
                    "validated_at": datetime.now(UTC).isoformat(),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "attempts": attempt + 1,
                }
                self.manifest.update_unit(symbol, day.isoformat(), entry)
                self._note_success()
                return DownloadUnitResult(
                    symbol, day, "downloaded", str(target), rows, digest
                )
            except DataValidationError:
                self.manifest.update_unit(
                    symbol,
                    day.isoformat(),
                    {
                        "status": "invalid",
                        "error": "DataValidationError",
                        "failure_reason": "source payload failed BI5 validation",
                        "source": "dukascopy-bid-m1",
                        "last_attempt_at": datetime.now(UTC).isoformat(),
                    },
                )
                return DownloadUnitResult(
                    symbol, day, "invalid", error="DataValidationError"
                )
            except PermanentSourceError as exc:
                error = type(exc).__name__
                self._pause_reason = f"provider refused request ({error})"
                self._paused.set()
                break
            except TransientSourceError as exc:
                error = (
                    f"HTTP {exc.status_code}"
                    if exc.status_code
                    else (exc.reason_code or type(exc).__name__)
                )
                if exc.status_code == 429:
                    self._pause_reason = "provider rate-limited request (HTTP 429)"
                    self._paused.set()
                    break
                if attempt < self.retries:
                    backoff = min(60.0, 3.0 * 2**attempt)
                    delay = max(exc.retry_after_seconds or 0.0, backoff)
                    delay += random.uniform(0.0, self.retry_jitter_seconds)
                    remaining = deadline - time.monotonic()
                    if delay >= remaining:
                        error = error or "retry time budget exhausted"
                        break
                    self.sleep(delay)
        error = error or "SourceError"
        self._note_failure(error)
        result = DownloadUnitResult(symbol, day, "failed", error=error)
        self.manifest.update_unit(
            symbol,
            day.isoformat(),
            {
                "status": "failed",
                "error": result.error,
                "failure_reason": result.error,
                "source": "dukascopy-bid-m1",
                "attempts": min(self.retries + 1, attempt + 1),
                "last_attempt_at": datetime.now(UTC).isoformat(),
            },
        )
        return result
