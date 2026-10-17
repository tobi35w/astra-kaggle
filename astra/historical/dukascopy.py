"""Dukascopy public daily M1 BID candle feed and BI5 decoder."""

from __future__ import annotations

import lzma
import math
import struct
import time
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..data import DataValidationError
from ..models import Candle
from .provider import PermanentSourceError, TransientSourceError

_M1_RECORD = struct.Struct(">IIIIIf")
_MAX_COMPRESSED_DAY = 2_000_000
_MAX_RETRY_AFTER_SECONDS = 300.0


def _retry_after_seconds(headers) -> float | None:
    """Parse a bounded Retry-After value without retaining source headers."""
    if not headers:
        return None
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return min(_MAX_RETRY_AFTER_SECONDS, max(0.0, seconds))


class DukascopyProvider:
    source_name = "Dukascopy public BID M1 candles"
    supports_deadline_controls = True

    def __init__(
        self,
        base_url: str = "https://datafeed.dukascopy.com/datafeed",
        *,
        transport=None,
    ):
        self.base_url = base_url.rstrip("/")
        self._transport = transport or urlopen

    def build_m1_url(self, symbol: str, day: date) -> str:
        if len(symbol) != 6 or not symbol.isalpha() or symbol.upper() != symbol:
            raise ValueError("Dukascopy FX symbols must be six uppercase letters")
        return (
            f"{self.base_url}/{symbol}/{day.year}/{day.month - 1:02d}/"
            f"{day.day:02d}/BID_candles_min_1.bi5"
        )

    def fetch_m1_day(
        self,
        symbol: str,
        day: date,
        timeout: float,
        *,
        read_timeout: float | None = None,
        deadline: float | None = None,
    ) -> bytes | None:
        url = self.build_m1_url(symbol, day)
        request = Request(url, headers={"User-Agent": "Astra historical data client"})
        try:
            with self._transport(request, timeout=timeout) as response:
                status = response.status
                if status != 200:
                    if status in (408, 425, 429) or status >= 500:
                        raise TransientSourceError(
                            f"Dukascopy returned HTTP {status}",
                            status_code=status,
                            retry_after_seconds=_retry_after_seconds(
                                getattr(response, "headers", None)
                            ),
                        )
                    raise PermanentSourceError(f"Dukascopy returned HTTP {status}")
                payload = self._read_bounded(
                    response,
                    read_timeout=read_timeout or timeout,
                    deadline=deadline,
                )
        except HTTPError as error:
            if error.code == 404:
                return None
            if error.code in (408, 425, 429) or error.code >= 500:
                raise TransientSourceError(
                    f"Dukascopy returned HTTP {error.code}",
                    status_code=error.code,
                    retry_after_seconds=_retry_after_seconds(error.headers),
                ) from None
            raise PermanentSourceError(
                f"Dukascopy returned HTTP {error.code}"
            ) from None
        except (TimeoutError, URLError, OSError) as error:
            raise TransientSourceError(
                f"Dukascopy request failed ({type(error).__name__})",
                reason_code=type(error).__name__,
            ) from None
        if len(payload) > _MAX_COMPRESSED_DAY:
            raise PermanentSourceError("Dukascopy daily response exceeded size limit")
        return payload or None

    @staticmethod
    def _read_bounded(
        response, *, read_timeout: float, deadline: float | None
    ) -> bytes:
        """Read bounded chunks under a read timeout and optional total deadline."""
        chunks = bytearray()
        raw = getattr(getattr(response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        while True:
            timeout = read_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("historical response exceeded its time budget")
                timeout = min(timeout, remaining)
            if sock is not None:
                sock.settimeout(timeout)
            chunk = response.read(min(65536, _MAX_COMPRESSED_DAY + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
            if len(chunks) > _MAX_COMPRESSED_DAY:
                break
        return bytes(chunks)


def price_scale_for(symbol: str) -> int:
    return 1000 if symbol.endswith("JPY") else 100000


def decode_m1_bi5(
    payload: bytes, symbol: str, day: date, price_scale: int | None = None
) -> tuple[Candle, ...]:
    """Decode one LZMA BI5 BID M1 file into UTC candle-close records.

    Dukascopy candle records contain second offset, open, close, low, high and
    a floating volume value. Source seconds refer to candle starts; Astra stores
    candle availability/close timestamps, so one minute is added exactly once.
    """
    if not payload:
        return ()
    try:
        decoder = lzma.LZMADecompressor(memlimit=32 * 1024 * 1024)
        raw = decoder.decompress(payload, max_length=1441 * _M1_RECORD.size)
    except lzma.LZMAError:
        raise DataValidationError(
            "Dukascopy BI5 payload is not valid LZMA data"
        ) from None
    if not decoder.eof or decoder.unused_data:
        raise DataValidationError(
            "Dukascopy BI5 payload is truncated or exceeds one day"
        )
    if not raw or len(raw) % _M1_RECORD.size:
        raise DataValidationError("Dukascopy BI5 payload has an incomplete M1 record")
    if len(raw) // _M1_RECORD.size > 1440:
        raise DataValidationError("Dukascopy BI5 payload contains too many M1 candles")

    scale = price_scale if price_scale is not None else price_scale_for(symbol)
    if scale <= 0:
        raise ValueError("price scale must be positive")
    day_start = datetime.combine(day, datetime.min.time(), UTC)
    candles: list[Candle] = []
    previous_seconds = -1
    for seconds, opening, closing, low, high, volume in _M1_RECORD.iter_unpack(raw):
        prices = (opening, closing, low, high)
        if seconds >= 86400 or seconds % 60 or min(prices) <= 0:
            raise DataValidationError("Dukascopy BI5 contains invalid M1 fields")
        if not math.isfinite(volume) or volume < 0:
            raise DataValidationError("Dukascopy BI5 contains invalid volume")
        if seconds <= previous_seconds:
            raise DataValidationError(
                "Dukascopy BI5 timestamps must be ordered and unique"
            )
        previous_seconds = seconds
        candle = Candle(
            timestamp=day_start + timedelta(seconds=seconds + 60),
            open=opening / scale,
            high=high / scale,
            low=low / scale,
            close=closing / scale,
            volume=volume,
            volume_type="provider",
        )
        if candle.low > min(candle.open, candle.close) or candle.high < max(
            candle.open, candle.close
        ):
            raise DataValidationError("Dukascopy BI5 contains impossible OHLC values")
        candles.append(candle)
    return tuple(candles)
