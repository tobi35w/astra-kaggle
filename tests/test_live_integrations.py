"""One tiny public-source request, opt-in only. No keys or broker connections."""

import os
from datetime import date

import pytest

from astra.historical.dukascopy import DukascopyProvider, decode_m1_bi5

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("ASTRA_RUN_LIVE_TESTS") != "true",
        reason="real internet checks require explicit opt-in",
    ),
]


def test_real_dukascopy_known_day():
    provider = DukascopyProvider(
        os.getenv("HISTORICAL_DATA_BASE_URL", "https://datafeed.dukascopy.com/datafeed")
    )
    day = date(2024, 1, 2)
    payload = provider.fetch_m1_day("EURUSD", day, timeout=15)
    assert payload
    candles = decode_m1_bi5(payload, "EURUSD", day)
    assert len(candles) == 1440
    assert candles[0].timestamp.isoformat() == "2024-01-02T00:01:00+00:00"
