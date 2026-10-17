"""Deterministic test fixtures; these are NOT historical market data."""

import math
import os
import socket
from datetime import UTC, datetime, timedelta

import pytest

from astra.models import Candle, MarketData
from astra.repositories import Journal, MemoryRepository


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setenv("ASTRA_PERSISTENCE_BACKEND", "memory")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch, request):
    if (
        request.node.get_closest_marker("live")
        and os.getenv("ASTRA_RUN_LIVE_TESTS") == "true"
    ):
        return
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect

    def local_only(host, *args, **kwargs):
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("Normal tests cannot resolve external services")
        return original_resolve(host, *args, **kwargs)

    def connect_local(sock, address):
        if isinstance(address, tuple) and address[0] not in (
            "127.0.0.1",
            "::1",
            "localhost",
        ):
            raise AssertionError("Normal tests cannot connect to external services")
        return original_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", local_only)
    monkeypatch.setattr(socket.socket, "connect", connect_local)


@pytest.fixture
def journal():
    return Journal(MemoryRepository())


@pytest.fixture
def market_data():
    def make(n=100, timeframe="M5", direction=1, oscillate=False, start=None):
        minutes = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}[timeframe]
        start = start or datetime(2025, 1, 1, tzinfo=UTC)
        candles = []
        for i in range(n):
            price = (
                1.1
                + direction * i * 0.0002
                + (math.sin(i * 0.5) * 0.003 if oscillate else 0)
            )
            candles.append(
                Candle(
                    start + timedelta(minutes=minutes * i),
                    price - 0.0001,
                    price + 0.0006,
                    price - 0.0006,
                    price,
                    10,
                )
            )
        return MarketData(
            "EURUSD", timeframe, tuple(candles), {"source": "synthetic_test_fixture"}
        )

    return make
