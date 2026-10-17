"""Tests for the validated future-adapter gateway and advisory export contracts."""

from dataclasses import replace
from datetime import timedelta

import pytest

from astra.data import DataValidationError
from astra.integration import MarketDataGateway, advisory_context
from astra.models import (
    Action,
    ConfidenceBreakdown,
    Decision,
    Regime,
    RiskApproval,
    Side,
    Signal,
)

PAIR, TIMEFRAME = "EURUSD", "M5"
METADATA = {
    "source": "synthetic_test_fixture",
    "timestamp_role": "bar_close",
    "price_basis": "mid",
    "dataset_version": "astra-sha256:synthetic",
    "quality_status": "PASS",
}


class _FakeSource:
    """Injected stand-in for a future market-data provider; no network."""

    def __init__(self, data):
        self.data = data
        self.calls = 0

    def closed_bars(self, pair, timeframe, as_of):
        self.calls += 1
        return replace(
            self.data,
            candles=tuple(c for c in self.data.candles if c.timestamp <= as_of),
        )


@pytest.fixture
def source(market_data):
    return _FakeSource(replace(market_data(40), metadata=dict(METADATA)))


@pytest.fixture
def as_of(source):
    return source.data.candles[-1].timestamp


def test_gateway_admits_valid_fresh_bid_basis_data(source, as_of):
    gateway = MarketDataGateway(source)
    data = gateway.load(PAIR, TIMEFRAME, as_of)
    assert len(data.candles) == 40
    assert source.calls == 1


def test_gateway_rejects_bad_requests(source, as_of):
    gateway = MarketDataGateway(source)
    with pytest.raises(DataValidationError, match="invalid market-data request"):
        gateway.load("NOTPAIR", TIMEFRAME, as_of)
    with pytest.raises(DataValidationError, match="invalid market-data request"):
        gateway.load(PAIR, "D1", as_of)
    with pytest.raises(DataValidationError, match="invalid market-data request"):
        gateway.load(PAIR, TIMEFRAME, as_of.replace(tzinfo=None))


def test_gateway_validates_source_freshness(source, as_of):
    gateway = MarketDataGateway(source)
    with pytest.raises(DataValidationError, match="stale"):
        gateway.load(PAIR, TIMEFRAME, as_of + timedelta(hours=2))

    class _FutureSource:
        """A misbehaving provider that leaks a candle after the decision time."""

        def closed_bars(self, pair, timeframe, at):
            return replace(
                source.data,
                candles=source.data.candles
                + (
                    replace(
                        source.data.candles[-1], timestamp=at + timedelta(minutes=5)
                    ),
                ),
            )

    with pytest.raises(DataValidationError, match="future-dated"):
        MarketDataGateway(_FutureSource()).load(PAIR, TIMEFRAME, as_of)

    # A future price inside an otherwise fresh series must not be admitted.
    poisoned = replace(
        source.data,
        candles=source.data.candles
        + (
            replace(
                source.data.candles[-1],
                close=999.0,
                timestamp=as_of + timedelta(minutes=5),
            ),
        ),
    )
    with pytest.raises(DataValidationError):
        MarketDataGateway(_FakeSource(poisoned)).load(
            PAIR, TIMEFRAME, as_of + timedelta(minutes=10)
        )


def test_gateway_requires_close_timestamps_and_declared_price_basis(source, as_of):
    wrong_role = replace(
        source.data, metadata=dict(source.data.metadata, timestamp_role="bar_open")
    )
    with pytest.raises(DataValidationError, match="close/availability"):
        MarketDataGateway(_FakeSource(wrong_role)).load(PAIR, TIMEFRAME, as_of)
    wrong_basis = replace(
        source.data, metadata=dict(source.data.metadata, price_basis="ask")
    )
    with pytest.raises(DataValidationError, match="price basis"):
        MarketDataGateway(_FakeSource(wrong_basis)).load(PAIR, TIMEFRAME, as_of)


def test_gateway_rejects_quality_insufficient_history(market_data, as_of):
    # A missing open-session bar is a data-quality failure, not something to fill.
    candles = market_data(40).candles
    holed = candles[:10] + candles[13:]
    data = replace(market_data(40), candles=holed, metadata=dict(METADATA))
    with pytest.raises(DataValidationError, match="DATA QUALITY INSUFFICIENT"):
        MarketDataGateway(_FakeSource(data)).load(PAIR, TIMEFRAME, as_of)


def test_gateway_rejects_mismatched_instrument_or_timeframe(source, as_of):
    wrong_pair = replace(source.data, pair="GBPUSD")
    with pytest.raises(DataValidationError, match="different instrument"):
        MarketDataGateway(_FakeSource(wrong_pair)).load(PAIR, TIMEFRAME, as_of)
    wrong_timeframe = replace(source.data, timeframe="M15")
    with pytest.raises(DataValidationError, match="different instrument"):
        MarketDataGateway(_FakeSource(wrong_timeframe)).load(PAIR, TIMEFRAME, as_of)


def _decision():
    signal = Signal("ema_trend", Side.LONG, 0.7, 1.1, 1.09, 1.13, "fixture", 2.0)
    return Decision(
        market_data_stamp(),
        PAIR,
        TIMEFRAME,
        Action.NO_TRADE,
        "weak setup: best candidate score is below threshold",
        Regime.RANGING,
        signal,
        RiskApproval(False, ("weak setup",)),
        ConfidenceBreakdown(0.8, 0.5, 0.7, 0.4, 1.0, 0.42, "weak evidence"),
        reason_codes=("WEAK_SETUP",),
        context={
            "dataset": {
                "source": "synthetic_test_fixture",
                "price_basis": "mid",
                "dataset_version": "astra-sha256:synthetic",
                "quality_status": "PASS",
                "canonical_m1_sha256": "synthetic",
                "path": "/must/not/be/exported",
            },
            "configuration": {"secret": "must-not-be-exported"},
        },
    )


def market_data_stamp():
    from datetime import UTC, datetime

    return datetime(2026, 10, 13, tzinfo=UTC)


def test_advisory_context_exports_facts_only():
    context = advisory_context(_decision())
    assert context.schema_version == 1
    assert context.decision_time == market_data_stamp()
    payload = context.payload_json
    assert "research interpretation only" in payload
    assert "WEAK_SETUP" in payload
    assert "astra-sha256:synthetic" in payload
    # No configuration, credentials, or local paths leave the process.
    assert "secret" not in payload
    assert "/must/not/be/exported" not in payload
    # The identity digest is stable for identical facts.
    assert advisory_context(_decision()).context_id == context.context_id
