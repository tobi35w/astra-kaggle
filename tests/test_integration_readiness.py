"""Synthetic contract fixtures; these are not empirical trading evidence."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from astra.backtest import BacktestCosts, run_report
from astra.config import Settings
from astra.data import DataValidationError
from astra.execution import OrderStatus, PaperExecutionProvider, PaperPortfolio
from astra.historical.dataset import load_dataset
from astra.historical.manifest import ManifestStore
from astra.historical.window import prepare_window
from astra.models import Candle, RiskApproval, Side, Signal
from astra.risk import RiskEngine
from astra.strategies import StrategyRegistry

NOW = datetime(2024, 1, 2, tzinfo=UTC)
LONG = Signal("fixture", Side.LONG, 0.9, 1.1, 1.09, 1.13, "synthetic")
SHORT = replace(LONG, side=Side.SHORT, stop_loss=1.11, take_profit=1.07)


@pytest.mark.parametrize("signal", [LONG, SHORT])
def test_bid_round_trip_charges_one_spread_and_two_slippages(signal, journal):
    book = PaperPortfolio(
        PaperExecutionProvider(0.00002, 0.00003, 1, price_basis="bid"), 1000, journal
    )
    position = book.open(
        signal, RiskApproval(True, (), 100), NOW, spread=0.0001, pair="EURUSD"
    )
    assert book.equity == pytest.approx(1000 - (0.0001 + 0.00002 + 0.000015) * 100)
    book.checkpoint(NOW)
    restored = PaperPortfolio.restore(journal, book.portfolio_id, NOW)
    assert restored.executor.price_basis == "bid"
    trade = restored.close(position.position_id, 1.1, NOW + timedelta(minutes=5))
    assert trade.pnl == pytest.approx(-(0.0001 + 2 * 0.00002 + 0.00003) * 100)
    assert restored.equity == pytest.approx(1000 + trade.pnl)


def test_bid_short_stops_trigger_on_ask_and_long_targets_on_bid():
    book = PaperPortfolio(PaperExecutionProvider(price_basis="bid"), 1000)
    book.open(SHORT, RiskApproval(True, (), 1), NOW, spread=0.001)
    # BID never reaches 1.11, but ASK does: the short stop must fire.
    closed = book.mark(NOW + timedelta(minutes=5), 1.1095, 1.099, 1.105, opening=1.1)
    assert closed[0].reason == "stop_loss"
    assert closed[0].exit == pytest.approx(1.11)
    book = PaperPortfolio(PaperExecutionProvider(price_basis="bid"), 1000)
    book.open(LONG, RiskApproval(True, (), 1), NOW, spread=0.001)
    assert not book.mark(NOW + timedelta(minutes=5), 1.1295, 1.1, 1.12, opening=1.1)


def test_bid_limit_uses_ask_and_risk_rechecks_actual_market_fill():
    risk = RiskEngine(Settings(trading_enabled=True, max_spread=0.01))
    book = PaperPortfolio(PaperExecutionProvider(price_basis="bid"), 1000)
    order = book.submit(
        replace(LONG, order_type="LIMIT", expires_after_bars=3),
        RiskApproval(True, (), 1),
        NOW,
        "EURUSD",
    )
    bar = Candle(NOW + timedelta(minutes=5), 1.1005, 1.101, 1.0995, 1.1005, 1)
    assert not book.process_orders(bar, "EURUSD", risk, 0.001)
    assert order.status == OrderStatus.PENDING
    next_bar = replace(bar, timestamp=bar.timestamp + timedelta(minutes=5), low=1.098)
    filled = book.process_orders(next_bar, "EURUSD", risk, 0.001)
    assert filled[0].entry == pytest.approx(1.1)

    book = PaperPortfolio(PaperExecutionProvider(price_basis="bid"), 1000)
    order = book.submit(LONG, RiskApproval(True, (), 1), NOW, "EURUSD")
    # Spread worsens reward/risk from 3 to below the independent 1.5 minimum.
    gapped = Candle(bar.timestamp, 1.11, 1.115, 1.105, 1.11, 1)
    assert not book.process_orders(gapped, "EURUSD", risk, 0.001)
    assert order.status == OrderStatus.REJECTED


def test_bid_backtests_require_explicit_matching_assumptions(market_data):
    data = replace(
        market_data(80),
        metadata={"price_basis": "bid", "source": "synthetic_test_fixture"},
    )
    strategy = StrategyRegistry().get("ema_trend")
    with pytest.raises(ValueError, match="midpoint"):
        run_report(data, strategy)
    report = run_report(data, strategy, BacktestCosts(spread=0.0001, price_basis="bid"))
    assert report.assumptions.price_basis == "bid"
    with pytest.raises(ValueError, match="price basis"):
        BacktestCosts(price_basis="ask")


def archive_fixture(tmp_path, *, missing_minute=False):
    archive = tmp_path / "archive"
    path = archive / "raw" / "EURUSD.json"
    path.parent.mkdir(parents=True)
    # A complete requested day is embedded in a larger source month.
    rows = [
        {
            "timestamp": int((NOW + timedelta(minutes=i)).timestamp() * 1000),
            "open": 1.1,
            "high": 1.101,
            "low": 1.099,
            "close": 1.1,
            "volume": 1,
        }
        for i in range(1440)
        if not (missing_minute and i == 100)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    ManifestStore(archive).update_unit(
        "EURUSD",
        "2024-01-01..2024-02-01",
        {
            "status": "downloaded",
            "path": "raw/EURUSD.json",
            "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return archive, path


def test_managed_json_window_validates_range_basis_and_changed_bytes(tmp_path):
    archive, path = archive_fixture(tmp_path)
    root = tmp_path / "out"
    result = prepare_window(archive, root, "EURUSD", date(2024, 1, 2), date(2024, 1, 2))
    assert set(result["timeframes"].values()) == {"PASS"}
    data = load_dataset(root, "EURUSD", "H1")
    assert len(data.candles) == 24
    assert data.metadata["price_basis"] == "bid"
    assert data.metadata["requested_start"] == NOW.isoformat()
    csv = root / "processed/h1/EURUSD.csv"
    csv.write_bytes(csv.read_bytes().replace(b"1.101", b"1.102", 1))
    with pytest.raises(DataValidationError, match="changed"):
        load_dataset(root, "EURUSD", "H1")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="checksum"):
        prepare_window(archive, root, "EURUSD", date(2024, 1, 2), date(2024, 1, 2))


def test_managed_json_window_cannot_hide_missing_minutes_or_range_edges(tmp_path):
    archive, _ = archive_fixture(tmp_path, missing_minute=True)
    root = tmp_path / "out"
    result = prepare_window(archive, root, "EURUSD", date(2024, 1, 2), date(2024, 1, 3))
    assert result["timeframes"]["M1"] != "PASS"
    with pytest.raises(DataValidationError):
        load_dataset(root, "EURUSD", "M5")
    with pytest.raises(DataValidationError, match="unverified"):
        prepare_window(archive, root, "GBPUSD", date(2024, 1, 2), date(2024, 1, 2))
