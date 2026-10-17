from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from astra.backtest import (
    BacktestCosts,
    ExecutionAssumptions,
    TradeResult,
    metrics,
    run_report,
)
from astra.config import Settings
from astra.execution import (
    OrderStatus,
    PaperExecutionProvider,
    PaperPortfolio,
    exit_trigger,
    order_price,
)
from astra.models import Candle, RiskApproval, RiskState, Side, Signal
from astra.risk import RiskEngine

NOW = datetime(2025, 1, 1, tzinfo=UTC)
SIGNAL = Signal("test", Side.LONG, 0.8, 1.1, 1.0, 1.3, "fixture")


class AlwaysLong:
    name = "test"

    def generate(self, data, context):
        price = data.candles[-1].close
        return replace(
            SIGNAL, entry=price, stop_loss=price - 0.002, take_profit=price + 0.004
        )


def test_paper_execution_rejects_invalid_costs_timestamps_and_geometry():
    executor = PaperExecutionProvider()
    with pytest.raises(ValueError, match="malformed strategy signal"):
        executor.execute(replace(SIGNAL, side=Side.FLAT), RiskApproval(True, (), 1))
    with pytest.raises(ValueError, match="spread must be finite"):
        executor.execute(SIGNAL, RiskApproval(True, (), 1), spread=-1)
    with pytest.raises(ValueError, match="execution costs"):
        PaperExecutionProvider(slippage=-0.1)
    with pytest.raises(ValueError, match="price value conversion"):
        PaperExecutionProvider(value_per_price_unit=0)

    portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
    with pytest.raises(ValueError, match="timezone-aware"):
        portfolio.open(
            SIGNAL,
            RiskApproval(True, (), 1),
            datetime(2025, 1, 1),  # noqa: DTZ001
        )
    with pytest.raises(ValueError, match="execution price invalidates"):
        PaperPortfolio(PaperExecutionProvider(slippage=0.2), 1000).open(
            SIGNAL, RiskApproval(True, (), 1), NOW
        )
    with pytest.raises(ValueError, match="initial equity"):
        PaperPortfolio(PaperExecutionProvider(), float("nan"))


def test_paper_executor_validates_stop_geometry_at_risk_checked_fill_price():
    signal = replace(
        SIGNAL,
        entry=1.0,
        stop_loss=1.00001,
        take_profit=1.1,
        order_type="LIMIT",
    )
    fill = PaperExecutionProvider(slippage=0.00002).execute(
        signal, RiskApproval(True, (), 1)
    )
    assert fill.fill_price == pytest.approx(1.00002)


def test_paper_portfolio_error_paths_and_pending_entry_mark():
    portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
    portfolio.advance_day(NOW)
    with pytest.raises(ValueError, match="clock cannot move backwards"):
        portfolio.advance_day(NOW - timedelta(days=1))
    with pytest.raises(PermissionError, match="risk engine vetoed paper order"):
        portfolio.submit(SIGNAL, RiskApproval(False, ("veto",), 0), NOW)
    with pytest.raises(ValueError, match="invalid paper close"):
        portfolio.open(SIGNAL, RiskApproval(True, (), 1), NOW)
        portfolio.close(1, 0, NOW + timedelta(minutes=1))
    with pytest.raises(ValueError, match="invalid OHLC mark"):
        portfolio.mark(NOW + timedelta(minutes=1), 1.0, 1.1, 1.05)

    pending = PaperPortfolio(PaperExecutionProvider(), 1000)
    limit = replace(SIGNAL, order_type="LIMIT", entry=1.05)
    position = pending.open(limit, RiskApproval(True, (), 1), NOW)
    assert not pending.mark(
        NOW, 1.4, 1.01, 1.2, include_new=True
    )  # A target touch on the entry candle has no assumed intrabar ordering.
    assert position.position_id in pending.positions
    with pytest.raises(ValueError, match="checkpoint requires persistence"):
        pending.checkpoint(NOW)

    later = PaperPortfolio(PaperExecutionProvider(), 1000)
    position = later.open(
        SIGNAL, RiskApproval(True, (), 1), NOW + timedelta(hours=1), pair="GBPUSD"
    )
    assert not later.mark(NOW, 1.2, 1.05, 1.1, pair="GBPUSD")
    assert position.position_id in later.positions


@pytest.mark.parametrize(
    "state,settings,fragment",
    [
        (RiskState(9500, 10000, 10000), {}, "daily loss"),
        (RiskState(7999, 8000, 10000), {}, "drawdown"),
        (RiskState(10000, 10000, 10000, consecutive_losses=5), {}, "consecutive-loss"),
        (RiskState(10000, 10000, 10000, open_positions=3), {}, "open positions"),
        (RiskState(10000, 10000, 10000, open_exposure=1), {}, "exposure"),
        (RiskState(10000, 10000, 10000, pair_exposure=1), {}, "concentration"),
        (RiskState(10000, 10000, 10000, correlated_exposure=2), {}, "correlated"),
        (RiskState(10000, 10000, 10000), {"kill_switch": True}, "kill switch"),
        (RiskState(10000, 10000, 10000), {"emergency_halt": True}, "emergency"),
        (RiskState(10000, 10000, 10000, open_positions=float("nan")), {}, "non-finite"),
        (RiskState(10000, 10000, 10000, open_risk=-1), {}, "negative"),
        (RiskState(10000, 10000, 0), {}, "equity"),
    ],
)
def test_risk_veto_paths(state, settings, fragment):
    result = RiskEngine(Settings(trading_enabled=True, **settings)).approve(
        SIGNAL, state
    )
    assert not result.approved and any(fragment in r for r in result.reasons)


def test_risk_sizes_against_remaining_daily_budget_including_reserved_risk_and_costs():
    risk = RiskEngine(
        Settings(
            trading_enabled=True,
            max_position_size=1e6,
            max_pair_exposure=1e6,
            max_correlated_exposure=1e6,
        )
    )
    state = RiskState(9510, 10000, 10000, open_risk=4)
    approval = risk.approve(SIGNAL, state, execution_cost_per_unit=0.02)
    assert approval.approved
    assert approval.position_size * (0.1 + 0.02) == pytest.approx(6)
    assert not risk.approve(SIGNAL, replace(state, open_risk=11)).approved
    dishonest = replace(SIGNAL, take_profit=1.11, risk_reward=1000)
    assert not risk.approve(dishonest, state).approved


def test_cost_slippage_and_invalid_config_veto():
    state = RiskState(1000, 1000, 1000)
    risk = RiskEngine(Settings(trading_enabled=True))
    assert not risk.approve(SIGNAL, state, spread=0.01).approved
    assert not risk.approve(SIGNAL, state, expected_slippage=0.01).approved
    assert not risk.approve(SIGNAL, state, value_per_price_unit=0).approved
    assert not risk.approve(SIGNAL, state, data_valid=False).approved
    assert (
        not RiskEngine(Settings(max_spread=float("nan")))
        .approve(SIGNAL, state)
        .approved
    )


def test_marked_equity_fees_spread_and_pair_isolation(journal):
    book = PaperPortfolio(
        PaperExecutionProvider(slippage=0.001, commission_per_unit=0.02), 1000, journal
    )
    first = book.open(
        SIGNAL, RiskApproval(True, (), 2), NOW, spread=0.002, pair="EURUSD"
    )
    second = book.open(SIGNAL, RiskApproval(True, (), 1), NOW, pair="GBPUSD")
    with pytest.raises(ValueError, match="pair"):
        book.mark(NOW + timedelta(minutes=5), 1.25, 1.05, 1.2)
    book.mark(NOW + timedelta(minutes=5), 1.25, 1.05, 1.2, pair="EURUSD")
    assert book.positions[second.position_id].last_price == SIGNAL.entry
    assert book.equity > book.balance
    before = book.balance
    trade = book.close(first.position_id, 1.2, NOW + timedelta(minutes=6))
    assert trade.pnl == pytest.approx((1.198 - 1.102) * 2 - 0.04)
    assert book.balance - before == pytest.approx(trade.pnl + 0.02)
    with pytest.raises(ValueError):
        book.mark(NOW, 1.2, 1.0, 1.1, pair="EURUSD")


def test_same_bar_stop_priority_gaps_and_no_optimistic_target_improvement():
    assert exit_trigger(SIGNAL, 1.4, 0.8, 0.9) == (0.9, "stop_loss")
    assert exit_trigger(SIGNAL, 1.5, 1.35, 1.4) == (1.3, "take_profit")
    short = replace(SIGNAL, side=Side.SHORT, stop_loss=1.2, take_profit=0.9)
    assert exit_trigger(short, 1.4, 0.8, 1.3) == (1.3, "stop_loss")
    assert exit_trigger(short, 0.85, 0.7, 0.8) == (0.9, "take_profit")


def test_orders_wait_expire_cancel_and_recheck_risk(journal):
    book = PaperPortfolio(PaperExecutionProvider(), 1000, journal)
    risk = RiskEngine(Settings(trading_enabled=True))
    approval = risk.approve(SIGNAL, book.risk_state("EURUSD"))
    pending = book.submit(SIGNAL, approval, NOW, "EURUSD")
    assert not book.process_orders(Candle(NOW, 1.1, 1.2, 1.05, 1.1), "EURUSD", risk)
    filled = book.process_orders(
        Candle(NOW + timedelta(minutes=5), 1.1, 1.2, 1.05, 1.1), "EURUSD", risk
    )
    assert filled and pending.status == OrderStatus.FILLED
    limit = book.submit(
        replace(SIGNAL, order_type="LIMIT", entry=1.01), approval, NOW, "EURUSD"
    )
    book.process_orders(
        Candle(NOW + timedelta(minutes=10), 1.1, 1.2, 1.05, 1.1), "EURUSD", risk
    )
    assert limit.status == OrderStatus.EXPIRED
    cancel = book.submit(SIGNAL, approval, NOW, "EURUSD")
    book.cancel(cancel.order_id, NOW + timedelta(minutes=1))
    assert cancel.status == OrderStatus.CANCELLED
    with pytest.raises(ValueError):
        book.cancel(cancel.order_id, NOW)
    veto = book.submit(SIGNAL, approval, NOW, "EURUSD")
    killed = RiskEngine(Settings(trading_enabled=True, kill_switch=True))
    book.process_orders(
        Candle(NOW + timedelta(minutes=15), 1.1, 1.2, 1.05, 1.1), "EURUSD", killed
    )
    assert veto.status == OrderStatus.REJECTED


@pytest.mark.parametrize(
    "kind,side,entry,expected",
    [
        ("LIMIT", Side.LONG, 1.05, 1.05),
        ("LIMIT", Side.SHORT, 1.15, 1.15),
        ("STOP", Side.LONG, 1.15, 1.15),
        ("STOP", Side.SHORT, 1.05, 1.05),
        ("LIMIT", Side.LONG, 0.9, None),
        ("STOP", Side.SHORT, 0.9, None),
    ],
)
def test_pending_order_trigger_prices(kind, side, entry, expected):
    assert (
        order_price(
            replace(SIGNAL, order_type=kind, side=side, entry=entry),
            Candle(NOW, 1.1, 1.2, 1.0, 1.1),
        )
        == expected
    )


def test_gap_invalidates_pending_trade_geometry():
    book = PaperPortfolio(PaperExecutionProvider(), 1000)
    risk = RiskEngine(Settings(trading_enabled=True))
    order = book.submit(SIGNAL, risk.approve(SIGNAL, book.risk_state()), NOW, "EURUSD")
    book.process_orders(
        Candle(NOW + timedelta(minutes=5), 1.4, 1.5, 1.35, 1.45), "EURUSD", risk
    )
    assert order.status == OrderStatus.REJECTED and not book.positions


def test_backtest_orders_risk_veto_marked_curve_and_pnl_units(market_data, journal):
    data = market_data(100)
    report = run_report(
        data,
        AlwaysLong(),
        BacktestCosts(0.0001, 0.00002, 0.0001),
        journal=journal,
        execution=ExecutionAssumptions(max_positions=2),
    )
    assert report.trades and report.orders
    assert report.trades[0].entry_index == 31
    assert len(report.marked_equity_curve) > len(report.equity_curve)
    assert sum(t.pnl for t in report.trades) == pytest.approx(report.metrics["net_pnl"])
    assert report.equity_curve[-1] - report.equity_curve[0] == pytest.approx(
        report.metrics["net_pnl"]
    )
    denied = run_report(
        data,
        AlwaysLong(),
        risk_engine=RiskEngine(Settings(trading_enabled=True, kill_switch=True)),
    )
    assert not denied.trades and all(
        o.status == OrderStatus.REJECTED for o in denied.orders
    )
    first = TradeResult("s", Side.LONG, 1, 2, 2, "target", size=3)
    assert metrics([first])["net_pnl"] == 6


def test_backtest_future_mutation_does_not_change_earlier_orders(market_data):
    data = market_data(100, oscillate=True)
    changed = replace(
        data,
        candles=data.candles[:70]
        + tuple(
            replace(
                c, open=c.open + 1, high=c.high + 1, low=c.low + 1, close=c.close + 1
            )
            for c in data.candles[70:]
        ),
    )
    original = run_report(data, AlwaysLong())
    poisoned = run_report(changed, AlwaysLong())
    assert [d for d in original.decisions if d["index"] < 70] == [
        d for d in poisoned.decisions if d["index"] < 70
    ]
    assert [t for t in original.trades if t.exit_index < 70] == [
        t for t in poisoned.trades if t.exit_index < 70
    ]


def test_paper_checkpoint_restores_risk_and_orders_without_reexecution(journal):
    book = PaperPortfolio(
        PaperExecutionProvider(commission_per_unit=0.01), 1000, journal
    )
    risk = RiskEngine(Settings(trading_enabled=True))
    approval = risk.approve(SIGNAL, book.risk_state("EURUSD"))
    first = book.open(
        SIGNAL, approval, NOW, pair="EURUSD", timeframe="M5", regime="TRENDING"
    )
    book.close(first.position_id, 1.2, NOW + timedelta(minutes=1))
    book.open(
        SIGNAL, approval, NOW + timedelta(minutes=2), pair="EURUSD", timeframe="M5"
    )
    book.submit(
        replace(SIGNAL, order_type="LIMIT", entry=1.01, expires_after_bars=3),
        approval,
        NOW + timedelta(minutes=2),
        "EURUSD",
    )
    candle = Candle(NOW + timedelta(minutes=5), 1.1, 1.15, 1.05, 1.12)
    book.process_orders(candle, "EURUSD", risk)
    book.process_orders(candle, "EURUSD", risk)
    assert next(iter(book.orders.values())).bars_waited == 1
    book.mark(candle.timestamp, candle.high, candle.low, candle.close, pair="EURUSD")
    book.checkpoint(candle.timestamp)
    restored = PaperPortfolio.restore(journal, book.portfolio_id, candle.timestamp)
    assert restored.equity == book.equity
    assert restored.risk_state("EURUSD") == book.risk_state("EURUSD")
    assert (
        restored.positions == book.positions
        and restored.orders == book.orders
        and restored.closed == book.closed
    )
    with pytest.raises(ValueError):
        PaperPortfolio.restore(journal, book.portfolio_id, NOW)
