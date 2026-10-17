import math
import unittest
from datetime import UTC, datetime, timedelta

from astra.config import Settings
from astra.data import DataValidationError, validate
from astra.decision import decide
from astra.execution import PaperExecutionProvider
from astra.indicators import adx, atr, bollinger_bands, ema, roc, rsi, sma
from astra.market import swing_points
from astra.models import (
    Action,
    Candle,
    MarketData,
    RiskApproval,
    RiskState,
    Side,
    Signal,
)
from astra.risk import RiskEngine
from astra.strategies import EmaTrend


def series(count=45, step=0.001):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = []
    for i in range(count):
        close = 1.1 + i * step
        candles.append(
            Candle(
                start + timedelta(minutes=15 * i),
                close - step / 2,
                close + step / 2,
                close - step,
                close,
                10,
            )
        )
    return MarketData("EURUSD", "M15", tuple(candles))


class DataTests(unittest.TestCase):
    def test_valid_series(self):
        validate(series())

    def test_rejects_naive_timestamp(self):
        data = series()
        bad = Candle(datetime(2025, 1, 1), 1, 1.1, 0.9, 1, 1)  # noqa: DTZ001
        with self.assertRaises(DataValidationError):
            validate(MarketData("EURUSD", "M15", (bad, data.candles[1])))

    def test_rejects_bad_ohlcv_and_order(self):
        data = series()
        bad = Candle(data.candles[0].timestamp, 2, 1, 0, 1, -1)
        with self.assertRaises(DataValidationError):
            validate(MarketData("EURUSD", "M15", (bad, data.candles[1])))
        with self.assertRaises(DataValidationError):
            validate(MarketData("EURUSD", "M15", tuple(reversed(data.candles))))

    def test_rejects_insufficient_and_non_finite(self):
        with self.assertRaises(DataValidationError):
            validate(MarketData("EURUSD", "M15", ()))
        d = series()
        c = d.candles[0]
        with self.assertRaises(DataValidationError):
            validate(
                MarketData(
                    d.pair,
                    d.timeframe,
                    (Candle(c.timestamp, math.nan, 1, 0, 0.5), d.candles[1]),
                )
            )


class IndicatorTests(unittest.TestCase):
    def test_sma_ema_and_insufficient(self):
        self.assertEqual(sma([1, 2, 3], 2), [None, 1.5, 2.5])
        self.assertEqual(ema([1, 2, 3], 3), [None, None, 2])
        self.assertEqual(ema([1, 2], 3), [None, None])

    def test_rsi_and_atr_edges(self):
        values = rsi([1, 2, 3, 4], 2)
        self.assertEqual(values[:2], [None, None])
        self.assertEqual(values[-1], 100)
        self.assertEqual(atr([3, 4], [1, 2], [2, 3], 2)[-1], 2)

    def test_invalid_period(self):
        with self.assertRaises(ValueError):
            sma([1], 0)

    def test_bands_and_rate_of_change(self):
        low, middle, high = bollinger_bands([1, 2, 3], 2)
        self.assertIsNone(middle[0])
        self.assertLess(low[-1], middle[-1])
        self.assertGreater(high[-1], middle[-1])
        self.assertAlmostEqual(roc([100, 110], 1)[-1], 10)

    def test_adx_warms_up_before_strength_is_available(self):
        highs = [1 + i * 0.01 for i in range(40)]
        lows = [value - 0.005 for value in highs]
        closes = [value - 0.002 for value in highs]
        values = adx(highs, lows, closes, 5)
        self.assertTrue(all(value is None for value in values[:9]))
        self.assertGreater(values[-1], 90)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.signal = Signal("test", Side.LONG, 0.8, 1.1, 1.09, 1.12, "test", 2)
        self.state = RiskState(10000, 10000, 10000)

    def test_daily_limit_halt_and_confidence_veto(self):
        risk = RiskEngine(Settings(trading_enabled=True, max_spread=0.0002))
        self.assertFalse(
            risk.approve(self.signal, RiskState(9500, 10000, 10000), 0.0001).approved
        )
        self.assertFalse(
            risk.approve(
                self.signal, RiskState(10000, 10000, 10000, halted=True), 0.0001
            ).approved
        )
        weak = Signal("weak", Side.LONG, 0.2, 1.1, 1.09, 1.12, "weak", 2)
        self.assertFalse(risk.approve(weak, self.state, 0.0001).approved)

    def test_daily_loss_ceiling_cannot_be_configured_above_five_percent(self):
        risk = RiskEngine(
            Settings(trading_enabled=True, max_spread=0.0002, max_daily_loss_pct=0.25)
        )
        under = risk.approve(self.signal, RiskState(9510, 10000, 10000), 0.0001)
        at_limit = risk.approve(self.signal, RiskState(9500, 10000, 10000), 0.0001)
        self.assertTrue(under.approved)
        self.assertFalse(at_limit.approved)

    def test_exposure_and_invalid_signal_are_vetoed(self):
        risk = RiskEngine(
            Settings(trading_enabled=True, max_spread=0.0002, max_position_size=2)
        )
        exposure = RiskState(10000, 10000, 10000, open_exposure=2)
        self.assertFalse(risk.approve(self.signal, exposure, 0.0001).approved)
        malformed = Signal(
            "bad", Side.LONG, float("nan"), 1.1, 1.09, 1.12, "invalid", 3
        )
        self.assertFalse(risk.approve(malformed, self.state, 0.0001).approved)

    def test_risk_size_and_paper_execution(self):
        risk = RiskEngine(
            Settings(
                trading_enabled=True,
                max_spread=0.0002,
                max_trade_risk_pct=0.00001,
                max_position_size=100,
                max_pair_exposure=100,
                max_correlated_exposure=100,
            )
        )
        approval = risk.approve(self.signal, self.state, 0.0001)
        self.assertTrue(approval.approved)
        self.assertAlmostEqual(approval.position_size, 10)
        with self.assertRaises(PermissionError):
            PaperExecutionProvider().execute(
                self.signal, RiskApproval(False, ("veto",))
            )
        fill = PaperExecutionProvider(slippage=0.00002).execute(
            self.signal, approval, 0.0001
        )
        self.assertGreater(fill.fill_price, self.signal.entry)
        converted = risk.approve(
            self.signal, self.state, 0.0001, value_per_price_unit=2
        )
        self.assertAlmostEqual(converted.position_size, approval.position_size / 2)

    def test_decision_veto_and_no_trade(self):
        risk = RiskEngine(Settings())
        decision = decide(series(), [EmaTrend()], risk, self.state)
        self.assertIn(decision.action, (Action.TRADE, Action.NO_TRADE, Action.WAIT))
        self.assertNotEqual(decision.action, Action.TRADE)
        self.assertEqual(
            decide(MarketData("EURUSD", "M15", ()), [], risk, self.state).action,
            Action.NO_TRADE,
        )


class MarketTests(unittest.TestCase):
    def test_swings_only_confirmed_after_right_bars(self):
        self.assertEqual(swing_points([1, 2, 5, 2, 1], 1, 1), [(3, "high", 5)])
        self.assertEqual(swing_points([1, 2, 5], 1, 1), [])


if __name__ == "__main__":
    unittest.main()
