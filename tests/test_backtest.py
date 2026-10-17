import unittest
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from astra.arena import compete
from astra.backtest import (
    BacktestCosts,
    ExecutionAssumptions,
    cost_sensitivity,
    metrics,
    run,
    run_report,
    walk_forward,
)
from astra.models import Candle, MarketData, Side, Signal
from astra.robustness import evaluate_walk_forward


@dataclass
class AlwaysLong:
    name: str = "test"

    def generate(self, data, context):
        price = data.candles[-1].close
        return Signal(
            self.name, Side.LONG, 0.8, price, price - 1, price + 100, "test", 2
        )


def data(n=40):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = tuple(
        Candle(
            start + timedelta(minutes=5 * i),
            1 + i * 0.01,
            1.01 + i * 0.01,
            0.99 + i * 0.01,
            1 + i * 0.01,
            1,
        )
        for i in range(n)
    )
    return MarketData("EURUSD", "M5", candles)


class BacktestTests(unittest.TestCase):
    def test_invalid_report_and_walk_forward_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial equity"):
            run_report(data(), AlwaysLong(), initial_equity=0)
        with self.assertRaisesRegex(ValueError, "window sizes"):
            walk_forward(data(), 0, 1)
        with self.assertRaisesRegex(ValueError, "positions and warmup"):
            ExecutionAssumptions(max_positions=0)

    def test_simulation_records_strategy_failure_and_cancels_unfilled_order(self):
        class Broken:
            name = "broken"

            def generate(self, *_):
                raise RuntimeError("fixture failure")

        failed = run_report(data(), Broken())
        self.assertTrue(
            any(item["reason"] == "strategy failure" for item in failed.decisions)
        )

        class NeverFilled:
            name = "never-filled"

            def generate(self, market, _context):
                entry = market.candles[-1].close + 1000
                return Signal(
                    self.name,
                    Side.SHORT,
                    0.9,
                    entry,
                    entry + 1,
                    entry - 10,
                    "limit far above market",
                    order_type="LIMIT",
                    expires_after_bars=100,
                )

        from astra.models import RiskApproval

        class ApproveForExecutionPath:
            def approve(self, *_args, **_kwargs):
                return RiskApproval(True, (), 1)

        report = run_report(
            data(), NeverFilled(), risk_engine=ApproveForExecutionPath()
        )
        self.assertTrue(report.orders)
        self.assertEqual(report.orders[-1].status.value, "CANCELLED")

    def test_next_bar_entry_and_costs(self):
        result = run(data(), AlwaysLong(), BacktestCosts(spread=0.02))
        self.assertTrue(result)
        self.assertGreater(result[0].entry, data().candles[30].close)
        baseline = run(data(), AlwaysLong())[0]
        self.assertLess(result[0].pnl_per_unit, baseline.pnl_per_unit)

    def test_metrics_empty_and_profit_factor(self):
        self.assertEqual(metrics([])["trade_count"], 0)
        from astra.backtest import TradeResult

        values = [
            TradeResult("x", Side.LONG, 1, 2, 1, "target"),
            TradeResult("x", Side.LONG, 1, 0.5, -0.5, "stop"),
        ]
        self.assertEqual(metrics(values)["profit_factor"], 2)

    def test_walk_forward_is_chronological_and_disjoint(self):
        folds = walk_forward(data(20), 8, 4)
        self.assertEqual(len(folds), 3)
        self.assertLess(
            folds[0][0].candles[-1].timestamp, folds[0][1].candles[0].timestamp
        )
        self.assertEqual(
            folds[0][1].candles[-1].timestamp + timedelta(minutes=5),
            folds[1][1].candles[0].timestamp,
        )

    def test_higher_costs_do_not_improve_fixed_long_results(self):
        results = cost_sensitivity(
            data(), AlwaysLong(), [BacktestCosts(), BacktestCosts(spread=0.01)]
        )
        self.assertLessEqual(results[1]["expectancy"], results[0]["expectancy"])

    def test_report_tracks_equity_drawdown_and_exposure(self):
        report = run_report(data(), AlwaysLong(), initial_equity=1000)
        self.assertEqual(len(report.equity_curve), len(report.trades) + 1)
        self.assertTrue(all(trade.size >= 0 for trade in report.trades))
        self.assertLessEqual(report.metrics["max_drawdown_pct"], 5)

    def test_research_lookback_bounds_causal_strategy_context(self):
        seen = []

        class ObserveWindow(AlwaysLong):
            def generate(self, market, context):
                seen.append(
                    (
                        market.candles[0].timestamp,
                        market.candles[-1].timestamp,
                        len(market.candles),
                    )
                )
                return super().generate(market, context)

        candles = data(90)
        report = run_report(
            candles,
            ObserveWindow(),
            execution=ExecutionAssumptions(context_lookback_bars=32),
        )
        self.assertTrue(report.decisions)
        self.assertLessEqual(max(count for _, _, count in seen), 32)
        self.assertTrue(all(start <= end for start, end, _ in seen))
        self.assertEqual(report.execution_assumptions["context_lookback_bars"], 32)

    def test_research_context_lookback_must_support_regime_warmup(self):
        with self.assertRaisesRegex(ValueError, "at least 30"):
            ExecutionAssumptions(context_lookback_bars=29)

    def test_arena_compares_all_strategies_same_instrument(self):
        results = compete(data(), [AlwaysLong()])
        self.assertEqual(results[0].pair, "EURUSD")
        self.assertEqual(results[0].timeframe, "M5")
        self.assertIn("insufficient", results[0].evidence)

    def test_costs_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            BacktestCosts(spread=-1)
        with self.assertRaises(ValueError):
            BacktestCosts(value_per_price_unit=0)

    def test_walk_forward_evaluator_reports_separate_oos_windows(self):
        results = evaluate_walk_forward(data(80), AlwaysLong(), 40, 40)
        self.assertEqual(len(results), 1)
        self.assertLess(results[0].train_end, results[0].test_start)
        self.assertIn("expectancy", results[0].out_of_sample_metrics)


if __name__ == "__main__":
    unittest.main()
