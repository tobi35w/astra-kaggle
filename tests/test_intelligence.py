import contextlib
import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from astra.__main__ import main as cli_main
from astra.config import Settings
from astra.database import (
    InMemoryDecisionRepository,
    InMemoryStrategyPerformanceRepository,
    PostgresDecisionRepository,
)
from astra.decision import decide
from astra.execution import PaperExecutionProvider, PaperPortfolio
from astra.exposure import OpenExposure, correlated_exposure, correlation, returns
from astra.health import check_health
from astra.history import load_csv, missing_candles
from astra.intelligence import confidence, detect_regime, select
from astra.learning import evaluate_degradation, record_result
from astra.market import momentum, multi_timeframe, structure, volatility
from astra.models import (
    Candle,
    MarketData,
    Regime,
    RiskApproval,
    RiskState,
    Side,
    Signal,
    StrategyPerformance,
)
from astra.pipeline import AstraPaperPipeline
from astra.risk import RiskEngine
from astra.strategies import Breakout, StrategyRegistry, valid_signal


def bars(n=50, start=None, step=0.001):
    start = start or datetime(2025, 1, 1, tzinfo=UTC)
    values = []
    for i in range(n):
        close = 1 + i * step
        values.append(
            Candle(
                start + timedelta(minutes=5 * i),
                close - 0.0002,
                close + 0.0005,
                close - 0.0005,
                close,
                10,
            )
        )
    return MarketData("EURUSD", "M5", tuple(values), {"timestamp_role": "bar_close"})


class IntelligenceTests(unittest.TestCase):
    def test_regime_requires_evidence_and_classifies_trend(self):
        self.assertEqual(detect_regime(bars(10)).regime, Regime.INSUFFICIENT_DATA)
        self.assertEqual(detect_regime(bars()).regime, Regime.TRENDING)

    def test_market_structure_and_features(self):
        raw = bars(7)
        peak_values = [1.0, 1.2, 1.0, 1.3, 1.0, 1.4, 1.0]
        structure_data = MarketData(
            raw.pair,
            raw.timeframe,
            tuple(
                Candle(c.timestamp, 1.05, peak, 0.95, 1.0, 1)
                for c, peak in zip(raw.candles, peak_values)
            ),
        )
        context = structure(structure_data, left=1, right=1)
        self.assertTrue(context.swings)
        self.assertIn("HH", [label for _, label in context.classifications])
        self.assertGreater(volatility(bars()).atr, 0)
        self.assertEqual(momentum(bars()).direction, Side.LONG)

    def test_breakout_uses_only_prior_range(self):
        from astra.models import MarketSnapshot

        data = bars(21)
        final = data.candles[-1]
        breakout_bar = Candle(final.timestamp, 1.05, 1.31, 1.04, 1.30, 10)
        breakout_data = MarketData(
            data.pair, data.timeframe, data.candles[:-1] + (breakout_bar,)
        )
        context = MarketSnapshot(
            data.pair,
            data.timeframe,
            final.timestamp,
            Side.LONG,
            0.5,
            0.01,
            Regime.RANGING,
            ("test context",),
        )
        signal = Breakout(lookback=20).generate(breakout_data, context)
        self.assertIsNotNone(signal)
        self.assertTrue(valid_signal(signal))

    def test_correlation_exposure_is_conservative(self):
        first, second = returns([1, 2, 3, 4]), returns([2, 4, 6, 8])
        self.assertAlmostEqual(correlation(first, second), 1)
        book = (
            OpenExposure("GBPUSD", Side.LONG, 0.4),
            OpenExposure("USDJPY", Side.SHORT, 0.2),
        )
        self.assertAlmostEqual(
            correlated_exposure("EURUSD", book, {"GBPUSD": 0.9, "USDJPY": 0.2}), 0.4
        )

    def test_mtf_excludes_future_candles_and_reports_conflict(self):
        data = bars()
        at = data.candles[-3].timestamp
        result = multi_timeframe({"M5": data}, at, ("M5", "H1"))
        self.assertEqual(result.snapshots["M5"].timestamp, at)
        self.assertFalse(result.aligned)
        self.assertIn("missing timeframe H1", result.conflicts)

    def test_conflicting_mtf_is_vetoed_by_decision(self):
        sample = bars()
        context = multi_timeframe(
            {"M5": sample}, sample.candles[-1].timestamp, ("M5", "H1")
        )
        decision = decide(
            sample,
            list(StrategyRegistry().all()),
            RiskEngine(Settings()),
            RiskState(1000, 1000, 1000),
            timeframe_context=context,
            decision_time=sample.candles[-1].timestamp,
        )
        self.assertIn("missing timeframe", decision.reason)

    def test_registry_and_selector_can_reject(self):
        registry = StrategyRegistry()
        self.assertEqual(len(registry.all()), 10)
        context = detect_regime(bars())
        result = select(bars(), context, list(registry.all()), min_fitness=2)
        self.assertFalse(result.candidates)

    def test_confidence_conflict_lowers_score_and_labels_it(self):
        plain = confidence(0.8, 0.8, 0.8, 0.8, 0.8)
        conflict = confidence(0.8, 0.8, 0.8, 0.8, 0.8, ("H1 conflicts with M5",))
        self.assertLess(conflict.overall, plain.overall)
        self.assertEqual(conflict.evidence_level, "conflicting evidence")


class SystemTests(unittest.TestCase):
    def test_end_to_end_pipeline_only_opens_paper_position_after_risk_approval(self):
        data = bars()
        portfolio = PaperPortfolio(PaperExecutionProvider(), 10_000)
        audit = InMemoryDecisionRepository()
        pipeline = AstraPaperPipeline(
            [StrategyRegistry().get("ema_trend")],
            RiskEngine(Settings(trading_enabled=True)),
            portfolio,
            audit,
        )
        result = pipeline.evaluate(
            data,
            RiskState(10_000, 10_000, 10_000),
            decision_time=data.candles[-1].timestamp,
        )
        self.assertEqual(result.decision.action.value, "TRADE")
        self.assertIsNotNone(result.paper_position)
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(audit.all(), (result.decision,))

    def test_postgres_repository_uses_parameterized_insert(self):
        cursor = MagicMock()
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor
        sample = bars()
        decision = decide(
            sample,
            [],
            RiskEngine(Settings()),
            RiskState(1000, 1000, 1000),
            decision_time=sample.candles[-1].timestamp,
        )
        with patch("psycopg.connect", return_value=connection) as connect:
            PostgresDecisionRepository("postgresql://example.invalid/db").append(
                decision
            )
        connect.assert_called_once_with("postgresql://example.invalid/db")
        self.assertIn("VALUES (%s,%s,%s", cursor.execute.call_args.args[0])
        self.assertEqual(cursor.execute.call_args.args[1][3], "NO_TRADE")

    def test_cli_runs_csv_backtest_and_emits_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bars.csv"
            rows = ["timestamp,open,high,low,close,volume"]
            for i in range(45):
                price = 1.1 + i * 0.001
                stamp = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i)
                rows.append(
                    f"{stamp.isoformat()},{price - 0.0005},{price + 0.0005},{price - 0.001},{price},10"
                )
            path.write_text("\n".join(rows), encoding="utf-8")
            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "astra",
                        "backtest",
                        str(path),
                        "--pair",
                        "EURUSD",
                        "--timeframe",
                        "M15",
                    ],
                ),
                contextlib.redirect_stdout(output),
            ):
                cli_main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["pair"], "EURUSD")
            self.assertEqual(result["strategy"], "ema_trend")

    def test_health_stale_and_critical_failure(self):
        old = bars()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        status = check_health(old, now=now, max_age_seconds=30, risk_engine_ok=False)
        self.assertFalse(status.healthy)
        self.assertTrue(any("stale" in reason for reason in status.reasons))

    def test_csv_import_and_gap_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bars.csv"
            path.write_text(
                "timestamp,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,1.1,.9,1,2\n2025-01-01T00:15:00Z,1,1.1,.9,1,2\n",
                encoding="utf-8",
            )
            imported = load_csv(path, "EURUSD", "M5")
            self.assertEqual(len(missing_candles(imported)), 1)
            self.assertEqual(imported.metadata["timestamp_role"], "bar_close")

    def test_performance_memory_is_append_only_and_detects_degradation(self):
        repository = InMemoryStrategyPerformanceRepository()
        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(20):
            record_result(
                repository,
                StrategyPerformance(
                    "s",
                    "TRENDING",
                    "EURUSD",
                    "M5",
                    5,
                    1.0 if i < 10 else -1.0,
                    0.1,
                    base + timedelta(days=i),
                ),
            )
        history = repository.history("s")
        result = evaluate_degradation(history)
        self.assertTrue(result.degraded)
        self.assertEqual(len(history), 20)

    def test_explicit_runtime_health_failure_is_a_veto(self):
        sample = bars()
        bad_health = check_health(sample, now=datetime.now(UTC), database_ok=False)
        decision = decide(
            sample,
            [],
            RiskEngine(Settings()),
            RiskState(1000, 1000, 1000),
            health=bad_health,
        )
        self.assertEqual(decision.action.value, "NO_TRADE")
        self.assertIn("database unavailable", decision.reason)

    def test_decision_audit_repository_records_veto(self):
        audit = InMemoryDecisionRepository()
        decision = decide(
            bars(),
            [],
            RiskEngine(Settings()),
            RiskState(1000, 1000, 1000),
            repository=audit,
        )
        self.assertEqual(audit.all(), (decision,))

    def test_paper_position_closes_on_stop_and_updates_equity(self):
        signal = Signal("s", Side.LONG, 0.8, 1.0, 0.9, 1.2, "test", 2)
        portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
        portfolio.open(signal, RiskApproval(True, ("approved",), 2), datetime.now(UTC))
        done = portfolio.mark(datetime.now(UTC), high=1.1, low=0.89, close=0.95)
        self.assertEqual(done[0].reason, "stop_loss")
        self.assertLess(portfolio.equity, 1000)
        self.assertFalse(portfolio.positions)


if __name__ == "__main__":
    unittest.main()
