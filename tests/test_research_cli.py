import json
import sys
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import pytest

from astra.__main__ import main
from astra.arena import evaluate_strategy
from astra.backtest import BacktestCosts
from astra.config import Settings
from astra.data import DataValidationError, validate
from astra.health import check_health
from astra.history import load_csv
from astra.learning import evaluate_degradation
from astra.ml import Example, chronological_split, load_model, monitor, train
from astra.models import Regime, StrategyPerformance
from astra.pipeline import replay_paper
from astra.repositories import Entity
from astra.robustness import (
    cost_grid,
    evaluate_holdout,
    evaluate_walk_forward,
    parameter_sensitivity,
    robustness_report,
)
from astra.strategies import EmaTrend


@pytest.fixture
def csv_file(tmp_path, market_data):
    path = tmp_path / "explicit_synthetic_test_fixture.csv"
    data = market_data(70, oscillate=True)
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "\n".join(
            f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
            for c in data.candles
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "command",
    [
        "validate",
        "analyze",
        "backtest",
        "walk-forward",
        "evaluate",
        "paper",
        "robustness",
    ],
)
def test_cli_dataset_commands_emit_json(command, csv_file, capsys):
    args = [
        "astra",
        command,
        str(csv_file),
        "--pair",
        "EURUSD",
        "--timeframe",
        "M5",
        "--train-size",
        "40",
        "--test-size",
        "20",
    ]
    with patch.object(sys, "argv", args):
        main()
    assert json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("command", ["config", "health", "fitness"])
def test_cli_inspection_commands_do_not_require_credentials(command, capsys):
    with patch.object(sys, "argv", ["astra", command]):
        main()
    assert json.loads(capsys.readouterr().out)


def test_main_routes_historical_sandbox_and_verification(monkeypatch, capsys):
    import sys

    from astra import database_verification, sandbox
    from astra.historical import cli as historical_cli

    monkeypatch.setattr(sys, "argv", ["astra", "historical", "validate"])
    monkeypatch.setattr(historical_cli, "historical_main", lambda args: 7)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 7

    served = []
    monkeypatch.setattr(
        sys, "argv", ["astra", "sandbox", "--port", "0", "--no-browser"]
    )
    monkeypatch.setattr(sandbox, "serve", lambda *args: served.append(args))
    main()
    assert served == [(0, ".sandbox", False, "/")]

    monkeypatch.setattr(sys, "argv", ["astra", "sandbox", "--research", "--no-browser"])
    main()
    assert served[-1] == (8787, ".sandbox", False, "/research")

    monkeypatch.setattr(sys, "argv", ["astra", "verify-database"])
    monkeypatch.setattr(
        database_verification, "verify_database", lambda _: {"status": "VERIFIED"}
    )
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "VERIFIED"

    monkeypatch.setattr(
        database_verification, "verify_database", lambda _: {"status": "FAILED"}
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


def test_main_closes_created_repository(monkeypatch, capsys):
    import sys

    class Repository:
        closed = False

        def query(self, *_args, **_kwargs):
            return ()

        def close(self):
            self.closed = True

    repository = Repository()
    monkeypatch.setattr("astra.adapters.repository_from_settings", lambda _: repository)
    monkeypatch.setattr(sys, "argv", ["astra", "fitness"])
    main()
    assert repository.closed
    assert json.loads(capsys.readouterr().out)["records"] == []


def test_dispatch_rejects_conflicting_or_missing_dataset_sources():
    from astra.__main__ import _dispatch, _parser
    from astra.config import Settings
    from astra.repositories import Journal, MemoryRepository

    journal = Journal(MemoryRepository())
    with pytest.raises(ValueError, match="either a CSV or a managed dataset"):
        _dispatch(
            _parser().parse_args(
                [
                    "backtest",
                    "fixture.csv",
                    "--dataset-root",
                    "root",
                    "--pair",
                    "EURUSD",
                    "--timeframe",
                    "M5",
                ]
            ),
            Settings(),
            journal,
        )
    with pytest.raises(ValueError, match="CSV or --dataset-root"):
        _dispatch(
            _parser().parse_args(["backtest", "--pair", "EURUSD", "--timeframe", "M5"]),
            Settings(),
            journal,
        )


def test_analyze_passes_causal_multitimeframe_context(monkeypatch, market_data):
    from astra.__main__ import _dispatch, _parser
    from astra.config import Settings
    from astra.repositories import Journal, MemoryRepository

    base = market_data(40, timeframe="M5")
    higher = market_data(40, timeframe="M15")
    context = object()
    monkeypatch.setattr(
        "astra.history.load_csv",
        lambda _path, _pair, timeframe, **_kwargs: (
            higher if timeframe == "M15" else base
        ),
    )
    monkeypatch.setattr("astra.market.multi_timeframe", lambda *_a, **_k: context)
    monkeypatch.setattr(
        "astra.decision.decide", lambda *_a, **kwargs: kwargs["timeframe_context"]
    )
    monkeypatch.setattr("astra.memory.PerformanceMemory.history", lambda *_a: ())
    args = _parser().parse_args(
        [
            "analyze",
            "base.csv",
            "--pair",
            "EURUSD",
            "--timeframe",
            "M5",
            "--m15",
            "higher.csv",
        ]
    )
    assert _dispatch(args, Settings(), Journal(MemoryRepository())) is context


def test_module_entrypoint_calls_main(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["astra", "config"])
    runpy.run_module("astra.__main__", run_name="__main__")
    assert "persistence_backend" in json.loads(capsys.readouterr().out)


def test_cli_defaults_and_safe_error(csv_file, capsys):
    with patch.object(sys, "argv", ["astra"]):
        main()
    assert "paper mode" in capsys.readouterr().out
    with (
        patch.object(
            sys,
            "argv",
            [
                "astra",
                "backtest",
                str(csv_file),
                "--pair",
                "EURUSD",
                "--timeframe",
                "M5",
                "--spread",
                "nan",
            ],
        ),
        pytest.raises(SystemExit) as error,
    ):
        main()
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_csv_validation_zero_price_duplicates_timezone_cadence_and_gaps(tmp_path):
    path = tmp_path / "fixture.csv"

    def write(rows):
        path.write_text(
            "timestamp,open,high,low,close,volume\n" + rows, encoding="utf-8"
        )

    for rows in (
        "2025-01-01T00:00:00Z,0,1,0,1,1\n2025-01-01T00:05:00Z,1,1.1,.9,1,1",
        "2025-01-01T00:00:00Z,1,1.1,.9,1,1\n2025-01-01T00:00:00Z,1,1.1,.9,1,1",
        "2025-01-01T00:00:00,1,1.1,.9,1,1\n2025-01-01T00:05:00Z,1,1.1,.9,1,1",
        "2025-01-01T00:00:00Z,1,1.1,.9,1,1\n2025-01-01T00:06:00Z,1,1.1,.9,1,1",
    ):
        write(rows)
        with pytest.raises(DataValidationError):
            load_csv(path, "EURUSD", "M5")
    write("2025-01-01T01:00:00+01:00,1,1.1,.9,1,1\n2025-01-01T00:10:00Z,1,1.1,.9,1,1")
    assert load_csv(path, "EURUSD", "M5").candles[0].timestamp.hour == 0
    with pytest.raises(DataValidationError, match="missing"):
        load_csv(path, "EURUSD", "M5", reject_gaps=True)
    with pytest.raises(DataValidationError):
        load_csv(path, "INVALID", "M5")
    # M1 is now the canonical historical-data timeframe; gaps remain explicit.
    assert load_csv(path, "EURUSD", "M1").timeframe == "M1"


def test_health_covers_all_critical_dependencies(market_data):
    data = market_data()
    status = check_health(
        data,
        now=data.candles[-1].timestamp,
        database_ok=False,
        risk_engine_ok=False,
        execution_ok=False,
        strategy_count=0,
        configuration_valid=False,
        persistence_ok=False,
        model_required=True,
    )
    assert not status.healthy and len(status.reasons) == 7
    assert not check_health(None).healthy
    assert not check_health(
        data, now=data.candles[-1].timestamp.replace(tzinfo=None)
    ).healthy


def test_robustness_and_fitting_only_see_training_windows(market_data, journal):
    data = market_data(120, oscillate=True)
    calls = []

    def fit(training):
        calls.append(training.candles[-1].timestamp)
        return EmaTrend()

    folds = evaluate_walk_forward(data, EmaTrend(), 60, 30, fit=fit, journal=journal)
    assert len(folds) == 2
    assert all(at == fold.train_end < fold.test_start for at, fold in zip(calls, folds))
    holdout = evaluate_holdout(data, EmaTrend())
    assert holdout.train.candles[-1].timestamp < holdout.test.candles[0].timestamp
    result = robustness_report(
        data,
        EmaTrend(),
        parameter_factory=lambda x: EmaTrend(stop_atr=x),
        parameters=(1.0, 1.5, 2.0),
        journal=journal,
    )
    assert result.state == "INSUFFICIENT_DATA"
    assert len(result.parameter_results) == 3
    assert len(cost_grid(data, EmaTrend(), (0, 0.0001), (0, 0.0001), (0, 0.0001))) == 8
    assert (
        len(parameter_sensitivity(data, lambda x: EmaTrend(stop_atr=x), (1.0, 1.5)))
        == 2
    )
    assert journal.repository.query(Entity.WALK_FORWARD)
    assert journal.repository.query(Entity.ROBUSTNESS)


def test_arena_persists_context_fitness_and_does_not_double_size(market_data, journal):
    from astra.backtest import run_report

    data = market_data(100)
    report = run_report(data, EmaTrend())
    evaluation = evaluate_strategy(data, EmaTrend(), journal=journal)
    assert evaluation.metrics["net_pnl"] == pytest.approx(
        sum(t.pnl for t in report.trades)
    )
    assert evaluation.regime_expectancy[Regime.TRENDING.value] == pytest.approx(
        report.metrics["expectancy"]
    )
    assert journal.repository.query(Entity.EVALUATION)
    assert journal.repository.query(Entity.FITNESS)
    records = journal.repository.query(Entity.PERFORMANCE)
    assert records
    assert all(row.payload["source"] == "BACKTEST" for row in records)
    assert all(row.payload["metric_unit"] == "R" for row in records)
    assert all(row.payload["outcome"] is not None for row in records)


def test_arena_scales_fitness_by_trade_evidence(market_data, monkeypatch):
    from types import SimpleNamespace

    from astra.robustness import RobustnessReport

    data = market_data(40)
    monkeypatch.setattr(
        "astra.arena.run_report",
        lambda *_args, **_kwargs: SimpleNamespace(
            trades=(),
            metrics={
                "expectancy": 1.0,
                "average_loss": -1.0,
                "max_drawdown_pct": 0.0,
                "risk_adjusted": 1.0,
                "consistency": 1.0,
                "trade_count": 1.0,
                "net_pnl": 1.0,
            },
        ),
    )
    monkeypatch.setattr(
        "astra.robustness.robustness_report",
        lambda *_args, **_kwargs: RobustnessReport(
            "ROBUST",
            (),
            {"net_pnl": 1.0},
            ({"net_pnl": 0.8}, {"net_pnl": 0.5}, {"net_pnl": 0.4}),
            {},
            0.0,
        ),
    )
    one_trade = evaluate_strategy(data, EmaTrend())
    assert one_trade.evidence == "insufficient trades"
    assert one_trade.fitness == pytest.approx(1 / 30)
    assert evaluate_strategy(data, EmaTrend()).fitness == one_trade.fitness


def test_performance_memory_uses_one_source_and_experiment_per_context(market_data):
    from astra.evidence import comparable_evidence

    at = market_data(2).candles[-1].timestamp
    records = (
        StrategyPerformance(
            "trend",
            "TRENDING",
            "EURUSD",
            "M5",
            2,
            1.0,
            0.2,
            at,
            source="BACKTEST",
            metric_unit="R",
            drawdown_unit="R",
            experiment_id="backtest-1",
            outcome=1.0,
        ),
        StrategyPerformance(
            "trend",
            "TRENDING",
            "EURUSD",
            "M5",
            3,
            0.5,
            0.3,
            at,
            source="OOS",
            metric_unit="R",
            drawdown_unit="R",
            experiment_id="fold-2",
            outcome=0.5,
        ),
        StrategyPerformance(
            "trend",
            "TRENDING",
            "EURUSD",
            "M5",
            4,
            -0.1,
            2.0,
            at,
            source="PAPER",
            metric_unit="account_currency",
            drawdown_unit="account_currency",
            experiment_id="paper-1",
            outcome=-0.1,
        ),
    )

    chosen = comparable_evidence(records)
    assert len(chosen) == 1
    assert chosen[0].source == "PAPER"
    assert chosen[0].metric_unit == "account_currency"
    assert chosen[0].experiment_id == "paper-1"


def test_readiness_report_labels_local_evidence_and_unverified_gates(tmp_path):
    from astra.readiness import readiness_report
    from astra.repositories import MemoryRepository

    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "experiment_id": "fixture-only",
                "completed_at": "2026-01-01T00:00:00+00:00",
                "results": [
                    {
                        "backtest": {"trade_count": 4, "net_pnl": 10},
                        "walk_forward": [{"out_of_sample_metrics": {"trade_count": 2}}],
                        "robustness": {
                            "state": "INSUFFICIENT_DATA",
                            "out_of_sample_metrics": {"trade_count": 3, "net_pnl": -2},
                            "stressed_metrics": [{"net_pnl": -3}],
                            "cost_sensitivity": 0.4,
                            "generalization_gap": 0.5,
                        },
                        "paper": {
                            "mode": "paper replay",
                            "trades": 2,
                            "orders": 2,
                            "open_positions": 0,
                            "approvals_enabled": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = readiness_report(
        Settings(), MemoryRepository(), campaign_path=str(campaign)
    )
    assert report["overall"] == "NOT_READY_FOR_MT5_DEMO_INTEGRATION"
    assert report["checks"]["risk_safety"]["status"] == "PASS — TESTED LOCALLY"
    assert report["checks"]["research_campaign"]["status"].startswith("PASS")
    assert report["checks"]["historical_backtest"]["metrics"]["trade_count"] == 4
    assert report["checks"]["chronological_holdout"]["metrics"]["net_pnl"] == -2
    assert report["checks"]["walk_forward"]["out_of_sample_trades"] == 2
    assert report["checks"]["cost_stress"]["scenarios"] == 1
    assert report["checks"]["robustness"]["state"] == "INSUFFICIENT_DATA"
    assert report["checks"]["paper_replay"]["trades"] == 2
    assert report["checks"]["hosted_schema_and_round_trip"]["status"] == "NOT RUN"
    assert report["profitability"] == "UNPROVEN"


def test_readiness_uses_saved_hosted_verification_without_network_calls(tmp_path):
    from dataclasses import replace

    from astra.readiness import readiness_report

    class RemoteRepository:
        def healthy(self):
            raise AssertionError("readiness must not contact remote persistence")

    database_report = tmp_path / "database.json"
    database_report.write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "schema": "VERIFIED",
                "journal_round_trip": "VERIFIED",
                "checked_at": "2026-10-11T18:43:27+00:00",
                "audit_record_id": "fixture-audit-id",
                "migrations_applied": False,
            }
        ),
        encoding="utf-8",
    )
    settings = replace(Settings(), persistence_backend="supabase")
    report = readiness_report(
        settings,
        RemoteRepository(),
        database_report_path=str(database_report),
    )
    assert report["checks"]["hosted_schema_and_round_trip"]["status"] == (
        "PASS — VERIFIED"
    )
    assert report["checks"]["journal_liveness"]["status"] == "PASS — VERIFIED"
    assert report["checks"]["historical_scope"]["status"] == "BLOCKED"
    assert report["checks"]["forward_paper_testing"]["status"] == "NOT RUN"
    assert report["overall"] == "NOT_READY_FOR_MT5_DEMO_INTEGRATION"


def test_degradation_disables_multiple_failure_modes(market_data):
    at = market_data().candles[0].timestamp
    records = tuple(
        StrategyPerformance(
            "trend",
            "TRENDING",
            "EURUSD",
            "M5",
            5,
            1 if i < 10 else -1,
            0.01 if i < 10 else 0.2,
            at + timedelta(days=i),
            win_rate=0.8 if i < 10 else 0.2,
            cost_sensitivity=0.1 if i < 10 else 0.8,
        )
        for i in range(20)
    )
    assert evaluate_degradation(records).severity == "DISABLED"
    assert not evaluate_degradation(records, as_of=at).degraded


def test_paper_replay_persists_audit_and_stays_disabled_by_default(
    market_data, journal
):
    data = market_data(50)
    result = replay_paper(
        data, [EmaTrend()], Settings(), BacktestCosts(), journal=journal
    )
    assert not result["positions"] and not result["trades"]
    assert len(journal.repository.query(Entity.DECISION)) == 50
    enabled = replay_paper(
        data,
        [EmaTrend()],
        Settings(trading_enabled=True),
        BacktestCosts(),
        journal=journal,
    )
    assert enabled["orders"]
    assert journal.repository.query(Entity.ORDER)
    assert journal.repository.query(Entity.EXECUTION)


def test_ml_gates_training_chronology_versioning_and_monitoring(market_data, journal):
    data = market_data(360, oscillate=True)
    assert train(data).status == "DISABLED"
    assert train(data, enabled=True).status == "DISABLED"
    # This explicit provenance flag is used only to exercise training on a synthetic test fixture.
    verified_fixture = replace(
        data, metadata={"real_data": True, "purpose": "test fixture only"}
    )
    assert (
        train(
            replace(verified_fixture, candles=verified_fixture.candles[:50]),
            enabled=True,
        ).status
        == "INSUFFICIENT_DATA"
    )
    result = train(
        verified_fixture,
        enabled=True,
        journal=journal,
        trained_at=data.candles[-1].timestamp,
    )
    assert result.status == "TRAINED"
    model = result.model
    assert model.training_end < model.validation_start <= model.trained_at
    assert not load_model(
        journal,
        pair="EURUSD",
        timeframe="M5",
        as_of=model.trained_at - timedelta(seconds=1),
    )
    restored = load_model(
        journal, pair="EURUSD", timeframe="M5", as_of=model.trained_at
    )
    values = (0.001, 0.002, 0.003, 0.2)
    assert restored.predict(values, model.trained_at) == model.predict(
        values, model.trained_at
    )
    with pytest.raises(ValueError, match="did not exist"):
        model.predict(values, model.training_end)
    with pytest.raises(ValueError, match="features"):
        model.predict((float("nan"),), model.trained_at)
    examples = (
        Example(model.trained_at, model.trained_at + timedelta(minutes=5), values, 1),
    )
    assert monitor(model, examples, journal)["sample_count"] == 1
    assert monitor(model, ())["status"] == "INSUFFICIENT_DATA"
    assert journal.repository.query(Entity.CALIBRATION)


def test_ml_single_class_and_insufficient_splits(market_data):
    data = market_data(360)
    data = replace(data, metadata={"real_data": True, "purpose": "test fixture only"})
    assert train(data, enabled=True).status == "INSUFFICIENT_DATA"
    with pytest.raises(ValueError):
        chronological_split(())
    with pytest.raises(ValueError):
        validate(data, min_candles=0)


def test_generalization_gap_penalizes_deterioration_independently_of_currency_scale():
    from astra.robustness import generalization_gap

    training = {"expectancy": 8, "average_win": 10, "average_loss": -10}
    holdout = {"expectancy": 1, "average_win": 10, "average_loss": -10}
    assert generalization_gap(training, holdout) == pytest.approx(0.7)
    assert generalization_gap(
        training, {k: v * 100 for k, v in holdout.items()}
    ) == pytest.approx(0.7)
    assert generalization_gap(holdout, training) == 0
