"""Fail-safe CLI for supplied historical data and disconnected paper execution."""

import argparse
import json
from datetime import UTC, datetime

from . import __version__
from .config import Settings
from .repositories import Entity, Journal, json_value


def _parser():
    from .history import PAIRS, TIMEFRAMES

    parser = argparse.ArgumentParser(
        prog="astra", description="Astra research and paper-only tools"
    )
    sub = parser.add_subparsers(dest="command")
    for command in (
        "validate",
        "analyze",
        "backtest",
        "walk-forward",
        "evaluate",
        "paper",
        "robustness",
    ):
        cmd = sub.add_parser(command)
        cmd.add_argument("csv", nargs="?")
        cmd.add_argument(
            "--dataset-root", help="load a checksum-verified managed historical dataset"
        )
        cmd.add_argument("--pair", required=True, choices=PAIRS)
        cmd.add_argument("--timeframe", required=True, choices=tuple(TIMEFRAMES))
        cmd.add_argument("--strategy", default="ema_trend")
        cmd.add_argument("--spread", type=float, default=0.0)
        cmd.add_argument("--slippage", type=float, default=0.0)
        cmd.add_argument("--commission", type=float, default=0.0)
        cmd.add_argument("--price-value", type=float, default=1.0)
        cmd.add_argument(
            "--price-basis",
            choices=("mid", "bid"),
            default="mid",
            help="BID uses a fixed assumed spread to reconstruct ASK quotes",
        )
        cmd.add_argument("--equity", type=float, default=10000.0)
        cmd.add_argument("--max-positions", type=int, default=1)
        cmd.add_argument("--train-size", type=int, default=120)
        cmd.add_argument("--test-size", type=int, default=60)
        cmd.add_argument("--reject-gaps", action="store_true")
        for tf in TIMEFRAMES:
            cmd.add_argument(
                "--" + tf.lower(),
                help=f"optional {tf} CSV for causal multi-timeframe context",
            )
    sub.add_parser("config")
    sub.add_parser("health")
    sub.add_parser(
        "verify-database",
        help="verify hosted schema and an idempotent Journal audit write",
    )
    sandbox = sub.add_parser("sandbox", help="open the local research dashboard")
    sandbox.add_argument("--port", type=int, default=8787)
    sandbox.add_argument("--workspace", default=".sandbox")
    sandbox.add_argument("--no-browser", action="store_true")
    sandbox.add_argument(
        "--research",
        action="store_true",
        help="open the historical research results page",
    )
    desk = sub.add_parser(
        "desk", help="open the live evaluation desk (tests, research, acquisition)"
    )
    desk.add_argument("--port", type=int, default=8789)
    desk.add_argument("--no-browser", action="store_true")
    fitness = sub.add_parser("fitness")
    fitness.add_argument("--pair", default="")
    fitness.add_argument("--timeframe", default="")
    fitness.add_argument("--strategy", default="")
    readiness = sub.add_parser(
        "readiness", help="report verified pre-MT5 readiness evidence"
    )
    readiness.add_argument("--dataset-root")
    readiness.add_argument("--pair", default="EURUSD", choices=PAIRS)
    readiness.add_argument("--timeframe", default="H4", choices=tuple(TIMEFRAMES))
    readiness.add_argument("--campaign-report")
    readiness.add_argument("--database-report")
    return parser


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "research":
        from .research import research_main

        raise SystemExit(research_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        from .historical.cli import historical_main

        raise SystemExit(historical_main(sys.argv[2:]))
    parser = _parser()
    args = parser.parse_args()
    if args.command == "sandbox":
        from .sandbox import serve

        serve(
            args.port,
            args.workspace,
            not args.no_browser,
            "/research" if args.research else "/",
        )
        return
    if args.command == "desk":
        from .progress_desk import serve as serve_desk

        serve_desk(port=args.port, open_browser=not args.no_browser)
        return
    repository = None
    try:
        settings = Settings.from_env()
        settings.validate()
        if not args.command:
            print(
                f"Astra {__version__}: research/paper mode; trading_enabled={settings.trading_enabled}"
            )
            return
        if args.command == "config":
            output = settings.public()
        elif args.command == "verify-database":
            from .database_verification import verify_database

            output = verify_database(settings)
        else:
            from .adapters import repository_from_settings

            repository = repository_from_settings(settings)
            journal = Journal(repository)
            output = _dispatch(args, settings, journal)
        print(json.dumps(json_value(output), indent=2, allow_nan=False))
        if args.command == "verify-database" and output["status"] != "VERIFIED":
            raise SystemExit(1)
        if (
            args.command == "readiness"
            and output["overall"] != "READY_FOR_MT5_DEMO_INTEGRATION"
        ):
            raise SystemExit(1)
        if args.command == "validate" and not output["valid"]:
            raise SystemExit(1)
    except Exception as error:  # noqa: BLE001 - never expose adapter credentials or tracebacks.
        from .data import DataValidationError

        quality_failure = isinstance(error, DataValidationError)
        print(
            json.dumps(
                {
                    "status": "DATA QUALITY INSUFFICIENT"
                    if quality_failure
                    else "ERROR",
                    "error_type": type(error).__name__,
                    "message": "Data validation rejected this operation; inspect the dataset quality report."
                    if quality_failure
                    else "Command failed safely; check inputs, configuration, dependencies and persistence.",
                }
            )
        )
        raise SystemExit(2) from None
    finally:
        if repository is not None and hasattr(repository, "close"):
            repository.close()


def _dispatch(args, settings, journal):
    from .backtest import BacktestCosts, ExecutionAssumptions, run_report
    from .history import TIMEFRAMES, load_csv, missing_candles
    from .strategies import StrategyRegistry

    registry = StrategyRegistry()
    if args.command == "readiness":
        from .readiness import readiness_report

        return readiness_report(
            settings,
            journal.repository,
            dataset_root=args.dataset_root,
            pair=args.pair,
            timeframe=args.timeframe,
            campaign_path=args.campaign_report,
            database_report_path=args.database_report,
        )
    if args.command == "health":
        ok = journal.healthy()
        model_ok = not settings.ml_enabled or bool(
            journal.repository.query(Entity.MODEL, as_of=datetime.now(UTC))
        )
        output = {
            "healthy": ok and model_ok,
            "checks": {
                "database_and_persistence": ok,
                "configuration": True,
                "strategies": len(registry.all()),
                "risk_engine": True,
                "execution": "paper only",
                "data": "not supplied",
                "model": "disabled"
                if not settings.ml_enabled
                else "available"
                if model_ok
                else "required model unavailable",
            },
            "decision": "NO_TRADE: no decision-time data supplied",
        }
        if ok:
            journal.record(Entity.HEALTH, datetime.now(UTC), output)
        return output
    if args.command == "fitness":
        records = journal.repository.query(
            Entity.FITNESS,
            pair=args.pair,
            timeframe=args.timeframe,
            strategy=args.strategy,
            as_of=datetime.now(UTC),
        )
        return {
            "records": records,
            "evidence": "No empirical performance evaluation has been performed."
            if not records
            else "stored research evidence",
        }
    if args.dataset_root:
        from .historical.dataset import load_dataset

        if args.csv:
            raise ValueError("supply either a CSV or a managed dataset root")
        data = load_dataset(args.dataset_root, args.pair, args.timeframe)
    else:
        if not args.csv:
            raise ValueError("a CSV or --dataset-root is required")
        data = load_csv(
            args.csv, args.pair, args.timeframe, reject_gaps=args.reject_gaps
        )
    if args.command == "validate":
        from .historical.validation import inspect_candles

        quality = inspect_candles(data.candles, data.pair, data.timeframe)
        return {
            "valid": quality.status == "PASS",
            "status": "PASS"
            if quality.status == "PASS"
            else "DATA QUALITY INSUFFICIENT",
            "quality": quality,
            "pair": data.pair,
            "timeframe": data.timeframe,
            "candles": len(data.candles),
            "gaps": missing_candles(data),
            "timestamp_role": "bar_close",
            "note": "Gaps require calendar review; imported prices have not been independently authenticated.",
        }
    from .historical.dataset import require_quality

    require_quality(data)
    costs = BacktestCosts(
        args.spread,
        args.slippage,
        args.commission,
        args.price_value,
        getattr(args, "price_basis", "mid"),
    )
    strategy = registry.get(args.strategy)
    if args.command == "backtest":
        report = run_report(
            data,
            strategy,
            costs,
            args.equity,
            execution=ExecutionAssumptions(max_positions=args.max_positions),
            journal=journal,
        )
        return {
            "strategy": strategy.name,
            "pair": data.pair,
            "timeframe": data.timeframe,
            "trade_count": len(report.trades),
            "metrics": report.metrics,
            "assumptions": costs,
            "execution": report.execution_assumptions,
            "orders": report.orders,
            "equity_curve": report.marked_equity_curve,
        }
    if args.command == "walk-forward":
        from .robustness import evaluate_walk_forward

        folds = evaluate_walk_forward(
            data, strategy, args.train_size, args.test_size, costs, journal=journal
        )
        return {"folds": folds, "status": "EVALUATED" if folds else "INSUFFICIENT_DATA"}
    if args.command == "robustness":
        from .robustness import robustness_report

        return robustness_report(data, strategy, costs, journal=journal)
    if args.command == "evaluate":
        from .arena import compete

        return {
            "evaluations": compete(data, list(registry.all()), costs, journal=journal)
        }
    if args.command == "paper":
        from .pipeline import replay_paper

        return replay_paper(
            data, list(registry.all()), settings, costs, args.equity, journal=journal
        )
    from .decision import decide
    from .market import multi_timeframe
    from .memory import PerformanceMemory
    from .models import RiskState
    from .risk import RiskEngine

    at = data.candles[-1].timestamp
    frames = {
        tf: load_csv(getattr(args, tf.lower()), data.pair, tf)
        for tf in TIMEFRAMES
        if getattr(args, tf.lower())
    }
    context = None
    if frames:
        frames[data.timeframe] = data
        context = multi_timeframe(frames, at, execution_timeframe=data.timeframe)
    return decide(
        data,
        list(registry.all()),
        RiskEngine(settings),
        RiskState(args.equity, args.equity, args.equity),
        args.spread,
        expected_slippage=args.slippage,
        value_per_price_unit=args.price_value,
        performance=PerformanceMemory(journal).history(data.pair, data.timeframe, at),
        decision_time=at,
        timeframe_context=context,
        journal=journal,
    )


if __name__ == "__main__":
    main()
