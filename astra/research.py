"""Reproducible offline campaigns over managed, validated historical windows."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from .backtest import BacktestCosts, run_report
from .decision import decide
from .historical.dataset import load_dataset
from .market import multi_timeframe
from .models import RiskState
from .repositories import Entity, Journal, json_value
from .risk import RiskEngine
from .robustness import evaluate_walk_forward, robustness_report
from .strategies import StrategyRegistry


def campaign(
    root,
    symbols,
    timeframe,
    costs,
    *,
    journal,
    settings,
    strategies=("ema_trend",),
    train_size=700,
    test_size=300,
    progress=None,
):
    """Completion means the experiment ran, never a claim of economic readiness."""
    if not symbols or not strategies or len(set(symbols)) != len(symbols):
        raise ValueError("a campaign requires unique symbols and strategies")
    registry = StrategyRegistry()
    selected = [registry.get(name) for name in strategies]
    datasets = {pair: load_dataset(root, pair, timeframe) for pair in symbols}
    if any(len(data.candles) < train_size + test_size for data in datasets.values()):
        raise ValueError("insufficient candles for a complete walk-forward fold")
    results = []
    for pair, data in datasets.items():
        assumptions = costs[pair]
        at = data.candles[-1].timestamp
        frames = {tf: load_dataset(root, pair, tf) for tf in ("M5", "M15", "H1", "H4")}
        context = multi_timeframe(frames, at, execution_timeframe=timeframe)
        analysis = decide(
            data,
            selected,
            RiskEngine(settings),
            RiskState(10000, 10000, 10000),
            assumptions.spread,
            decision_time=at,
            timeframe_context=context,
            expected_slippage=assumptions.slippage,
            journal=journal,
        )
        from .pipeline import replay_paper

        paper = replay_paper(data, selected, settings, assumptions, journal=journal)
        paper_summary = {
            key: paper[key] for key in ("mode", "portfolio_id", "equity", "balance")
        }
        paper_summary.update(
            trades=len(paper["trades"]),
            orders=len(paper["orders"]),
            open_positions=len(paper["positions"]),
            approvals_enabled=settings.trading_enabled,
        )
        for strategy in selected:
            if progress:
                progress(f"Evaluating {pair} {timeframe} {strategy.name}")
            backtest = run_report(data, strategy, assumptions, journal=journal)
            folds = evaluate_walk_forward(
                data, strategy, train_size, test_size, assumptions, journal=journal
            )
            robustness = robustness_report(data, strategy, assumptions, journal=journal)
            results.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy.name,
                    "dataset": data.metadata,
                    "first_close": data.candles[0].timestamp,
                    "last_close": at,
                    "candles": len(data.candles),
                    "analysis": analysis,
                    "paper": paper_summary,
                    "assumptions": assumptions,
                    "backtest": backtest.metrics,
                    "walk_forward": folds,
                    "robustness": robustness,
                }
            )
    request = {
        "datasets": {p: d.metadata["dataset_version"] for p, d in datasets.items()},
        "timeframe": timeframe,
        "strategies": strategies,
        "costs": {p: asdict(c) for p, c in costs.items()},
        "train_size": train_size,
        "test_size": test_size,
        "settings": settings.public(),
    }
    report = {
        "status": "COMPLETED",
        "completed_at": datetime.now(UTC),
        "experiment_id": sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest(),
        "request": request,
        "results": results,
        "limitations": [
            "Fixed assumed spreads, not historical ASK quotes.",
            "Selected research windows do not complete a continuous archive for the four active pairs.",
            "Results describe this experiment; no profitability or live-readiness claim.",
        ],
    }
    record = journal.record(Entity.AUDIT, report["completed_at"], report)
    report["journal_record_id"] = record.record_id
    return report


def research_main(argv):
    import argparse
    import sys

    from .adapters import SQLiteRepository
    from .config import Settings
    from .history import PAIRS

    parser = argparse.ArgumentParser(prog="astra research")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", nargs="+", required=True, choices=PAIRS)
    parser.add_argument("--timeframe", default="H4", choices=("M5", "M15", "H1", "H4"))
    parser.add_argument("--strategies", nargs="+", default=["ema_trend"])
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--spread-pips", type=float, required=True)
    parser.add_argument("--slippage-pips", type=float, required=True)
    parser.add_argument("--commission", type=float, default=0)
    parser.add_argument("--price-value", type=float, default=1)
    parser.add_argument(
        "--paper-enabled",
        action="store_true",
        help="allow risk-gated orders in this offline replay",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    repo = SQLiteRepository(str(output / "journal.sqlite3"))
    try:
        result = campaign(
            args.dataset_root,
            args.symbols,
            args.timeframe,
            {
                pair: BacktestCosts(
                    args.spread_pips * (0.01 if pair.endswith("JPY") else 0.0001),
                    args.slippage_pips * (0.01 if pair.endswith("JPY") else 0.0001),
                    args.commission,
                    args.price_value,
                    "bid",
                )
                for pair in args.symbols
            },
            journal=Journal(repo),
            settings=replace(
                Settings(),
                trading_enabled=args.paper_enabled,
                max_spread=args.spread_pips * 0.01,
                max_slippage=args.slippage_pips * 0.01,
            ),
            strategies=tuple(args.strategies),
            train_size=args.train_size,
            test_size=args.test_size,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        path = output / f"campaign-{result['experiment_id'][:16]}.json"
        path.write_text(
            json.dumps(json_value(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "report": str(path),
                    "experiments": len(result["results"]),
                }
            )
        )
    finally:
        repo.close()
    return 0
