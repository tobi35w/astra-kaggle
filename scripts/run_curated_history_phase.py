"""Run a resumable, offline ML and strategy holdout campaign on curated windows.

Each checksum-approved contiguous window stays independent. The final 30% is a
chronological paper holdout; strategy settings are frozen, and every run is
repeated with doubled spread and slippage. This follow-up focuses on H1/H4,
where the prior evaluation identified the EMA concern. It is separate from the
Yllvar Training Arena campaign and publishes no selected/production model.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, uuid5

from astra.adapters import SQLiteRepository
from astra.backtest import BacktestCosts, ExecutionAssumptions, run_report
from astra.historical.dataset import load_dataset, require_quality
from astra.ml import Model, train
from astra.models import MarketData
from astra.repositories import Entity, Journal, Record, json_value
from astra.strategies import StrategyRegistry
from scripts.run_training_arena_phase import LearnedDirection, model_from_payload

ROOT = Path(__file__).resolve().parents[1]
CURATED_ROOT = ROOT / "data" / "astra-training-four-pairs"
CURATION_PATH = CURATED_ROOT / "curation.json"
OUTPUT_ROOT = ROOT / "data" / "training-arena" / "curated-history-h1-h4-v1"
REPORT_PATH = OUTPUT_ROOT / "evaluation-report.json"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
RESULTS_PATH = OUTPUT_ROOT / "results.jsonl"
MODELS_PATH = OUTPUT_ROOT / "models.jsonl"
TRADES_PATH = OUTPUT_ROOT / "trades.jsonl"
JOURNAL_PATH = ROOT / "data" / "training-arena" / "arena-journal.sqlite3"
CAMPAIGN_ID = "astra-curated-four-pair-windows-h1-h4-v1"
TIMEFRAMES = ("H1", "H4")
STARTING_USD = 10_000.0
RISK_FRACTION = 0.005
TRAIN_FRACTION = 0.70
MINIMUM_MODEL_SAMPLES = 300
EXECUTION = ExecutionAssumptions(
    max_positions=1, warmup_bars=30, context_lookback_bars=128
)
COST_MULTIPLIERS = {"base": 1.0, "double_cost": 2.0}


def now() -> datetime:
    return datetime.now(UTC)


def experiment_id(
    pair: str, timeframe: str, window_id: str, scenario: str, strategy: str
) -> str:
    return f"{scenario}:{pair}:{timeframe}:{window_id}:{strategy}"


def record_id(kind: str, identity: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{CAMPAIGN_ID}:{kind}:{identity}"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(json_value(value), separators=(",", ":"), allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def costs_for(pair: str, multiplier: float) -> BacktestCosts:
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    return BacktestCosts(
        spread=pip * multiplier,
        slippage=pip * 0.2 * multiplier,
        commission_per_unit=0.0,
        value_per_price_unit=1.0,
        price_basis="bid",
    )


def starting_equity(pair: str, data: MarketData, split: int) -> float:
    if pair.endswith("JPY"):
        # USDJPY quote-currency P&L is JPY; mark the USD-equivalent deposit at
        # the last known close before the holdout begins.
        return STARTING_USD * data.candles[split - 1].close
    return STARTING_USD


def load_window(root: Path, pair: str, timeframe: str) -> MarketData:
    data = load_dataset(root, pair, timeframe)
    require_quality(data)
    return replace(
        data,
        metadata=data.metadata
        | {
            "real_data": True,
            "campaign_id": CAMPAIGN_ID,
            "window_id": root.name,
            "price_authentication": "Dukascopy-derived source; local checksums and structural checks do not independently authenticate prices",
        },
    )


def save_model(
    journal: Journal,
    model: Model,
    identity: str,
    pair: str,
    timeframe: str,
    dataset_version: str,
) -> None:
    payload = {
        "source": "CURATED_FOUR_PAIR_HISTORICAL",
        "campaign_id": CAMPAIGN_ID,
        "window_id": identity,
        "dataset_version": dataset_version,
        "model": json_value(model),
        "warning": "research artifact only; not registered as Astra's active inference model",
    }
    journal.record(
        Entity.AUDIT,
        model.trained_at,
        payload,
        pair=pair,
        timeframe=timeframe,
        record_id=record_id("model-audit", identity),
    )


def persist_run(journal: Journal, result: dict, simulation) -> None:
    identity = result["id"]
    pair, timeframe, strategy = result["pair"], result["timeframe"], result["strategy"]
    stamp = result["test_end"]
    dataset_version = result["dataset_version"]
    summary = {
        key: result[key]
        for key in (
            "source",
            "campaign_id",
            "window_id",
            "experiment_id",
            "dataset_version",
            "evidence",
            "cost_scenario",
            "test_start",
            "test_end",
            "native_currency",
            "starting_balance_native",
            "starting_balance_usd_equivalent",
            "net_pnl_native",
            "net_return_pct",
            "trade_count",
            "metrics",
            "assumptions",
        )
    }
    records = [
        Record(
            Entity.BACKTEST,
            stamp,
            summary,
            pair=pair,
            timeframe=timeframe,
            strategy=strategy,
            record_id=record_id("backtest", identity),
        ),
        Record(
            Entity.METRICS,
            stamp,
            summary,
            pair=pair,
            timeframe=timeframe,
            strategy=strategy,
            record_id=record_id("metrics", identity),
        ),
    ]
    for sequence, trade in enumerate(simulation.trades):
        at = trade.closed_at or trade.opened_at or stamp
        trade_id = f"{identity}:{sequence}:{at.isoformat()}"
        records.append(
            Record(
                Entity.TRADE,
                at,
                {
                    "source": "CURATED_FOUR_PAIR_HISTORICAL",
                    "campaign_id": CAMPAIGN_ID,
                    "window_id": result["window_id"],
                    "experiment_id": identity,
                    "dataset_version": dataset_version,
                    "trade_sequence": sequence,
                    "native_currency": result["native_currency"],
                    "trade": json_value(trade),
                    "pnl": trade.pnl,
                },
                pair=pair,
                timeframe=timeframe,
                strategy=strategy,
                correlation_id=identity,
                record_id=record_id("trade", trade_id),
            )
        )
    journal.repository.append_batch(tuple(records))


def campaign_windows() -> list[tuple[str, str, Path]]:
    curation = json.loads(CURATION_PATH.read_text(encoding="utf-8"))
    if curation.get("status") not in {"COMPLETE", "COMPLETE_WITH_EXCLUSIONS"}:
        raise ValueError("curated four-pair manifest is not marked PASS")
    windows = []
    for pair, details in curation["pairs"].items():
        for item in details["segments"]:
            # The curation manifest retains rejected candidates for audit. Only
            # explicitly accepted, five-frame PASS windows enter this campaign.
            if item.get("status") != "PASS":
                continue
            if len(item.get("loadable_timeframes", [])) != 5:
                raise ValueError(
                    f"accepted window is missing a timeframe: {pair}/{item.get('id')}"
                )
            windows.append((pair, item["id"], CURATED_ROOT / pair / item["id"]))
    if not windows:
        raise ValueError("curation manifest contains no accepted windows")
    return windows


def result_row(
    data: MarketData,
    window_id: str,
    scenario: str,
    strategy: str,
    model_info: dict | None,
    simulation,
    split: int,
    initial_equity: float,
) -> dict:
    pnl = float(simulation.metrics["net_pnl"])
    return {
        "id": experiment_id(data.pair, data.timeframe, window_id, scenario, strategy),
        "experiment_id": experiment_id(
            data.pair, data.timeframe, window_id, scenario, strategy
        ),
        "campaign_id": CAMPAIGN_ID,
        "window_id": window_id,
        "pair": data.pair,
        "timeframe": data.timeframe,
        "strategy": strategy,
        "source": "Dukascopy-curated four-pair local windows",
        "source_type": "MODEL_PAPER_OOS"
        if strategy == "ml_direction"
        else "STRATEGY_PAPER_OOS",
        "evidence": "CHRONOLOGICAL_HOLDOUT_PAPER",
        "cost_scenario": scenario,
        "dataset_version": data.metadata["dataset_version"],
        "canonical_m1_sha256": data.metadata.get("canonical_m1_sha256"),
        "test_start": data.candles[split].timestamp,
        "test_end": data.candles[-1].timestamp,
        "native_currency": "JPY" if data.pair.endswith("JPY") else "USD",
        "starting_balance_native": initial_equity,
        "ending_equity_native": initial_equity + pnl,
        "starting_balance_usd_equivalent": STARTING_USD,
        "start_fx_usdjpy": data.candles[split - 1].close
        if data.pair.endswith("JPY")
        else None,
        "net_pnl_native": pnl,
        "net_return_pct": pnl / initial_equity * 100,
        "trade_count": len(simulation.trades),
        "metrics": json_value(simulation.metrics),
        "model": model_info,
        "assumptions": {
            "spread_pips": 1.0 if scenario == "base" else 2.0,
            "slippage_pips": 0.2 if scenario == "base" else 0.4,
            "commission_per_unit": 0.0,
            "price_basis": "BID",
            "risk_per_trade_pct": RISK_FRACTION * 100,
            "max_open_positions": EXECUTION.max_positions,
            "warmup_bars": EXECUTION.warmup_bars,
            "leverage": "not modeled",
            "account_currency": "JPY" if data.pair.endswith("JPY") else "USD",
        },
    }


def write_progress(status: str, completed: int, total: int, **values) -> None:
    write_json(
        PROGRESS_PATH,
        {
            "campaign_id": CAMPAIGN_ID,
            "status": status,
            "completed_tests": completed,
            "total_tests": total,
            "updated_at": now(),
            **values,
        },
    )


def run_campaign() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    windows = campaign_windows()
    strategies = StrategyRegistry().all()
    # One ML model is fitted per window/frame and tested only on its final 30%.
    total = (
        len(windows) * len(TIMEFRAMES) * (len(strategies) + 1) * len(COST_MULTIPLIERS)
    )
    existing_results = read_jsonl(RESULTS_PATH)
    completed = {row["id"] for row in existing_results}
    existing_models = {row["id"]: row for row in read_jsonl(MODELS_PATH)}
    repository = SQLiteRepository(str(JOURNAL_PATH))
    journal = Journal(repository)
    failures = []
    try:
        journal.record(
            Entity.AUDIT,
            now(),
            {
                "event": "curated_four_pair_training_evaluation_started",
                "campaign_id": CAMPAIGN_ID,
                "window_count": len(windows),
                "pair_count": len({pair for pair, _window, _path in windows}),
                "timeframes": TIMEFRAMES,
                "strategies": [strategy.name for strategy in strategies],
                "holdout_fraction": 1 - TRAIN_FRACTION,
                "cost_scenarios": COST_MULTIPLIERS,
                "run_count_upper_bound": total,
                "assumptions": "BID OHLC, 0.5% risk per trade, one open position, no leverage; USDJPY accounts are JPY denominated.",
            },
            record_id=record_id("campaign-start", CAMPAIGN_ID),
        )
        for pair, window_id, window_root in windows:
            for timeframe in TIMEFRAMES:
                dataset_key = f"{pair}:{timeframe}:{window_id}"
                write_progress(
                    "RUNNING",
                    len(completed),
                    total,
                    current_pair=pair,
                    current_timeframe=timeframe,
                    current_window=window_id,
                    current_dataset=dataset_key,
                )
                data = load_window(window_root, pair, timeframe)
                split = int(len(data.candles) * TRAIN_FRACTION)
                if split < 2 or len(data.candles) - split < 100:
                    raise ValueError(
                        f"chronological holdout is too short: {dataset_key}"
                    )
                train_data = MarketData(
                    pair, timeframe, data.candles[:split], data.metadata
                )
                model_info = existing_models.get(dataset_key)
                model = None
                if model_info is None:
                    trained = train(
                        train_data,
                        enabled=True,
                        minimum_samples=MINIMUM_MODEL_SAMPLES,
                        trained_at=train_data.candles[-1].timestamp,
                    )
                    model_info = {
                        "id": dataset_key,
                        "campaign_id": CAMPAIGN_ID,
                        "pair": pair,
                        "timeframe": timeframe,
                        "window_id": window_id,
                        "dataset_version": data.metadata["dataset_version"],
                        "status": trained.status,
                        "reason": trained.reason,
                        "validation": json_value(trained.model)
                        if trained.model
                        else None,
                        "oos_start": data.candles[split].timestamp,
                        "training_end": train_data.candles[-1].timestamp,
                    }
                    append_jsonl(MODELS_PATH, model_info)
                    existing_models[dataset_key] = model_info
                    if trained.model:
                        model = trained.model
                        save_model(
                            journal,
                            model,
                            dataset_key,
                            pair,
                            timeframe,
                            data.metadata["dataset_version"],
                        )
                elif model_info.get("status") == "TRAINED":
                    model = model_from_payload(model_info["validation"])

                test_strategies = [(s.name, s) for s in strategies]
                if model is not None:
                    test_strategies.append(("ml_direction", LearnedDirection(model)))
                for scenario, multiplier in COST_MULTIPLIERS.items():
                    for strategy_name, strategy in test_strategies:
                        identity = experiment_id(
                            pair, timeframe, window_id, scenario, strategy_name
                        )
                        if identity in completed:
                            continue
                        costs = costs_for(pair, multiplier)
                        initial_equity = starting_equity(pair, data, split)
                        simulation = run_report(
                            data,
                            strategy,
                            costs,
                            initial_equity,
                            RISK_FRACTION,
                            execution=EXECUTION,
                            start_index=split,
                            evidence_source="OOS_PAPER",
                        )
                        result = result_row(
                            data,
                            window_id,
                            scenario,
                            strategy_name,
                            model_info,
                            simulation,
                            split,
                            initial_equity,
                        )
                        persist_run(journal, result, simulation)
                        append_jsonl(RESULTS_PATH, result)
                        completed.add(identity)
                        write_progress(
                            "RUNNING",
                            len(completed),
                            total,
                            current_pair=pair,
                            current_timeframe=timeframe,
                            current_window=window_id,
                            current_experiment=identity,
                            latest_trade_count=result["trade_count"],
                            latest_net_return_pct=result["net_return_pct"],
                        )
                        print(
                            f"{len(completed)}/{total} {identity}: "
                            f"{result['trade_count']} trades, "
                            f"{result['net_return_pct']:+.3f}% {result['native_currency']}",
                            flush=True,
                        )
        results = read_jsonl(RESULTS_PATH)
        trained_count = sum(
            row.get("status") == "TRAINED" for row in existing_models.values()
        )
        actual_total = len(windows) * len(TIMEFRAMES) * len(strategies) * len(
            COST_MULTIPLIERS
        ) + trained_count * len(COST_MULTIPLIERS)
        summary = summarize(results)
        report = {
            "campaign_id": CAMPAIGN_ID,
            "status": "COMPLETED",
            "completed_at": now(),
            "source": "Dukascopy-curated, checksum-verified historical windows",
            "windows": len(windows),
            "models": list(existing_models.values()),
            "test_count": len(results),
            "upper_bound_test_count": total,
            "actual_total_test_count": int(actual_total),
            "strategies": [strategy.name for strategy in strategies] + ["ml_direction"],
            "timeframes": TIMEFRAMES,
            "holdout": "Final 30% of each independent window; no joins across gaps; model fit/validation use only prior 70% with label-boundary purge.",
            "cost_scenarios": COST_MULTIPLIERS,
            "summary": summary,
            "results_jsonl": str(RESULTS_PATH.relative_to(ROOT)),
            "trades_jsonl": str(TRADES_PATH.relative_to(ROOT)),
            "models_jsonl": str(MODELS_PATH.relative_to(ROOT)),
            "limits": [
                "Fixed assumed spread/slippage and zero commission; historical bid/ask quotes are unavailable.",
                "The 22 accepted windows are disjoint and are summarized separately; do not treat them as one continuous account path.",
                "Historical backtests and ML holdouts do not prove future profitability or replace forward paper evidence.",
                "USDJPY P&L is kept in JPY; other pair P&L is USD-denominated.",
            ],
        }
        write_json(REPORT_PATH, report)
        write_trades_ledger()
        journal.record(
            Entity.AUDIT,
            now(),
            {
                "event": "curated_four_pair_training_evaluation_completed",
                "campaign_id": CAMPAIGN_ID,
                "test_count": len(results),
                "model_count": len(existing_models),
                "summary": summary,
                "report_path": str(REPORT_PATH.relative_to(ROOT)),
            },
            record_id=record_id("campaign-complete", CAMPAIGN_ID),
        )
        write_progress(
            "COMPLETED",
            len(completed),
            int(actual_total),
            report_path=str(REPORT_PATH.relative_to(ROOT)),
        )
        return 0
    except (
        Exception
    ) as error:  # preserve checkpoint; keep errors actionable but sanitized
        failures.append({"type": type(error).__name__, "at": now().isoformat()})
        write_json(OUTPUT_ROOT / "failures.json", failures)
        write_progress(
            "FAILED",
            len(completed),
            total,
            last_error=type(error).__name__,
            last_completed_experiment=next(iter(completed), None),
        )
        raise
    finally:
        repository.close()


def summarize(results: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = {}
    for result in results:
        key = (
            f"{result['strategy']}|{result['pair']}|{result['timeframe']}|"
            f"{result['cost_scenario']}"
        )
        buckets.setdefault(key, []).append(result)
    summary = {}
    for key, rows in sorted(buckets.items()):
        pnl_pct = [float(row["net_return_pct"]) for row in rows]
        trade_counts = [int(row["trade_count"]) for row in rows]
        summary[key] = {
            "window_runs": len(rows),
            "positive_windows": sum(value > 0 for value in pnl_pct),
            "negative_windows": sum(value < 0 for value in pnl_pct),
            "flat_windows": sum(value == 0 for value in pnl_pct),
            "median_return_pct": median(pnl_pct),
            "mean_return_pct": sum(pnl_pct) / len(pnl_pct),
            "total_closed_trades": sum(trade_counts),
            "runs_below_30_trades": sum(value < 30 for value in trade_counts),
        }
    return summary


def write_trades_ledger() -> None:
    import sqlite3

    path = JOURNAL_PATH.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(path, uri=True) as connection:
        rows = connection.execute(
            "SELECT row_json FROM astra_records WHERE entity = ? ORDER BY occurred_at, record_id",
            (Entity.TRADE.value,),
        ).fetchall()
    temp = TRADES_PATH.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for (raw,) in rows:
            row = json.loads(raw)
            if row.get("payload", {}).get("campaign_id") == CAMPAIGN_ID:
                stream.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
    os.replace(temp, TRADES_PATH)


if __name__ == "__main__":
    raise SystemExit(run_campaign())
