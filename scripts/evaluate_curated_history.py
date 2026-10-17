"""Evaluate approved four-pair history and publish atomic local research snapshots."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial, wraps
from pathlib import Path

from astra.adapters import SQLiteRepository
from astra.backtest import BacktestCosts, ExecutionAssumptions, run_report, walk_forward
from astra.historical.dataset import load_dataset
from astra.ml import dataset as ml_dataset
from astra.ml import train as train_model
from astra.models import MarketData
from astra.repositories import Entity, Journal, json_value
from astra.strategies import StrategyRegistry

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "astra-training-four-pairs"
OUT = DATA_ROOT / "research-results.json"
STATE = DATA_ROOT / "research-progress.json"
CONTROL = DATA_ROOT / "research-control.json"
LOCK = DATA_ROOT / "research-evaluation.lock"
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
TIMEFRAMES = ("M5", "M15", "H1", "H4")
LOOKBACK = 128
PIP_COSTS = {"spread_pips": 1.0, "slippage_pips": 0.2, "commission_per_unit": 0.0}


def save(path, value):
    contents = json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                if path == STATE:
                    path.write_text(contents, encoding="utf-8")
                    temporary.unlink(missing_ok=True)
                    return
                raise
            time.sleep(0.25)


@contextmanager
def evaluation_lock():
    """Prevent launchers and manual commands from running duplicate evaluations."""
    handle = LOCK.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = partial(msvcrt.locking, handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = partial(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        handle.close()
        raise RuntimeError(
            "Another Astra historical evaluation is already running"
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    try:
        yield
    finally:
        unlock()
        handle.close()


def prevent_concurrent_evaluation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with evaluation_lock():
            return function(*args, **kwargs)

    return wrapped


def prepare_resume(previous, previous_state, expected_scope, expected_assumptions):
    if previous.get("status") not in {"FAILED", "RUNNING"}:
        raise ValueError("Resume requires an unfinished research-results.json")
    if previous.get("scope") != json_value(expected_scope) or previous.get(
        "assumptions"
    ) != json_value(expected_assumptions):
        raise ValueError("Saved research scope or assumptions differ from this run")
    previous["resume_history"] = previous.get("resume_history", [])
    previous["resume_history"].append(
        {
            "resumed_at": datetime.now(UTC),
            "previous_status": previous["status"],
            "previous_progress": previous_state.get("completed", 0),
            "previous_failure": previous.get("errors", []),
        }
    )
    previous["errors"] = [
        error for error in previous.get("errors", []) if "type" not in error
    ]
    previous["status"] = "RUNNING"
    completed = max(previous_state.get("completed", 0), len(previous.get("items", [])))
    return previous, completed


@prevent_concurrent_evaluation
def run(*, resume=False):
    manifest = json.loads((DATA_ROOT / "curation.json").read_text(encoding="utf-8"))
    segments = [
        (pair, segment["id"])
        for pair in PAIRS
        for segment in manifest["pairs"][pair]["segments"]
        if all(tf in segment.get("loadable_timeframes", ()) for tf in TIMEFRAMES)
    ]
    registry = StrategyRegistry()
    strategies = registry.all()
    backtest_tasks = sum(
        len(strategies) * len(TIMEFRAMES)
        for pair, segment in segments
        if all(
            tf
            in next(
                s for s in manifest["pairs"][pair]["segments"] if s["id"] == segment
            )["loadable_timeframes"]
            for tf in TIMEFRAMES
        )
    )
    wf_tasks = sum(
        len(strategies) * len(TIMEFRAMES)
        for pair, segment in segments
        if all(
            tf
            in next(
                s for s in manifest["pairs"][pair]["segments"] if s["id"] == segment
            )["loadable_timeframes"]
            for tf in ("H1", "H4")
        )
    )
    ml_tasks = sum(
        len(TIMEFRAMES)
        for pair, segment in segments
        if all(
            tf
            in next(
                s for s in manifest["pairs"][pair]["segments"] if s["id"] == segment
            )["loadable_timeframes"]
            for tf in TIMEFRAMES
        )
    )
    total = backtest_tasks * 3 + wf_tasks + ml_tasks
    results = {
        "status": "RUNNING",
        "started_at": datetime.now(UTC),
        "scope": {"pairs": PAIRS, "timeframes": TIMEFRAMES, "segments": len(segments)},
        "assumptions": {
            **PIP_COSTS,
            "context_lookback_bars": LOOKBACK,
            "price_basis": "BID; historical ASK quotes unavailable",
            "strategies": [s.name for s in strategies],
            "risk": "existing risk engine; 0.5% initial-equity fraction per trade",
        },
        "items": [],
        "errors": [],
        "profitability": "UNPROVEN",
        "limitations": [
            "Historical evaluation cannot prove future profitability.",
            "Costs use fixed spread and slippage; acquired archive contains BID bars only.",
            "Windows remain separate; no missing month is filled or stitched.",
            "Bounded 128-bar decision context is recorded for reproducibility.",
            "Data source output was quality-checked but not independently authenticated.",
        ],
    }
    completed = 0
    if resume:
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        previous_state = json.loads(STATE.read_text(encoding="utf-8"))
        results, completed = prepare_resume(
            previous, previous_state, results["scope"], results["assumptions"]
        )
        completed = min(total, completed)

    # The JSON report is written in batches, while Journal records are durable
    # after each evaluation. Use both sources when resuming so an interrupted
    # batch is not evaluated and counted again.
    repo = SQLiteRepository(str(DATA_ROOT / "research-journal.sqlite3"))
    journal = Journal(repo)
    existing_results = {
        (
            item.get("kind", ""),
            item.get("pair", ""),
            item.get("segment", ""),
            item.get("timeframe", ""),
            item.get("strategy", ""),
        )
        for item in results["items"]
    }
    journal_entities = (
        (Entity.BACKTEST, "BACKTEST"),
        (Entity.METRICS, "OOS_HOLDOUT"),
        (Entity.ROBUSTNESS, "COST_STRESS"),
        (Entity.WALK_FORWARD, "WALK_FORWARD"),
    )
    for entity, default_kind in journal_entities:
        for record in repo.query(entity):
            payload = record.payload
            parts = payload.get("experiment_key", "").split(":", 3)
            if len(parts) == 4:
                pair, segment, timeframe, strategy = parts
                existing_results.add(
                    (
                        payload.get("source", default_kind),
                        pair,
                        segment,
                        timeframe,
                        strategy,
                    )
                )
    completed = min(total, len(existing_results))
    CONTROL.unlink(missing_ok=True)
    save(OUT, results)
    save(
        STATE,
        {
            "status": "RUNNING",
            "stage": "Starting",
            "completed": completed,
            "total": total,
            "updated_at": datetime.now(UTC),
        },
    )

    def progress(stage, pair, segment, timeframe, strategy="", detail=""):
        nonlocal completed
        key = (stage, pair, segment, timeframe, strategy)
        if key not in existing_results:
            existing_results.add(key)
            completed = min(total, completed + 1)
        state = {
            "status": "RUNNING",
            "stage": stage,
            "pair": pair,
            "segment": segment,
            "timeframe": timeframe,
            "strategy": strategy,
            "detail": detail,
            "completed": completed,
            "total": total,
            "updated_at": datetime.now(UTC),
        }
        save(STATE, state)

    def honor_pause():
        paused = False
        while CONTROL.exists():
            try:
                action = json.loads(CONTROL.read_text(encoding="utf-8")).get("action")
            except (OSError, json.JSONDecodeError):
                action = None
            if action != "pause":
                CONTROL.unlink(missing_ok=True)
                break
            paused = True
            save(
                STATE,
                {
                    "status": "PAUSED",
                    "stage": "Paused safely between evaluation steps",
                    "completed": completed,
                    "total": total,
                    "updated_at": datetime.now(UTC),
                },
            )
            time.sleep(1)
        if paused:
            save(
                STATE,
                {
                    "status": "RUNNING",
                    "stage": "Resuming evaluation",
                    "completed": completed,
                    "total": total,
                    "updated_at": datetime.now(UTC),
                },
            )

    def has_result(kind, pair, segment, timeframe, strategy=None):
        return (
            kind,
            pair,
            segment,
            timeframe,
            strategy or "",
        ) in existing_results

    execution = ExecutionAssumptions(context_lookback_bars=LOOKBACK)
    try:
        for pair, segment in segments:
            segment_root = DATA_ROOT / pair / segment
            for timeframe in TIMEFRAMES:
                honor_pause()
                save(
                    STATE,
                    {
                        "status": "RUNNING",
                        "stage": "Validating dataset",
                        "pair": pair,
                        "segment": segment,
                        "timeframe": timeframe,
                        "completed": completed,
                        "total": total,
                        "updated_at": datetime.now(UTC),
                    },
                )
                try:
                    data = load_dataset(segment_root, pair, timeframe)
                except (OSError, ValueError, KeyError) as error:
                    results["errors"].append(
                        {
                            "pair": pair,
                            "segment": segment,
                            "timeframe": timeframe,
                            "reason": type(error).__name__,
                        }
                    )
                    continue
                ml_identity = ("ML_TRAINING", pair, segment, timeframe, "")
                if ml_identity not in existing_results and any(
                    record.occurred_at == data.candles[-1].timestamp
                    for record in repo.query(
                        Entity.MODEL, pair=pair, timeframe=timeframe
                    )
                ):
                    existing_results.add(ml_identity)
                pip = 0.01 if pair.endswith("JPY") else 0.0001
                costs = BacktestCosts(
                    PIP_COSTS["spread_pips"] * pip,
                    PIP_COSTS["slippage_pips"] * pip,
                    PIP_COSTS["commission_per_unit"],
                    1.0,
                    "bid",
                )
                for strategy in strategies:
                    honor_pause()
                    if len(data.candles) < 64:
                        continue
                    experiment_key = f"{pair}:{segment}:{timeframe}:{strategy.name}"
                    if not has_result(
                        "BACKTEST", pair, segment, timeframe, strategy.name
                    ):
                        t0 = time.perf_counter()
                        report = run_report(data, strategy, costs, execution=execution)
                        journal.record(
                            Entity.BACKTEST,
                            data.candles[-1].timestamp,
                            {
                                "experiment_key": experiment_key,
                                "source": "BACKTEST",
                                "dataset_version": data.metadata["dataset_version"],
                                "metrics": report.metrics,
                                "trade_count": len(report.trades),
                            },
                            pair=pair,
                            timeframe=timeframe,
                            strategy=strategy.name,
                        )
                        results["items"].append(
                            {
                                "kind": "BACKTEST",
                                "pair": pair,
                                "segment": segment,
                                "timeframe": timeframe,
                                "strategy": strategy.name,
                                "dataset_version": data.metadata["dataset_version"],
                                "first_close": data.candles[0].timestamp,
                                "last_close": data.candles[-1].timestamp,
                                "candles": len(data.candles),
                                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                                "metrics": report.metrics,
                                "trades": report.trades,
                            }
                        )
                        progress("BACKTEST", pair, segment, timeframe, strategy.name)
                        honor_pause()
                    split = int(len(data.candles) * 0.7)
                    if split >= 30 and len(data.candles) - split >= 30:
                        if not has_result(
                            "OOS_HOLDOUT", pair, segment, timeframe, strategy.name
                        ):
                            holdout = run_report(
                                data,
                                strategy,
                                costs,
                                execution=execution,
                                start_index=split - 1,
                                evidence_source="OOS",
                            )
                            journal.record(
                                Entity.METRICS,
                                data.candles[-1].timestamp,
                                {
                                    "experiment_key": experiment_key,
                                    "source": "OOS_HOLDOUT",
                                    "dataset_version": data.metadata["dataset_version"],
                                    "split_close": data.candles[split - 1].timestamp,
                                    "metrics": holdout.metrics,
                                    "trade_count": len(holdout.trades),
                                },
                                pair=pair,
                                timeframe=timeframe,
                                strategy=strategy.name,
                            )
                            results["items"].append(
                                {
                                    "kind": "OOS_HOLDOUT",
                                    "pair": pair,
                                    "segment": segment,
                                    "timeframe": timeframe,
                                    "strategy": strategy.name,
                                    "split_close": data.candles[split - 1].timestamp,
                                    "test_start": data.candles[split].timestamp,
                                    "test_end": data.candles[-1].timestamp,
                                    "metrics": holdout.metrics,
                                    "trades": holdout.trades,
                                }
                            )
                            progress(
                                "OOS_HOLDOUT", pair, segment, timeframe, strategy.name
                            )
                            honor_pause()
                        train_data = replace(data, candles=data.candles[:split])
                        stressed_costs = replace(
                            costs,
                            spread=costs.spread * 2,
                            slippage=costs.slippage * 2,
                        )
                        if not has_result(
                            "COST_STRESS", pair, segment, timeframe, strategy.name
                        ):
                            stressed = run_report(
                                train_data,
                                strategy,
                                stressed_costs,
                                execution=execution,
                            )
                            journal.record(
                                Entity.ROBUSTNESS,
                                train_data.candles[-1].timestamp,
                                {
                                    "experiment_key": experiment_key,
                                    "source": "COST_STRESS",
                                    "cost_multiplier": 2,
                                    "dataset_version": data.metadata["dataset_version"],
                                    "metrics": stressed.metrics,
                                    "trade_count": len(stressed.trades),
                                },
                                pair=pair,
                                timeframe=timeframe,
                                strategy=strategy.name,
                            )
                            results["items"].append(
                                {
                                    "kind": "COST_STRESS",
                                    "pair": pair,
                                    "segment": segment,
                                    "timeframe": timeframe,
                                    "strategy": strategy.name,
                                    "cost_multiplier": 2,
                                    "metrics": stressed.metrics,
                                    "trades": stressed.trades,
                                }
                            )
                            progress(
                                "COST_STRESS", pair, segment, timeframe, strategy.name
                            )
                            honor_pause()
                    if timeframe in TIMEFRAMES and not has_result(
                        "WALK_FORWARD", pair, segment, timeframe, strategy.name
                    ):
                        train_size = max(30, int(split * 0.5))
                        test_size = max(10, int(split * 0.1))
                        development_data = replace(data, candles=data.candles[:split])
                        folds = walk_forward(development_data, train_size, test_size)
                        wf_items = []
                        for fold_index, (train, test) in enumerate(folds, 1):
                            honor_pause()
                            combined = MarketData(
                                pair,
                                timeframe,
                                train.candles + test.candles,
                                data.metadata,
                            )
                            oos = run_report(
                                combined,
                                strategy,
                                costs,
                                execution=execution,
                                start_index=len(train.candles) - 1,
                                evidence_source="OOS",
                            )
                            wf_items.append(
                                {
                                    "fold": fold_index,
                                    "train_start": train.candles[0].timestamp,
                                    "train_end": train.candles[-1].timestamp,
                                    "test_start": test.candles[0].timestamp,
                                    "test_end": test.candles[-1].timestamp,
                                    "metrics": oos.metrics,
                                    "trades": oos.trades,
                                }
                            )
                        if wf_items:
                            journal.record(
                                Entity.WALK_FORWARD,
                                wf_items[-1]["test_end"],
                                {
                                    "experiment_key": experiment_key,
                                    "dataset_version": data.metadata["dataset_version"],
                                    "folds": [
                                        {
                                            key: fold[key]
                                            for key in (
                                                "fold",
                                                "train_start",
                                                "train_end",
                                                "test_start",
                                                "test_end",
                                                "metrics",
                                            )
                                        }
                                        for fold in wf_items
                                    ],
                                },
                                pair=pair,
                                timeframe=timeframe,
                                strategy=strategy.name,
                            )
                            results["items"].append(
                                {
                                    "kind": "WALK_FORWARD",
                                    "pair": pair,
                                    "segment": segment,
                                    "timeframe": timeframe,
                                    "strategy": strategy.name,
                                    "folds": wf_items,
                                }
                            )
                            progress(
                                "WALK_FORWARD", pair, segment, timeframe, strategy.name
                            )
                            honor_pause()
                if (
                    timeframe in TIMEFRAMES
                    and len(data.candles) >= 100
                    and not has_result("ML_TRAINING", pair, segment, timeframe)
                ):
                    ml_data = replace(
                        data, metadata=data.metadata | {"real_data": True}
                    )
                    training = train_model(
                        ml_data,
                        enabled=True,
                        minimum_samples=300,
                        journal=journal,
                        trained_at=data.candles[-1].timestamp,
                    )
                    examples = ml_dataset(ml_data)
                    results["items"].append(
                        {
                            "kind": "ML_TRAINING",
                            "pair": pair,
                            "segment": segment,
                            "timeframe": timeframe,
                            "dataset_version": data.metadata["dataset_version"],
                            "status": training.status,
                            "reason": training.reason,
                            "sample_count": len(examples),
                            "model": training.model,
                        }
                    )
                    progress(
                        "ML_TRAINING", pair, segment, timeframe, detail=training.status
                    )
                    honor_pause()
                save(OUT, results)
        results["status"] = "COMPLETED"
        results["completed_at"] = datetime.now(UTC)
        results["profitability"] = "UNPROVEN"
        save(OUT, results)
        save(
            STATE,
            {
                "status": "COMPLETED",
                "completed": completed,
                "total": total,
                "updated_at": datetime.now(UTC),
            },
        )
    except Exception as error:
        results["status"] = "FAILED"
        results["errors"].append(
            {"type": type(error).__name__, "detail": str(error)[:400]}
        )
        save(OUT, results)
        save(
            STATE,
            {
                "status": "FAILED",
                "completed": completed,
                "total": total,
                "updated_at": datetime.now(UTC),
            },
        )
        raise
    finally:
        repo.close()


if __name__ == "__main__":
    import argparse

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with (
        (DATA_ROOT / "research-supabase-sync.stdout.log").open("a", encoding="utf-8") as stdout,
        (DATA_ROOT / "research-supabase-sync.stderr.log").open("a", encoding="utf-8") as stderr,
    ):
        flags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0
        )
        try:
            subprocess.Popen(
                [sys.executable, "-m", "scripts.sync_research_to_supabase", "--watch"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                creationflags=flags,
            )
        except OSError:
            # The research remains durable locally and the launcher also starts
            # the sync worker; a later launch will retry the remote mirror.
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the saved failed run without recomputing recorded results.",
    )
    run(resume=parser.parse_args().resume)
