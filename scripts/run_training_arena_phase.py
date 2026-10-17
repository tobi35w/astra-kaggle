"""Offline, resumable second-phase ML and paper simulations for Training Arena."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pyarrow.parquet as pq

from astra.adapters import SQLiteRepository
from astra.backtest import BacktestCosts, ExecutionAssumptions, run_report
from astra.indicators import true_ranges
from astra.ml import Model, train
from astra.models import Candle, MarketData, Side, Signal
from astra.repositories import Entity, Journal, Record, json_value
from astra.strategies import StrategyRegistry

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
ARENA_ROOT = ROOT / "data" / "training-arena"
MANIFEST_PATH = DATA_ROOT / "four-pairs-manifest.json"
QUALITY_PATH = DATA_ROOT / "quality-report.json"
RESEARCH_STATE = ROOT / "data" / "astra-training-four-pairs" / "research-progress.json"
RESEARCH_RESULTS = ROOT / "data" / "astra-training-four-pairs" / "research-results.json"
STATE_PATH = ARENA_ROOT / "arena-state.json"
RESULTS_PATH = ARENA_ROOT / "arena-results.json"
CONTROL_PATH = ARENA_ROOT / "arena-control.json"
LOCK_PATH = ARENA_ROOT / "arena-worker.lock"
JOURNAL_PATH = ARENA_ROOT / "arena-journal.sqlite3"
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4")
SIM_TIMEFRAMES = ("M1", "M5", "M15", "H1", "H4")
BENCHMARKS = ("ema_trend", "mean_reversion", "breakout")
MINIMUM_SAMPLES = 300
STARTING_BALANCE = 10_000.0
RISK_PER_TRADE = 0.005
SPREAD_PIPS = 1.0
SLIPPAGE_PIPS = 0.2


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def update_state(**values) -> dict:
    state = read_json(STATE_PATH, {}) or {}
    state.update(values, updated_at=utc_now())
    write_json(STATE_PATH, state)
    return state


def claim_worker() -> bool:
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            os.kill(int(LOCK_PATH.read_text(encoding="utf-8").strip()), 0)
            return False
        except (OSError, ValueError):
            LOCK_PATH.unlink(missing_ok=True)
            return claim_worker()


def verify_data() -> tuple[list[dict], dict]:
    manifest = read_json(MANIFEST_PATH)
    quality = read_json(QUALITY_PATH)
    if not manifest or not quality:
        raise ValueError("the saved four-pair FX dataset manifest or quality audit is missing")
    if quality.get("structural_fail_files") or quality.get("structural_pass_files") != 20:
        raise ValueError("dataset structural audit did not pass for all 20 pair/timeframe files")
    entries = manifest.get("files", [])
    expected = {(pair, timeframe) for pair in PAIRS for timeframe in TIMEFRAMES}
    actual = {(entry.get("symbol"), entry.get("timeframe")) for entry in entries}
    if actual != expected:
        raise ValueError("saved dataset does not contain exactly the approved four pairs and five timeframes")
    for entry in entries:
        path = DATA_ROOT / entry["labeled_file"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"curated Parquet file is missing: {entry['symbol']} {entry['timeframe']}")
    return entries, quality


def load_market_data(entry: dict) -> tuple[MarketData, str]:
    path = DATA_ROOT / entry["labeled_file"]
    table = pq.read_table(path)
    required = {"Time", "Open", "High", "Low", "Close", "Volume", "Symbol", "Timeframe"}
    if not required.issubset(table.column_names) or table.num_rows != entry["rows"]:
        raise ValueError(f"curated Parquet schema or row count mismatch: {entry['symbol']} {entry['timeframe']}")
    if table["Symbol"][0].as_py() != entry["symbol"] or table["Timeframe"][0].as_py() != entry["timeframe"]:
        raise ValueError("pair/timeframe identity in curated dataset does not match its manifest")
    raw_times = table["Time"].to_pylist()
    columns = {
        name: table[name].to_pylist()
        for name in ("Open", "High", "Low", "Close", "Volume")
    }
    duration = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}[entry["timeframe"]]
    delta = timedelta(minutes=duration)
    candles = []
    previous = None
    for index, raw_time in enumerate(raw_times):
        if raw_time.tzinfo is None:
            raise ValueError("dataset candle timestamps must be timezone-aware")
        # Source Time is a bar-open time; decision availability is the bar close.
        stamp = raw_time.astimezone(UTC) + delta
        if previous is not None and stamp <= previous:
            raise ValueError("dataset timestamps must be strictly chronological")
        previous = stamp
        values = [float(columns[name][index]) for name in ("Open", "High", "Low", "Close", "Volume")]
        o, h, low, close, volume = values
        if not all(np.isfinite(values)) or min(o, h, low, close) <= 0 or volume < 0:
            raise ValueError("dataset has non-finite or invalid OHLCV values")
        if low > min(o, close) or h < max(o, close):
            raise ValueError("dataset violates OHLC bounds")
        candles.append(Candle(stamp, o, h, low, close, volume, "tick"))
    if len(candles) < 1000:
        raise ValueError("dataset file has too few rows for this research phase")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {
        "source": "Yllvar/fx-historical-data",
        "dataset_version": f"sha256:{digest}",
        "price_basis": "bid",
        "timestamp_semantics": "source bar-open timestamp converted to close/availability time",
        "quality_status": "PASS",
        "real_data": True,
        "price_authentication": "not independently authenticated",
    }
    return MarketData(entry["symbol"], entry["timeframe"], tuple(candles), source), digest


class LearnedDirection:
    """Fixed chronological model inference; every order still uses the risk engine."""

    name = "ml_direction"

    def __init__(self, model: Model):
        self.model = model

    def generate(self, data: MarketData, context):
        candles = data.candles
        if len(candles) < 15 or candles[-1].timestamp < self.model.trained_at:
            return None
        closes = [c.close for c in candles[-6:]]
        recent = candles[-15:]
        ranges = true_ranges(
            [c.high for c in recent],
            [c.low for c in recent],
            [c.close for c in recent],
        )
        atr = sum(ranges[-14:]) / 14
        last = candles[-1]
        values = (
            closes[-1] / closes[-2] - 1,
            closes[-1] / closes[0] - 1,
            atr / last.close,
            (last.close - last.open) / max(last.high - last.low, 1e-12),
        )
        probability = self.model.predict(values, last.timestamp)
        if atr <= 0 or 0.45 < probability < 0.55:
            return None
        side = Side.LONG if probability >= 0.55 else Side.SHORT
        distance = 1.5 * atr
        entry = last.close
        stop = entry - distance if side == Side.LONG else entry + distance
        target = entry + distance * 1.5 if side == Side.LONG else entry - distance * 1.5
        confidence = probability if side == Side.LONG else 1 - probability
        return Signal(
            self.name, side, float(confidence), entry, stop, target,
            f"chronological model p(up)={probability:.3f}; 0.55/0.45 threshold; 1.5 ATR stop; 1.5R target",
            1.5,
        )


def experiment_id(kind: str, pair: str, timeframe: str, strategy: str) -> str:
    return f"{kind}:{pair}:{timeframe}:{strategy}"


def journal_trade_rows(journal: Journal, test_id: str, dataset_version: str, report) -> None:
    pair = test_id.split(":")[1]
    timeframe = test_id.split(":")[2]
    records = []
    for sequence, trade in enumerate(report.trades):
        payload = {
            "source": "TRAINING_ARENA_PAPER",
            "experiment_id": test_id,
            "dataset_version": dataset_version,
            "trade_sequence": sequence,
            "trade": json_value(trade),
        }
        stamp = trade.closed_at or trade.opened_at
        if stamp is None:
            continue
        rid = str(uuid5(NAMESPACE_URL, f"astra-arena-trade:{test_id}:{sequence}:{stamp.isoformat()}"))
        records.append(
            Record(
                Entity.TRADE,
                stamp,
                payload,
                pair=pair,
                timeframe=timeframe,
                strategy=trade.strategy,
                correlation_id=test_id,
                record_id=rid,
            )
        )
    if records:
        journal.repository.append_batch(tuple(records))


def result_from_report(test_id: str, pair: str, timeframe: str, strategy: str, dataset_version: str, model_info, report, data: MarketData, split: int) -> dict:
    metrics = json_value(report.metrics)
    return {
        "id": test_id,
        "pair": pair,
        "timeframe": timeframe,
        "strategy": strategy,
        "kind": "ML_PAPER" if strategy == "ml_direction" else "STRATEGY_BENCHMARK",
        "evidence": "OOS_PAPER_SIMULATION",
        "source": "Yllvar/fx-historical-data",
        "dataset_version": dataset_version,
        "test_start": data.candles[split].timestamp,
        "test_end": data.candles[-1].timestamp,
        "model": model_info,
        "starting_balance": STARTING_BALANCE,
        "ending_equity": STARTING_BALANCE + metrics["net_pnl"],
        "net_pnl": metrics["net_pnl"],
        "metrics": metrics,
        "assumptions": {
            "spread_pips": SPREAD_PIPS,
            "slippage_pips": SLIPPAGE_PIPS,
            "commission_per_unit": 0.0,
            "price_basis": "bid; long entries pay the full modeled spread, short exits pay the spread",
            "risk_per_trade_pct": RISK_PER_TRADE * 100,
            "max_open_positions": 1,
            "starting_balance": STARTING_BALANCE,
            "leverage": "not modeled; risk-sized units only",
            "account_currency_conversion": "1 price-unit value assumption; USD account P&L approximation",
        },
        "execution_assumptions": json_value(report.execution_assumptions),
        "trades": [json_value(trade) for trade in report.trades],
        "completed_at": utc_now(),
    }


def pause_checkpoint(journal: Journal, pair: str, timeframe: str, strategy: str, phase: str, stats: dict) -> None:
    while True:
        control = read_json(CONTROL_PATH, {}) or {}
        if control.get("action") != "pause":
            if read_json(STATE_PATH, {}).get("status") == "PAUSED":
                update_state(status="RUNNING", stage="Resumed", updated_at=utc_now())
            return
        update_state(
            status="PAUSED",
            stage="Paused safely at a simulation checkpoint",
            active_test={"pair": pair, "timeframe": timeframe, "strategy": strategy, "phase": phase, **stats},
        )
        time.sleep(1)


def run_phase() -> None:
    ARENA_ROOT.mkdir(parents=True, exist_ok=True)
    research = read_json(RESEARCH_STATE, {}) or {}
    if research.get("status") != "COMPLETED":
        update_state(status="WAITING_FOR_RESEARCH", stage="Waiting for the current Research Sandbox evaluation to complete")
        return
    entries, quality = verify_data()
    report = read_json(RESULTS_PATH, {}) or {
        "campaign_id": "astra-training-arena-yllvar-v1",
        "source_dataset": "Yllvar/fx-historical-data",
        "scope": {"pairs": PAIRS, "timeframes": TIMEFRAMES},
        "training_window_fraction": 0.70,
        "initial_balance_per_test": STARTING_BALANCE,
        "tests": [],
        "models": [],
        "assumptions": [
            "oos_uses_final_30_percent_chronologically",
            "market_timestamp_is_open_plus_timeframe_to_close_availability",
            "no_orders_or_broker_connections",
            "one_open_position_per_independent_test_account",
        ],
        "source_limitations": quality.get("research_readiness", {}).get("requires_followup", []),
        "current_research_snapshot": read_json(RESEARCH_RESULTS, {}).get("profitability", "UNPROVEN"),
    }
    completed = {item.get("id") for item in report.get("tests", [])}
    existing_models = {item.get("id") for item in report.get("models", [])}
    total_tests = 20 + (4 * 4 * len(BENCHMARKS))
    total_models = 20
    update_state(
        status="RUNNING",
        stage="Loading verified local FX dataset",
        completed_tests=len(completed),
        total_tests=total_tests,
        completed_models=len(existing_models),
        total_models=total_models,
        current_research="COMPLETED; carried forward as source-separated context, not mixed into Yllvar model labels",
        dataset={
            "source": "Yllvar/fx-historical-data",
            "pairs": list(PAIRS),
            "timeframes": list(TIMEFRAMES),
            "rows": sum(item["rows"] for item in entries),
            "files": len(entries),
            "structural_pass_files": quality.get("structural_pass_files"),
            "structural_fail_files": quality.get("structural_fail_files"),
            "created_at_utc": read_json(MANIFEST_PATH, {}).get("created_at_utc"),
            "offline_ready": True,
            "limitations": quality.get("research_readiness", {}).get("requires_followup", []),
        },
        active_test=None,
        last_error="",
    )
    repository = SQLiteRepository(str(JOURNAL_PATH))
    journal = Journal(repository)
    try:
        journal.record(
            Entity.AUDIT,
            datetime.now(UTC),
            {
                "event": "training_arena_phase_started",
                "campaign_id": report["campaign_id"],
                "source_dataset": report["source_dataset"],
                "current_research": "prior phase remains source-separated context",
                "scope": report["scope"],
                "assumptions": report["assumptions"],
                "limitations": report["source_limitations"],
            },
            record_id=str(uuid5(NAMESPACE_URL, "astra-training-arena-phase:v1")),
        )
        benchmarks = StrategyRegistry()
        split_fraction = 0.70
        for entry in entries:
            pair, timeframe = entry["symbol"], entry["timeframe"]
            update_state(stage="Validating and loading curated source", current_pair=pair, current_timeframe=timeframe)
            data, digest = load_market_data(entry)
            split = int(len(data.candles) * split_fraction)
            if split < MINIMUM_SAMPLES + 100 or len(data.candles) - split < 100:
                continue
            version = f"sha256:{digest}"
            model_id = experiment_id("ML_MODEL", pair, timeframe, "direction_lr")
            model_info = next((item for item in report["models"] if item.get("id") == model_id), None)
            if model_info is None:
                training_data = MarketData(pair, timeframe, data.candles[:split], data.metadata)
                update_state(stage="Training chronological model", current_pair=pair, current_timeframe=timeframe, current_strategy="direction_lr")
                trained = train(
                    training_data,
                    enabled=True,
                    minimum_samples=MINIMUM_SAMPLES,
                    trained_at=training_data.candles[-1].timestamp,
                )
                if trained.model is not None:
                    journal.record(
                        Entity.MODEL,
                        trained.model.trained_at,
                        trained.model,
                        pair=pair,
                        timeframe=timeframe,
                        record_id=str(uuid5(NAMESPACE_URL, "astra-arena-model:" + model_id)),
                    )
                model_info = {
                    "id": model_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "dataset_version": version,
                    "status": trained.status,
                    "reason": trained.reason,
                    "validation": json_value(trained.model) if trained.model else None,
                    "oos_start": data.candles[split].timestamp,
                    "sample_count": trained.model.sample_count if trained.model else 0,
                }
                report["models"].append(model_info)
                write_json(RESULTS_PATH, report)
                update_state(completed_models=len(report["models"]), stage="Model trained; starting isolated OOS simulations")
            if model_info.get("status") == "TRAINED":
                test_id = experiment_id("ML_PAPER", pair, timeframe, "ml_direction")
                if test_id not in completed:
                    model = trained.model if "trained" in locals() and model_info.get("id") == model_id else model_from_payload(model_info["validation"])
                    strategy = LearnedDirection(model)
                    update_state(stage="Simulating model-driven paper trades", current_pair=pair, current_timeframe=timeframe, current_strategy=strategy.name, current_test={"id": test_id, "bars": 0, "total_bars": len(data.candles) - split})

                    def progress(values, _pair=pair, _timeframe=timeframe, _strategy=strategy.name, _id=test_id):
                        pause_checkpoint(journal, _pair, _timeframe, _strategy, "OOS simulation", values)
                        update_state(
                            current_pair=_pair,
                            current_timeframe=_timeframe,
                            current_strategy=_strategy,
                            active_test={"id": _id, **json_value(values)},
                            stage="Simulating model-driven paper trades",
                        )

                    costs = costs_for(pair)
                    simulated = run_report(
                        data,
                        strategy,
                        costs,
                        STARTING_BALANCE,
                        RISK_PER_TRADE,
                        execution=ExecutionAssumptions(max_positions=1, warmup_bars=30, context_lookback_bars=128),
                        start_index=split,
                        evidence_source="OOS",
                        progress_callback=progress,
                    )
                    result = result_from_report(test_id, pair, timeframe, strategy.name, version, model_info, simulated, data, split)
                    persist_result(journal, result, simulated)
                    report["tests"].append(result)
                    completed.add(test_id)
                    write_json(RESULTS_PATH, report)
                    update_state(completed_tests=len(completed), current_trade_count=len(simulated.trades), active_test=None, stage="Model OOS simulation completed")
            if timeframe != "M1":
                for strategy_name in BENCHMARKS:
                    test_id = experiment_id("BENCHMARK", pair, timeframe, strategy_name)
                    if test_id in completed:
                        continue
                    strategy = benchmarks.get(strategy_name)
                    update_state(stage="Running strategy benchmark paper simulation", current_pair=pair, current_timeframe=timeframe, current_strategy=strategy_name, active_test={"id": test_id, "bars": 0, "total_bars": len(data.candles) - split})

                    def progress(values, _pair=pair, _timeframe=timeframe, _strategy=strategy_name, _id=test_id):
                        pause_checkpoint(journal, _pair, _timeframe, _strategy, "benchmark simulation", values)
                        update_state(
                            current_pair=_pair,
                            current_timeframe=_timeframe,
                            current_strategy=_strategy,
                            active_test={"id": _id, **json_value(values)},
                            stage="Running strategy benchmark paper simulation",
                        )

                    simulated = run_report(
                        data,
                        strategy,
                        costs_for(pair),
                        STARTING_BALANCE,
                        RISK_PER_TRADE,
                        execution=ExecutionAssumptions(max_positions=1, warmup_bars=30, context_lookback_bars=128),
                        start_index=split,
                        evidence_source="OOS",
                        progress_callback=progress,
                    )
                    result = result_from_report(test_id, pair, timeframe, strategy_name, version, None, simulated, data, split)
                    persist_result(journal, result, simulated)
                    report["tests"].append(result)
                    completed.add(test_id)
                    write_json(RESULTS_PATH, report)
                    update_state(completed_tests=len(completed), current_trade_count=len(simulated.trades), active_test=None, stage="Strategy benchmark completed")
        report["status"] = "COMPLETED"
        report["completed_at"] = utc_now()
        report["summary"] = summarize(report["tests"])
        write_json(RESULTS_PATH, report)
        update_state(status="COMPLETED", stage="All chronological models and OOS paper tests completed", completed_tests=len(completed), total_tests=total_tests, active_test=None)
        journal.record(
            Entity.AUDIT,
            datetime.now(UTC),
            {"event": "training_arena_phase_completed", "campaign_id": report["campaign_id"], "summary": report["summary"]},
            record_id=str(uuid5(NAMESPACE_URL, "astra-training-arena-phase-completed:v1")),
        )
        total_tests = 48 + sum(1 for item in report["models"] if item.get("status") == "TRAINED")
        update_state(total_tests=total_tests, completed_tests=len(completed))
    finally:
        repository.close()


def model_from_payload(payload: dict) -> Model:
    values = dict(payload)
    for key in ("trained_at", "training_end", "validation_start"):
        values[key] = datetime.fromisoformat(values[key])
    for key in ("means", "scales", "coefficients", "feature_schema"):
        values[key] = tuple(values[key])
    return Model(**values)


def costs_for(pair: str) -> BacktestCosts:
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    return BacktestCosts(SPREAD_PIPS * pip, SLIPPAGE_PIPS * pip, 0.0, 1.0, "bid")


def persist_result(journal: Journal, result: dict, report) -> None:
    pair, timeframe, strategy = result["pair"], result["timeframe"], result["strategy"]
    raw_available_at = result["test_end"]
    available_at = raw_available_at if isinstance(raw_available_at, datetime) else datetime.fromisoformat(str(raw_available_at))
    journal.record(
        Entity.BACKTEST,
        available_at,
        {"source": "TRAINING_ARENA_PAPER", "experiment_id": result["id"], "dataset_version": result["dataset_version"], "metrics": result["metrics"], "trade_count": len(report.trades), "assumptions": result["assumptions"]},
        pair=pair, timeframe=timeframe, strategy=strategy, record_id=str(uuid5(NAMESPACE_URL, "astra-arena-backtest:" + result["id"])),
    )
    journal.record(
        Entity.METRICS,
        available_at,
        {"source": "TRAINING_ARENA_PAPER", "experiment_id": result["id"], "dataset_version": result["dataset_version"], "metrics": result["metrics"]},
        pair=pair, timeframe=timeframe, strategy=strategy, record_id=str(uuid5(NAMESPACE_URL, "astra-arena-metrics:" + result["id"])),
    )
    journal_trade_rows(journal, result["id"], result["dataset_version"], report)


def summarize(results: list[dict]) -> dict:
    pnls = [float(item.get("net_pnl", 0)) for item in results]
    trades = [trade for item in results for trade in item.get("trades", [])]
    trade_pnls = [
        float(trade.get("pnl", float(trade.get("pnl_per_unit", 0) or 0) * float(trade.get("size", 0) or 0)) or 0)
        for trade in trades
    ]
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl < 0]
    return {
        "completed_test_accounts": len(results),
        "starting_balance_per_test": STARTING_BALANCE,
        "aggregate_starting_balance": STARTING_BALANCE * len(results),
        "aggregate_net_pnl_across_independent_tests": sum(pnls),
        "aggregate_ending_equity_across_independent_tests": STARTING_BALANCE * len(results) + sum(pnls),
        "closed_trade_count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "risk_note": "Aggregate P&L sums independent test accounts; it is not one deployable portfolio or a profitability claim.",
    }


def main() -> int:
    ARENA_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        if not claim_worker():
            print("Training Arena phase worker is already running.", flush=True)
            return 0
        try:
            while True:
                research = read_json(RESEARCH_STATE, {}) or {}
                if research.get("status") == "COMPLETED":
                    run_phase()
                    return 0
                if research.get("status") == "FAILED":
                    update_state(status="WAITING_FOR_RESEARCH", stage="Research evaluation failed; resume it before the next phase can start", current_research="FAILED")
                else:
                    update_state(status="WAITING_FOR_RESEARCH", stage="Waiting for the current Research Sandbox evaluation to complete", current_research=research.get("status", "NOT_STARTED"))
                time.sleep(15)
        except Exception as error:  # noqa: BLE001 - persist a safe failure state at the worker boundary.
            update_state(status="FAILED", stage="Training Arena phase stopped safely", last_error=type(error).__name__)
            traceback.print_exc(file=sys.stderr)
            print(f"Training Arena phase failed: {type(error).__name__}", file=sys.stderr, flush=True)
            return 1
        finally:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
