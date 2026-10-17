"""Loopback-only research application; all sandbox data stays in its own local journal."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .adapters import SQLiteRepository
from .backtest import BacktestCosts, ExecutionAssumptions, run_report
from .config import Settings
from .data import DataValidationError
from .history import PAIRS, TIMEFRAMES, load_csv, missing_candles
from .repositories import Entity, Journal, json_value
from .strategies import SPECS, StrategyRegistry

MAX_BODY = 5 * 1024 * 1024
MAX_CANDLES = 4000
MODES = ("analyze", "backtest", "arena", "walk_forward", "robustness", "paper")


class SandboxError(ValueError):
    """Actionable message safe to show in the local application."""


def number(values, key, default, low, high):
    try:
        value = float(values.get(key, default))
    except (TypeError, ValueError):
        raise SandboxError(
            f"{key.replace('_', ' ').capitalize()} must be a number."
        ) from None
    if not isfinite(value) or not low <= value <= high:
        raise SandboxError(
            f"{key.replace('_', ' ').capitalize()} must be between {low:g} and {high:g}."
        )
    return value


class Sandbox:
    def __init__(self, workspace: str | Path):
        self.root = Path(workspace).resolve()
        self.data_dir = self.root / "datasets"
        self.run_dir = self.root / "runs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="astra-sandbox"
        )
        self.jobs: dict[str, dict] = {}
        self.datasets: dict[str, dict] = {}
        self._research_summary_mtime: tuple | None = None
        self._research_summary_cache: dict | None = None
        for path in self.data_dir.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if (
                    item["id"] == path.stem
                    and (self.data_dir / (path.stem + ".csv")).is_file()
                ):
                    self.datasets[item["id"]] = item
            except (ValueError, KeyError, TypeError):
                continue

    def close(self):
        self._worker.shutdown(wait=True)

    def state(self):
        return {
            "token": self.token,
            "datasets": list(self.datasets.values()),
            "runs": self.history(),
            "strategies": [
                {"name": s.name, **json_value(SPECS[s.name])}
                for s in StrategyRegistry().all()
            ],
            "pairs": PAIRS,
            "timeframes": tuple(TIMEFRAMES),
            "max_candles": MAX_CANDLES,
            "mode": "local paper sandbox",
            "version": "0.1.0",
        }

    def curated_research_dir(self):
        return (
            Path(__file__).resolve().parents[1] / "data" / "astra-training-four-pairs"
        )

    def curated_research_status(self):
        path = self.curated_research_dir() / "research-progress.json"
        if not path.is_file():
            return {"status": "NOT_STARTED", "completed": 0, "total": 0}
        status = json.loads(path.read_text(encoding="utf-8"))
        research_dir = self.curated_research_dir()
        control = research_dir / "research-control.json"
        if status.get("status") == "RUNNING" and control.is_file():
            try:
                pending = json.loads(control.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pending = {}
            if pending.get("action") == "pause":
                status = status | {"status": "PAUSE_REQUESTED"}
        elif (
            status.get("status") == "PAUSED"
            and not control.is_file()
            and self._evaluation_is_active(research_dir / "research-evaluation.lock")
        ):
            status = status | {
                "status": "RUNNING",
                "stage": "Resuming evaluation",
            }
        return status

    @staticmethod
    def _evaluation_is_active(lock_path):
        if not lock_path.is_file():
            return False
        try:
            handle = lock_path.open("r+")
        except PermissionError:
            return True
        except OSError:
            return False
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return True
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return True
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
        finally:
            handle.close()

    def curated_research_control(self, action):
        if action not in {"pause", "resume"}:
            raise ValueError("Research action must be pause or resume.")
        control = self.curated_research_dir() / "research-control.json"
        if action == "pause":
            control.write_text('{"action":"pause"}\n', encoding="utf-8")
        else:
            control.unlink(missing_ok=True)
        return {"action": action, "requested": True}

    def curated_research_results(self):
        research_dir = self.curated_research_dir()
        path = research_dir / "research-results.json"
        journal_path = research_dir / "research-journal.sqlite3"
        if not path.is_file() and not journal_path.is_file():
            return {"status": "NOT_STARTED", "items": []}
        modified = (
            path.stat().st_mtime_ns if path.is_file() else 0,
            journal_path.stat().st_mtime_ns if journal_path.is_file() else 0,
            journal_path.stat().st_size if journal_path.is_file() else 0,
        )
        if modified == self._research_summary_mtime and self._research_summary_cache:
            return self._research_summary_cache
        report = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"status": "RUNNING", "items": []}
        )
        rows = []
        seen = set()
        for index, item in enumerate(report.get("items", ())):
            if item.get("kind") in ("BACKTEST", "OOS_HOLDOUT", "COST_STRESS"):
                rows.append(
                    {
                        key: item.get(key)
                        for key in (
                            "kind",
                            "pair",
                            "segment",
                            "timeframe",
                            "strategy",
                            "candles",
                            "metrics",
                        )
                    }
                    | {"id": index, "trade_count": len(item.get("trades", ()))}
                )
                seen.add(
                    (
                        item.get("kind"),
                        item.get("pair"),
                        item.get("segment"),
                        item.get("timeframe"),
                        item.get("strategy", ""),
                    )
                )
            elif item.get("kind") == "WALK_FORWARD":
                rows.append(
                    {
                        key: item.get(key)
                        for key in ("kind", "pair", "segment", "timeframe", "strategy")
                    }
                    | {
                        "id": index,
                        "folds": [
                            {
                                "fold": f.get("fold"),
                                "test_start": f.get("test_start"),
                                "test_end": f.get("test_end"),
                                "metrics": f.get("metrics"),
                                "trade_count": len(f.get("trades", ())),
                            }
                            for f in item.get("folds", ())
                        ],
                    }
                )
                seen.add(
                    (
                        item.get("kind"),
                        item.get("pair"),
                        item.get("segment"),
                        item.get("timeframe"),
                        item.get("strategy", ""),
                    )
                )
            else:
                rows.append(
                    {
                        key: item.get(key)
                        for key in (
                            "kind",
                            "pair",
                            "segment",
                            "timeframe",
                            "strategy",
                            "status",
                            "reason",
                            "sample_count",
                        )
                    }
                    | {"id": index}
                )
        if journal_path.is_file():
            store = SQLiteRepository(str(journal_path))
            try:
                for entity, default_kind in (
                    (Entity.BACKTEST, "BACKTEST"),
                    (Entity.METRICS, "OOS_HOLDOUT"),
                    (Entity.ROBUSTNESS, "COST_STRESS"),
                    (Entity.WALK_FORWARD, "WALK_FORWARD"),
                ):
                    for record in store.query(entity):
                        payload = record.payload
                        parts = payload.get("experiment_key", "").split(":", 3)
                        if len(parts) != 4:
                            continue
                        pair, segment, timeframe, strategy = parts
                        kind = payload.get("source", default_kind)
                        identity = (kind, pair, segment, timeframe, strategy)
                        if identity in seen:
                            continue
                        row = {
                            "id": "j-" + record.record_id,
                            "kind": kind,
                            "pair": pair,
                            "segment": segment,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "journal_only": True,
                        }
                        if kind == "WALK_FORWARD":
                            row["folds"] = [
                                {
                                    "fold": fold.get("fold"),
                                    "test_start": fold.get("test_start"),
                                    "test_end": fold.get("test_end"),
                                    "metrics": fold.get("metrics", {}),
                                    "trade_count": fold.get("metrics", {}).get(
                                        "trade_count", 0
                                    ),
                                }
                                for fold in payload.get("folds", ())
                            ]
                        else:
                            row["metrics"] = payload.get("metrics", {})
                            row["trade_count"] = payload.get("trade_count", 0)
                        rows.append(row)
                        seen.add(identity)
            finally:
                store.close()
        result = {
            "status": report.get("status", "RUNNING"),
            "started_at": report.get("started_at"),
            "completed_at": report.get("completed_at"),
            "scope": report.get("scope", {}),
            "assumptions": report.get("assumptions", {}),
            "profitability": report.get("profitability", "UNPROVEN"),
            "limitations": report.get("limitations", ()),
            "errors": report.get("errors", ()),
            "items": rows,
        }
        self._research_summary_mtime = modified
        self._research_summary_cache = result
        return result

    def curated_research_trades(self, identity):
        if identity.startswith("j-"):
            record_id = identity[2:]
            research_journal = self.curated_research_dir() / "research-journal.sqlite3"
            if not research_journal.is_file():
                raise SandboxError("This research result could not be found.")
            store = SQLiteRepository(str(research_journal))
            try:
                for entity, kind in (
                    (Entity.BACKTEST, "BACKTEST"),
                    (Entity.METRICS, "OOS_HOLDOUT"),
                    (Entity.ROBUSTNESS, "COST_STRESS"),
                    (Entity.WALK_FORWARD, "WALK_FORWARD"),
                ):
                    for record in store.query(entity):
                        if record.record_id != record_id:
                            continue
                        payload = record.payload
                        pair, segment, timeframe, strategy = payload.get(
                            "experiment_key",
                            ":::",
                        ).split(":", 3)
                        return {
                            "kind": payload.get("source", kind),
                            "pair": pair,
                            "segment": segment,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "metrics": payload.get("metrics"),
                            "folds": payload.get("folds", ()),
                            "trades": (),
                            "journal_only": True,
                        }
            finally:
                store.close()
            raise SandboxError("This research result could not be found.")
        try:
            index = int(identity)
        except (TypeError, ValueError):
            raise SandboxError("This research result could not be found.") from None
        path = self.curated_research_dir() / "research-results.json"
        if not path.is_file():
            raise SandboxError("No curated research has been saved yet.")
        report = json.loads(path.read_text(encoding="utf-8"))
        items = report.get("items", ())
        if index < 0 or index >= len(items):
            raise SandboxError("This research result could not be found.")
        return items[index]

    def upload(self, body):
        pair, timeframe = str(body.get("pair", "")), str(body.get("timeframe", ""))
        csv_text = body.get("csv", "")
        if not isinstance(csv_text, str) or not csv_text.strip():
            raise SandboxError("Choose a CSV file or paste your candle data first.")
        if len(csv_text.encode("utf-8")) > MAX_BODY:
            raise SandboxError(
                "This file is too large. Upload a CSV smaller than 5 MB."
            )
        identity = hashlib.sha256((pair + timeframe + csv_text).encode()).hexdigest()[
            :24
        ]
        path = self.data_dir / (identity + ".csv")
        path.write_text(csv_text, encoding="utf-8")
        try:
            data = load_csv(path, pair, timeframe)
            if len(data.candles) > MAX_CANDLES:
                raise SandboxError(
                    f"Use a window of up to {MAX_CANDLES:,} candles for interactive tests."
                )
        except (ValueError, OSError):
            if identity not in self.datasets:
                path.unlink(missing_ok=True)
            raise
        name = (
            str(body.get("name", "pasted-candles.csv"))
            .replace("\\", "/")
            .rsplit("/", 1)[-1][:120]
        )
        summary = {
            "id": identity,
            "name": name,
            "pair": pair,
            "timeframe": timeframe,
            "count": len(data.candles),
            "start": data.candles[0].timestamp.isoformat(),
            "end": data.candles[-1].timestamp.isoformat(),
            "gaps": len(missing_candles(data)),
        }
        (self.data_dir / (identity + ".json")).write_text(
            json.dumps(summary), encoding="utf-8"
        )
        with self._lock:
            self.datasets[identity] = summary
        return summary

    def data(self, identity):
        summary = self.datasets.get(identity)
        if not summary:
            raise SandboxError("Choose an imported dataset first.")
        return load_csv(
            self.data_dir / (identity + ".csv"), summary["pair"], summary["timeframe"]
        )

    def preview(self, identity):
        data = self.data(identity)
        return {
            "summary": self.datasets[identity],
            "candles": json_value(data.candles),
            "gaps": json_value(missing_candles(data)),
        }

    def history(self):
        result = []
        for path in sorted(
            self.run_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:30]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                result.append(
                    {
                        k: value[k]
                        for k in (
                            "id",
                            "mode",
                            "created_at",
                            "dataset",
                            "settings",
                            "headline",
                        )
                    }
                )
            except (ValueError, KeyError):
                continue
        return result

    def result(self, identity):
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise SandboxError("This saved run could not be found.")
        path = self.run_dir / (identity + ".json")
        if not path.is_file():
            raise SandboxError("This saved run could not be found.")
        return json.loads(path.read_text(encoding="utf-8"))

    def submit(self, body):
        if body.get("mode") not in MODES:
            raise SandboxError("Choose a supported experiment.")
        self.data(body.get("dataset_id"))
        self.options(body)
        with self._lock:
            if any(j["status"] in ("queued", "running") for j in self.jobs.values()):
                raise SandboxError(
                    "An experiment is already running. Wait for it to finish."
                )
            identity = uuid4().hex
            self.jobs[identity] = {
                "id": identity,
                "status": "queued",
                "message": "Preparing your experiment",
            }
        self._worker.submit(self._run_job, identity, dict(body))
        return {"id": identity, "status": "queued"}

    def job(self, identity):
        with self._lock:
            if identity not in self.jobs:
                raise SandboxError(
                    "This experiment is no longer active. Check saved runs."
                )
            return dict(self.jobs[identity])

    def _run_job(self, identity, body):
        with self._lock:
            self.jobs[identity].update(
                status="running", message="Running Astra on your candle history"
            )
        try:
            result = self.execute(body, identity)
            (self.run_dir / (identity + ".json")).write_text(
                json.dumps(result, allow_nan=False), encoding="utf-8"
            )
            with self._lock:
                self.jobs[identity].update(
                    status="complete", message="Experiment complete", result=result
                )
        except (ValueError, KeyError) as error:
            with self._lock:
                self.jobs[identity].update(status="failed", message=str(error))
        except Exception:  # noqa: BLE001 - keep internal paths and credentials out of the UI.
            with self._lock:
                self.jobs[identity].update(
                    status="failed",
                    message="Astra could not finish this run. Check the dataset and try a smaller window.",
                )

    def options(self, body):
        selected = body.get("strategies", ["ema_trend"])
        names = {s.name for s in StrategyRegistry().all()}
        if (
            not isinstance(selected, list)
            or not selected
            or any(s not in names for s in selected)
        ):
            raise SandboxError("Select at least one of Astra's ten strategies.")
        raw = body.get("settings", {})
        if not isinstance(raw, dict):
            raise SandboxError("Experiment settings must be an object.")
        return {
            "equity": number(raw, "equity", 10000, 1, 1e9),
            "risk_percent": number(raw, "risk_percent", 0.5, 0.01, 1),
            "spread_pips": number(raw, "spread_pips", 1, 0, 100),
            "slippage_pips": number(raw, "slippage_pips", 0.2, 0, 100),
            "commission": number(raw, "commission", 0, 0, 10000),
            "price_value": number(raw, "price_value", 1, 0.000001, 1e9),
            "max_units": number(raw, "max_units", 100000, 0.000001, 1e9),
            "max_positions": int(number(raw, "max_positions", 1, 1, 20)),
            "drawdown_percent": number(raw, "drawdown_percent", 20, 1, 100),
            "train_size": int(number(raw, "train_size", 120, 31, MAX_CANDLES)),
            "test_size": int(number(raw, "test_size", 60, 2, MAX_CANDLES)),
            "kill_switch": bool(raw.get("kill_switch", False)),
        }

    def execute(self, body, identity=None):
        """Compose existing Astra engines. Settings never come from a production .env."""
        data = self.data(body.get("dataset_id"))
        mode = body["mode"]
        from .historical.dataset import require_quality

        require_quality(data)
        opts = self.options(body)
        pip = 0.01 if data.pair.endswith("JPY") else 0.0001
        costs = BacktestCosts(
            opts["spread_pips"] * pip,
            opts["slippage_pips"] * pip,
            opts["commission"],
            opts["price_value"],
        )
        settings = Settings(
            trading_enabled=True,
            persistence_backend="sqlite",
            max_trade_risk_pct=opts["risk_percent"] / 100,
            max_spread=max(costs.spread * 2, pip),
            max_slippage=max(costs.slippage * 2, pip),
            max_position_size=opts["max_units"],
            max_pair_exposure=opts["max_units"],
            max_correlated_exposure=opts["max_units"],
            max_open_positions=opts["max_positions"],
            max_drawdown_pct=opts["drawdown_percent"] / 100,
            kill_switch=opts["kill_switch"],
        )
        registry = StrategyRegistry()
        strategies = [
            registry.get(name) for name in body.get("strategies", ["ema_trend"])
        ]
        store = SQLiteRepository(str(self.root / "research.sqlite3"))
        journal = Journal(store)
        value: Any
        try:
            from .risk import RiskEngine

            if mode == "analyze":
                from .decision import decide
                from .intelligence import detect_regime
                from .market import momentum, multi_timeframe, structure
                from .models import RiskState

                frames = {data.timeframe: data}
                for tf, dataset_id in body.get("timeframes", {}).items():
                    if dataset_id:
                        frames[tf] = self.data(dataset_id)
                mtf = (
                    multi_timeframe(
                        frames,
                        data.candles[-1].timestamp,
                        execution_timeframe=data.timeframe,
                    )
                    if len(frames) > 1
                    else None
                )
                decision = decide(
                    data,
                    strategies,
                    RiskEngine(settings),
                    RiskState(opts["equity"], opts["equity"], opts["equity"]),
                    costs.spread,
                    decision_time=data.candles[-1].timestamp,
                    expected_slippage=costs.slippage,
                    value_per_price_unit=costs.value_per_price_unit,
                    timeframe_context=mtf,
                    journal=journal,
                    execution_cost_per_unit=costs.commission_per_unit
                    + (costs.spread + 2 * costs.slippage) * costs.value_per_price_unit,
                )
                value = {
                    "decision": decision,
                    "market": detect_regime(data),
                    "structure": structure(data),
                    "momentum": momentum(data),
                    "timeframes": mtf,
                }
                headline = decision.action.value.replace("_", " ")
            elif mode == "backtest":
                report = run_report(
                    data,
                    strategies[0],
                    costs,
                    opts["equity"],
                    opts["risk_percent"] / 100,
                    execution=ExecutionAssumptions(max_positions=opts["max_positions"]),
                    risk_engine=RiskEngine(settings),
                    journal=journal,
                )
                value = report
                headline = f"{len(report.trades)} trades · {report.metrics['total_return_pct']:+.2f}%"
            elif mode == "arena":
                from .arena import compete

                evaluations = compete(data, strategies, costs, journal=journal)
                value = {
                    "evaluations": evaluations,
                    "assumption_note": "Arena uses identical 10,000 capital, 0.5% risk and one position for every strategy. Cost controls apply.",
                }
                headline = f"{len(evaluations)} strategies compared"
            elif mode == "walk_forward":
                from .robustness import evaluate_walk_forward

                folds = evaluate_walk_forward(
                    data,
                    strategies[0],
                    opts["train_size"],
                    opts["test_size"],
                    costs,
                    journal=journal,
                )
                if not folds:
                    raise SandboxError(
                        "This dataset is shorter than your training + test windows. Reduce the windows or import more candles."
                    )
                value = {
                    "folds": folds,
                    "assumption_note": "Each fold uses 10,000 capital, 0.5% risk and one position. Cost controls apply.",
                }
                headline = f"{len(folds)} chronological folds"
            elif mode == "robustness":
                from .robustness import robustness_report

                value = robustness_report(data, strategies[0], costs, journal=journal)
                headline = value.state.replace("_", " ")
            elif mode == "paper":
                from .pipeline import replay_paper

                paper_result = replay_paper(
                    data, strategies, settings, costs, opts["equity"], journal=journal
                )
                value = paper_result
                headline = f"{len(paper_result['trades'])} closed · {len(paper_result['positions'])} open"
            else:
                raise SandboxError("Choose a supported experiment.")
            result = json_value(
                {
                    "id": identity or uuid4().hex,
                    "mode": mode,
                    "created_at": datetime.now(UTC),
                    "dataset": self.datasets[body["dataset_id"]],
                    "settings": opts,
                    "strategies": [s.name for s in strategies],
                    "headline": headline,
                    "result": value,
                    "costs": costs,
                    "limitations": "Historical OHLC simulation. Heuristic scores are not probabilities; results do not establish future profitability.",
                }
            )
            journal.record(
                Entity.AUDIT,
                datetime.now(UTC),
                {"sandbox_run": result["id"], "mode": mode, "headline": headline},
            )
            return result
        finally:
            store.close()


class SandboxServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, application):
        self.application = application
        super().__init__(("127.0.0.1", port), SandboxHandler)


class SandboxHandler(BaseHTTPRequestHandler):
    server: SandboxServer

    def log_message(self, format, *args):
        return  # No CSV bodies, credentials or request tokens in logs.

    def respond(self, status, value, content_type="application/json"):
        body = (
            json.dumps(value, allow_nan=False).encode()
            if content_type == "application/json"
            else value
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def allowed(self, mutation=False):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != origin.removeprefix("http://"):
            self.respond(
                403, {"error": "Open Astra using its local 127.0.0.1 address."}
            )
            return False
        if self.headers.get("Origin", origin) != origin:
            self.respond(
                403, {"error": "This request must come from the Astra sandbox."}
            )
            return False
        if mutation and not secrets.compare_digest(
            self.headers.get("X-Astra-Token", ""), self.server.application.token
        ):
            self.respond(403, {"error": "Refresh the sandbox before continuing."})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        app = self.server.application
        try:
            if path == "/api/state":
                self.respond(200, app.state())
            elif path == "/api/curated-research/status":
                self.respond(200, app.curated_research_status())
            elif path == "/api/curated-research/results":
                self.respond(200, app.curated_research_results())
            elif path.startswith("/api/curated-research/trades/"):
                self.respond(200, app.curated_research_trades(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/datasets/"):
                self.respond(200, app.preview(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/jobs/"):
                self.respond(200, app.job(path.rsplit("/", 1)[-1]))
            elif path.startswith("/api/runs/"):
                self.respond(200, app.result(path.rsplit("/", 1)[-1]))
            elif path in (
                "/",
                "/app.js",
                "/styles.css",
                "/research",
                "/research.js",
                "/research.css",
            ):
                name = {"/": "index.html", "/research": "research.html"}.get(
                    path, path[1:]
                )
                mime = {
                    "index.html": "text/html; charset=utf-8",
                    "research.html": "text/html; charset=utf-8",
                    "app.js": "application/javascript",
                    "styles.css": "text/css",
                    "research.js": "application/javascript",
                    "research.css": "text/css",
                }[name]
                self.respond(
                    200,
                    (Path(__file__).parent / "sandbox_web" / name).read_bytes(),
                    mime,
                )
            else:
                self.respond(404, {"error": "Page not found."})
        except (ValueError, KeyError):
            self.respond(404, {"error": "This dataset or saved run was not found."})

    def do_POST(self):
        if not self.allowed(mutation=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.respond(413, {"error": "Upload a CSV smaller than 5 MB."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.respond(415, {"error": "Expected a JSON request."})
                return
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise SandboxError("Expected experiment settings or a CSV upload.")
            if self.path == "/api/datasets":
                self.respond(201, self.server.application.upload(body))
            elif self.path == "/api/runs":
                self.respond(202, self.server.application.submit(body))
            elif self.path == "/api/curated-research/control":
                self.respond(
                    202,
                    self.server.application.curated_research_control(
                        body.get("action")
                    ),
                )
            else:
                self.respond(404, {"error": "Action not found."})
        except (SandboxError, DataValidationError) as error:
            self.respond(400, {"error": str(error)})
        except (ValueError, TypeError, KeyError):
            self.respond(
                400,
                {
                    "error": "Check your CSV format, pair, timeframe and experiment settings."
                },
            )
        except Exception:  # noqa: BLE001 - never reveal internal details to a browser.
            self.respond(
                500,
                {
                    "error": "The local sandbox could not save this request. Check available disk space."
                },
            )


def serve(port=8787, workspace=".sandbox", open_browser=True, page="/"):
    application = Sandbox(workspace)
    server = SandboxServer(port, application)
    url = f"http://127.0.0.1:{server.server_port}"
    launch_url = url if page == "/" else url.rstrip("/") + page
    print(
        f"Astra sandbox: {launch_url}\nPaper research only. Press Ctrl+C to stop.",
        flush=True,
    )
    if open_browser:
        webbrowser.open(launch_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Astra sandbox stopped.")
    finally:
        server.server_close()
        application.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Astra local sandbox")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--workspace", default=".sandbox")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--research", action="store_true")
    args = parser.parse_args()
    serve(
        args.port,
        args.workspace,
        not args.no_browser,
        "/research" if args.research else "/",
    )
