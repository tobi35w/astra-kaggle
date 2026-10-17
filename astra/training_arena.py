"""Separate loopback-only web application for offline paper training runs."""

from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .repositories import Entity

ROOT = Path(__file__).resolve().parents[1]
ARENA_ROOT = ROOT / "data" / "training-arena"
CURATED_HISTORY_ROOT = ARENA_ROOT / "curated-history-h1-h4-v1"
WEB_ROOT = Path(__file__).with_name("training_arena_web")
RESEARCH_ROOT = ROOT / "data" / "astra-training-four-pairs"
DATASET_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
MAX_BODY = 4096
_CACHE_LOCK = threading.Lock()
_JSON_CACHE: dict[str, tuple[int, int, object]] = {}


def read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def read_cached_json(path: Path, fallback=None):
    try:
        stat = path.stat()
        with _CACHE_LOCK:
            cached = _JSON_CACHE.get(str(path))
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                return cached[2]
            value = json.loads(path.read_text(encoding="utf-8"))
            _JSON_CACHE[str(path)] = (stat.st_mtime_ns, stat.st_size, value)
            return value
    except (OSError, json.JSONDecodeError):
        return fallback


def arena_state() -> dict:
    state = read_json(ARENA_ROOT / "arena-state.json", {}) or {}
    results = read_cached_json(ARENA_ROOT / "arena-results.json", {}) or {}
    research = read_json(RESEARCH_ROOT / "research-progress.json", {}) or {}
    report = read_cached_json(RESEARCH_ROOT / "research-results.json", {}) or {}
    manifest = read_json(DATASET_ROOT / "four-pairs-manifest.json", {}) or {}
    quality = read_json(DATASET_ROOT / "quality-report.json", {}) or {}
    curated_history = read_json(CURATED_HISTORY_ROOT / "progress.json", {}) or {}
    curated_report = (
        read_cached_json(CURATED_HISTORY_ROOT / "evaluation-report.json", {}) or {}
    )
    tests = results.get("tests", [])
    # The test list is appended as runs complete. Put the newest completed run
    # first, then its latest trade, so the live UI feed advances after each test
    # even when its historical candle dates precede another run's dates.
    trades = [
        trade
        | {
            "test_id": test.get("id"),
            "pair": test.get("pair"),
            "timeframe": test.get("timeframe"),
            "strategy": test.get("strategy"),
        }
        for test in reversed(tests)
        for trade in reversed(test.get("trades", []))
    ]
    total_pnl = sum(float(test.get("net_pnl", 0) or 0) for test in tests)
    start_total = sum(float(test.get("starting_balance", 0) or 0) for test in tests)
    wins = [
        float(
            t.get(
                "pnl",
                float(t.get("pnl_per_unit", 0) or 0) * float(t.get("size", 0) or 0),
            )
            or 0
        )
        for t in trades
        if float(
            t.get(
                "pnl",
                float(t.get("pnl_per_unit", 0) or 0) * float(t.get("size", 0) or 0),
            )
            or 0
        )
        > 0
    ]
    losses = [
        float(
            t.get(
                "pnl",
                float(t.get("pnl_per_unit", 0) or 0) * float(t.get("size", 0) or 0),
            )
            or 0
        )
        for t in trades
        if float(
            t.get(
                "pnl",
                float(t.get("pnl_per_unit", 0) or 0) * float(t.get("size", 0) or 0),
            )
            or 0
        )
        < 0
    ]
    completed = len(tests)
    latest_test: dict = next(iter(tests), {})
    return {
        "arena": state,
        "campaign": {
            "status": results.get(
                "status", state.get("status", "WAITING_FOR_RESEARCH")
            ),
            "tests": [
                {key: value for key, value in test.items() if key != "trades"}
                for test in tests
            ],
            "models": results.get("models", []),
            "summary": {
                "completed_test_accounts": completed,
                "total_test_accounts": state.get("total_tests", 68),
                "starting_balance_per_test": 10000.0,
                "aggregate_starting_balance": start_total,
                "aggregate_net_pnl": total_pnl,
                "aggregate_ending_equity": start_total + total_pnl,
                "trade_count": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
                "risk_note": "P&L sums separate, independent test accounts; choose a test account to inspect its own balance.",
            },
            "recent_trades": trades,
            "updated_at": state.get("updated_at") or results.get("completed_at"),
            "source_limitations": results.get(
                "source_limitations",
                quality.get("research_readiness", {}).get("requires_followup", []),
            ),
            "assumptions": results.get("assumptions", []),
            "account": {
                "mode": "offline simulated paper account",
                "default_starting_balance": 10000.0,
                "risk_per_trade_pct": 0.5,
                "maximum_open_positions_per_test": 1,
                "spread_pips": 1.0,
                "slippage_pips": 0.2,
                "leverage": "not modeled",
                "currency_conversion": "1 price-unit value assumption; USD account P&L approximation",
            },
            "latest_test": latest_test,
        },
        "research": {
            "status": research.get("status", "NOT_STARTED"),
            "stage": research.get("stage", ""),
            "completed": research.get("completed", 0),
            "total": research.get("total", 0),
            "updated_at": research.get("updated_at"),
            "prior_phase_profitability": report.get("profitability", "UNPROVEN"),
            "prior_phase_result_count": len(report.get("items", [])),
        },
        "curated_history": {
            "campaign_id": curated_history.get("campaign_id"),
            "status": curated_history.get("status", "NOT_STARTED"),
            "completed_tests": curated_history.get("completed_tests", 0),
            "total_tests": curated_history.get("total_tests", 0),
            "current_pair": curated_history.get("current_pair"),
            "current_timeframe": curated_history.get("current_timeframe"),
            "current_window": curated_history.get("current_window"),
            "current_experiment": curated_history.get("current_experiment"),
            "latest_trade_count": curated_history.get("latest_trade_count"),
            "latest_net_return_pct": curated_history.get("latest_net_return_pct"),
            "updated_at": curated_history.get("updated_at"),
            "report": curated_report if curated_report else None,
        },
        "dataset": {
            "source": manifest.get("dataset_id", "Yllvar/fx-historical-data"),
            "rows": manifest.get("rows", 0),
            "files": len(manifest.get("files", [])),
            "pairs": manifest.get("scope", {}).get("symbols", []),
            "timeframes": manifest.get("scope", {}).get("timeframes", []),
            "structural_pass_files": quality.get("structural_pass_files", 0),
            "structural_fail_files": quality.get("structural_fail_files", []),
            "offline_ready": bool(manifest and quality),
            "limitations": quality.get("research_readiness", {}).get(
                "requires_followup", []
            ),
        },
        "journal_counts": {
            entity.value: count_entity(entity)
            for entity in (Entity.MODEL, Entity.BACKTEST, Entity.METRICS, Entity.TRADE)
        },
        "storage_sync": read_json(
            ARENA_ROOT / "supabase-sync-status.json", {"status": "LOCAL SPOOL READY"}
        ),
    }


def count_entity(entity: Entity) -> int:
    path = ARENA_ROOT / "arena-journal.sqlite3"
    if not path.exists():
        return 0
    import sqlite3

    try:
        with sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2
        ) as connection:
            row = connection.execute(
                "SELECT count(*) FROM astra_records WHERE entity = ?", (entity.value,)
            ).fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def serve(
    host: str = "127.0.0.1", port: int = 8788, *, open_browser: bool = False
) -> None:
    ARENA_ROOT.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AstraTrainingArena/1.0"

        def log_message(self, format, *args):
            print("Training Arena: " + (format % args), flush=True)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/state":
                state = arena_state()
                state["control_token"] = token
                body = json.dumps(state, allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            name = "index.html" if route in ("/", "/index.html") else route.lstrip("/")
            if name not in {"index.html", "app.css", "app.js"}:
                self.send_error(404)
                return
            path = WEB_ROOT / name
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            content_type = (
                "text/html; charset=utf-8"
                if name.endswith(".html")
                else "text/css; charset=utf-8"
                if name.endswith(".css")
                else "text/javascript; charset=utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            route = urlsplit(self.path).path
            if route != "/api/control" or self.headers.get("X-Astra-Token") != token:
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY:
                self.send_error(413)
                return
            try:
                payload = json.loads(self.rfile.read(length))
                action = payload.get("action")
            except (ValueError, AttributeError):
                self.send_error(400)
                return
            if action not in {"pause", "resume"}:
                self.send_error(400)
                return
            control = ARENA_ROOT / "arena-control.json"
            with lock:
                control.write_text(
                    json.dumps(
                        {
                            "action": action,
                            "requested_at": datetime.now(UTC).isoformat(),
                        }
                    ),
                    encoding="utf-8",
                )
            body = json.dumps({"ok": True, "action": action}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"Astra Training Arena available at http://{host}:{port}/", flush=True)
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
