"""Evaluation Desk: a loopback-only live view of Astra's tests and evaluations.

Mirrors the Research Sandbox / Training Arena patterns: 127.0.0.1 binding, Host
and Origin validation, a control token for mutations, strict static-file
whitelisting, and sanitized errors. It only reads local JSON artifacts and can
launch the offline test suite; it never contacts a broker or a feed.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from .project_status import project_assessment
from .verification import SuiteRunner

ROOT = Path(__file__).resolve().parents[1]
DESK_ROOT = ROOT / "data" / "progress-desk"
RESEARCH_ROOT = ROOT / "data" / "astra-training-four-pairs"
TRAINING_ROOT = ROOT / "data" / "training-arena"
CURATED_HISTORY_ROOT = TRAINING_ROOT / "curated-history-h1-h4-v1"
ACQUISITION_ROOT = ROOT / "data" / "astra-continuous-2026-tail"
REHEARSAL_ROOT = ROOT / "data" / "paper-rehearsal-2026"
QLEAP_ROOT = ROOT / "data" / "future-phase" / "qleap-fx-majors-tick-ohlcv"
QLEAP_SOURCE_ROOT = ROOT / "data" / "astra-qleap-source"
WEB_ROOT = Path(__file__).with_name("progress_desk_web")
MAX_BODY = 4096
STALE_AFTER_SECONDS = 15 * 60
_TAIL_LOCK = threading.Lock()
_TAIL_CACHE: dict[str, tuple[tuple[int, int], object]] = {}


def read_json(path: Path, fallback=None):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_invalid_number,
            parse_float=_json_float,
        )
    except (OSError, ValueError, UnicodeError):
        return fallback


def _invalid_number(value):
    raise ValueError("Non-finite JSON number")


def _json_float(value):
    number = float(value)
    if not isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def read_cached_json(path: Path, fallback=None):
    """mtime/size-keyed cache so large result files are parsed only when changed."""
    try:
        stat = path.stat()
    except OSError:
        return fallback
    key = (stat.st_mtime_ns, stat.st_size)
    with _TAIL_LOCK:
        cached = _TAIL_CACHE.get(str(path))
        if cached and cached[0] == key:
            return cached[1]
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_invalid_number,
            parse_float=_json_float,
        )
    except (OSError, ValueError, UnicodeError):
        return fallback
    with _TAIL_LOCK:
        _TAIL_CACHE[str(path)] = (key, value)
    return value


def _artifact(path: Path, *, cached: bool = False) -> tuple[dict, list[str]]:
    value = (read_cached_json if cached else read_json)(path)
    if isinstance(value, dict):
        return value, []
    if path.is_file():
        return {}, [f"{path.name} is unreadable or is not a JSON object."]
    return {}, []


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _rows(value) -> list[dict]:
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


def _messages(value) -> list[str]:
    return [str(item) for item in value[:6]] if isinstance(value, list) else []


def _finite(value, default=None):
    try:
        number = float(value)
        return number if not isinstance(value, bool) and isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _count(value) -> int:
    number = _finite(value, 0)
    return max(0, int(number))


def _progress(completed, total, recorded_status: str) -> dict:
    done, planned = _count(completed), _count(total)
    invalid = any(
        _finite(value) is None or _finite(value) != normalized
        for value, normalized in ((completed, done), (total, planned))
    )
    return {
        "completed": done,
        "total": planned,
        "progress_fraction": min(1.0, done / planned) if planned else None,
        "count_inconsistent": invalid
        or done > planned
        or (recorded_status == "COMPLETED" and planned > 0 and done != planned),
    }


def _timestamp(value) -> datetime | None:
    if isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value)
            return stamp if stamp.tzinfo is not None else None
        except (ValueError, OverflowError):
            pass
    return None


def _snapshot_meta(
    progress: dict, results: dict, issues: list[str], *, now: datetime | None = None
) -> dict:
    """Describe recorded evidence without claiming that a worker is still alive.

    Fifteen minutes without an update makes an active snapshot stale, not failed:
    a long experiment can legitimately take longer. Historical completed results
    keep their recorded verdict even when their timestamp is old.
    """
    available = bool(progress or results)
    recorded = str(progress.get("status") or results.get("status") or "NOT_STARTED")
    updated = progress.get("updated_at") or results.get("completed_at")
    age = None
    freshness = "UNKNOWN" if available else "MISSING"
    if (stamp := _timestamp(updated)) is not None:
        age = ((now or datetime.now(UTC)) - stamp).total_seconds()
        if age >= 0:
            freshness = "STALE" if age > STALE_AFTER_SECONDS else "CURRENT"
    status = recorded
    note = "Saved local result; worker activity has not been checked."
    if recorded in {"RUNNING", "IN_PROGRESS", "ACTIVITY_RECORDED"}:
        status = "STALE" if freshness == "STALE" else "RECORDED_ACTIVE"
        note = (
            "The last activity update is over 15 minutes old; worker activity is unknown."
            if freshness == "STALE"
            else "A saved update records activity; this does not verify a running worker."
        )
    elif recorded in {"COMPLETE", "COMPLETED"}:
        note = "The saved run reports completion; this is historical evidence."
    elif not available:
        status = "UNAVAILABLE" if issues else "NOT_STARTED"
        note = (
            "The saved state could not be read."
            if issues
            else "No saved run is available."
        )
    return {
        "available": available,
        "status": status,
        "recorded_status": recorded,
        "status_note": note,
        "updated_at": updated if isinstance(updated, str) else None,
        "freshness": {
            "state": freshness,
            "updated_at": updated if isinstance(updated, str) else None,
            "age_seconds": round(age, 1) if age is not None and age >= 0 else None,
            "threshold_seconds": STALE_AFTER_SECONDS,
        },
        "artifact_issues": issues,
    }


def _research_state(*, now: datetime | None = None) -> dict:
    progress, progress_issues = _artifact(RESEARCH_ROOT / "research-progress.json")
    results, result_issues = _artifact(
        RESEARCH_ROOT / "research-results.json", cached=True
    )
    meta = _snapshot_meta(progress, results, progress_issues + result_issues, now=now)
    items = _rows(results.get("items"))
    kinds: dict[str, int] = {}
    for item in items:
        kind = str(item.get("kind") or "OTHER")
        kinds[kind] = kinds.get(kind, 0) + 1
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    return {
        **meta,
        **_progress(completed, total, meta["recorded_status"]),
        "profitability": results.get("profitability", "UNPROVEN"),
        "experiments": len(items),
        "experiment_kinds": kinds,
        "errors": len(_list(results.get("errors"))),
        "limitations": _messages(results.get("limitations")),
        "current": {
            key: progress.get(key)
            for key in ("stage", "pair", "timeframe", "segment", "strategy")
        },
    }


def _acquisition_state(*, now: datetime | None = None) -> dict:
    manifest_path = ACQUISITION_ROOT / "manifests" / "dukascopy_manifest.json"
    manifest, manifest_issues = _artifact(manifest_path)
    downloads = _mapping(manifest.get("downloads"))
    by_symbol: dict[str, dict[str, int]] = {}
    latest_day: dict[str, str] = {}
    for key, unit in downloads.items():
        symbol, _, day = key.partition(":")
        bucket = by_symbol.setdefault(symbol, {})
        status = str(_mapping(unit).get("status") or "unknown")
        bucket[status] = bucket.get(status, 0) + 1
        if status == "downloaded" and (
            symbol not in latest_day or day > latest_day[symbol]
        ):
            latest_day[symbol] = day
    summary, summary_issues = _artifact(ACQUISITION_ROOT / "acquisition-summary.json")
    latest_id = manifest.get("latest_run_id")
    latest_run = (
        _mapping(_mapping(manifest.get("runs")).get(latest_id))
        if isinstance(latest_id, str)
        else {}
    )
    summary_updated = summary.get("acquired_at") or summary.get("completed_at")
    manifest_updated = (
        latest_run.get("completed_at")
        or manifest.get("updated_at")
        or latest_run.get("started_at")
    )
    summary_stamp, manifest_stamp = (
        _timestamp(summary_updated),
        _timestamp(manifest_updated),
    )
    use_manifest = bool(latest_run or downloads) and (
        not summary
        or (latest_run and summary_stamp is None)
        or (
            manifest_stamp is not None
            and summary_stamp is not None
            and manifest_stamp > summary_stamp
        )
    )
    selected = latest_run if use_manifest else summary
    snapshot = (
        {
            "status": selected.get("status")
            or ("ACTIVITY_RECORDED" if downloads else "NOT_STARTED"),
            "updated_at": manifest_updated
            if use_manifest
            else summary_updated or manifest_updated,
        }
        if manifest or summary
        else {}
    )
    meta = _snapshot_meta(snapshot, summary, manifest_issues + summary_issues, now=now)
    return {
        **meta,
        "running": False,
        "units": len(downloads),
        "by_symbol": by_symbol,
        "latest_days": latest_day,
        "summary_status": meta["status"],
        "symbols": _mapping(selected.get("symbols")),
        "run_id": selected.get("run_id"),
        "snapshot_source": "manifest-run"
        if use_manifest and latest_run
        else "manifest"
        if use_manifest
        else "acquisition-summary",
        "manifest_updated_at": manifest.get("updated_at"),
    }


def _training_state(*, now: datetime | None = None) -> dict:
    progress, progress_issues = _artifact(TRAINING_ROOT / "arena-state.json")
    results, result_issues = _artifact(
        TRAINING_ROOT / "arena-results.json", cached=True
    )
    meta = _snapshot_meta(progress, results, progress_issues + result_issues, now=now)
    tests = _rows(results.get("tests"))
    pnls = [
        value for row in tests if (value := _finite(row.get("net_pnl"))) is not None
    ]
    active = _mapping(progress.get("active_test"))
    return {
        **meta,
        **_progress(
            progress.get("completed_tests", len(tests)),
            progress.get("total_tests", len(tests)),
            meta["recorded_status"],
        ),
        "saved_results": len(tests),
        "current": {
            "stage": progress.get("stage"),
            "pair": active.get("pair") or progress.get("current_pair"),
            "timeframe": active.get("timeframe") or progress.get("current_timeframe"),
            "strategy": active.get("strategy") or progress.get("current_strategy"),
            "experiment": _mapping(progress.get("current_test")).get("id"),
            "window": None,
        },
        "summary": {
            "trade_count": sum(
                _count(row.get("trade_count", len(_rows(row.get("trades")))))
                for row in tests
            ),
            "net_pnl": _finite(sum(pnls)) if pnls else None,
            "positive_tests": sum(value > 0 for value in pnls),
            "negative_tests": sum(value < 0 for value in pnls),
            "model_count": len(_rows(results.get("models"))),
            "units_note": "P&L sums independent test accounts with a 1 price-unit value assumption; it is not portfolio P&L.",
        },
        "limitations": _messages(
            results.get("source_limitations")
            or _mapping(progress.get("dataset")).get("limitations")
        ),
    }


def _curated_history_state(*, now: datetime | None = None) -> dict:
    progress, progress_issues = _artifact(CURATED_HISTORY_ROOT / "progress.json")
    report, report_issues = _artifact(
        CURATED_HISTORY_ROOT / "evaluation-report.json", cached=True
    )
    meta = _snapshot_meta(progress, report, progress_issues + report_issues, now=now)
    groups = []
    scenarios: dict[str, dict] = {}
    for key, value in _mapping(report.get("summary")).items():
        parts = key.split("|")
        if len(parts) != 4 or not isinstance(value, dict):
            continue
        strategy, pair, timeframe, scenario = parts
        group = {
            "strategy": strategy,
            "pair": pair,
            "timeframe": timeframe,
            "scenario": scenario,
            **{
                name: _count(value.get(name))
                for name in (
                    "window_runs",
                    "positive_windows",
                    "negative_windows",
                    "flat_windows",
                    "total_closed_trades",
                    "runs_below_30_trades",
                )
            },
            "median_return_pct": _finite(value.get("median_return_pct")),
            "mean_return_pct": _finite(value.get("mean_return_pct")),
        }
        groups.append(group)
        bucket = scenarios.setdefault(
            scenario,
            {
                "tests": 0,
                "positive_tests": 0,
                "negative_tests": 0,
                "flat_tests": 0,
                "trade_count": 0,
                "weighted_return": 0.0,
                "return_samples": 0,
            },
        )
        for target, source in (
            ("tests", "window_runs"),
            ("positive_tests", "positive_windows"),
            ("negative_tests", "negative_windows"),
            ("flat_tests", "flat_windows"),
            ("trade_count", "total_closed_trades"),
        ):
            bucket[target] += group[source]
        if group["mean_return_pct"] is not None:
            bucket["weighted_return"] += group["mean_return_pct"] * group["window_runs"]
            bucket["return_samples"] += group["window_runs"]
    for bucket in scenarios.values():
        samples = bucket.pop("return_samples")
        weighted = bucket.pop("weighted_return")
        bucket["mean_return_pct"] = _finite(weighted / samples) if samples else None
    saved = sum(bucket["tests"] for bucket in scenarios.values())
    return {
        **meta,
        **_progress(
            progress.get("completed_tests", saved),
            progress.get("total_tests", report.get("actual_total_test_count", 0)),
            meta["recorded_status"],
        ),
        "saved_results": saved,
        "current": {
            "stage": progress.get("stage"),
            "pair": progress.get("current_pair"),
            "timeframe": progress.get("current_timeframe"),
            "window": progress.get("current_window"),
            "experiment": progress.get("current_experiment"),
            "strategy": progress.get("current_strategy"),
        },
        "summary": {
            "trade_count": sum(bucket["trade_count"] for bucket in scenarios.values()),
            "net_pnl": None,
            "positive_tests": sum(
                bucket["positive_tests"] for bucket in scenarios.values()
            ),
            "negative_tests": sum(
                bucket["negative_tests"] for bucket in scenarios.values()
            ),
            "model_count": len(_rows(report.get("models"))),
            "groups": groups,
            "scenarios": scenarios,
            "units_note": "Returns describe independent historical test windows. Mean returns are weighted by test count; native-currency P&L is not summed across pairs.",
        },
        "limitations": _messages(report.get("limits")),
    }


def _rehearsal_state() -> dict:
    summary = _mapping(read_json(REHEARSAL_ROOT / "summary.json"))
    if not summary:
        return {"available": False}
    return {"available": True, **summary}


def _qleap_state() -> dict:
    audit = _mapping(read_cached_json(QLEAP_ROOT / "astra-audit.json"))
    if not audit:
        return {"available": False}
    pairs = {}
    for symbol, item in _mapping(audit.get("pairs")).items():
        if not isinstance(item, dict):
            continue
        counts = [
            len(item[name]) if isinstance(item.get(name), list) else None
            for name in ("non_monotonic_files", "duplicate_timestamp_files")
        ] + [
            _finite(item.get(name))
            for name in (
                "non_positive_or_nan_prices",
                "ohlc_inconsistent",
                "negative_volume",
            )
        ]
        known = all(
            value is not None and value >= 0 and int(value) == value for value in counts
        )
        issues = sum(int(value) for value in counts) if known else None
        pairs[symbol] = {
            "files": item.get("files"),
            "rows": item.get("rows"),
            "range": [item.get("first_timestamp"), item.get("last_timestamp")],
            "non_monotonic_files": counts[0],
            "duplicate_timestamp_files": counts[1],
            "structural_issues": issues,
            "structural_status": "UNKNOWN"
            if issues is None
            else "ISSUES"
            if issues
            else "CLEAN",
            "bad_prices": item.get("non_positive_or_nan_prices"),
            "bad_ohlc": item.get("ohlc_inconsistent"),
            "negative_volume": item.get("negative_volume"),
            "closed_session_bars": item.get("closed_session_bars"),
        }
    converted = _mapping(
        read_json(QLEAP_SOURCE_ROOT / "reports" / "eurusd-qleap-parquet.json")
    )
    return {
        "available": True,
        "audited_at": audit.get("audited_at"),
        "pairs": pairs,
        "converted_status": converted.get("status", "NOT CONVERTED"),
    }


class Desk:
    def __init__(self, workspace: Path | None = None, command: list[str] | None = None):
        self.workdir = Path(workspace or DESK_ROOT)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.token = self._load_or_create_token()
        self.runner = SuiteRunner(self.workdir, command)

    def _load_or_create_token(self) -> str:
        path = self.workdir / "desk-token.txt"
        try:
            existing = path.read_text(encoding="utf-8").strip()
            if len(existing) >= 32:
                return existing
        except OSError:
            pass
        token = secrets.token_urlsafe(32)
        path.write_text(token, encoding="utf-8")
        return token

    def state(self) -> dict:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "assessment": project_assessment(ROOT),
            "research": _research_state(),
            "training": _training_state(),
            "curated_history": _curated_history_state(),
            "acquisition": _acquisition_state(),
            "rehearsal": _rehearsal_state(),
            "qleap": _qleap_state(),
            "test_run": self.runner.status(),
            "test_history": self.runner.history(),
        }

    def control(self, action) -> dict:
        if action not in {"run-tests"}:
            raise ValueError("unsupported desk action")
        started = self.runner.start()
        return {"action": action, "started": started}


class DeskServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, application: Desk):
        self.application = application
        super().__init__(("127.0.0.1", port), DeskHandler)

    def server_bind(self):
        # Windows otherwise permits two listeners to bind the same local port.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class DeskHandler(BaseHTTPRequestHandler):
    server: DeskServer
    server_version = "AstraEvaluationDesk/1.0"

    def log_message(self, format, *args):
        return

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
                403, {"error": "Open the desk using its local 127.0.0.1 address."}
            )
            return False
        if self.headers.get("Origin", origin) != origin:
            self.respond(403, {"error": "This request must come from the Astra desk."})
            return False
        if mutation and not secrets.compare_digest(
            self.headers.get("X-Astra-Token", ""), self.server.application.token
        ):
            self.respond(403, {"error": "Refresh the desk before continuing."})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        app = self.server.application
        if path == "/api/state":
            try:
                state = app.state()
                state["control_token"] = app.token
                self.respond(200, state)
            except Exception:  # noqa: BLE001 - sanitize storage and serialization errors.
                self.respond(
                    500,
                    {
                        "error": "The desk could not load its saved state. Refresh to retry."
                    },
                )
            return
        try:
            if path in ("/", "/index.html", "/app.css", "/app.js"):
                name = "index.html" if path == "/" else path[1:]
                mime = {
                    "index.html": "text/html; charset=utf-8",
                    "app.css": "text/css; charset=utf-8",
                    "app.js": "application/javascript; charset=utf-8",
                }[name]
                self.respond(200, (WEB_ROOT / name).read_bytes(), mime)
            else:
                self.respond(404, {"error": "Page not found."})
        except OSError:
            self.respond(404, {"error": "This page could not be loaded."})

    def do_POST(self):
        if not self.allowed(mutation=True):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.respond(413, {"error": "Request too large."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.respond(415, {"error": "Expected a JSON request."})
                return
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise TypeError("expected an object")
            if urlsplit(self.path).path != "/api/control":
                self.respond(404, {"error": "Action not found."})
                return
            self.respond(202, self.server.application.control(body.get("action")))
        except (ValueError, TypeError) as error:
            self.respond(400, {"error": str(error) or "Invalid request."})
        except Exception:  # noqa: BLE001 - never reveal internals to the browser.
            self.respond(500, {"error": "The desk could not handle this request."})


def serve(
    host: str = "127.0.0.1",
    port: int = 8789,
    *,
    open_browser: bool = False,
    workspace=None,
    command: list[str] | None = None,
) -> None:
    if host != "127.0.0.1":
        raise ValueError("The Evaluation Desk only supports host 127.0.0.1.")
    application = Desk(workspace, command)
    server = DeskServer(port, application)
    print(
        f"Astra Evaluation Desk available at http://{host}:{server.server_port}/",
        flush=True,
    )
    if open_browser:
        webbrowser.open(f"http://{host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Astra Evaluation Desk")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
