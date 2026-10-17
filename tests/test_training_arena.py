"""Offline tests for the training arena state assembly and loopback API.

All fixture records are synthetic; the arena never touches the real data root
because the module paths are monkeypatched for every test.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from astra import training_arena
from astra.repositories import Entity


@pytest.fixture
def arena_paths(monkeypatch, tmp_path):
    arena = tmp_path / "arena"
    research = tmp_path / "research"
    dataset = tmp_path / "dataset"
    for path in (arena, research, dataset):
        path.mkdir()
    monkeypatch.setattr(training_arena, "ARENA_ROOT", arena)
    monkeypatch.setattr(training_arena, "RESEARCH_ROOT", research)
    monkeypatch.setattr(training_arena, "DATASET_ROOT", dataset)
    return {"arena": arena, "research": research, "dataset": dataset}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_json_helpers_fall_back_on_missing_and_corrupt_files(tmp_path):
    missing = tmp_path / "missing.json"
    assert training_arena.read_json(missing, {"default": True}) == {"default": True}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert training_arena.read_cached_json(corrupt, "fallback") == "fallback"


def test_read_cached_json_reuses_value_until_file_changes(tmp_path):
    path = tmp_path / "cached.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    assert training_arena.read_cached_json(path) == {"value": 1}
    # Same content and mtime: served from cache without raising.
    assert training_arena.read_cached_json(path) == {"value": 1}
    path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    assert training_arena.read_cached_json(path) == {"value": 2}


def test_arena_state_defaults_when_nothing_has_run(arena_paths):
    state = training_arena.arena_state()
    assert state["campaign"]["status"] == "WAITING_FOR_RESEARCH"
    assert state["campaign"]["summary"]["completed_test_accounts"] == 0
    assert state["campaign"]["summary"]["aggregate_net_pnl"] == 0
    assert state["campaign"]["recent_trades"] == []
    assert state["research"]["status"] == "NOT_STARTED"
    assert state["research"]["prior_phase_profitability"] == "UNPROVEN"
    assert state["dataset"]["offline_ready"] is False
    assert state["storage_sync"] == {"status": "LOCAL SPOOL READY"}


def test_arena_state_aggregates_completed_tests(arena_paths):
    _write(
        arena_paths["arena"] / "arena-state.json",
        {
            "status": "RUNNING",
            "total_tests": 12,
            "updated_at": "2026-10-13T00:00:00+00:00",
        },
    )
    _write(
        arena_paths["arena"] / "arena-results.json",
        {
            "status": "COMPLETED",
            "completed_at": "2026-10-13T01:00:00+00:00",
            "tests": [
                {
                    "id": "test-1",
                    "pair": "EURUSD",
                    "timeframe": "M5",
                    "strategy": "ema_trend",
                    "net_pnl": 120.5,
                    "starting_balance": 10000,
                    "trades": [{"pnl": 150.0}, {"pnl": -29.5}],
                },
                {
                    "id": "test-2",
                    "pair": "USDJPY",
                    "timeframe": "H1",
                    "strategy": "breakout",
                    "net_pnl": -40.0,
                    "starting_balance": 10000,
                    "trades": [{"pnl_per_unit": 2.0, "size": -20.0}],
                },
            ],
            "assumptions": ["synthetic fixture assumptions"],
            "source_limitations": ["synthetic fixture limitation"],
        },
    )
    _write(
        arena_paths["research"] / "research-progress.json",
        {"status": "COMPLETED", "stage": "done", "completed": 10, "total": 12},
    )
    _write(
        arena_paths["research"] / "research-results.json",
        {"profitability": "UNPROVEN", "items": [{"kind": "BACKTEST"}]},
    )
    _write(
        arena_paths["dataset"] / "four-pairs-manifest.json",
        {
            "dataset_id": "synthetic/manifest",
            "rows": 1000,
            "files": [1, 2],
            "scope": {"symbols": ["EURUSD"], "timeframes": ["M5"]},
        },
    )
    _write(
        arena_paths["dataset"] / "quality-report.json",
        {
            "structural_pass_files": 2,
            "structural_fail_files": [],
            "research_readiness": {"requires_followup": ["fixture"]},
        },
    )

    state = training_arena.arena_state()
    summary = state["campaign"]["summary"]
    assert state["campaign"]["status"] == "COMPLETED"
    assert summary["completed_test_accounts"] == 2
    assert summary["aggregate_net_pnl"] == pytest.approx(80.5)
    assert summary["aggregate_ending_equity"] == pytest.approx(20080.5)
    assert summary["trade_count"] == 3
    assert summary["wins"] == 1  # only the explicit +150 trade
    assert (
        summary["losses"] == 2
    )  # the per-unit trade falls back to pnl_per_unit * size = -40
    assert summary["risk_note"].startswith("P&L sums separate")

    # Newest completed run first, then its latest trade; trade annotations attached.
    trades = state["campaign"]["recent_trades"]
    assert trades[0]["test_id"] == "test-2"
    assert trades[0]["pair"] == "USDJPY"
    assert (trades[0]["pnl_per_unit"], trades[0]["size"]) == (2.0, -20.0)
    assert trades[1]["test_id"] == "test-1"
    assert trades[1]["pnl"] == -29.5
    assert "trades" not in state["campaign"]["tests"][0]
    assert state["research"]["status"] == "COMPLETED"
    assert state["research"]["prior_phase_result_count"] == 1
    assert state["dataset"]["offline_ready"] is True


def test_count_entity_reports_journal_rows(arena_paths):
    from astra.adapters import SQLiteRepository
    from astra.repositories import Record

    journal_path = arena_paths["arena"] / "arena-journal.sqlite3"
    assert training_arena.count_entity(Entity.MODEL) == 0
    store = SQLiteRepository(str(journal_path))
    records = tuple(
        Record(Entity.MODEL, _stamp(), {"version": f"m{i}"}) for i in range(3)
    ) + (Record(Entity.BACKTEST, _stamp(), {"run": 1}),)
    store.append_batch(records)
    store.close()
    assert training_arena.count_entity(Entity.MODEL) == 3
    assert training_arena.count_entity(Entity.BACKTEST) == 1
    assert training_arena.count_entity(Entity.TRADE) == 0


def _stamp():
    from datetime import UTC, datetime

    return datetime(2026, 10, 13, tzinfo=UTC)


def test_main_parses_arguments_without_starting_a_server(monkeypatch):
    calls = []

    def fake_serve(host, port, *, open_browser=False):
        calls.append((host, port, open_browser))

    monkeypatch.setattr(training_arena, "serve", fake_serve)
    monkeypatch.setattr("sys.argv", ["arena", "--port", "9001", "--open"])
    training_arena.main()
    assert calls == [("127.0.0.1", 9001, True)]


@pytest.fixture
def arena_http(arena_paths):
    captured = []

    class Capturing(ThreadingHTTPServer):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            captured.append(self)

    original = training_arena.ThreadingHTTPServer
    training_arena.ThreadingHTTPServer = Capturing
    thread = threading.Thread(
        target=lambda: training_arena.serve("127.0.0.1", 0, open_browser=False),
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not captured and time.monotonic() < deadline:
            time.sleep(0.05)
        server = captured[0]
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        training_arena.ThreadingHTTPServer = original


@pytest.fixture
def control_token(arena_http):
    with urlopen(arena_http + "/api/state", timeout=5) as response:
        return json.loads(response.read())["control_token"]


@pytest.mark.socket
def test_arena_state_endpoint_issues_a_control_token(arena_http, control_token):
    assert len(control_token) >= 32


@pytest.mark.socket
def test_arena_serves_whitelisted_static_files_only(arena_http):
    with urlopen(arena_http + "/", timeout=5) as response:
        assert response.status == 200
        assert "text/html" in response.headers["Content-Type"]
    for path in ("/app.js", "/app.css"):
        with urlopen(arena_http + path, timeout=5) as response:
            assert response.status == 200
    for path in ("/nope.js", "/index.md"):
        with pytest.raises(HTTPError) as error:
            urlopen(arena_http + path, timeout=5)
        assert error.value.code == 404


@pytest.mark.socket
def test_arena_control_requires_token_and_valid_action(
    arena_http, control_token, arena_paths
):
    control = arena_paths["arena"] / "arena-control.json"

    def post(payload, token=None):
        request = Request(
            arena_http + "/api/control",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if token is not None:
            request.add_header("X-Astra-Token", token)
        return urlopen(request, timeout=5)

    with pytest.raises(HTTPError) as error:
        post({"action": "pause"})
    assert error.value.code == 403
    assert not control.exists()

    with pytest.raises(HTTPError) as error:
        post({"action": "explode"}, token=control_token)
    assert error.value.code == 400
    assert not control.exists()

    with post({"action": "pause"}, token=control_token) as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"ok": True, "action": "pause"}
    assert json.loads(control.read_text(encoding="utf-8"))["action"] == "pause"

    # The arena worker reads its control action from the file; resume rewrites it.
    with post({"action": "resume"}, token=control_token) as response:
        assert response.status == 200
    assert json.loads(control.read_text(encoding="utf-8"))["action"] == "resume"
