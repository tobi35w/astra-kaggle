"""Offline tests for the Evaluation Desk state assembly, runner, and API."""

import json
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from astra import progress_desk
from astra.progress_desk import Desk, SuiteRunner, read_cached_json, read_json
from astra.repositories import PersistenceError


@pytest.fixture
def desk_paths(monkeypatch, tmp_path):
    roots = {
        "DESK_ROOT": tmp_path / "desk",
        "RESEARCH_ROOT": tmp_path / "research",
        "TRAINING_ROOT": tmp_path / "training",
        "CURATED_HISTORY_ROOT": tmp_path / "curated-history",
        "ACQUISITION_ROOT": tmp_path / "acquisition",
        "REHEARSAL_ROOT": tmp_path / "rehearsal",
        "QLEAP_ROOT": tmp_path / "qleap",
        "QLEAP_SOURCE_ROOT": tmp_path / "qleap-source",
    }
    for name, path in roots.items():
        monkeypatch.setattr(progress_desk, name, path)
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_helpers_fall_back_on_missing_and_corrupt_files(tmp_path):
    missing = tmp_path / "missing.json"
    assert read_json(missing, {"default": 1}) == {"default": 1}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    assert read_cached_json(corrupt, "fallback") == "fallback"
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert read_cached_json(valid) == {"a": 1}
    assert read_cached_json(valid) == {"a": 1}  # served from the mtime cache


def test_research_state_aggregates_progress_and_results(desk_paths):
    _write(
        desk_paths["RESEARCH_ROOT"] / "research-progress.json",
        {
            "status": "COMPLETED",
            "completed": 42,
            "total": 50,
            "updated_at": "2026-10-13T10:00:00+00:00",
        },
    )
    _write(
        desk_paths["RESEARCH_ROOT"] / "research-results.json",
        {
            "status": "COMPLETED",
            "profitability": "UNPROVEN",
            "items": [
                {"kind": "BACKTEST"},
                {"kind": "BACKTEST"},
                {"kind": "OOS_HOLDOUT"},
                {"kind": "WALK_FORWARD"},
            ],
            "errors": [{"symbol": "EURUSD"}],
            "limitations": ["synthetic fixture limit"],
        },
    )
    state = progress_desk._research_state()
    assert state["status"] == "COMPLETED"
    assert state["completed"] == 42 and state["total"] == 50
    assert state["progress_fraction"] == pytest.approx(0.84)
    assert state["experiment_kinds"] == {
        "BACKTEST": 2,
        "OOS_HOLDOUT": 1,
        "WALK_FORWARD": 1,
    }
    assert state["errors"] == 1
    assert state["profitability"] == "UNPROVEN"


def test_acquisition_state_counts_manifest_units(desk_paths):
    manifest = {
        "updated_at": "2026-10-13T17:00:00+00:00",
        "downloads": {
            "EURUSD:2026-09-01": {"status": "downloaded"},
            "EURUSD:2026-09-02": {"status": "failed", "error": "TransientSourceError"},
            "EURUSD:2026-09-03": {"status": "downloaded"},
            "USDJPY:2026-09-01": {"status": "downloaded"},
        },
    }
    _write(
        desk_paths["ACQUISITION_ROOT"] / "manifests" / "dukascopy_manifest.json",
        manifest,
    )
    state = progress_desk._acquisition_state()
    assert state["running"] is False
    assert state["units"] == 4
    assert state["by_symbol"]["EURUSD"] == {"downloaded": 2, "failed": 1}
    assert state["by_symbol"]["USDJPY"] == {"downloaded": 1}
    assert state["latest_days"]["EURUSD"] == "2026-09-03"
    assert state["summary_status"] in ("RECORDED_ACTIVE", "STALE")
    assert "worker" in state["status_note"]


def test_rehearsal_and_qleap_states(desk_paths):
    assert progress_desk._rehearsal_state() == {"available": False}
    _write(
        desk_paths["REHEARSAL_ROOT"] / "summary.json",
        {"net_pnl": 184.46, "trades": 19, "wins": 7, "final_equity": 10184.46},
    )
    rehearsal = progress_desk._rehearsal_state()
    assert rehearsal["available"] is True
    assert rehearsal["net_pnl"] == 184.46

    assert progress_desk._qleap_state() == {"available": False}
    _write(
        desk_paths["QLEAP_ROOT"] / "astra-audit.json",
        {
            "audited_at": "2026-10-13T14:00:00+00:00",
            "pairs": {
                "EURUSD": {
                    "files": 77,
                    "rows": 2374500,
                    "first_timestamp": "2020-04-01T00:01:00",
                    "last_timestamp": "2026-09-01T00:00:00",
                    "non_monotonic_files": [],
                    "duplicate_timestamp_files": [],
                    "non_positive_or_nan_prices": 0,
                    "ohlc_inconsistent": 0,
                    "negative_volume": 0,
                    "closed_session_bars": 0,
                }
            },
        },
    )
    _write(
        desk_paths["QLEAP_SOURCE_ROOT"] / "reports" / "eurusd-qleap-parquet.json",
        {"status": "DATA QUALITY INSUFFICIENT"},
    )
    qleap = progress_desk._qleap_state()
    assert qleap["pairs"]["EURUSD"]["rows"] == 2374500
    assert qleap["converted_status"] == "DATA QUALITY INSUFFICIENT"


def test_test_runner_streams_a_command_and_records_exit_code(tmp_path):
    runner = SuiteRunner(
        tmp_path, command=[sys.executable, "-c", "print('desk-fake-line')"]
    )
    assert runner.start() is True
    # The default pytest basetemp needs its parent directory to exist first.
    assert (tmp_path / "test-tmp").is_dir()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = runner.status()
        if not status["running"]:
            break
        time.sleep(0.05)
    status = runner.status()
    assert status["running"] is False
    assert status["exit_code"] == 0
    assert "desk-fake-line" in status["log"]
    history = runner.history()
    assert len(history) == 1 and history[0]["last_line"] == "desk-fake-line"
    # A second start while idle is allowed; a concurrent one is not.
    assert runner.start() is True


def test_desk_token_persists_and_control_validates_actions(tmp_path):
    desk = Desk(tmp_path)
    assert len(desk.token) >= 32
    assert Desk(tmp_path).token == desk.token
    with pytest.raises(ValueError, match="unsupported desk action"):
        desk.control("delete-everything")


def test_desk_state_shape(tmp_path, desk_paths):
    desk = Desk(tmp_path)
    state = desk.state()
    for key in (
        "research",
        "training",
        "curated_history",
        "acquisition",
        "rehearsal",
        "qleap",
        "test_run",
        "test_history",
    ):
        assert key in state
    assert state["research"]["status"] in ("NOT_STARTED", "IN_PROGRESS", "COMPLETED")


@pytest.fixture
def desk_http(desk_paths, monkeypatch):
    command = [sys.executable, "-c", "print('desk-api-line')"]
    captured = []

    class Capturing(progress_desk.DeskServer):
        def __init__(self, port, application):
            super().__init__(port, application)
            captured.append(self)

    monkeypatch.setattr(progress_desk, "DeskServer", Capturing)
    thread = threading.Thread(
        target=lambda: progress_desk.serve(
            "127.0.0.1", 0, workspace=desk_paths["DESK_ROOT"], command=command
        ),
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not captured and time.monotonic() < deadline:
        time.sleep(0.05)
    server = captured[0]
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base, desk_paths["DESK_ROOT"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.socket
def test_desk_api_state_static_and_control(desk_http):
    base, _ = desk_http
    with urlopen(base + "/api/state", timeout=5) as response:
        state = json.loads(response.read())
    token = state["control_token"]
    assert len(token) >= 32
    assert "research" in state and "acquisition" in state

    with urlopen(base + "/", timeout=5) as response:
        assert "Evaluation Desk" in response.read().decode()
    for path in ("/nope.js", "/app.map.json"):
        with pytest.raises(HTTPError) as error:
            urlopen(base + path, timeout=5)
        assert error.value.code == 404

    request = Request(
        base + "/api/control",
        data=json.dumps({"action": "explode"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 403

    run_request = Request(
        base + "/api/control",
        data=b'{"action":"run-tests"}',
        headers={"Content-Type": "application/json", "X-Astra-Token": token},
        method="POST",
    )
    with urlopen(run_request, timeout=5) as response:
        assert response.status == 202
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with urlopen(base + "/api/state", timeout=5) as response:
            latest = json.loads(response.read())
        if not latest["test_run"]["running"]:
            break
        time.sleep(0.2)
    assert "desk-api-line" in latest["test_run"]["log"]
    assert latest["test_run"]["exit_code"] == 0


@pytest.mark.socket
def test_desk_server_rejects_a_second_listener_on_the_same_port(desk_http):
    base, workspace = desk_http
    port = urlsplit(base).port
    with pytest.raises(OSError), progress_desk.DeskServer(port, Desk(workspace)):
        pass


@pytest.mark.socket
@pytest.mark.parametrize("failure", ["storage", "serialization"])
def test_desk_state_failure_is_sanitized_and_static_missing_files_remain_404(
    desk_http, monkeypatch, tmp_path, failure
):
    def failed_state(self):
        if failure == "storage":
            raise PersistenceError("sensitive internal database path")
        return {"invalid_value": float("nan")}

    monkeypatch.setattr(Desk, "state", failed_state)
    base, _ = desk_http
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/state", timeout=5)
    assert error.value.code == 500
    payload = json.loads(error.value.read())
    assert payload == {
        "error": "The desk could not load its saved state. Refresh to retry."
    }

    monkeypatch.setattr(progress_desk, "WEB_ROOT", tmp_path / "missing")
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/app.css", timeout=5)
    assert error.value.code == 404
