"""Synthetic monitor artifacts test status honesty, units, and robust polling."""

import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace

import pytest

from astra import progress_desk as desk
from astra.repositories import PersistenceError

NOW = datetime(2026, 10, 14, 12, tzinfo=UTC)


@pytest.fixture
def monitor_paths(tmp_path, monkeypatch):
    paths = {}
    for name in (
        "RESEARCH_ROOT",
        "ACQUISITION_ROOT",
        "TRAINING_ROOT",
        "CURATED_HISTORY_ROOT",
        "REHEARSAL_ROOT",
        "QLEAP_ROOT",
        "QLEAP_SOURCE_ROOT",
    ):
        paths[name] = tmp_path / name.lower()
        paths[name].mkdir()
        monkeypatch.setattr(desk, name, paths[name])
    return paths


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize(
    ("offset", "status", "freshness"),
    [(0, "RECORDED_ACTIVE", "CURRENT"), (901, "STALE", "STALE")],
)
def test_running_research_is_a_timestamped_snapshot_not_worker_proof(
    monitor_paths, offset, status, freshness
):
    write(
        monitor_paths["RESEARCH_ROOT"] / "research-progress.json",
        {
            "status": "RUNNING",
            "completed": 5,
            "total": 20,
            "updated_at": (NOW - timedelta(seconds=offset)).isoformat(),
            "stage": "holdout",
            "pair": "EURUSD",
            "timeframe": "H1",
        },
    )
    state = desk._research_state(now=NOW)
    assert state["status"] == status
    assert state["recorded_status"] == "RUNNING"
    assert state["freshness"]["state"] == freshness
    assert state["freshness"]["threshold_seconds"] == 900
    assert "worker" in state["status_note"]
    assert state["current"]["pair"] == "EURUSD"
    assert state["progress_fraction"] == 0.25


def test_completed_history_survives_age_but_counter_overrun_is_visible(monitor_paths):
    write(
        monitor_paths["RESEARCH_ROOT"] / "research-progress.json",
        {
            "status": "COMPLETED",
            "completed": 2955,
            "total": 2904,
            "updated_at": (NOW - timedelta(days=2)).isoformat(),
        },
    )
    state = desk._research_state(now=NOW)
    assert state["status"] == "COMPLETED"
    assert state["freshness"]["state"] == "STALE"
    assert state["count_inconsistent"] is True
    assert state["completed"] == 2955
    assert state["total"] == 2904
    assert state["progress_fraction"] == 1


@pytest.mark.parametrize(
    "stamp", [None, "garbled", "2026-10-14T12:00:00", "2026-10-15T12:00:00Z"]
)
def test_invalid_naive_and_future_timestamps_do_not_claim_current_activity(
    monitor_paths, stamp
):
    write(
        monitor_paths["RESEARCH_ROOT"] / "research-progress.json",
        {
            "status": "RUNNING",
            "updated_at": stamp,
        },
    )
    state = desk._research_state(now=NOW)
    assert state["freshness"]["state"] == "UNKNOWN"
    assert state["freshness"]["age_seconds"] is None
    assert state["status"] == "RECORDED_ACTIVE"


def test_missing_and_corrupt_artifacts_are_distinguished(monitor_paths):
    assert desk._research_state(now=NOW)["status"] == "NOT_STARTED"
    path = monitor_paths["RESEARCH_ROOT"] / "research-progress.json"
    for text in (
        "{partial",
        "[]",
        '{"completed": NaN}',
        '{"completed": Infinity}',
        '{"completed": 1e999}',
    ):
        path.write_text(text, encoding="utf-8")
        state = desk._research_state(now=NOW)
        assert state["status"] == "UNAVAILABLE"
        assert state["artifact_issues"]
        json.dumps(state, allow_nan=False)


def test_wrong_nested_shapes_and_invalid_counts_do_not_break_polling(monitor_paths):
    write(
        monitor_paths["RESEARCH_ROOT"] / "research-progress.json",
        {
            "status": "RUNNING",
            "completed": -3,
            "total": "unknown",
        },
    )
    write(
        monitor_paths["RESEARCH_ROOT"] / "research-results.json",
        {
            "items": [None, 1, {"kind": "BACKTEST"}],
            "errors": None,
            "limitations": "invalid list",
        },
    )
    state = desk._research_state(now=NOW)
    assert state["count_inconsistent"] is True
    assert state["completed"] == state["total"] == 0
    assert state["progress_fraction"] is None
    assert state["experiments"] == 1
    assert state["limitations"] == []


def test_acquisition_manifest_never_proves_worker_activity(monitor_paths):
    write(
        monitor_paths["ACQUISITION_ROOT"] / "manifests" / "dukascopy_manifest.json",
        {
            "updated_at": NOW.isoformat(),
            "downloads": {
                "EURUSD:2026-09-01": {"status": "downloaded"},
                "EURUSD:2026-09-02": {"status": "failed"},
            },
        },
    )
    state = desk._acquisition_state(now=NOW)
    assert state["running"] is False
    assert state["status"] == "RECORDED_ACTIVE"
    assert state["by_symbol"]["EURUSD"] == {"downloaded": 1, "failed": 1}
    assert state["latest_days"]["EURUSD"] == "2026-09-01"
    write(
        monitor_paths["ACQUISITION_ROOT"] / "acquisition-summary.json",
        {"status": "COMPLETED"},
    )
    assert desk._acquisition_state(now=NOW)["status"] == "COMPLETED"


def test_acquisition_summary_uses_its_acquired_at_timestamp(monitor_paths):
    write(
        monitor_paths["ACQUISITION_ROOT"] / "manifests" / "dukascopy_manifest.json",
        {
            "updated_at": (NOW - timedelta(days=1)).isoformat(),
        },
    )
    write(
        monitor_paths["ACQUISITION_ROOT"] / "acquisition-summary.json",
        {
            "status": "INTERRUPTED",
            "acquired_at": NOW.isoformat(),
            "run_id": "summary-run",
        },
    )
    state = desk._acquisition_state(now=NOW)
    assert state["status"] == "INTERRUPTED"
    assert state["updated_at"] == NOW.isoformat()
    assert state["freshness"]["state"] == "CURRENT"
    assert state["run_id"] == "summary-run"


@pytest.mark.parametrize("explicit_run", [True, False])
def test_new_acquisition_activity_is_not_masked_by_an_older_summary(
    monitor_paths, explicit_run
):
    manifest = {
        "updated_at": NOW.isoformat(),
        "downloads": {"EURUSD:2026-10-01": {"status": "downloaded"}},
    }
    if explicit_run:
        manifest.update(
            {
                "latest_run_id": "new-run",
                "runs": {
                    "new-run": {
                        "run_id": "new-run",
                        "status": "RUNNING",
                        "started_at": NOW.isoformat(),
                    }
                },
            }
        )
    write(
        monitor_paths["ACQUISITION_ROOT"] / "manifests" / "dukascopy_manifest.json",
        manifest,
    )
    write(
        monitor_paths["ACQUISITION_ROOT"] / "acquisition-summary.json",
        {
            "status": "COMPLETE",
            "acquired_at": (NOW - timedelta(days=1)).isoformat(),
            "run_id": "old-run",
            "symbols": {"EURUSD": {"status": "PASS"}},
        },
    )
    state = desk._acquisition_state(now=NOW)
    assert state["status"] == "RECORDED_ACTIVE"
    assert state["running"] is False
    assert state["symbols"] == {}
    assert state["snapshot_source"] == ("manifest-run" if explicit_run else "manifest")
    assert state["run_id"] == ("new-run" if explicit_run else None)


def test_training_reports_current_test_and_keeps_trade_rows_out_of_api(monitor_paths):
    write(
        monitor_paths["TRAINING_ROOT"] / "arena-state.json",
        {
            "status": "RUNNING",
            "updated_at": NOW.isoformat(),
            "completed_tests": 2,
            "total_tests": 4,
            "active_test": {
                "pair": "USDJPY",
                "timeframe": "H4",
                "strategy": "ema_trend",
            },
        },
    )
    write(
        monitor_paths["TRAINING_ROOT"] / "arena-results.json",
        {
            "tests": [
                {"net_pnl": 5.0, "trades": [{"pnl_per_unit": 1, "size": 5}]},
                {"net_pnl": -2.0, "trades": [{"pnl_per_unit": -1, "size": 2}]},
            ],
            "models": [{"status": "TRAINED"}],
            "source_limitations": ["Synthetic tests only."],
        },
    )
    state = desk._training_state(now=NOW)
    assert state["current"]["pair"] == "USDJPY"
    assert state["saved_results"] == state["completed"] == 2
    assert state["total"] == 4
    assert state["summary"]["net_pnl"] == 3
    assert state["summary"]["trade_count"] == 2
    assert state["summary"]["positive_tests"] == 1
    assert state["summary"]["negative_tests"] == 1
    assert state["summary"]["model_count"] == 1
    assert "independent" in state["summary"]["units_note"]
    assert "trades" not in json.dumps(state)


def test_curated_returns_weight_independent_test_counts_and_keep_scenarios_separate(
    monitor_paths,
):
    write(
        monitor_paths["CURATED_HISTORY_ROOT"] / "progress.json",
        {
            "status": "COMPLETED",
            "completed_tests": 5,
            "total_tests": 5,
            "updated_at": NOW.isoformat(),
        },
    )
    write(
        monitor_paths["CURATED_HISTORY_ROOT"] / "evaluation-report.json",
        {
            "status": "COMPLETED",
            "models": [{"status": "TRAINED"}],
            "summary": {
                "ema_trend|EURUSD|H1|base": {
                    "window_runs": 1,
                    "positive_windows": 1,
                    "negative_windows": 0,
                    "mean_return_pct": 10,
                    "median_return_pct": 10,
                    "total_closed_trades": 5,
                },
                "ema_trend|USDJPY|H1|base": {
                    "window_runs": 3,
                    "positive_windows": 0,
                    "negative_windows": 3,
                    "mean_return_pct": -2,
                    "median_return_pct": -1,
                    "total_closed_trades": 9,
                },
                "ema_trend|EURUSD|H1|double_cost": {
                    "window_runs": 1,
                    "positive_windows": 0,
                    "negative_windows": 1,
                    "mean_return_pct": -3,
                    "median_return_pct": -3,
                    "total_closed_trades": 5,
                },
            },
        },
    )
    state = desk._curated_history_state(now=NOW)
    summary = state["summary"]
    assert state["saved_results"] == 5
    assert summary["net_pnl"] is None
    assert summary["scenarios"]["base"]["mean_return_pct"] == 1
    assert summary["scenarios"]["double_cost"]["mean_return_pct"] == -3
    assert summary["scenarios"]["base"]["trade_count"] == 14
    assert summary["negative_tests"] == 4
    assert summary["groups"][1]["median_return_pct"] == -1
    assert "native-currency" in summary["units_note"]


def test_cached_results_recover_after_partial_rewrite(tmp_path):
    path = tmp_path / "saved.json"
    write(path, {"phase": "first"})
    assert desk.read_cached_json(path)["phase"] == "first"
    path.write_text('{"phase":', encoding="utf-8")
    assert desk.read_cached_json(path) is None
    write(path, {"phase": "completed"})
    assert desk.read_cached_json(path)["phase"] == "completed"


def test_source_audit_bad_shapes_do_not_break_other_monitor_views(monitor_paths):
    write(
        monitor_paths["QLEAP_ROOT"] / "astra-audit.json",
        {
            "pairs": {
                "EURUSD": {
                    "non_monotonic_files": 12,
                    "duplicate_timestamp_files": None,
                },
                "GBPUSD": "invalid pair record",
            },
        },
    )
    write(monitor_paths["REHEARSAL_ROOT"] / "summary.json", ["invalid summary"])
    write(
        monitor_paths["QLEAP_SOURCE_ROOT"] / "reports" / "eurusd-qleap-parquet.json", []
    )
    state = desk._qleap_state()
    assert state["pairs"]["EURUSD"]["non_monotonic_files"] is None
    assert state["pairs"]["EURUSD"]["structural_status"] == "UNKNOWN"
    assert state["pairs"]["EURUSD"]["structural_issues"] is None
    assert "GBPUSD" not in state["pairs"]
    assert desk._rehearsal_state() == {"available": False}
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "192.168.1.1"])
def test_desk_rejects_hosts_that_do_not_match_its_loopback_origin(host, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Host validation must happen before creating the application.")

    monkeypatch.setattr(desk, "Desk", unexpected)
    with pytest.raises(ValueError, match="127.0.0.1"):
        desk.serve(host=host)


@pytest.mark.parametrize("failure", ["storage", "serialization", "filesystem"])
def test_saved_state_handler_returns_sanitized_errors_without_a_network(failure):
    def failed_state():
        if failure == "storage":
            raise PersistenceError("sensitive database details")
        if failure == "filesystem":
            raise OSError("sensitive file path")
        return {"invalid": float("nan")}

    handler = desk.DeskHandler.__new__(desk.DeskHandler)
    handler.server = SimpleNamespace(
        server_port=8789,
        application=SimpleNamespace(
            state=failed_state, token="synthetic-control-token"
        ),
    )
    handler.path = "/api/state"
    handler.headers = {"Host": "127.0.0.1:8789"}
    handler.requestline = "GET /api/state HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.wfile = BytesIO()
    handler.do_GET()
    headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
    assert b" 500 " in headers.splitlines()[0]
    assert json.loads(body) == {
        "error": "The desk could not load its saved state. Refresh to retry."
    }
