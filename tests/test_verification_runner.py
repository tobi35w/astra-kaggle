"""Offline runner fixtures launch tiny Python commands, never another suite."""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import pytest

from astra.adapters import SQLiteRepository
from astra.repositories import Entity, Journal
from astra.verification import RECORD_KIND, SuiteRunner, _environment


def finish(runner: SuiteRunner) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = runner.status()
        if not state["running"]:
            return state
        time.sleep(0.02)
    pytest.fail("verification fixture did not finish")


def command(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_results_survive_restart_and_runs_have_distinct_artifacts(tmp_path):
    runner = SuiteRunner(tmp_path, command("print('first fixture run')"))
    assert runner.start()
    first = finish(runner)
    restored = SuiteRunner(tmp_path, command("print('second fixture run')"))
    assert restored.status()["status"] == "PASSED"
    assert restored.status()["run_id"] == first["run_id"]
    assert "first fixture run" in restored.status()["log"]
    assert restored.start()
    second = finish(restored)
    assert second["run_id"] != first["run_id"]
    assert second["log_path"] != first["log_path"]
    history = restored.history()
    assert len(history) == 2
    assert history[0]["last_line"] == "second fixture run"
    assert history[1]["last_line"] == "first fixture run"
    repository = SQLiteRepository(str(tmp_path / "verification-journal.sqlite3"))
    try:
        records = repository.query(Entity.AUDIT, strategy=RECORD_KIND)
        assert {row.correlation_id for row in records} == {
            first["run_id"],
            second["run_id"],
        }
        assert all(row.occurred_at.tzinfo is not None for row in records)
    finally:
        repository.close()


def test_process_lock_blocks_other_instances_until_run_finishes(tmp_path):
    gate = tmp_path / "release"
    source = (
        "from pathlib import Path; import time; "
        f"gate = Path({str(gate)!r}); "
        "print('waiting', flush=True)\n"
        "while not gate.exists(): time.sleep(0.02)"
    )
    runner = SuiteRunner(tmp_path, command(source))
    assert runner.start()
    other = SuiteRunner(tmp_path, command("print('next')"))
    try:
        assert other.status()["running"] is True
        assert other.start() is False
        assert runner.start() is False
        assert other.status()["status"] == "RUNNING"
    finally:
        gate.touch()
        finish(runner)
    assert other.start()
    assert finish(other)["status"] == "PASSED"


def test_abandoned_run_is_interrupted_and_history_is_recovered(tmp_path):
    runner = SuiteRunner(tmp_path)
    saved = runner.status()
    saved.update(
        run_id="interrupted-fixture",
        status="RUNNING",
        running=True,
        started_at=datetime.now(UTC).isoformat(),
        phase="Tests and coverage",
        phases=[{"name": "Tests and coverage", "status": "RUNNING"}],
    )
    repository = SQLiteRepository(str(tmp_path / "verification-journal.sqlite3"))
    try:
        Journal(repository).record(
            Entity.AUDIT,
            datetime.now(UTC),
            {"kind": RECORD_KIND, "run": saved},
            strategy=RECORD_KIND,
            correlation_id=saved["run_id"],
        )
    finally:
        repository.close()
    restored = SuiteRunner(tmp_path)
    assert restored.status()["status"] == "INTERRUPTED"
    assert restored.status()["running"] is False
    assert restored.status()["phases"][0]["status"] == "INTERRUPTED"
    assert restored.history()[0]["finished_at"]
    assert SuiteRunner(tmp_path).status()["status"] == "INTERRUPTED"


def test_process_lock_also_blocks_a_separate_python_process(tmp_path):
    source = (
        "from pathlib import Path; from astra.verification import SuiteRunner; "
        f"other = SuiteRunner(Path({str(tmp_path)!r}), ['invalid-fixture-command']); "
        "assert other.status()['running']; "
        "assert not other.start(); print('cross-process lock held')"
    )
    runner = SuiteRunner(tmp_path, command(source))
    assert runner.start()
    state = finish(runner)
    assert state["status"] == "PASSED"
    assert "cross-process lock held" in state["log"]


def test_storage_failure_prevents_process_start_and_releases_lock(
    tmp_path, monkeypatch
):
    marker = tmp_path / "must-not-exist"
    source = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    runner = SuiteRunner(tmp_path, command(source))

    def cannot_persist():
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(runner, "_persist", cannot_persist)
    with pytest.raises(RuntimeError, match="journal storage"):
        runner.start()
    assert runner.status()["status"] == "FAILED"
    assert not marker.exists()
    other = SuiteRunner(tmp_path, command("print('storage fixture recovered')"))
    assert other.start()
    assert finish(other)["status"] == "PASSED"


def test_default_phases_include_required_checks_and_report_artifacts(tmp_path):
    runner = SuiteRunner(tmp_path)
    steps = runner._steps("fixture")
    assert [name for name, _ in steps] == [
        "Tests and coverage",
        "Ruff lint",
        "Ruff format",
    ]
    pytest_command = steps[0][1]
    assert "--cov=astra" in pytest_command
    assert any(value.startswith("--junitxml=") for value in pytest_command)
    assert any(value.startswith("--cov-report=json:") for value in pytest_command)
    assert "--no-cov" not in pytest_command
    assert steps[1][1][-3:] == ["check", "astra", "tests"]
    assert steps[2][1][-4:] == ["format", "--check", "astra", "tests"]


def test_phases_report_failures_and_continue_collecting_checks(tmp_path, monkeypatch):
    runner = SuiteRunner(tmp_path)

    def steps(run_id):
        artifacts = tmp_path / "test-runs" / run_id
        junit = '<testsuites><testsuite tests="4" failures="1" errors="0" skipped="1"/></testsuites>'
        coverage = json.dumps({"totals": {"percent_covered": 92.345}})
        source = (
            "from pathlib import Path; "
            f"Path({str(artifacts / 'junit.xml')!r}).write_text({junit!r}); "
            f"Path({str(artifacts / 'coverage.json')!r}).write_text({coverage!r}); "
            "print('one synthetic assertion failed'); raise SystemExit(1)"
        )
        return [
            ("Tests and coverage", command(source)),
            ("Ruff lint", command("print('lint fixture passed')")),
            ("Ruff format", command("print('format fixture passed')")),
        ]

    monkeypatch.setattr(runner, "_steps", steps)
    assert runner.start()
    state = finish(runner)
    assert state["status"] == "FAILED" and state["exit_code"] == 1
    assert [phase["status"] for phase in state["phases"]] == [
        "FAILED",
        "PASSED",
        "PASSED",
    ]
    assert state["counts"] == {
        "tests": 4,
        "passed": 2,
        "failed": 1,
        "errors": 0,
        "skipped": 1,
    }
    assert state["coverage_pct"] == 92.34
    assert "format fixture passed" in state["log"]
    assert SuiteRunner(tmp_path).status()["counts"] == state["counts"]


def test_missing_verification_reports_cannot_be_a_pass(tmp_path, monkeypatch):
    runner = SuiteRunner(tmp_path)
    monkeypatch.setattr(
        runner,
        "_steps",
        lambda _: [("Tests and coverage", command("print('no reports')"))],
    )
    assert runner.start()
    state = finish(runner)
    assert state["status"] == "FAILED"
    assert state["counts"] is None and state["coverage_pct"] is None
    assert "reports missing or unreadable" in state["log"]


def test_all_skipped_tests_do_not_pass_verification(tmp_path, monkeypatch):
    runner = SuiteRunner(tmp_path)

    def steps(run_id):
        artifacts = tmp_path / "test-runs" / run_id
        junit = '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="2"/></testsuites>'
        coverage = json.dumps({"totals": {"percent_covered": 50}})
        source = (
            "from pathlib import Path; "
            f"Path({str(artifacts / 'junit.xml')!r}).write_text({junit!r}); "
            f"Path({str(artifacts / 'coverage.json')!r}).write_text({coverage!r})"
        )
        return [("Tests and coverage", command(source))]

    monkeypatch.setattr(runner, "_steps", steps)
    assert runner.start()
    state = finish(runner)
    assert state["status"] == "FAILED"
    assert state["counts"]["passed"] == 0
    assert state["counts"]["skipped"] == 2


def test_relative_workdir_is_resolved_before_child_changes_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runner = SuiteRunner("relative-desk", command("print('absolute artifact fixture')"))
    assert runner.start()
    state = finish(runner)
    assert state["status"] == "PASSED"
    assert os.path.isabs(state["log_path"])
    assert (tmp_path / "relative-desk" / "verification-journal.sqlite3").exists()


def test_guard_attachment_failure_stops_child_and_fails_run(tmp_path, monkeypatch):
    from astra.verification_process import ChildLifetime

    attached = []

    def cannot_attach(self, process):
        attached.append(process)
        raise OSError("synthetic guard failure")

    monkeypatch.setattr(ChildLifetime, "attach", cannot_attach)
    runner = SuiteRunner(tmp_path, command("import time; time.sleep(30)"))
    assert runner.start()
    assert finish(runner)["status"] == "FAILED"
    assert len(attached) == 1 and attached[0].poll() is not None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job lifetime regression")
def test_windows_crash_kills_verification_child_and_allows_recovery(tmp_path):
    import ctypes

    run_root = tmp_path / "crash-desk"
    child_ready = tmp_path / "child-ready"
    parent_ready = tmp_path / "parent-ready"
    crash_now = tmp_path / "crash-now"
    child_source = (
        "import os, time; from pathlib import Path; "
        f"Path({str(child_ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_source = (
        "import os, sys, time; from pathlib import Path; from astra.verification import SuiteRunner\n"
        f"runner = SuiteRunner(Path({str(run_root)!r}), [sys.executable, '-c', {child_source!r}])\n"
        "assert runner.start()\n"
        f"while runner._process is None or not Path({str(child_ready)!r}).exists():\n"
        "    if not runner.status()['running']: raise SystemExit(4)\n"
        "    time.sleep(0.01)\n"
        f"Path({str(parent_ready)!r}).touch()\n"
        f"while not Path({str(crash_now)!r}).exists(): time.sleep(0.01)\n"
        "os._exit(73)\n"
    )
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenProcess.restype = ctypes.c_void_p
    api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    api.WaitForSingleObject.restype = ctypes.c_uint32
    api.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    api.TerminateProcess.restype = ctypes.c_int
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    child_handle = None
    parent = subprocess.Popen(
        command(parent_source),
        env=_environment(tmp_path / "parent-env"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 15
        while not parent_ready.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                pytest.fail(f"crash fixture exited early: {parent.communicate()!r}")
            time.sleep(0.02)
        assert parent_ready.exists(), "guarded child did not become ready"
        # Retain a handle to this exact fixture process before the parent dies;
        # PID reuse cannot turn this assertion or cleanup into another process.
        child_handle = api.OpenProcess(0x100001, False, int(child_ready.read_text()))
        assert child_handle
        crash_now.touch()
        assert parent.wait(timeout=5) == 73
        assert api.WaitForSingleObject(child_handle, 5000) == 0
        restored = SuiteRunner(run_root, command("print('crash recovered')"))
        assert restored.status()["status"] == "INTERRUPTED"
        assert restored.start()
        assert finish(restored)["status"] == "PASSED"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        parent.communicate(timeout=5)
        if child_handle:
            if api.WaitForSingleObject(child_handle, 0) != 0:
                api.TerminateProcess(child_handle, 1)
                api.WaitForSingleObject(child_handle, 5000)
            api.CloseHandle(child_handle)


def test_runner_disables_inherited_live_access_and_isolates_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_RUN_LIVE_TESTS", "true")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-m live")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "synthetic-value-must-not-reach-child")
    source = (
        "import os, tempfile; from pathlib import Path; "
        "assert os.environ['PYTHON_DOTENV_DISABLED'] == '1'; "
        "assert os.environ['ASTRA_RUN_LIVE_TESTS'] == 'false'; "
        "assert os.environ['ASTRA_PERSISTENCE_BACKEND'] == 'memory'; "
        "assert 'PYTEST_ADDOPTS' not in os.environ; "
        "assert 'SUPABASE_SECRET_KEY' not in os.environ; "
        f"assert Path(tempfile.gettempdir()).is_relative_to(Path({str(tmp_path)!r})); "
        "print('offline environment verified')"
    )
    runner = SuiteRunner(tmp_path, command(source))
    assert runner.start()
    assert finish(runner)["status"] == "PASSED"


def test_log_memory_is_bounded_and_spawn_error_is_durable(tmp_path):
    runner = SuiteRunner(
        tmp_path, command("print('x' * 150000); print('tail-fixture')")
    )
    assert runner.start()
    state = finish(runner)
    assert len(state["log"]) <= 64000
    assert state["log"].endswith("tail-fixture")
    broken = SuiteRunner(tmp_path, [str(tmp_path / "missing-executable")])
    assert broken.start()
    assert finish(broken)["status"] == "FAILED"
    assert SuiteRunner(tmp_path).history()[0]["exit_code"] == -1
