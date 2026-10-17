"""Durable, offline verification runs for the local Evaluation Desk.

Each run has isolated coverage, JUnit and log artifacts. Lifecycle snapshots are
append-only Journal records; an operating-system file lock serializes runners
using the same desk directory. An abandoned run becomes INTERRUPTED on restart.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from .adapters import SQLiteRepository
from .repositories import Entity, Journal
from .verification_process import ChildLifetime

ROOT = Path(__file__).resolve().parents[1]
MAX_LOG_LINES = 400
MAX_LOG_BYTES = 64_000
RECORD_KIND = "verification_run"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _RunLock:
    """Nonblocking file lock, automatically released when its process exits."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: BinaryIO | None = None

    def acquire(self) -> bool:
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _environment(artifact_dir: Path) -> dict[str, str]:
    # A GUI process can inherit test opt-ins or pytest-cov instrumentation.
    # Never propagate those into a new offline suite or a nested test fixture.
    excluded = ("ASTRA_", "SUPABASE_", "HISTORICAL_", "COV_CORE_", "COVERAGE_")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(excluded) and not key.startswith("PYTEST_")
    }
    temporary = artifact_dir / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(
        PYTHON_DOTENV_DISABLED="1",
        ASTRA_RUN_LIVE_TESTS="false",
        ASTRA_TRADING_ENABLED="false",
        ASTRA_ML_ENABLED="false",
        ASTRA_AI_ENABLED="false",
        ASTRA_PERSISTENCE_BACKEND="memory",
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
        COVERAGE_FILE=str(artifact_dir / ".coverage"),
        TMP=str(temporary),
        TEMP=str(temporary),
        TMPDIR=str(temporary),
    )
    return env


def _report_counts(path: Path) -> dict | None:
    try:
        if path.stat().st_size > 16_000_000:
            return None
        raw = path.read_text(encoding="utf-8")
        if "<!DOCTYPE" in raw or "<!ENTITY" in raw:
            return None
        root = ET.fromstring(raw)
        suites = [
            suite for suite in root.iter("testsuite") if not suite.findall("testsuite")
        ]
        if not suites:
            return None
        totals = {
            key: sum(int(suite.get(attribute, "0")) for suite in suites)
            for key, attribute in (
                ("tests", "tests"),
                ("failed", "failures"),
                ("errors", "errors"),
                ("skipped", "skipped"),
            )
        }
        totals["passed"] = totals["tests"] - sum(
            totals[key] for key in ("failed", "errors", "skipped")
        )
        return totals if all(value >= 0 for value in totals.values()) else None
    except (OSError, UnicodeError, ET.ParseError, ValueError):
        return None


def _coverage_percentage(path: Path) -> float | None:
    try:
        value = float(
            json.loads(path.read_text(encoding="utf-8"))["totals"]["percent_covered"]
        )
        return round(value, 2) if math.isfinite(value) and 0 <= value <= 100 else None
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None


class SuiteRunner:
    """Run pytest with coverage and both Ruff checks, preserving legacy API."""

    def __init__(self, workdir: Path, command: list[str] | None = None):
        self._workdir = Path(workdir).resolve()
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._command = command
        self._lock = threading.RLock()
        self._lease: _RunLock | None = None
        self._process: subprocess.Popen | None = None
        self._database = self._workdir / "verification-journal.sqlite3"
        self._state: dict[str, Any] = {
            "run_id": None,
            "status": "NOT_STARTED",
            "running": False,
            "phase": "Ready to verify",
            "phases": [],
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "counts": None,
            "coverage_pct": None,
            "log": "",
            "log_path": None,
            "last_line": "",
            "bytes": 0,
        }
        self._refresh()

    def _saved_runs(self) -> list[dict]:
        if not self._database.exists():
            return []
        repository = SQLiteRepository(str(self._database))
        try:
            rows = repository.query(Entity.AUDIT, strategy=RECORD_KIND)
        finally:
            repository.close()
        latest: dict[str, dict] = {}
        for row in rows:
            if row.payload.get("kind") == RECORD_KIND:
                snapshot = row.payload.get("run", {})
                if snapshot.get("run_id"):
                    latest[snapshot["run_id"]] = snapshot
        return sorted(latest.values(), key=lambda run: run["started_at"], reverse=True)

    def _persist(self) -> None:
        snapshot = {key: value for key, value in self._state.items() if key != "log"}
        repository = SQLiteRepository(str(self._database))
        try:
            Journal(repository).record(
                Entity.AUDIT,
                datetime.now(UTC),
                {"kind": RECORD_KIND, "run": snapshot},
                strategy=RECORD_KIND,
                correlation_id=self._state["run_id"],
            )
        finally:
            repository.close()

    def _log_tail(self, snapshot: dict) -> str:
        # Only resolve the artifact belonging to this run, never arbitrary paths
        # from a modified saved report.
        run_id = snapshot.get("run_id")
        if not isinstance(run_id, str) or Path(run_id).name != run_id:
            return ""
        path = self._workdir / "test-runs" / f"{run_id}.log"
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - MAX_LOG_BYTES))
                return "\n".join(
                    handle.read()
                    .decode("utf-8", errors="replace")
                    .splitlines()[-MAX_LOG_LINES:]
                )
        except OSError:
            return ""

    def _refresh(self) -> None:
        if self._lease is not None or self._state.get("storage_error"):
            return
        saved = self._saved_runs()
        if not saved:
            return
        self._state = {**saved[0], "log": self._log_tail(saved[0])}
        if not self._state["running"]:
            return
        lease = _RunLock(self._workdir / "verification.lock")
        if not lease.acquire():
            return
        try:
            # Read again under the lease: another runner may have just finished.
            current = self._saved_runs()[0]
            self._state = {**current, "log": self._log_tail(current)}
            if self._state["running"]:
                self._state.update(
                    status="INTERRUPTED",
                    running=False,
                    phase="Interrupted before completion",
                    finished_at=_now(),
                    exit_code=None,
                )
                for phase in self._state["phases"]:
                    if phase["status"] == "RUNNING":
                        phase.update(status="INTERRUPTED", finished_at=_now())
                self._persist()
        finally:
            lease.release()

    def _steps(self, run_id: str) -> list[tuple[str, list[str]]]:
        if self._command is not None:
            return [("Custom verification", self._command)]
        artifacts = self._workdir / "test-runs" / run_id
        return [
            (
                "Tests and coverage",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--cov=astra",
                    "--cov-report=term-missing",
                    f"--cov-report=json:{artifacts / 'coverage.json'}",
                    f"--junitxml={artifacts / 'junit.xml'}",
                    "-v",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={self._workdir / 'test-tmp' / run_id}",
                ],
            ),
            ("Ruff lint", [sys.executable, "-m", "ruff", "check", "astra", "tests"]),
            (
                "Ruff format",
                [sys.executable, "-m", "ruff", "format", "--check", "astra", "tests"],
            ),
        ]

    def start(self) -> bool:
        with self._lock:
            if self._lease is not None:
                return False
            lease = _RunLock(self._workdir / "verification.lock")
            if not lease.acquire():
                self._refresh()
                return False
            self._lease = lease
            try:
                run_id = (
                    datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                    + "-"
                    + uuid4().hex[:8]
                )
                artifacts = self._workdir / "test-runs" / run_id
                artifacts.mkdir(parents=True)
                (self._workdir / "test-tmp").mkdir(parents=True, exist_ok=True)
                steps = self._steps(run_id)
                self._state = {
                    "run_id": run_id,
                    "status": "RUNNING",
                    "running": True,
                    "phase": "Starting verification",
                    "phases": [
                        {
                            "name": name,
                            "status": "QUEUED",
                            "started_at": None,
                            "finished_at": None,
                            "exit_code": None,
                        }
                        for name, _ in steps
                    ],
                    "started_at": _now(),
                    "finished_at": None,
                    "exit_code": None,
                    "counts": None,
                    "coverage_pct": None,
                    "log": "",
                    "log_path": str(self._workdir / "test-runs" / f"{run_id}.log"),
                    "junit_path": str(artifacts / "junit.xml")
                    if self._command is None
                    else None,
                    "coverage_path": str(artifacts / "coverage.json")
                    if self._command is None
                    else None,
                    "last_line": "",
                    "bytes": 0,
                }
                self._persist()
                threading.Thread(
                    target=self._run, args=(steps, artifacts), daemon=True
                ).start()
                return True
            except Exception:  # noqa: BLE001 - release the lease on every startup failure.
                self._state.update(
                    running=False,
                    status="FAILED",
                    exit_code=-1,
                    phase="Verification could not start",
                    storage_error=True,
                )
                self._lease.release()
                self._lease = None
                raise RuntimeError(
                    "Verification could not start; check local journal storage."
                ) from None

    def _run(self, steps: list[tuple[str, list[str]]], artifacts: Path) -> None:
        lines: deque[str] = deque(maxlen=MAX_LOG_LINES)
        failure = 0
        try:
            with Path(self._state["log_path"]).open("w", encoding="utf-8") as handle:

                def emit(line: str) -> None:
                    lines.append(line.rstrip("\r\n"))
                    handle.write(line)
                    handle.flush()
                    with self._lock:
                        self._state.update(
                            log="\n".join(lines)[-MAX_LOG_BYTES:],
                            last_line=lines[-1][:200],
                            bytes=handle.tell(),
                        )

                for index, (name, command) in enumerate(steps):
                    with self._lock:
                        phase = self._state["phases"][index]
                        phase.update(status="RUNNING", started_at=_now())
                        self._state["phase"] = name
                        self._persist()
                    if self._command is None:
                        emit(f"\n{name}\n")
                    try:
                        with ChildLifetime() as lifetime:
                            process = subprocess.Popen(
                                command,
                                cwd=str(ROOT),
                                env=_environment(artifacts),
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW
                                if os.name == "nt"
                                else 0,
                            )
                            try:
                                lifetime.attach(process)
                                with self._lock:
                                    self._process = process
                                assert process.stdout is not None
                                with process.stdout:
                                    while chunk := process.stdout.readline(4096):
                                        emit(chunk)
                                code = process.wait()
                            finally:
                                if process.poll() is None:
                                    process.kill()
                                    process.wait(timeout=5)
                                if process.stdout is not None:
                                    process.stdout.close()
                    except OSError as error:
                        emit(f"Could not start {name}: {type(error).__name__}\n")
                        code = -1
                    with self._lock:
                        self._process = None
                        if self._command is None and index == 0:
                            self._state["counts"] = _report_counts(
                                artifacts / "junit.xml"
                            )
                            self._state["coverage_pct"] = _coverage_percentage(
                                artifacts / "coverage.json"
                            )
                            if code == 0 and (
                                self._state["counts"] is None
                                or self._state["coverage_pct"] is None
                            ):
                                code = -1
                                emit("Verification reports missing or unreadable.\n")
                            elif code == 0 and (
                                self._state["counts"]["tests"]
                                == self._state["counts"]["skipped"]
                                or self._state["counts"]["failed"] > 0
                                or self._state["counts"]["errors"] > 0
                            ):
                                code = 1
                                emit(
                                    "JUnit reports failed checks or no executed tests.\n"
                                )
                        phase.update(
                            status="PASSED" if code == 0 else "FAILED",
                            finished_at=_now(),
                            exit_code=code,
                        )
                        failure = failure or code
                        self._persist()
        except Exception as error:  # noqa: BLE001 - retain a sanitized failed lifecycle.
            failure = -1
            with self._lock:
                message = f"Verification runner failed: {type(error).__name__}"
                self._state["log"] = (self._state["log"] + "\n" + message)[
                    -MAX_LOG_BYTES:
                ]
                self._state["last_line"] = message
                try:
                    with Path(self._state["log_path"]).open(
                        "a", encoding="utf-8"
                    ) as failed_log:
                        failed_log.write(message + "\n")
                except OSError:
                    pass
                for phase in self._state["phases"]:
                    if phase["status"] == "RUNNING":
                        phase.update(status="FAILED", exit_code=-1, finished_at=_now())
        finally:
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    try:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
                            self._process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        failure = -1
                self._process = None
                self._state.update(
                    running=False,
                    status="PASSED" if failure == 0 else "FAILED",
                    phase="Verification complete"
                    if failure == 0
                    else "Verification needs attention",
                    finished_at=_now(),
                    exit_code=failure,
                )
                try:
                    self._persist()
                except Exception:  # noqa: BLE001 - don't report a nondurable success.
                    self._state.update(
                        status="FAILED",
                        exit_code=-1,
                        phase="Verification journal write failed",
                        storage_error=True,
                    )
                finally:
                    assert self._lease is not None
                    self._lease.release()
                    self._lease = None

    def status(self) -> dict:
        with self._lock:
            self._refresh()
            return copy.deepcopy(self._state)

    def history(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return copy.deepcopy(self._saved_runs()[:10])
