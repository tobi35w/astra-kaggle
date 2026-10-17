"""Start or resume local historical research, then open its dashboard."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from astra.sandbox import serve

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "astra-training-four-pairs"
RESULTS = DATA_ROOT / "research-results.json"
PORT = 8787
URL = f"http://127.0.0.1:{PORT}/research"


def start_or_resume_evaluation() -> str:
    """Start one detached evaluator; its process lock rejects duplicate launches."""
    if RESULTS.is_file():
        try:
            report = json.loads(RESULTS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return f"Cannot resume: the saved research report is unreadable ({type(error).__name__})."
        if report.get("status") == "COMPLETED":
            return "The saved historical evaluation is already complete."
        command = [sys.executable, "-m", "scripts.evaluate_curated_history", "--resume"]
    else:
        command = [sys.executable, "-m", "scripts.evaluate_curated_history"]

    stdout_path = DATA_ROOT / "research-run.stdout.log"
    stderr_path = DATA_ROOT / "research-run.stderr.log"
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                creationflags=flags,
            )
        except OSError as error:
            return f"Could not start the historical evaluator: {error}"
    return "Historical evaluation started or resumed in the background."


def start_supabase_sync() -> None:
    """Run an idempotent uploader alongside research without owning its process."""
    stdout_path = DATA_ROOT / "research-supabase-sync.stdout.log"
    stderr_path = DATA_ROOT / "research-supabase-sync.stderr.log"
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
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
        except OSError as error:
            print(f"Could not start the Supabase sync worker ({type(error).__name__}).", flush=True)


def dashboard_is_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    start_supabase_sync()
    print(start_or_resume_evaluation(), flush=True)
    if dashboard_is_open():
        webbrowser.open(URL)
        print(f"Astra Research Sandbox: {URL}", flush=True)
        return
    serve(PORT, ".sandbox", open_browser=True, page="/research")


if __name__ == "__main__":
    main()
