"""Open the separate Training Arena and ensure its offline workers are armed."""

from __future__ import annotations

import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "training-arena"
PORT = 8788
URL = f"http://127.0.0.1:{PORT}/"


def port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
            return True
    except OSError:
        return False


def start_process(module: str, args: list[str], label: str) -> None:
    stdout_path = DATA_ROOT / f"{label}.stdout.log"
    stderr_path = DATA_ROOT / f"{label}.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            [sys.executable, "-m", module, *args],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if not port_open():
            start_process("astra.training_arena", ["--host", "127.0.0.1", "--port", str(PORT)], "training-arena-web")
        start_process("scripts.run_training_arena_phase", [], "training-arena-phase")
        start_process("scripts.sync_training_arena_to_supabase", ["--watch"], "training-arena-sync")
        webbrowser.open(URL)
        print(f"Astra Training Arena opened at {URL}")
        print("The offline phase worker is armed and will wait for the current research evaluation.")
        print("The Supabase mirror queues locally while offline and retries when internet returns.")
        return 0
    except OSError as error:
        print(f"Could not start Training Arena ({type(error).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
