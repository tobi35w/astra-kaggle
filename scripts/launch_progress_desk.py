"""Open the local Evaluation Desk, reusing its existing server when available."""

from __future__ import annotations

import http.client
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8789
URL = f"http://127.0.0.1:{PORT}/"


def desk_available() -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        return response.status == 200 and response.getheader("Server", "").startswith(
            "AstraEvaluationDesk/"
        )
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def start_server() -> None:
    output = ROOT / "data" / "progress-desk"
    output.mkdir(parents=True, exist_ok=True)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with (
        (output / "server.stdout.log").open("a", encoding="utf-8") as stdout,
        (output / "server.stderr.log").open("a", encoding="utf-8") as stderr,
    ):
        subprocess.Popen(
            [sys.executable, "-m", "astra", "desk", "--no-browser"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )


def main() -> int:
    try:
        if not desk_available():
            start_server()
            deadline = time.monotonic() + 10
            while not desk_available():
                if time.monotonic() >= deadline:
                    print(
                        "Astra could not open its desk on port 8789. See data/progress-desk/server.stderr.log."
                    )
                    return 1
                time.sleep(0.1)
        webbrowser.open(URL)
        print(f"Astra Evaluation Desk: {URL}")
        return 0
    except OSError as error:
        print(f"Could not start the Evaluation Desk ({type(error).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
