"""Mirror the active historical evaluation into Astra's private Supabase journal.

The evaluator keeps its local durable journal. This sidecar reads committed rows,
uploads them idempotently, and retries outages without touching evaluator state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from astra.adapters import repository_from_settings
from astra.config import Settings
from astra.repositories import Entity, PersistenceError, Record

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "astra-training-four-pairs"
JOURNAL_PATH = DATA_ROOT / "research-journal.sqlite3"
RESULTS_PATH = DATA_ROOT / "research-results.json"
PROGRESS_PATH = DATA_ROOT / "research-progress.json"
STATE_PATH = DATA_ROOT / "supabase-sync-state.json"
STATUS_PATH = DATA_ROOT / "research-supabase-sync-status.json"
LOCK_PATH = DATA_ROOT / "supabase-sync.lock"
BATCH_SIZE = 50


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def availability_time(item: dict, fallback: str) -> datetime:
    for key in ("last_close", "test_end", "trained_at", "available_at"):
        value = item.get(key)
        if value:
            try:
                parsed = datetime.fromisoformat(str(value))
                if parsed.tzinfo is not None:
                    return parsed.astimezone(UTC)
            except ValueError:
                pass
    parsed = datetime.fromisoformat(fallback)
    return parsed.astimezone(UTC) if parsed.tzinfo else datetime.now(UTC)


def report_records(report: dict, progress: dict, uploaded: set[str]) -> list[Record]:
    records: list[Record] = []
    run_id = str(report.get("started_at") or "astra-curated-history")
    report_time = str(progress.get("updated_at") or report.get("updated_at") or now_iso())
    manifest = {key: value for key, value in report.items() if key != "items"}
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    manifest_id = str(uuid5(NAMESPACE_URL, f"astra-evaluation-manifest:{run_id}:{manifest_digest}"))
    if manifest_id not in uploaded:
        records.append(
            Record(
                Entity.AUDIT,
                availability_time({}, report_time),
                {"event": "historical_evaluation_manifest", "run_id": run_id, "report": manifest},
                record_id=manifest_id,
            )
        )
    for item in report.get("items", []):
        if not isinstance(item, dict):
            continue
        identity = ":".join(
            str(item.get(key, ""))
            for key in ("kind", "pair", "segment", "timeframe", "strategy")
        )
        if item.get("kind") == "ML_TRAINING":
            identity += ":" + str(item.get("dataset_version", ""))
        record_id = str(uuid5(NAMESPACE_URL, f"astra-evaluation:{run_id}:{identity}"))
        if record_id in uploaded:
            continue
        records.append(
            Record(
                Entity.AUDIT,
                availability_time(item, report_time),
                {"event": "historical_evaluation_result", "run_id": run_id, "result": item},
                pair=str(item.get("pair", "")),
                timeframe=str(item.get("timeframe", "")),
                strategy=str(item.get("strategy", "")),
                record_id=record_id,
            )
        )
    stamp = str(progress.get("updated_at", ""))
    if stamp:
        progress_id = str(uuid5(NAMESPACE_URL, f"astra-evaluation-progress:{run_id}:{stamp}"))
        if progress_id not in uploaded:
            records.append(
                Record(
                    Entity.AUDIT,
                    availability_time({}, stamp),
                    {"event": "historical_evaluation_progress", "run_id": run_id, "progress": progress},
                    record_id=progress_id,
                )
            )
    return records


def committed_journal_records(uploaded: set[str]) -> list[Record]:
    if not JOURNAL_PATH.exists():
        return []
    # Read-only connection avoids DDL, writes, and evaluator-side locking.
    uri = JOURNAL_PATH.resolve().as_uri() + "?mode=ro"
    for attempt in range(4):
        try:
            with sqlite3.connect(uri, uri=True, timeout=2) as connection:
                rows = connection.execute(
                    "SELECT row_json FROM astra_records ORDER BY occurred_at, record_id"
                ).fetchall()
            return [
                Record.from_row(json.loads(row[0]))
                for row in rows
                if json.loads(row[0]).get("record_id") not in uploaded
            ]
        except sqlite3.OperationalError:
            if attempt == 3:
                raise
            time.sleep(0.25 * (attempt + 1))
    return []


def sync_once(remote, state: dict) -> dict:
    uploaded = set(state.get("uploaded_ids", []))
    records = committed_journal_records(uploaded)
    report = read_json(RESULTS_PATH)
    progress = read_json(PROGRESS_PATH)
    records.extend(report_records(report, progress, uploaded))
    total = len(records)
    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        # Keep request size bounded when trade detail payloads are large.
        current: list[Record] = []
        size = 0
        batches: list[list[Record]] = []
        for record in batch:
            encoded_size = len(json.dumps(record.row(), allow_nan=False).encode("utf-8"))
            if current and size + encoded_size > 400_000:
                batches.append(current)
                current, size = [], 0
            current.append(record)
            size += encoded_size
        if current:
            batches.append(current)
        for chunk in batches:
            remote.append_batch(tuple(chunk))
            uploaded.update(record.record_id for record in chunk)
            state["uploaded_ids"] = sorted(uploaded)
            state["last_success_at"] = now_iso()
            write_json(STATE_PATH, state)
    state.update(
        {
            "uploaded_ids": sorted(uploaded),
            "last_success_at": now_iso(),
            "last_local_journal_rows": len(uploaded),
            "last_report_items": len(report.get("items", [])),
            "evaluation_status": report.get("status", progress.get("status", "UNKNOWN")),
        }
    )
    write_json(STATE_PATH, state)
    write_json(
        STATUS_PATH,
        {
            "status": "CONNECTED",
            "last_success_at": state["last_success_at"],
            "uploaded_records": len(uploaded),
            "evaluation_status": state["evaluation_status"],
            "last_error": "",
        },
    )
    return state


def claim_worker() -> bool:
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            return False
        except (OSError, ValueError):
            LOCK_PATH.unlink(missing_ok=True)
            return claim_worker()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="keep syncing new records")
    parser.add_argument("--interval", type=float, default=4.0)
    args = parser.parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.watch and not claim_worker():
        print("Supabase research sync is already running.", flush=True)
        return 0
    state = read_json(STATE_PATH)
    try:
        remote = repository_from_settings(Settings.from_env())
        if not remote.healthy():
            raise PersistenceError("Supabase connection/write readiness check failed")
        print("Supabase research sync connected.", flush=True)
        while True:
            try:
                state = sync_once(remote, state)
                print(
                    f"Synced {state['last_local_journal_rows']} journal rows and "
                    f"{state['last_report_items']} detailed results; "
                    f"evaluation {state['evaluation_status']}.",
                    flush=True,
                )
            except (PersistenceError, OSError, sqlite3.Error, ValueError) as error:
                # The exception classes are safe; never log settings, headers, or response bodies.
                write_json(
                    STATUS_PATH,
                    {
                        "status": "RETRYING",
                        "last_success_at": state.get("last_success_at", ""),
                        "uploaded_records": len(state.get("uploaded_ids", [])),
                        "last_error": type(error).__name__,
                        "retry_in_seconds": max(2.0, args.interval),
                    },
                )
                print(f"Supabase sync retrying after {type(error).__name__}.", flush=True)
            if not args.watch:
                return 0
            time.sleep(max(2.0, args.interval))
    except (PersistenceError, ValueError) as error:
        write_json(STATUS_PATH, {"status": "ERROR", "last_error": type(error).__name__})
        print(f"Supabase sync could not start ({type(error).__name__}).", file=sys.stderr)
        return 1
    finally:
        if args.watch:
            LOCK_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
