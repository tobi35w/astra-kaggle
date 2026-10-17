"""Retryable online mirror for Training Arena's offline journal and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from astra.adapters import repository_from_settings
from astra.config import Settings
from astra.repositories import Entity, PersistenceError, Record

ROOT = Path(__file__).resolve().parents[1]
ARENA_ROOT = ROOT / "data" / "training-arena"
JOURNAL_PATH = ARENA_ROOT / "arena-journal.sqlite3"
RESULTS_PATH = ARENA_ROOT / "arena-results.json"
STATE_PATH = ARENA_ROOT / "supabase-sync-state.json"
SYNC_STATUS_PATH = ARENA_ROOT / "supabase-sync-status.json"
LOCK_PATH = ARENA_ROOT / "supabase-sync.lock"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def journal_rows(uploaded):
    if not JOURNAL_PATH.exists():
        return []
    uri = JOURNAL_PATH.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as connection:
        rows = connection.execute("SELECT row_json FROM astra_records ORDER BY occurred_at,record_id").fetchall()
    records = []
    for (raw,) in rows:
        row = json.loads(raw)
        if row.get("record_id") not in uploaded:
            records.append(Record.from_row(row))
    return records


def report_rows(uploaded, report):
    records = []
    campaign = str(report.get("campaign_id", "astra-training-arena-yllvar-v1"))
    summary = {key: value for key, value in report.items() if key != "tests"}
    digest = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    manifest_id = str(uuid5(NAMESPACE_URL, f"training-arena-manifest:{campaign}:{digest}"))
    if manifest_id not in uploaded:
        records.append(Record(Entity.AUDIT, datetime.now(UTC), {"event":"training_arena_campaign_snapshot","campaign_id":campaign,"report":summary}, record_id=manifest_id))
    for item in report.get("models", []):
        identity = str(item.get("id", ""))
        if not identity:
            continue
        record_id = str(uuid5(NAMESPACE_URL, "training-arena-model:" + identity))
        if record_id in uploaded:
            continue
        stamp = item.get("validation", {}).get("trained_at")
        occurred = datetime.fromisoformat(stamp) if stamp else datetime.now(UTC)
        records.append(Record(Entity.AUDIT, occurred, {"event":"training_arena_model_result","campaign_id":campaign,"model":item}, pair=item.get("pair", ""), timeframe=item.get("timeframe", ""), record_id=record_id))
    for item in report.get("tests", []):
        identity = str(item.get("id", ""))
        if not identity:
            continue
        record_id = str(uuid5(NAMESPACE_URL, "training-arena-test:" + identity))
        if record_id in uploaded:
            continue
        stamp = item.get("test_end")
        occurred = datetime.fromisoformat(stamp) if stamp else datetime.now(UTC)
        records.append(Record(Entity.AUDIT, occurred, {"event":"training_arena_oos_test_result","campaign_id":campaign,"result":item}, pair=item.get("pair", ""), timeframe=item.get("timeframe", ""), strategy=item.get("strategy", ""), record_id=record_id))
    return records


def sync_once(remote, state):
    uploaded = set(state.get("uploaded_ids", []))
    results = read_json(RESULTS_PATH)
    records = journal_rows(uploaded) + report_rows(uploaded, results)
    for start in range(0, len(records), 50):
        batch = records[start:start+50]
        remote.append_batch(tuple(batch))
        uploaded.update(record.record_id for record in batch)
        state["uploaded_ids"] = sorted(uploaded)
        state["last_success_at"] = datetime.now(UTC).isoformat()
        write_json(STATE_PATH, state)
    state.update(uploaded_ids=sorted(uploaded), last_success_at=datetime.now(UTC).isoformat(), evaluation_status=results.get("status", "WAITING"))
    write_json(STATE_PATH, state)
    write_json(SYNC_STATUS_PATH, {"status":"CONNECTED","uploaded_records":len(uploaded),"last_success_at":state["last_success_at"],"last_error":""})
    return state


def claim_worker():
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            os.kill(int(LOCK_PATH.read_text(encoding="utf-8").strip()), 0)
            return False
        except (OSError, ValueError):
            LOCK_PATH.unlink(missing_ok=True)
            return claim_worker()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    ARENA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.watch and not claim_worker():
        return 0
    state = read_json(STATE_PATH)
    try:
        remote = repository_from_settings(Settings.from_env())
        while True:
            try:
                if not remote.healthy():
                    raise PersistenceError("Supabase connection/write readiness check failed")
                state = sync_once(remote, state)
            except (PersistenceError, OSError, sqlite3.Error, ValueError) as error:
                write_json(SYNC_STATUS_PATH, {"status":"RETRYING","uploaded_records":len(state.get("uploaded_ids",[])),"last_success_at":state.get("last_success_at",""),"last_error":type(error).__name__})
            if not args.watch:
                return 0
            time.sleep(max(3, args.interval))
    except (PersistenceError, ValueError) as error:
        write_json(SYNC_STATUS_PATH, {"status":"ERROR","last_error":type(error).__name__})
        return 1
    finally:
        if args.watch:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
