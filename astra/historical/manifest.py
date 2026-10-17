"""Atomic, local metadata for resumable source downloads and datasets."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import DATASET_VERSION, TIMEFRAMES


class ManifestStore:
    def __init__(self, root: str | Path):
        self.path = Path(root) / "manifests" / "dukascopy_manifest.json"
        self._lock = threading.RLock()
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise TypeError("historical manifest must be an object")
                self.data = loaded
            except (OSError, json.JSONDecodeError):
                raise ValueError(
                    "historical manifest is unreadable; restore it before resuming"
                ) from None
        else:
            self.data = {}
        self.data.setdefault("dataset_version", DATASET_VERSION)
        self.data.setdefault("source", "Dukascopy public M1 BID candle feed")
        self.data.setdefault("timeframes", list(TIMEFRAMES))
        self.data.setdefault("symbols", {})
        self.data.setdefault("downloads", {})
        self.data.setdefault("runs", {})
        if any(
            not isinstance(self.data[key], dict)
            for key in ("symbols", "downloads", "runs")
        ):
            raise ValueError(
                "historical manifest has invalid dataset or download entries"
            )
        self.data.setdefault("created_at", datetime.now(UTC).isoformat())
        self.data["updated_at"] = datetime.now(UTC).isoformat()

    def unit(self, symbol: str, day: str) -> dict[str, Any] | None:
        with self._lock:
            value = self.data["downloads"].get(f"{symbol}:{day}")
            return dict(value) if isinstance(value, dict) else None

    def update_unit(self, symbol: str, day: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self.data["downloads"][f"{symbol}:{day}"] = entry
            self.data["updated_at"] = datetime.now(UTC).isoformat()
            self._save_locked()

    def begin_run(self, *, symbols, start_date, end_date, source: str) -> str:
        """Append an explicit run record without replacing cumulative unit history."""
        with self._lock:
            run_id = (
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            )
            started_at = datetime.now(UTC).isoformat()
            self.data["runs"][run_id] = {
                "run_id": run_id,
                "status": "RUNNING",
                "started_at": started_at,
                "scope": {
                    "symbols": list(symbols),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "source": source,
                },
            }
            self.data["latest_run_id"] = run_id
            self.data["updated_at"] = started_at
            self._save_locked()
            return run_id

    def finish_run(self, run_id: str, *, status: str, summary=None) -> None:
        """Finish an existing run record, preserving its original scope/start time."""
        with self._lock:
            run = self.data["runs"].get(run_id)
            if not isinstance(run, dict):
                raise KeyError(f"unknown acquisition run: {run_id}")
            run["status"] = status
            run["completed_at"] = datetime.now(UTC).isoformat()
            if summary is not None:
                run["summary"] = summary
            self.data["updated_at"] = run["completed_at"]
            self._save_locked()

    def update_dataset(
        self, *, symbols, start_date, end_date, results, run_id=None
    ) -> None:
        with self._lock:
            snapshot = {
                "symbols": results,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            encoded = json.dumps(snapshot, sort_keys=True, allow_nan=False)
            version = (
                DATASET_VERSION + ":" + hashlib.sha256(encoded.encode()).hexdigest()
            )
            self.data["dataset_version"] = version
            self.data.setdefault("versions", {})[version] = json.loads(encoded)
            self.data["symbols"] = results
            self.data["requested_symbols"] = list(symbols)
            self.data["start_date"] = start_date.isoformat()
            self.data["end_date"] = end_date.isoformat()
            self.data["latest_dataset"] = {
                "dataset_version": version,
                "symbols": list(symbols),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            if run_id is not None:
                run = self.data["runs"].get(run_id)
                if isinstance(run, dict):
                    run["dataset_version"] = version
            self.data["updated_at"] = datetime.now(UTC).isoformat()
            self._save_locked()

    def save(self) -> None:
        with self._lock:
            self.data["updated_at"] = datetime.now(UTC).isoformat()
            self._save_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        encoded = json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
