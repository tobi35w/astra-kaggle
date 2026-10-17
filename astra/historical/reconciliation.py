"""Read-only reconciliation of acquisition claims against raw daily files."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ..data import DataValidationError
from .downloader import _file, iter_days
from .dukascopy import decode_m1_bi5
from .validation import _market_open


def _open_minutes(day: date) -> tuple[datetime, ...]:
    start = datetime.combine(day, time.min, UTC)
    return tuple(
        start + timedelta(minutes=minute)
        for minute in range(1440)
        if _market_open(start + timedelta(minutes=minute))
    )


def reconcile_acquisition(root: str | Path, symbols, start: date, end: date) -> dict:
    """Check expected open-session M1 units, manifest claims, and raw-file bytes.

    This function is read-only. A failed manifest entry on a fully closed FX
    date is reported as a non-blocking closed session; no missing bars are
    fabricated and no manifest state is repaired here.
    """
    root = Path(root)
    symbols = tuple(symbols)
    manifest_path = root / "manifests" / "dukascopy_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "status": "FAIL",
                "errors": ["manifest unreadable"],
                "manifest_status_counts": {},
                "expected_session_units": 0,
                "verified_session_units": 0,
                "closed_session_units": 0,
                "missing_sessions": [],
                "invalid_files": [],
                "untracked_files": [],
            }
    else:
        manifest = {}
    entries = manifest.get("downloads", {})
    if not isinstance(entries, dict):
        return {
            "status": "FAIL",
            "errors": ["manifest downloads field is invalid"],
            "manifest_status_counts": {},
            "expected_session_units": 0,
            "verified_session_units": 0,
            "closed_session_units": 0,
            "missing_sessions": [],
            "invalid_files": [],
            "untracked_files": [],
        }

    status_counts: dict[str, int] = {}
    for entry in entries.values():
        status = (
            entry.get("status", "unknown") if isinstance(entry, dict) else "invalid"
        )
        status_counts[status] = status_counts.get(status, 0) + 1

    expected_count = verified_count = closed_count = 0
    missing_sessions = []
    invalid_files = []
    referenced: set[Path] = set()
    for symbol in symbols:
        for day in iter_days(start, end):
            open_times = _open_minutes(day)
            key = f"{symbol}:{day.isoformat()}"
            entry = entries.get(key)
            path = _file(root, symbol, day)
            if not open_times:
                closed_count += 1
                # Some historical manifests contain downloaded weekend files.
                # They are not required for coverage, but if present they are
                # still tracked and integrity-checked rather than called orphaned.
                if path.is_file():
                    referenced.add(path.resolve())
                    try:
                        path.resolve().relative_to(root.resolve())
                    except ValueError:
                        invalid_files.append(
                            {"unit": key, "reason": "raw path escapes dataset root"}
                        )
                        continue
                    try:
                        raw = path.read_bytes()
                        digest = hashlib.sha256(raw).hexdigest()
                        candles = decode_m1_bi5(raw, symbol, day)
                        if (
                            not isinstance(entry, dict)
                            or entry.get("status") != "downloaded"
                            or entry.get("path", str(path.relative_to(root)))
                            != str(path.relative_to(root))
                            or digest != entry.get("checksum")
                            or len(candles) != entry.get("rows")
                        ):
                            invalid_files.append(
                                {
                                    "unit": key,
                                    "reason": "closed-session file integrity mismatch",
                                }
                            )
                    except (DataValidationError, ValueError, OSError) as error:
                        invalid_files.append(
                            {
                                "unit": key,
                                "reason": f"closed-session file invalid: {type(error).__name__}",
                            }
                        )
                elif not isinstance(entry, dict) or entry.get("status") not in {
                    "closed",
                    "failed",
                    "deferred",
                    "interrupted",
                }:
                    # Closed sessions need no raw file; old failed/deferred
                    # entries are retained as history and do not create gaps.
                    invalid_files.append(
                        {"unit": key, "reason": "closed session not recorded as closed"}
                    )
                continue

            expected_count += 1
            if not path.is_file():
                missing_sessions.append(
                    {
                        "unit": key,
                        "reason": entry.get("error", "raw file missing")
                        if isinstance(entry, dict)
                        else "manifest entry missing",
                    }
                )
                continue
            referenced.add(path.resolve())
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError:
                invalid_files.append(
                    {"unit": key, "reason": "raw path escapes dataset root"}
                )
                continue
            if not isinstance(entry, dict) or entry.get("status") != "downloaded":
                missing_sessions.append(
                    {
                        "unit": key,
                        "reason": "file exists without a downloaded manifest claim",
                    }
                )
                continue
            if entry.get("path", str(path.relative_to(root))) != str(
                path.relative_to(root)
            ):
                invalid_files.append({"unit": key, "reason": "manifest path mismatch"})
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != entry.get("checksum"):
                invalid_files.append({"unit": key, "reason": "SHA-256 mismatch"})
                continue
            try:
                candles = decode_m1_bi5(raw, symbol, day)
            except (DataValidationError, ValueError, OSError) as error:
                invalid_files.append(
                    {
                        "unit": key,
                        "reason": f"BI5 validation failed: {type(error).__name__}",
                    }
                )
                continue
            if (
                not candles
                or len(candles) != entry.get("rows")
                or entry.get("size_bytes", len(raw)) != len(raw)
            ):
                invalid_files.append(
                    {"unit": key, "reason": "row count or file size mismatch"}
                )
                continue
            actual_open_times = {
                candle.timestamp - timedelta(minutes=1)
                for candle in candles
                if _market_open(candle.timestamp - timedelta(minutes=1))
            }
            missing_minutes = [
                stamp for stamp in open_times if stamp not in actual_open_times
            ]
            if missing_minutes:
                missing_sessions.append(
                    {
                        "unit": key,
                        "reason": "verified file is missing open-session M1 candles",
                        "missing_open_minutes": len(missing_minutes),
                    }
                )
                continue
            verified_count += 1

    raw_root = root / "raw" / "dukascopy"
    raw_files = list(raw_root.rglob("*.bi5")) if raw_root.is_dir() else []
    untracked_files = [
        str(path) for path in raw_files if path.resolve() not in referenced
    ]
    partial_files = list(raw_root.rglob("*.part")) if raw_root.is_dir() else []
    errors = []
    if missing_sessions:
        errors.append("required open-session units are missing or unapproved")
    if invalid_files:
        errors.append("raw files failed integrity or manifest validation")
    if untracked_files:
        errors.append("untracked raw BI5 files exist")
    if partial_files:
        errors.append("partial download files exist")
    if expected_count != verified_count:
        errors.append("verified coverage does not equal required open-session coverage")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scope": {
            "symbols": list(symbols),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timeframe": "M1",
        },
        "manifest_status_counts": status_counts,
        "expected_session_units": expected_count,
        "verified_session_units": verified_count,
        "closed_session_units": closed_count,
        "missing_sessions": missing_sessions,
        "invalid_files": invalid_files,
        "untracked_files": untracked_files,
        "partial_files": [str(path) for path in partial_files],
    }
