"""One-command continuous acquisition of the post-curation Dukascopy tail.

Run from a machine with network access. Downloads M1 BID candles for the active
pairs from --start to the latest complete source day, normalizes, resamples to
M5/M15/H1/H4, and quality-checks the full requested range per symbol.

Example:
    .venv/Scripts/python.exe scripts/download_continuous_tail.py \
        --start 2026-09-01 --root data/astra-continuous-2026-tail
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.historical.config import HistoricalSettings
from astra.historical.dukascopy import DukascopyProvider
from astra.historical.pipeline import discover_latest, run_pipeline

ACTIVE_SYMBOLS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")


def run(
    start: date,
    *,
    end: date | None = None,
    root: Path,
    symbols=ACTIVE_SYMBOLS,
    provider=None,
    timeout: float = 15.0,
    workers: int = 1,
    request_interval: float = 15.0,
    retries: int = 1,
    read_timeout: float = 15.0,
    max_elapsed_seconds: float = 180.0,
    retry_jitter_seconds: float = 0.5,
    max_consecutive_failures: int = 3,
    progress=None,
) -> dict:
    """Acquire [start, end] for every symbol into one dataset root.

    Defaults are deliberately gentle (one worker, three seconds between source
    requests) because Dukascopy rate-limits aggressive crawlers. Returns a
    summary dict; per-symbol failures are isolated, never fatal.
    """
    symbols = tuple(symbols)
    provider = provider or DukascopyProvider()
    if end is None:
        end = discover_latest(provider, timeout=timeout)
    if end < start:
        raise ValueError("end date precedes start date")
    settings = HistoricalSettings(
        root=root,
        symbols=tuple(symbols),
        start_date=start,
        end_date=end,
        workers=workers,
        retries=retries,
        timeout=timeout,
        read_timeout=read_timeout,
        max_elapsed_seconds=max_elapsed_seconds,
        retry_jitter_seconds=retry_jitter_seconds,
        max_consecutive_failures=max_consecutive_failures,
    )
    summary = run_pipeline(
        settings,
        provider=provider,
        request_interval=request_interval,
        progress=progress,
    )
    pipeline_report = json.loads(
        Path(summary.report_json_path).read_text(encoding="utf-8")
    )
    requested_scope_complete = pipeline_report.get("requested_scope_complete", False)
    required_pair_scope_complete = set(symbols) == set(ACTIVE_SYMBOLS)
    complete = requested_scope_complete and required_pair_scope_complete
    run_id = summary.run_id or uuid4().hex
    report = {
        "run_id": run_id,
        "acquired_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "requested_scope_complete": requested_scope_complete,
        "required_pair_scope_complete": required_pair_scope_complete,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "root": str(root),
        "scope": {
            "symbols": list(symbols),
            "timeframes": ["M1", "M5", "M15", "H1", "H4"],
        },
        "coverage_reconciliation": pipeline_report.get("coverage_reconciliation", {}),
        "cumulative_manifest_status_counts": _manifest_status_counts(root),
        "symbols": {
            item.symbol: {
                "status": item.status,
                "errors": item.errors,
                "downloads": (
                    {
                        "requested": item.downloads.requested,
                        "downloaded": item.downloads.downloaded,
                        "cached": item.downloads.cached,
                        "missing": item.downloads.missing,
                        "failed": item.downloads.failed,
                        "invalid": item.downloads.invalid,
                        "deferred": item.downloads.deferred,
                        "failed_errors": dict(
                            Counter(
                                result.error
                                for result in item.downloads.results
                                if result.status == "failed"
                            )
                        ),
                    }
                    if item.downloads
                    else None
                ),
                "timeframes": {
                    timeframe: {"status": value.status, "rows": value.rows}
                    for timeframe, value in item.timeframes.items()
                },
            }
            for item in summary.symbols
        },
        "failed_files": [
            {
                "symbol": item.symbol,
                "date": result.day.isoformat(),
                "status": result.status,
                "reason": result.error,
            }
            for item in summary.symbols
            if item.downloads
            for result in item.downloads.results
            if result.status in {"failed", "invalid", "missing", "deferred"}
        ],
        "remaining_gaps": pipeline_report.get("coverage_reconciliation", {}).get(
            "missing_sessions", []
        ),
        "manifest": summary.manifest_path,
        "report_json": summary.report_json_path,
    }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "reports" / "acquisition-runs" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    _atomic_json(root / "acquisition-summary.json", report)
    return report


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _manifest_status_counts(root: Path) -> dict[str, int]:
    path = root / "manifests" / "dukascopy_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(
        entry.get("status", "unknown")
        for entry in manifest.get("downloads", {}).values()
        if isinstance(entry, dict)
    )
    return dict(counts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--root", default="data/astra-continuous-2026-tail")
    parser.add_argument("--symbols", nargs="+", default=list(ACTIVE_SYMBOLS))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel source workers; 1 is the rate-limit-friendly default",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=15.0,
        help="minimum seconds between source requests",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="connect timeout in seconds"
    )
    parser.add_argument(
        "--read-timeout", type=float, default=15.0, help="read timeout in seconds"
    )
    parser.add_argument(
        "--max-elapsed",
        type=float,
        default=180.0,
        help="maximum elapsed seconds per daily file",
    )
    parser.add_argument(
        "--retry-jitter",
        type=float,
        default=0.5,
        help="maximum retry jitter in seconds",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="additional attempts for transient source errors (0 disables retries)",
    )
    args = parser.parse_args(argv)

    def report_progress(result, completed, total):
        detail = f" ({result.error})" if result.error else ""
        print(
            f"[TAIL] {completed}/{total} {result.symbol} "
            f"{result.day.isoformat()} {result.status}{detail}",
            file=sys.stderr,
            flush=True,
        )

    try:
        report = run(
            args.start,
            end=args.end,
            root=Path(args.root),
            symbols=args.symbols,
            workers=args.workers,
            timeout=args.timeout,
            request_interval=args.request_interval,
            retries=args.retries,
            read_timeout=args.read_timeout,
            max_elapsed_seconds=args.max_elapsed,
            retry_jitter_seconds=args.retry_jitter,
            max_consecutive_failures=args.max_consecutive_failures,
            progress=report_progress,
        )
    except KeyboardInterrupt:
        _write_interrupted_summary(Path(args.root), args.start, args.end, args.symbols)
        print(
            "[TAIL] Interrupted. Completed daily files are safe; rerun the same command to resume.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__}))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "symbols": report["symbols"],
            },
            indent=2,
        )
    )
    return int(report["status"] != "COMPLETE")


def _write_interrupted_summary(root: Path, start: date, end, symbols) -> None:
    manifest_path = root / "manifests" / "dukascopy_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    # Interrupts can happen during discovery, before a new manifest run exists.
    # A receipt always has its own identity so it cannot overwrite a prior run.
    run_id = uuid4().hex
    report = {
        "run_id": run_id,
        "interruption_receipt": True,
        "manifest_latest_run_id": manifest.get("latest_run_id"),
        "status": "INTERRUPTED",
        "acquired_at": datetime.now(UTC).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat() if end else None,
        "scope": {
            "symbols": list(symbols),
            "timeframes": ["M1", "M5", "M15", "H1", "H4"],
        },
        "cumulative_manifest_status_counts": _manifest_status_counts(root),
        "manifest": str(manifest_path),
    }
    if root.exists():
        run_path = root / "reports" / "acquisition-runs" / f"{run_id}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_path, report)
        _atomic_json(root / "acquisition-summary.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
