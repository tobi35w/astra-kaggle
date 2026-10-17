"""End-to-end local historical-data workflow and quality reports."""

from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from .config import HistoricalSettings
from .downloader import HistoricalDownloader
from .dukascopy import DukascopyProvider, decode_m1_bi5
from .manifest import ManifestStore
from .models import PipelineSummary, SymbolPipelineResult
from .normalizer import normalize_symbol
from .reconciliation import reconcile_acquisition
from .resampler import resample
from .validation import inspect_candles, inspect_csv

_OUTPUTS = ("M1", "M5", "M15", "H1", "H4")


def _write_candles(path, candles, *, volume_unit=None, price_basis="bid"):
    if price_basis not in {"mid", "bid", "ask"}:
        raise ValueError("unsupported price basis")
    candles = tuple(candles)
    volume_types = {c.volume_type for c in candles}
    if len(volume_types) > 1:
        raise ValueError("cannot write mixed candle volume types to one CSV")
    volume_type = next(iter(volume_types), "tick")
    volume_column = "tick_volume" if volume_type == "tick" else "volume"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fields = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            volume_column,
            "volume_type",
            "price_basis",
        ]
        if volume_unit:
            fields.append("volume_unit")
        writer.writerow(fields)
        for c in candles:
            row = [
                c.timestamp.isoformat(),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.volume_type,
                price_basis,
            ]
            if volume_unit:
                row.append(volume_unit)
            writer.writerow(row)
    temporary.replace(path)


def discover_latest(
    provider=None, *, timeout=15, lookback_days=14, symbol="EURUSD", now=None
) -> date:
    if not 1 <= lookback_days <= 14:
        raise ValueError("latest-day discovery is limited to 1–14 probes")
    provider = provider or DukascopyProvider()
    current = (now or datetime.now(UTC)).date()
    for offset in range(1, lookback_days + 1):
        candidate = current - timedelta(days=offset)
        payload = provider.fetch_m1_day(symbol, candidate, timeout)
        if payload:
            candles = decode_m1_bi5(payload, symbol, candidate)
            quality = inspect_candles(
                candles,
                symbol,
                "M1",
                expected_start=datetime.combine(candidate, datetime.min.time(), UTC),
                expected_end=datetime.combine(
                    candidate + timedelta(days=1), datetime.min.time(), UTC
                ),
            )
            if quality.status == "PASS":
                return candidate
    raise RuntimeError("could not discover a recent Dukascopy M1 data day")


def run_pipeline(
    settings: HistoricalSettings,
    *,
    start=None,
    end=None,
    dry_run=False,
    force=False,
    provider=None,
    request_interval: float | None = None,
    progress=None,
):
    start = start or settings.start_date
    settings.validate()
    provider = provider or DukascopyProvider(settings.base_url)
    if dry_run and end is None and settings.end_date is None:
        raise ValueError("--end is required for an offline dry-run")
    end = (
        end or settings.end_date or discover_latest(provider, timeout=settings.timeout)
    )
    if end < start or end > datetime.now(UTC).date():
        raise ValueError("invalid historical date range")
    if dry_run:
        # Planning must not contact a provider, touch a manifest, or write reports.
        return PipelineSummary(
            "astra-historical-v1",
            "Dukascopy BID M1",
            start,
            end,
            tuple(
                SymbolPipelineResult(symbol, "PLANNED") for symbol in settings.symbols
            ),
            "",
            "",
            "",
        )
    manifest = ManifestStore(settings.root)
    run_id = manifest.begin_run(
        symbols=settings.symbols,
        start_date=start,
        end_date=end,
        source=getattr(provider, "source_name", type(provider).__name__),
    )
    downloader = HistoricalDownloader(
        provider,
        root=settings.root,
        workers=settings.workers,
        retries=settings.retries,
        timeout=settings.timeout,
        read_timeout=settings.read_timeout,
        max_elapsed_seconds=settings.max_elapsed_seconds,
        retry_jitter_seconds=settings.retry_jitter_seconds,
        max_consecutive_failures=settings.max_consecutive_failures,
        manifest=manifest,
        progress=progress,
        **(
            {"request_interval": request_interval}
            if request_interval is not None
            else {}
        ),
    )
    results = []
    for symbol in settings.symbols:
        item = SymbolPipelineResult(symbol, "FAIL")
        try:
            try:
                item.downloads = downloader.download(
                    (symbol,), start, end, dry_run=dry_run, force=force
                )
            except KeyboardInterrupt:
                unit_counts = _scope_unit_counts(
                    manifest.data.get("downloads", {}),
                    settings.symbols,
                    start,
                    end,
                )
                runs_dir = Path(settings.root) / "reports" / "runs"
                runs_dir.mkdir(parents=True, exist_ok=True)
                interrupted_path = runs_dir / f"{run_id}.json"
                _atomic_write(
                    interrupted_path,
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "INTERRUPTED",
                            "scope": {
                                "symbols": list(settings.symbols),
                                "start_date": start.isoformat(),
                                "end_date": end.isoformat(),
                                "timeframes": list(_OUTPUTS),
                            },
                            "unit_status_counts": unit_counts,
                        },
                        indent=2,
                    )
                    + "\n",
                )
                manifest.finish_run(
                    run_id,
                    status="INTERRUPTED",
                    summary={
                        "scope": {
                            "symbols": list(settings.symbols),
                            "start_date": start.isoformat(),
                            "end_date": end.isoformat(),
                        },
                        "unit_status_counts": unit_counts,
                        "report": str(interrupted_path),
                    },
                )
                raise
            if (
                item.downloads.failed
                or item.downloads.invalid
                or item.downloads.missing
                or item.downloads.deferred
            ):
                item.errors.append(
                    "one or more required daily source units are unavailable or invalid"
                )
                results.append(item)
                continue
            m1_path = normalize_symbol(settings.root, symbol, start, end)
            from ..history import load_csv

            market = load_csv(
                m1_path, symbol, "M1", allowed_pairs=(symbol,), min_candles=1
            )
            for timeframe in _OUTPUTS:
                if timeframe != "M1":
                    bars = resample(market.candles, timeframe)
                    target = (
                        Path(settings.root)
                        / "processed"
                        / timeframe.lower()
                        / f"{symbol}.csv"
                    )
                    _write_candles(target, bars, volume_unit="provider_raw")
                else:
                    target = m1_path
                if target.is_file():
                    item.timeframes[timeframe] = inspect_csv(
                        target,
                        symbol,
                        timeframe,
                        max_gap_minutes=settings.max_gap_minutes,
                        expected_start=datetime.combine(
                            start, datetime.min.time(), UTC
                        ),
                        expected_end=datetime.combine(
                            end + timedelta(days=1), datetime.min.time(), UTC
                        ),
                    )
            statuses = {report.status for report in item.timeframes.values()}
            if not statuses or "FAIL" in statuses:
                item.status = "FAIL"
            elif "WARN" in statuses or item.errors:
                item.status = "WARN"
                item.errors.append("DATA QUALITY INSUFFICIENT")
            else:
                item.status = "PASS"
        except Exception as error:  # noqa: BLE001 - isolate symbol failures
            item.errors.append(type(error).__name__)
        results.append(item)
    manifest.update_dataset(
        symbols=settings.symbols,
        start_date=start,
        end_date=end,
        results={
            r.symbol: {
                "status": r.status,
                "timeframes": {
                    k: {"status": v.status, "rows": v.rows, "checksum": v.checksum}
                    for k, v in r.timeframes.items()
                },
                "errors": r.errors,
            }
            for r in results
        },
        run_id=run_id,
    )
    coverage = reconcile_acquisition(settings.root, settings.symbols, start, end)
    all_timeframes_pass = all(
        set(item.timeframes) == set(_OUTPUTS)
        and all(report.status == "PASS" for report in item.timeframes.values())
        for item in results
    )
    requested_scope_complete = (
        all(item.status == "PASS" for item in results)
        and all_timeframes_pass
        and coverage["status"] == "PASS"
    )
    report_dir = Path(settings.root) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "run_id": run_id,
        "status": "COMPLETE" if requested_scope_complete else "INCOMPLETE",
        "requested_scope_complete": requested_scope_complete,
        "scope": {
            "symbols": list(settings.symbols),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timeframes": list(_OUTPUTS),
        },
        "coverage_reconciliation": coverage,
        "dataset_version": manifest.data["dataset_version"],
        "source": "Dukascopy BID M1",
        "request_base_url": settings.base_url,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "symbols": [
            {
                "symbol": r.symbol,
                "status": r.status,
                "errors": r.errors,
                "downloads": r.downloads,
                "timeframes": r.timeframes,
            }
            for r in results
        ],
    }
    json_path = report_dir / "latest.json"
    runs_dir = report_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = runs_dir / f"{run_id}.json"
    encoded_report = json.dumps(
        report_data,
        default=lambda value: (
            value.__dict__
            if hasattr(value, "__dict__")
            else value.isoformat()
            if hasattr(value, "isoformat")
            else str(value)
        ),
        indent=2,
    )
    _atomic_write(json_path, encoded_report + "\n")
    _atomic_write(run_json_path, encoded_report + "\n")
    text_path = report_dir / "latest.txt"
    text_path.write_text(
        "\n".join(
            f"{r.symbol}: {r.status} ({', '.join(r.errors) or 'no symbol errors'})"
            for r in results
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.finish_run(
        run_id,
        status=report_data["status"],
        summary={
            "scope": report_data["scope"],
            "report": str(run_json_path),
            "coverage_status": coverage["status"],
            "symbol_statuses": {item.symbol: item.status for item in results},
        },
    )
    return PipelineSummary(
        manifest.data["dataset_version"],
        "Dukascopy BID M1",
        start,
        end,
        tuple(results),
        str(manifest.path),
        str(json_path),
        str(text_path),
        run_id,
    )


def _scope_unit_counts(downloads, symbols, start, end):
    counts: dict[str, int] = {}
    for symbol in symbols:
        day = start
        while day <= end:
            entry = downloads.get(f"{symbol}:{day.isoformat()}", {})
            status = entry.get("status", "not_attempted")
            counts[status] = counts.get(status, 0) + 1
            day += timedelta(days=1)
    return counts


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)
