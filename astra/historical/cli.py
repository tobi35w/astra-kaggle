"""Command-line interface for staged historical-data work."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from ..history import load_csv
from .config import HistoricalSettings
from .downloader import HistoricalDownloader
from .dukascopy import DukascopyProvider
from .normalizer import normalize_symbol
from .pipeline import run_pipeline
from .resampler import resample
from .validation import inspect_csv


def historical_main(argv=None):
    try:
        return _historical_main(argv)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "message": "Historical command failed; data is not verified. Check range, source, reports and configuration.",
                }
            )
        )
        return 2


def _historical_main(argv=None):
    parser = argparse.ArgumentParser(prog="astra historical")
    commands = parser.add_subparsers(dest="action", required=True)
    latest = commands.add_parser("latest")
    latest.add_argument("--symbol", default="EURUSD")
    latest.add_argument("--base-url")
    pipeline = commands.add_parser("pipeline")
    download_command = commands.add_parser("download")
    for command in (pipeline, download_command):
        command.add_argument("--start", type=date.fromisoformat)
        command.add_argument("--end", type=date.fromisoformat)
        command.add_argument("--symbols", nargs="+")
        command.add_argument("--root")
        command.add_argument("--base-url")
        command.add_argument("--workers", type=int)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--force", action="store_true")
    normalize = commands.add_parser("normalize")
    normalize.add_argument("symbol")
    normalize.add_argument("--start", type=date.fromisoformat, required=True)
    normalize.add_argument("--end", type=date.fromisoformat, required=True)
    normalize.add_argument("--root")
    validate = commands.add_parser("validate")
    validate.add_argument("symbol")
    validate.add_argument("timeframe")
    validate.add_argument("path")
    resampling = commands.add_parser("resample")
    resampling.add_argument("symbol")
    resampling.add_argument("timeframe", choices=("M5", "M15", "H1", "H4"))
    resampling.add_argument("path")
    importing = commands.add_parser("import")
    importing.add_argument("symbol")
    importing.add_argument("timeframe")
    importing.add_argument("path")
    json_source = commands.add_parser("process-json")
    json_source.add_argument("symbol")
    json_source.add_argument("path", nargs="+")
    json_source.add_argument("--root")
    json_source.add_argument("--max-gap-minutes", type=int, default=1)
    parquet_source = commands.add_parser("process-parquet")
    parquet_source.add_argument("symbol")
    parquet_source.add_argument("source_dir")
    parquet_source.add_argument("--root")
    parquet_source.add_argument("--start", type=date.fromisoformat, required=True)
    parquet_source.add_argument("--end", type=date.fromisoformat, required=True)
    parquet_source.add_argument("--max-gap-minutes", type=int, default=1)
    json_pipeline = commands.add_parser("pipeline-json")
    json_pipeline.add_argument("--start", type=date.fromisoformat, required=True)
    json_pipeline.add_argument("--end", type=date.fromisoformat, required=True)
    json_pipeline.add_argument("--symbols", nargs="+", required=True)
    json_pipeline.add_argument("--root")
    json_pipeline.add_argument("--workers", type=int, default=1, choices=(1, 2))
    json_pipeline.add_argument("--max-gap-minutes", type=int, default=1)
    json_pipeline.add_argument("--dry-run", action="store_true")
    json_pipeline.add_argument("--force", action="store_true")
    window = commands.add_parser(
        "prepare-window", help="validate an existing archive slice without downloading"
    )
    window.add_argument("--archive-root", required=True)
    window.add_argument("--root", required=True)
    window.add_argument("--symbols", nargs="+", required=True)
    window.add_argument("--start", type=date.fromisoformat, required=True)
    window.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    settings = HistoricalSettings.from_env()
    result: Any
    if args.action == "prepare-window":
        from .window import prepare_window

        result = {
            symbol: prepare_window(
                args.archive_root, args.root, symbol, args.start, args.end
            )
            for symbol in args.symbols
        }
    elif args.action == "latest":
        from dataclasses import replace

        from .pipeline import discover_latest

        settings = replace(settings, base_url=args.base_url or settings.base_url)
        settings.validate()
        day = discover_latest(
            DukascopyProvider(settings.base_url),
            timeout=settings.timeout,
            symbol=args.symbol.upper(),
        )
        result = {
            "status": "VERIFIED",
            "symbol": args.symbol.upper(),
            "latest_complete_source_day": day,
            "base_url": settings.base_url,
            "checked_at": datetime.now(UTC),
        }
    elif args.action == "validate":
        result = inspect_csv(
            args.path,
            args.symbol.upper(),
            args.timeframe.upper(),
            max_gap_minutes=settings.max_gap_minutes,
        )
    elif args.action == "normalize":
        result = normalize_symbol(
            args.root or settings.root, args.symbol.upper(), args.start, args.end
        )
    elif args.action == "resample":
        data = load_csv(
            args.path, args.symbol.upper(), "M1", allowed_pairs=(args.symbol.upper(),)
        )
        result = resample(data.candles, args.timeframe)
    elif args.action == "import":
        from ..adapters import repository_from_settings
        from ..config import Settings
        from ..repositories import Journal

        repo = repository_from_settings(Settings.from_env())
        try:
            from .importer import import_csv

            result = {
                "imported_records": import_csv(
                    args.path,
                    args.symbol.upper(),
                    args.timeframe.upper(),
                    Journal(repo),
                    batch_size=settings.batch_size,
                )
            }
        finally:
            if hasattr(repo, "close"):
                repo.close()
    elif args.action == "process-json":
        from .json_source import process_dukascopy_json

        result = process_dukascopy_json(
            args.path,
            args.symbol,
            args.root or settings.root,
            max_gap_minutes=args.max_gap_minutes,
        )
    elif args.action == "process-parquet":
        from .qleap_source import process_qleap_parquet

        result = process_qleap_parquet(
            args.source_dir,
            args.symbol,
            args.root or settings.root,
            max_gap_minutes=args.max_gap_minutes,
            expected_start=datetime.combine(args.start, time.min, UTC),
            expected_end=datetime.combine(args.end + timedelta(days=1), time.min, UTC),
        )
    elif args.action == "pipeline-json":
        from .dukascopy_node import DukascopyNodeDownloader
        from .json_source import process_dukascopy_json

        symbols = tuple(symbol.upper() for symbol in args.symbols)
        settings = HistoricalSettings(
            root=args.root or settings.root,
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
            workers=args.workers,
            retries=settings.retries,
            timeout=settings.timeout,
            max_gap_minutes=args.max_gap_minutes,
            batch_size=settings.batch_size,
        )
        settings.validate()
        source = DukascopyNodeDownloader(settings.root, workers=settings.workers)
        download_result = source.download(
            symbols,
            args.start,
            args.end,
            dry_run=args.dry_run,
            force=args.force,
        )
        processed = {}
        for symbol in symbols:
            item = download_result["symbols"][symbol]
            if item["status"] in ("downloaded", "cached") and not args.dry_run:
                try:
                    processed[symbol] = process_dukascopy_json(
                        item["files"],
                        symbol,
                        settings.root,
                        max_gap_minutes=settings.max_gap_minutes,
                        expected_start=datetime.combine(args.start, time.min, UTC),
                        expected_end=datetime.combine(
                            args.end + timedelta(days=1), time.min, UTC
                        ),
                    )
                except Exception as error:  # noqa: BLE001 - isolate symbol-level failures
                    processed[symbol] = {
                        "status": "failed",
                        "error": type(error).__name__,
                    }
        from .manifest import ManifestStore

        if not args.dry_run:
            ManifestStore(settings.root).update_dataset(
                symbols=symbols,
                start_date=args.start,
                end_date=args.end,
                results={
                    symbol: processed.get(symbol, download_result["symbols"][symbol])
                    for symbol in symbols
                },
            )
        result = {"download": download_result, "processed": processed}
    else:
        symbols = (
            tuple(value.upper() for value in args.symbols)
            if args.symbols
            else settings.symbols
        )
        settings = HistoricalSettings(
            root=args.root or settings.root,
            symbols=symbols,
            start_date=args.start or settings.start_date,
            end_date=args.end or settings.end_date,
            workers=args.workers if args.workers is not None else settings.workers,
            retries=settings.retries,
            timeout=settings.timeout,
            max_gap_minutes=settings.max_gap_minutes,
            batch_size=settings.batch_size,
            base_url=args.base_url or settings.base_url,
        )
        settings.validate()
        if args.dry_run and settings.end_date is None:
            raise ValueError("--end is required for an offline dry-run")
        if args.action == "pipeline":
            result = run_pipeline(settings, dry_run=args.dry_run, force=args.force)
        else:
            end = settings.end_date
            if end is None:
                from .pipeline import discover_latest

                end = discover_latest(
                    DukascopyProvider(settings.base_url), timeout=settings.timeout
                )
            start = args.start or settings.start_date
            result = HistoricalDownloader(
                DukascopyProvider(settings.base_url),
                root=settings.root,
                workers=settings.workers,
                retries=settings.retries,
                timeout=settings.timeout,
            ).download(
                settings.symbols, start, end, dry_run=args.dry_run, force=args.force
            )
    print(
        json.dumps(
            result,
            default=lambda obj: (
                obj.isoformat()
                if hasattr(obj, "isoformat")
                else obj.__dict__
                if hasattr(obj, "__dict__")
                else str(obj)
            ),
            indent=2,
        )
    )
    if args.action == "prepare-window":
        return int(
            any(
                status != "PASS"
                for item in result.values()
                for status in item["timeframes"].values()
            )
        )
    if args.action == "pipeline-json" and any(
        item["status"] == "failed" for item in result["download"]["symbols"].values()
    ):
        return 1
    if args.action == "pipeline-json" and any(
        item.get("status") == "failed"
        or any(value != "PASS" for value in item.get("timeframes", {}).values())
        for item in result["processed"].values()
    ):
        return 1
    if args.action == "process-json":
        return int(any(value != "PASS" for value in result["timeframes"].values()))
    if args.action == "process-parquet":
        return int(any(value != "PASS" for value in result["timeframes"].values()))
    if args.action == "validate":
        return int(result.status != "PASS")
    if args.action in ("pipeline", "download"):
        return 0 if args.dry_run else int(not result.successful)
    return 0
