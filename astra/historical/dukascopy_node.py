"""Rate-limited, resumable acquisition through the public Dukascopy CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ..models import Candle
from .manifest import ManifestStore
from .validation import _bar_crosses_closed_market, _market_open, inspect_candles

_PACKAGE = "dukascopy-node@1.50.0"


class DukascopyNodeDownloader:
    """Download flat-inclusive BID M1 JSON files in resumable month chunks."""

    def __init__(
        self, root: str | Path, *, workers: int = 1, runner=None, timeout: float = 900
    ):
        if not 1 <= workers <= 2:
            raise ValueError("Dukascopy CLI workers must be 1 or 2")
        self.root = Path(root).resolve()
        self.workers = workers
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Dukascopy CLI timeout must be positive and finite")
        self.timeout = timeout
        self.manifest = ManifestStore(self.root)
        self.runner: Callable | None = runner

    def download(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        if end < start:
            raise ValueError("end date precedes start date")
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be nonempty and unique")
        if any(
            len(symbol) != 6 or not symbol.isalpha() or symbol != symbol.upper()
            for symbol in symbols
        ):
            raise ValueError("symbols must be six uppercase FX letters")

        results = {}
        if self.workers == 1:
            for symbol in symbols:
                results[symbol] = self._download_symbol(
                    symbol, start, end, force, dry_run
                )
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {
                    pool.submit(
                        self._download_symbol, symbol, start, end, force, dry_run
                    ): symbol
                    for symbol in symbols
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
        return {
            "source": "Dukascopy via dukascopy-node 1.50.0",
            "start_date": start.isoformat(),
            "end_date_inclusive": end.isoformat(),
            "include_flats": True,
            "dry_run": dry_run,
            "symbols": {symbol: results[symbol] for symbol in symbols},
        }

    def _download_symbol(self, symbol, start, end, force, dry_run):
        chunks = [
            self._download_one(symbol, chunk_start, chunk_end, force, dry_run)
            for chunk_start, chunk_end in _month_ranges(start, end)
        ]
        successful = [
            item for item in chunks if item["status"] in ("downloaded", "cached")
        ]
        if any(item["status"] == "failed" for item in chunks):
            status = "failed"
        elif dry_run:
            status = "planned"
        elif all(item["status"] == "cached" for item in chunks):
            status = "cached"
        else:
            status = "downloaded"
        return {
            "status": status,
            "files": [item["path"] for item in successful],
            "chunks": chunks,
            "rows": sum(item.get("rows", 0) for item in successful),
        }

    def _download_one(self, symbol, start, end, force, dry_run):
        exclusive_end = end + timedelta(days=1)
        key = f"{start.isoformat()}..{exclusive_end.isoformat()}"
        target = (
            self.root
            / "raw"
            / "dukascopy-node"
            / symbol
            / f"{symbol.lower()}-m1-bid-{start.isoformat()}-{exclusive_end.isoformat()}.json"
        )
        if dry_run:
            return {"status": "planned", "path": target.as_posix()}

        prior = self.manifest.unit(symbol, key)
        if target.is_file() and not force:
            digest = _sha256(target)
            if (
                prior
                and prior.get("status") == "downloaded"
                and prior.get("checksum") == digest
                and _valid_json_export(target, start, exclusive_end)
            ):
                row_count = len(json.loads(target.read_text(encoding="utf-8")))
                upgraded = {
                    **prior,
                    "rows": row_count,
                    "volume_type": "provider",
                    "volume_unit": "millions",
                    "include_flats": True,
                }
                self.manifest.update_unit(symbol, key, upgraded)
                return {
                    "status": "cached",
                    "path": target.as_posix(),
                    "checksum": digest,
                    "rows": row_count,
                }

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="dukascopy-", dir=target.parent
            ) as tmp:
                output_dir = Path(tmp)
                command = self._command(symbol, start, exclusive_end, output_dir, force)
                self._run(command, output_dir)
                staged = output_dir / target.name
                if not staged.is_file() or not _valid_json_export(
                    staged, start, exclusive_end
                ):
                    raise ValueError(
                        "Dukascopy CLI did not produce a valid nonempty JSON export"
                    )
                staged.replace(target)
            digest = _sha256(target)
            rows = len(json.loads(target.read_text(encoding="utf-8")))
            self.manifest.update_unit(
                symbol,
                key,
                {
                    "status": "downloaded",
                    "path": target.relative_to(self.root).as_posix(),
                    "checksum": digest,
                    "rows": rows,
                    "source": "dukascopy-node-1.50.0-bid-m1",
                    "include_flats": True,
                    "volume_type": "provider",
                    "volume_unit": "millions",
                    "start_date": start.isoformat(),
                    "end_date_exclusive": exclusive_end.isoformat(),
                },
            )
            return {
                "status": "downloaded",
                "path": target.as_posix(),
                "checksum": digest,
                "rows": rows,
            }
        except Exception as error:  # noqa: BLE001 - sanitize external CLI errors
            self.manifest.update_unit(
                symbol, key, {"status": "failed", "error": type(error).__name__}
            )
            return {"status": "failed", "error": type(error).__name__}

    def _command(self, symbol, start, end_exclusive, output_dir, force):
        base = _npm_exec_prefix()
        cache_path = self.root / "cache" / "dukascopy-node" / symbol
        command = base + [
            "-i",
            symbol.lower(),
            "-from",
            start.isoformat(),
            "-to",
            end_exclusive.isoformat(),
            "-t",
            "m1",
            "-p",
            "bid",
            "-v",
            "-vu",
            "millions",
            "-fl",
            "-f",
            "json",
            "-dir",
            str(output_dir),
            "-bs",
            "1",
            "-bp",
            "1000",
            "-r",
            "3",
            "-rp",
            "30000",
        ]
        if not force:
            command.extend(["-ch", "-chpath", str(cache_path)])
        return command

    def _run(self, command, output_dir):
        if self.runner is not None:
            self.runner(command, output_dir)
            return
        environment = os.environ.copy()
        environment["npm_config_cache"] = str(
            Path(tempfile.gettempdir()) / "astra-npm-cache"
        )
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=environment,
            check=False,
            timeout=self.timeout,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Dukascopy CLI exited with status {completed.returncode}"
            )


def _npm_exec_prefix() -> list[str]:
    node = shutil.which("node")
    if node:
        npm_cli = (
            Path(node).resolve().parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
        )
        if npm_cli.is_file():
            return [
                node,
                str(npm_cli),
                "exec",
                "--yes",
                f"--package={_PACKAGE}",
                "--",
                "dukascopy-node",
            ]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", _PACKAGE]
    raise RuntimeError("Node.js/npm is required to download Dukascopy JSON data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_json_export(path: Path, start: date, end_exclusive: date) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            return False
        start_ms = int(datetime.combine(start, time.min, UTC).timestamp() * 1000)
        end_ms = int(datetime.combine(end_exclusive, time.min, UTC).timestamp() * 1000)
        previous_timestamp = None
        for row in rows:
            if not isinstance(row, dict) or not {
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }.issubset(row):
                return False
            timestamp = row["timestamp"]
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                return False
            if timestamp % 60_000:
                return False
            if previous_timestamp is not None:
                if timestamp <= previous_timestamp:
                    return False
                for missing_ms in range(previous_timestamp + 60_000, timestamp, 60_000):
                    if _market_open(datetime.fromtimestamp(missing_ms / 1000, UTC)):
                        return False
            previous_timestamp = timestamp
            opening, high, low, close, volume = (
                float(row[key]) for key in ("open", "high", "low", "close", "volume")
            )
            if (
                timestamp < start_ms
                or timestamp >= end_ms
                or not all(map(math.isfinite, (opening, high, low, close, volume)))
                or min(opening, high, low, close) <= 0
                or volume < 0
                or low > min(opening, close)
                or high < max(opening, close)
            ):
                return False
        candles = tuple(
            Candle(
                datetime.fromtimestamp(row["timestamp"] / 1000, UTC)
                + timedelta(minutes=1),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                "provider",
            )
            for row in rows
            if not _bar_crosses_closed_market(
                datetime.fromtimestamp(row["timestamp"] / 1000, UTC),
                timedelta(minutes=1),
            )
        )
        report = inspect_candles(
            candles,
            "EURUSD",
            "M1",
            expected_start=datetime.combine(start, time.min, UTC),
            expected_end=datetime.combine(end_exclusive, time.min, UTC),
        )
        return not report.suspicious_gaps and not report.invalid_count
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _month_ranges(start: date, end: date):
    """Yield inclusive date chunks no larger than one calendar month."""
    current = start
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)
