"""Build bounded, checksum-verified research windows without filling missing bars."""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from ..data import DataValidationError
from .dukascopy_node import _month_ranges, _sha256
from .json_source import process_dukascopy_json
from .manifest import ManifestStore


def prepare_window(archive_root, root, symbol: str, start: date, end: date):
    if end < start:
        raise ValueError("end date precedes start date")
    from ..history import PAIRS

    if symbol not in PAIRS:
        raise ValueError("unsupported currency pair")
    archive = Path(archive_root).resolve()
    manifest = ManifestStore(archive)
    paths = []
    for first, last in _month_ranges(start.replace(day=1), end):
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        # Prefer a complete month; acquisition may have ended partway through it.
        keys = (f"{first}..{next_month}", f"{first}..{last + timedelta(days=1)}")
        item = next(
            (manifest.unit(symbol, key) for key in keys if manifest.unit(symbol, key)),
            None,
        )
        if not item or item.get("status") != "downloaded":
            raise DataValidationError(
                f"DATA QUALITY INSUFFICIENT: unverified archive month {symbol} {first:%Y-%m}"
            )
        path = (archive / item["path"]).resolve()
        if (
            not path.is_relative_to(archive)
            or not path.is_file()
            or _sha256(path) != item["checksum"]
        ):
            raise DataValidationError(
                "DATA QUALITY INSUFFICIENT: archive checksum or path mismatch"
            )
        paths.append(path)
    return process_dukascopy_json(
        paths,
        symbol,
        root,
        expected_start=datetime.combine(start, time.min, UTC),
        expected_end=datetime.combine(end + timedelta(days=1), time.min, UTC),
        clip_to_range=True,
    )
