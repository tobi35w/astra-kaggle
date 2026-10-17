"""The boundary between acquired historical files and quantitative research."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..data import DataValidationError
from ..history import PAIRS, TIMEFRAMES, load_csv
from ..models import MarketData
from .validation import inspect_candles


def require_quality(data: MarketData) -> None:
    report = inspect_candles(data.candles, data.pair, data.timeframe)
    if report.status != "PASS":
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: "
            + "; ".join(report.warnings or ("invalid candles",))
        )


def load_dataset(root: str | Path, symbol: str, timeframe: str) -> MarketData:
    """Load only the exact bytes approved by a successful managed BI5 pipeline.

    This checks software provenance and integrity. It is not independent price
    authentication, and does not establish economic value or ML eligibility.
    """
    if symbol not in PAIRS or timeframe not in TIMEFRAMES:
        raise ValueError("unsupported historical dataset")
    root = Path(root)
    if not (root / "reports" / "latest.json").is_file():
        return _load_json_dataset(root, symbol, timeframe)
    report = json.loads((root / "reports" / "latest.json").read_text(encoding="utf-8"))
    item = next((s for s in report["symbols"] if s["symbol"] == symbol), None)
    if item is None or item["status"] != "PASS":
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: successful acquisition report required"
        )
    path = (
        root
        / ("normalized" if timeframe == "M1" else "processed")
        / timeframe.lower()
        / f"{symbol}.csv"
    )
    data = load_csv(path, symbol, timeframe, min_candles=1)
    approved = item["timeframes"][timeframe]
    if (
        approved["status"] != "PASS"
        or data.metadata["dataset_version"] != "astra-sha256:" + approved["checksum"]
    ):
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: dataset changed after validation"
        )
    start = datetime.fromisoformat(report["start_date"]).replace(tzinfo=UTC)
    end = datetime.fromisoformat(report["end_date"]).replace(tzinfo=UTC) + timedelta(
        days=1
    )
    quality = inspect_candles(
        data.candles, symbol, timeframe, expected_start=start, expected_end=end
    )
    if quality.status != "PASS":
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: requested range is incomplete"
        )
    return replace(
        data,
        metadata=data.metadata
        | {
            "source": report["source"],
            "price_basis": "bid",
            "quality_status": "PASS",
            "canonical_m1_sha256": item["timeframes"]["M1"]["checksum"],
            "acquisition_version": report["dataset_version"],
            "requested_start": start.isoformat(),
            "requested_end_exclusive": end.isoformat(),
            "provenance": "managed Dukascopy acquisition; not independently authenticated",
        },
    )


def _load_json_dataset(root: Path, symbol: str, timeframe: str) -> MarketData:
    report_path = root / "reports" / f"{symbol.lower()}-dukascopy-json.json"
    if not report_path.is_file():
        # Converted qleap parquet snapshots keep the same managed-report contract.
        report_path = root / "reports" / f"{symbol.lower()}-qleap-parquet.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("symbol") != symbol:
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: successful acquisition report required"
        )
    if not report.get("requested_start") or not report.get("requested_end_exclusive"):
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: regenerate JSON data with an explicit requested range"
        )
    path = (
        root
        / ("normalized" if timeframe == "M1" else "processed")
        / timeframe.lower()
        / f"{symbol}.csv"
    )
    data = load_csv(path, symbol, timeframe, min_candles=1)
    approved = report["outputs"][timeframe]
    if (
        approved["status"] != "PASS"
        or data.metadata["dataset_version"]
        != "astra-sha256:" + approved["checksum_sha256"]
        or data.metadata["price_basis"] != "bid"
    ):
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: dataset changed after validation"
        )
    start = datetime.fromisoformat(report["requested_start"])
    end = datetime.fromisoformat(report["requested_end_exclusive"])
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise DataValidationError("DATA QUALITY INSUFFICIENT: invalid requested range")
    quality = inspect_candles(
        data.candles, symbol, timeframe, expected_start=start, expected_end=end
    )
    if quality.status != "PASS":
        raise DataValidationError(
            "DATA QUALITY INSUFFICIENT: requested range is incomplete"
        )
    return replace(
        data,
        metadata=data.metadata
        | {
            "source": report["source"],
            "quality_status": "PASS",
            "canonical_m1_sha256": report["outputs"]["M1"]["checksum_sha256"],
            "requested_start": start.isoformat(),
            "requested_end_exclusive": end.isoformat(),
            "source_sha256": report["source_sha256"],
            "provenance": report.get(
                "provenance", "managed Dukascopy JSON; not independently authenticated"
            ),
        },
    )
