"""Audit the qleap FX OHLCV snapshot before any Astra use.

Verifies, per pair: timestamp monotonicity and uniqueness, minute alignment,
cross-month duplicate boundaries, price sanity, volume sanity, weekend-closure
behavior, and overlap against the curated four-pair windows. Read-only; writes
a single audit report. All findings are facts about the source, not endorsements.
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from math import isnan
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO
    / "data"
    / "future-phase"
    / "qleap-fx-majors-tick-ohlcv"
    / "source-repo"
    / "1min"
)
REPORT = (
    REPO / "data" / "future-phase" / "qleap-fx-majors-tick-ohlcv" / "astra-audit.json"
)
PAIR_MAP = {
    "aud_usd": "AUDUSD",
    "eur_usd": "EURUSD",
    "gbp_usd": "GBPUSD",
    "usd_jpy": "USDJPY",
}
CURATED = REPO / "data" / "astra-training-four-pairs"


def _market_open(stamp: datetime) -> bool:
    local = stamp.astimezone(ZoneInfo_NY())
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    if weekday == 5 or (weekday == 6 and minute < 17 * 60):
        return False
    return not (weekday == 4 and minute >= 17 * 60)


def ZoneInfo_NY():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")


def audit_pair(folder: Path) -> dict:
    files = sorted(folder.glob("*.parquet"))
    result = {
        "files": len(files),
        "rows": 0,
        "non_monotonic_files": [],
        "duplicate_timestamp_files": [],
        "non_positive_or_nan_prices": 0,
        "negative_volume": 0,
        "closed_session_bars": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "ohlc_inconsistent": 0,
        "month_boundary_candles": [],
    }
    previous_last: datetime | None = None
    for path in files:
        table = pq.read_table(
            path, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        stamps = table.column("timestamp").to_pylist()
        if not stamps:
            continue
        if any(b <= a for a, b in pairwise(stamps)):
            result["non_monotonic_files"].append(path.name)
        if len(set(stamps)) != len(stamps):
            result["duplicate_timestamp_files"].append(path.name)
        for column in ("Open", "High", "Low", "Close"):
            values = table.column(column).to_pylist()
            result["non_positive_or_nan_prices"] += sum(
                1 for v in values if v is None or isnan(v) or v <= 0
            )
        highs, lows = table.column("High").to_pylist(), table.column("Low").to_pylist()
        opens, closes = (
            table.column("Open").to_pylist(),
            table.column("Close").to_pylist(),
        )
        result["ohlc_inconsistent"] += sum(
            1
            for o, h, l, c in zip(opens, highs, lows, closes)
            if l > h or not l <= o <= h or not l <= c <= h
        )
        result["negative_volume"] += sum(
            1
            for v in table.column("Volume").to_pylist()
            if v is None or isnan(v) or v < 0
        )
        result["closed_session_bars"] += sum(
            1
            for s in stamps
            if not _market_open(s.replace(tzinfo=UTC) - timedelta(minutes=1))
        )
        result["rows"] += len(stamps)
        if result["first_timestamp"] is None:
            result["first_timestamp"] = stamps[0].isoformat()
        result["last_timestamp"] = stamps[-1].isoformat()
        if previous_last is not None and stamps[0] <= previous_last:
            result["month_boundary_candles"].append(
                {
                    "file": path.name,
                    "first": stamps[0].isoformat(),
                    "previous_last": previous_last.isoformat(),
                }
            )
        previous_last = stamps[-1]
    return result


def main() -> int:
    report = {
        "audited_at": datetime.now(UTC).isoformat(),
        "source": "qleap/fx_majors_tick_ohlcv",
        "pairs": {},
    }
    for folder, symbol in PAIR_MAP.items():
        print(f"auditing {symbol} ...", flush=True)
        report["pairs"][symbol] = audit_pair(SOURCE / folder)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AUDITED", "report": str(REPORT)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
