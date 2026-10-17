"""Compare overlapping FX bars against Astra's previously curated archive."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astra.history import load_csv

ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = ROOT / "data" / "future-phase" / "Yllvar-fx-historical-data"
OLD_ROOT = ROOT / "data" / "astra-training-four-pairs"
CURATION = json.loads((OLD_ROOT / "curation.json").read_text(encoding="utf-8"))
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
TIMEFRAMES = tuple(TF_MINUTES)


def main() -> None:
    comparisons = []
    for pair in PAIRS:
        for timeframe in TIMEFRAMES:
            source_path = (
                NEW_ROOT
                / "four-pairs-source-parquet"
                / pair
                / f"{pair}_{timeframe}.parquet"
            )
            table = pq.read_table(source_path)
            opened_ms = (
                table["Time"]
                .cast(pa.timestamp("ms", tz="UTC"))
                .cast(pa.int64())
                .to_numpy()
            )
            closed_ms = opened_ms + TF_MINUTES[timeframe] * 60_000
            new_prices = np.column_stack(
                [table[name].to_numpy() for name in ("Open", "High", "Low", "Close")]
            )
            new_by_close = {
                int(timestamp): prices
                for timestamp, prices in zip(closed_ms, new_prices, strict=True)
            }
            new_start, new_end = int(closed_ms.min()), int(closed_ms.max())
            intersections = []

            for segment in CURATION["pairs"][pair]["segments"]:
                if segment["status"] != "PASS" or timeframe not in segment.get(
                    "loadable_timeframes", ()
                ):
                    continue
                segment_start = int(
                    datetime.fromisoformat(segment["start"])
                    .replace(tzinfo=UTC)
                    .timestamp()
                    * 1000
                )
                segment_end = int(
                    (
                        datetime.fromisoformat(segment["end_inclusive"])
                        + timedelta(days=1)
                    )
                    .replace(tzinfo=UTC)
                    .timestamp()
                    * 1000
                )
                if segment_end < new_start or segment_start > new_end:
                    continue

                report_path = ROOT / segment["result"]["report_json"]
                old_report = json.loads(report_path.read_text(encoding="utf-8"))
                frame_report = old_report["outputs"][timeframe]
                if frame_report["status"] != "PASS":
                    continue
                old_path = ROOT / frame_report["path"]
                digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
                if digest != frame_report["checksum_sha256"]:
                    raise ValueError(f"Curated source checksum changed: {old_path}")
                data = load_csv(old_path, pair, timeframe, min_candles=1)
                old = [
                    (int(candle.timestamp.timestamp() * 1000), candle)
                    for candle in data.candles
                    if new_start <= int(candle.timestamp.timestamp() * 1000) <= new_end
                    and int(candle.timestamp.timestamp() * 1000) in new_by_close
                ]
                if not old:
                    continue
                old_values = np.array(
                    [
                        [candle.open, candle.high, candle.low, candle.close]
                        for _, candle in old
                    ],
                    dtype=float,
                )
                new_values = np.array(
                    [new_by_close[timestamp] for timestamp, _ in old], dtype=float
                )
                diff = np.abs(old_values - new_values)
                pip = 0.01 if pair.endswith("JPY") else 0.0001
                close_diff_pips = diff[:, 3] / pip
                intersections.append(
                    {
                        "segment": segment["id"],
                        "matched_timestamps": len(old),
                        "exact_ohlc_rows": int(
                            np.count_nonzero(np.all(diff == 0, axis=1))
                        ),
                        "close_mae_pips": float(close_diff_pips.mean()),
                        "close_p95_abs_pips": float(np.quantile(close_diff_pips, 0.95)),
                        "close_max_abs_pips": float(close_diff_pips.max()),
                    }
                )

            comparisons.append(
                {
                    "symbol": pair,
                    "timeframe": timeframe,
                    "new_source_first_close_utc": str(
                        table["Time"][0].as_py()
                        + timedelta(minutes=TF_MINUTES[timeframe])
                    ),
                    "new_source_last_close_utc": str(
                        table["Time"][-1].as_py()
                        + timedelta(minutes=TF_MINUTES[timeframe])
                    ),
                    "overlapping_segments": intersections,
                    "matched_timestamps": sum(
                        item["matched_timestamps"] for item in intersections
                    ),
                }
            )

    report = {
        "source": "Yllvar/fx-historical-data versus Astra's managed Dukascopy archive",
        "interpretation": "The new dataset's Time field is a bar-open timestamp; comparisons add one timeframe duration to obtain close time. Different provider processing can produce legitimate price discrepancies.",
        "comparisons": comparisons,
        "caution": "This is a source-consistency comparison, not independent authentication or proof either feed is the executable broker price.",
    }
    output = NEW_ROOT / "overlap-comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "symbol": item["symbol"],
                    "timeframe": item["timeframe"],
                    "matched_timestamps": item["matched_timestamps"],
                    "overlapping_segments": len(item["overlapping_segments"]),
                }
                for item in comparisons
            ],
            indent=2,
        )
    )
    print(f"Detailed overlap comparison: {output}")


if __name__ == "__main__":
    main()
