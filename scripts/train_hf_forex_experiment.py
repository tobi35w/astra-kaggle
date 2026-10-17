"""Reproduce a diagnostic model experiment using only the saved HF Arrow files.

Source timestamps are assumed to be close/availability times, as in the original
EURUSD experiment. That convention and the price basis are unverified. The label
is the direction after three observed bars, which can span gaps. No model is
activated and no trading performance is inferred from classification scores.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.data import DataValidationError, validate
from astra.historical.validation import _market_open, inspect_candles
from astra.ml import chronological_split, dataset, train
from astra.models import Candle, MarketData
from astra.repositories import Entity, Journal, MemoryRepository, json_value

DATASET_ID = "keithtwesigye/OHLCV-1m-Forex"
DATA_ROOT = REPO / "data" / "future-phase" / "OHLCV-1m-Forex"
CACHE = DATA_ROOT / "hf-cache"
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
SUPPORTED_PAIRS = {*PAIRS, "USDCAD", "USDCHF", "NZDUSD"}
MINUTE = timedelta(minutes=1)
CAVEAT = (
    "Historical diagnostic; overlapping training labels are purged, but this "
    "is not a certified fresh holdout. EURUSD repeats an already inspected period. "
    "External source is not independently authenticated; timestamp semantics "
    "are unspecified and treated as candle close/availability times. "
    "Labels use three observed bars, not necessarily three elapsed minutes. "
    "OHLCV has no identified bid/ask basis; classification metrics do not measure "
    "net trading returns or qualify a model for execution."
)
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume", "ticker"}


def to_market_data(rows, pair: str, *, max_rows: int = 400_000) -> MarketData:
    """Build a validated M1 series from already ticker-filtered rows."""
    if max_rows < 15:
        raise ValueError("max_rows must be at least 15")
    if pair not in SUPPORTED_PAIRS:
        raise ValueError("ticker is not supported by Astra's ML pair interface")
    selected = [row for row in rows if row["ticker"] == pair]
    if max_rows and len(selected) > max_rows:
        selected = selected[-max_rows:]
    for row in selected:
        stamp = row["timestamp"]
        # astimezone() on a naive datetime silently assumes the host's timezone.
        if (
            not isinstance(stamp, datetime)
            or stamp.tzinfo is None
            or stamp.utcoffset() is None
        ):
            raise DataValidationError("source timestamps must be timezone-aware")
    candles = tuple(
        Candle(
            timestamp=row["timestamp"].astimezone(UTC),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            volume_type="provider",
        )
        for row in selected
    )
    metadata = {
        "source": DATASET_ID,
        "source_split": "train",
        "price_basis": "unknown",
        "provenance": (
            "external Hugging Face dataset; OHLCV, UTC-awareness, ordering, and "
            "duplicate checks passed; not independently authenticated"
        ),
        "timestamp_convention": "unspecified; treated as close/availability time",
        "real_data": True,
    }
    data = MarketData(pair, "M1", candles, metadata)
    validate(data, min_candles=15)
    if any(c.timestamp.second or c.timestamp.microsecond for c in candles):
        raise ValueError("source timestamps are not minute aligned")
    digest = sha256(f"{DATASET_ID}:{pair}:M1:assumed-close\n".encode())
    for candle in candles:
        fields = [candle.timestamp.isoformat()] + [
            value.hex()
            for value in (
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
        ]
        digest.update(("|".join(fields) + "\n").encode())
    metadata["dataset_version"] = f"sha256:{digest.hexdigest()}"
    return data


def _cached_source_files() -> tuple[tuple[Path, ...], dict]:
    """Select one complete local snapshot, never fetch a missing cache remotely."""
    infos = sorted(
        (CACHE / "keithtwesigye___ohlcv-1m-forex").glob(
            "default/0.0.0/*/dataset_info.json"
        )
    )
    if not infos:
        raise FileNotFoundError(
            "dataset cache is missing; run scripts/download_hf_forex_dataset.py first"
        )
    if len(infos) != 1:
        raise ValueError(
            "multiple cached snapshots; select one before comparing models"
        )
    info = json.loads(infos[0].read_text(encoding="utf-8"))
    split = info["splits"]["train"]
    count = len(split["shard_lengths"])
    paths = tuple(sorted(infos[0].parent.glob("ohlcv-1m-forex-train-*.arrow")))
    expected = tuple(
        f"ohlcv-1m-forex-train-{index:05d}-of-{count:05d}.arrow"
        for index in range(count)
    )
    if not count or tuple(path.name for path in paths) != expected:
        raise ValueError("cached Arrow shards are incomplete or ambiguous")
    return paths, info


def source_inventory() -> dict:
    """Fingerprint the local inputs; hashes establish identity, not authenticity."""
    paths, info = _cached_source_files()
    entries = []
    for path in (*paths, paths[0].parent / "dataset_info.json"):
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            {
                "path": path.relative_to(CACHE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "dataset_id": DATASET_ID,
        "cache_snapshot": paths[0].parent.name,
        "declared_rows": info["splits"]["train"]["num_examples"],
        "files": entries,
    }


def _load_ticker_rows(pair: str, max_rows: int):
    if pair not in SUPPORTED_PAIRS:
        raise ValueError("ticker is not supported by Astra's ML pair interface")
    if max_rows < 15:
        raise ValueError("max_rows must be at least 15")
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow import ipc

    paths, info = _cached_source_files()
    pieces, retained = deque(), 0
    previous = None
    total_rows = 0
    for path, expected_rows in zip(paths, info["splits"]["train"]["shard_lengths"]):
        shard_rows = 0
        with pa.memory_map(str(path), "r") as source:
            for batch in ipc.open_stream(source):
                if set(batch.schema.names) != REQUIRED_COLUMNS:
                    raise ValueError("unexpected Hugging Face schema")
                shard_rows += batch.num_rows
                filtered = batch.filter(pc.equal(batch.column("ticker"), pair))
                if not filtered.num_rows:
                    continue
                stamps = filtered.column("timestamp")
                if (
                    not pa.types.is_timestamp(stamps.type)
                    or not stamps.type.tz
                    or stamps.null_count
                ):
                    raise DataValidationError(
                        "source timestamps must be timezone-aware"
                    )
                first, last = stamps[0].as_py(), stamps[-1].as_py()
                if (previous is not None and first <= previous) or pc.any(
                    pc.less_equal(stamps.slice(1), stamps.slice(0, len(stamps) - 1))
                ).as_py():
                    raise DataValidationError(
                        "source timestamps must be strictly ordered"
                    )
                previous = last
                pieces.append(pa.Table.from_batches([filtered]))
                retained += filtered.num_rows
                # Keep only the requested tail while checking the entire ticker's order.
                while retained - pieces[0].num_rows >= max_rows:
                    retained -= pieces.popleft().num_rows
                if retained > max_rows:
                    pieces[0] = pieces[0].slice(retained - max_rows)
                    retained = max_rows
        if shard_rows != expected_rows:
            raise ValueError("cached Arrow shard row count differs from its manifest")
        total_rows += shard_rows
    if total_rows != info["splits"]["train"]["num_examples"]:
        raise ValueError("cached dataset row count differs from its manifest")
    if not pieces:
        raise ValueError(f"dataset contains no rows for {pair}")
    return pa.concat_tables(tuple(pieces)).to_pylist()


def score_predictions(labels, probabilities, *, training_up_fraction: float) -> dict:
    """Use only training labels to choose the constant and majority baselines."""
    if not labels:
        return {"sample_count": 0, "status": "INSUFFICIENT_DATA"}
    if len(labels) != len(probabilities):
        raise ValueError("prediction and label counts differ")
    count = len(labels)
    accuracy = sum((p >= 0.5) == y for p, y in zip(probabilities, labels)) / count
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / count
    majority = int(training_up_fraction >= 0.5)
    baseline_accuracy = sum(y == majority for y in labels) / count
    baseline_brier = sum((training_up_fraction - y) ** 2 for y in labels) / count
    return {
        "sample_count": count,
        "accuracy": accuracy,
        "brier": brier,
        "observed_up_fraction": sum(labels) / count,
        "predicted_up_fraction": sum(p >= 0.5 for p in probabilities) / count,
        "majority_baseline_class": majority,
        "majority_baseline_accuracy": baseline_accuracy,
        "constant_probability_baseline": training_up_fraction,
        "constant_probability_brier": baseline_brier,
        "accuracy_improvement_percentage_points": (accuracy - baseline_accuracy) * 100,
        "brier_improvement": baseline_brier - brier,
    }


def validation_diagnostics(data: MarketData, model) -> dict:
    examples = dataset(data)
    training, validation = chronological_split(examples)
    probability = sum(example.label for example in training) / len(training)
    # These are retrospective scores computed after fitting, not historical orders
    # or predictions from a model falsely claimed to exist before trained_at.
    predictions = [model.predict(e.values, model.trained_at) for e in validation]
    labels = [example.label for example in validation]
    buckets = defaultdict(list)
    for index, example in enumerate(validation):
        buckets[example.timestamp.strftime("%Y-%m")].append(index)

    breaks, closed = [0], [0]
    previous = None
    for candle in data.candles:
        breaks.append(
            breaks[-1]
            + int(previous is not None and candle.timestamp - previous != MINUTE)
        )
        closed.append(closed[-1] + int(not _market_open(candle.timestamp - MINUTE)))
        previous = candle.timestamp
    start = len(examples) - len(validation) + 14
    clean = []
    for position, index in enumerate(range(start, start + len(validation))):
        # ATR uses the preceding close, so all 15 feature bars and 3 label bars
        # must be consecutive and inside the assumed FX session for this subset.
        first, last = index - 14, index + 3
        if breaks[last + 1] == breaks[first + 1] and closed[last + 1] == closed[first]:
            clean.append(position)

    def subset(indices):
        return score_predictions(
            [labels[i] for i in indices],
            [predictions[i] for i in indices],
            training_up_fraction=probability,
        )

    calibration = []
    for lower in range(10):
        indices = [i for i, p in enumerate(predictions) if min(int(p * 10), 9) == lower]
        if indices:
            calibration.append(
                {
                    "probability_from": lower / 10,
                    "probability_to": (lower + 1) / 10,
                    "sample_count": len(indices),
                    "mean_probability": sum(predictions[i] for i in indices)
                    / len(indices),
                    "observed_up_fraction": sum(labels[i] for i in indices)
                    / len(indices),
                }
            )
    return {
        "training_samples": len(training),
        "validation_samples": len(validation),
        "purged_training_samples": len(examples) - len(training) - len(validation),
        "training_label_end": training[-1].label_available_at.isoformat(),
        "validation_start": validation[0].timestamp.isoformat(),
        "validation_label_end": validation[-1].label_available_at.isoformat(),
        "overall": subset(range(len(validation))),
        "continuous_open_session_windows": subset(clean),
        "excluded_from_continuous_subset": len(validation) - len(clean),
        "labels_spanning_more_than_three_minutes": sum(
            e.label_available_at - e.timestamp > 3 * MINUTE for e in validation
        ),
        "by_decision_month": {
            month: subset(indices) for month, indices in sorted(buckets.items())
        },
        "calibration": calibration,
        "interpretation": (
            "Same frozen model in every diagnostic subset; no retuning. "
            "Overlapping labels are dependent observations, not independent trials."
        ),
    }


def run(pair: str = "EURUSD", *, max_rows: int = 400_000, journal=None) -> dict:
    rows = _load_ticker_rows(pair, max_rows)
    data = to_market_data(rows, pair, max_rows=max_rows)
    del rows
    quality = inspect_candles(data.candles, pair, "M1")
    journal = journal if journal is not None else Journal(MemoryRepository())
    result = train(data, enabled=True, journal=journal)
    report = {
        "dataset_id": DATASET_ID,
        "pair": pair,
        "timeframe": "M1",
        "status": result.status,
        "reason": result.reason,
        "source_rows": len(data.candles),
        "first_available_at": data.candles[0].timestamp.isoformat(),
        "last_available_at": data.candles[-1].timestamp.isoformat(),
        "dataset_version": data.metadata["dataset_version"],
        "timestamp_convention": data.metadata["timestamp_convention"],
        "data_quality": {
            "scope": "selected ticker tail; no rows filled, sorted, or deduplicated",
            "status": quality.status,
            "open_session_gap_count": len(quality.suspicious_gaps),
            "missing_open_session_minutes": sum(
                g.missing_minutes for g in quality.suspicious_gaps
            ),
            "expected_closure_gap_count": quality.expected_gap_count,
            "largest_timestamp_gap_minutes": max(
                (
                    int((b.timestamp - a.timestamp).total_seconds() // 60)
                    for a, b in pairwise(data.candles)
                ),
                default=0,
            ),
            "warnings": list(quality.warnings[:12]),
            "source_authentication": "UNVERIFIED",
        },
        "promotion_eligible": False,
        "profitability_evaluated": False,
        "caveat": CAVEAT,
    }
    if result.model is not None:
        report["model"] = json_value(result.model)
        report["validation"] = validation_diagnostics(data, result.model)
    report["completed_at"] = max(
        datetime.now(UTC), data.candles[-1].timestamp
    ).isoformat()
    # train() saves the standard Model schema. Reports have their own entity so
    # they cannot become malformed "latest model" payloads in load_model().
    journal.record(
        Entity.EVALUATION,
        datetime.fromisoformat(report["completed_at"]),
        {"kind": "hf_forex_ml_training_experiment", **report},
        pair=pair,
        timeframe="M1",
    )
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--max-rows", type=int, default=400_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.output:
        from astra.adapters import SQLiteRepository

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        repository = SQLiteRepository(str(output.with_suffix(".sqlite3")))
        try:
            report = run(args.pair, max_rows=args.max_rows, journal=Journal(repository))
            output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        finally:
            repository.close()
    else:
        report = run(args.pair, max_rows=args.max_rows)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "TRAINED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
