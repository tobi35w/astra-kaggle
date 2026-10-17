"""Retrospective HF classification with continuous feature and label windows.

Source candles keep their original positions. The 70/30 chronological boundary
is chosen before purging or continuity filtering; no removed bars are joined.
Every retained example uses 15 consecutive open-session M1 feature bars and
the next three M1 bars. The legacy default retains assumed availability times;
cloud runs explicitly convert source opening timestamps using sampled provenance.
These previously consumed data are not a fresh holdout or evidence of profitability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

from astra.historical.validation import _market_open
from astra.ml import FEATURE_NAMES, Model, chronological_split, dataset
from astra.repositories import Entity, Journal, MemoryRepository, json_value
from scripts.hf_source_clock import availability_view
from scripts.train_hf_forex_experiment import (
    DATASET_ID,
    MINUTE,
    _load_ticker_rows,
    score_predictions,
    to_market_data,
)

FEATURE_BARS = 15
HORIZON = 3
TRAINING_FRACTION = 0.7
MINIMUM_SAMPLES = 300
FITTING_SETTINGS = {
    "training_fraction": TRAINING_FRACTION,
    "split_basis": "original examples before purging or continuity filtering",
    "purge_rule": "training label availability strictly before original validation start",
    "feature_schema": list(FEATURE_NAMES),
    "feature_bars": FEATURE_BARS,
    "label_horizon_bars": HORIZON,
    "label_horizon_minutes": HORIZON,
    "continuity_rule": "all 18 bars consecutive M1 and open at close minus one minute",
    "session_calendar": "America/New_York Sunday 17:00 through Friday 17:00",
    "minimum_retained_samples": MINIMUM_SAMPLES,
    "scaler": "StandardScaler",
    "estimator": "LogisticRegression",
    "random_state": 0,
    "max_iter": 1000,
}
CAVEAT = (
    "Retrospective diagnostic on previously consumed historical data; this is "
    "not a certified fresh holdout. The continuity rule was adopted after "
    "inspection of the original experiment. External source authenticity is "
    "unverified; unspecified timestamps are assumed to be candle close/availability "
    "times and OHLCV price basis is unknown. The New York FX session calendar "
    "is an assumption, not provider-specific holiday verification. Every retained "
    "label covers exactly three elapsed minutes. Overlapping labels are dependent "
    "observations. Model availability is recorded no earlier than this run and "
    "the latest source close; scores are retrospective, not historical forecasts. "
    "Classification metrics do not measure net trading returns or "
    "qualify a model for execution."
)


def _stamp(example, field="timestamp"):
    return getattr(example, field).isoformat() if example is not None else None


def continuous_split(data):
    """Return clean partitions, counts, and a linear-time source continuity profile.

    ATR's first true range needs the preceding close, so the complete window
    starts fourteen bars before the decision and ends three bars after it.
    Prefix sums inspect only the original source sequence, without filling gaps
    or scanning every missing minute in a long closure.
    """
    examples = dataset(data, horizon=HORIZON)
    split_index = int(len(examples) * TRAINING_FRACTION)
    original_training = examples[:split_index]
    original_validation = examples[split_index:]
    boundary = original_validation[0].timestamp if original_validation else None
    # Tiny inputs can have no eligible training labels. Preserve their counts
    # instead of letting chronological_split's insufficient-data error escape.
    eligible = tuple(
        example
        for example in original_training
        if boundary is not None and example.label_available_at < boundary
    )
    if len(examples) >= 4 and eligible:
        eligible, original_validation = chronological_split(
            examples, fraction=TRAINING_FRACTION
        )

    breaks, closed = [0], [0]
    largest_gap = 0
    previous = None
    for candle in data.candles:
        elapsed = candle.timestamp - previous if previous is not None else MINUTE
        breaks.append(breaks[-1] + int(elapsed != MINUTE))
        closed.append(closed[-1] + int(not _market_open(candle.timestamp - MINUTE)))
        if previous is not None:
            largest_gap = max(largest_gap, int(elapsed.total_seconds() // 60))
        previous = candle.timestamp

    clean = []
    for offset in range(len(examples)):
        first = offset
        last = offset + FEATURE_BARS - 1 + HORIZON
        # The break immediately before the first feature bar is irrelevant;
        # every bar itself, including that first bar, must be in the FX session.
        clean.append(
            breaks[last + 1] == breaks[first + 1] and closed[last + 1] == closed[first]
        )
    # Label availability increases with source position, so purging removes
    # only the original training suffix. The validation start never moves.
    training = tuple(e for e, keep in zip(eligible, clean) if keep)
    validation = tuple(
        e for e, keep in zip(original_validation, clean[split_index:]) if keep
    )
    diagnostics = {
        "original_examples": len(examples),
        "original_training_samples": len(original_training),
        "original_validation_samples": len(original_validation),
        "purged_training_samples": len(original_training) - len(eligible),
        "excluded_training_samples": len(eligible) - len(training),
        "excluded_validation_samples": len(original_validation) - len(validation),
        "training_samples": len(training),
        "validation_samples": len(validation),
        "fixed_validation_start": boundary.isoformat() if boundary else None,
        "validation_start": boundary.isoformat() if boundary else None,
        "original_validation_label_end": _stamp(
            original_validation[-1] if original_validation else None,
            "label_available_at",
        ),
        "retained_training_start": _stamp(training[0] if training else None),
        "retained_training_end": _stamp(training[-1] if training else None),
        "training_label_end": _stamp(
            training[-1] if training else None, "label_available_at"
        ),
        "retained_validation_start": _stamp(validation[0] if validation else None),
        "retained_validation_end": _stamp(validation[-1] if validation else None),
        "validation_label_end": _stamp(
            validation[-1] if validation else None, "label_available_at"
        ),
        "retained_labels_spanning_more_than_three_minutes": sum(
            example.label_available_at - example.timestamp != HORIZON * MINUTE
            for partition in (training, validation)
            for example in partition
        ),
    }
    quality = {
        "scope": "selected ticker tail; source rows unchanged; continuity profile only",
        "status": "WARN" if breaks[-1] or closed[-1] else "PASS",
        "timestamp_gap_count": breaks[-1],
        "closed_session_bar_count": closed[-1],
        "largest_timestamp_gap_minutes": largest_gap,
        "source_authentication": "UNVERIFIED",
    }
    return training, validation, diagnostics, quality


def _training_version(pair, training, settings):
    """Identify the actual fit inputs, independent of future validation prices."""
    digest = sha256()
    digest.update(
        json.dumps(
            {"dataset_id": DATASET_ID, "pair": pair, "timeframe": "M1", **settings},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    for example in training:
        digest.update(
            json.dumps(
                json_value(example),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _diagnostics(training, validation, predictions):
    labels = [example.label for example in validation]
    training_up = sum(example.label for example in training) / len(training)

    def score(indices):
        return score_predictions(
            [labels[i] for i in indices],
            [predictions[i] for i in indices],
            training_up_fraction=training_up,
        )

    months, bins = defaultdict(list), defaultdict(list)
    for index, (example, probability) in enumerate(zip(validation, predictions)):
        months[example.timestamp.strftime("%Y-%m")].append(index)
        bins[min(int(probability * 10), 9)].append(index)
    calibration = [
        {
            "probability_from": bucket / 10,
            "probability_to": (bucket + 1) / 10,
            "sample_count": len(indices),
            "mean_probability": sum(predictions[i] for i in indices) / len(indices),
            "observed_up_fraction": sum(labels[i] for i in indices) / len(indices),
        }
        for bucket, indices in sorted(bins.items())
    ]
    return {
        "overall": score(range(len(validation))),
        "by_decision_month": {
            month: score(indices) for month, indices in sorted(months.items())
        },
        "calibration": calibration,
        "interpretation": (
            "Retrospective scores from one frozen model at its creation availability "
            "time; no historical inference or orders. Training-only baselines and "
            "scaling; no retuning by month or probability bin."
        ),
    }


def run(
    pair: str = "EURUSD",
    *,
    max_rows: int = 400_000,
    journal=None,
    timestamp_shift_minutes=0,
) -> dict:
    if timestamp_shift_minutes not in (0, 1):
        raise ValueError("supported timestamp shifts are zero or one minute")
    rows = _load_ticker_rows(pair, max_rows)
    data = to_market_data(rows, pair, max_rows=max_rows)
    data = availability_view(data, timestamp_shift_minutes)
    del rows
    training, validation, split, quality = continuous_split(data)
    journal = journal if journal is not None else Journal(MemoryRepository())
    report = {
        "dataset_id": DATASET_ID,
        "pair": pair,
        "timeframe": "M1",
        "status": "INSUFFICIENT_DATA",
        "reason": "insufficient retained labeled observations after continuity filtering",
        "source_rows": len(data.candles),
        "first_available_at": data.candles[0].timestamp.isoformat(),
        "last_available_at": data.candles[-1].timestamp.isoformat(),
        "dataset_version": data.metadata["dataset_version"],
        "raw_dataset_version": data.metadata.get(
            "raw_dataset_version", data.metadata["dataset_version"]
        ),
        "source_timestamp_shift_minutes": timestamp_shift_minutes,
        "timestamp_convention": data.metadata["timestamp_convention"],
        "price_basis": data.metadata["price_basis"],
        "fitting_settings": dict(FITTING_SETTINGS),
        "data_quality": quality,
        "validation": split,
        "promotion_eligible": False,
        "profitability_evaluated": False,
        "caveat": CAVEAT,
    }
    if timestamp_shift_minutes:
        report["fitting_settings"]["source_timestamp_shift_minutes"] = 1
        report["sampled_price_basis"] = data.metadata["sampled_price_basis"]
        report["caveat"] = CAVEAT.replace(
            "External source authenticity is unverified; unspecified timestamps are assumed to be candle close/availability times and OHLCV price basis is unknown.",
            "Sampled EURUSD/USDJPY OHLC matches support BID prices and source opening timestamps. This run shifts source times +1 minute to availability. The whole snapshot and volume semantics remain unauthenticated; the same convention for other pairs is an assumption.",
        )
    if training and validation and len(training) + len(validation) >= MINIMUM_SAMPLES:
        if len({example.label for example in training}) < 2:
            report["reason"] = "retained training requires both outcome classes"
        else:
            from sklearn import __version__ as sklearn_version
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit([example.values for example in training])
            estimator = LogisticRegression(random_state=0, max_iter=1000).fit(
                scaler.transform([example.values for example in training]),
                [example.label for example in training],
            )
            report["fitting_settings"].update(
                sklearn_version=sklearn_version,
                scaler_parameters=scaler.get_params(),
                estimator_parameters=estimator.get_params(),
            )
            version = _training_version(pair, training, report["fitting_settings"])
            report["training_content_version"] = f"sha256:{version}"
            # This report inspected the entire selected tail. Neither the model
            # nor its retrospective scores may claim to exist before that tail
            # was available or before this actual fit was performed.
            available_at = max(datetime.now(UTC), data.candles[-1].timestamp)
            model = Model(
                version[:16],
                available_at,
                training[-1].label_available_at,
                datetime.fromisoformat(split["fixed_validation_start"]),
                tuple(float(value) for value in scaler.mean_),
                tuple(float(value) for value in scaler.scale_),
                tuple(float(value) for value in estimator.coef_[0]),
                float(estimator.intercept_[0]),
                0.0,
                0.0,
                len(training) + len(validation),
            )
            predictions = [
                model.predict(example.values, model.trained_at)
                for example in validation
            ]
            split.update(_diagnostics(training, validation, predictions))
            model = replace(
                model,
                validation_accuracy=split["overall"]["accuracy"],
                validation_brier=split["overall"]["brier"],
            )
            journal.record(
                Entity.MODEL, model.trained_at, model, pair=pair, timeframe="M1"
            )
            report.update(
                status="TRAINED",
                reason="fixed chronological holdout with purged labels and continuous windows",
                model=json_value(model),
            )
    report["completed_at"] = max(
        datetime.now(UTC), data.candles[-1].timestamp
    ).isoformat()
    journal.record(
        Entity.EVALUATION,
        datetime.fromisoformat(report["completed_at"]),
        {"kind": "hf_forex_continuity_experiment", **report},
        pair=pair,
        timeframe="M1",
    )
    return report
