"""Optional chronological learning. No trained model or fabricated performance is bundled."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import exp, isfinite

from .data import validate
from .indicators import atr, true_ranges
from .repositories import Entity, json_value

FEATURE_NAMES = ("return_1", "return_5", "relative_atr", "body_fraction")


def features(data, at):
    candles = tuple(c for c in data.candles if c.timestamp <= at)
    from dataclasses import replace

    validate(replace(data, candles=candles), 15)
    closes = [c.close for c in candles]
    last = candles[-1]
    average_range = atr([c.high for c in candles], [c.low for c in candles], closes)[-1]
    return (
        closes[-1] / closes[-2] - 1,
        closes[-1] / closes[-6] - 1,
        average_range / last.close,
        (last.close - last.open) / max(last.high - last.low, 1e-12),
    )


@dataclass(frozen=True)
class Example:
    timestamp: datetime
    label_available_at: datetime
    values: tuple[float, ...]
    label: int


def dataset(data, horizon=3):
    validate(data)
    if horizon < 1:
        raise ValueError("label horizon must be positive")
    candles = data.candles
    closes = [c.close for c in candles]
    ranges = true_ranges([c.high for c in candles], [c.low for c in candles], closes)
    return tuple(
        Example(
            c.timestamp,
            candles[i + horizon].timestamp,
            (
                closes[i] / closes[i - 1] - 1,
                closes[i] / closes[i - 5] - 1,
                sum(ranges[i - 13 : i + 1]) / 14 / closes[i],
                (c.close - c.open) / max(c.high - c.low, 1e-12),
            ),
            int(candles[i + horizon].close > c.close),
        )
        for i, c in enumerate(candles)
        if 14 <= i < len(candles) - horizon
    )


def chronological_split(examples, fraction=0.7):
    if not 0 < fraction < 1 or len(examples) < 4:
        raise ValueError("insufficient chronological training/validation data")
    if any(a.timestamp >= b.timestamp for a, b in pairwise(examples)):
        raise ValueError("ML examples must be strictly chronological")
    split = int(len(examples) * fraction)
    test = examples[split:]
    # Purge labels crossing the validation boundary, including labels at that timestamp.
    train = tuple(
        e for e in examples[:split] if e.label_available_at < test[0].timestamp
    )
    if not train:
        raise ValueError("no training labels available before validation")
    return train, test


@dataclass(frozen=True)
class Model:
    version: str
    trained_at: datetime
    training_end: datetime
    validation_start: datetime
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    validation_accuracy: float
    validation_brier: float
    sample_count: int
    feature_schema: tuple[str, ...] = FEATURE_NAMES

    def predict(self, values, at):
        if at < self.trained_at:
            raise ValueError("model did not exist at inference time")
        if len(values) != len(FEATURE_NAMES) or not all(isfinite(v) for v in values):
            raise ValueError("invalid inference features")
        z = self.intercept + sum(
            w * (v - m) / s
            for v, m, s, w in zip(values, self.means, self.scales, self.coefficients)
        )
        return 1 / (1 + exp(-max(-700, min(700, z))))


@dataclass(frozen=True)
class TrainingResult:
    status: str
    reason: str
    model: Model | None = None


def train(data, *, enabled=False, minimum_samples=300, journal=None, trained_at=None):
    if not enabled:
        return TrainingResult(
            "DISABLED",
            "ML requires explicit opt-in and sufficient verified real history",
        )
    if data.metadata.get("real_data") is not True:
        return TrainingResult(
            "DISABLED", "real historical data provenance has not been confirmed"
        )
    examples = dataset(data)
    if len(examples) < max(100, minimum_samples):
        return TrainingResult(
            "INSUFFICIENT_DATA", "insufficient labeled real historical observations"
        )
    training, validation = chronological_split(examples)
    if len({e.label for e in training}) < 2:
        return TrainingResult(
            "INSUFFICIENT_DATA", "training requires both outcome classes"
        )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return TrainingResult("DISABLED", "install Astra's optional ml extra")
    scaler = StandardScaler().fit([e.values for e in training])
    estimator = LogisticRegression(random_state=0, max_iter=1000).fit(
        scaler.transform([e.values for e in training]), [e.label for e in training]
    )
    predictions = estimator.predict_proba(
        scaler.transform([e.values for e in validation])
    )[:, 1]
    accuracy = sum(
        (p >= 0.5) == e.label for p, e in zip(predictions, validation)
    ) / len(validation)
    brier = sum(
        (float(p) - e.label) ** 2 for p, e in zip(predictions, validation)
    ) / len(validation)
    version = sha256(
        json.dumps(json_value(examples), sort_keys=True).encode()
    ).hexdigest()[:16]
    available_at = max(
        trained_at or datetime.now(UTC), validation[-1].label_available_at
    )
    model = Model(
        version,
        available_at,
        training[-1].label_available_at,
        validation[0].timestamp,
        tuple(float(v) for v in scaler.mean_),
        tuple(float(v) for v in scaler.scale_),
        tuple(float(v) for v in estimator.coef_[0]),
        float(estimator.intercept_[0]),
        accuracy,
        brier,
        len(examples),
    )
    if journal:
        journal.record(
            Entity.MODEL,
            model.trained_at,
            model,
            pair=data.pair,
            timeframe=data.timeframe,
        )
    return TrainingResult(
        "TRAINED",
        "chronological holdout; probabilities require ongoing calibration monitoring",
        model,
    )


def load_model(journal, *, pair, timeframe, as_of):
    rows = journal.repository.query(
        Entity.MODEL, pair=pair, timeframe=timeframe, as_of=as_of
    )
    if not rows:
        return None
    payload = dict(rows[-1].payload)
    for key in ("trained_at", "training_end", "validation_start"):
        payload[key] = datetime.fromisoformat(payload[key])
    for key in ("means", "scales", "coefficients", "feature_schema"):
        payload[key] = tuple(payload[key])
    if payload["feature_schema"] != FEATURE_NAMES or payload["trained_at"] > as_of:
        raise ValueError("model schema or availability mismatch")
    return Model(**payload)


def monitor(model, examples, journal=None):
    if not examples:
        return {"status": "INSUFFICIENT_DATA", "sample_count": 0}
    scores = [model.predict(e.values, e.timestamp) for e in examples]
    brier = sum((p - e.label) ** 2 for p, e in zip(scores, examples)) / len(examples)
    result = {
        "status": "DEGRADED" if brier > model.validation_brier + 0.1 else "HEALTHY",
        "sample_count": len(examples),
        "brier": brier,
        "model_version": model.version,
    }
    if journal:
        journal.record(
            Entity.CALIBRATION, max(e.label_available_at for e in examples), result
        )
    return result
