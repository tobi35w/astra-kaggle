"""Descriptive uncertainty for conditional replays on previously inspected data."""

from __future__ import annotations

import math
import random
from datetime import date
from itertools import pairwise
from statistics import mean

CAVEAT = (
    "Descriptive moving-block bootstrap on reused development history; not an "
    "independent significance test or forecast. Blocks contain at most five "
    "observed days and never cross an observation gap longer than four calendar "
    "days. Recorded inactive days are included; missing days are not fabricated."
)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def _validate(result):
    daily = result["daily"]
    if not daily or not result.get("account_currency"):
        raise ValueError(
            "observed daily equity and explicit account currency are required"
        )
    initial = result["initial_equity"]
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("initial capital must be positive")
    stamps = [date.fromisoformat(d["date"]) for d in daily]
    if any(a >= b for a, b in pairwise(stamps)):
        raise ValueError("daily observations must be strictly chronological")
    previous = initial
    for day in daily:
        values = [day[k] for k in ("start_equity", "end_equity", "pnl", "return")]
        if not all(math.isfinite(v) for v in values) or day["observed_bars"] <= 0:
            raise ValueError("daily rows must be finite observed days")
        if not _close(day["start_equity"], previous) or not _close(
            day["pnl"], day["end_equity"] - previous
        ):
            raise ValueError("daily equity does not reconcile")
        if not _close(day["return"], day["pnl"] / previous if previous else 0):
            raise ValueError("daily return does not reconcile")
        previous = day["end_equity"]
    for trade in result["trades"]:
        if not _close(trade["pnl"], trade["pnl_per_unit"] * trade["size"]):
            raise ValueError("total and per-unit trade P&L do not reconcile")
    net = previous - initial
    if not _close(net, sum(t["pnl"] for t in result["trades"])):
        raise ValueError("closed trades and final equity do not reconcile")
    if sum(d["trade_count"] for d in daily) != len(result["trades"]):
        raise ValueError("daily and total trade counts differ")
    return stamps


def block_interval(stamps, values, *, replicates=2000, seed=20261016):
    if len(stamps) != len(values) or not values or replicates < 100:
        raise ValueError(
            "nonempty aligned days and at least 100 bootstrap replicates required"
        )
    segments = [[]]
    for i, value in enumerate(values):
        if i and (stamps[i] - stamps[i - 1]).days > 4:
            segments.append([])
        segments[-1].append(value)
    blocks = []
    for segment in segments:
        width = min(5, len(segment))
        blocks.extend(segment[i : i + width] for i in range(len(segment) - width + 1))
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sample = []
        while len(sample) < len(values):
            sample.extend(rng.choice(blocks))
        means.append(mean(sample[: len(values)]))
    means.sort()
    return {
        "low_95": means[int(0.025 * (replicates - 1))],
        "high_95": means[int(0.975 * (replicates - 1))],
        "replicates": replicates,
        "seed": seed,
        "block_days": 5,
        "segment_lengths": list(map(len, segments)),
        "observed_days": len(values),
        "caveat": CAVEAT,
    }


def summarize(result, *, bootstrap_replicates=2000, seed=20261016):
    stamps = _validate(result)
    daily, trades = result["daily"], result["trades"]
    initial = result["initial_equity"]
    final = daily[-1]["end_equity"]
    profits = sum(max(0, t["pnl"]) for t in trades)
    losses = -sum(min(0, t["pnl"]) for t in trades)
    peak, drawdown = initial, 0.0
    months = {}
    for day in daily:
        peak = max(peak, day["end_equity"])
        drawdown = max(drawdown, (peak - day["end_equity"]) / peak)
        month = months.setdefault(
            day["date"][:7], {"net_pnl": 0, "trade_count": 0, "observed_days": 0}
        )
        month["net_pnl"] += day["pnl"]
        month["trade_count"] += day["trade_count"]
        month["observed_days"] += 1
    return {
        "account_currency": result["account_currency"],
        "trade_count": len(trades),
        "net_pnl": final - initial,
        "total_return_pct": (final / initial - 1) * 100,
        "win_rate": sum(t["pnl"] > 0 for t in trades) / len(trades) if trades else None,
        "profit_factor": profits / losses if losses else None,
        "expectancy_pnl": (final - initial) / len(trades) if trades else None,
        "observed_days": len(daily),
        "active_days": sum(d["trade_count"] > 0 for d in daily),
        "daily_close_max_drawdown_pct": drawdown * 100,
        "replay_max_drawdown_pct": result["max_drawdown_pct"] * 100,
        "mean_daily_return": mean(d["return"] for d in daily),
        "bootstrap_mean_daily_return": block_interval(
            stamps,
            [d["return"] for d in daily],
            replicates=bootstrap_replicates,
            seed=seed,
        ),
        "by_month": months,
        "pnl_reconciled": True,
        "profitability_proven": False,
    }


def paired_difference(
    model_result, baseline_result, *, bootstrap_replicates=2000, seed=20261016
):
    stamps = _validate(model_result)
    other_stamps = _validate(baseline_result)
    if stamps != other_stamps:
        raise ValueError("paired scenarios must have identical observed dates")
    if (
        model_result["account_currency"] != baseline_result["account_currency"]
        or model_result["initial_equity"] != baseline_result["initial_equity"]
    ):
        raise ValueError("paired scenarios must share currency and initial capital")
    values = [
        a["return"] - b["return"]
        for a, b in zip(model_result["daily"], baseline_result["daily"])
    ]
    return {
        "mean_daily_return_difference": mean(values),
        "bootstrap_mean_daily_return_difference": block_interval(
            stamps, values, replicates=bootstrap_replicates, seed=seed
        ),
        "net_pnl_difference": model_result["daily"][-1]["end_equity"]
        - baseline_result["daily"][-1]["end_equity"],
        "account_currency": model_result["account_currency"],
        "caveat": CAVEAT,
    }
