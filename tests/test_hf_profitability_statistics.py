"""Synthetic P&L/statistics fixtures; not historical market evidence."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

from scripts.hf_profitability_statistics import paired_difference, summarize


def sample(returns, *, gap_at=None):
    equity = 1000.0
    days, trades = [], []
    for i, rate in enumerate(returns):
        stamp = date(2025, 1, 6) + timedelta(
            days=i + (180 if gap_at is not None and i >= gap_at else 0)
        )
        pnl = equity * rate
        days.append(
            {
                "date": str(stamp),
                "start_equity": equity,
                "end_equity": equity + pnl,
                "pnl": pnl,
                "return": rate,
                "trade_count": int(pnl != 0),
                "observed_bars": 50,
            }
        )
        if pnl:
            trades.append({"pnl": pnl, "pnl_per_unit": pnl / 2, "size": 2})
        equity += pnl
    return {
        "initial_equity": 1000.0,
        "final_equity": equity,
        "account_currency": "JPY",
        "daily": days,
        "trades": trades,
        "max_drawdown_pct": 0.0,
    }


def test_negative_constant_returns_and_seed_are_reproducible():
    result = sample([-0.001] * 20)
    first = summarize(result, bootstrap_replicates=100)
    assert first == summarize(result, bootstrap_replicates=100)
    assert first["bootstrap_mean_daily_return"]["high_95"] == pytest.approx(-0.001)
    assert first["profit_factor"] == 0
    assert first["net_pnl"] == pytest.approx(sum(t["pnl"] for t in result["trades"]))
    assert first["pnl_reconciled"]


def test_inactive_observed_days_retained_and_outage_never_crossed_by_blocks():
    result = sample([0.01, 0, 0, 0, 0, -0.01, 0, 0, 0, 0], gap_at=5)
    report = summarize(result, bootstrap_replicates=100)
    assert report["observed_days"] == 10
    assert report["active_days"] == 2
    assert report["bootstrap_mean_daily_return"]["segment_lengths"] == [5, 5]


@pytest.mark.parametrize("issue", ["pnl", "return", "date", "observed"])
def test_invalid_evidence_fails_reconciliation(issue):
    result = sample([0.01, -0.01])
    if issue == "pnl":
        result["trades"][0]["pnl_per_unit"] *= 2
    elif issue == "return":
        result["daily"][0]["return"] = 42
    elif issue == "date":
        result["daily"][1]["date"] = result["daily"][0]["date"]
    else:
        result["daily"][0]["observed_bars"] = 0
    with pytest.raises(ValueError):
        summarize(result, bootstrap_replicates=100)


def test_paired_days_have_exact_zero_difference_for_identical_results():
    result = sample([0.01, -0.01, 0, 0.002, 0] * 4)
    report = paired_difference(result, deepcopy(result), bootstrap_replicates=100)
    assert report["bootstrap_mean_daily_return_difference"]["low_95"] == 0
    assert report["bootstrap_mean_daily_return_difference"]["high_95"] == 0
    assert report["net_pnl_difference"] == 0


def test_paired_currency_and_dates_cannot_be_silently_mixed():
    result = sample([0.01, -0.01])
    changed = deepcopy(result)
    changed["account_currency"] = "USD"
    with pytest.raises(ValueError, match="currency"):
        paired_difference(result, changed, bootstrap_replicates=100)
    changed = sample([0.01, -0.01], gap_at=1)
    with pytest.raises(ValueError, match="dates"):
        paired_difference(result, changed, bootstrap_replicates=100)


def test_replay_drawdown_fraction_becomes_percentage():
    result = sample([-0.01] * 10)
    result["max_drawdown_pct"] = 0.12
    assert summarize(result, bootstrap_replicates=100)["replay_max_drawdown_pct"] == 12
