"""Offline regression tests for the research campaign runner; all prices are synthetic."""

import json
import lzma
import struct
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from astra.backtest import BacktestCosts
from astra.config import Settings
from astra.historical.config import HistoricalSettings
from astra.historical.pipeline import run_pipeline
from astra.repositories import Entity, Journal, MemoryRepository
from astra.research import campaign, research_main

# Monday 2024-01-01 through Tuesday 2024-01-02 avoids FX weekend closures so the
# synthetic full-day files pass the same quality gate as real data.
START = date(2024, 1, 1)
END = date(2024, 1, 2)
COSTS = {"EURUSD": BacktestCosts(0.0001, 0.00002, 0.0, 1.0, "bid")}
SETTINGS = Settings(trading_enabled=True, max_spread=0.0001, max_slippage=0.00002)


def _day_payload(day_index: int) -> bytes:
    """Deterministic intraday wobble around a drifting base; synthetic prices only."""
    records = []
    state = 110000 + day_index * 3
    for minute in range(1440):
        state += ((minute * 7 + day_index * 13) % 5) - 2
        opening = state
        closing = state + (((minute * 3 + day_index) % 3) - 1)
        records.append(
            struct.pack(
                ">IIIIIf",
                minute * 60,
                opening,
                closing,
                min(opening, closing) - 1,
                max(opening, closing) + 1,
                1.0,
            )
        )
    return lzma.compress(b"".join(records))


class _SyntheticProvider:
    """Deterministic stand-in for the Dukascopy feed; no network."""

    def fetch_m1_day(self, symbol, day, timeout):
        return _day_payload((day - START).days)


@pytest.fixture(scope="module")
def managed_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("managed-dataset")
    settings = HistoricalSettings(
        root=root / "dataset",
        symbols=("EURUSD",),
        start_date=START,
        end_date=END,
    )
    assert run_pipeline(settings, provider=_SyntheticProvider()).successful
    return settings.root


def _run_campaign(root, **kwargs):
    options = {
        "root": root,
        "symbols": ("EURUSD",),
        "timeframe": "M15",
        "costs": COSTS,
        "journal": Journal(MemoryRepository()),
        "settings": SETTINGS,
        "strategies": ("ema_trend",),
        "train_size": 50,
        "test_size": 25,
    }
    options.update(kwargs)
    return campaign(**options)


def test_campaign_completes_and_records_evidence(managed_root):
    journal = Journal(MemoryRepository())
    report = _run_campaign(managed_root, journal=journal)

    assert report["status"] == "COMPLETED"
    assert len(report["experiment_id"]) == 64
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result["pair"] == "EURUSD"
    assert result["timeframe"] == "M15"
    assert result["strategy"] == "ema_trend"
    # Evidence-generating stages all ran and were journalled.
    for key in ("analysis", "paper", "backtest", "walk_forward", "robustness"):
        assert key in result
    assert result["walk_forward"], "walk-forward folds must exist for a 192-bar window"
    assert result["paper"]["approvals_enabled"] is True
    assert report["limitations"], "campaigns must carry explicit limitations"

    records = journal.repository.query(Entity.AUDIT)
    assert any(
        record.payload.get("status") == "COMPLETED"
        and record.payload.get("experiment_id") == report["experiment_id"]
        for record in records
    )
    assert report["journal_record_id"] == records[-1].record_id
    # The research evidence sources are separated, not pooled.
    sources = {
        record.payload.get("source")
        for record in journal.repository.query(Entity.PERFORMANCE)
    }
    assert sources <= {"BACKTEST", "OOS", "PAPER"}


def test_campaign_is_deterministic_for_identical_inputs(managed_root):
    first = _run_campaign(managed_root)
    second = _run_campaign(managed_root)
    assert first["experiment_id"] == second["experiment_id"]
    assert first["results"][0]["backtest"] == second["results"][0]["backtest"]
    assert first["results"][0]["walk_forward"] == second["results"][0]["walk_forward"]


def test_campaign_reports_progress_per_pair_and_strategy(managed_root):
    seen = []
    _run_campaign(managed_root, progress=seen.append)
    assert seen == ["Evaluating EURUSD M15 ema_trend"]


def test_campaign_rejects_ambiguous_or_impossible_requests(managed_root):
    base = {
        "root": managed_root,
        "symbols": ("EURUSD",),
        "timeframe": "M15",
        "costs": COSTS,
        "journal": Journal(MemoryRepository()),
        "settings": SETTINGS,
        "train_size": 50,
        "test_size": 25,
    }
    with pytest.raises(ValueError, match="unique symbols and strategies"):
        campaign(**{**base, "symbols": ("EURUSD", "EURUSD")})
    with pytest.raises(ValueError, match="unique symbols and strategies"):
        campaign(**base, strategies=())
    with pytest.raises(KeyError):
        campaign(**base, strategies=("not_a_strategy",))
    with pytest.raises(KeyError):
        campaign(**{**base, "costs": {}})
    with pytest.raises(ValueError, match="insufficient candles"):
        campaign(**{**base, "train_size": 10_000})


def test_research_main_writes_campaign_report_and_journal(tmp_path, capsys):
    settings = HistoricalSettings(
        root=tmp_path / "dataset",
        symbols=("EURUSD",),
        start_date=START,
        end_date=END,
    )
    assert run_pipeline(settings, provider=_SyntheticProvider()).successful
    output = tmp_path / "out"
    argv = [
        "--dataset-root",
        str(settings.root),
        "--output",
        str(output),
        "--symbols",
        "EURUSD",
        "--timeframe",
        "M15",
        "--strategies",
        "ema_trend",
        "--train-size",
        "50",
        "--test-size",
        "25",
        "--spread-pips",
        "1",
        "--slippage-pips",
        "0.2",
    ]
    with patch.object(sys, "argv", ["astra", "research", *argv]):
        assert research_main(argv) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "COMPLETED"
    assert summary["experiments"] == 1
    report_path = Path(summary["report"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["experiment_id"][:16] in report_path.name
    assert report["results"][0]["pair"] == "EURUSD"
    assert (output / "journal.sqlite3").is_file()


def test_research_main_denies_paper_orders_without_explicit_flag(tmp_path, capsys):
    settings = HistoricalSettings(
        root=tmp_path / "dataset",
        symbols=("EURUSD",),
        start_date=START,
        end_date=END,
    )
    assert run_pipeline(settings, provider=_SyntheticProvider()).successful
    argv = [
        "--dataset-root",
        str(settings.root),
        "--output",
        str(tmp_path / "out"),
        "--symbols",
        "EURUSD",
        "--timeframe",
        "M15",
        "--train-size",
        "50",
        "--test-size",
        "25",
        "--spread-pips",
        "1",
        "--slippage-pips",
        "0.2",
    ]
    with patch.object(sys, "argv", ["astra", "research", *argv]):
        assert research_main(argv) == 0
    report_path = Path(json.loads(capsys.readouterr().out)["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Without --paper-enabled, risk approvals stay off and no orders fill.
    assert report["request"]["settings"]["trading_enabled"] is False
    assert report["results"][0]["paper"]["approvals_enabled"] is False
    assert report["results"][0]["paper"]["trades"] == 0
