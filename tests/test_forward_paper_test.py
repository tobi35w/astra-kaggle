"""Offline tests for the forward paper-test runner; all fixture prices are synthetic."""

import json
import lzma
import struct
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_forward_paper_test import ACTIVE_SYMBOLS, main, run

START = date(2024, 1, 1)
END = date(2024, 1, 2)


def _day_payload(day_index: int) -> bytes:
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
    """Deterministic Dukascopy stand-in; no network."""

    def fetch_m1_day(self, symbol, day, timeout):
        return _day_payload((day - START).days)


@pytest.fixture(scope="module")
def managed_root(tmp_path_factory):
    from astra.historical.config import HistoricalSettings
    from astra.historical.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("forward-root")
    settings = HistoricalSettings(
        root=root / "dataset",
        symbols=("EURUSD",),
        start_date=START,
        end_date=END,
    )
    assert run_pipeline(settings, provider=_SyntheticProvider()).successful
    return settings.root


def test_forward_paper_test_replays_every_pair(managed_root):
    report = run(managed_root, symbols=("EURUSD",), timeframe="M5")
    assert report["status"] == "COMPLETED"
    assert report["caveat"].startswith("Finite historical replay")
    eur = report["symbols"]["EURUSD"]
    assert eur["status"] == "COMPLETED"
    assert eur["candles"] == 576
    assert eur["decisions"] == 576
    assert eur["orders"] >= eur["filled_orders"]
    assert set(eur) >= {"trades", "wins", "net_pnl", "final_equity", "dataset_version"}
    assert report["assumptions"]["price_basis"] == "bid"
    assert len(report["assumptions"]["strategies"]) == 10


def test_forward_paper_test_isolates_symbol_failures(managed_root):
    report = run(managed_root, symbols=("EURUSD", "GBPUSD"), timeframe="M5")
    assert report["status"] == "PARTIAL"
    assert report["symbols"]["EURUSD"]["status"] == "COMPLETED"
    assert report["symbols"]["GBPUSD"]["status"] == "FAILED"
    assert "error_type" in report["symbols"]["GBPUSD"]


def test_forward_paper_test_cli_writes_report(tmp_path, managed_root, capsys):
    output = tmp_path / "forward.json"
    code = main(
        [
            "--dataset-root",
            str(managed_root),
            "--symbols",
            "EURUSD",
            "--timeframe",
            "M5",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED"
    assert report["symbols"]["EURUSD"]["status"] == "COMPLETED"


def test_active_symbols_match_the_curated_universe():
    assert ACTIVE_SYMBOLS == ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
