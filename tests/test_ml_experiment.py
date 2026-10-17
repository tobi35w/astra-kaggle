"""Offline tests for the ML training experiment runner; synthetic fixtures only."""

import json
import lzma
import struct
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_models_experiment import (
    _is_managed_real_data,
    main,
    run,
)

START = date(2024, 1, 1)
END = date(2024, 1, 4)


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
    def fetch_m1_day(self, symbol, day, timeout):
        return _day_payload((day - START).days)


@pytest.fixture(scope="module")
def managed_root(tmp_path_factory):
    from astra.historical.config import HistoricalSettings
    from astra.historical.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("ml-root")
    settings = HistoricalSettings(
        root=root / "dataset",
        symbols=("EURUSD",),
        start_date=START,
        end_date=END,
    )
    assert run_pipeline(settings, provider=_SyntheticProvider()).successful
    return settings.root


def test_provenance_gate_requires_managed_quality_pass_data(market_data):
    synthetic = market_data(40)
    assert _is_managed_real_data(synthetic) is False
    labelled = market_data(40)
    labelled = type(labelled)(
        labelled.pair,
        labelled.timeframe,
        labelled.candles,
        {
            "source": "synthetic_test_fixture",
            "quality_status": "PASS",
            "provenance": "managed Dukascopy acquisition; not independently authenticated",
        },
    )
    assert _is_managed_real_data(labelled) is False
    managed = type(labelled)(
        labelled.pair,
        labelled.timeframe,
        labelled.candles,
        {
            "source": "Dukascopy-node M1 JSON, BID",
            "quality_status": "PASS",
            "provenance": "managed Dukascopy JSON; not independently authenticated",
        },
    )
    assert _is_managed_real_data(managed) is True


def test_runner_trains_or_reports_insufficient_on_a_managed_window(managed_root):
    """Exercises the runner's mechanics on a synthetic fixture window, not a real experiment."""
    from astra.repositories import Entity, Journal, MemoryRepository

    journal = Journal(MemoryRepository())
    report = run(managed_root, pair="EURUSD", timeframe="M5", journal=journal)
    assert report["status"] == "TRAINED"
    assert report["caveat"].startswith("Chronological split")
    model = report["model"]
    assert 0.0 <= model["validation_accuracy"] <= 1.0
    assert 0.0 <= model["validation_brier"] <= 1.0
    assert model["sample_count"] >= 300
    # The purge discipline must hold: training ends before validation starts.
    assert model["training_end"] < model["validation_start"]
    assert journal.repository.query(Entity.MODEL)


def test_runner_refuses_unmanaged_data(managed_root, tmp_path, monkeypatch):
    import scripts.train_models_experiment as runner

    original = runner.load_dataset

    def stripped(root, pair, timeframe):
        data = original(root, pair, timeframe)
        return type(data)(
            data.pair,
            data.timeframe,
            data.candles,
            data.metadata | {"provenance": "unverified external copy"},
        )

    monkeypatch.setattr(runner, "load_dataset", stripped)
    report = runner.run(managed_root, pair="EURUSD", timeframe="M5")
    assert report["status"] == "REFUSED"
    assert "managed" in report["reason"]


def test_cli_reports_and_writes_output(managed_root, tmp_path, capsys):
    output = tmp_path / "ml-report.json"
    code = main(
        [
            "--dataset-root",
            str(managed_root),
            "--pair",
            "EURUSD",
            "--timeframe",
            "M5",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "TRAINED"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "TRAINED"
