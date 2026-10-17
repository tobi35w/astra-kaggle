"""Offline tests for the qleap parquet source bridge; all fixture bars are synthetic."""

import json
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astra.data import DataValidationError
from astra.historical.dataset import load_dataset
from astra.historical.qleap_source import process_qleap_parquet

SYMBOL = "EURUSD"
# Monday 2024-01-08 00:00 UTC through 23:59 UTC is entirely inside the FX session.
DAY = datetime(2024, 1, 8, tzinfo=UTC)
START = datetime(2024, 1, 8, tzinfo=UTC)
END_EXCLUSIVE = datetime(2024, 1, 9, tzinfo=UTC)


def _table(rows):
    """rows: iterable of (close-timestamp, open, high, low, close, volume)."""
    return pa.table(
        {
            "timestamp": [r[0] for r in rows],
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [r[5] for r in rows],
        }
    )


def _session_bars():
    # qleap convention: the first bar of a UTC day closes at 00:01 (opening 00:00).
    rows = []
    for minute in range(1, 1441):
        stamp = DAY + timedelta(minutes=minute)
        base = 1.1 + minute * 1e-6
        rows.append(
            (stamp, base, base + 2e-5, base - 2e-5, base + 1e-5, 10.0 + minute % 7)
        )
    return rows


def _weekend_bars():
    # Friday 22:30 UTC, Saturday midday, Sunday 12:00 UTC are all closed-session bars.
    return [
        (datetime(2024, 1, 5, 22, 30, tzinfo=UTC), 1.1, 1.10002, 1.09998, 1.10001, 5.0),
        (datetime(2024, 1, 6, 12, 0, tzinfo=UTC), 1.1, 1.10002, 1.09998, 1.10001, 5.0),
        (datetime(2024, 1, 7, 12, 0, tzinfo=UTC), 1.1, 1.10002, 1.09998, 1.10001, 5.0),
    ]


@pytest.fixture
def source_dir(tmp_path):
    source = tmp_path / "qleap"
    source.mkdir()
    pq.write_table(
        _table(_session_bars() + _weekend_bars()), source / "2024_01a.parquet"
    )
    return source


@pytest.fixture
def converted(source_dir, tmp_path):
    root = tmp_path / "astra-qleap-source"
    return root, process_qleap_parquet(
        source_dir,
        SYMBOL,
        root,
        max_gap_minutes=1,
        expected_start=START,
        expected_end=END_EXCLUSIVE,
    )


def test_conversion_excludes_closed_sessions_and_keeps_session_bars(converted):
    root, result = converted
    assert result["symbol"] == SYMBOL
    assert result["unique_m1_rows"] == 1440
    assert result["excluded_closed_session_rows"] == 3
    assert result["duplicate_rows"] == 0
    assert result["timeframes"] == {
        "M1": "PASS",
        "M5": "PASS",
        "M15": "PASS",
        "H1": "PASS",
        "H4": "PASS",
    }
    report = json.loads((root / "reports" / "eurusd-qleap-parquet.json").read_text())
    assert report["status"] == "PASS"
    assert report["timestamp_conversion"].startswith("naive UTC candle-close")
    assert any("do not mix sources" in item for item in report["limitations"])
    assert report["outputs"]["M5"]["rows"] == 288
    assert len(report["source_sha256"]) == 1


def test_converted_root_loads_through_the_managed_boundary(converted):
    root, _ = converted
    data = load_dataset(root, SYMBOL, "M5")
    assert data.metadata["quality_status"] == "PASS"
    assert data.metadata["price_basis"] == "bid"
    assert data.metadata["provenance"].startswith("converted qleap")
    assert len(data.candles) == 288
    assert data.candles[0].timestamp == DAY + timedelta(minutes=5)
    # A tampered file must be rejected by the checksum gate.
    target = root / "processed" / "m5" / f"{SYMBOL}.csv"
    target.write_text(target.read_text().replace("1.100001", "1.100009"))
    with pytest.raises(DataValidationError, match="changed after validation"):
        load_dataset(root, SYMBOL, "M5")


def test_conflicting_duplicate_candles_are_rejected(tmp_path):
    source = tmp_path / "conflict"
    source.mkdir()
    rows = _session_bars()[:10]
    conflict = (
        rows[0][0],
        rows[0][1] + 1e-4,
        rows[0][2],
        rows[0][3],
        rows[0][4],
        rows[0][5],
    )
    pq.write_table(_table(rows + [conflict]), source / "2024_01.parquet")
    with pytest.raises(ValueError, match="conflicting duplicate candle"):
        process_qleap_parquet(source, SYMBOL, tmp_path / "out")


def test_identical_duplicate_candles_are_counted_not_fatal(tmp_path):
    source = tmp_path / "dupe"
    source.mkdir()
    rows = _session_bars()[:10]
    pq.write_table(_table(rows + [rows[0]]), source / "2024_01a.parquet")
    result = process_qleap_parquet(source, SYMBOL, tmp_path / "out")
    assert result["duplicate_rows"] == 1
    assert result["unique_m1_rows"] == 10


def test_missing_parquet_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no parquet files"):
        process_qleap_parquet(tmp_path / "absent", SYMBOL, tmp_path / "out")


def test_cli_process_parquet_reports_quality_and_exit_code(
    source_dir, tmp_path, capsys
):
    from astra.historical.cli import historical_main

    root = tmp_path / "out"
    code = historical_main(
        [
            "process-parquet",
            SYMBOL,
            str(source_dir),
            "--root",
            str(root),
            "--start",
            "2024-01-08",
            "--end",
            "2024-01-08",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["timeframes"]["M1"] == "PASS"

    # A range the source cannot satisfy fails closed instead of passing quietly.
    code = historical_main(
        [
            "process-parquet",
            SYMBOL,
            str(source_dir),
            "--root",
            str(tmp_path / "out2"),
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )
    assert code == 1
