from __future__ import annotations

import json
import lzma
import struct
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from astra.historical.config import HistoricalSettings
from astra.historical.downloader import HistoricalDownloader, iter_days
from astra.historical.dukascopy import DukascopyProvider, decode_m1_bi5
from astra.historical.dukascopy_node import (
    DukascopyNodeDownloader,
    _valid_json_export,
)
from astra.historical.importer import import_csv
from astra.historical.json_source import process_dukascopy_json
from astra.historical.manifest import ManifestStore
from astra.historical.pipeline import run_pipeline
from astra.historical.resampler import resample
from astra.historical.validation import _market_open, inspect_csv
from astra.history import load_csv
from astra.models import Candle
from astra.repositories import Journal, MemoryRepository


def _payload(rows):
    return lzma.compress(b"".join(struct.pack(">IIIIIf", *row) for row in rows))


def _complete_json_rows(start, end):
    """Synthetic minute records for the requested FX session; never real market data."""
    current = datetime.combine(start, datetime.min.time(), UTC)
    stop = datetime.combine(end, datetime.min.time(), UTC)
    rows = []
    while current < stop:
        if _market_open(current):
            rows.append(
                {
                    "timestamp": int(current.timestamp() * 1000),
                    "open": 1.1,
                    "high": 1.1,
                    "low": 1.1,
                    "close": 1.1,
                    "volume": 0,
                }
            )
        current += timedelta(minutes=1)
    return rows


def test_bi5_utc_close_timestamp_and_volume_type():
    candle = decode_m1_bi5(
        _payload([(0, 100000, 110000, 90000, 120000, 3.5)]), "EURUSD", date(2024, 1, 1)
    )[0]
    assert candle.timestamp == datetime(2024, 1, 1, 0, 1, tzinfo=UTC)
    assert (candle.open, candle.close, candle.low, candle.high) == (1.0, 1.1, 0.9, 1.2)
    assert candle.volume_type == "provider"


def test_url_uses_zero_based_month_and_expected_filename():
    url = DukascopyProvider().build_m1_url("EURUSD", date(2024, 1, 2))
    assert url.endswith("/EURUSD/2024/00/02/BID_candles_min_1.bi5")


def test_iter_days_includes_bounds():
    assert tuple(iter_days(date(2024, 1, 1), date(2024, 1, 3))) == (
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    )


def test_resample_only_emits_complete_utc_bars():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = tuple(
        Candle(start + timedelta(minutes=i + 1), 1, 2, 0.5, 1.5, 2) for i in range(6)
    )
    result = resample(candles, "M5")
    assert len(result) == 1
    assert result[0].timestamp == start + timedelta(minutes=5)
    assert result[0].volume == 10
    assert result[0].high == 2


def test_downloader_atomic_cache_resume_and_dry_run(tmp_path):
    class Provider:
        calls = 0

        def fetch_m1_day(self, symbol, day, timeout):
            self.calls += 1
            return _payload([(0, 100000, 110000, 90000, 120000, 1)])

    provider = Provider()
    manifest = ManifestStore(tmp_path)
    downloader = HistoricalDownloader(
        provider, root=tmp_path, manifest=manifest, sleep=lambda _: None
    )
    day = date(2024, 1, 1)
    planned = downloader.download(("EURUSD",), day, day, dry_run=True)
    assert planned.results[0].status == "planned"
    assert provider.calls == 0
    first = downloader.download(("EURUSD",), day, day)
    second = downloader.download(("EURUSD",), day, day)
    assert first.downloaded == 1
    assert second.cached == 1
    assert provider.calls == 1


def test_importer_is_idempotent_by_deterministic_record_id(tmp_path):
    path = tmp_path / "candles.csv"
    path.write_text(
        "timestamp,open,high,low,close,tick_volume\n2024-01-01T00:01:00+00:00,1,2,0.5,1.5,2\n",
        encoding="utf-8",
    )
    repo = MemoryRepository()
    journal = Journal(repo)
    assert import_csv(path, "EURUSD", "M1", journal, batch_size=1) == 1
    assert import_csv(path, "EURUSD", "M1", journal, batch_size=1) == 1
    from astra.repositories import Entity

    assert len(repo.query(Entity.OBSERVATION)) == 1
    assert repo.query(Entity.OBSERVATION)[0].payload["price_basis"] == "mid"


def test_validator_reports_csv_hash_and_bars(tmp_path):
    path = tmp_path / "m1.csv"
    path.write_text(
        "timestamp,open,high,low,close,tick_volume\n2024-01-01T00:01:00+00:00,1,2,0.5,1.5,2\n2024-01-01T00:02:00+00:00,1.5,2,1,1.6,1\n",
        encoding="utf-8",
    )
    report = inspect_csv(path, "EURUSD", "M1")
    assert report.status == "PASS"
    assert report.rows == 2
    assert len(report.checksum) == 64


def test_pipeline_runs_locally_with_injected_deterministic_provider(tmp_path):
    class Provider:
        def fetch_m1_day(self, symbol, day, timeout):
            return _payload(
                [
                    (minute * 60, 100000, 100010, 99990, 100020, 1.0)
                    for minute in range(1440)
                ]
            )

    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 1),
    )
    summary = run_pipeline(settings, provider=Provider())
    assert summary.symbols[0].status == "PASS", summary.symbols[0].timeframes
    assert summary.symbols[0].timeframes["M1"].rows == 1440
    assert summary.symbols[0].timeframes["M5"].rows == 288
    assert (tmp_path / "processed" / "h1" / "EURUSD.csv").is_file()
    assert (tmp_path / "processed" / "h4" / "EURUSD.csv").is_file()
    assert (tmp_path / "reports" / "latest.json").is_file()


def test_process_dukascopy_json_preserves_flats_and_builds_complete_timeframes(
    tmp_path,
):
    start = datetime(2024, 1, 2, tzinfo=UTC)
    rows = []
    for minute in range(2 * 24 * 60):
        opened = start + timedelta(minutes=minute)
        price = 1.1 if minute == 10 else 1.1001
        rows.append(
            {
                "timestamp": int(opened.timestamp() * 1000),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0 if minute == 10 else 3,
            }
        )
    source_a = tmp_path / "source-a.json"
    source_b = tmp_path / "source-b.json"
    source_a.write_text(json.dumps(rows[:1440]), encoding="utf-8")
    source_b.write_text(json.dumps(rows[1440:]), encoding="utf-8")

    result = process_dukascopy_json((source_a, source_b), "EURUSD", tmp_path / "out")

    assert result["unique_m1_rows"] == 2880
    assert result["source_rows"] == 2880
    assert result["zero_volume_rows"] == 1
    assert result["timeframes"] == {
        "M1": "PASS",
        "M5": "PASS",
        "M15": "PASS",
        "H1": "PASS",
        "H4": "PASS",
    }
    assert (tmp_path / "out" / "processed" / "h4" / "EURUSD.csv").is_file()
    report = json.loads(
        (tmp_path / "out" / "reports" / "eurusd-dukascopy-json.json").read_text(
            encoding="utf-8"
        )
    )
    import hashlib

    assert (
        report["source_sha256"][source_a.as_posix()]
        == hashlib.sha256(source_a.read_bytes()).hexdigest()
    )
    assert report["zero_volume_rows"] == 1
    assert report["volume_type"] == "provider"
    assert report["volume_unit"] == "millions (dukascopy-node 1.50.0 default)"
    assert report["outputs"]["M1"]["rows"] == 2880
    market = load_csv(
        tmp_path / "out" / "normalized" / "m1" / "EURUSD.csv",
        "EURUSD",
        "M1",
    )
    assert market.metadata["volume_type"] == "provider"
    assert market.metadata["volume_unit"] == "millions"
    assert market.metadata["price_basis"] == "bid"
    assert market.candles[10].volume_type == "provider"

    from astra.backtest import run_report
    from astra.strategies import EmaTrend

    with pytest.raises(ValueError, match="midpoint"):
        run_report(market, EmaTrend())


def test_process_dukascopy_json_reports_gaps_when_flat_candles_are_omitted(tmp_path):
    source = tmp_path / "sparse.json"
    start = datetime(2024, 1, 2, tzinfo=UTC)
    rows = [
        {
            "timestamp": int((start + timedelta(minutes=i)).timestamp() * 1000),
            "open": 1.1,
            "high": 1.1,
            "low": 1.1,
            "close": 1.1,
            "volume": 1,
        }
        for i in (0, 2)
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")

    result = process_dukascopy_json(source, "EURUSD", tmp_path / "out")

    assert result["timeframes"]["M1"] == "WARN"
    report = json.loads(
        (tmp_path / "out" / "reports" / "eurusd-dukascopy-json.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["outputs"]["M1"]["suspicious_gaps"][0]["missing_candles"] == 1


def test_process_dukascopy_json_excludes_flat_weekend_bars(tmp_path):
    start = datetime(2024, 1, 5, tzinfo=UTC)
    rows = [
        {
            "timestamp": int((start + timedelta(minutes=minute)).timestamp() * 1000),
            "open": 1.1,
            "high": 1.1,
            "low": 1.1,
            "close": 1.1,
            "volume": 0,
        }
        for minute in range(3 * 24 * 60)
    ]
    source = tmp_path / "friday-through-weekend.json"
    source.write_text(json.dumps(rows), encoding="utf-8")

    result = process_dukascopy_json(
        source,
        "EURUSD",
        tmp_path / "out",
        expected_start=start,
        expected_end=start + timedelta(days=3),
    )

    assert result["source_rows"] == 3 * 24 * 60
    assert result["excluded_closed_session_rows"] == 2 * 24 * 60
    assert result["unique_m1_rows"] == 24 * 60
    assert result["timeframes"] == {
        "M1": "PASS",
        "M5": "PASS",
        "M15": "PASS",
        "H1": "PASS",
        "H4": "PASS",
    }


def test_validator_classifies_higher_timeframe_weekend_gaps_as_expected(tmp_path):
    path = tmp_path / "m5-weekend.csv"
    path.write_text(
        "timestamp,open,high,low,close,tick_volume\n"
        "2024-01-05T22:00:00+00:00,1.1,1.1,1.1,1.1,0\n"
        "2024-01-07T22:05:00+00:00,1.1,1.1,1.1,1.1,0\n",
        encoding="utf-8",
    )

    report = inspect_csv(path, "EURUSD", "M5", max_gap_minutes=1)

    assert report.status == "PASS"
    assert report.expected_gap_count == 1
    assert report.suspicious_gaps == ()


def test_validator_still_reports_missing_open_session_higher_timeframe_bars(tmp_path):
    path = tmp_path / "m5-open-gap.csv"
    path.write_text(
        "timestamp,open,high,low,close,tick_volume\n"
        "2024-01-02T12:00:00+00:00,1.1,1.1,1.1,1.1,1\n"
        "2024-01-02T12:15:00+00:00,1.1,1.1,1.1,1.1,1\n",
        encoding="utf-8",
    )

    report = inspect_csv(path, "EURUSD", "M5", max_gap_minutes=1)

    assert report.status == "WARN"
    assert report.expected_gap_count == 0
    assert len(report.suspicious_gaps) == 1


def test_dukascopy_node_download_manifest_resume_and_force(tmp_path):
    calls = []

    def runner(command, output_dir):
        calls.append(command)
        filename = "eurusd-m1-bid-2024-01-01-2024-02-01.json"
        rows = [
            {
                "timestamp": 1704067200000,
                "open": 1.1,
                "high": 1.1,
                "low": 1.1,
                "close": 1.1,
                "volume": 0,
            }
        ]
        rows = _complete_json_rows(date(2024, 1, 1), date(2024, 2, 1))
        (output_dir / filename).write_text(json.dumps(rows), encoding="utf-8")

    downloader = DukascopyNodeDownloader(tmp_path, runner=runner)
    first = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))
    second = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))
    target = Path(first["symbols"]["EURUSD"]["files"][0])
    target.write_text("corrupted", encoding="utf-8")
    repaired = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))
    forced = downloader.download(
        ("EURUSD",), date(2024, 1, 1), date(2024, 1, 31), force=True
    )

    assert first["symbols"]["EURUSD"]["status"] == "downloaded"
    assert second["symbols"]["EURUSD"]["status"] == "cached"
    assert second["symbols"]["EURUSD"]["rows"] == len(
        _complete_json_rows(date(2024, 1, 1), date(2024, 2, 1))
    )
    assert repaired["symbols"]["EURUSD"]["status"] == "downloaded"
    assert forced["symbols"]["EURUSD"]["status"] == "downloaded"
    assert len(calls) == 3
    assert "-fl" in calls[0]
    assert calls[0][calls[0].index("-to") + 1] == "2024-02-01"
    assert calls[0][calls[0].index("-bp") + 1] == "1000"
    assert "-ch" not in calls[-1]


def test_dukascopy_node_recovers_after_interrupted_chunk(tmp_path):
    calls = 0

    def runner(_command, output_dir):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("simulated interruption")
        filename = "eurusd-m1-bid-2024-01-01-2024-02-01.json"
        rows = _complete_json_rows(date(2024, 1, 1), date(2024, 2, 1))
        (output_dir / filename).write_text(json.dumps(rows), encoding="utf-8")

    downloader = DukascopyNodeDownloader(tmp_path, runner=runner)
    failed = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))
    resumed = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))
    cached = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 31))

    assert failed["symbols"]["EURUSD"]["status"] == "failed"
    assert failed["symbols"]["EURUSD"]["chunks"][0]["error"] == "TimeoutError"
    assert resumed["symbols"]["EURUSD"]["status"] == "downloaded"
    assert resumed["symbols"]["EURUSD"]["rows"] == len(
        _complete_json_rows(date(2024, 1, 1), date(2024, 2, 1))
    )
    assert cached["symbols"]["EURUSD"]["status"] == "cached"
    assert calls == 2


def test_dukascopy_node_downloader_dry_run_and_bounded_concurrency(tmp_path):
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def runner(command, output_dir):
        nonlocal active, maximum_active
        symbol = command[command.index("-i") + 1]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        (output_dir / f"{symbol}-m1-bid-2024-01-01-2024-01-02.json").write_text(
            json.dumps(_complete_json_rows(date(2024, 1, 1), date(2024, 1, 2))),
            encoding="utf-8",
        )
        with lock:
            active -= 1

    dry = DukascopyNodeDownloader(tmp_path / "dry", runner=runner).download(
        ("EURUSD",), date(2024, 1, 1), date(2024, 1, 1), dry_run=True
    )
    assert dry["symbols"]["EURUSD"]["status"] == "planned"
    assert not (tmp_path / "dry" / "manifests" / "dukascopy_manifest.json").exists()

    concurrent = DukascopyNodeDownloader(
        tmp_path / "concurrent", workers=2, runner=runner
    )
    result = concurrent.download(
        ("EURUSD", "GBPUSD"), date(2024, 1, 1), date(2024, 1, 1)
    )
    assert result["symbols"]["EURUSD"]["status"] == "downloaded"
    assert result["symbols"]["GBPUSD"]["status"] == "downloaded"
    assert maximum_active == 2


def test_dukascopy_node_downloader_chunks_ranges_by_calendar_month(tmp_path):
    def runner(command, output_dir):
        symbol = command[command.index("-i") + 1]
        start_date = command[command.index("-from") + 1]
        end_date = command[command.index("-to") + 1]
        name = f"{symbol}-m1-bid-{start_date}-{end_date}.json"
        (output_dir / name).write_text(
            json.dumps(
                _complete_json_rows(
                    date.fromisoformat(start_date), date.fromisoformat(end_date)
                )
            ),
            encoding="utf-8",
        )

    result = DukascopyNodeDownloader(tmp_path, runner=runner).download(
        ("EURUSD",), date(2024, 1, 31), date(2024, 2, 1)
    )

    chunks = result["symbols"]["EURUSD"]["chunks"]
    assert [chunk["status"] for chunk in chunks] == ["downloaded", "downloaded"]
    assert [Path(chunk["path"]).name for chunk in chunks] == [
        "eurusd-m1-bid-2024-01-31-2024-02-01.json",
        "eurusd-m1-bid-2024-02-01-2024-02-02.json",
    ]


def test_node_downloader_command_execution_and_month_rollover(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import astra.historical.dukascopy_node as node

    downloader = DukascopyNodeDownloader(tmp_path, runner=None)
    calls = []
    monkeypatch.setattr(
        node.subprocess,
        "run",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or SimpleNamespace(returncode=0)
        ),
    )
    downloader._run(["node", "--version"], tmp_path)
    assert calls[0][0][0] == ["node", "--version"]
    assert calls[0][1]["env"]["npm_config_cache"].endswith("astra-npm-cache")

    monkeypatch.setattr(
        node.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=4)
    )
    with pytest.raises(RuntimeError, match="status 4"):
        downloader._run(["node"], tmp_path)

    chunks = tuple(node._month_ranges(date(2024, 12, 31), date(2025, 1, 1)))
    assert chunks == (
        (date(2024, 12, 31), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 1)),
    )


def test_node_downloader_rejects_bad_request_and_missing_cli_output(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        DukascopyNodeDownloader(tmp_path, workers=3)
    with pytest.raises(ValueError, match="timeout"):
        DukascopyNodeDownloader(tmp_path, timeout=float("inf"))
    downloader = DukascopyNodeDownloader(tmp_path, runner=lambda *_: None)
    with pytest.raises(ValueError, match="end date precedes"):
        downloader.download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 1))
    with pytest.raises(ValueError, match="nonempty and unique"):
        downloader.download((), date(2024, 1, 1), date(2024, 1, 1))
    with pytest.raises(ValueError, match="nonempty and unique"):
        downloader.download(("EURUSD", "EURUSD"), date(2024, 1, 1), date(2024, 1, 1))
    with pytest.raises(ValueError, match="six uppercase FX letters"):
        downloader.download(("eurusd",), date(2024, 1, 1), date(2024, 1, 1))

    result = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 1))
    assert result["symbols"]["EURUSD"]["status"] == "failed"


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        "[null]",
        '[{"timestamp":true,"open":1,"high":1,"low":1,"close":1,"volume":1}]',
        '[{"timestamp":1,"open":1,"high":1,"low":1,"close":1,"volume":1}]',
        '[{"timestamp":1704153660000,"open":1,"high":1,"low":1,"close":1,"volume":1},{"timestamp":1704153600000,"open":1,"high":1,"low":1,"close":1,"volume":1}]',
        '[{"timestamp":0,"open":1,"high":1,"low":1,"close":1,"volume":-1}]',
    ],
)
def test_node_json_export_rejects_malformed_rows(tmp_path, contents):
    source = tmp_path / "bad.json"
    source.write_text(contents, encoding="utf-8")
    assert not _valid_json_export(source, date(2024, 1, 2), date(2024, 1, 3))


def test_pipeline_records_failed_source_and_isolates_symbol_processing_errors(
    monkeypatch, tmp_path
):
    from astra.historical import pipeline

    day = date(2024, 1, 2)
    settings = HistoricalSettings(
        root=tmp_path / "failed-source",
        symbols=("EURUSD",),
        start_date=day,
        end_date=day,
        retries=0,
    )

    class InvalidProvider:
        def fetch_m1_day(self, *_):
            return b"not a BI5 file"

    failed = run_pipeline(settings, provider=InvalidProvider())
    assert failed.symbols[0].status == "FAIL"
    assert (
        "one or more required daily source units are unavailable or invalid"
        in failed.symbols[0].errors[0]
    )

    class ValidProvider:
        def fetch_m1_day(self, *_):
            return _payload(
                [
                    (minute * 60, 100000, 100010, 99990, 100020, 1.0)
                    for minute in range(1440)
                ]
            )

    failing_settings = HistoricalSettings(
        root=tmp_path / "normalizer-error",
        symbols=("EURUSD",),
        start_date=day,
        end_date=day,
        retries=0,
    )

    def fail_normalize(*_args, **_kwargs):
        raise RuntimeError("fixture normalization failure")

    monkeypatch.setattr(pipeline, "normalize_symbol", fail_normalize)
    isolated = run_pipeline(failing_settings, provider=ValidProvider())
    assert isolated.symbols[0].status == "FAIL"
    assert isolated.symbols[0].errors == ["RuntimeError"]


def test_pipeline_date_discovery_and_mixed_writer_edges(tmp_path):
    from astra.historical.pipeline import _write_candles, discover_latest
    from astra.models import Candle

    class MissingProvider:
        def fetch_m1_day(self, *_):
            return None

    with pytest.raises(RuntimeError, match="could not discover"):
        discover_latest(
            MissingProvider(), lookback_days=1, now=datetime(2024, 1, 3, tzinfo=UTC)
        )
    with pytest.raises(ValueError, match="limited to 1–14"):
        discover_latest(MissingProvider(), lookback_days=15)
    with pytest.raises(ValueError, match="mixed candle volume types"):
        _write_candles(
            tmp_path / "mixed.csv",
            (
                Candle(datetime(2024, 1, 2, 0, 1, tzinfo=UTC), 1, 1, 1, 1, 1, "tick"),
                Candle(
                    datetime(2024, 1, 2, 0, 2, tzinfo=UTC), 1, 1, 1, 1, 1, "provider"
                ),
            ),
        )
    with pytest.raises(ValueError, match="unsupported price basis"):
        _write_candles(tmp_path / "invalid-basis.csv", (), price_basis="unknown")


@pytest.mark.parametrize("status", ["FAIL", "WARN"])
def test_pipeline_propagates_timeframe_quality_status(monkeypatch, tmp_path, status):
    from types import SimpleNamespace

    from astra.historical import pipeline

    day = date(2024, 1, 2)
    settings = HistoricalSettings(
        root=tmp_path / status.lower(),
        symbols=("EURUSD",),
        start_date=day,
        end_date=day,
        retries=0,
    )

    class Provider:
        def fetch_m1_day(self, *_):
            return _payload(
                [
                    (minute * 60, 100000, 100010, 99990, 100020, 1.0)
                    for minute in range(1440)
                ]
            )

    monkeypatch.setattr(
        pipeline,
        "inspect_csv",
        lambda *_args, **_kwargs: SimpleNamespace(
            status=status, rows=1, checksum="fixture", expected_gap_count=0
        ),
    )
    result = run_pipeline(settings, provider=Provider())
    assert result.symbols[0].status == status
    if status == "WARN":
        assert "DATA QUALITY INSUFFICIENT" in result.symbols[0].errors


def test_pipeline_rejects_explicit_reversed_range(tmp_path):
    start, end = date(2024, 1, 2), date(2024, 1, 1)
    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
    )
    with pytest.raises(ValueError, match="invalid historical date range"):
        run_pipeline(settings, start=start, end=end)


def test_dukascopy_json_export_rejects_open_session_gaps_but_allows_weekend(tmp_path):
    def row(timestamp):
        return {
            "timestamp": int(timestamp.timestamp() * 1000),
            "open": 1.1,
            "high": 1.1,
            "low": 1.1,
            "close": 1.1,
            "volume": 0,
        }

    open_gap = tmp_path / "open-gap.json"
    open_gap.write_text(
        json.dumps(
            [
                row(datetime(2024, 1, 2, 12, 0, tzinfo=UTC)),
                row(datetime(2024, 1, 2, 12, 2, tzinfo=UTC)),
            ]
        ),
        encoding="utf-8",
    )
    assert not _valid_json_export(open_gap, date(2024, 1, 1), date(2024, 2, 1))

    weekend = tmp_path / "weekend-gap.json"
    weekend.write_text(
        json.dumps(
            sorted(
                _complete_json_rows(date(2024, 1, 5), date(2024, 1, 8))
                + [
                    row(datetime(2024, 1, 6, 12, tzinfo=UTC)),
                    row(datetime(2024, 1, 7, 12, tzinfo=UTC)),
                ],
                key=lambda item: item["timestamp"],
            )
        ),
        encoding="utf-8",
    )
    assert _valid_json_export(weekend, date(2024, 1, 5), date(2024, 1, 8))
