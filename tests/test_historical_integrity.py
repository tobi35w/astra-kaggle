"""Offline regression fixtures for historical integrity; all prices are synthetic."""

import io
import json
import lzma
import struct
from datetime import UTC, date, datetime, timedelta
from urllib.error import HTTPError

import pytest

from astra.data import DataValidationError
from astra.historical.cli import historical_main
from astra.historical.config import HistoricalSettings
from astra.historical.downloader import HistoricalDownloader
from astra.historical.dukascopy import DukascopyProvider, decode_m1_bi5
from astra.historical.dukascopy_node import _valid_json_export
from astra.historical.importer import import_csv
from astra.historical.manifest import ManifestStore
from astra.historical.pipeline import run_pipeline
from astra.historical.provider import PermanentSourceError, TransientSourceError
from astra.historical.validation import inspect_csv
from astra.history import load_csv
from astra.repositories import Entity, Journal, MemoryRepository


def payload(seconds):
    return lzma.compress(
        b"".join(
            struct.pack(">IIIIIf", s, 110000, 110010, 109990, 110020, 1.0)
            for s in seconds
        )
    )


def csv_file(tmp_path, minutes=(1, 2), timeframe="M1"):
    path = tmp_path / f"{timeframe}.csv"
    start = datetime(2024, 1, 2, tzinfo=UTC)
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "".join(
            f"{(start + timedelta(minutes=m)).isoformat()},1.1,1.2,1,1.15,1\n"
            for m in minutes
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("seconds", [(0, 0), (60, 0), (1,), (86400,)])
def test_bi5_rejects_duplicate_reversed_or_misaligned_source(seconds):
    with pytest.raises(DataValidationError):
        decode_m1_bi5(payload(seconds), "EURUSD", date(2024, 1, 2))


def test_bi5_limits_decompression_and_rejects_truncation():
    for raw in (
        payload((0,))[:-2],
        lzma.compress(b"x" * 100000),
        payload((0,)) + b"junk",
    ):
        with pytest.raises(DataValidationError):
            decode_m1_bi5(raw, "EURUSD", date(2024, 1, 2))


@pytest.mark.parametrize(
    "code,expected",
    [
        (503, TransientSourceError),
        (429, TransientSourceError),
        (403, PermanentSourceError),
    ],
)
def test_http_failure_classification_is_injected_and_sanitized(code, expected):
    def transport(request, *, timeout):
        assert timeout == 3
        raise HTTPError(
            request.full_url, code, "private detail", {}, io.BytesIO(b"private detail")
        )

    with pytest.raises(expected, match=f"HTTP {code}") as result:
        DukascopyProvider(transport=transport).fetch_m1_day(
            "EURUSD", date(2024, 1, 2), 3
        )
    assert "private detail" not in str(result.value)


def test_http_503_preserves_status_and_bounded_retry_after():
    def transport(request, *, timeout):
        raise HTTPError(
            request.full_url,
            503,
            "private detail",
            {"Retry-After": "17"},
            io.BytesIO(b"private detail"),
        )

    with pytest.raises(TransientSourceError) as result:
        DukascopyProvider(transport=transport).fetch_m1_day(
            "EURUSD", date(2024, 1, 2), 3
        )
    assert result.value.status_code == 503
    assert result.value.retry_after_seconds == 17
    assert "private detail" not in str(result.value)


def test_network_error_preserves_safe_reason_code_without_details():
    def transport(request, *, timeout):
        raise TimeoutError("private network detail")

    with pytest.raises(TransientSourceError) as result:
        DukascopyProvider(transport=transport).fetch_m1_day(
            "EURUSD", date(2024, 1, 2), 3
        )

    assert result.value.reason_code == "TimeoutError"
    assert "private network detail" not in str(result.value)


def test_downloader_reports_safe_network_reason_code(tmp_path):
    class TimeoutProvider:
        def fetch_m1_day(self, *_):
            raise TransientSourceError(
                "Dukascopy request failed", reason_code="TimeoutError"
            )

    summary = HistoricalDownloader(
        TimeoutProvider(),
        root=tmp_path,
        workers=1,
        retries=0,
        request_interval=0,
        sleep=lambda _: None,
    ).download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 2))

    assert summary.results[0].error == "TimeoutError"
    assert (
        ManifestStore(tmp_path).unit("EURUSD", "2024-01-02")["error"] == "TimeoutError"
    )


def test_downloader_records_http_status_and_respects_retry_after(tmp_path):
    delays = []

    class UnavailableProvider:
        def fetch_m1_day(self, *_):
            raise TransientSourceError(
                "Dukascopy returned HTTP 503",
                status_code=503,
                retry_after_seconds=30,
            )

    summary = HistoricalDownloader(
        UnavailableProvider(),
        root=tmp_path,
        workers=1,
        retries=1,
        request_interval=0,
        sleep=delays.append,
    ).download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 2))

    assert summary.failed == 1
    assert len(delays) == 1
    assert 30 <= delays[0] <= 30.5  # Retry-After is a floor; bounded jitter follows it.
    saved = ManifestStore(tmp_path).unit("EURUSD", "2024-01-02")
    assert saved["status"] == "failed"
    assert saved["error"] == "HTTP 503"


def test_downloader_stops_requesting_after_http_429_and_preserves_resume_units(
    tmp_path,
):
    class RateLimitedProvider:
        def __init__(self):
            self.calls = []

        def fetch_m1_day(self, symbol, day, *_):
            self.calls.append((symbol, day))
            raise TransientSourceError(
                "Dukascopy returned HTTP 429",
                status_code=429,
                retry_after_seconds=60,
            )

    provider = RateLimitedProvider()
    summary = HistoricalDownloader(
        provider,
        root=tmp_path,
        workers=1,
        retries=4,
        request_interval=0,
        sleep=lambda _: None,
    ).download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 4))

    assert provider.calls == [("EURUSD", date(2024, 1, 2))]
    assert summary.failed == 1
    assert summary.deferred == 2
    manifest = ManifestStore(tmp_path)
    assert manifest.unit("EURUSD", "2024-01-02")["error"] == "HTTP 429"
    assert manifest.unit("EURUSD", "2024-01-03")["status"] == "deferred"
    assert manifest.unit("EURUSD", "2024-01-04")["status"] == "deferred"


def test_single_worker_downloader_propagates_interrupt_without_thread_pool(
    tmp_path, monkeypatch
):
    import astra.historical.downloader as downloader_module

    class InterruptingProvider:
        def fetch_m1_day(self, *_):
            raise KeyboardInterrupt

    def forbidden_thread_pool(*args, **kwargs):
        pytest.fail("single-worker acquisition must not create a thread pool")

    monkeypatch.setattr(downloader_module, "ThreadPoolExecutor", forbidden_thread_pool)
    downloader = HistoricalDownloader(
        InterruptingProvider(), root=tmp_path, workers=1, request_interval=0
    )

    with pytest.raises(KeyboardInterrupt):
        downloader.download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 2))


def test_missing_minute_cannot_be_hidden_by_large_gap_threshold(tmp_path):
    report = inspect_csv(
        csv_file(tmp_path, (1, 3)), "EURUSD", "M1", max_gap_minutes=1440
    )
    assert report.status == "WARN"
    assert report.suspicious_gaps[0].missing_minutes == 1


def test_gap_units_and_missing_range_edges(tmp_path):
    report = inspect_csv(csv_file(tmp_path, (5, 15), "M5"), "EURUSD", "M5")
    assert report.suspicious_gaps[0].missing_minutes == 5
    report = inspect_csv(
        csv_file(tmp_path, (2, 3)),
        "EURUSD",
        "M1",
        expected_start=datetime(2024, 1, 2, tzinfo=UTC),
        expected_end=datetime(2024, 1, 2, 0, 5, tzinfo=UTC),
    )
    assert report.status == "WARN"
    assert sum(g.missing_minutes for g in report.suspicious_gaps) == 3


def test_invalid_late_row_prevents_all_import_writes(tmp_path):
    path = csv_file(tmp_path)
    path.write_text(path.read_text() + "2024-01-02T00:03:00+00:00,1,0.5,2,1,1\n")
    repo = MemoryRepository()
    with pytest.raises(DataValidationError):
        import_csv(path, "EURUSD", "M1", Journal(repo), batch_size=1)
    assert not repo.query(Entity.OBSERVATION)


def test_changed_dataset_does_not_silently_reuse_old_import_ids(tmp_path):
    path = csv_file(tmp_path)
    repo = MemoryRepository()
    import_csv(path, "EURUSD", "M1", Journal(repo))
    path.write_text(path.read_text().replace(",1.15,", ",1.16,"))
    import_csv(path, "EURUSD", "M1", Journal(repo))
    rows = repo.query(Entity.OBSERVATION)
    assert len(rows) == 4
    assert len({row.payload["dataset_version"] for row in rows}) == 2


def test_corrupt_manifest_is_preserved_and_reported(tmp_path):
    path = tmp_path / "manifests" / "dukascopy_manifest.json"
    path.parent.mkdir()
    path.write_text("{broken")
    with pytest.raises(ValueError, match="manifest"):
        ManifestStore(tmp_path)
    assert path.read_text() == "{broken"


def test_manifest_keeps_immutable_run_scopes_across_narrow_retries(tmp_path):
    store = ManifestStore(tmp_path)
    broad_id = store.begin_run(
        symbols=("AUDUSD", "EURUSD", "GBPUSD", "USDJPY"),
        start_date=date(2026, 9, 1),
        end_date=date(2026, 10, 12),
        source="Dukascopy",
    )
    store.finish_run(broad_id, status="INCOMPLETE", summary={"failed": 83})
    retry_id = store.begin_run(
        symbols=("EURUSD",),
        start_date=date(2026, 10, 12),
        end_date=date(2026, 10, 12),
        source="Dukascopy",
    )
    store.finish_run(retry_id, status="INCOMPLETE", summary={"failed": 1})

    saved = ManifestStore(tmp_path).data
    assert saved["runs"][broad_id]["scope"]["start_date"] == "2026-09-01"
    assert saved["runs"][broad_id]["scope"]["end_date"] == "2026-10-12"
    assert saved["runs"][retry_id]["scope"]["symbols"] == ["EURUSD"]


def test_reconciliation_fails_closed_when_manifest_has_no_verified_raw_file(tmp_path):
    from astra.historical.reconciliation import reconcile_acquisition

    report = reconcile_acquisition(
        tmp_path, ("EURUSD",), date(2026, 9, 7), date(2026, 9, 7)
    )
    assert report["status"] == "FAIL"
    assert report["expected_session_units"] == 1
    assert report["verified_session_units"] == 0
    assert report["missing_sessions"][0]["unit"] == "EURUSD:2026-09-07"


def test_dry_run_does_not_discover_dates_or_create_files(tmp_path):
    class NoNetwork:
        def fetch_m1_day(self, *args):
            pytest.fail("dry-run attempted network access")

    root = tmp_path / "untouched"
    settings = HistoricalSettings(
        root=root, symbols=("EURUSD",), start_date=date(2024, 1, 2)
    )
    with pytest.raises(ValueError, match="--end"):
        run_pipeline(settings, dry_run=True, provider=NoNetwork())
    result = run_pipeline(
        settings, end=date(2024, 1, 2), dry_run=True, provider=NoNetwork()
    )
    assert result.symbols[0].status == "PLANNED"
    assert not root.exists()


def test_cli_returns_failure_for_bad_quality_and_never_imports(tmp_path, capsys):
    path = csv_file(tmp_path, (1, 3))
    assert historical_main(["validate", "EURUSD", "M1", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "WARN"
    assert historical_main(["download", "--start", "2024-01-02", "--dry-run"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_partial_json_range_is_not_a_complete_download(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            [
                {
                    "timestamp": 1704153600000,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 0,
                }
            ]
        )
    )
    assert not _valid_json_export(path, date(2024, 1, 2), date(2024, 1, 3))


def test_valid_but_changed_cache_is_refetched(tmp_path):
    class Provider:
        calls = 0

        def fetch_m1_day(self, *args):
            self.calls += 1
            return payload((0, 60))

    provider = Provider()
    downloader = HistoricalDownloader(provider, root=tmp_path, retries=0)
    day = date(2024, 1, 2)
    first = downloader.download(("EURUSD",), day, day)
    from pathlib import Path

    Path(first.results[0].path).write_bytes(payload((0,)))
    assert downloader.download(("EURUSD",), day, day).downloaded == 1
    assert provider.calls == 2


def test_managed_loader_carries_provenance_and_rejects_modified_files(tmp_path):
    from astra.historical.dataset import load_dataset

    class Provider:
        def fetch_m1_day(self, *args):
            return payload(range(0, 86400, 60))

    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
    )
    assert run_pipeline(settings, provider=Provider()).successful
    data = load_dataset(tmp_path, "EURUSD", "M5")
    assert data.metadata["price_basis"] == "bid"
    assert data.metadata["quality_status"] == "PASS"
    assert data.metadata["dataset_version"].startswith("astra-sha256:")
    assert len(data.candles) == 288
    report_path = tmp_path / "reports" / "latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["symbols"][0]["status"] = "FAIL"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataValidationError, match="successful acquisition report"):
        load_dataset(tmp_path, "EURUSD", "M5")
    report["symbols"][0]["status"] = "PASS"
    report["start_date"] = "2024-01-01"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DataValidationError, match="requested range is incomplete"):
        load_dataset(tmp_path, "EURUSD", "M5")
    report["start_date"] = "2024-01-02"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    from astra.__main__ import _dispatch, _parser
    from astra.config import Settings

    args = _parser().parse_args(
        [
            "analyze",
            "--dataset-root",
            str(tmp_path),
            "--pair",
            "EURUSD",
            "--timeframe",
            "M5",
        ]
    )
    decision = _dispatch(args, Settings(), Journal(MemoryRepository()))
    assert (
        decision.context["dataset"]["dataset_version"]
        == data.metadata["dataset_version"]
    )
    path = tmp_path / "processed" / "m5" / "EURUSD.csv"
    path.write_text(path.read_text().replace("1.1001", "1.10015"))
    with pytest.raises(DataValidationError, match="changed after validation"):
        load_dataset(tmp_path, "EURUSD", "M5")


def test_normalizer_rejects_manifest_checksum_mismatch(tmp_path):
    from astra.historical.normalizer import normalize_symbol

    class Provider:
        def fetch_m1_day(self, *_):
            return payload((0,))

    day = date(2024, 1, 2)
    downloader = HistoricalDownloader(Provider(), root=tmp_path, retries=0)
    assert downloader.download(("EURUSD",), day, day).downloaded == 1
    manifest = ManifestStore(tmp_path)
    manifest.update_unit(
        "EURUSD", day.isoformat(), {"status": "downloaded", "checksum": "wrong"}
    )
    with pytest.raises(ValueError, match="does not match"):
        normalize_symbol(tmp_path, "EURUSD", day, day)


def test_normalizer_rejects_conflicting_duplicate_timestamps(monkeypatch, tmp_path):
    from astra.historical.normalizer import normalize_symbol
    from astra.models import Candle

    class Provider:
        def fetch_m1_day(self, *_):
            return payload((0,))

    start, end = date(2024, 1, 2), date(2024, 1, 3)
    HistoricalDownloader(Provider(), root=tmp_path, retries=0).download(
        ("EURUSD",), start, end
    )
    calls = 0

    def duplicate_timestamp(_raw, _symbol, _day):
        nonlocal calls
        calls += 1
        return (
            Candle(
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
                1,
                1,
                1,
                1 if calls == 1 else 1.1,
                1,
            ),
        )

    monkeypatch.setattr(
        "astra.historical.normalizer.decode_m1_bi5", duplicate_timestamp
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        normalize_symbol(tmp_path, "EURUSD", start, end)


def test_quality_report_preserves_failure_counts_and_rejects_bad_bounds(tmp_path):
    from astra.historical.validation import inspect_candles
    from astra.models import Candle

    path = tmp_path / "bad.csv"
    path.write_bytes(b"\xff\xfe")
    unreadable = inspect_csv(path, "EURUSD", "M1")
    assert unreadable.status == "FAIL" and unreadable.invalid_count == 1

    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "bad-time,1,1,1,1,1\n"
        "2024-01-02T00:01:00,1,1,1,1,1\n"
        "2024-01-02T00:01:30+00:00,1,1,1,1,1\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1,1\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )
    counted = inspect_csv(path, "EURUSD", "M1")
    assert counted.invalid_count == 3 and counted.out_of_order_count == 1

    candle = Candle(datetime(2024, 1, 2, 0, 1, tzinfo=UTC), 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="quality-check configuration"):
        inspect_candles((candle,), "EURUSD", "M1", max_gap_minutes=-1)
    with pytest.raises(ValueError, match="bounds must be timezone-aware"):
        inspect_candles(
            (candle,),
            "EURUSD",
            "M1",
            expected_start=datetime(2024, 1, 2),  # noqa: DTZ001
        )
    with pytest.raises(ValueError, match="range must be nonempty"):
        inspect_candles(
            (candle,),
            "EURUSD",
            "M1",
            expected_start=datetime(2024, 1, 2, tzinfo=UTC),
            expected_end=datetime(2024, 1, 2, tzinfo=UTC),
        )
    misaligned = Candle(datetime(2024, 1, 2, 0, 1, 30, tzinfo=UTC), 1, 1, 1, 1, 1)
    assert inspect_candles((misaligned,), "EURUSD", "M1").status == "FAIL"
    out_of_range = inspect_candles(
        (candle,),
        "EURUSD",
        "M1",
        expected_start=datetime(2024, 1, 2, tzinfo=UTC),
        expected_end=datetime(2024, 1, 2, 0, 0, 30, tzinfo=UTC),
    )
    assert any(
        "outside the requested range" in warning for warning in out_of_range.warnings
    )
    weekend = Candle(datetime(2024, 1, 6, 0, 1, tzinfo=UTC), 1, 1, 1, 1, 1)
    assert any(
        "weekend closure" in warning
        for warning in inspect_candles((weekend,), "EURUSD", "M1").warnings
    )


def test_csv_loader_rejects_missing_or_mixed_volume_metadata(tmp_path):
    path = tmp_path / "fixture.csv"
    with pytest.raises(ValueError, match="unsupported timeframe"):
        load_csv(path, "EURUSD", "BAD")

    path.write_text(
        "timestamp,open,high,low,close\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="volume column"):
        load_csv(path, "EURUSD", "M1")

    path.write_text(
        "timestamp,open,high,low,close,volume,volume_unit\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1,contracts\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1,1,lots\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="mixes volume units"):
        load_csv(path, "EURUSD", "M1")

    path.write_text(
        "timestamp,open,high,low,close,volume,price_basis\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1,bid\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1,1,mid\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="mixes price bases"):
        load_csv(path, "EURUSD", "M1")


def test_csv_loader_preserves_price_basis(tmp_path):
    path = tmp_path / "bid.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume,price_basis\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1,bid\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1,1,bid\n",
        encoding="utf-8",
    )
    assert load_csv(path, "EURUSD", "M1").metadata["price_basis"] == "bid"

    path.write_text(
        "timestamp,open,high,low,close,volume,price_basis\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1,unknown\n"
        "2024-01-02T00:02:00+00:00,1,1,1,1,1,unknown\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="unsupported price basis"):
        load_csv(path, "EURUSD", "M1")

    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1\n"
        "2024-01-02T00:30:00+00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="missing candles"):
        load_csv(path, "EURUSD", "M1", reject_gaps=True)


def test_data_quality_veto_applies_before_strategy_selection(market_data):
    from dataclasses import replace

    from astra.config import Settings
    from astra.decision import decide
    from astra.models import Action, RiskState
    from astra.risk import RiskEngine
    from astra.strategies import StrategyRegistry

    data = market_data()
    data = replace(data, metadata={"quality_status": "WARN"})
    decision = decide(
        data,
        list(StrategyRegistry().all()),
        RiskEngine(Settings(trading_enabled=True)),
        RiskState(10000, 10000, 10000),
        decision_time=data.candles[-1].timestamp,
    )
    assert decision.action == Action.NO_TRADE
    assert "DATA QUALITY INSUFFICIENT" in decision.reason


def test_latest_discovery_rejects_incomplete_source_day():
    from astra.historical.pipeline import discover_latest

    class Provider:
        def fetch_m1_day(self, symbol, day, timeout):
            return payload((0,)) if day.day == 3 else payload(range(0, 86400, 60))

    result = discover_latest(
        Provider(), now=datetime(2024, 1, 4, tzinfo=UTC), lookback_days=2
    )
    assert result == date(2024, 1, 2)


def test_quality_report_counts_conflicting_duplicates(tmp_path):
    path = csv_file(tmp_path)
    path.write_text(path.read_text() + "2024-01-02T00:02:00+00:00,1.1,1.2,1,1.16,1\n")
    report = inspect_csv(path, "EURUSD", "M1")
    assert report.status == "FAIL"
    assert report.rows == 3
    assert report.duplicate_count == report.conflicting_duplicate_count == 1


def test_manifest_versions_change_with_data_and_retain_prior_summary(tmp_path):
    manifest = ManifestStore(tmp_path)
    args = {
        "symbols": ("EURUSD",),
        "start_date": date(2024, 1, 2),
        "end_date": date(2024, 1, 2),
    }
    manifest.update_dataset(**args, results={"EURUSD": {"checksum": "a"}})
    first = manifest.data["dataset_version"]
    manifest.update_dataset(**args, results={"EURUSD": {"checksum": "b"}})
    assert first != manifest.data["dataset_version"]
    assert len(manifest.data["versions"]) == 2


@pytest.mark.parametrize("command,exit_code", [("validate", 1), ("backtest", 2)])
def test_main_validator_rejects_an_open_session_gap(
    tmp_path, monkeypatch, capsys, command, exit_code
):
    import sys

    from astra.__main__ import main

    path = csv_file(tmp_path, (1, 3))
    monkeypatch.setattr(
        sys,
        "argv",
        ["astra", command, str(path), "--pair", "EURUSD", "--timeframe", "M1"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == exit_code
    result = json.loads(capsys.readouterr().out)
    if command == "validate":
        assert result["valid"] is False
    assert result["status"] == "DATA QUALITY INSUFFICIENT"


def test_bid_provenance_cannot_silently_enter_a_midpoint_simulator(market_data):
    from dataclasses import replace

    from astra.backtest import BacktestCosts, run_report
    from astra.config import Settings
    from astra.pipeline import replay_paper
    from astra.strategies import EmaTrend

    data = replace(market_data(), metadata={"price_basis": "bid"})
    with pytest.raises(ValueError, match="midpoint"):
        run_report(data, EmaTrend())
    with pytest.raises(ValueError, match="midpoint"):
        replay_paper(data, [EmaTrend()], Settings(), BacktestCosts())
