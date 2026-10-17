"""Offline tests for the continuous-tail acquisition runbook; synthetic prices only."""

import json
import lzma
import struct
from datetime import UTC, date, datetime, timedelta

import pytest

from astra.historical.config import HistoricalSettings
from astra.historical.downloader import HistoricalDownloader
from astra.historical.pipeline import run_pipeline
from astra.historical.provider import TransientSourceError
from astra.historical.validation import _market_open
from scripts.download_continuous_tail import ACTIVE_SYMBOLS, main, run


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
        return _day_payload((day - date(2024, 1, 1)).days)


def test_run_acquires_validated_timeframes_for_every_symbol(tmp_path):
    report = run(
        date(2024, 1, 1),
        end=date(2024, 1, 2),
        root=tmp_path / "tail",
        symbols=("EURUSD",),
        provider=_SyntheticProvider(),
    )
    assert report["start"] == "2024-01-01"
    assert report["end"] == "2024-01-02"
    eur = report["symbols"]["EURUSD"]
    assert eur["status"] == "PASS"
    assert set(eur["timeframes"]) == {"M1", "M5", "M15", "H1", "H4"}
    assert (tmp_path / "tail" / "acquisition-summary.json").is_file()


def test_run_isolates_per_symbol_failures(tmp_path):
    class _FailingDay:
        def fetch_m1_day(self, symbol, day, timeout):
            return None  # every day reports as missing

    report = run(
        date(2024, 1, 1),
        end=date(2024, 1, 2),
        root=tmp_path / "tail",
        symbols=("EURUSD",),
        provider=_FailingDay(),
    )
    assert report["symbols"]["EURUSD"]["status"] != "PASS"
    assert report["symbols"]["EURUSD"]["errors"]


def test_main_reports_status_and_exit_code(tmp_path, capsys, monkeypatch):
    import scripts.download_continuous_tail as runbook

    monkeypatch.setattr(runbook, "DukascopyProvider", lambda: _SyntheticProvider())
    code = main(
        [
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--root",
            str(tmp_path / "tail"),
            "--symbols",
            "EURUSD",
        ]
    )
    assert code == 1  # A one-pair request cannot certify the required four-pair tail.
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "INCOMPLETE"
    assert payload["symbols"]["EURUSD"]["status"] == "PASS"
    assert payload["symbols"]["EURUSD"]["downloads"]["downloaded"] == 2
    assert "[TAIL] 1/2 EURUSD 2024-01-01 downloaded" in captured.err


def test_main_reports_incomplete_downloads_and_error_counts(
    tmp_path, capsys, monkeypatch
):
    import scripts.download_continuous_tail as runbook
    from astra.historical.provider import TransientSourceError

    class _UnavailableProvider:
        def fetch_m1_day(self, *_):
            raise TransientSourceError("Dukascopy returned HTTP 503", status_code=503)

    monkeypatch.setattr(runbook, "DukascopyProvider", lambda: _UnavailableProvider())
    code = main(
        [
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-01",
            "--root",
            str(tmp_path / "tail"),
            "--symbols",
            "EURUSD",
            "--retries",
            "0",
            "--request-interval",
            "0",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["status"] == "INCOMPLETE"
    assert payload["symbols"]["EURUSD"]["downloads"]["failed"] == 1
    assert payload["symbols"]["EURUSD"]["downloads"]["failed_errors"] == {"HTTP 503": 1}


def test_main_handles_keyboard_interrupt_without_traceback(
    capsys, monkeypatch, tmp_path
):
    import scripts.download_continuous_tail as runbook

    def interrupted_run(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runbook, "run", interrupted_run)
    # Exercise the default relative output path in an isolated workspace.
    # The interrupt handler writes a receipt even when run() never starts.
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "data" / "astra-continuous-2026-tail"
    output.mkdir(parents=True)
    code = main(["--start", "2024-01-01", "--end", "2024-01-02"])

    captured = capsys.readouterr()
    assert code == 130
    assert "Interrupted" in captured.err
    assert "Traceback" not in captured.err
    receipt = json.loads((output / "acquisition-summary.json").read_text())
    assert receipt["status"] == "INTERRUPTED"
    assert receipt["start"] == "2024-01-01"


def test_interrupt_does_not_replace_a_previous_acquisition_run(tmp_path, monkeypatch):
    import scripts.download_continuous_tail as runbook

    output = tmp_path / "tail"
    (output / "manifests").mkdir(parents=True)
    reports = output / "reports" / "acquisition-runs"
    reports.mkdir(parents=True)
    previous = {
        "run_id": "previous-run",
        "status": "COMPLETE",
        "synthetic_fixture": True,
    }
    previous_path = reports / "previous-run.json"
    previous_path.write_text(json.dumps(previous), encoding="utf-8")
    before = previous_path.read_bytes()
    (output / "manifests" / "dukascopy_manifest.json").write_text(
        json.dumps({"latest_run_id": "previous-run", "downloads": {}}),
        encoding="utf-8",
    )

    def interrupted_run(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runbook, "run", interrupted_run)
    assert main(["--start", "2024-01-01", "--root", str(output)]) == 130
    assert previous_path.read_bytes() == before
    receipt = json.loads((output / "acquisition-summary.json").read_text())
    assert receipt["run_id"] != "previous-run"
    assert receipt["interruption_receipt"] is True
    assert (reports / (receipt["run_id"] + ".json")).is_file()


def test_main_fails_closed_on_an_invalid_range(capsys):
    code = main(["--start", "2024-01-02", "--end", "2024-01-01", "--root", "unused"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_active_symbols_match_the_curated_universe():
    assert ACTIVE_SYMBOLS == ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")


def _session_payload(day: date, *, corrupt_closed: bool = False) -> bytes:
    """Synthetic BI5 with the source's 1440-row closed-session flat padding."""
    opening = datetime.combine(day, datetime.min.time(), UTC)
    records = []
    for minute in range(1440):
        tradeable = _market_open(opening + timedelta(minutes=minute))
        price = 110000 + minute if tradeable else 110000
        volume = 1.0 if tradeable else 0.0
        if corrupt_closed and not tradeable:
            volume = 1.0
        records.append(
            struct.pack(">IIIIIf", minute * 60, price, price, price, price, volume)
        )
    return lzma.compress(b"".join(records))


def test_wholly_closed_weekend_day_never_calls_the_feed(tmp_path):
    class UnavailableProvider:
        def fetch_m1_day(self, *args):
            pytest.fail("the public feed must not be contacted on a closed day")

    saturday = date(2026, 9, 5)
    summary = HistoricalDownloader(UnavailableProvider(), root=tmp_path).download(
        ("EURUSD",), saturday, saturday
    )
    assert summary.successful and summary.closed == 1
    assert summary.results[0].status == "closed"
    assert not tuple(tmp_path.rglob("*.bi5"))


def test_weekend_padding_is_excluded_but_verified_raw_is_preserved(tmp_path):
    class Provider:
        def __init__(self):
            self.calls = []

        def fetch_m1_day(self, symbol, day, timeout):
            self.calls.append(day)
            return _session_payload(day)

    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2026, 9, 4),
        end_date=date(2026, 9, 7),
        workers=1,
    )
    provider = Provider()
    report = run_pipeline(settings, provider=provider, request_interval=0)
    assert report.successful, report.symbols[0].errors
    assert date(2026, 9, 5) not in provider.calls
    from astra.historical.dataset import load_dataset
    from astra.historical.downloader import _file
    from astra.historical.manifest import ManifestStore

    assert ManifestStore(tmp_path).unit("EURUSD", "2026-09-05")["status"] == "closed"
    assert not _file(tmp_path, "EURUSD", date(2026, 9, 5)).exists()
    assert _file(tmp_path, "EURUSD", date(2026, 9, 6)).is_file()
    data = load_dataset(tmp_path, "EURUSD", "M1")
    assert len(data.candles) == 1260 + 180 + 1440  # Fri open, Sun evening, Mon
    assert all(_market_open(c.timestamp - timedelta(minutes=1)) for c in data.candles)


def test_nonflat_closed_session_padding_fails_closed(tmp_path):
    class Provider:
        def fetch_m1_day(self, symbol, day, timeout):
            return _session_payload(day, corrupt_closed=True)

    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2026, 9, 4),
        end_date=date(2026, 9, 4),
        workers=1,
    )
    report = run_pipeline(settings, provider=Provider(), request_interval=0)
    assert not report.successful
    assert report.symbols[0].status == "FAIL"
    assert "ValueError" in report.symbols[0].errors


def test_weekday_source_failures_remain_unapproved(tmp_path):
    class Provider:
        def fetch_m1_day(self, symbol, day, timeout):
            raise TransientSourceError("unavailable", status_code=503)

    settings = HistoricalSettings(
        root=tmp_path,
        symbols=("EURUSD",),
        start_date=date(2026, 9, 7),
        end_date=date(2026, 9, 7),
        workers=1,
        retries=0,
    )
    report = run_pipeline(settings, provider=Provider(), request_interval=0)
    assert not report.successful
    assert report.symbols[0].timeframes == {}
    assert report.symbols[0].downloads.failed == 1
