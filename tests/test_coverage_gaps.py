"""Focused edge-path tests for Astra's safety boundaries and adapters."""

from __future__ import annotations

import runpy
import sys
from datetime import UTC, date, datetime
from types import ModuleType
from urllib.error import HTTPError

import pytest

from astra.adapters import PostgresRepository, SupabaseRepository
from astra.config import Settings
from astra.data import DataValidationError, InMemoryProvider, validate
from astra.database import PostgresDecisionRepository, _jsonable
from astra.database_verification import (
    failure_category,
    read_schema,
    schema_issues,
    verify_database,
)
from astra.exposure import OpenExposure, correlated_exposure, correlation, returns
from astra.historical import cli as historical_cli
from astra.historical.config import HistoricalSettings
from astra.historical.downloader import HistoricalDownloader
from astra.historical.models import DownloadSummary
from astra.history import load_csv
from astra.models import Candle, MarketData, Side
from astra.repositories import Entity, MemoryRepository


def _market(candles, **metadata):
    return MarketData("EURUSD", "M5", tuple(candles), metadata)


def test_data_validation_rejects_bad_status_and_invalid_minimum(market_data):
    data = market_data(2)
    with pytest.raises(DataValidationError, match="DATA QUALITY"):
        validate(_market(data.candles, quality_status="FAIL"))
    with pytest.raises(ValueError, match="min_candles"):
        validate(data, min_candles=0)
    with pytest.raises(KeyError):
        InMemoryProvider(data).get_market_data("GBPUSD", "M5", 2)
    assert (
        InMemoryProvider(data).get_market_data("EURUSD", "M5", 1).candles
        == data.candles[-1:]
    )


@pytest.mark.parametrize(
    "replace, message",
    [
        (
            lambda c: Candle(
                c.timestamp.replace(tzinfo=None),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
            ),
            "timezone-aware",
        ),
        (
            lambda c: Candle(
                c.timestamp, float("nan"), c.high, c.low, c.close, c.volume
            ),
            "non-finite",
        ),
        (
            lambda c: Candle(c.timestamp, 0, c.high, c.low, c.close, c.volume),
            "non-positive",
        ),
        (
            lambda c: Candle(c.timestamp, c.open, c.low - 1, c.low, c.close, c.volume),
            "impossible OHLC",
        ),
        (
            lambda c: Candle(c.timestamp, c.open, c.high, c.low, c.close, -1),
            "negative volume",
        ),
        (
            lambda c: Candle(
                c.timestamp, c.open, c.high, c.low, c.close, c.volume, "other"
            ),
            "volume_type",
        ),
    ],
)
def test_data_validation_rejects_invalid_candle_fields(market_data, replace, message):
    data = market_data(2)
    with pytest.raises(DataValidationError, match=message):
        validate(_market((replace(data.candles[0]), data.candles[1])))


def test_data_validation_requires_nonempty_pair_and_timeframe(market_data):
    candle = market_data(1).candles[0]
    with pytest.raises(DataValidationError, match="pair and timeframe"):
        validate(MarketData(" ", "M5", (candle,)))


def test_data_module_cli_delegates_historical_arguments(monkeypatch):
    calls = []
    historical_cli = ModuleType("astra.historical.cli")
    historical_cli.historical_main = lambda args: calls.append(args) or 7
    monkeypatch.setitem(sys.modules, "astra.historical.cli", historical_cli)
    monkeypatch.delitem(sys.modules, "astra.data")
    monkeypatch.setattr(sys, "argv", ["astra.data", "historical", "validate"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("astra.data", run_name="__main__")
    assert exc.value.code == 7
    assert calls == [["validate"]]


def test_exposure_helpers_cover_short_flat_and_threshold_cases():
    assert returns([0, 2, 1]) == (-0.5,)
    assert correlation((1.0,), (2.0,)) is None
    assert correlation((1.0, 1.0), (2.0, 3.0)) is None
    assert correlation((1.0, 2.0), (3.0, 4.0)) == pytest.approx(1)
    book = (OpenExposure("EURUSD", Side.LONG, 9), OpenExposure("GBPUSD", Side.SHORT, 2))
    assert correlated_exposure("EURUSD", book, {"GBPUSD": 0.7}) == 2
    assert correlated_exposure("EURUSD", book, {"GBPUSD": -0.8}, threshold=0.8) == 2
    with pytest.raises(ValueError, match="threshold"):
        correlated_exposure("EURUSD", (), {}, threshold=1.1)


def test_history_csv_rejects_invalid_schema_and_mixed_metadata(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,open\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="columns"):
        load_csv(path, "EURUSD", "M5")
    path.write_text(
        "timestamp,open,high,low,close,volume,volume_type,volume_unit\n"
        "2025-01-01T00:00:00+00:00,1.1,1.2,1.0,1.1,1,tick,lot\n"
        "2025-01-01T00:05:00+00:00,1.1,1.2,1.0,1.1,1,real,share\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="mixes volume types"):
        load_csv(path, "EURUSD", "M5")


def test_settings_validation_and_environment_fallback(monkeypatch):
    for setting in (
        Settings(max_trade_risk_pct=0),
        Settings(max_daily_loss_pct=0),
        Settings(max_drawdown_pct=2),
        Settings(minimum_confidence=2),
        Settings(minimum_stop_distance=0),
        Settings(persistence_backend="unknown"),
    ):
        with pytest.raises(ValueError):
            setting.validate()

    original_import = __import__

    def import_without_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_dotenv)
    monkeypatch.setenv("ASTRA_LOG_LEVEL", "not-a-level")
    monkeypatch.setenv("ASTRA_MAX_SPREAD", "-1")
    with pytest.raises(ValueError, match="ASTRA_MAX_SPREAD"):
        Settings.from_env()
    monkeypatch.delenv("ASTRA_MAX_SPREAD")
    assert Settings.from_env().log_level == "INFO"


def _complete_schema():
    entities = ("",) + tuple(item.value for item in Entity)
    return {
        "columns": {
            "record_id": "uuid",
            "entity": "text",
            "occurred_at": "timestamp with time zone",
            "recorded_at": "timestamp with time zone",
            "pair": "text",
            "timeframe": "text",
            "strategy": "text",
            "correlation_id": "text",
            "schema_version": "integer",
            "payload": "jsonb",
        },
        "constraints": [
            "PRIMARY KEY (record_id)",
            "CHECK (schema_version = 1)",
            "CHECK (jsonb_typeof(payload) = 'object')",
            "CHECK (timeframe IN ('','M1','M5','M15','H1','H4'))",
            "CHECK (entity IN (" + ",".join(f"'{item}'" for item in entities) + "))",
        ],
        "indexes": [
            "(entity, pair, timeframe, strategy, occurred_at, record_id)",
            "(correlation_id)",
            "(recorded_at)",
        ],
        "rls": True,
        "triggers": [
            "CREATE TRIGGER astra_journal_immutable BEFORE DELETE OR UPDATE ON public.astra_records FOR EACH ROW EXECUTE FUNCTION astra_reject_journal_mutation()"
        ],
    }


@pytest.mark.parametrize(
    "message, expected",
    [
        ("other", "OPERATION_FAILED"),
        ("timeout", "TIMEOUT"),
        ("certificate verify failed", "TLS_FAILED"),
        ("connection refused", "CONNECTION_FAILED"),
        ("HTTP 403 forbidden", "AUTHENTICATION_FAILED"),
        ("could not resolve host", "DNS_FAILED"),
    ],
)
def test_database_failure_categories(message, expected):
    assert failure_category(RuntimeError(message)) == expected


def test_read_schema_runs_all_read_only_catalog_queries(monkeypatch):
    class Cursor:
        def __init__(self):
            self.index = -1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, _query):
            self.index += 1

        def fetchall(self):
            return [
                [("record_id", "uuid")],
                [("PRIMARY KEY (record_id)",)],
                [("indexdef",)],
                [],
                [("triggerdef",)],
            ][self.index]

        def fetchone(self):
            return (True,)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self):
            return Cursor()

    monkeypatch.setattr("psycopg.connect", lambda dsn, **kw: Connection())
    result = read_schema("fake-dsn")
    assert result["columns"] == {"record_id": "uuid"}
    assert result["constraints"] == ["PRIMARY KEY (record_id)"]
    assert result["indexes"] == ["indexdef"]
    assert result["rls"] is True
    assert result["triggers"] == ["triggerdef"]


def test_schema_issues_report_every_security_and_compatibility_gap():
    invalid = {
        "columns": {},
        "constraints": [],
        "indexes": [],
        "rls": False,
        "triggers": [],
    }
    issues = schema_issues(invalid)
    assert "missing or incompatible column: record_id" in issues
    assert "schema constraint does not admit M1" in issues
    assert "journal UUID primary key missing" in issues
    assert "payload or schema-version constraint missing" in issues
    assert "required lookup index missing: correlation_id" in issues
    assert "row-level security is not enabled" in issues
    assert "append-only journal trigger missing" in issues


def test_database_verification_stops_before_schema_for_local_or_missing_dsn():
    local = verify_database(
        Settings(), schema_reader=lambda _: pytest.fail("unexpected")
    )
    assert local["reason"] == "A hosted PostgreSQL or Supabase backend is required"
    no_dsn = verify_database(
        Settings(persistence_backend="supabase"),
        schema_reader=lambda _: pytest.fail("unexpected"),
    )
    assert (
        no_dsn["reason"]
        == "SUPABASE_DB_URL is required to inspect hosted constraints and indexes"
    )


def test_database_verification_reports_schema_failure_and_journal_exception():
    settings = Settings(persistence_backend="postgres", supabase_db_url="test")
    schema_failed = verify_database(
        settings, schema_reader=lambda _: (_ for _ in ()).throw(OSError("timeout"))
    )
    assert schema_failed["schema"] == "FAILED"
    assert schema_failed["reason"] == "TIMEOUT"

    class BrokenRepository(MemoryRepository):
        def append_batch(self, records):
            raise OSError("connection refused; secret is not reported")

    result = verify_database(
        settings,
        schema_reader=lambda _: _complete_schema(),
        repository=BrokenRepository(),
    )
    assert result["schema"] == "VERIFIED"
    assert result["journal_round_trip"] == "FAILED"
    assert result["reason"] == "CONNECTION_FAILED"


def test_database_verification_closes_owned_repository(monkeypatch):
    import astra.database_verification as verification

    class OwnedRepository(MemoryRepository):
        was_closed = False

        def close(self):
            self.was_closed = True

    repo = OwnedRepository()
    monkeypatch.setattr(verification, "repository_from_settings", lambda _: repo)
    result = verify_database(
        Settings(persistence_backend="postgres", supabase_db_url="test"),
        schema_reader=lambda _: _complete_schema(),
    )
    assert result["status"] == "VERIFIED"
    assert repo.was_closed


def test_repository_adapter_construction_and_failure_boundaries(monkeypatch):
    from astra.repositories import PersistenceError

    with pytest.raises(ValueError, match="connection credential"):
        PostgresRepository("")

    original_import = __import__

    def without_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_psycopg)
    with pytest.raises(PersistenceError, match="postgres extra"):
        PostgresRepository("postgresql://fixture")


def test_supabase_adapter_http_transport_handles_empty_json_and_sanitizes_errors(
    monkeypatch,
):
    from astra.repositories import PersistenceError

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.body

    monkeypatch.setattr("astra.adapters.urlopen", lambda *a, **kw: Response(b"[]"))
    repo = SupabaseRepository("https://fixture.supabase.co", "key")
    assert repo._http("GET", "https://fixture.supabase.co/rest/v1", {}, None) == []
    monkeypatch.setattr("astra.adapters.urlopen", lambda *a, **kw: Response(b""))
    assert repo._http("GET", "https://fixture.supabase.co/rest/v1", {}, None) == []

    with pytest.raises(ValueError, match="credential is missing"):
        SupabaseRepository("https://fixture.supabase.co")

    repo._transport = lambda *a: (_ for _ in ()).throw(RuntimeError("private detail"))
    with pytest.raises(PersistenceError, match="Supabase request failed") as error:
        repo.query(Entity.AUDIT)
    assert "private detail" not in str(error.value)

    def http_failure(*_args, **_kwargs):
        raise HTTPError("https://fixture.invalid", 503, "private", {}, None)

    repo._transport = http_failure
    with pytest.raises(PersistenceError, match="HTTP 503"):
        repo.query(Entity.AUDIT)


def test_supabase_empty_batch_is_noop_and_db_schema_without_rls_is_false():
    low_privilege = SupabaseRepository(
        "https://fixture.supabase.co",
        secret_key="secret-fixture",
        use_secret=True,
        transport=lambda *a: [],
    )
    low_privilege.append_batch(())

    class Cursor:
        def __init__(self):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, _query):
            self.index += 1

        def fetchall(self):
            return [[], [], [], [], []][self.index - 1]

        def fetchone(self):
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self):
            return Cursor()

    assert read_schema("fixture", connect=lambda *a, **kw: Connection())["rls"] is False


def test_legacy_postgres_decision_adapter_and_json_conversion(monkeypatch):
    from astra.models import Action, Decision, Regime, RiskApproval, Signal

    with pytest.raises(ValueError, match="connection string"):
        PostgresDecisionRepository("")

    import psycopg

    class Cursor:
        executed = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, *args):
            self.executed = args

    class Connection:
        def __init__(self):
            self.value = Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self):
            return self.value

    connection = Connection()
    monkeypatch.setattr(psycopg, "connect", lambda _: connection)
    decision = Decision(
        datetime(2025, 1, 1, tzinfo=UTC),
        "EURUSD",
        "M5",
        Action.NO_TRADE,
        "test",
        regime=Regime.RANGING,
        signal=Signal("test", Side.LONG, 0.8, 1.1, 1.0, 1.3, "fixture"),
        risk=RiskApproval(True, (), 1),
    )
    PostgresDecisionRepository("postgresql://fixture").append(decision)
    values = connection.value.executed[1]
    assert values[3] == "NO_TRADE"
    assert values[5] == "RANGING"
    assert values[7].obj["side"] == "LONG"
    assert values[8].obj["approved"] is True
    assert _jsonable({"at": datetime(2025, 1, 1, tzinfo=UTC), "side": Side.LONG}) == {
        "at": "2025-01-01T00:00:00+00:00",
        "side": "LONG",
    }
    with pytest.raises(TypeError, match="cannot encode object"):
        _jsonable({"unsupported": object()})

    from enum import Enum

    class PlainState(Enum):
        READY = "ready"

    assert _jsonable({"state": PlainState.READY}) == {"state": "ready"}

    import builtins

    original_import = builtins.__import__

    def missing_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_psycopg)
    with pytest.raises(RuntimeError, match="install Astra"):
        PostgresDecisionRepository("postgresql://fixture").append(decision)


def test_historical_node_downloader_sanitizes_runner_failure(tmp_path):
    from datetime import date

    from astra.historical.dukascopy_node import DukascopyNodeDownloader
    from astra.historical.manifest import ManifestStore

    def failed_runner(_command, _output_dir):
        raise TimeoutError("private transport detail")

    downloader = DukascopyNodeDownloader(tmp_path, runner=failed_runner)
    result = downloader.download(("EURUSD",), date(2024, 1, 1), date(2024, 1, 1))
    item = result["symbols"]["EURUSD"]["chunks"][0]
    assert item == {"status": "failed", "error": "TimeoutError"}
    assert ManifestStore(tmp_path).unit("EURUSD", "2024-01-01..2024-01-02") == {
        "status": "failed",
        "error": "TimeoutError",
    }


def test_historical_node_command_resolution_fallbacks(monkeypatch, tmp_path):
    from astra.historical import dukascopy_node

    monkeypatch.setattr(dukascopy_node.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Node.js/npm is required"):
        dukascopy_node._npm_exec_prefix()

    monkeypatch.setattr(
        dukascopy_node.shutil,
        "which",
        lambda name: "C:/node/npx.cmd" if name == "npx" else None,
    )
    assert dukascopy_node._npm_exec_prefix() == [
        "C:/node/npx.cmd",
        "--yes",
        "dukascopy-node@1.50.0",
    ]


def test_historical_normalizer_requires_manifest_verified_raw_file(tmp_path):
    from astra.historical.normalizer import normalize_symbol

    with pytest.raises(ValueError, match="verified raw M1 file is missing"):
        normalize_symbol(tmp_path, "EURUSD", date(2024, 1, 1), date(2024, 1, 1))


def test_historical_resampler_rejects_mixed_volume_types():
    from datetime import UTC, datetime, timedelta

    from astra.historical.resampler import resample
    from astra.models import Candle

    opened = datetime(2024, 1, 1, tzinfo=UTC)
    candles = tuple(
        Candle(
            opened + timedelta(minutes=i + 1),
            1,
            1,
            1,
            1,
            1,
            "provider" if i < 2 else "tick",
        )
        for i in range(5)
    )
    with pytest.raises(ValueError, match="mixed volume types"):
        resample(candles, "M5")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        resample((), "M3")


def test_arena_skips_strategies_that_raise_or_return_no_candidate(market_data):
    from astra.arena import rank
    from astra.models import MarketSnapshot, Regime, Side

    data = market_data(40)
    context = MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        Side.LONG,
        0.5,
        0.001,
        Regime.TRENDING,
        (),
    )

    class Raises:
        def generate(self, *_):
            raise ArithmeticError("fixture failure")

    class NoSignal:
        def generate(self, *_):
            return None

    assert rank(data, context, [Raises(), NoSignal()]) == []


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [None],
        [{"timestamp": True}],
        [{"timestamp": 0}],
        [{"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
    ],
)
def test_historical_json_processor_rejects_empty_or_malformed_exports(
    tmp_path, payload
):
    import json

    from astra.historical.json_source import process_dukascopy_json

    source = tmp_path / "fixture.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        process_dukascopy_json(source, "EURUSD", tmp_path / "output")


def test_historical_json_processor_rejects_invalid_configuration(tmp_path):
    from astra.historical.json_source import process_dukascopy_json

    with pytest.raises(ValueError, match="six-letter FX pair"):
        process_dukascopy_json((), "BAD", tmp_path)
    with pytest.raises(ValueError, match="cannot be negative"):
        process_dukascopy_json((), "EURUSD", tmp_path, max_gap_minutes=-1)
    with pytest.raises(ValueError, match="at least one"):
        process_dukascopy_json((), "EURUSD", tmp_path)


def test_historical_json_processor_rejects_conflicting_duplicates(tmp_path):
    import json
    from datetime import UTC, datetime

    from astra.historical.json_source import process_dukascopy_json

    stamp = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1000)
    row = {
        "timestamp": stamp,
        "open": 1.1,
        "high": 1.1,
        "low": 1.1,
        "close": 1.1,
        "volume": 1,
    }
    conflicting = dict(row, close=1.1001)
    source = tmp_path / "fixture.json"
    source.write_text(json.dumps([row, conflicting]), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        process_dukascopy_json(source, "EURUSD", tmp_path / "output")


def test_historical_json_processor_counts_identical_duplicate_bars(tmp_path):
    import json
    from datetime import UTC, datetime

    from astra.historical.json_source import process_dukascopy_json

    stamp = int(datetime(2024, 1, 2, tzinfo=UTC).timestamp() * 1000)
    row = {
        "timestamp": stamp,
        "open": 1.1,
        "high": 1.1,
        "low": 1.1,
        "close": 1.1,
        "volume": 1,
    }
    source = tmp_path / "duplicate.json"
    source.write_text(json.dumps([row, row]), encoding="utf-8")
    report = process_dukascopy_json(source, "EURUSD", tmp_path / "output")
    assert report["source_rows"] == 2
    assert report["unique_m1_rows"] == 1
    assert report["duplicate_rows"] == 1


def test_historical_import_rejects_invalid_ranges_and_low_quality(tmp_path, journal):
    from astra.historical.importer import import_csv

    with pytest.raises(ValueError, match="invalid timeframe or batch size"):
        import_csv(tmp_path / "absent.csv", "EURUSD", "INVALID", journal)

    source = tmp_path / "gapped.csv"
    source.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T00:01:00+00:00,1,1,1,1,1\n"
        "2024-01-02T00:30:00+00:00,1,1,1,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DATA QUALITY INSUFFICIENT"):
        import_csv(source, "EURUSD", "M1", journal)


def test_historical_downloader_isolates_unexpected_worker_errors(tmp_path, monkeypatch):
    downloader = HistoricalDownloader(
        root=tmp_path, workers=1, retries=0, sleep=lambda _: None, request_interval=0
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture worker failure")

    monkeypatch.setattr(downloader, "_download_one", fail)
    result = downloader.download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 2))
    assert result.failed == 1
    assert result.results[0].error == "RuntimeError"


def test_empty_provider_payload_is_marked_invalid(tmp_path):
    class Provider:
        def fetch_m1_day(self, *_args):
            return b""

    downloader = HistoricalDownloader(
        Provider(), root=tmp_path, retries=0, sleep=lambda _: None, request_interval=0
    )
    result = downloader.download(("EURUSD",), date(2024, 1, 2), date(2024, 1, 2))
    assert result.invalid == 1
    assert result.results[0].error == "DataValidationError"


def test_historical_dataset_loader_rejects_unsupported_requests(tmp_path):
    from astra.historical.dataset import load_dataset

    with pytest.raises(ValueError, match="unsupported historical dataset"):
        load_dataset(tmp_path, "NOTPAIR", "M5")


def test_historical_manifest_rejects_bad_shapes_and_saves(tmp_path):
    from astra.historical.manifest import ManifestStore

    path = tmp_path / "manifests" / "dukascopy_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="must be an object"):
        ManifestStore(tmp_path)

    path.write_text('{"symbols": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid dataset or download entries"):
        ManifestStore(tmp_path)

    path.unlink()
    store = ManifestStore(tmp_path)
    store.save()
    assert path.is_file()


def test_decision_rejects_naive_decision_time_and_unclear_or_volatile_market(
    market_data, monkeypatch
):
    from astra.config import Settings
    from astra.decision import decide
    from astra.models import Action, MarketSnapshot, Regime, RiskState, Side
    from astra.risk import RiskEngine

    data = market_data(40)

    class Strategy:
        name = "fixture"

    strategies = [Strategy()]
    state = RiskState(10000, 10000, 10000)
    naive = decide(
        data,
        strategies,
        RiskEngine(Settings()),
        state,
        decision_time=datetime(2025, 1, 1),  # noqa: DTZ001
    )
    assert naive.action == Action.NO_TRADE
    assert "decision time must be timezone-aware" in naive.reason

    def context(regime, volatility):
        return MarketSnapshot(
            data.pair,
            data.timeframe,
            data.candles[-1].timestamp,
            Side.LONG,
            0.5,
            volatility,
            regime,
            ("controlled test context",),
        )

    monkeypatch.setattr(
        "astra.decision.detect_regime", lambda *_: context(Regime.UNCLEAR, 0.001)
    )
    unclear = decide(
        data,
        strategies,
        RiskEngine(Settings()),
        state,
        decision_time=data.candles[-1].timestamp,
    )
    assert unclear.action == Action.NO_TRADE and "unclear regime" in unclear.reason

    monkeypatch.setattr(
        "astra.decision.detect_regime", lambda *_: context(Regime.TRENDING, 1.0)
    )
    volatile = decide(
        data,
        strategies,
        RiskEngine(Settings()),
        state,
        decision_time=data.candles[-1].timestamp,
    )
    assert (
        volatile.action == Action.NO_TRADE and "excessive volatility" in volatile.reason
    )


def test_decision_fails_closed_on_model_and_strategy_errors(
    market_data, monkeypatch, journal
):
    from astra.config import Settings
    from astra.decision import decide
    from astra.models import Action, MarketSnapshot, Regime, RiskState, Side
    from astra.risk import RiskEngine

    data = market_data(40)
    state = RiskState(10000, 10000, 10000)

    class Strategy:
        name = "fixture"

    strategies = [Strategy()]
    monkeypatch.setattr(
        "astra.ml.load_model",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    settings = Settings(ml_enabled=True)
    model_failure = decide(
        data,
        strategies,
        RiskEngine(settings),
        state,
        journal=journal,
        decision_time=data.candles[-1].timestamp,
    )
    assert model_failure.action == Action.NO_TRADE
    assert "model" in model_failure.reason.lower()

    context = MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        Side.LONG,
        0.5,
        0.001,
        Regime.TRENDING,
        ("test",),
    )
    monkeypatch.setattr("astra.decision.detect_regime", lambda *_: context)
    monkeypatch.setattr(
        "astra.decision.select",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("fixture")),
    )
    strategy_failure = decide(
        data,
        strategies,
        RiskEngine(Settings()),
        state,
        decision_time=data.candles[-1].timestamp,
    )
    assert strategy_failure.action == Action.NO_TRADE
    assert "strategy evaluation failed: ValueError" in strategy_failure.reason


def test_decision_vetoes_conflicting_strategy_candidates(market_data, monkeypatch):
    from astra.config import Settings
    from astra.decision import decide
    from astra.intelligence import Selection
    from astra.models import Action, MarketSnapshot, Regime, RiskState, Side, Signal
    from astra.risk import RiskEngine

    data = market_data(40)
    stamp = data.candles[-1].timestamp
    close = data.candles[-1].close
    context = MarketSnapshot(
        data.pair,
        data.timeframe,
        stamp,
        Side.LONG,
        0.5,
        0.001,
        Regime.TRENDING,
        ("test",),
    )
    candidates = Selection(
        (
            (
                0.9,
                Signal(
                    "long-fixture",
                    Side.LONG,
                    0.8,
                    close,
                    close - 0.001,
                    close + 0.002,
                    "test",
                ),
            ),
            (
                0.9,
                Signal(
                    "short-fixture",
                    Side.SHORT,
                    0.8,
                    close,
                    close + 0.001,
                    close - 0.002,
                    "test",
                ),
            ),
        ),
        (),
        "fixture candidates",
    )
    monkeypatch.setattr("astra.decision.detect_regime", lambda *_: context)
    monkeypatch.setattr("astra.decision.select", lambda *_a, **_k: candidates)

    class Strategy:
        name = "fixture"

    decision = decide(
        data,
        [Strategy()],
        RiskEngine(Settings(minimum_confidence=1.0)),
        RiskState(10000, 10000, 10000),
        min_score=0.0,
        decision_time=stamp,
    )
    assert decision.action == Action.NO_TRADE
    assert "conflicting strategy evidence" in decision.reason


def _run_historical_cli(monkeypatch, capsys, args, root, *, end_date=date(2025, 1, 2)):
    settings = HistoricalSettings(
        root=root, symbols=("EURUSD",), start_date=date(2025, 1, 1), end_date=end_date
    )
    monkeypatch.setattr(historical_cli.HistoricalSettings, "from_env", lambda: settings)
    result = historical_cli.historical_main(args)
    import json

    output = json.loads(capsys.readouterr().out)
    return result, output


def test_historical_cli_latest_and_validate_paths(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "astra.historical.pipeline.discover_latest", lambda *a, **kw: date(2025, 1, 1)
    )
    code, output = _run_historical_cli(
        monkeypatch,
        capsys,
        [
            "latest",
            "--symbol",
            "eurusd",
            "--base-url",
            "https://www.dukascopy.com/datafeed",
        ],
        tmp_path,
    )
    assert code == 0 and output["symbol"] == "EURUSD"
    monkeypatch.setattr(
        historical_cli,
        "inspect_csv",
        lambda *a, **kw: SimpleNamespace(status="PASS", rows=2),
    )
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["validate", "eurusd", "m1", "fixture.csv"], tmp_path
    )
    assert code == 0 and output["status"] == "PASS"
    monkeypatch.setattr(
        historical_cli,
        "inspect_csv",
        lambda *a, **kw: SimpleNamespace(status="FAIL", rows=1),
    )
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["validate", "EURUSD", "M1", "fixture.csv"], tmp_path
    )
    assert code == 1 and output["status"] == "FAIL"


def test_historical_cli_normalize_resample_import_and_json_paths(
    monkeypatch, capsys, tmp_path, market_data
):
    monkeypatch.setattr(
        historical_cli, "normalize_symbol", lambda *a: tmp_path / "EURUSD.csv"
    )
    code, output = _run_historical_cli(
        monkeypatch,
        capsys,
        ["normalize", "eurusd", "--start", "2025-01-01", "--end", "2025-01-02"],
        tmp_path,
    )
    assert code == 0 and output.endswith("EURUSD.csv")

    monkeypatch.setattr(historical_cli, "load_csv", lambda *a, **kw: market_data(2))
    monkeypatch.setattr(historical_cli, "resample", lambda *a: ("resampled",))
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["resample", "eurusd", "M5", "fixture.csv"], tmp_path
    )
    assert code == 0 and output == ["resampled"]

    class Repository:
        closed = False

        def close(self):
            self.closed = True

    repo = Repository()
    monkeypatch.setattr("astra.adapters.repository_from_settings", lambda _: repo)
    monkeypatch.setattr("astra.historical.importer.import_csv", lambda *a, **kw: 2)
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["import", "eurusd", "M5", "fixture.csv"], tmp_path
    )
    assert code == 0 and output["imported_records"] == 2 and repo.closed

    monkeypatch.setattr(
        "astra.historical.json_source.process_dukascopy_json",
        lambda *a, **kw: {"timeframes": {"M1": "PASS"}},
    )
    code, output = _run_historical_cli(
        monkeypatch,
        capsys,
        ["process-json", "EURUSD", "fixture.json", "--root", str(tmp_path)],
        tmp_path,
    )
    assert code == 0 and output["timeframes"]["M1"] == "PASS"


def test_historical_cli_pipeline_json_download_and_error_paths(
    monkeypatch, capsys, tmp_path
):
    from astra.historical import dukascopy_node, json_source, manifest

    class Downloader:
        def __init__(self, *a, **kw):
            pass

        def download(self, symbols, *a, **kw):
            return {
                "symbols": {
                    symbol: {"status": "downloaded", "files": ["fixture.json"]}
                    for symbol in symbols
                }
            }

    class Store:
        updated = False

        def __init__(self, root):
            pass

        def update_dataset(self, **kwargs):
            self.updated = True

    monkeypatch.setattr(dukascopy_node, "DukascopyNodeDownloader", Downloader)
    monkeypatch.setattr(
        json_source,
        "process_dukascopy_json",
        lambda *a, **kw: {"status": "PASS", "timeframes": {"M1": "PASS"}},
    )
    monkeypatch.setattr(manifest, "ManifestStore", Store)
    args = [
        "pipeline-json",
        "--start",
        "2025-01-01",
        "--end",
        "2025-01-01",
        "--symbols",
        "eurusd",
        "--root",
        str(tmp_path),
    ]
    code, output = _run_historical_cli(monkeypatch, capsys, args, tmp_path)
    assert code == 0 and output["processed"]["EURUSD"]["status"] == "PASS"
    code, output = _run_historical_cli(
        monkeypatch, capsys, args + ["--dry-run"], tmp_path
    )
    assert code == 0 and output["processed"] == {}

    monkeypatch.setattr(
        json_source,
        "process_dukascopy_json",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("fixture processing failure")
        ),
    )
    code, output = _run_historical_cli(monkeypatch, capsys, args, tmp_path)
    assert code == 1 and output["processed"]["EURUSD"]["error"] == "RuntimeError"
    monkeypatch.setattr(
        json_source,
        "process_dukascopy_json",
        lambda *a, **kw: {"status": "PASS", "timeframes": {"M1": "PASS"}},
    )

    class FailedDownloader(Downloader):
        def download(self, symbols, *a, **kw):
            return {
                "symbols": {
                    symbol: {"status": "failed", "files": []} for symbol in symbols
                }
            }

    monkeypatch.setattr(dukascopy_node, "DukascopyNodeDownloader", FailedDownloader)
    code, _ = _run_historical_cli(monkeypatch, capsys, args, tmp_path)
    assert code == 1

    code, output = _run_historical_cli(
        monkeypatch, capsys, ["pipeline", "--dry-run"], tmp_path, end_date=None
    )
    assert code == 2 and output["status"] == "ERROR"


def test_historical_cli_pipeline_and_download_use_configured_ranges(
    monkeypatch, capsys, tmp_path
):
    summary = DownloadSummary(1, 1, 0, 0, 0, 0)
    monkeypatch.setattr(historical_cli, "run_pipeline", lambda *a, **kw: summary)
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["pipeline", "--dry-run", "--force"], tmp_path
    )
    assert code == 0 and output["requested"] == 1

    class Downloader:
        def __init__(self, *a, **kw):
            pass

        def download(self, *a, **kw):
            return summary

    monkeypatch.setattr(historical_cli, "HistoricalDownloader", Downloader)
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["download", "--dry-run"], tmp_path
    )
    assert code == 0 and output["downloaded"] == 1

    monkeypatch.setattr(
        "astra.historical.pipeline.discover_latest",
        lambda *a, **kw: date(2025, 1, 3),
    )
    code, output = _run_historical_cli(
        monkeypatch, capsys, ["download"], tmp_path, end_date=None
    )
    assert code == 0 and output["downloaded"] == 1, output

    class FailedSummaryDownloader(Downloader):
        def download(self, *a, **kw):
            return DownloadSummary(1, 0, 0, 0, 1, 0)

    monkeypatch.setattr(historical_cli, "HistoricalDownloader", FailedSummaryDownloader)
    code, output = _run_historical_cli(monkeypatch, capsys, ["download"], tmp_path)
    assert code == 1 and output["failed"] == 1


def test_historical_configuration_parses_environment_and_rejects_bad_values(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HISTORICAL_DATA_SYMBOLS", "eurusd, gbpusd")
    monkeypatch.setenv("HISTORICAL_DATA_START_DATE", "2024-01-01")
    monkeypatch.setenv("HISTORICAL_DATA_END_DATE", "2024-01-02")
    monkeypatch.setenv("HISTORICAL_DATA_ROOT", str(tmp_path))
    settings = HistoricalSettings.from_env()
    assert settings.symbols == ("EURUSD", "GBPUSD")
    assert settings.end_date == date(2024, 1, 2)

    monkeypatch.setenv("HISTORICAL_DATA_START_DATE", "not-a-date")
    with pytest.raises(ValueError, match="date configuration"):
        HistoricalSettings.from_env()
    monkeypatch.setenv("HISTORICAL_DATA_START_DATE", "2024-01-01")
    monkeypatch.setenv("HISTORICAL_DATA_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="numeric historical"):
        HistoricalSettings.from_env()


@pytest.mark.parametrize(
    "changes",
    [
        {"start_date": date(2099, 1, 1)},
        {"start_date": date(2025, 1, 2), "end_date": date(2025, 1, 1)},
        {"end_date": date(2099, 1, 1)},
        {"symbols": ()},
        {"symbols": ("EURUSD", "EURUSD")},
        {"symbols": ("eurusd",)},
        {"workers": 0},
        {"retries": 11},
        {"timeout": 0},
        {"max_gap_minutes": 1441},
        {"batch_size": 0},
        {"root": " "},
        {"base_url": "https://example.invalid"},
    ],
)
def test_historical_settings_reject_invalid_combinations(changes):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(HistoricalSettings(), **changes).validate()


def test_historical_downloader_classifies_missing_invalid_and_transient(tmp_path):
    class Provider:
        def __init__(self, responses):
            self.responses = iter(responses)

        def fetch_m1_day(self, *_):
            value = next(self.responses)
            if isinstance(value, Exception):
                raise value
            return value

    day = date(2024, 1, 1)
    missing = HistoricalDownloader(
        Provider([None]),
        root=tmp_path / "missing",
        retries=0,
        sleep=lambda _: None,
        request_interval=0,
    )
    assert missing.download(("EURUSD",), day, day).missing == 1

    invalid = HistoricalDownloader(
        Provider([b"not-lzma"]),
        root=tmp_path / "invalid",
        retries=0,
        sleep=lambda _: None,
        request_interval=0,
    )
    assert invalid.download(("EURUSD",), day, day).invalid == 1

    from astra.historical.provider import PermanentSourceError, TransientSourceError

    permanent = HistoricalDownloader(
        Provider([PermanentSourceError("not-found")]),
        root=tmp_path / "permanent",
        retries=4,
        sleep=lambda _: None,
        request_interval=0,
    )
    failed = permanent.download(("EURUSD",), day, day)
    assert failed.failed == 1 and failed.results[0].error == "PermanentSourceError"

    transient = HistoricalDownloader(
        Provider([TransientSourceError("retry"), None]),
        root=tmp_path / "transient",
        retries=1,
        sleep=lambda _: None,
        request_interval=0,
    )
    assert transient.download(("EURUSD",), day, day).missing == 1

    with pytest.raises(ValueError, match="request interval"):
        HistoricalDownloader(root=tmp_path, request_interval=-1)


def test_dukascopy_provider_retries_only_transient_source_failures():
    from urllib.error import HTTPError, URLError

    from astra.historical.dukascopy import DukascopyProvider
    from astra.historical.provider import PermanentSourceError, TransientSourceError

    class Response:
        def __init__(self, status, payload=b""):
            self.status, self.payload = status, payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _limit):
            return self.payload

    day = date(2024, 1, 1)
    with pytest.raises(ValueError, match="uppercase"):
        DukascopyProvider().build_m1_url("eurusd", day)
    assert (
        DukascopyProvider(transport=lambda *a, **kw: Response(200)).fetch_m1_day(
            "EURUSD", day, 1
        )
        is None
    )
    with pytest.raises(TransientSourceError):
        DukascopyProvider(transport=lambda *a, **kw: Response(503)).fetch_m1_day(
            "EURUSD", day, 1
        )
    with pytest.raises(PermanentSourceError):
        DukascopyProvider(transport=lambda *a, **kw: Response(400)).fetch_m1_day(
            "EURUSD", day, 1
        )

    def error(code):
        return HTTPError("https://feed.invalid", code, "private", {}, None)

    assert (
        DukascopyProvider(
            transport=lambda *a, **kw: (_ for _ in ()).throw(error(404))
        ).fetch_m1_day("EURUSD", day, 1)
        is None
    )
    with pytest.raises(TransientSourceError):
        DukascopyProvider(
            transport=lambda *a, **kw: (_ for _ in ()).throw(error(429))
        ).fetch_m1_day("EURUSD", day, 1)
    with pytest.raises(PermanentSourceError):
        DukascopyProvider(
            transport=lambda *a, **kw: (_ for _ in ()).throw(error(401))
        ).fetch_m1_day("EURUSD", day, 1)
    with pytest.raises(TransientSourceError):
        DukascopyProvider(
            transport=lambda *a, **kw: (_ for _ in ()).throw(URLError("offline"))
        ).fetch_m1_day("EURUSD", day, 1)
    with pytest.raises(PermanentSourceError, match="size limit"):
        DukascopyProvider(
            transport=lambda *a, **kw: Response(200, b"x" * 2_000_001)
        ).fetch_m1_day("EURUSD", day, 1)


def test_dukascopy_decoder_rejects_corrupt_records_and_bad_scale():
    import lzma
    import struct

    from astra.data import DataValidationError
    from astra.historical.dukascopy import decode_m1_bi5

    day = date(2024, 1, 1)
    with pytest.raises(DataValidationError, match="LZMA"):
        decode_m1_bi5(b"invalid", "EURUSD", day)
    with pytest.raises(DataValidationError, match="incomplete M1 record"):
        decode_m1_bi5(lzma.compress(b"x"), "EURUSD", day)
    with pytest.raises(DataValidationError, match="incomplete M1 record"):
        decode_m1_bi5(lzma.compress(b""), "EURUSD", day)
    good = (0, 100000, 100000, 100000, 100000, 1.0)
    compressed = lzma.compress(struct.pack(">IIIIIf", *good))
    with pytest.raises(DataValidationError, match="truncated or exceeds"):
        decode_m1_bi5(compressed[:-1], "EURUSD", day)
    with pytest.raises(DataValidationError, match="truncated or exceeds"):
        decode_m1_bi5(compressed + compressed, "EURUSD", day)
    full_day = b"".join(
        struct.pack(">IIIIIf", minute * 60, 100000, 100000, 100000, 100000, 1.0)
        for minute in range(1441)
    )
    with pytest.raises(DataValidationError, match="too many"):
        decode_m1_bi5(lzma.compress(full_day), "EURUSD", day)
    with pytest.raises(ValueError, match="scale"):
        decode_m1_bi5(lzma.compress(struct.pack(">IIIIIf", *good)), "EURUSD", day, 0)
    cases = [
        (86400, 100000, 100000, 100000, 100000, 1.0),
        (0, 100000, 100000, 100000, 100000, -1.0),
        (0, 100000, 100000, 100000, 100000, float("inf")),
        (61, 100000, 100000, 100000, 100000, 1.0),
        (0, 0, 100000, 100000, 100000, 1.0),
        (0, 100000, 110000, 120000, 130000, 1.0),
    ]
    for row in cases:
        with pytest.raises(DataValidationError):
            decode_m1_bi5(lzma.compress(struct.pack(">IIIIIf", *row)), "EURUSD", day)
    dup = lzma.compress(struct.pack(">IIIIIf", *good) + struct.pack(">IIIIIf", *good))
    with pytest.raises(DataValidationError, match="ordered and unique"):
        decode_m1_bi5(dup, "EURUSD", day)
