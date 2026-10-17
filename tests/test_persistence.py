import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from urllib.error import HTTPError

import pytest

from astra.adapters import (
    PostgresRepository,
    SQLiteRepository,
    SupabaseRepository,
    repository_from_settings,
)
from astra.config import Settings
from astra.decision import decide
from astra.execution import PaperExecutionProvider, PaperPortfolio
from astra.memory import PerformanceMemory
from astra.models import (
    Action,
    RiskApproval,
    RiskState,
    Side,
    Signal,
    StrategyPerformance,
)
from astra.pipeline import AstraPaperPipeline
from astra.repositories import (
    Entity,
    Journal,
    MemoryRepository,
    PersistenceError,
    Record,
    json_value,
)
from astra.risk import RiskEngine
from astra.strategies import EmaTrend

NOW = datetime(2025, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("entity", list(Entity))
def test_each_entity_roundtrips_with_idempotency_and_asof(tmp_path, entity):
    path = str(tmp_path / "journal.sqlite3")
    store = SQLiteRepository(path)
    record = Record(
        entity,
        NOW,
        {"nested": {"number": 1}},
        pair="EURUSD",
        timeframe="M5",
        strategy="trend",
    )
    later = replace(record, record_id="future", occurred_at=NOW + timedelta(days=1))
    store.append_batch((record, later))
    store.append_batch((replace(record, payload={"changed": True}),))
    store.close()
    reopened = SQLiteRepository(path)
    assert reopened.query(
        entity, as_of=NOW, pair="EURUSD", timeframe="M5", strategy="trend"
    ) == (record,)
    assert len(reopened.query(entity)) == 2
    assert reopened.healthy()
    reopened.close()
    assert not reopened.healthy()


def test_memory_does_not_alias_or_overwrite_payload():
    store = MemoryRepository()
    record = Record(Entity.AUDIT, NOW, {"value": [1]})
    store.append_batch((record,))
    record.payload["value"].append(2)
    store.append_batch((record,))
    result = store.query(Entity.AUDIT)[0]
    assert result.payload == {"value": [1]}
    result.payload["value"].append(3)
    assert store.query(Entity.AUDIT)[0].payload == {"value": [1]}


def test_sqlite_batch_validation_is_atomic_and_errors_sanitized(tmp_path):
    store = SQLiteRepository(str(tmp_path / "db"))
    valid = Record(Entity.AUDIT, NOW, {"ok": True})
    invalid = replace(valid, record_id="bad", payload={"unserializable": object()})
    with pytest.raises(PersistenceError):
        store.append_batch((valid, invalid))
    assert not store.query(Entity.AUDIT)
    store.close()
    with pytest.raises(PersistenceError):
        store.query(Entity.AUDIT)


def test_supabase_uses_requested_privilege_and_parameterized_filters():
    calls = []
    record = Record(Entity.PERFORMANCE, NOW, {"ok": True}, strategy="x&select=*")

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        return [record.row()] if method == "GET" else []

    low = SupabaseRepository(
        "https://example.supabase.co", "sb_publishable_fixture", transport=transport
    )
    assert low.query(Entity.PERFORMANCE, as_of=NOW, strategy=record.strategy) == (
        record,
    )
    assert "x%26select%3D%2A" in calls[-1][1]
    assert calls[-1][2]["apikey"] == "sb_publishable_fixture"
    assert "Authorization" not in calls[-1][2]
    with pytest.raises(PersistenceError, match="server-side"):
        low.append_batch((record,))
    assert not low.healthy()
    high = SupabaseRepository(
        "https://example.supabase.co",
        secret_key="sb_secret_fixture",
        use_secret=True,
        transport=transport,
    )
    high.append_batch((record,))
    assert json.loads(calls[-1][3])[0]["record_id"] == record.record_id
    assert high.healthy()


def test_supabase_pagination_jwt_and_sanitized_failures():
    record = Record(Entity.AUDIT, NOW, {})
    pages = [[record.row()] * 500, [record.row()]]
    calls = []

    def transport(method, url, headers, body):
        calls.append((url, headers))
        return pages.pop(0)

    store = SupabaseRepository(
        "https://example.supabase.co", "eyJfixture", transport=transport
    )
    assert len(store.query(Entity.AUDIT)) == 501
    assert "offset=500" in calls[-1][0]
    assert calls[-1][1]["Authorization"] == "Bearer eyJfixture"

    def failing(*args):
        raise HTTPError("https://secret.invalid", 403, "credential-secret", {}, None)

    store._transport = failing
    with pytest.raises(PersistenceError) as error:
        store.query(Entity.AUDIT)
    assert "credential-secret" not in str(error.value)
    assert not store.healthy()
    store._transport = lambda *args: {"bad": "shape"}
    with pytest.raises(PersistenceError):
        store.query(Entity.AUDIT)
    store._transport = lambda *args: [{}]
    with pytest.raises(PersistenceError):
        store.query(Entity.AUDIT)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:pass@example.com",
        "https://example.com?key=secret",
        "",
    ],
)
def test_supabase_rejects_unsafe_configuration(url):
    with pytest.raises(ValueError):
        SupabaseRepository(url, "fixture")


def test_postgres_batch_and_asof_queries_are_parameterized():
    record = Record(
        Entity.DECISION, NOW, {"action": "NO_TRADE"}, strategy="'; DROP TABLE x; --"
    )
    cursor, connection = MagicMock(), MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchall.return_value = [(record.row(),)]
    connect = MagicMock(return_value=connection)
    store = PostgresRepository("postgresql://fixture", connect=connect)
    store.append_batch((record,))
    assert record.strategy not in cursor.execute.call_args.args[0]
    assert record.strategy in cursor.execute.call_args.args[1]
    assert store.query(
        Entity.DECISION,
        strategy=record.strategy,
        pair="EURUSD",
        timeframe="M5",
        as_of=NOW,
    ) == (record,)
    assert record.strategy not in cursor.execute.call_args.args[0]
    assert store.healthy()
    connect.side_effect = RuntimeError("dsn-secret")
    with pytest.raises(PersistenceError) as error:
        store.append_batch((record,))
    assert "dsn-secret" not in str(error.value)
    assert not store.healthy()


def test_repository_factory_and_redacted_configuration(tmp_path, monkeypatch):
    assert isinstance(
        repository_from_settings(Settings(persistence_backend="memory")),
        MemoryRepository,
    )
    store = repository_from_settings(Settings(sqlite_path=str(tmp_path / "db")))
    assert store.healthy()
    store.close()
    assert isinstance(
        repository_from_settings(
            Settings(
                persistence_backend="postgres", database_url="postgresql://fixture"
            )
        ),
        PostgresRepository,
    )
    cfg = Settings(
        persistence_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_publishable_key="private-test-value",
        database_url="private-dsn",
    )
    assert isinstance(repository_from_settings(cfg), SupabaseRepository)
    assert "private-test-value" not in repr(cfg) + json.dumps(cfg.public())
    assert "private-dsn" not in repr(cfg) + json.dumps(cfg.public())
    monkeypatch.setenv("ASTRA_MAX_SPREAD", "nan")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_decision_journal_records_reconstructable_bundle(journal, market_data):
    data = market_data()
    decision = decide(
        data,
        [EmaTrend()],
        RiskEngine(Settings(trading_enabled=True)),
        RiskState(1000, 1000, 1000),
        decision_time=data.candles[-1].timestamp,
        journal=journal,
    )
    assert decision.action == Action.TRADE
    rows = journal.repository.query(Entity.DECISION)
    assert rows[0].payload["context"]["selection"]["candidates"]
    assert (
        journal.repository.query(Entity.RISK)[0].correlation_id
        == rows[0].correlation_id
    )
    assert journal.repository.query(Entity.OBSERVATION)
    observation = journal.repository.query(Entity.OBSERVATION)[0].payload
    assert observation["bar_count"] == len(data.candles)
    assert observation["candle"]["timestamp"] == data.candles[-1].timestamp.isoformat()
    assert "candles" not in observation
    assert len(json.dumps(observation)) < 1000
    assert journal.repository.query(Entity.CONFIDENCE)


def test_failed_journal_prevents_paper_execution(market_data):
    class Failing(MemoryRepository):
        def append_batch(self, records):
            raise PersistenceError("offline")

    data = market_data()
    portfolio = PaperPortfolio(PaperExecutionProvider(), 1000)
    pipeline = AstraPaperPipeline(
        [EmaTrend()],
        RiskEngine(Settings(trading_enabled=True)),
        portfolio,
        journal=Journal(Failing()),
    )
    result = pipeline.evaluate(
        data, RiskState(1000, 1000, 1000), decision_time=data.candles[-1].timestamp
    )
    assert result.decision.action == Action.NO_TRADE
    assert "PERSISTENCE_FAILURE" in result.decision.reason_codes
    assert not portfolio.positions


def test_performance_and_calibration_are_only_available_after_close(journal):
    portfolio = PaperPortfolio(
        PaperExecutionProvider(commission_per_unit=0.01), 1000, journal
    )
    signal = Signal("trend", Side.LONG, 0.8, 1, 0.9, 1.2, "fixture")
    position = portfolio.open(
        signal, RiskApproval(True, (), 2), NOW, pair="EURUSD", timeframe="M5"
    )
    trade = portfolio.close(position.position_id, 1.2, NOW + timedelta(minutes=5))
    memory = PerformanceMemory(journal)
    memory.observe_trade(trade, regime="TRENDING", confidence=0.8, setup_type="ema")
    assert not memory.history("EURUSD", "M5", NOW)
    assert memory.history("EURUSD", "M5", trade.closed_at)[0].outcome == pytest.approx(
        0.38
    )
    assert memory.summarize("EURUSD", "M5", trade.closed_at)[0].trades == 1
    assert journal.repository.query(Entity.CALIBRATION)
    assert journal.repository.query(Entity.TRADE)
    assert journal.repository.query(Entity.PNL)
    future = StrategyPerformance(
        "trend", "TRENDING", "EURUSD", "M5", 1, 1, 0, NOW + timedelta(days=2)
    )
    journal.record(Entity.PERFORMANCE, NOW, future, pair="EURUSD", timeframe="M5")
    assert len(memory.history("EURUSD", "M5", trade.closed_at)) == 1


def test_json_rejects_naive_timestamps_and_unsupported_schema():
    with pytest.raises(ValueError):
        json_value(datetime(2025, 1, 1))  # noqa: DTZ001 - intentional invalid input.
    with pytest.raises(ValueError):
        Record(Entity.AUDIT, NOW, {}, schema_version=2).row()
    assert json_value(float("inf")) is None
