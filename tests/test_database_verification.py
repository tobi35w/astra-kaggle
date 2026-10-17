"""Deterministic schema/Journal verification; no hosted database is used here."""

from copy import deepcopy

from astra.config import Settings
from astra.database_verification import failure_category, verify_database
from astra.repositories import Entity, MemoryRepository


def schema_fixture():
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
            "CHECK (entity IN (" + ",".join(f"'{e.value}'" for e in Entity) + "))",
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


def settings():
    return Settings(
        persistence_backend="postgres", supabase_db_url="test-dsn-never-used"
    )


def test_verification_proves_idempotent_roundtrip_and_keeps_one_audit_event():
    repo = MemoryRepository()
    result = verify_database(
        settings(), schema_reader=lambda _: schema_fixture(), repository=repo
    )
    assert result["status"] == "VERIFIED"
    assert result["migrations_applied"] is False
    assert len(repo.query(Entity.AUDIT)) == 1


def test_incompatible_schema_prevents_any_write():
    repo = MemoryRepository()
    schema = deepcopy(schema_fixture())
    schema["constraints"][3] = "CHECK (timeframe IN ('','M5','M15','H1','H4'))"
    schema["rls"] = False
    result = verify_database(
        settings(), schema_reader=lambda _: schema, repository=repo
    )
    assert result["schema"] == "INCOMPATIBLE"
    assert any("M1" in issue for issue in result["issues"])
    assert not repo.query(Entity.AUDIT)


def test_authentication_errors_are_classified_without_secret_output():
    def failing_reader(_):
        raise RuntimeError(
            "password authentication failed for user secret-user; password=secret-value"
        )

    result = verify_database(settings(), schema_reader=failing_reader)
    assert result["reason"] == "AUTHENTICATION_FAILED"
    assert "secret" not in str(result)
    assert result["journal_round_trip"] == "NOT_CHECKED"


def test_failed_readback_never_reports_verified():
    class DroppedWrite(MemoryRepository):
        def append_batch(self, records):
            return None

    result = verify_database(
        settings(), schema_reader=lambda _: schema_fixture(), repository=DroppedWrite()
    )
    assert result["journal_round_trip"] == "FAILED"
    assert result["status"] == "NOT_VERIFIED"


def test_local_adapter_cannot_be_mislabeled_as_hosted_verified():
    assert verify_database(Settings())["status"] == "NOT_VERIFIED"
    assert failure_category(OSError("getaddrinfo failed private-host")) == "DNS_FAILED"
