"""Explicit hosted-schema and Journal round-trip verification; never applies DDL."""

from __future__ import annotations

from datetime import UTC, datetime

from .adapters import repository_from_settings
from .config import Settings
from .repositories import Entity, Journal, Record


def failure_category(error: Exception) -> str:
    """Return a useful diagnostic without exposing DSNs, usernames, or responses."""
    message = str(error).lower()
    for category, fragments in (
        (
            "AUTHENTICATION_FAILED",
            (
                "authentication failed",
                "password authentication",
                "http 401",
                "http 403",
            ),
        ),
        (
            "DNS_FAILED",
            ("getaddrinfo", "could not translate", "resolve", "name or service"),
        ),
        ("TIMEOUT", ("timed out", "timeout")),
        ("TLS_FAILED", ("certificate", "ssl", "handshake")),
        ("CONNECTION_FAILED", ("unreachable", "refused", "permission denied")),
    ):
        if any(fragment in message for fragment in fragments):
            return category
    return "OPERATION_FAILED"


def read_schema(dsn: str, *, connect=None) -> dict:
    if connect is None:
        import psycopg

        connect = psycopg.connect
    with connect(dsn, connect_timeout=5) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT column_name,data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='astra_records'"
        )
        columns = dict(cursor.fetchall())
        cursor.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid=to_regclass('public.astra_records')"
        )
        constraints = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='astra_records'"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        cursor.execute(
            "SELECT relrowsecurity FROM pg_class WHERE oid=to_regclass('public.astra_records')"
        )
        row = cursor.fetchone()
        rls = bool(row and row[0])
        cursor.execute(
            "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgrelid=to_regclass('public.astra_records') AND NOT tgisinternal AND tgenabled <> 'D'"
        )
        triggers = [row[0] for row in cursor.fetchall()]
    return {
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        "rls": rls,
        "triggers": triggers,
    }


def schema_issues(schema: dict) -> list[str]:
    required = {
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
    }
    issues = [
        f"missing or incompatible column: {name}"
        for name, kind in required.items()
        if schema["columns"].get(name) != kind
    ]
    constraints = " ".join(schema["constraints"])
    for value in ("M1", "M5", "M15", "H1", "H4", *(e.value for e in Entity)):
        if f"'{value}'" not in constraints:
            issues.append(f"schema constraint does not admit {value}")
    if "PRIMARY KEY (record_id)" not in constraints:
        issues.append("journal UUID primary key missing")
    if (
        "jsonb_typeof(payload)" not in constraints
        or "schema_version = 1" not in constraints
    ):
        issues.append("payload or schema-version constraint missing")
    indexes = " ".join(schema["indexes"])
    for columns in (
        "entity, pair, timeframe, strategy, occurred_at, record_id",
        "correlation_id",
        "recorded_at",
    ):
        if columns not in indexes:
            issues.append(f"required lookup index missing: {columns}")
    if not schema["rls"]:
        issues.append("row-level security is not enabled")
    if not any(
        all(
            part in trigger
            for part in (
                "BEFORE",
                "UPDATE",
                "DELETE",
                "FOR EACH ROW",
                "astra_reject_journal_mutation",
            )
        )
        for trigger in schema["triggers"]
    ):
        issues.append("append-only journal trigger missing")
    return issues


def verify_database(
    settings: Settings, *, schema_reader=read_schema, repository=None
) -> dict:
    """Check hosted schema before writing one retained audit event through Journal.

    Repeated insertion of the same UUID must be idempotent. The event remains as
    evidence; no existing rows are updated or deleted. This does not apply a migration.
    """
    settings.validate()
    result = {
        "status": "NOT_VERIFIED",
        "checked_at": datetime.now(UTC).isoformat(),
        "backend": settings.persistence_backend,
        "schema": "NOT_CHECKED",
        "journal_round_trip": "NOT_CHECKED",
        "migrations_applied": False,
    }
    if settings.persistence_backend not in {"postgres", "supabase"}:
        return result | {
            "reason": "A hosted PostgreSQL or Supabase backend is required"
        }
    dsn = settings.supabase_db_url or settings.database_url
    if not dsn:
        return result | {
            "reason": "SUPABASE_DB_URL is required to inspect hosted constraints and indexes"
        }
    try:
        schema = schema_reader(dsn)
    except Exception as error:  # noqa: BLE001 - external drivers may include credentials in exceptions.
        return result | {
            "schema": "FAILED",
            "reason": failure_category(error),
            "error_type": type(error).__name__,
        }
    issues = schema_issues(schema)
    if issues:
        return result | {
            "schema": "INCOMPATIBLE",
            "issues": issues,
            "reason": "Reconcile the existing schema with the documented migrations before retrying",
        }
    result["schema"] = "VERIFIED"
    owned = repository is None
    try:
        repository = repository or repository_from_settings(settings)
        journal = Journal(repository)
        record = Record(
            Entity.AUDIT,
            datetime.now(UTC),
            {
                "event": "database_verification",
                "purpose": "explicit hosted Journal round-trip",
            },
            strategy="database_verification",
        )
        journal.repository.append_batch((record,))
        journal.repository.append_batch((record,))
        matches = [
            r
            for r in journal.repository.query(
                Entity.AUDIT, strategy="database_verification"
            )
            if r.record_id == record.record_id
        ]
        if len(matches) != 1 or matches[0].row() != record.row():
            return result | {
                "journal_round_trip": "FAILED",
                "reason": "Journal read-back or idempotency mismatch",
            }
        return result | {
            "status": "VERIFIED",
            "journal_round_trip": "VERIFIED",
            "audit_record_id": record.record_id,
            "updates": "Append-only schema; state changes are new events",
        }
    except Exception as error:  # noqa: BLE001 - retain stage and sanitized external failure only.
        return result | {
            "journal_round_trip": "FAILED",
            "reason": failure_category(error),
            "error_type": type(error).__name__,
        }
    finally:
        if owned and repository is not None and hasattr(repository, "close"):
            repository.close()
