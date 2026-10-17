"""SQLite, PostgreSQL and Supabase adapters for the same domain journal contract."""

from __future__ import annotations

import json
import sqlite3
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .repositories import Entity, MemoryRepository, PersistenceError, Record, json_value


class SQLiteRepository:
    def __init__(self, path: str):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS astra_records (record_id TEXT PRIMARY KEY, "
            "entity TEXT NOT NULL, occurred_at TEXT NOT NULL, pair TEXT NOT NULL, "
            "timeframe TEXT NOT NULL, strategy TEXT NOT NULL, row_json TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS astra_records_lookup "
            "ON astra_records(entity,pair,timeframe,strategy,occurred_at)"
        )

    def append_batch(self, records):
        try:
            rows = [r.row() for r in records]
            with self.connection:
                self.connection.executemany(
                    "INSERT OR IGNORE INTO astra_records VALUES (?,?,?,?,?,?,?)",
                    [
                        (
                            r["record_id"],
                            r["entity"],
                            r["occurred_at"],
                            r["pair"],
                            r["timeframe"],
                            r["strategy"],
                            json.dumps(r, allow_nan=False),
                        )
                        for r in rows
                    ],
                )
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            raise PersistenceError("SQLite journal write failed") from None

    def query(self, entity, *, as_of=None, pair="", timeframe="", strategy=""):
        clauses, args = ["entity = ?"], [entity.value]
        for key, value in (
            ("pair", pair),
            ("timeframe", timeframe),
            ("strategy", strategy),
        ):
            if value:
                clauses.append(f"{key} = ?")
                args.append(value)
        if as_of is not None:
            clauses.append("occurred_at <= ?")
            args.append(json_value(as_of))
        try:
            rows = self.connection.execute(
                "SELECT row_json FROM astra_records WHERE "
                + " AND ".join(clauses)
                + " ORDER BY occurred_at, record_id",
                args,
            ).fetchall()
            return tuple(Record.from_row(json.loads(row[0])) for row in rows)
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            raise PersistenceError("SQLite journal read failed") from None

    def healthy(self):
        try:
            self.connection.execute("SELECT record_id FROM astra_records LIMIT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self):
        self.connection.close()


class PostgresRepository:
    def __init__(self, connection_string: str, connect=None):
        if not connection_string:
            raise ValueError("PostgreSQL connection credential is required")
        self._dsn = connection_string
        if connect is None:
            try:
                import psycopg
            except ImportError:
                raise PersistenceError("install the postgres extra") from None
            connect = psycopg.connect
        self._connect = connect

    def append_batch(self, records):
        try:
            from psycopg.types.json import Jsonb

            rows = [r.row() for r in records]
            with (
                self._connect(self._dsn, connect_timeout=5) as conn,
                conn.cursor() as cursor,
            ):
                for row in rows:
                    cursor.execute(
                        "INSERT INTO astra_records "
                        "(record_id,entity,occurred_at,pair,timeframe,strategy,correlation_id,schema_version,payload) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (record_id) DO NOTHING",
                        tuple(
                            row[k]
                            for k in (
                                "record_id",
                                "entity",
                                "occurred_at",
                                "pair",
                                "timeframe",
                                "strategy",
                                "correlation_id",
                                "schema_version",
                            )
                        )
                        + (Jsonb(row["payload"]),),
                    )
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            raise PersistenceError("PostgreSQL journal write failed") from None

    def query(self, entity, *, as_of=None, pair="", timeframe="", strategy=""):
        clauses, args = ["entity = %s"], [entity.value]
        for key, value in (
            ("pair", pair),
            ("timeframe", timeframe),
            ("strategy", strategy),
        ):
            if value:
                clauses.append(f"{key} = %s")
                args.append(value)
        if as_of is not None:
            clauses.append("occurred_at <= %s")
            args.append(as_of)
        try:
            with (
                self._connect(self._dsn, connect_timeout=5) as conn,
                conn.cursor() as cursor,
            ):
                cursor.execute(
                    "SELECT row_to_json(r) FROM astra_records r WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY occurred_at,record_id",
                    args,
                )
                return tuple(Record.from_row(row[0]) for row in cursor.fetchall())
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            raise PersistenceError("PostgreSQL journal read failed") from None

    def healthy(self):
        try:
            self.query(Entity.HEALTH)
            return True
        except PersistenceError:
            return False


class SupabaseRepository:
    """PostgREST transport. Private journal writes require explicit server credential opt-in.

    Publishable keys are never promoted to elevated credentials. No permissive RLS policy
    is installed. HTTP bodies/headers/errors are never included in application exceptions.
    """

    def __init__(
        self,
        url: str,
        publishable_key: str = "",
        *,
        secret_key: str = "",
        use_secret: bool = False,
        transport=None,
    ):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
        ):
            raise ValueError("SUPABASE_URL must be a plain HTTPS project URL")
        self._key = secret_key if use_secret else publishable_key
        if not self._key:
            raise ValueError("the selected Supabase credential is missing")
        self._write_enabled = use_secret
        self._base = url.rstrip("/") + "/rest/v1/astra_records"
        self._transport = transport or self._http

    @staticmethod
    def _http(method, url, headers, body):
        request = Request(url, data=body, headers=headers, method=method)
        with urlopen(request, timeout=10) as response:
            raw = response.read()
        return json.loads(raw) if raw else []

    def _request(self, method, query="", body=None):
        headers = {
            "apikey": self._key,
            "Content-Type": "application/json",
            "Prefer": "resolution=ignore-duplicates,return=minimal",
        }
        # Legacy JWT project keys need Authorization; new sb_* keys are not JWTs.
        if self._key.startswith("eyJ"):
            headers["Authorization"] = "Bearer " + self._key
        try:
            return self._transport(
                method,
                self._base + query,
                headers,
                json.dumps(body, allow_nan=False).encode()
                if body is not None
                else None,
            )
        except HTTPError as error:
            raise PersistenceError(
                f"Supabase request failed (HTTP {error.code})"
            ) from None
        except Exception:  # noqa: BLE001 - fail closed at an external boundary.
            raise PersistenceError("Supabase request failed") from None

    def append_batch(self, records):
        if not self._write_enabled:
            raise PersistenceError(
                "private journal writes require explicit server-side secret access"
            )
        if records:
            self._request("POST", "?on_conflict=record_id", [r.row() for r in records])

    def query(self, entity, *, as_of=None, pair="", timeframe="", strategy=""):
        filters = {
            "entity": "eq." + entity.value,
            "order": "occurred_at.asc,record_id.asc",
            "limit": "500",
        }
        for key, value in (
            ("pair", pair),
            ("timeframe", timeframe),
            ("strategy", strategy),
        ):
            if value:
                filters[key] = "eq." + value
        if as_of is not None:
            filters["occurred_at"] = "lte." + json_value(as_of)
        rows: list[Record] = []
        offset = 0
        while True:
            page = self._request("GET", "?" + urlencode(filters | {"offset": offset}))
            if not isinstance(page, list):
                raise PersistenceError("invalid Supabase journal response")
            try:
                rows.extend(Record.from_row(row) for row in page)
            except (ValueError, TypeError, KeyError):
                raise PersistenceError("invalid Supabase journal record") from None
            if len(page) < 500:
                return tuple(rows)
            offset += len(page)

    def healthy(self):
        try:
            self._request("GET", "?select=record_id&limit=1")
            # RLS can hide everything from a publishable key; it is not write readiness.
            return self._write_enabled
        except PersistenceError:
            return False


def repository_from_settings(settings):
    settings.validate()
    if settings.persistence_backend == "memory":
        return MemoryRepository()
    if settings.persistence_backend == "sqlite":
        return SQLiteRepository(settings.sqlite_path)
    if settings.persistence_backend == "postgres":
        return PostgresRepository(
            settings.supabase_db_url or settings.database_url or ""
        )
    return SupabaseRepository(
        settings.supabase_url or "",
        settings.supabase_publishable_key or "",
        secret_key=settings.supabase_secret_key or "",
        use_secret=settings.supabase_use_secret,
    )
