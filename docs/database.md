# Supabase and database setup

The Data API base is `/rest/v1/`. The adapter obtains the project URL and credentials from Settings/environment. `.env.example` contains placeholders only. No credentials are hard-coded, logged or committed.

An earlier network-enabled verification on 2026-10-11 passed. It confirmed read-only REST access, inspected the hosted PostgreSQL columns, constraints, indexes, RLS and append-only trigger, then appended one audit event twice and read back one matching record. Audit record ID: `b229a2a4-a2ea-4914-9ba1-26849ecbe445`. No migration ran during that verification; migration-history state was not independently inspected. The `003_historical_m1.sql` migration was applied through a separately recorded pooler operation on 2026-10-06 and the resulting entity constraint was verified. After an intervening timeout, a fresh verifier passed again on 2026-10-11 at 18:43 UTC; its audit record ID is `47fb7011-d8dd-49de-893b-f8c1dd2d671b`. The sanitized result is at `data/astra-post-audit-20261011/reports/database-verification-current.json`. Keep credential values out of chat.

An earlier failed check was an execution-environment timeout, not proof that the hosted Supabase project was down. The latest verification passed; rerun `verify-database` before depending on hosted persistence again.

The verifier reads existing columns, constraints, indexes, RLS and the append-only trigger before any write. Only a compatible schema permits one retained Journal audit record, inserted twice and read back to verify idempotency. It never applies DDL or updates/deletes existing records. State changes remain appended events. Failures return nonzero with a sanitized stage and category. `health` alone does not verify migrations or writes.

Inspect the existing schema before following the setup instructions below. The complete setup script is for new deployments; do not blindly run it against an existing project. An existing journal missing M1 may need only `003_historical_m1.sql`, after reconciliation with `002_journal.sql`. The recorded 2026-10-06 migration changed the timeframe constraint to admit M1; no migration ran during the current continuation.

## Local operation

Default `ASTRA_PERSISTENCE_BACKEND=sqlite` persists to ignored `astra.sqlite3`. `memory` is deterministic test storage. Both implement the same append/query/health interface as remote adapters. All journal entities are round-trip tested without production credentials.

## Hosted setup required

1. In the Supabase Dashboard, open **SQL Editor → New query**, paste all of `database/supabase_setup.sql`, and run it. This single transaction creates the legacy compatibility tables and the append-only journal in `public`, enables RLS, and denies `anon`/`authenticated` access. It creates no public policies. Alternatively, apply `database/schema.sql` followed by `database/migrations/002_journal.sql` through a trusted PostgreSQL migration workflow. Neither path has been applied to the hosted project by this build.
2. Fill the appropriate values locally in `.env`. Do not paste secrets into chat. For direct PostgreSQL, set `ASTRA_PERSISTENCE_BACKEND=postgres` and `SUPABASE_DB_URL`. For private server-side Data API persistence, set `ASTRA_PERSISTENCE_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, and explicitly set `ASTRA_SUPABASE_USE_SECRET=true`.
3. Run `python -m astra health` from the project environment to check the selected adapter. Subsequent local analysis/paper/research commands write their records through it.

`SUPABASE_PUBLISHABLE_KEY` is used for normal low-privilege access when secret mode is false. The private journal has RLS enabled and grants no anonymous/authenticated policy. Therefore the default publishable credential cannot write this backend journal and does not count as write-ready health. Do not solve a permission failure by adding a public-all policy. Only server-side private journal writes warrant elevated access. New `sb_*` credentials use the apikey header; legacy JWT project keys also use Authorization. Errors omit headers, DSNs and remote bodies.

## Schema and recovery

The new `astra_records` table stores a versioned object payload and typed entity envelope. Categories cover observations, snapshots, regimes, signals, evaluations, fitness, TRADE/NO TRADE decisions, risk results, orders, positions, executions, paper trades, P&L, research runs/metrics, walk-forward, robustness, models, confidence/calibration, health and audit. Indexes support context/time and correlation lookups. UUIDs make retries idempotent; batches are atomic. Timestamps distinguish domain availability (`occurred_at`) from database ingestion (`recorded_at`). Queries can restrict availability with `as_of`.

The migration enables RLS on legacy Astra tables too, revokes public/anon/authenticated access, and grants only SELECT/INSERT on the new journal to Supabase's service_role when that role exists. Update/delete triggers protect append-only history; administrative owners can still deliberately alter database policy. Legacy tables and adapters are retained, not automatically copied into the journal.

PostgreSQL tests use a fake connection/cursor; Supabase tests use an injected deterministic transport; SQLite tests reopen actual local databases. Hosted schema and idempotent write/read behavior were directly verified on 2026-10-11, and the M1 migration was directly applied and inspected on 2026-10-06. The latest connectivity check passed. There is no cross-process book locking or automatic crash reconciliation beyond explicit checkpoints and the event journal.
