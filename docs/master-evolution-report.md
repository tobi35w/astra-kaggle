# Master evolution checkpoint — 6 October 2026

**Later data update (11 October 2026):** the archive was re-audited and curated for the active four-pair scope. Twenty-two disjoint windows passed all five timeframe checks; unmanifested invalid exports were removed. Use the [current curation report](historical-data-curation.md) and [readiness report](pre-mt5-readiness-report.md) for present status. The historical checkpoint and addenda below preserve what was known at their recorded times.

## Current recheck addendum — 11 October 2026

- Python is 3.14.6. The prescribed pytest coverage run completed with **332 passed, 1 skipped, 100% coverage**, and one `runpy` warning. Ruff lint and format checks, mypy, compileall, and `pip check` passed. `coverage.json` (2026-10-06) and `.codex-coverage.json` (2026-10-10) predate this run; the latest result is from the live test command.
- Supabase persistence is verified. A network-enabled read-only REST check passed; read-only PostgreSQL inspection passed all schema, constraint, index, RLS and trigger checks. `verify-database` appended one audit event, retried the identical record ID, and read back exactly one matching row (`b229a2a4-a2ea-4914-9ba1-26849ecbe445`). No migration ran, and migration-history state was not independently inspected. The restricted sandbox's initial DNS failures were environmental; its later network-enabled check succeeded.
- The local `astra-full-history` manifest contains **345 downloaded and 71 failed entries** (416 total) across AUDUSD, EURUSD, GBPUSD and USDJPY. The listed shards total **1.218 GiB / 12,695,038 M1 rows**. Checksums, row counts, JSON shape, OHLC values and within-shard ordering passed for every listed shard. There are 16 unmanifested JSON files: twelve structurally valid but without manifest checksums, and four truncated. The combined timeline has open-session gaps around failed periods. No canonical conversion, real-data backtest, walk-forward campaign, or ML training was run.
- Ruff formatting was applied to nine files; all 60 Python source/test files pass `ruff format --check`. A separate safety fix makes Astra's historical CSV writer persist `price_basis=bid`, carries the field through CSV loading and Journal imports, and rejects mixed/unsupported basis labels. Regression tests verify that the generated BID data is rejected by the current midpoint-only simulator. No BID backtest was run.
- The next gate is data completeness and a side-aware price model (or an explicitly supported midpoint dataset) before genuine research evaluation.

The 6 October checkpoint below is historical evidence and remains unchanged.

**Incomplete: blocked at Phase F, hosted persistence verification.** At the 6 October 2026 recheck, Supabase REST authentication and a read through Astra's repository succeeded. Direct PostgreSQL still fails DNS in the application environment; its hostname publishes an IPv6 address only, and a TCP probe to that address timed out. No hosted migration or write was executed. The requested phase order prevents proceeding to historical research, learning and AI implementation until the schema and Journal round-trip are verified. Use the project's actual Session pooler connection details for an IPv4 connection. Do not put credentials in chat or source code.

## A. Audited

Reviewed the existing configuration, CLI, indicators, market context, strategies/registry, arena, risk, execution, backtests, chronological validation, ML gates, persistence adapters, migrations, sandbox, historical BI5/JSON paths and documentation. Ran the entire existing suite before editing: 176 passed, 85% coverage, Ruff lint and mypy passed. Ruff formatting failed in the historical files. The project had no tracked source baseline; all project files were already untracked. No commit was created, and the working architecture and compatibility interfaces were retained.

## B. Fixed

- BI5 decoding now bounds decompression and rejects truncated, duplicate, reversed, malformed and oversized records. Its volume is provider-reported, not tick count.
- Unverified or changed raw caches are fetched again. Normalization requires matching manifest checksums. Corrupt manifests stop processing instead of silently discarding provenance.
- Quality reports expose every missing open-session interval, including requested-range edges. Higher-timeframe gap units are minutes. Weekend-crossing candles are flagged. A large gap threshold cannot conceal missing observations.
- Historical imports validate the full input snapshot before the first database write. Changed content gets different observation identities instead of silently retaining old candles under the same IDs.
- JSON chunks must cover their entire requested period. Integer timestamps are checked without truncating fractional input. The external CLI has a timeout and bounded output handling.
- Historical CLI failures/WARN reports exit nonzero. Offline dry-runs require an end date and create no data, directories or reports. Download date/worker settings are validated.
- Normal tests block external DNS/connections and do not read the local `.env`. Real integration tests require explicit opt-in. `.env.example` now contains placeholders only.

## C. Built

Added an injected HTTP transport, bounded latest-complete-day discovery, explicit selection of Dukascopy's alternate public entry route, request spacing, content-derived acquisition versions with retained manifest summaries, a checksum-verified historical loader, research quality gates, and an explicit `verify-database` command. The database verifier reads schema before writing one retained audit event through Journal, then checks read-back and idempotency. It applies no migrations or updates/deletes. Failed schema checks prevent writes.

## D. Historical data

**LIVE VERIFIED, bounded sample.** EURUSD on 2024-01-02 passed through the original HTTPS endpoint. A recent request to that host encountered a TLS handshake timeout. Dukascopy's own `https://www.dukascopy.com/datafeed` route returned the 2026-10-05 files for all seven requested pairs. It redirected to the public feed over HTTP; those responses are not TLS-authenticated after the redirect. The previous HTTP 503 was not reproduced on the known historical day, so its original cause is unproven. No provider substitution or full historical download was performed.

October 5 was the most recent fully completed UTC day at verification. Partial current-day data is excluded by discovery policy. Each of EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF and NZDUSD has:

| Timeframe | Candles per pair | Quality |
|---|---:|---|
| M1 | 1,440 | PASS |
| M5 | 288 | PASS |
| M15 | 96 | PASS |
| H1 | 24 | PASS |
| H4 | 6 | PASS |

Local evidence is under `data/astra-master-smoke/` and `data/astra-master-latest/`, including raw BI5 files, normalized M1, derived CSVs, versioned manifests and quality reports. These are ignored artifacts. The full 2018-to-latest archive is **not downloaded**. The JSON adapter is **unit tested**, not live verified in this continuation.

A later opt-in internet test of the known historical day through the `www` entry route **failed with TimeoutError**. It was not hidden or converted into a passing fixture. Source access remains intermittent; the successful downloads and a subsequent offline verification of all 35 datasets establish only the bounded evidence above, not reliable long-run acquisition.

## E. Database

**PARTIALLY VERIFIED; Phase F remains incomplete.** Earlier checks failed PostgreSQL authentication and REST DNS. The 6 October 2026 recheck supersedes those connection observations:

- The saved Codex MCP configuration and Astra's `.env` reference the same Supabase project. No Supabase MCP tools are exposed to the current chat, so authenticated MCP access is not verified.
- The REST hostname now resolves. An authenticated, read-only request to `astra_records` returned HTTP 200 and zero rows. A read through Astra's configured `SupabaseRepository` also succeeded.
- The REST OpenAPI description exposes all ten expected journal columns with compatible types. It does not establish the check constraints, indexes, RLS, append-only trigger, or migration history.
- The direct PostgreSQL hostname publishes an AAAA record and no A record. The application's resolver fails for that endpoint; a TCP probe to its IPv6 address timed out. `verify-database` at `2026-10-06T12:00:59.961309+00:00` returned `NOT_VERIFIED`, `DNS_FAILED`, `schema: FAILED`, `journal_round_trip: NOT_CHECKED`, and `migrations_applied: false`.

No hosted schema change or insert was executed. The remaining route is the actual Session pooler connection from Supabase's Connect dialog, followed by full schema inspection and the retained Journal audit round-trip. Do not invent a pooler hostname from the project's region. Read-only recheck evidence is in `data/astra-master-latest/reports/database-recheck-20261006.json`.

## F. Analysis

The data-to-analysis boundary is implemented and regression tested. `load_dataset(root, symbol, timeframe)` requires a successful report, matching content hash and complete requested range; it carries the source, BID price basis and canonical M1 hash into context. Direct NO TRADE decisions honor a failed quality status. Research CLI/dashboard execution rejects insufficient quality. The requested historical analysis campaign (Phase G) has not run because Phase F is blocked. This one-day sample also lacks the 30 H1/H4 bars needed for full regime analysis.

## G. Backtesting

Existing next-bar simulation, fill-price risk rechecks, costs, stop-first OHLC handling and P&L-unit tests are preserved. **No real-data backtest ran.** Before that phase, reconcile the simulator's midpoint-price assumption with Dukascopy's BID source. Backtests and paper replay now explicitly reject data labeled BID rather than silently treating those prices as midpoints. Generic legacy CSVs still require the operator to supply the correct price basis.

## H. Walk-forward / fronttesting

Existing chronological folds, training-only fitting, label purging and finite paper replay remain tested. No real-data walk-forward or paper experiment ran. There is no continuous market feed/fronttest service.

## I. Learning

Existing performance memory, heuristic fitness/degradation and gated ML remain in place. The requested source-separated BACKTEST/OOS/PAPER evidence ledger, granular learning events, reproducible research records and recovery-aware fitness loop have **not been implemented**. Existing memory can mix incompatible sources and drawdown units; it must be strengthened in its ordered phase before empirical adaptive use. ML remains disabled.

## J. Astra awareness

Existing structured `Decision.context`, health checks and NO TRADE reason codes are retained; dataset provenance is now attached. The requested full operational state and component-level HEALTHY/DEGRADED/FAILED model are not complete. No learning or AI authority was added to risk.

## K–L. AI infrastructure and OpenRouter

**Deferred, not implemented or activated.** No provider calls were made. The repository has no OpenRouter provider, request cache/budgets, AI audit lifecycle or external-information service. Configuration names in the master request remain planned requirements, not working features. See `ai_integration.md`. Proceed only after the earlier phases are healthy.

## M. Verification

Final verification: **206 passed, 1 opt-in live test skipped; 88% total coverage** (3,988 statements, 492 missed). The separately enabled live test was executed and **failed with TimeoutError**, as recorded in section D. It is not included in the offline pass count. Tests otherwise use explicitly synthetic fixtures; no synthetic results are represented as trading evidence.

Commands executed:

```powershell
.venv/Scripts/python.exe -m pytest --cov=astra --cov-report=term --cov-report=json:coverage.json
.venv/Scripts/ruff.exe check astra tests
.venv/Scripts/ruff.exe format --check astra tests
.venv/Scripts/python.exe -m mypy astra
.venv/Scripts/python.exe -m compileall -q astra tests
.venv/Scripts/python.exe -m pip check
git diff --check
```

Ruff lint/format passed for 58 Python files; mypy passed for all 43 source modules. Compilation and dependency checks passed. A separate scan of actual project files found no trailing whitespace or credential-pattern matches, and an AST import scan found no MT5/MetaTrader imports. This separate scan matters because `git diff --check` does not inspect the project's untracked files. Current safety settings were re-read without displaying credentials: trading=false, ML=false, daily loss=0.05, trade risk=0.005, drawdown=0.20, positions=3 and consecutive losses=5.

Baseline and final per-file audit metadata are retained locally under `.sandbox/master-audit/`. Thirty-nine source, test, documentation and configuration files were added or changed, including formatting repairs. The original compatibility tests remain in the passing suite.

## N. Remaining limitations

The master acceptance criteria are not all met. The direct PostgreSQL connection, full hosted schema inspection and write/read idempotency remain unverified; REST access now succeeds. The full historical archive, empirical analysis/backtesting/walk-forward, persistent learning expansion and AI infrastructure are deferred. One-symbol normalization currently materializes its requested window in memory; use bounded windows until streaming support exists. The weekend calendar does not automatically excuse holiday gaps. Hashes detect changed bytes but cannot recreate deleted old datasets; retain source artifacts for research reproducibility. Raw BI5 volume units remain `provider_raw` rather than an unverified conversion.

Trading and ML remain disabled in the checked local configuration. No MT5, broker credentials, live execution, autonomous code modification or AI trading path was introduced. Existing risk defaults were not raised.

## O. Exact next command

Set `SUPABASE_DB_URL` locally in `.env` using **Connect → Session pooler** for the existing project. Copy its actual hostname and username, retain the database password with appropriate URL encoding, and require SSL. The current `SUPABASE_URL` passed the REST checks and does not need correction based on this evidence.

From **Command Prompt** in the Astra folder:

```cmd
".venv\Scripts\python.exe" -m astra verify-database
```

From **PowerShell**:

```powershell
& ".\.venv\Scripts\python.exe" -m astra verify-database
```

If schema inspection then reports incompatibilities, reconcile the reported objects with `database/schema.sql`, `002_journal.sql` and `003_historical_m1.sql` before applying only the necessary migration. Do not blindly run the full setup script against an existing project.
