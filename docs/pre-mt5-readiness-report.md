# Astra pre-MT5 readiness handover — 11 October 2026

## Decision

**Not ready to begin MT5 demo integration.** The repository remains research and paper only, with trading disabled by default. The code boundary has no MT5 adapter, broker credentials, live order route, or live mode. The evidence supports continuing offline research, not integration or deployment. Profitability is unproven.

## Gate status after post-audit recovery

| Gate | Status | Evidence |
|---|---|---|
| Offline automated suite | `PASS — TESTED LOCALLY` | 347 passed, 1 existing skip, 1 separately marked socket test deselected; 96% coverage. |
| Sandbox route behavior | `PASS — TESTED LOCALLY` | HTTP handler rules run against in-process request doubles. |
| Real socket integration | `PASS — TESTED LOCALLY` | Explicit `-m socket` loopback HTTP smoke test passed on this run (1 passed). |
| Python compilation | `PASS — TESTED LOCALLY` | Python 3.14.6 `compileall` exited 0; interpreter still prints the stale-base-path warning described below. |
| Ruff, mypy, dependency check | `PASS — TESTED LOCALLY` | Current checks pass after the route-test refactor. |
| Local SQLite journal | `PASS — TESTED LOCALLY` | Campaign round trip and repository entity checks completed locally. |
| Hosted schema and Journal round trip | `PASS — VERIFIED` | Fresh verification at 2026-10-11 19:01:21 UTC; audit event `0115084b-a213-4574-8a5c-d0d62fb2d395`; sanitized result saved at `data/astra-post-audit-20261011/reports/database-verification-current.json`. The verifier applied no migration; migration history was not inspected. |
| Four-pair curated historical data | `PASS — VERIFIED` for selected windows | 22 uninterrupted windows (six or more consecutive months) across AUDUSD, EURUSD, GBPUSD, and USDJPY passed M1/M5/M15/H1/H4 validation. The continuous archive still has failed months and gaps; see `docs/historical-data-curation.md`. |
| Full four-pair 2018-to-latest archive | `BLOCKED` | The source manifest has 345 successful and 71 failed monthly chunks. The current-day provider check failed; only validated uninterrupted windows are approved for research. |
| Historical backtest and walk-forward | `PASS — VERIFIED` as completed evaluations | Results are adverse/inconclusive and sample-limited; details below. |
| Historical paper replay | `PASS — VERIFIED` | Finite offline replay, 25 trades, no open positions at completion. |
| Genuine forward paper testing | `NOT RUN` | No continuously updated feed or forward paper observation is recorded. |
| ML training and profitability | `NOT RUN` / unproven | ML remains disabled and no empirical model was trained. Profitability is `UNPROVEN`. |

The machine-readable readiness command now reports each of these separately. A successful database check or segmented historical dataset does not clear the continuous-archive or forward-paper gates.

## Verified work

- **Bid/ask execution and risk:** long entries and short exits use ASK-side pricing for BID data; pending orders are risk-checked at their assumed fill price. Spread accounting, stop behavior, pending orders, and P&L unit handling have targeted tests.
- **Curated four-pair data:** `data/astra-training-four-pairs/curation.json` records 22 validated, uninterrupted windows and the per-timeframe totals. Each window has source BID M1 plus M5/M15/H1/H4 resampled only from complete constituent minutes. Every accepted segment passed the managed loader for all five timeframes. Exact date windows and exclusions are in `docs/historical-data-curation.md`. This is prepared data, not multi-pair strategy or ML training evidence.
- **Prior EURUSD evaluation:** Dukascopy-node EURUSD BID M1 data covers 1 February 2022 through 30 November 2023. The validated set has 688,320 unique M1 bars and derived M5/M15/H1/H4 data. See `data/astra-verified-window/reports/eurusd-dukascopy-json.json`; the H4 dataset version is `astra-sha256:d77f…b047a`.
- **Research campaign:** `data/astra-research-paper-20261011/campaign-067dfc3cc7bd2118.json` completed with backtest, chronological holdout, six walk-forward folds, cost stress, robustness assessment, and finite paper replay. Backtest: 25 trades, +0.70% net return. Holdout: 13 trades, -2.10%, expectancy -16.12 account-currency units per trade. Walk-forward: 34 trades across separate fold accounts, one positive and five negative. Robustness is `INSUFFICIENT_DATA`. These results are weak and not a deployment case.
- **Paper persistence:** SQLite round-trip was inspected after the campaign. It contains 25 paper trades; paper performance, backtest, OOS, walk-forward and robustness records are source-separated and timestamped. Paper replay finished with no open positions.
- **Hosted persistence evidence:** an earlier Supabase verification passed on 11 October (audit `b229a2a4-a2ea-4914-9ba1-26849ecbe445`). The later timeout was superseded by the fresh successful post-audit verification recorded above.
- **AI scaffold:** optional advisory provider is disabled by default and uses injected transport tests. No key was read, no external AI call was made, and no trade authority is delegated to it.

## Blockers and limitations

- The full continuous four-pair archive from 2018 to latest is not complete. The source manifest has 345 successful exports and 71 failures across AUDUSD (82/95), EURUSD (89/107), GBPUSD (89/107), and USDJPY (85/107). Curated windows deliberately stop at gaps; their date ranges vary by pair. Short windows and incomplete segments are excluded from the training root.
- Latest-data acquisition remains unverified: the 2026-10-11 `historical latest --symbol EURUSD` command exited with `TransientSourceError`, after the bounded BI5 endpoints timed out before headers and the one-day Dukascopy-node attempt recorded `RuntimeError`. The CLI sanitizes the cause and latest-day discovery does not persist a report. Do not treat the downloader as currently reliable based on the older bounded acquisition.
- **Post-audit database:** the fresh verification passed schema inspection and the idempotent Journal round trip at 2026-10-11 19:01:21 UTC (audit `0115084b-a213-4574-8a5c-d0d62fb2d395`). No migration was applied and migration history remains uninspected.
- **Post-audit test suite:** sandbox route and request-validation tests run in-process and preserve status, security-header, content-type, body-size, token and sanitized-error assertions. Full suite: `347 passed, 1 skipped, 1 deselected`, exit 0, 96% coverage. The existing skip is the opt-in live integration test. The explicit loopback smoke test passed (`1 passed` on this run). One existing `runpy` import-order warning remains. A deterministic interrupted-chunk test verifies the node downloader records failure, resumes successfully, then uses the validated cache.
- **Python environment:** `.venv/Scripts/python.exe` runs Python 3.14.6, and its `sys._base_executable` exists at `C:\Python314\python.exe`. `pyvenv.cfg` also points there. The launcher emits `Failed to find real location of C:\Python314\python.exe` but compilation exits successfully. The installed Python 3.12 WindowsApps executable could not be launched (`Access is denied`), so the venv was not deleted or rebuilt.
- **Dukascopy diagnostic:** bounded HTTPS probes to both approved BI5 base URLs timed out before receiving response headers (therefore no HTTP status or content type was available). A one-day EURUSD Dukascopy-node 1.50.0 attempt for 2026-10-09 failed with a sanitized `RuntimeError`, zero rows, and no output file; see `data/astra-post-audit-20261011/historical-sample/manifests/dukascopy_manifest.json`. The 2026-10-11 `historical latest --symbol EURUSD` retry returned `TransientSourceError`; latest-day discovery does not persist a report and its CLI hides the underlying fetch error. These attempts do not establish a provider HTTP response code.
- **Source cleanup:** all 16 raw JSON files missing from the archive manifest were removed after audit. Twelve parsed but failed continuous open-session/range checks (including unmodeled year-end or weekend-edge gaps); four were truncated JSON. Their hash-pinned paths and diagnostic results are retained in `data/astra-training-four-pairs/unmanifested-source-audit.json`. No manifest-listed raw source was deleted.
- **Provider implementation audit:** direct BI5 URLs use Dukascopy’s lowercase zero-based month path and uppercase symbols. HTTP 404 is classified as a missing day; 408/425/429/5xx and socket errors are transient; other HTTP errors are permanent. The BI5 parser bounds compressed/uncompressed size, validates the 24-byte big-endian LZMA records, price scale, UTC candle-open-to-close conversion and OHLC. The daily downloader has four retries with exponential jitter, two workers, global 0.5-second pacing, checksum-verified cache reads and atomic `.part` replacement. Dukascopy-node uses single-worker/month chunks, CLI retry flags, validated JSON, manifest checksums and atomic temp directories. Its sanitized failure currently retains only `RuntimeError`, not npm/HTTP details; that diagnostic limitation remains.
- **Research-method audit:** the saved campaign request contains one preselected `ema_trend` strategy and no parameter sweep. The holdout was evaluated after the full-history backtest and was not used for tuning in this campaign; treat that holdout as consumed and do not reuse it as independent validation. Walk-forward tests are chronological and their test folds are disjoint; training uses only the preceding window. Signals at a candle close are eligible for execution from the next bar. Prefix/future-mutation tests passed. Fixed costs were 1 pip assumed spread, 0.2 pip slippage and zero commission. OHLC cannot reconstruct tick order or variable historical spreads.
- **Learning and recovery:** the campaign’s BACKTEST, OOS and PAPER records remain in separate units and source streams; the SQLite journal was inspected. Tests cover append/idempotency, causal availability, portfolio checkpoint restoration without order re-execution, risk rechecks, and risk veto independent of strategy output. Arena fitness scales linearly to full weight at 30 trades and labels fewer than 10 as insufficient; degradation requires 20 observations. A new regression test verifies a one-trade evaluation receives 1/30 fitness weight and is reproducible. These are safeguards, not proof of learned edge.
- The campaign uses a fixed assumed spread/slippage, zero commission, OHLC execution assumptions, and one instrument unit for paper. It cannot validate historical variable spread, tick paths, queueing, financing, latency, liquidity, or broker-specific margin and fills.
- ML has not been trained on verified empirical labels. The campaign is not training evidence. There is no claim of strategy profitability.

## Local checks

- Ruff lint: passed (`.venv/Scripts/ruff.exe check astra tests`).
- Ruff formatting: passed (`.venv/Scripts/ruff.exe format --check astra tests`, 68 files).
- Mypy: passed (49 source files).
- Dependency consistency: passed (`pip check`).
- Full tests: `347 passed, 1 skipped, 1 deselected` (exit 0, 96% coverage); explicit socket smoke test: `1 passed`.
- `compileall`: exit code 0 under the active 3.14.6 venv, with a launcher warning from its base-executable path metadata.

The readiness CLI can emit an evidence snapshot without contacting remote persistence. To check the reference EURUSD window from the current curated root:

```powershell
.venv/Scripts/python.exe -m astra readiness `
  --dataset-root data/astra-training-four-pairs/EURUSD/2022-02-01_to-2023-11-30 `
  --pair EURUSD --timeframe H4 `
  --campaign-report data/astra-research-paper-20261011/campaign-067dfc3cc7bd2118.json
```

With the saved hosted report included, run:

```powershell
.venv/Scripts/python.exe -m astra readiness `
  --dataset-root data/astra-training-four-pairs/EURUSD/2022-02-01_to-2023-11-30 `
  --pair EURUSD --timeframe H4 `
  --campaign-report data/astra-research-paper-20261011/campaign-067dfc3cc7bd2118.json `
  --database-report data/astra-post-audit-20261011/reports/database-verification-current.json
```

It correctly remains `NOT_READY_FOR_MT5_DEMO_INTEGRATION` because the archive and genuine forward-paper gates are open. The report always labels profitability `UNPROVEN`.

## Next evidence needed

1. The offline suite and socket smoke test have passed locally. Keep these checks in routine pre-change verification.
2. Restore provider reliability and fill failed ranges for the four active pairs. Keep each new interval separate until complete M1 and derived-timeframe checks pass.
3. The current hosted database gate is closed by the saved 19:01 UTC check; repeat it before relying on the service again. It will append a new uniquely identified audit record.
4. Record genuine forward paper results on a live-updating feed and evaluate adequate unseen data. Keep ML disabled until labels and leakage-safe out-of-sample validation are defensible.
5. Reconsider demo integration only after these gates and current adverse research results receive a fresh review. Keep live trading disabled.
