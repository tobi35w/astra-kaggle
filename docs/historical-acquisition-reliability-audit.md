# Historical acquisition reliability audit

**Audit date:** 2026-10-14  
**Dataset:** `data/astra-continuous-2026-tail`  
**Required tail:** 2026-09-01 through 2026-10-12, AUDUSD/EURUSD/GBPUSD/USDJPY, Dukascopy BID M1 source with derived M1/M5/M15/H1/H4 outputs.

## Finding

The tail is **INCOMPLETE**. Read-only reconciliation finds 144 pair-days with open FX sessions: 72 have verified raw M1 coverage and 72 are missing. The manifest has 168 day-pair records overall (85 marked downloaded, 83 failed); those totals include 24 fully closed pair-days, so they are not a valid completeness measure by themselves.

All 85 files claimed as downloaded were present in the expected raw directory, matched their recorded SHA-256 and row counts, decoded as BI5, and passed session-coverage checks where a session was open. There were no corrupt, untracked, or partial raw BI5 files. Thirteen downloaded files fall on fully closed dates and do not contribute to required open-session coverage. No normalized or derived timeframe files currently exist for this tail.

The acquisition report for the latest narrow retry records one EURUSD request for 2026-10-12 and HTTP 503. The manifest retains the earlier broader unit records, but its top-level range fields represent the latest dataset update. The primary historical cause remains provider-side HTTP 429/503 refusal despite conservative single-worker pacing; the available evidence does not establish a local file-integrity defect or justify bypassing provider controls.

HistData remains separate and unimported. Its audited archives end on 2026-10-09, use fixed UTC-5 timestamps and zero M1 volume, and cannot fill the missing October 12 session or be blended into the Dukascopy stream.

## Changes made

- Added append-only acquisition run scopes and outcomes to the manifest, plus immutable per-run report files. Narrow retries no longer obscure the historical run scope.
- Added explicit read-timeout, per-file elapsed-time, retry-jitter, and consecutive-failure limits. Retries remain finite and honor `Retry-After`; HTTP 429 stops new provider requests, and repeated failures open a circuit. No traffic-masking or restriction-bypass behavior was added.
- Changed the downloader to submit only a bounded number of jobs, record unsubmitted work as deferred, and preserve unit progress on interruption. Staged files are reread, hashed, decoded, and atomically moved only after validation.
- Added read-only manifest/file reconciliation for expected open sessions, checksums, row counts, file sizes, claimed paths, partial files, and untracked raw files. Dataset and runbook completion now requires verified coverage and passing required timeframes; a one-pair run cannot certify the four-pair tail.
- No database schema change or migration is required. HistData remains untouched; trading, risk limits, and execution settings are unchanged.

## Verification and next action

Offline suite: **430 passed, 1 skipped, 5 deselected; 93% coverage**. Ruff: **all checks passed**. The initial full-suite attempt had two Windows Temp permission failures; the successful rerun redirected `TEMP` and `TMP` to the workspace.

No manual action is needed to repair local files. The remaining action is to retry the exact tail acquisition only when Dukascopy accepts requests again. The pipeline resumes from checksum-verified cache, records any further refusals, and will remain **INCOMPLETE** until every required open session and derived timeframe passes validation. No live download was run as part of this audit/recovery change.
