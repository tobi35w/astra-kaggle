# Replacement FX data: executed audit, 15 October 2026

**Later scope decisions:** on 15 October the user retired the requirement to recover the 72 missing daily files. On **16 October 2026** the user clarified that the [active research data plan](fresh-validation-plan.md) should use the downloaded HF `keithtwesigye/OHLCV-1m-Forex` snapshot for four-pair model development, superseding the recorded curated-archive selection. The findings below remain an audit of the retired acquisition target; ForexSB coverage and admission results are unchanged.

**Result: a fresh four-pair archive was downloaded, but no complete replacement is approved.** ForexSB supplies observations on every requested trading date, including 12 October. Its M1 series omits 906 open-session minutes across the four pairs. The original Dukascopy acquisition remains **72/144 required pair-days verified**, with 72 pair-days unresolved.

The fixed scope is AUDUSD, EURUSD, GBPUSD and USDJPY, from source opens on **1 September through 12 October 2026 inclusive**. Coverage and derivation use Astra's existing New York 17:00 weekly FX calendar. Missing minutes are not filled, different source representations are not combined, and no model training or market-data evaluation was run.

## Downloaded ForexSB evidence

The [public ForexSB application](https://data.forexsb.com/) publishes the metadata and archive URLs in its own JavaScript. All four archive requests returned HTTP 200 on 15 October. Each compressed archive contains 200,000 ordered M1 records and reaches **15 October 2026 at 10:59 UTC source-open time**. All four passed gzip integrity, record-layout, timestamp uniqueness/order, OHLC, nonnegative volume and nonnegative spread checks.

The provider metadata identifies Dukascopy. A [publisher explanation](https://forexsb.com/forum/topic/9014/brokerdata-or-dukacopy-date/) describes composing its bars from Dukascopy ticks; that explanation is from 2022 and does not independently authenticate today's feed. This distribution is a separate representation of the same attributed underlying source, not independent price confirmation.

| Pair | M1 rows in requested window | Missing open-session minutes | Complete open-session dates | Exact OHLC matches / overlapping minutes |
|---|---:|---:|---:|---:|
| AUDUSD | 42,866 | 334 | 5 / 36 | 31,258 / 31,258 |
| EURUSD | 42,962 | 238 | 5 / 36 | 24,166 / 24,166 |
| GBPUSD | 43,001 | 199 | 5 / 36 | 15,040 / 15,042 |
| USDJPY | 43,065 | 135 | 9 / 36 | 15,781 / 15,781 |
| Total | 171,894 | 906 | 24 / 144 | 86,245 / 86,247 |

Each pair has observations on all 36 expected open-session dates. There should be 43,200 open-session minutes per pair under the existing calendar. Date presence therefore does not establish complete daily coverage. ForexSB has observations on all 72 originally missing pair-days, but only 13 of those alternative pair-days have all expected minutes; none were spliced into the original archive.

The two GBPUSD discrepancies occur at close times **2 September 12:00 UTC** and **2 September 15:08 UTC**. One closing price differs by 1.2 pips; the other row differs in its open/high. Exact prices are saved in the audit. A close match over the available overlap supports BID consistency there, but does not prove equivalence on dates without reference data or justify treating the volume fields as interchangeable.

Where original data exists, the 513 minutes omitted by ForexSB correspond to flat, zero-volume original candles. The original archive does not supply reference observations for the other 393 missing minutes. The audit records this distinction and does not infer or fabricate those prices.

## Timestamp, volume and derivation rules

The observed archive layout has seven little-endian signed 32-bit fields per row: minutes since 1 January 2000 UTC, open, high, low, close, volume and spread. Price scales come from the saved metadata and must match the requested symbol: 100,000 for AUDUSD/EURUSD/GBPUSD and 1,000 for USDJPY. The decoder rejects older or unsupported layouts instead of guessing.

Source timestamps represent minute opens. The audit adds exactly one minute to obtain candle close/availability time. It selects source-open `[2026-09-01T00:00Z, 2026-10-13T00:00Z)` before comparison and resampling; later observations in the archive do not enter the requested window.

Raw volume integers are retained as provider fields with unverified units. The website's zero-to-one substitution and rounded volume scaling are not applied. Raw spread points are checked for nonnegative values and are not treated as a historical execution-cost model.

Astra's existing causal resampler was executed in memory. It emits a target candle only when every constituent M1 minute exists; no source-supplied higher timeframe was substituted.

| Pair | M5 bars | M15 bars | H1 bars | H4 bars |
|---|---:|---:|---:|---:|
| AUDUSD | 8,408 | 2,718 | 633 | 142 |
| EURUSD | 8,475 | 2,754 | 643 | 142 |
| GBPUSD | 8,519 | 2,788 | 651 | 144 |
| USDJPY | 8,575 | 2,832 | 687 | 153 |

**All 20 pair/timeframe checks returned WARN for incomplete requested coverage.** There is no managed dataset manifest, imported replacement, qualified fresh holdout or promoted model from this work.

## Other availability checks executed

| Source | Executed check | Result and limit |
|---|---|---|
| HistData | Downloaded the current EURUSD October M1 ZIP through its public download form and verified its CRC/hash | 9,626 rows; ends 9 October at 16:58 fixed EST. Its 88,632 bytes and SHA-256 match the earlier archive. No October 12 data in this file. |
| FXCM | Requested the public EURUSD M1 weekly files for 2026 weeks 41 and 42 | Both HTTP 404. These specific files cannot complete the required four-pair window. |
| Dukascopy primary host | One request for EURUSD, 12 October, with finite connect/read timeouts and no retry | Failed with `URLError`; no archive was received. This request did execute. |
| Dukascopy documented main-site host | One request for the same public daily BID archive | HTTP 503; no archive was received. No repeated requests were launched. |
| qleap Hugging Face OHLCV | Fetched the current EURUSD M1 file listing | 77 monthly files, ending `2026_08.parquet`; September and October are absent from that listing. |

The preceding Hugging Face model experiment used a different cached source ending in July 2025 or February 2026 by pair. It cannot provide the requested 2026 tail.

Other researched leads did not produce a verified replacement: MarketData Hub downloads the same Dukascopy archive; GetData describes mid-price CFD quotes; TrueFX requires account access; Adona's documented historical API is paid, with a limited recent-data trial. No registration, subscription, broker connection or purchase was performed.

## Reproduce and inspect

Saved source files, HTTP receipts and the research Journal are under `data/astra-alternative-source-research-20261015/`. Raw archives and prior acquisition/experiment artifacts remain intact.

The reusable offline auditor is [scripts/audit_forexsb_source.py](../scripts/audit_forexsb_source.py). It requires explicit paths and dates, refuses output inside the reference dataset, rejects malformed or conflicting records, verifies reference SHA-256 and row counts, rechecks its inputs, and writes a unique audit directory. Audit records go through `Journal`/`SQLiteRepository` with readback verification. A successful process exit means the audit executed, not that the dataset was admitted.

```powershell
.venv/Scripts/python.exe scripts/audit_forexsb_source.py --source-root data/astra-alternative-source-research-20261015 --reference-root data/astra-continuous-2026-tail --output-root data/astra-alternative-source-research-20261015/audits --start 2026-09-01 --end 2026-10-12
```

The executed run is [20261015T121850-6f8e5171/audit.json](../data/astra-alternative-source-research-20261015/audits/20261015T121850-6f8e5171/audit.json), with its own `audit.sqlite3`. It includes source/input checksums, every missing minute, per-day coverage, all timeframe reports, exact mismatch examples and verified reference-file identities. Its audit record ID is `526ae84b-cc6e-4226-8e56-27a0a6ce24da`.

The 20 focused tests use synthetic fixtures in isolated temporary directories. They verify close timestamps, price scales, retained zero/raw volume, corrupt/truncated archives, duplicates/conflicts, ordering, bounded decompression, causal clipping, missing-bar rejection, reference tampering, source preservation, separate outputs and Journal readback. Full-suite verification evidence is stored separately under `data/astra-replacement-verification-20261015/`.

Full offline verification passed **514 tests, 1 skipped, 8 socket tests deselected; 93.224842% Astra coverage**. Ruff lint and formatting passed for `astra`, `tests` and the new audit script. SHA-256 comparison before and after the suite confirmed that **all 105 protected evidence files were unchanged**, including the original acquisition summary/manifest/raw files, the new source archives and audit, the original EURUSD experiment report and the completed four-model comparison artifacts. The verification directory retains pytest logs, JUnit, coverage and the integrity receipt.

## Remaining acquisition gate

Keep this snapshot as an audited candidate source. Complete BID M1 coverage is still required before all derived timeframes can pass. The original acquisition can resume from verified cache when its provider accepts bounded requests again; another source must independently pass the same coverage and provenance checks. This search did not relax those requirements or declare the 72 missing original downloads recovered.
