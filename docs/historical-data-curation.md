# Four-pair historical data curation

## Active research universe

The research universe is AUDUSD, EURUSD, GBPUSD, and USDJPY. This guide records the verified curated archive in `data/astra-training-four-pairs/`; `curation.json` records the source manifest hash, selected windows, chunk hashes, exclusions, timeframes, and row totals. Each window is a separate loadable dataset. Never concatenate windows across gaps. As clarified on **16 October 2026**, current model development uses the downloaded HF OHLCV snapshot under the [active data plan](fresh-validation-plan.md); this curated archive remains available for its existing research workflows.

Only uninterrupted windows of at least six consecutive calendar months were considered large enough for this initial training dataset. All source chunks were checked against their acquisition-manifest SHA-256. Candidate windows were converted from source M1 and independently passed the managed loader for M1, M5, M15, H1, and H4 before acceptance.

| Pair | Accepted complete windows (inclusive dates) |
|---|---|
| AUDUSD | 2018-01-01–2018-11-30; 2019-01-01–2019-11-30; 2021-05-01–2021-10-31; 2022-01-01–2023-10-31; 2024-01-01–2024-10-31 |
| EURUSD | 2018-01-01–2018-11-30; 2020-05-01–2020-11-30; 2022-02-01–2023-11-30; 2024-01-01–2024-11-30; 2025-01-01–2025-11-30; 2026-01-01–2026-08-31 |
| GBPUSD | 2018-01-01–2018-11-30; 2019-05-01–2019-11-30; 2021-02-01–2022-11-30; 2023-06-01–2023-11-30; 2024-01-01–2024-09-30; 2025-01-01–2025-11-30 |
| USDJPY | 2018-01-01–2018-11-30; 2019-01-01–2019-11-30; 2021-04-01–2021-10-31; 2022-05-01–2023-11-30; 2024-01-01–2024-09-30 |

These are 22 disjoint windows with **7,923,900 M1 bars** in total. Every accepted window passed each listed timeframe's quality gate. The derived bar counts are:

| Timeframe | Bars |
|---|---:|
| M1 | 7,923,900 |
| M5 | 1,584,780 |
| M15 | 528,260 |
| H1 | 132,065 |
| H4 | 31,918 |

The derived frames use UTC-aligned OHLCV aggregation of M1 candles. A target bar is emitted only when all of its M1 constituents are present. Partial boundary bars are omitted, and no bars are filled or interpolated. Timestamps are candle-close/availability times. Weekend closures and session boundaries follow Astra's documented generic FX calendar; the validator does not have a broker-specific holiday or early-close calendar.

## Exclusions and cleanup

- The original acquisition manifest has 345 successful monthly chunks and 71 failed chunks: AUDUSD 13 failures, EURUSD 18, GBPUSD 18, and USDJPY 22. Failed months have no accepted training data.
- Thirty-three intact but shorter continuous runs (under six months) are recorded and excluded from the curated training root by the minimum-window policy. They remain in the raw acquisition archive for recovery and audit, but are not in the approved training set.
- AUDUSD November 2024 was withheld: the combined M1 window reported one open-session gap at the final Friday minute (2024-11-29 22:00 UTC) under the generic FX calendar. M5, M15, and H1 also warned; H4 passed. The underlying reason could be an early close, but that has not been verified. January–October 2024 passed and is retained separately.
- Sixteen JSON files outside the acquisition manifest were audited. Twelve parsed but failed complete open-session coverage checks; four were truncated JSON. All sixteen were removed from the raw export directory. Their paths, SHA-256 hashes, and quality findings remain in `data/astra-training-four-pairs/unmanifested-source-audit.json`.
- The source archive and its manifest remain available for provenance and recovery. This curated root retains its verified managed-dataset status; selecting the HF cache for classification development does not confer that status on the HF source.

## Use and limits

For example, to load EURUSD H4 from the validated 2022–2023 window, use:

```powershell
.venv/Scripts/python.exe -m astra readiness `
  --dataset-root data/astra-training-four-pairs/EURUSD/2022-02-01_to-2023-11-30 `
  --pair EURUSD --timeframe H4
```

These curated windows were subsequently used by the completed historical research campaigns documented in [project status](project-status.md). The recorded 15 October selection was superseded on 16 October by the user's clarification to use the downloaded HF dataset. The 15 October verification remains preserved: all 110 output-file hashes and row counts matched, and all 44 H1/H4 datasets passed the managed loader again. They are previously used history, not a newly untouched holdout. Prices are Dukascopy BID; local checksums and structural validation do not establish ASK history or profitability. Keep the disjoint windows separate, split chronologically inside each window, and purge labels at every validation boundary.

## qleap HF snapshot audit (2026-10-13)

A 1.4 GB HuggingFace snapshot (`qleap/fx_majors_tick_ohlcv`, 924 parquet files) was audited offline before any training or evaluation use. Findings, with the machine-readable report at `data/future-phase/qleap-fx-majors-tick-ohlcv/astra-audit.json`:

- **Structure is clean**: 588 files across the four active pairs (~8.6 M USDJPY rows back to 2003-05; ~2.37 M rows each for the others from 2020-04-01 to 2026-08-31). No out-of-order timestamps, no duplicate timestamps within or across month files, no impossible OHLC, no non-positive prices, no negative volumes, and zero closed-session bars.
- **Timestamps are naive UTC candle-close minutes**, matching Astra's semantics (a month file's first bar closes at 00:01; the 2026-08 file starts at the 2026-08-02 22:01 weekly open).
- **It is a different extraction of the same claimed source, not equivalent data.** A February-2022 EURUSD overlap check against the curated window found identical timestamps but different prices on 28,689 of 28,689 common bars (worst absolute difference 4.35 pips) and non-proportionally different volumes.
- **Weekly session edges differ from Astra's DST-aware calendar in summer weeks.** In EST weeks both sources agree (Sunday 22:00 UTC open, Friday 22:00 UTC close). In EDT weeks qleap opens Sunday 22:00 UTC where the curated calendar expects 21:00 UTC, producing systematic 60-minute Sunday-edge gaps (~250 per pair at M5).
- **Scattered multi-minute holes**: EURUSD M5 shows 6,287 suspicious gaps, dominated by 4,691 gaps of exactly 5 minutes (the export drops no-tick minutes in 5-minute runs).

**Verdict**: the snapshot does not meet Astra's PASS quality bar and was not admitted to any evaluation root. Converted output (kept under `data/astra-qleap-source/`, git-ignored) reports `DATA QUALITY INSUFFICIENT` and the managed loader refuses it, as intended. The bridge (`astra/historical/qleap_source.py`, CLI `astra historical process-parquet`) exists for future re-evaluation if a verified continuous run ever justifies treating it as a separate source. Do not stitch qleap output onto curated windows inside one dataset.

## Retired continuous tail acquisition (2026-09-01 onward)

The earlier plan chose the September–October tail for new evaluation evidence. That specific acquisition was retired at the user's request on 15 October; the [active data plan](fresh-validation-plan.md), clarified on 16 October, uses the downloaded HF OHLCV snapshot for model development. EURUSD's latest curated window ends on 2026-08-31; the other pairs' accepted windows end earlier as listed in the table. Previously consumed experiment dates must be checked directly before calling any different period an independent holdout. The old downloader and cached files remain available as historical work, with no further retry planned.

**First run outcome (2026-10-13)**: after 1 h 07 m the run reported FAIL for all four pairs — Dukascopy refused most daily requests (TransientSourceError) despite backoff; only ~34 of 172 day-files were verified and cached. Re-running the same command retries only the refused days. Until a re-run reports PASS for all active pairs and timeframes in `data/astra-continuous-2026-tail/acquisition-summary.json`, the tail is not approved for evaluation use. Note that unthrottled latest-day probing (as in `astra historical latest`) is itself rate-limited; always pass an explicit `--end`.
