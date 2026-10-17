# Research data plan

Updated **16 October 2026** following the user's clarification: current four-pair model development uses the downloaded Hugging Face `keithtwesigye/OHLCV-1m-Forex` snapshot. This supersedes the recorded 15 October selection of the older curated archive. The September–12 October download target remains retired; recovering its 72 missing daily files is no longer an active requirement.

## Selected dataset

Use `data/future-phase/OHLCV-1m-Forex/`, downloaded on 15 October. Its [manifest](../data/future-phase/OHLCV-1m-Forex/manifest.json) records 44,024,447 cached M1 rows across 49 tickers. Current model development is limited to AUDUSD, EURUSD, GBPUSD and USDJPY.

| Pair | Cached rows | First source timestamp (UTC) | Last source timestamp (UTC) |
|---|---:|---|---|
| AUDUSD | 958,658 | 2023-01-01 22:31 | 2025-07-31 23:59 |
| EURUSD | 1,015,292 | 2023-01-01 22:04 | 2026-02-24 16:58 |
| GBPUSD | 960,890 | 2023-01-01 23:04 | 2025-07-31 23:59 |
| USDJPY | 1,016,143 | 2023-01-01 22:00 | 2026-02-24 16:58 |

The source does not declare candle-open/close convention, bid/ask basis, volume meaning or license metadata. Experiments retain the explicit assumption that timestamps are candle close/availability times. EURUSD and USDJPY have large internal gaps; no observations are filled, spliced or silently deduplicated. This cache cannot complete the retired September–October 2026 interval and is selected for historical classification development, not qualified trading evaluation.

## Use for continued research

1. Freeze each experiment's source/code identities, settings and comparison criteria before fitting, using a separate output identity. The [continuity comparison runner](../scripts/compare_hf_forex_continuity.py) follows the completed 15 October four-pair experiment documented in [HF experiment evidence](hf-forex-experiment.md).
2. Fix chronological training/validation boundaries before filtering continuous feature/label windows. Fit preprocessing and baseline controls only on training observations, exclude windows that cross missing minutes, and purge training labels available at or after validation begins.
3. Preserve explicit close/availability assumptions and label results as development and internal validation. The prior comparison has already been inspected; reused periods are not an untouched holdout. A new run is complete only when its saved outputs and Journal records are verified.
4. Before model qualification, separately verify and reserve a previously unused period with identified price and timestamp semantics. Assess risk-gated paper simulations at base and doubled costs, with consistent P&L units and fill-price risk rechecks. Classification accuracy alone does not establish net trading performance.

The source selection does not promote a model or relax the managed data, risk or paper-only execution boundaries.

The first continuation completed on 16 October: all four models were retrained with continuous feature/label windows at the unchanged validation boundaries. Its [frozen source selection and settings](../data/future-phase/OHLCV-1m-Forex/training/continuity/20261016T113345-01771fbe/plan.json), [verified results](hf-forex-experiment.md), and Journal evidence now pin this active experiment. USDJPY retained an accuracy gain over its baseline; the other three did not. No independent holdout or trading performance is established by these results.

## Preserved curated archive

The existing verified Dukascopy BID archive remains at `data/astra-training-four-pairs/`. Its recorded 15 October selection is historical evidence; it is no longer the selected source for the current HF model development. Keep its accepted windows separate.

| Pair | Complete windows | M1 candles |
|---|---:|---:|
| AUDUSD | 5 | 1,879,380 |
| EURUSD | 6 | 2,190,000 |
| GBPUSD | 6 | 2,067,540 |
| USDJPY | 5 | 1,786,980 |
| Total | 22 | 7,923,900 |

The available timeframes are M1, M5, M15, H1 and H4. The [curation guide](historical-data-curation.md) lists each window's dates. That selection included every accepted window and did not depend on its strategy return.

A fresh read-only verification on 15 October checked the SHA-256 and row counts of **all 110 selected output files**, as well as the curation's source-manifest identity. All matched their approved reports. The managed loader also rechecked requested-range quality, BID basis and integrity for **all 44 H1/H4 datasets**. No input file was changed.

The [superseded selection](../data/astra-research-data-plan-20261015/selection.json) pins every window, timeframe, path, row count and checksum. Its Journal record is `77ed1324-0121-46a9-98bc-be526e44cf37`, stored through `Journal`/`SQLiteRepository` in `data/astra-research-data-plan-20261015/plan.sqlite3`. These windows already contributed to earlier experiments and are not a newly untouched holdout.

## Retired acquisition and preserved evidence

The old `data/astra-continuous-2026-tail/` acquisition remains historically **INCOMPLETE: 72/144 required pair-days verified**. It is no longer the active data requirement, and further retries are not planned. Its original summary, manifest and raw files are retained unchanged.

The ForexSB, HistData and other source checks remain dated audit evidence. Their quality findings are preserved; this change does not fill missing prices or combine their records with the selected archive. See the [replacement-data audit](replacement-data-audit-20261015.md) and [acquisition reliability audit](historical-acquisition-reliability-audit.md).

The [superseded plan](../data/astra-research-data-plan-20261015/superseded-fresh-validation-plan.md) preserves the previous acquisition requirement and evidence-repair history.
