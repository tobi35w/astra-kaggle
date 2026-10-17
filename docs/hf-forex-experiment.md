# Hugging Face Forex model experiment

## Continuity retraining — 16 October 2026

At the user's clarification, the downloaded HF snapshot is the active source for this model-development phase. Run **20261016T113345-01771fbe** completed at **12:37 Africa/Lagos** on 16 October. It used the same latest 400,000 source rows per pair and the same original 70/30 chronological boundaries as the earlier experiment, then excluded incomplete feature/label windows from both fitting and scoring.

| Pair | Retained training examples | Retained validation examples | Model accuracy | Training-majority baseline | Gain, percentage points |
|---|---:|---:|---:|---:|---:|
| AUDUSD | 266,805 | 114,873 | 51.0311% | 51.0407% | -0.0096 |
| EURUSD | 271,837 | 116,392 | 51.1281% | 51.1315% | -0.0034 |
| GBPUSD | 269,949 | 115,380 | 50.6613% | 50.6873% | -0.0260 |
| USDJPY | 271,746 | 116,500 | 51.1356% | 50.0961% | +1.0395 |

USDJPY remains the clearest research lead. Its accuracy increased **0.1373 percentage points** over the original model on exactly the same continuous validation subset. The other three changes were -0.0270, -0.0180 and -0.0043 points respectively. All four models have slightly lower Brier scores than their training-frequency constants; USDJPY's score is 0.24985453 versus 0.25001799. These small probability-score gains do not establish profitable trading.

USDJPY beats its fixed training-majority baseline in June and July 2025 and January and February 2026, but trails it by 1.0068 points in the small May 2025 subset (2,483 examples). There are no intervening August–December validation observations to infer performance from. This is reused development evidence with dependent overlapping labels, not independent confirmation or a significance test.

The split is fixed **before** filtering. Each retained example requires all 15 feature bars, including ATR's preceding close, and the next three label bars to be consecutive M1 observations inside the assumed New York FX session. Each retained label therefore covers exactly three elapsed minutes. Three training examples per pair are separately purged at the original validation boundary. Continuity filtering then excludes 13,180/8,148/10,036/8,239 training examples and 5,122/3,603/4,615/3,495 validation examples in AUDUSD/EURUSD/GBPUSD/USDJPY order. No candles are filled, removed from the source sequence, or joined across gaps.

Scaling, logistic-regression fitting, and both baseline controls use only the retained training examples. No threshold or model settings were tuned. Future label-bar availability is used retrospectively to establish dataset eligibility; it is not a decision-time trading filter. Models retain their actual creation/availability time and are not represented as historically available forecasts. Source price basis, timestamp convention and volume semantics remain unverified; filtering does not authenticate the source or make it an approved execution dataset.

The frozen plan pins the cached Arrow files, prior reports, relevant original implementation, new implementation, and dependency versions before fitting. Reference checks verify the same dataset identity, fixed boundary and retained validation count; a regression also checks exact timestamp equivalence between the original and new continuity masks. Training content has its own deterministic identity, separate from full-dataset identity, and tests verify that changing validation prices cannot change fitted coefficients or preprocessing.

Saved evidence: [comparison report](../data/future-phase/OHLCV-1m-Forex/training/continuity/20261016T113345-01771fbe/comparison.md), [complete results](../data/future-phase/OHLCV-1m-Forex/training/continuity/20261016T113345-01771fbe/comparison.json), [frozen plan](../data/future-phase/OHLCV-1m-Forex/training/continuity/20261016T113345-01771fbe/plan.json), and [verification receipt](../data/future-phase/OHLCV-1m-Forex/training/continuity/20261016T113345-01771fbe/verification.json). Models and core evaluation reports were read back and matched through `Journal`/`SQLiteRepository`. Source, reference and implementation identities were rechecked before completion. Earlier evidence remains unchanged.

Reproduce into a new run directory:

```powershell
.venv/Scripts/python.exe -m scripts.compare_hf_forex_continuity --pairs AUDUSD EURUSD GBPUSD USDJPY --max-rows 400000 --reference-root data/future-phase/OHLCV-1m-Forex/training/comparisons/20261015T113615-648c4411
```

Verification passed **537 tests, 1 skipped, 8 socket tests deselected; 93.22% Astra coverage**. Ruff lint and format checks passed for `astra`, `tests`, and both original/new HF scripts. The 23 new synthetic checks cover gap/session exclusions, original-boundary purging, future-data isolation, model persistence, reference equivalence, changed inputs, and interruption/persistence failure receipts. All **107 protected source, acquisition and prior experiment files** retained their SHA-256 identities. Logs, JUnit, coverage and integrity receipts are in `data/astra-hf-continuity-verification-20261016/`. The initial focused runner test attempt needed its missing temporary parent directory created; its subsequent complete run and the full suite passed. Existing runpy/SQLite resource warnings remain.

No model was promoted and no cost-aware trading simulation or hosted operation was performed. Independent qualification still requires identified price/timestamp semantics, a separately reserved unused period, and risk-gated paper results at base and doubled costs. The [active data plan](fresh-validation-plan.md) retains the downloaded HF source and the retired acquisition decision.

## Original four-model experiment — 15 October 2026

The 15 October 2026 continuation adds a reproducible comparison for AUDUSD, EURUSD, GBPUSD and USDJPY using the already downloaded `keithtwesigye/OHLCV-1m-Forex` snapshot. These are historical classification experiments. Their scores do not establish net trading performance or qualify a model for paper decisions.

## Executed results

Run **20261015T113615-648c4411** completed on 15 October 2026 at **12:48 Africa/Lagos**. All four models used 400,000 source rows each, with 279,985 training examples after purging and 119,995 validation examples. The saved JSON reports and all four model records were read back from the local Journal and verified. Source/code identity checks passed. EURUSD reproduced the original model version, parameters and scores exactly; only the new model's creation time differs.

| Pair | Model accuracy | Training-majority baseline accuracy | Gain, percentage points | Gain on continuous windows, percentage points | Model Brier | Constant-probability Brier |
|---|---:|---:|---:|---:|---:|---:|
| AUDUSD | 51.1755% | 51.1046% | +0.0708 | +0.0174 | 0.24974222 | 0.24990965 |
| EURUSD | 51.2005% | 51.1663% | +0.0342 | +0.0146 | 0.24977205 | 0.24986627 |
| GBPUSD | 50.7521% | 50.7229% | +0.0292 | -0.0217 | 0.24991492 | 0.24998115 |
| USDJPY | 51.1313% | 50.1846% | +0.9467 | +0.9021 | 0.24984168 | 0.25001105 |

USDJPY has the largest accuracy improvement over its own baseline, and that improvement survives the continuous-window diagnostic. It is a candidate for further research, not a qualified trading model: probability-score gains are small, monthly outcomes vary, and neither costs nor execution returns have been evaluated. The other three accuracy gains are marginal; GBPUSD falls below its baseline on continuous windows. Selecting USDJPY from these results consumes this comparison as development evidence.

The next empirical test would freeze the selected candidate and evaluate a later, separately verified and reserved period, with identified price/timestamp semantics and risk-gated paper simulations at base and doubled costs. This run did not perform that test or certify an untouched holdout.

Saved evidence: [comparison report](../data/future-phase/OHLCV-1m-Forex/training/comparisons/20261015T113615-648c4411/comparison.md), [complete results](../data/future-phase/OHLCV-1m-Forex/training/comparisons/20261015T113615-648c4411/comparison.json), [frozen plan](../data/future-phase/OHLCV-1m-Forex/training/comparisons/20261015T113615-648c4411/plan.json), and [verification with dependency versions](../data/future-phase/OHLCV-1m-Forex/training/comparisons/20261015T113615-648c4411/verification.json).

## Data scope

A read-only scan of all six cached Arrow shards found the following coverage. Timestamps below are source timestamps in UTC; their open/close convention is not supplied by the source.

| Pair | Cached rows | Earliest timestamp | Latest timestamp |
|---|---:|---|---|
| AUDUSD | 958,658 | 2023-01-01 22:31 | 2025-07-31 23:59 |
| EURUSD | 1,015,292 | 2023-01-01 22:04 | 2026-02-24 16:58 |
| GBPUSD | 960,890 | 2023-01-01 23:04 | 2025-07-31 23:59 |
| USDJPY | 1,016,143 | 2023-01-01 22:00 | 2026-02-24 16:58 |

The whole source cache contains 44,024,447 rows across 49 tickers. That total does not imply that all pairs reach the latest date. The cache cannot replace Astra's missing September–October 2026 window. EURUSD's selected history contains a gap of 153 days, 22 hours and 5 minutes. The actual pair reports quantify missing open-session minutes, weekend/session warnings, and gaps within the selected tail.

The experiment retains the original close/availability-time assumption to reproduce the first EURUSD run. Naive timestamps are now rejected before conversion to UTC; they are never silently assigned the computer's timezone. The source's timestamp convention, bid/ask basis and volume meaning remain unverified. No source data is filled, sorted, spliced into another provider, or silently deduplicated.

## Fixed method

Each pair uses its latest 400,000 source rows, unless an explicit smaller limit is supplied. The reader filters the saved Arrow shards directly without a Hub connection or row-by-row Hugging Face iteration. It checks the complete ticker's timestamp ordering before cropping the tail, and verifies shard identities and row counts.

The existing `astra.ml` model is unchanged: one- and five-bar returns, relative ATR and candle body fraction; a scaler fitted on training rows only; deterministic logistic regression; and a chronological 70/30 split. Training examples whose labels become available at or after the first validation decision are purged. A label is 1 when the close after three observed bars is higher, otherwise 0, including ties. Three observed bars can span more than three elapsed minutes when data has gaps.

The comparison adds two controls fitted only from training labels: the majority class for accuracy and the training up-frequency as a constant probability for Brier score. It also saves predicted class frequencies, monthly validation results and calibration bins. Overlapping labels are dependent observations; the sample count is not an independent-trial count.

A separate diagnostic subset requires all 15 feature bars and the next 3 label bars to be consecutive one-minute observations inside the assumed FX session. It uses the same fitted model and training baseline. Neither the model nor its threshold is tuned using that subset or the calendar-month results.

These periods have not been certified as untouched. In particular, the EURUSD comparison repeats the earlier reported validation period. No P&L or transaction-cost result is inferred from direction accuracy, and no model is automatically selected or activated.

## Run and inspect

```powershell
.venv/Scripts/python.exe -m scripts.compare_hf_forex_models --pairs AUDUSD EURUSD GBPUSD USDJPY --max-rows 400000
```

Each invocation creates a new directory under `data/future-phase/OHLCV-1m-Forex/training/comparisons/`. It writes `plan.json` before fitting, recording the source hashes, code hashes, pair list, row limit, split, label, model settings and controls. Input and code identities are checked again before completion. Existing comparison directories and the original `training/EURUSD-M1-report.json` are preserved.

The directory contains a per-pair report, `comparison.json`, `comparison.md`, and `experiment.sqlite3`. Models use Astra's standard `Entity.MODEL` schema; diagnostic reports use `Entity.EVALUATION`, with planning/completion records in `Entity.AUDIT`. All records go through `Journal` and `SQLiteRepository`, and the runner verifies model/evaluation counts by reading them back. A failed input recheck or interrupted run cannot be reported as completed.

The original single-pair interface remains available:

```powershell
.venv/Scripts/python.exe scripts/train_hf_forex_experiment.py --pair EURUSD --max-rows 400000 --output data/astra-hf-single/EURUSD-M1-report.json
```

An explicit single-pair output also saves a local journal beside the report. These experiment journals are separate from application decision and execution records.

## Software verification

The continuation passed `.venv/Scripts/python.exe -m pytest --cov=astra`: **494 passed, 1 skipped, 8 socket tests deselected; 93.22% Astra coverage**. Ruff lint and format checks passed for `astra`, `tests`, and the HF scripts. New tests use explicitly identified synthetic fixtures and cover timestamp conversion, cache integrity, chronological ordering before cropping, training-only baselines, gap-window diagnostics, loadable model records, durable comparison output and interrupted/changed-input runs.

The full-suite check also confirmed that the real acquisition summary and original EURUSD report retained their SHA-256 identities. Logs, JUnit, coverage JSON and the integrity receipt are under `data/astra-hf-verification-20261015/`. Existing runpy/SQLite resource warnings remain; they did not fail verification. No hosted database operation was performed.
