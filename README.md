# Astra

**Run the next experiments on Kaggle:** use [the Kaggle guide](docs/kaggle.md), upload `exports/astra-kaggle.zip` as a private Dataset, and import `exports/Astra_Research.ipynb`. The notebook downloads the pinned Hugging Face source, trains USDJPY by default, runs the prepared cost-aware paper replays, and packages results for download. CPU is sufficient; Internet is required. Set `RUN_TESTS = True` for cloud software checks. Heavy local work is paused at the user's request.

Rebuild the small source-only export with `.venv/Scripts/python.exe -m scripts.export_kaggle`. It excludes local datasets, results, environments and credentials; existing research evidence stays on disk. The export setup has not yet been executed on Kaggle, and the new profitability experiment has not yet run on real data.

Historical data acquisition, normalization, quality checks, and UTC resampling are documented in [docs/historical_data.md](docs/historical_data.md). The local pipeline runs independently of Supabase; optional journal persistence requires explicit configuration.

Astra is a causal Forex research, decision, and paper-execution system. It has no MT5 connection, broker connection, or live trading mode.

**Current stage (2026-10-16): research development and validation; no candidate qualified for promotion.** At the user's clarification, current four-pair model development uses the downloaded `keithtwesigye/OHLCV-1m-Forex` snapshot in `data/future-phase/OHLCV-1m-Forex/`. The completed comparison uses continuous feature/label windows and fixed chronological boundaries through [compare_hf_forex_continuity.py](scripts/compare_hf_forex_continuity.py). This is historical development: a [subsequent sample audit](kaggle/provenance/audit.md) supports BID prices and opening timestamps for sampled EURUSD/USDJPY rows. New cloud runs add one minute to availability; whole-cache authentication remains incomplete. The cache ends in July 2025 or February 2026 by pair. The existing curated Dukascopy archive and its verification remain available. The September–October acquisition with 72 missing daily files remains retired as an active requirement. See the [active data plan](docs/fresh-validation-plan.md), [experiment evidence](docs/hf-forex-experiment.md), and [project status](docs/project-status.md).

**Hugging Face experiment follow-up (2026-10-16):** all four models were retrained using uninterrupted feature and label windows. USDJPY scored 51.1356% accuracy against its 50.0961% training-majority baseline; the other three scored slightly below their baselines. Source, implementation and prior-result identities passed rechecks, and saved models/reports were verified through the Journal. Profitability remains untested. See the [executed results and reproducible experiment](docs/hf-forex-experiment.md). Software verification passed 537 tests with 93.22% coverage and clean Ruff checks; 107 protected prior evidence files remained unchanged.

**Replacement-data follow-up (2026-10-15):** four fresh ForexSB archives cover every requested September–12 October trading date, but omit 906 open-session minutes. Their source and derived timeframes remain unapproved; the original acquisition still has 72 missing pair-days. See the [executed source checks and reproducible audit](docs/replacement-data-audit-20261015.md).

## Monitor Astra

Double-click **Launch Astra Evaluation Desk.cmd** (the existing **Launch Astra Progress Desk.cmd** also works), or run `.venv/Scripts/python.exe -m scripts.launch_progress_desk`. Open [the Evaluation Desk](http://127.0.0.1:8789/) while its local server is running.

The overview explains the current stage, saved evidence, and next research gate. Separate views cover software verification, historical research, both training campaigns, data acquisition, source audits, and paper rehearsal. **Run software checks** executes the offline pytest suite with coverage, then Ruff lint and format checks. Phase outcomes, test counts, coverage, bounded logs, and recent runs are saved locally through the Journal and restored when the desk restarts. Each run has isolated temporary files and coverage output; a process lock prevents duplicate desk verification runs. On Windows, a private Job Object terminates attached test workers if the desk exits abruptly.

The desk refreshes every few seconds. Research and acquisition state comes from saved artifacts: a recent active record is labelled unconfirmed, and one older than 15 minutes is marked stale. A completed run is historical evidence, not proof of current profitability. The desk can start software verification; research and training controls remain in their own apps. Opening it does not launch acquisition or training. New verification records and logs stay under ignored `data/progress-desk/`.

## Install

Python 3.12 or newer, using the existing project environment:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[dev,postgres]"
# Optional, only for the gated learning module:
.venv/Scripts/python.exe -m pip install -e ".[ml]"
# If .env does not already exist:
Copy-Item .env.example .env
.venv/Scripts/python.exe -m astra
```

Supabase is Astra's configured primary journal for application records. Put the project URL and server-side secret in the ignored local `.env` file after following [database setup](docs/database.md); never put the secret in a client or source control. The four-pair historical evaluator keeps a local SQLite spool so an outage cannot block or lose an active evaluation, and its background sync mirrors journal rows, full trade details, walk-forward folds, model results, and progress into the private append-only Supabase journal. The Windows Research Sandbox launcher starts that sync automatically. `.env`, SQLite journals, caches, and model binaries are ignored by Git. Configuration output redacts credential values.

## CLI

Supply genuine, timezone-aware OHLCV CSV data. Timestamps mean candle close/availability time. The current research universe is AUDUSD, EURUSD, GBPUSD, and USDJPY. The parser retains compatibility with other supported pairs. The curated four-pair archive remains verified input for these strategy commands; the active HF M1 classification experiment has separate source assumptions and does not admit that cache to the managed strategy-data boundary. Use M5, M15, H1, or H4 for strategy research.

```powershell
.venv/Scripts/python.exe -m astra config
.venv/Scripts/python.exe -m astra health
.venv/Scripts/python.exe -m astra verify-database
.venv/Scripts/python.exe -m astra validate data/EURUSD_M5.csv --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra analyze data/EURUSD_M5.csv --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra backtest data/EURUSD_M5.csv --pair EURUSD --timeframe M5 --strategy ema_trend --spread 0.0001 --slippage 0.00002
.venv/Scripts/python.exe -m astra walk-forward data/EURUSD_M5.csv --pair EURUSD --timeframe M5 --train-size 500 --test-size 200
.venv/Scripts/python.exe -m astra evaluate data/EURUSD_M5.csv --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra robustness data/EURUSD_M5.csv --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra fitness --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra paper data/EURUSD_M5.csv --pair EURUSD --timeframe M5
.venv/Scripts/python.exe -m astra readiness --dataset-root data/astra-training-four-pairs/EURUSD/2022-02-01_to-2023-11-30 --pair EURUSD --timeframe H4 --campaign-report data/astra-research-paper-20261011/campaign-067dfc3cc7bd2118.json
.venv/Scripts/python.exe -m astra sandbox
```

To reproduce the saved EURUSD campaign into a new output directory, run:

```powershell
.venv/Scripts/python.exe -m astra research --dataset-root data/astra-verified-window --output data/astra-research-reproduction --symbols EURUSD --timeframe H4 --strategies ema_trend --train-size 700 --test-size 300 --spread-pips 1 --slippage-pips 0.2 --paper-enabled
```

This performs a finite historical replay only. `--paper-enabled` permits risk-gated simulated orders in that campaign; it does not connect to a broker.

The optional local research dashboard is available through `Launch Astra Sandbox.cmd`. `Launch Astra Research Sandbox.cmd` starts or resumes the unfinished four-pair historical evaluation and opens its results page. The evaluator uses a process lock so repeated launches do not create duplicate runs. Both dashboard commands bind to `127.0.0.1`; the general sandbox accepts validated CSV candle history and stores imported data, reports and its SQLite journal under `.sandbox/`. The dashboard supports decision analysis, backtests, strategy comparisons, walk-forward evaluation, robustness checks and paper replay. It does not connect to a feed or broker; dataset provenance, costs and OHLC fill assumptions remain the operator's responsibility.

`Launch Astra Training Arena.cmd` opens the separate Training Arena on `http://127.0.0.1:8788/`. Its background phase worker waits for the current four-pair evaluation to report `COMPLETED`, then uses the saved Yllvar dataset offline for chronological logistic-model training and isolated, risk-gated OOS paper simulations. It resumes from saved test results after interruption. Every test starts with its own $10,000 account; aggregate P&L is explicitly the sum of separate test accounts, not a deployable portfolio. No broker, live order route, market feed, or internet access is used. The optional Supabase sync queues locally and retries when online.

`analyze` accepts optional `--m5`, `--m15`, `--h1`, `--h4` CSV paths. Once any is provided, all four frames are required for full alignment. Default research quality checks distinguish FX weekend closures from missing open-session bars; `--reject-gaps` also rejects expected closure gaps. `--price-value` specifies the account-currency value of one price unit per position unit, not an inferred FX lot conversion. `--commission` is round-trip commission per position unit.

Research CLI and dashboard runs reject open-session gaps and candles outside the assumed FX session with `DATA QUALITY INSUFFICIENT`. The managed historical loader also verifies the acquisition report, requested range and checksum. Imported CSVs have content-derived dataset versions. Validation alone does not authenticate prices or establish strategy performance.

After hosted persistence is verified, `analyze --dataset-root data/astra-master-latest --pair EURUSD --timeframe M5` uses the managed data boundary. The existing CSV form remains supported. A dataset root and CSV cannot be supplied together.

Historical `backtest` and `evaluate` commands explicitly run a research simulation with their supplied costs. Adaptive `analyze` and `paper` honor `ASTRA_TRADING_ENABLED=false` by default and configured risk limits; enabling this flag only permits paper approvals. Paper mode is a finite CSV replay. It saves the resulting positions, orders, decisions, trade outcomes and a recovery checkpoint. It does not subscribe to any feed.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest --cov=astra
.venv/Scripts/ruff.exe check astra tests
.venv/Scripts/ruff.exe format --check astra tests
.venv/Scripts/python.exe -m compileall -q astra tests
.venv/Scripts/python.exe -m mypy astra
.venv/Scripts/python.exe -m pip check
git diff --check
```

The socket-level sandbox smoke test is isolated from the offline suite. Run `.venv/Scripts/python.exe -m pytest -m socket tests/test_sandbox.py::test_sandbox_loopback_http_smoke` on a host that permits loopback HTTP.

See [architecture](docs/architecture.md), [backtesting](docs/backtesting.md), [strategies](docs/strategies.md), [risk](docs/risk.md), [configuration](docs/configuration.md), [database](docs/database.md), and [learning](docs/learning.md). The [continuation report](docs/continuation-report.md) records files changed, checks, and remaining external setup.

To create the Supabase schema, paste [`database/supabase_setup.sql`](database/supabase_setup.sql) into the Supabase Dashboard SQL Editor and run it. See the [database setup guide](docs/database.md) before configuring Astra's server-side credentials.
