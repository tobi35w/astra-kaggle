# Astra: current stage and next decisions

Initial assessment on 14 October 2026 from the code, local saved artifacts, and software verification; subsequent evidence is dated below. Active data selection was corrected on 16 October. The initial assessment inspected historical experiments without rerunning them and performed no hosted database check or migration.

## Kaggle handoff — 16 October 2026

At the user's request, heavy local work is paused. The [Kaggle workflow](kaggle.md) packages source, tests, documentation and sampled provenance into a small export. It downloads the pinned public HF snapshot on Kaggle, trains USDJPY by default (or all four pairs), and then runs the prepared USDJPY cost-aware paper experiment. Software tests are optional cloud work. Source data, existing research outputs and credentials are excluded from export; prior local evidence is preserved.

The [executed sample audit](../kaggle/provenance/audit.md) found 240 matching EURUSD/USDJPY BID OHLC samples at source opening timestamps, with zero matches at adjacent minutes. Volume matched within tolerance rather than independently established exact equality. New cloud training and replay shift opening timestamps +1 minute to close/availability. This does not authenticate every cached row, and earlier model reports retain their original clock assumptions.

The profitability runner is prepared but has not been executed on real data. The Kaggle workflow has not been executed on Kaggle. Local handoff validation is limited to source/notebook syntax, Ruff and export integrity; no fresh local training, dataset scan, download or full test run forms part of this handoff. Added regression tests are included for cloud execution. Profitability remains unproven and no model is promoted.

## Decision

Astra is in **research development and validation**. The causal research engine, independent risk approval, paper replay, persistence, training workflows, and local apps already exist. On 16 October the user clarified that current four-pair model development should use the downloaded `keithtwesigye/OHLCV-1m-Forex` snapshot. The existing verified curated archive remains available. The incomplete September–October acquisition target was retired on 15 October; recovering its 72 missing daily files is no longer a prerequisite for development. Independent confirmation on previously unused data remains a later model-qualification requirement.

## Evidence inspected

| Evidence | Recorded result | Interpretation |
|---|---|---|
| Four-pair historical research | Completed; progress records 2,955 against 2,904 planned | Counts disagree; the desk flags this and caps the bar, without changing the source artifacts |
| Yllvar Training Arena | 68 independent test accounts; 20 models | Separate account simulations; their sum is not a portfolio return |
| Curated Dukascopy H1/H4 campaign | 792 runs; 44 models; 12,334 trade records | Completed historical baseline across independent windows |
| Curated base costs | 396 runs; mean run return -0.8067% | Baseline has not established an edge |
| Curated doubled costs | 396 runs; mean run return -1.1834% | Worse under increased costs |
| Retired continuous acquisition | Offline reconciliation: 72/144 required open-session units verified | Retired as an active requirement on 15 October; the incomplete acquisition and its evidence repair remain historical records |
| EURUSD paper rehearsal | 19 closed trades; saved P&L +184.46 | Finite historical replay; not independent forward evidence |

Curated return means are weighted by the number of runs in each saved strategy/pair/timeframe group. USD and JPY cash P&L are never summed. Most curated strategy/window runs contain fewer than 30 trades. The saved loss diagnosis labels promotion `NOT_QUALIFIED`. The historical windows used for that diagnosis cannot confirm a candidate selected or tuned using their results.

Primary local evidence: `data/training-arena/curated-history-h1-h4-v1/evaluation-report.json`, `loss-diagnostics.json`, and `next-phase-readiness.json`; `data/training-arena/arena-state.json`; `data/astra-training-four-pairs/research-progress.json`; `data/astra-continuous-2026-tail/acquisition-summary.json`. These are recorded local artifacts, not independent authentication of the price source.

## Subsequent Hugging Face experiment — 15 October 2026

The cached `keithtwesigye/OHLCV-1m-Forex` experiment completed four models using 400,000 rows per pair. USDJPY achieved 51.1313% validation accuracy against a 50.1846% training-majority baseline; its improvement persisted on uninterrupted feature/label windows. AUDUSD, EURUSD and GBPUSD had much smaller gains. These are historical classification diagnostics, with no demonstrated net trading edge and no promoted model. The snapshot ends in July 2025 for AUDUSD/GBPUSD and February 2026 for EURUSD/USDJPY, with large internal gaps in the latter two, so it does not complete the missing September–October 2026 data.

The [experiment report](hf-forex-experiment.md) records methodology, exact results, caveats, saved Journal evidence and the next empirical test. Full software verification passed **494 tests, 1 skipped, 8 socket tests deselected; 93.22% coverage**, with Ruff lint and formatting passing. This follow-up performed no hosted operation and did not alter prior acquisition or experiment evidence.

## Subsequent replacement-data search — 15 October 2026

Four current ForexSB M1 archives were acquired for the September–12 October window. They contain observations on every requested open-session date, but omit **906 required minutes** in total. Overlap against the original Dukascopy BID files matched 86,245 of 86,247 OHLC rows; two GBPUSD rows differed. All M1/M5/M15/H1/H4 checks reported incomplete coverage, so the alternative remains unapproved and the original acquisition still has 72 missing pair-days. Current HistData, FXCM, Dukascopy and qleap availability checks did not provide a complete replacement. The [executed replacement-data audit](replacement-data-audit-20261015.md) records source hashes, exact gaps, comparison results and the reproducible offline audit command.

The new audit tool passed the full offline suite: **514 tests, 1 skipped, 8 socket tests deselected; 93.22% coverage**, with Ruff lint and formatting passing. All 105 protected source, acquisition and experiment evidence files retained their SHA-256 identities across testing.

## Active data selection — 16 October 2026

The [active research data plan](fresh-validation-plan.md) now selects `data/future-phase/OHLCV-1m-Forex/` for AUDUSD, EURUSD, GBPUSD and USDJPY model development. The [continuity comparison runner](../scripts/compare_hf_forex_continuity.py) completed run `20261016T113345-01771fbe`, retraining all four models after excluding incomplete feature/label windows while preserving the original chronological boundaries. USDJPY scored **51.1356% accuracy versus a 50.0961% baseline**; AUDUSD, EURUSD and GBPUSD scored slightly below their respective baselines. Source/reference/code checks and Journal read-backs passed. The [experiment report](hf-forex-experiment.md) contains exact results, exclusions and reproducible commands. Source semantics remain unverified, the historical periods have already been inspected, and profitability was not evaluated.

The continuation passed **537 tests, 1 skipped, 8 socket tests deselected; 93.22% coverage**, with Ruff lint and formatting passing. All 107 protected prior evidence files retained their SHA-256 identities.

The superseded 15 October plan selected 22 curated Dukascopy windows. Its verification remains valid historical evidence: SHA-256 and row counts matched for all 110 M1/M5/M15/H1/H4 outputs, and all 44 H1/H4 datasets passed the managed loader. Those previously used windows remain available for their existing research workflows; they are not an untouched holdout.

## What the Evaluation Desk provides

- One overview for project stage, evidence, outstanding checks, and the next research decision.
- Views for software checks, research, both training campaigns, acquisition, audits, and paper replay, with links to the existing Research Sandbox and Training Arena.
- A full offline verification run: pytest coverage, Ruff lint, and Ruff formatting. Logs stream while a phase runs; structured test counts and coverage appear when pytest finishes.
- Durable Journal-backed phase history and results after restart. A crashed run is marked interrupted. A process lock prevents duplicate verification in the same desk workspace.
- Clear missing, incomplete, recorded-active, stale, and completed states. The 15-minute research activity threshold is a display heuristic; the desk cannot infer worker liveness from files alone.

Use **Launch Astra Evaluation Desk.cmd** or **Launch Astra Progress Desk.cmd**, then open `http://127.0.0.1:8789/`. Reopening the launcher reuses the server. The CLI form is `.venv/Scripts/python.exe -m scripts.launch_progress_desk`. The foreground server command is `.venv/Scripts/python.exe -m astra desk --no-browser`.

## Next research steps

1. Continue four-pair model development using the selected HF cache and pinned source identities.
2. Use the completed continuity comparison as development evidence. Freeze any further experiment's features, chronological boundaries and training-only controls before fitting, with a new output identity.
3. Exclude feature/label windows that cross missing minutes, fit preprocessing only on earlier observations and purge overlapping labels. Label reused-history results as internal validation.
4. For independent model qualification, separately verify and reserve a previously unused period before examining its outcomes. This period need not be the retired September–October interval.
5. Review sample size, calibration and risk-gated paper results at base and doubled costs, including repeated net outcomes, drawdown and risk/P&L reconciliation, before model promotion or ongoing paper observation.

No candidate is promoted by the desk. No fresh holdout has been acquired, reserved, or evaluated by this change. The MT5/broker/live-order boundary remains in force.

The [research data plan](fresh-validation-plan.md) records the active HF selection, preserved curated verification, retired September–October acquisition and separate requirement for independent confirmation.

## Verification of this change

The final native desk verification run, `20261014T113504869518-42396110`, passed all three phases:

- Pytest with `--cov=astra`: **479 passed, 1 skipped, 8 socket tests deselected; 93.22% coverage**.
- Ruff lint and formatting: passed. The new launcher also passed separate Ruff checks.
- Mypy: passed for all 56 source files. Compileall and dependency consistency checks passed.
- Frontend JavaScript syntax and DOM smoke checks using actual saved state passed, including cost summaries, bounded progress, filtering, missing evidence, and escaping artifact strings.
- Windows process tests verified duplicate-run exclusion, durable history, interrupted-run recovery, and termination of an attached worker after abrupt parent exit. The same-port duplicate-bind regression passed without making an outgoing connection.

The first baseline suite encountered two Windows temporary-folder permission failures. Per-run workspace temporary directories resolved both. Later lint/format failures were corrected before the final recorded pass. The suite still emits existing runpy/SQLite resource warnings; they did not fail the run.

Full socket/browser verification could not be completed: loopback connection attempts were blocked in the sandbox, and the automatic approval reviewer returned a review-system error when asked to launch the desk or interact with the browser. The user confirmed opening the dashboard manually. Its visual appearance and HTTP interactions have not been independently verified in a browser during this change.

Run logs, JUnit results, coverage JSON, and the verification Journal are retained under ignored `data/progress-desk/`. The saved successful result is restored by the desk after restart.
