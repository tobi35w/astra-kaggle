# Run Astra on Kaggle

The export moves the Hugging Face Forex experiment to Kaggle. A CPU notebook is sufficient for the current logistic-regression models. Your machine only prepares the source ZIP; installation, dataset loading, training, evaluation and optional tests run on Kaggle.

## Export on your computer

From the project directory:

```powershell
.venv/Scripts/python.exe -m scripts.export_kaggle --output-dir exports
```

This creates:

- `exports/astra-kaggle.zip`: the source bundle for a private Kaggle Dataset.
- `exports/Astra_Research.ipynb`: the notebook to import into Kaggle.

The bundle contains source, tests, documentation and selected provenance. It excludes credentials, `.env`, local databases, downloaded datasets, run outputs, virtual environments and local caches. Existing local experiments and datasets remain on your computer. Review the export manifest if you want to inspect the exact packaged files.

## Start a Kaggle notebook

1. Create a **private Kaggle Dataset** from `exports/astra-kaggle.zip`.
2. Create a Kaggle notebook and import `exports/Astra_Research.ipynb`.
3. Use **Add Input** to attach the private source Dataset.
4. Turn **Internet on** in notebook settings. Use CPU; no GPU is required.
5. Run the cells in order, or **Run All** with the defaults.
6. Save a notebook version after completion, then download `astra-results-*.zip` from the final cell or Kaggle's **Output** tab.

The notebook accepts either the ZIP or source files Kaggle has already extracted. Keep `EXPORT-MANIFEST.json` with the source: the first cell verifies the exact file inventory, sizes and SHA-256 hashes before installing or running project code, and rejects missing or extra files. It creates a fresh project directory under `/kaggle/working`.

Astra requires Python 3.12 or later: an older notebook kernel triggers a managed Python 3.12 installation through `uv`. The isolated runtime, `uv` bootstrap files and installer caches use the session's temporary directory, outside `/kaggle/working`, so saved notebook outputs do not include dependencies. Commands use that environment without changing the notebook kernel.

The first package installation and dataset download need Internet. Kaggle may require account verification to enable Internet. This export does not upload anything or start a hosted run by itself.

## Configure a run

The notebook's configuration cell defaults to:

```python
MODE = "all"
PAIRS = ["USDJPY"]
MAX_ROWS = 400000
INPUT_CACHE = None
REFERENCE_ROOT = None
RESULTS = WORKING / "astra-results"
RUN_TESTS = False
```

| Setting | Meaning |
| --- | --- |
| `MODE = "all"` | Train continuity-filtered models for the selected pairs, then evaluate USDJPY historical profitability. |
| `MODE = "train"` | Save trained models and classification diagnostics. |
| `MODE = "profitability"` | Evaluate an existing compatible continuity-training run supplied in `REFERENCE_ROOT`. |
| `MODE = "check"` | Check Python and installed runtime dependencies without downloading data or running experiments. |
| `PAIRS` | Start with USDJPY, or use `["AUDUSD", "EURUSD", "GBPUSD", "USDJPY"]` to train all four. The profitability experiment currently evaluates USDJPY only, so `all` must include USDJPY. |
| `MAX_ROWS` | Latest source rows per training pair, at least 300. Keep 400,000 for the established experiment size; smaller runs are exploratory. Profitability uses the row limit saved in its training reference. |
| `INPUT_CACHE` | Optional read-only input path to the existing Hugging Face `hf-cache` directory. |
| `REFERENCE_ROOT` | Existing compatible training run for profitability-only mode. `all` creates the required reference. |
| `RUN_TESTS` | Enable pytest coverage and Ruff on Kaggle. Defaults to false. |

With no input cache, the runner downloads the pinned `keithtwesigye/OHLCV-1m-Forex` source snapshot on Kaggle. The source cache uses temporary session storage, outside saved notebook outputs and the result directory. The first download includes the upstream dataset even when only one pair is selected; allow space and time for it. A fresh Kaggle session downloads the source again unless you attach the existing cache as an input.

The bundled sample audit corroborates BID prices and candle-open timestamps for sampled USDJPY rows. The cloud training pipeline adds one minute to source timestamps to represent close/availability times. Applying that interpretation to the full cache remains an explicit assumption. Earlier local runs used a different clock assumption and are not directly equivalent references; `MODE = "all"` creates a new compatible training reference before replay.

To use your already downloaded source, upload the existing `data/future-phase/OHLCV-1m-Forex/hf-cache` directory as a separate **private** Kaggle Dataset. Preserve its directory structure, attach it through **Add Input**, and set, for example:

```python
INPUT_CACHE = "/kaggle/input/astra-hf-snapshot/hf-cache"
```

Use the actual path displayed by Kaggle. Do not point this setting at the ForexSB archives, the different Yllvar dataset, CSV conversions or a Hugging Face `save_to_disk` directory. Package installation still needs Internet unless you separately provide an offline environment.

To rerun only profitability evaluation, attach prior downloaded results as another private input and set `REFERENCE_ROOT` to the compatible continuity-training run directory. Keep its reports, frozen plan, verification receipt and experiment journal together. The runner validates references before evaluating.

## Outputs and interpretation

`/kaggle/working/astra-results` holds runtime provenance and the training/profitability run directories. Each completed experiment writes its own reports and verification evidence. The final notebook cell creates a downloadable ZIP of all saved runs in this directory, including incomplete-run receipts and optional software-check logs, while excluding source data and environments.

A successful notebook execution is not itself evidence of profit. Read the actual reports for net returns after costs, trade count, drawdown, uncertainty and sensitivity to doubled costs. The historical period has already been used for development. Sampled source evidence does not authenticate the full cache or establish actual bid/ask spreads and execution costs. A profitable replay still needs independent data and an unused evaluation period before any profitability claim is justified. The system remains research and paper execution only.

The exported setup has not been run on Kaggle by the assistant. Local package installation, training and the full test suite are intentionally not part of export. To run software verification on Kaggle, set `RUN_TESTS = True`; the notebook invokes `python -m pytest --cov=astra` and `python -m ruff check astra scripts tests`. It saves JUnit results, coverage JSON and command logs under `astra-results/verification` for download.

## If a cell fails

- **No bundle found:** attach the source Dataset. If multiple copies are attached, set `CODE_ZIP` or `PROJECT_INPUT` in the first code cell.
- **Manifest mismatch:** re-export and upload the complete source bundle. Keep `EXPORT-MANIFEST.json` and avoid editing, adding or removing files inside the uploaded source Dataset.
- **Python or package download fails:** enable Internet, then rerun setup. The notebook keeps older project and result folders.
- **Reference or cache rejected:** use the original directory structure and compatible snapshot/reference. Read the reported mismatch before changing anything.
- **Session ends or storage fills:** save/download completed outputs, then start a new session. Uploaded inputs persist; unsaved working files may not.
- **Experiment fails partway through:** inspect the error and any incomplete-run receipt. Rerunning creates new experiment run directories; incomplete work is not reported as completed.

For the experiment's prior results and data assumptions, see [Hugging Face Forex experiment](hf-forex-experiment.md).
