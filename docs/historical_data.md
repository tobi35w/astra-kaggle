# Astra historical data engine

The engine uses Dukascopy public M1 BID candles as its canonical source. The direct BI5 path is live verified for a bounded sample and stores files under `data/raw/dukascopy/`. The optional `pipeline-json` path uses pinned `dukascopy-node` and preserves monthly source JSON under `data/raw/dukascopy-node/`. Its existing archive has now been curated and converted into the approved four-pair training windows documented in [historical data curation](historical-data-curation.md). Both source formats label volume `provider`. Raw BI5 units remain `provider_raw` without an unverified conversion; JSON uses the explicitly requested unit `millions`. Raw BID prices are not midpoint prices.

Current readiness is in [the pre-MT5 report](pre-mt5-readiness-report.md). The active historical scope is AUDUSD, EURUSD, GBPUSD, and USDJPY. Their curated data comprises separate validated intervals, each with M1/M5/M15/H1/H4; the full continuous four-pair archive is incomplete. See [four-pair curation](historical-data-curation.md) for exact dates, row counts, assumptions, and exclusions. A latest-day provider check failed on 2026-10-11. Hashes establish local integrity, not independent price authentication.

**Local archive audit (2026-10-11):** `data/astra-full-history/` contains 345 manifest-listed successful monthly M1 chunks and 71 failed chunks across the four active pairs, with manifest ranges from 2018-01-01 through 2026-09-01 exclusive. The full archive is discontinuous. Twenty-two uninterrupted windows of at least six months were reprocessed and passed managed quality checks in all five timeframes; these are saved under `data/astra-training-four-pairs/`. The curation manifest records source checksums, row counts, selected windows, and exclusions. Sixteen raw JSON files outside the manifest were audited and removed: twelve failed full open-session/range coverage, and four failed JSON parsing. Their hashes and findings remain in `unmanifested-source-audit.json`. Source manifest-listed files are retained for provenance and recovery; excluded shorter windows and failed months are not in the curated training root. All prices are BID, volume is the provider's millions unit, and data hashes do not independently authenticate the provider.

`HISTORICAL_DATA_BASE_URL` or `--base-url` explicitly selects `https://datafeed.dukascopy.com/datafeed` or `https://www.dukascopy.com/datafeed`. There is no silent provider substitution. Discovery checks at most 14 completed UTC days, rejects incomplete source days and aborts on network failure instead of treating it as missing data:

```powershell
.venv/Scripts/python.exe -m astra historical latest --base-url https://www.dukascopy.com/datafeed
.venv/Scripts/python.exe -m astra historical pipeline --start 2026-10-05 --end 2026-10-05 --symbols AUDUSD EURUSD GBPUSD USDJPY --root data/astra-master-latest --workers 1 --base-url https://www.dukascopy.com/datafeed
```

Every missing open-session interval is reported, including range edges. `HISTORICAL_DATA_MAX_GAP_MINUTES` only annotates severity; it cannot conceal a gap or produce PASS for incomplete data. Gap sizes use minutes for every timeframe. Weekend-crossing candles are flagged. The New York calendar does not automatically excuse holidays. WARN and FAIL exit nonzero. Offline dry-runs require an explicit end date and create no files or directories.

Compressed responses and decompression are bounded. Invalid or changed caches are fetched again; normalization requires matching manifest checksums. Corrupt manifests stop processing. Requests are spaced at least half a second apart per downloader. JSON exports must cover the complete requested chunk, and the external process has a timeout. No missing candle is fabricated.

`astra.historical.dataset.load_dataset(root, symbol, timeframe)` checks a successful managed BI5 report, its checksum and its requested range before returning `MarketData` with the canonical M1 hash, content version and BID price basis. `require_quality(data)` gates CLI/dashboard research. Imports validate the complete snapshot before a Journal write and use content versions in UUID identities, preserving changed data as distinct observations. Manifests retain prior version summaries; retain corresponding source files to reproduce prior research. Processing still materializes one symbol's requested window, so use bounded windows until streaming normalization is implemented.

Normal pytest blocks external network access and does not load local `.env`. The separate tiny real integration test is opt-in:

```powershell
$env:ASTRA_RUN_LIVE_TESTS = "true"
.venv/Scripts/python.exe -m pytest tests/test_live_integrations.py
Remove-Item Env:ASTRA_RUN_LIVE_TESTS
```

All timestamps exposed to Astra are UTC candle-close/availability times. Source minute timestamps are advanced one minute exactly once. Higher timeframes use UTC clock boundaries and are emitted only when every constituent M1 candle is present. Missing candles are not synthesized. Weekend closures use the New York 17:00 FX session boundary and DST-aware `America/New_York` rules.

## Commands

```powershell
# No network access; show the date units the pipeline would request.
.venv/Scripts/python.exe -m astra historical pipeline --start 2024-01-01 --end 2024-01-02 --symbols EURUSD --dry-run

# Download, normalize, validate, and derive all supported timeframes.
.venv/Scripts/python.exe -m astra historical pipeline --start 2024-01-01 --end 2024-01-02 --symbols EURUSD

# Download only; omitted end discovers a recent source day.
.venv/Scripts/python.exe -m astra historical download --start 2024-01-01 --symbols EURUSD

# Explicit processing stages.
.venv/Scripts/python.exe -m astra historical normalize EURUSD --start 2024-01-01 --end 2024-01-02
.venv/Scripts/python.exe -m astra historical validate EURUSD M1 data/normalized/m1/EURUSD.csv
```

For JSON exported by `dukascopy-node`, include flat candles (`-fl` / `--flats`) so zero-volume minutes are not omitted and resampled bars can be complete. `-to` is exclusive; for Jan 2 and Jan 3, use Jan 4 as the end date:

```powershell
npx --yes dukascopy-node@1.50.0 -i eurusd -from 2024-01-02 -to 2024-01-04 -t m1 -p bid -v -fl -f json -dir .\data\dukascopy-json -bs 1 -bp 1000 -r 0
.venv/Scripts/python.exe -m astra historical process-json EURUSD .\data\dukascopy-json\eurusd-m1-bid-2024-01-02-2024-01-04.json --root .\data\astra-dukascopy
```

`process-json` treats JSON timestamps as UTC candle-open timestamps and advances them one minute to Astra's close/availability convention. It preserves zero-volume flat records, validates the M1 OHLCV stream, emits only complete M5/M15/H1/H4 bars, and writes local quality reports. JSON CSVs use a generic `volume` column with `volume_type=provider` and `volume_unit=millions`; volume labels and units survive CSV loading and optional Journal import. Any missing timestamps remain flagged; they are not filled or concealed. It does not connect to or write to Supabase.

For managed multi-symbol downloads, use `pipeline-json`. It splits ranges into calendar-month chunks for bounded files and resumable progress, verifies source checksums from the manifest before reuse, and processes all successful chunks together. `--workers` is limited to one or two symbols at a time. Each source request uses batch size 1 and a one-second pause, enables flat candles, explicitly requests provider volume in millions, and retries transient artifact failures three times with a 30-second pause. `--force` bypasses both Astra's completed-chunk manifest and the Dukascopy-node cache. A month-scale example is:

```powershell
.venv/Scripts/python.exe -m astra historical pipeline-json --start 2024-01-01 --end 2024-01-31 --symbols EURUSD GBPUSD USDJPY --root data/dukascopy-month-test --workers 1
```

Rerunning the same command resumes from verified monthly chunks. Use `--dry-run` to list planned monthly chunks without launching Node or making network requests. The first use requires Node.js/npm and network access to the npm registry and Dukascopy. When range downloads complete, the manifest stores each chunk's row count, SHA-256, dates, and flat-inclusion setting.

For the legacy BI5 pipeline, `--force` re-fetches existing units; otherwise a raw file is reused only when it can be decoded and checked against its manifest checksum. Each daily response is size-limited, validated before an atomic `.part` rename, and tracked with SHA-256/checksum and status. That path is bounded by configured start/end dates, symbol list, maximum eight workers, retry count, timeout, and batch size. Dry-run only reports planned work and creates no download files.

Defaults are configurable with `HISTORICAL_DATA_ROOT`, `HISTORICAL_DATA_SYMBOLS`, `HISTORICAL_DATA_START_DATE`, `HISTORICAL_DATA_END_DATE`, `HISTORICAL_DATA_WORKERS`, `HISTORICAL_DATA_RETRIES`, `HISTORICAL_DATA_TIMEOUT`, `HISTORICAL_DATA_MAX_GAP_MINUTES`, and `HISTORICAL_DATA_BATCH_SIZE`. The acquisition default and validation allowlist are limited to AUDUSD, EURUSD, GBPUSD, and USDJPY. Omit the end date to discover a recent available source date; a complete acquisition should pin an end date after discovery for reproducibility.

To optionally store candles, invoke the `historical import SYMBOL TIMEFRAME PATH` stage after configuring Astra persistence. It batches through the existing `Journal`/`Repository`; deterministic IDs make re-imports idempotent. No connection to Supabase is needed for download or local processing. To permit M1 in Supabase's journal, apply `database/migrations/003_historical_m1.sql` after the journal migration; do not apply it automatically. `database/supabase_setup.sql` contains the equivalent M1 constraint for new deployments.

`data/manifests/dukascopy_manifest.json` records source-unit checksums and per-symbol dataset results. `data/reports/latest.json` and `latest.txt` summarize the most recent run. These generated datasets are intentionally git-ignored. The documented public endpoint can be temporarily unavailable; HTTP errors are isolated to their symbol/date units, and no report should describe an unavailable source range as downloaded or verified.
