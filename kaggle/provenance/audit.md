# Hugging Face sampled source provenance

Already executed checks on 16 October 2026 matched **240/240 sampled OHLC rows exactly** to hash-verified Dukascopy BID raw data at the same candle-open timestamps. Volumes matched within 0.0001 for all 240 rows. Neither a minus-one-minute nor a plus-one-minute join matched any complete OHLC row.

The samples cover EURUSD and USDJPY from 12:00 through 12:59 UTC on 3 January 2023 and 15 January 2026. Source timestamps therefore require **one minute added to obtain candle availability** under the strongly corroborated sampled convention. Applying this convention beyond the samples remains an assumption; the whole cache is not authenticated.

The cached revision's [duplicate commit](https://huggingface.co/datasets/keithtwesigye/OHLCV-1m-Forex/commit/f38b5658b02d1a081f9c2abb314d8a3972caae1e) names `mito0o852/OHLCV-1m-Forex`. The original repository's [January 2023 commit](https://huggingface.co/datasets/mito0o852/OHLCV-1m-Forex/commit/dcf7e34fe9efc58ffdd76c47e39f09a53c46d04e) identifies Dukascopy OHLCV. Both repositories returned HTTP 404 for README.md.

`audit.json` records the exact source URLs, revision IDs, sampled windows, first rows, reference paths, reference SHA-256 checksums, comparison tolerances, and limitations. This receipt preserves existing observations for the Kaggle export. It does not claim a completed profitability evaluation or an untouched holdout. No additional scans or training were needed to save these notes.
