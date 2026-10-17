# Astra development instructions

- Preserve the hard boundary: no MT5 imports, broker credentials, live order routes, or live trading mode.
- Keep strategies independent of risk approval; every candidate passes through the risk engine.
- Prefer deterministic, causal calculations and document assumptions. Avoid look-ahead in all evaluators.
- Treat timestamps as candle close/availability times when doing multi-timeframe joins; exclude bars after decision time.
- Risk checks derive reward/risk from entry/stop/target fields, not a strategy-supplied ratio.
- Add meaningful tests for behavioral changes; run `.venv/Scripts/python.exe -m pytest --cov=astra` and Ruff.
- Never commit `.env` or credentials. Paper execution is the only execution provider in this stage.
- Preserve the existing repository interfaces and compatibility tests. New persistence goes through `Journal`/`Repository`; inject remote adapters and use deterministic transports in tests.
- Never mark a hosted migration or real-data evaluation completed without executing and verifying it. Keep synthetic fixtures explicitly identified as tests.
- Keep P&L units consistent: `pnl_per_unit` is per unit; `pnl` is total. Recheck pending orders through risk at their actual assumed fill price.
- Use close/availability timestamps for performance and model records too. Purge overlapping ML labels at chronological validation boundaries.
