# Configuration

`Settings.from_env()` loads `.env` when python-dotenv is installed, with process environment taking precedence. Invalid/nonfinite/negative numeric settings fail safely instead of silently loosening limits. Direct Settings objects are validated at repository creation and risk approval. Secret fields are excluded from dataclass repr and displayed only as presence booleans by `config`.

| Variable | Default | Meaning |
|---|---|---|
| ASTRA_TRADING_ENABLED | false | Permit paper approvals only |
| ASTRA_MAX_DAILY_LOSS_PCT | 0.05 | At most 5%; smaller configured limits apply |
| ASTRA_MAX_TRADE_RISK_PCT | 0.005 | Normal per-trade risk; environment values capped at 1% |
| ASTRA_MAX_DRAWDOWN_PCT | 0.20 | Peak-equity drawdown veto |
| ASTRA_MAX_CONSECUTIVE_LOSSES | 5 | Closed-loss sequence veto |
| ASTRA_MAX_POSITION_SIZE | 1 | Total units across open positions |
| ASTRA_MAX_OPEN_POSITIONS | 3 | Maximum open positions |
| ASTRA_MAX_PAIR_EXPOSURE | 1 | Units in one pair |
| ASTRA_MAX_CORRELATED_EXPOSURE | 2 | Units in correlated pairs |
| ASTRA_MAX_SPREAD / ASTRA_MAX_SLIPPAGE | 0 / 0 | Admitted costs in price units |
| ASTRA_MAX_DATA_AGE_SECONDS | 3600 | Decision input freshness; MTF checks also use each interval |
| ASTRA_MINIMUM_CONFIDENCE | 0.5 | Signal and aggregate confidence thresholds |
| ASTRA_MINIMUM_REWARD_RISK | 1.0 | Minimum ratio derived from price levels |
| ASTRA_MAX_RELATIVE_VOLATILITY | 0.03 | Decision veto for ATR/price above limit |
| ASTRA_KILL_SWITCH / ASTRA_EMERGENCY_HALT | false / false | Independent risk halts |
| ASTRA_ML_ENABLED | false | Require an available model when enabled |
| ASTRA_PERSISTENCE_BACKEND | sqlite | memory, sqlite, postgres, supabase |
| ASTRA_SQLITE_PATH | astra.sqlite3 | Local journal path |
| ASTRA_DATABASE_URL | unset | Legacy PostgreSQL credential |
| SUPABASE_URL | unset | HTTPS project URL; placeholder only in .env.example |
| SUPABASE_PUBLISHABLE_KEY | unset | Low-privilege Data API credential |
| SUPABASE_SECRET_KEY | unset | Private server-side Data API credential |
| SUPABASE_DB_URL | unset | PostgreSQL credential, preferred over legacy URL |
| ASTRA_SUPABASE_USE_SECRET | false | Explicit server-side journal-write opt-in |

`ASTRA_ENV` and `ASTRA_LOG_LEVEL` are retained operational labels. All defaults are in `config.py`. No setting enables live trading. Do not put a credential on a command line or in a committed file.
