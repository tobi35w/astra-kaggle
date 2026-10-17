"""Centralized, conservative environment-backed settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from math import isfinite


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    log_level: str = "INFO"
    trading_enabled: bool = False
    max_daily_loss_pct: float = 0.05
    max_trade_risk_pct: float = 0.005
    max_drawdown_pct: float = 0.20
    max_consecutive_losses: int = 5
    minimum_stop_distance: float = 1e-8
    max_position_size: float = 1.0
    max_open_positions: int = 3
    max_correlated_exposure: float = 2.0
    max_pair_exposure: float = 1.0
    max_spread: float = 0.0
    max_slippage: float = 0.0
    max_data_age_seconds: int = 3600
    database_url: str | None = field(default=None, repr=False)
    persistence_backend: str = "sqlite"
    sqlite_path: str = "astra.sqlite3"
    supabase_url: str | None = field(default=None, repr=False)
    supabase_publishable_key: str | None = field(default=None, repr=False)
    supabase_secret_key: str | None = field(default=None, repr=False)
    supabase_db_url: str | None = field(default=None, repr=False)
    supabase_use_secret: bool = False
    minimum_confidence: float = 0.5
    minimum_reward_risk: float = 1.0
    max_relative_volatility: float = 0.03
    kill_switch: bool = False
    emergency_halt: bool = False
    ml_enabled: bool = False
    ai_enabled: bool = False
    ai_provider: str = "openrouter"
    ai_api_key: str | None = field(default=None, repr=False)
    ai_model: str = "openrouter/free"
    ai_base_url: str = "https://openrouter.ai/api/v1"
    ai_timeout_seconds: float = 10.0
    ai_max_requests: int = 20

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and (not isfinite(value) or value < 0)
            ):
                raise ValueError(f"invalid configuration: {item.name}")
        if not 0 < self.max_trade_risk_pct <= 0.01:
            raise ValueError("per-trade risk must be in (0, 1%]")
        if not 0 < self.max_daily_loss_pct or not 0 < self.max_drawdown_pct <= 1:
            raise ValueError("loss and drawdown limits must be positive")
        if not 0 <= self.minimum_confidence <= 1 or self.minimum_stop_distance <= 0:
            raise ValueError("invalid confidence or stop distance")
        if self.persistence_backend not in {"memory", "sqlite", "postgres", "supabase"}:
            raise ValueError("unknown persistence backend")
        if self.ai_provider != "openrouter":
            raise ValueError("unknown AI provider")
        if self.ai_enabled and not self.ai_api_key:
            raise ValueError("ASTRA_AI_API_KEY is required when AI is enabled")
        if self.ai_timeout_seconds <= 0 or self.ai_max_requests < 1:
            raise ValueError("AI timeout and request limit must be positive")
        from urllib.parse import urlsplit

        ai_url = urlsplit(self.ai_base_url)
        if (
            ai_url.scheme != "https"
            or not ai_url.hostname
            or ai_url.username
            or ai_url.password
            or ai_url.query
            or ai_url.fragment
        ):
            raise ValueError("ASTRA_AI_BASE_URL must be a plain HTTPS URL")
        if not self.ai_model or len(self.ai_model) > 128:
            raise ValueError("invalid AI model identifier")

    def public(self) -> dict:
        """Never include credential values or credential-bearing connection URLs."""
        return {
            item.name: (
                bool(getattr(self, item.name))
                if not item.repr
                else getattr(self, item.name)
            )
            for item in fields(self)
        }

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # Plain environment variables remain sufficient in minimal deployments.
            pass

        def number(key: str, default: float) -> float:
            try:
                value = float(os.getenv(key, default))
                if not isfinite(value) or value < 0:
                    raise ValueError
                return value
            except ValueError:
                raise ValueError(f"invalid numeric configuration: {key}") from None

        return cls(
            environment=os.getenv("ASTRA_ENV", "development"),
            log_level=(
                os.getenv("ASTRA_LOG_LEVEL", "INFO").upper()
                if os.getenv("ASTRA_LOG_LEVEL", "INFO").upper()
                in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
                else "INFO"
            ),
            trading_enabled=os.getenv("ASTRA_TRADING_ENABLED", "false").lower()
            == "true",
            max_daily_loss_pct=min(number("ASTRA_MAX_DAILY_LOSS_PCT", 0.05), 0.05),
            max_trade_risk_pct=min(number("ASTRA_MAX_TRADE_RISK_PCT", 0.005), 0.01),
            max_drawdown_pct=number("ASTRA_MAX_DRAWDOWN_PCT", 0.20),
            max_consecutive_losses=int(number("ASTRA_MAX_CONSECUTIVE_LOSSES", 5)),
            max_data_age_seconds=int(number("ASTRA_MAX_DATA_AGE_SECONDS", 3600)),
            max_position_size=number("ASTRA_MAX_POSITION_SIZE", 1.0),
            max_open_positions=int(number("ASTRA_MAX_OPEN_POSITIONS", 3)),
            max_correlated_exposure=number("ASTRA_MAX_CORRELATED_EXPOSURE", 2.0),
            max_pair_exposure=number("ASTRA_MAX_PAIR_EXPOSURE", 1.0),
            max_spread=number("ASTRA_MAX_SPREAD", 0.0),
            max_slippage=number("ASTRA_MAX_SLIPPAGE", 0.0),
            database_url=os.getenv("ASTRA_DATABASE_URL") or None,
            persistence_backend=os.getenv("ASTRA_PERSISTENCE_BACKEND", "sqlite"),
            sqlite_path=os.getenv("ASTRA_SQLITE_PATH", "astra.sqlite3"),
            supabase_url=os.getenv("SUPABASE_URL") or None,
            supabase_publishable_key=os.getenv("SUPABASE_PUBLISHABLE_KEY") or None,
            supabase_secret_key=os.getenv("SUPABASE_SECRET_KEY") or None,
            supabase_db_url=os.getenv("SUPABASE_DB_URL") or None,
            supabase_use_secret=os.getenv("ASTRA_SUPABASE_USE_SECRET", "false").lower()
            == "true",
            minimum_confidence=number("ASTRA_MINIMUM_CONFIDENCE", 0.5),
            minimum_reward_risk=number("ASTRA_MINIMUM_REWARD_RISK", 1.0),
            max_relative_volatility=number("ASTRA_MAX_RELATIVE_VOLATILITY", 0.03),
            kill_switch=os.getenv("ASTRA_KILL_SWITCH", "false").lower() == "true",
            emergency_halt=os.getenv("ASTRA_EMERGENCY_HALT", "false").lower() == "true",
            ml_enabled=os.getenv("ASTRA_ML_ENABLED", "false").lower() == "true",
            ai_enabled=os.getenv("ASTRA_AI_ENABLED", "false").lower() == "true",
            ai_provider=os.getenv("ASTRA_AI_PROVIDER", "openrouter").lower(),
            ai_api_key=os.getenv("ASTRA_AI_API_KEY") or None,
            ai_model=os.getenv("ASTRA_AI_MODEL", "openrouter/free"),
            ai_base_url=os.getenv("ASTRA_AI_BASE_URL", "https://openrouter.ai/api/v1"),
            ai_timeout_seconds=number("ASTRA_AI_TIMEOUT_SECONDS", 10.0),
            ai_max_requests=int(number("ASTRA_AI_MAX_REQUESTS", 20)),
        )
