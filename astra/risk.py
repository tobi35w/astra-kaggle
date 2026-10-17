"""Independent risk veto; strategies cannot override limits or kill switch."""

from __future__ import annotations

from math import isfinite

from .config import Settings
from .models import RiskApproval, RiskState, Signal
from .strategies import risk_reward, valid_signal


class RiskEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def approve(
        self,
        signal: Signal,
        state: RiskState,
        spread: float = 0.0,
        data_valid: bool = True,
        expected_slippage: float = 0.0,
        value_per_price_unit: float = 1.0,
        execution_cost_per_unit: float = 0.0,
    ) -> RiskApproval:
        reasons = []
        try:
            self.settings.validate()
        except ValueError:
            return RiskApproval(False, ("configuration invalid",))
        if not valid_signal(signal):
            return RiskApproval(False, ("strategy output is malformed",))
        if not all(
            isfinite(value)
            for value in (
                state.equity,
                state.day_start_equity,
                state.peak_equity,
                state.open_exposure,
                state.correlated_exposure,
                state.pair_exposure,
                state.open_risk,
                state.open_positions,
                state.consecutive_losses,
                spread,
                expected_slippage,
                value_per_price_unit,
                execution_cost_per_unit,
            )
        ):
            return RiskApproval(False, ("risk input contains non-finite values",))
        if (
            state.open_positions < 0
            or state.consecutive_losses < 0
            or min(
                state.open_exposure,
                state.correlated_exposure,
                state.pair_exposure,
                state.open_risk,
                execution_cost_per_unit,
            )
            < 0
        ):
            return RiskApproval(
                False, ("risk state contains negative exposure or counters",)
            )
        if value_per_price_unit <= 0:
            return RiskApproval(False, ("price value conversion must be positive",))
        if not self.settings.trading_enabled:
            reasons.append("trading disabled; paper decisions only")
        if state.halted or self.settings.kill_switch:
            reasons.append("global kill switch is active")
        if self.settings.emergency_halt:
            reasons.append("emergency halt is active")
        if not data_valid:
            reasons.append("market data failed validation")
        if signal.confidence < self.settings.minimum_confidence:
            reasons.append("signal confidence below minimum threshold")
        if spread < 0 or spread > self.settings.max_spread:
            reasons.append("spread exceeds configured limit")
        if expected_slippage < 0 or expected_slippage > self.settings.max_slippage:
            reasons.append("expected slippage exceeds configured limit")
        if state.equity <= 0 or state.day_start_equity <= 0 or state.peak_equity <= 0:
            reasons.append("equity is invalid")
        elif (state.day_start_equity - state.equity) / state.day_start_equity >= min(
            self.settings.max_daily_loss_pct, 0.05
        ):
            reasons.append("daily loss limit reached")
        if (
            state.peak_equity > 0
            and (state.peak_equity - state.equity) / state.peak_equity
            >= self.settings.max_drawdown_pct
        ):
            reasons.append("drawdown limit reached")
        if state.consecutive_losses >= self.settings.max_consecutive_losses:
            reasons.append("consecutive-loss protection triggered")
        stop = abs(signal.entry - signal.stop_loss)
        if stop < self.settings.minimum_stop_distance:
            reasons.append("stop distance is invalid or too small")
        if risk_reward(signal) < self.settings.minimum_reward_risk:
            reasons.append("expected risk/reward is below 1:1")
        if state.open_exposure >= self.settings.max_position_size:
            reasons.append("position exposure limit reached")
        if state.open_positions >= self.settings.max_open_positions:
            reasons.append("maximum open positions reached")
        if state.correlated_exposure >= self.settings.max_correlated_exposure:
            reasons.append("correlated exposure limit reached")
        if state.pair_exposure >= self.settings.max_pair_exposure:
            reasons.append("pair concentration limit reached")
        if reasons:
            return RiskApproval(False, tuple(reasons))
        # Keep per-trade risk meaningfully below the immutable 5% daily ceiling.
        risk_budget = state.equity * min(
            self.settings.max_trade_risk_pct, self.settings.max_daily_loss_pct, 0.01
        )
        daily_remaining = max(
            0.0,
            state.equity
            - state.day_start_equity * (1 - min(self.settings.max_daily_loss_pct, 0.05))
            - state.open_risk,
        )
        drawdown_remaining = max(
            0.0,
            state.equity
            - state.peak_equity * (1 - self.settings.max_drawdown_pct)
            - state.open_risk,
        )
        risk_budget = min(risk_budget, daily_remaining, drawdown_remaining)
        size = min(
            risk_budget / (stop * value_per_price_unit + execution_cost_per_unit),
            self.settings.max_position_size - state.open_exposure,
            self.settings.max_correlated_exposure - state.correlated_exposure,
            self.settings.max_pair_exposure - state.pair_exposure,
        )
        if size <= 0:
            return RiskApproval(False, ("calculated position size is zero",))
        return RiskApproval(True, ("all configured risk checks passed",), size)
