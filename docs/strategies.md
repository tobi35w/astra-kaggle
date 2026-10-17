# Strategy library and adaptive selection

The original common `Strategy.name` / `generate(data, context)` protocol remains. Built-ins expose `StrategySpec`: identity, family, thesis, suitable/unsuitable regimes, required bars, setup, entry, stop/exit and risk assumptions. Signals supply directional price levels, confidence/evidence and invalidation. No strategy approves risk.

| Strategy | Distinct hypothesis / setup | Exit/invalidation |
|---|---|---|
| ema_trend | Persistent fast/slow EMA direction | ATR stop and configurable reward multiple |
| breakout | Close through preceding range, excluding current bar | ATR stop, 2R target; return into range invalidates thesis |
| pullback | Trend rejection near short mean | 1.25 ATR stop, 2R; structure invalidation |
| momentum | Eight-bar displacement exceeds ATR in trend direction | ATR stop and 2R |
| mean_reversion | Deviation from 20-bar mean in a range | ATR stop, mean target |
| range_fade | Outer 15% of ten-bar range | ATR stop, midpoint target, minimum 1R |
| price_action | Rejection wick exceeds twice candle body | Greater of ATR/range stop, 2R |
| volatility | Directional range exceeds 1.5 times preceding average | ATR stop and 2R |
| structure_continuation | Pullback to confirmed swing support inside an HH/HL structure | Stop beyond the swing low, 2R; close back below invalidates |
| session_opening_range | Breakout of the first bars' range in the London/NY overlap | 1.5 ATR stop, 2R; return into range invalidates |

Strategy thesis invalidations are explanatory. Execution implements the signal's stop/target and configured lifecycle; it does not dynamically reevaluate every thesis on each tick. All strategies are research hypotheses, not claims of demonstrated edge.

Selection first applies regime/data suitability and valid price geometry. Setup scoring weights confidence, actual reward-risk and regime match. Available pair/timeframe/regime fitness then adjusts ranking. Negative expectancy and degradation penalize candidates; multiple degradation modes can disable them. Future observations are excluded. Candidate disagreement reduces confidence and can veto all candidates; the strongest remaining candidate faces independent risk. No historical record means no invented history.

Arena fitness bounds expectancy relative to losses, drawdown, risk-adjusted results and chronological consistency. It discounts small samples and cost sensitivity, and penalizes fragile/insufficient holdout robustness and deterioration in scale-normalized holdout expectancy. This generalization gap is an overfitting warning, not proof of its cause. Return alone cannot win. Metrics include win rate, average win/loss, profit factor, net P&L, return percentage, drawdown, sample count, sessions, regimes, pair and timeframe. Evaluations and contextual fitness are persisted when a journal is supplied.

Performance memory records strategy, pair, timeframe, entry regime, UTC session, setup type, confidence bucket, outcome, expectancy, drawdown and degradation evidence. Baseline/recent comparisons examine expectancy, win rate, drawdown and cost sensitivity. Confidence has separate market, regime, strategy, setup, timeframe alignment and execution components, with STRONG/MODERATE/WEAK/CONFLICTED/INSUFFICIENT states. The score is an uncalibrated heuristic, not a probability.
