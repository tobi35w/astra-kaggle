# Independent risk policy

RiskEngine has final veto authority. It validates finite signal/risk inputs, positive price geometry, actual entry/stop/target reward-risk, equity, drawdown, daily losses, consecutive losses, position count, exposure, pair concentration, correlated exposure, spread, slippage, confidence, emergency halt and global kill switch. Strategy-supplied `risk_reward` is never trusted.

The maximum daily loss threshold is **5%** of UTC day-start equity, even when a caller supplies a higher daily setting. This is distinct from the default **0.5% per-trade risk**. Per-trade risk is capped at 1%; directly supplied invalid Settings fail validation. A smaller configured daily threshold is respected.

Sizing uses the smallest of per-trade budget, remaining daily budget and remaining drawdown budget, less reserved open stop risk. The denominator is stop distance times account-currency value per unit plus explicitly supplied execution costs. Position, pair and correlated size caps then apply. The direct risk API defaults additional execution cost to zero for compatibility; paper pipeline and backtester pass their actual round-trip cost assumptions.

Pending orders are risk-approved at creation and reapproved when their fill price is known. The portfolio derives realized/unrealized equity, consecutive losses, position count, pair exposure and reserved risk from its own ledger. Without a supplied empirical correlation model, other pairs are conservatively treated as correlated. `exposure.py` also exposes Pearson correlation and conservative exposure aggregation for callers with properly paired return histories. Size limits are generic instrument units, not inferred FX lots or margin limits; callers must supply correct account-currency conversion.

The immediate paper pipeline validates available data and full confidence/context before approval, then persists its decision before opening. A persistence failure vetoes new execution. Every NO TRADE includes explicit reason codes and reconstructable context when persistence is available. A wholly unavailable store cannot itself record its own outage; the returned veto remains available to the caller.

These policies constrain approved risk. A discontinuous price gap can exceed a stop-based budget; no exact realized loss guarantee is made. No broker or live order provider exists.
