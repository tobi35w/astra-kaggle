from dataclasses import replace
from datetime import timedelta

import pytest

from astra.intelligence import confidence, detect_regime
from astra.market import structure, support_resistance_zones
from astra.models import MarketSnapshot, Regime, Side
from astra.strategies import StrategyRegistry, valid_signal


def _staircase(candles):
    """Zigzag staircase: rising peaks and troughs (HH/HL) ending in a pullback
    that holds above the last confirmed swing low."""
    out = []
    for index, candle in enumerate(candles):
        # Up legs of four bars alternate with down legs, with the final leg
        # descending so the last close sits between support and the prior close.
        leg = ((index // 4) + 1) % 2
        base = (
            1.10
            + 0.001 * (index // 4)
            + (
                0.001 * (index - 4 * (index // 4))
                if leg
                else -0.0005 * (index - 4 * (index // 4))
            )
        )
        opening, closing = base, base + (0.0002 if leg else -0.0002)
        out.append(
            replace(
                candle,
                open=opening,
                high=max(opening, closing) + 0.00005,
                low=min(opening, closing) - 0.00005,
                close=closing,
            )
        )
    return out


@pytest.mark.parametrize("name", [s.name for s in StrategyRegistry().all()])
@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_each_distinct_family_produces_causal_entry_stop_target(
    name, side, market_data
):
    data = market_data(40)
    candles = [
        replace(c, open=1.1, high=1.101, low=1.099, close=1.1) for c in data.candles
    ]
    regime = (
        Regime.RANGING if name in {"range_fade", "mean_reversion"} else Regime.TRENDING
    )
    volatility = 0.002
    if name in {"ema_trend", "momentum", "breakout"}:
        candles[-1] = replace(
            candles[-1], open=1.105, high=1.111, low=1.104, close=1.11
        )
    elif name == "pullback":
        candles[-1] = replace(candles[-1], open=1.0999, close=1.1001)
    elif name in {"range_fade", "mean_reversion"}:
        candles = [replace(c, high=1.2, low=1.0) for c in candles]
        candles[-1] = replace(candles[-1], open=1.01, close=1.01)
        volatility = 0.01
    elif name == "price_action":
        candles[-1] = replace(
            candles[-1], open=1.1, high=1.1002, low=1.09, close=1.1001
        )
    elif name == "volatility":
        candles[-1] = replace(candles[-1], open=1.10, high=1.13, low=1.07, close=1.11)
    elif name == "structure_continuation":
        candles = _staircase(candles)
    elif name == "session_opening_range":
        # Shift the fixture into the London/NY overlap (12:00-16:00 UTC) and
        # break out of the first four bars' range on the final bar.
        session_start = data.candles[0].timestamp.replace(hour=12, minute=0)
        candles = [
            replace(candle, timestamp=session_start + timedelta(minutes=5 * index))
            for index, candle in enumerate(candles)
        ]
        candles[-1] = replace(
            candles[-1], open=1.105, high=1.111, low=1.104, close=1.11
        )
    if side == Side.SHORT:
        candles = [
            replace(
                c,
                open=2.2 - c.open,
                high=2.2 - c.low,
                low=2.2 - c.high,
                close=2.2 - c.close,
            )
            for c in candles
        ]
    data = replace(data, candles=tuple(candles))
    context = MarketSnapshot(
        data.pair,
        data.timeframe,
        data.candles[-1].timestamp,
        side,
        0.5,
        volatility,
        regime,
        (),
    )
    strategy = StrategyRegistry().get(name)
    signal = strategy.generate(data, context)
    assert signal is not None and signal.side == side and valid_signal(signal)
    assert signal.entry == data.candles[-1].close
    spec = strategy.specification
    assert (
        spec.thesis
        and spec.setup
        and spec.entry
        and spec.stop
        and spec.exit
        and spec.risk_assumptions
    )
    assert Regime.INSUFFICIENT_DATA in spec.unsuitable_regimes


def test_structure_continuation_is_causal_and_requires_structure(market_data):
    from astra.strategies import StructureContinuation

    strategy = StructureContinuation()
    base = market_data(40)
    # A monotonic fixture has no confirmed swings, so no structural setup exists.
    assert strategy.generate(base, detect_regime(base)) is None

    stair = replace(base, candles=tuple(_staircase(base.candles)))
    signal = strategy.generate(stair, detect_regime(stair))
    assert signal is not None and valid_signal(signal)
    assert signal.side == Side.LONG
    assert signal.stop_loss < signal.entry < signal.take_profit
    assert signal.entry == stair.candles[-1].close

    # Future bars cannot change a decision made at the same decision time.
    future = tuple(
        replace(
            base.candles[index],
            timestamp=stair.candles[-1].timestamp + timedelta(minutes=5 * (index + 1)),
            open=base.candles[index].open + 1,
            high=base.candles[index].high + 1,
            low=base.candles[index].low + 1,
            close=base.candles[index].close + 1,
        )
        for index in range(5)
    )
    poisoned = replace(stair, candles=stair.candles + future)
    sliced = replace(
        poisoned,
        candles=tuple(
            c for c in poisoned.candles if c.timestamp <= stair.candles[-1].timestamp
        ),
    )
    assert sliced.candles == stair.candles
    assert strategy.generate(sliced, detect_regime(sliced)) == signal


def test_session_opening_range_requires_session_and_complete_range(market_data):
    from astra.strategies import SessionOpeningRange

    strategy = SessionOpeningRange(range_bars=4)
    base = market_data(40)
    overlap = base.candles[0].timestamp.replace(hour=12, minute=0)
    candles = [
        replace(candle, timestamp=overlap + timedelta(minutes=5 * index))
        for index, candle in enumerate(base.candles)
    ]
    data = replace(base, candles=tuple(candles))

    # Asia-session bars: no signal even after a large move.
    asia = base.candles[0].timestamp.replace(hour=2, minute=0)
    asia_candles = [
        replace(candle, timestamp=asia + timedelta(minutes=5 * index))
        for index, candle in enumerate(base.candles)
    ]
    asia_candles[-1] = replace(asia_candles[-1], close=1.13)
    assert (
        strategy.generate(
            replace(base, candles=tuple(asia_candles)), detect_regime(base)
        )
        is None
    )

    # Opening range incomplete: only four in-session bars so far.
    partial = tuple(
        replace(candle, timestamp=asia + timedelta(minutes=5 * index))
        for index, candle in enumerate(base.candles[:4])
    ) + tuple(candles[:4])
    assert (
        strategy.generate(replace(base, candles=partial), detect_regime(base)) is None
    )

    # Still inside the opening range: flat in-session candles produce nothing.
    flat = tuple(
        replace(candle, open=1.1, high=1.101, low=1.099, close=1.1)
        for candle in candles
    )
    assert strategy.generate(replace(base, candles=flat), detect_regime(base)) is None

    # A close beyond the opening range produces a geometric, valid breakout.
    broken = data.candles[:-1] + (
        replace(data.candles[-1], open=1.105, high=1.111, low=1.104, close=1.11),
    )
    signal = strategy.generate(replace(data, candles=broken), detect_regime(data))
    assert signal is not None and valid_signal(signal)
    assert signal.side == Side.LONG
    assert signal.stop_loss < signal.entry < signal.take_profit
    assert "opening range" in signal.reason

    with pytest.raises(ValueError, match="opening range needs at least one bar"):
        SessionOpeningRange(range_bars=0)


def test_structure_zones_failed_breakout_and_equal_pivots(market_data):
    data = market_data(40, oscillate=True)
    zones = support_resistance_zones(data, 0.001, 1, 1)
    assert zones and all(low <= high and count > 0 for low, high, count in zones)
    with pytest.raises(ValueError):
        support_resistance_zones(data, -1)
    prior_high = max(c.high for c in data.candles[-21:-1])
    last = replace(
        data.candles[-1],
        high=prior_high + 0.01,
        close=prior_high - 0.001,
        open=prior_high - 0.002,
        low=prior_high - 0.003,
    )
    assert (
        structure(replace(data, candles=data.candles[:-1] + (last,))).failed_breakout
        == "up"
    )


def test_regime_breakout_high_volatility_low_volatility_and_confidence(market_data):
    data = market_data(60, oscillate=True)
    last = data.candles[-1]
    breakout = replace(last, high=1.25, low=1.10, open=1.11, close=1.24)
    assert (
        detect_regime(replace(data, candles=data.candles[:-1] + (breakout,))).regime
        == Regime.BREAKOUT
    )
    quiet = replace(
        data,
        candles=tuple(
            replace(c, open=1.1, close=1.1, high=1.10001, low=1.09999)
            for c in data.candles
        ),
    )
    assert detect_regime(quiet).regime == Regime.LOW_VOLATILITY
    volatile = replace(
        data,
        candles=tuple(
            replace(c, high=c.close + 0.05, low=c.close - 0.05) for c in data.candles
        ),
    )
    assert detect_regime(volatile).regime == Regime.VOLATILE
    for value, state in (
        (0.9, "STRONG"),
        (0.6, "MODERATE"),
        (0.4, "WEAK"),
        (0.1, "INSUFFICIENT"),
    ):
        result = confidence(
            value, value, value, value, value, timeframe_alignment=value
        )
        assert result.state == state
    with pytest.raises(ValueError):
        confidence(1.1, 0.5, 0.5, 0.5, 0.5)
