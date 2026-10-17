from scripts.analyze_curated_history_losses import (
    summarize_results,
    summarize_trades,
)


def test_summarize_results_pairs_cost_scenarios_without_mixing_them():
    base = {
        "pair": "EURUSD",
        "timeframe": "H1",
        "window_id": "window-a",
        "strategy": "ml_direction",
        "cost_scenario": "base",
        "net_return_pct": 1.0,
        "trade_count": 25,
    }
    double = base | {
        "cost_scenario": "double_cost",
        "net_return_pct": -0.5,
        "trade_count": 20,
    }

    summary = summarize_results([base, double])

    assert summary["by_cost_scenario"]["base"]["positive_runs"] == 1
    assert summary["by_cost_scenario"]["double_cost"]["negative_runs"] == 1
    assert summary["by_cost_scenario"]["base"]["runs_below_30_trades"] == 1
    assert (
        summary["paired_cost_sensitivity"]["median_return_change_double_minus_base_pp"]
        == -1.5
    )
    assert (
        summary["paired_cost_sensitivity"]["positive_base_to_nonpositive_double"] == 1
    )


def test_summarize_trades_uses_r_units_for_mixed_native_currencies():
    def trade_row(pair, pnl):
        return {
            "strategy": "ml_direction",
            "payload": {
                "experiment_id": f"base:{pair}:H1:window:ml_direction",
                "trade": {
                    "pnl_per_unit": pnl / 10,
                    "risk_per_unit": 1,
                    "size": 10,
                    "reason": "take_profit" if pnl > 0 else "stop_loss",
                    "regime": "TRENDING",
                    "side": "LONG",
                },
                "pnl": pnl,
            },
        }

    summary = summarize_trades([trade_row("EURUSD", 20), trade_row("USDJPY", -10)])
    base = summary["by_cost_scenario"]["base"]

    assert base["trade_count"] == 2
    assert base["wins"] == 1
    assert base["mean_r"] == 0.5
    assert base["exit_reasons"] == {"stop_loss": 1, "take_profit": 1}
