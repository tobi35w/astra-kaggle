"""Forward paper testing over a managed dataset root.

Runs the risk-gated paper replay chronologically over every requested pair and
writes a combined report. Intended for the post-curation tail first (the only
data that counts as new evaluation evidence) and for rehearsals on curated
windows. This is a finite historical replay, not live trading.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from astra.backtest import BacktestCosts
from astra.config import Settings
from astra.historical.dataset import load_dataset
from astra.pipeline import replay_paper
from astra.repositories import Entity, Journal, MemoryRepository
from astra.strategies import StrategyRegistry

ACTIVE_SYMBOLS = ("AUDUSD", "EURUSD", "GBPUSD", "USDJPY")
CAVEAT = (
    "Finite historical replay over a managed dataset. Outcomes describe this "
    "dataset only and do not establish future performance."
)


def run(
    root,
    *,
    symbols=ACTIVE_SYMBOLS,
    timeframe: str = "H4",
    spread_pips: float = 1.0,
    slippage_pips: float = 0.2,
    commission: float = 0.0,
    price_value: float = 1.0,
    equity: float = 10_000.0,
    price_basis: str = "bid",
    journal=None,
) -> dict:
    """Replay paper decisions over each pair's managed dataset.

    Failures are isolated per symbol; the run always returns a report.
    """
    registry = StrategyRegistry()
    strategies = list(registry.all())
    journal = journal or Journal(MemoryRepository())
    settings = replace(
        Settings(),
        trading_enabled=True,
        max_spread=spread_pips * 0.01,
        max_slippage=slippage_pips * 0.01,
    )
    results = {}
    for symbol in symbols:
        jpy = symbol.endswith("JPY")
        pip = 0.01 if jpy else 0.0001
        costs = BacktestCosts(
            spread_pips * pip,
            slippage_pips * pip,
            commission,
            price_value,
            price_basis,
        )
        try:
            data = load_dataset(root, symbol, timeframe)
            replay = replay_paper(
                data, strategies, settings, costs, equity, journal=journal
            )
        except Exception as error:  # noqa: BLE001 - isolate per-symbol failures
            results[symbol] = {"status": "FAILED", "error_type": type(error).__name__}
            continue
        trades = replay["trades"]
        wins = sum(1 for trade in trades if trade.pnl > 0)
        results[symbol] = {
            "status": "COMPLETED",
            "candles": len(data.candles),
            "window": [
                data.candles[0].timestamp.isoformat(),
                data.candles[-1].timestamp.isoformat(),
            ],
            "decisions": len(replay["decisions"]),
            "trade_decisions": sum(
                1
                for decision in replay["decisions"]
                if decision.action.value == "TRADE"
            ),
            "orders": len(replay["orders"]),
            "filled_orders": sum(
                1 for order in replay["orders"] if order.status.value == "FILLED"
            ),
            "trades": len(trades),
            "wins": wins,
            "net_pnl": round(sum(trade.pnl for trade in trades), 2),
            "final_equity": round(replay["equity"], 2),
            "dataset_version": data.metadata.get("dataset_version", ""),
        }
        journal.record(
            Entity.AUDIT,
            data.candles[-1].timestamp,
            {
                "kind": "forward_paper_test",
                "symbol": symbol,
                "timeframe": timeframe,
                **results[symbol],
            },
            pair=symbol,
            timeframe=timeframe,
        )
    completed = [item for item in results.values() if item["status"] == "COMPLETED"]
    return {
        "status": "COMPLETED" if len(completed) == len(symbols) else "PARTIAL",
        "dataset_root": str(root),
        "timeframe": timeframe,
        "symbols": results,
        "assumptions": {
            "equity": equity,
            "spread_pips": spread_pips,
            "slippage_pips": slippage_pips,
            "commission_per_unit": commission,
            "price_value": price_value,
            "price_basis": price_basis,
            "strategies": [strategy.name for strategy in strategies],
        },
        "caveat": CAVEAT,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--symbols", nargs="+", default=list(ACTIVE_SYMBOLS))
    parser.add_argument("--timeframe", default="H4", choices=("M5", "M15", "H1", "H4"))
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--price-value", type=float, default=1.0)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    try:
        report = run(
            args.dataset_root,
            symbols=tuple(args.symbols),
            timeframe=args.timeframe,
            spread_pips=args.spread_pips,
            slippage_pips=args.slippage_pips,
            commission=args.commission,
            price_value=args.price_value,
            equity=args.equity,
        )
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "ERROR", "error_type": type(error).__name__}))
        return 2
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps({"status": report["status"], "symbols": report["symbols"]}, indent=2)
    )
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
