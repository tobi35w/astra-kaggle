"""Source-separated, idempotent research evidence through the existing Journal."""

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .models import StrategyPerformance
from .repositories import Entity, Record, json_value


def digest(value):
    return sha256(
        json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def code_version():
    root = Path(__file__).parent
    return digest(
        {
            p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


def record_experiment(
    journal,
    data,
    strategy,
    report,
    *,
    source="BACKTEST",
    initial_equity=10000,
    fixed_fraction=0.005,
    start_index=None,
    risk_engine=None,
    available_at=None,
):
    if source not in ("BACKTEST", "OOS"):
        raise ValueError("research evidence source must be BACKTEST or OOS")
    at = available_at or datetime.now(UTC)
    if at.tzinfo is None or at.utcoffset() is None or at < data.candles[-1].timestamp:
        raise ValueError("research evidence cannot predate its final candle")
    configuration = {
        "source": source,
        "dataset": data.metadata,
        "candle_sha256": digest(data.candles),
        "strategy": strategy.name,
        "parameters": asdict(strategy)
        if is_dataclass(strategy) and not isinstance(strategy, type)
        else vars(strategy),
        "costs": report.assumptions,
        "execution": report.execution_assumptions,
        "initial_equity": initial_equity,
        "fixed_fraction": fixed_fraction,
        "risk": risk_engine.settings.public()
        if risk_engine
        else "default research risk",
        "start_index": start_index,
        "code_version": code_version(),
    }
    identity = digest(configuration)
    keys = {
        "pair": data.pair,
        "timeframe": data.timeframe,
        "strategy": strategy.name,
        "correlation_id": identity,
    }

    def record(entity, value, suffix):
        return Record(
            entity,
            at,
            json_value(value),
            **keys,
            record_id=str(uuid5(NAMESPACE_URL, "astra:" + identity + suffix)),
        )

    records = [
        record(
            Entity.AUDIT,
            {
                "kind": "research_experiment",
                "experiment_id": identity,
                "configuration": configuration,
                "metrics": report.metrics,
                "available_at": at,
            },
            ":experiment",
        )
    ]
    running = peak = 0.0
    for i, trade in enumerate(report.trades):
        costs = report.assumptions
        exit_cost = costs.slippage + (
            costs.spread / 2 if costs.price_basis == "mid" else 0
        )
        risk = (
            trade.risk_per_unit + exit_cost
        ) * costs.value_per_price_unit + costs.commission_per_unit
        if risk <= 0:
            raise ValueError("learning requires positive independently calculated risk")
        multiple = trade.pnl_per_unit / risk
        running += multiple
        peak = max(peak, running)
        performance = StrategyPerformance(
            strategy.name,
            trade.regime,
            data.pair,
            data.timeframe,
            1,
            multiple,
            peak - running,
            at,
            win_rate=float(trade.pnl > 0),
            outcome=multiple,
            source=source,
            metric_unit="R",
            drawdown_unit="R",
            dataset_version=data.metadata.get(
                "dataset_version", "sha256:" + configuration["candle_sha256"]
            ),
            experiment_id=identity,
            sequence=i,
        )
        records.append(record(Entity.PERFORMANCE, performance, f":performance:{i}"))
        records.append(
            record(
                Entity.AUDIT,
                {
                    "kind": "trade_learning",
                    "source": source,
                    "experiment_id": identity,
                    "trade": trade,
                    "pnl": trade.pnl,
                    "pnl_per_unit": trade.pnl_per_unit,
                    "r_multiple": multiple,
                    "risk_per_unit_account_currency": risk,
                    "duration_bars": trade.exit_index - trade.entry_index,
                    "available_at": at,
                },
                f":trade:{i}",
            )
        )
    journal.repository.append_batch(tuple(records))
    return identity


def comparable_evidence(records):
    """Use one source/unit/experiment stream per strategy and market context.

    PAPER takes priority over OOS, then BACKTEST, then legacy records. Latest
    completed experiment wins within a source; reruns never inflate sample size.
    """
    priority = {"LEGACY": 0, "BACKTEST": 1, "OOS": 2, "PAPER": 3}
    groups: dict[tuple, dict[tuple, list[StrategyPerformance]]] = {}
    for item in records:
        key = (item.strategy, item.pair, item.timeframe, item.regime, item.session)
        stream = (item.source, item.metric_unit, item.drawdown_unit, item.experiment_id)
        groups.setdefault(key, {}).setdefault(stream, []).append(item)
    result = []
    for streams in groups.values():
        chosen = max(
            streams,
            key=lambda s: (
                priority.get(s[0], -1),
                max(r.observed_at for r in streams[s]),
                s,
            ),
        )
        result.extend(streams[chosen])
    return tuple(result)
