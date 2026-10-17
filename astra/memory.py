"""Causal performance memory and confidence outcome summaries."""

from dataclasses import replace
from datetime import datetime

from .models import StrategyPerformance
from .repositories import Entity, Journal


class PerformanceMemory:
    def __init__(self, journal: Journal):
        self.journal = journal

    def record(self, performance: StrategyPerformance, **keys):
        self.journal.record(
            Entity.PERFORMANCE,
            performance.observed_at,
            performance,
            pair=performance.pair,
            timeframe=performance.timeframe,
            strategy=performance.strategy,
            **keys,
        )

    def history(self, pair="", timeframe="", as_of=None, *, source=None):
        records = self.journal.repository.query(
            Entity.PERFORMANCE, pair=pair, timeframe=timeframe, as_of=as_of
        )
        result = []
        for record in records:
            value = dict(record.payload)
            value["observed_at"] = datetime.fromisoformat(value["observed_at"])
            # Both envelope availability and payload availability must be causal.
            if (as_of is None or value["observed_at"] <= as_of) and (
                source is None or value.get("source", "LEGACY") == source
            ):
                result.append(StrategyPerformance(**value))
        return tuple(
            sorted(result, key=lambda r: (r.observed_at, r.experiment_id, r.sequence))
        )

    def observe_trade(self, trade, *, regime=None, confidence=None, setup_type=None):
        from .market import session_context

        regime = regime or trade.regime
        confidence = trade.confidence if confidence is None else confidence
        setup_type = setup_type or trade.setup_type

        bucket = (
            "unknown"
            if confidence is None
            else "high"
            if confidence >= 0.75
            else "medium"
            if confidence >= 0.55
            else "low"
        )
        record = StrategyPerformance(
            trade.strategy,
            regime,
            trade.pair,
            trade.timeframe,
            1,
            trade.pnl,
            0.0,
            trade.closed_at,
            session=session_context(trade.opened_at),
            setup_type=setup_type,
            confidence_bucket=bucket,
            win_rate=float(trade.pnl > 0),
            outcome=trade.pnl,
            source="PAPER",
            metric_unit="account_currency",
            drawdown_unit="account_currency",
            experiment_id=getattr(trade, "portfolio_id", ""),
            sequence=trade.position_id,
        )
        from uuid import NAMESPACE_URL, uuid5

        from .evidence import digest

        identity = "astra:paper:" + digest(trade)
        self.record(record, record_id=str(uuid5(NAMESPACE_URL, identity)))
        self.journal.record(
            Entity.CALIBRATION,
            trade.closed_at,
            {
                "score": confidence,
                "bucket": bucket,
                "outcome": trade.pnl,
                "meaning": "observed outcome; score is not a calibrated probability",
            },
            pair=trade.pair,
            timeframe=trade.timeframe,
            strategy=trade.strategy,
            record_id=str(uuid5(NAMESPACE_URL, identity + ":calibration")),
        )

    def summarize(self, pair, timeframe, as_of):
        groups: dict[tuple[str, ...], list[StrategyPerformance]] = {}
        for record in self.history(pair, timeframe, as_of):
            groups.setdefault(
                (
                    record.strategy,
                    record.regime,
                    record.session,
                    record.setup_type,
                    record.confidence_bucket,
                    record.source,
                    record.metric_unit,
                    record.drawdown_unit,
                    record.experiment_id,
                ),
                [],
            ).append(record)
        results = []
        for records in groups.values():
            count = sum(r.trades for r in records)
            if not count:
                continue
            running = peak = drawdown = 0.0
            for r in records:
                running += r.expectancy * r.trades
                peak = max(peak, running)
                drawdown = max(drawdown, peak - running)
            granular = all(r.outcome is not None for r in records)
            results.append(
                replace(
                    records[-1],
                    trades=count,
                    expectancy=sum(r.expectancy * r.trades for r in records) / count,
                    win_rate=sum(r.win_rate * r.trades for r in records) / count,
                    max_drawdown=drawdown
                    if granular
                    else max(r.max_drawdown for r in records),
                    drawdown_unit=records[-1].metric_unit
                    if granular
                    else records[-1].drawdown_unit,
                )
            )
        return tuple(results)
