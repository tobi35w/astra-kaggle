"""Evidence-labelled pre-MT5 readiness snapshot without remote side effects."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def readiness_report(
    settings,
    repository,
    *,
    dataset_root: str | None = None,
    pair: str = "EURUSD",
    timeframe: str = "H4",
    campaign_path: str | None = None,
    database_report_path: str | None = None,
) -> dict:
    backend = settings.persistence_backend
    local_backend = backend in {"memory", "sqlite"}
    durable_backend = backend == "sqlite"
    local_healthy = repository.healthy() if local_backend else False
    checks = {
        "configuration": {
            "status": "PASS — TESTED LOCALLY",
            "persistence_backend": backend,
        },
        "risk_safety": {
            "status": "PASS — TESTED LOCALLY"
            if not settings.trading_enabled
            and settings.max_daily_loss_pct <= 0.05
            and settings.max_trade_risk_pct <= 0.005
            and settings.max_drawdown_pct <= 0.20
            and settings.max_open_positions <= 3
            and settings.max_consecutive_losses <= 5
            else "FAIL",
            "trading_enabled": settings.trading_enabled,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "max_trade_risk_pct": settings.max_trade_risk_pct,
            "max_drawdown_pct": settings.max_drawdown_pct,
            "max_open_positions": settings.max_open_positions,
            "max_consecutive_losses": settings.max_consecutive_losses,
            "kill_switch": settings.kill_switch,
            "emergency_halt": settings.emergency_halt,
        },
        "journal_liveness": {
            "status": "PASS — TESTED LOCALLY"
            if local_healthy and durable_backend
            else "BLOCKED"
            if not local_backend
            else "NOT RUN",
            "backend": backend,
            "durable": durable_backend,
            "detail": "Memory storage is volatile."
            if backend == "memory"
            else "Remote health requires an explicit hosted verification report."
            if not local_backend
            else "",
        },
        "hosted_schema_and_round_trip": {
            "status": "NOT RUN",
            "instruction": "Run `python -m astra verify-database` and provide its saved JSON report.",
        },
        "historical_dataset": {"status": "NOT RUN"},
        "historical_scope": {
            "status": "BLOCKED",
            "detail": "Curated history covers four active pairs in separate quality-checked windows; failed months and gaps prevent a continuous four-pair archive.",
        },
        "research_campaign": {"status": "NOT RUN"},
        "historical_backtest": {"status": "NOT RUN"},
        "chronological_holdout": {"status": "NOT RUN"},
        "walk_forward": {"status": "NOT RUN"},
        "cost_stress": {"status": "NOT RUN"},
        "robustness": {"status": "NOT RUN"},
        "paper_replay": {"status": "NOT RUN"},
        "forward_paper_testing": {
            "status": "NOT RUN",
            "detail": "No genuine chronological paper observation on a live-updating market feed is recorded.",
        },
        "ai": {
            "status": "PASS — TESTED LOCALLY" if not settings.ai_enabled else "NOT RUN",
            "enabled": settings.ai_enabled,
            "provider": settings.ai_provider,
            "model": settings.ai_model,
        },
        "execution_boundary": {
            "status": "PASS — TESTED LOCALLY",
            "mt5": "NOT INTEGRATED",
            "live_trading": "NOT IMPLEMENTED",
            "execution_provider": "paper only",
        },
    }
    if dataset_root:
        try:
            from .historical.dataset import load_dataset

            data = load_dataset(dataset_root, pair, timeframe)
            checks["historical_dataset"] = {
                "status": "PASS — VERIFIED",
                "pair": data.pair,
                "timeframe": data.timeframe,
                "candles": len(data.candles),
                "first_available": data.candles[0].timestamp,
                "last_available": data.candles[-1].timestamp,
                "dataset_version": data.metadata.get("dataset_version", ""),
                "quality_status": data.metadata.get("quality_status", ""),
                "price_basis": data.metadata.get("price_basis", "unspecified"),
            }
        except Exception as error:  # noqa: BLE001 - report bounded local validation only.
            checks["historical_dataset"] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "detail": "Managed dataset failed verification; inspect its quality report.",
            }
    if campaign_path:
        try:
            campaign = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
            results = campaign.get("results")
            valid = (
                campaign.get("status") == "COMPLETED"
                and isinstance(results, list)
                and bool(results)
                and all(isinstance(item, dict) for item in results)
            )
            checks["research_campaign"] = {
                "status": "PASS — VERIFIED" if valid else "FAIL",
                "experiment_id": campaign.get("experiment_id", ""),
                "completed_at": campaign.get("completed_at", ""),
                "experiment_count": len(results) if isinstance(results, list) else 0,
            }
            if valid:
                first = results[0]
                if isinstance(first.get("backtest"), dict):
                    checks["historical_backtest"] = {
                        "status": "PASS — VERIFIED",
                        "metrics": first["backtest"],
                    }
                robustness = first.get("robustness")
                if isinstance(robustness, dict):
                    holdout = robustness.get("out_of_sample_metrics")
                    if isinstance(holdout, dict):
                        checks["chronological_holdout"] = {
                            "status": "PASS — VERIFIED",
                            "metrics": holdout,
                        }
                    stress = robustness.get("stressed_metrics")
                    if isinstance(stress, list):
                        checks["cost_stress"] = {
                            "status": "PASS — VERIFIED",
                            "scenarios": len(stress),
                            "metrics": stress,
                        }
                    checks["robustness"] = {
                        "status": "PASS — VERIFIED",
                        "state": robustness.get("state", "UNKNOWN"),
                        "cost_sensitivity": robustness.get("cost_sensitivity"),
                        "generalization_gap": robustness.get("generalization_gap"),
                    }
                folds = first.get("walk_forward")
                if isinstance(folds, list) and folds:
                    checks["walk_forward"] = {
                        "status": "PASS — VERIFIED",
                        "folds": len(folds),
                        "out_of_sample_trades": sum(
                            item.get("out_of_sample_metrics", {}).get("trade_count", 0)
                            for item in folds
                            if isinstance(item, dict)
                        ),
                    }
                paper = first.get("paper")
                if isinstance(paper, dict):
                    checks["paper_replay"] = {
                        "status": "PASS — VERIFIED",
                        "type": "finite historical replay",
                        "mode": paper.get("mode", ""),
                        "trades": paper.get("trades", 0),
                        "orders": paper.get("orders", 0),
                        "open_positions": paper.get("open_positions", 0),
                        "approvals_enabled": paper.get("approvals_enabled", False),
                    }
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            KeyError,
        ) as error:
            checks["research_campaign"] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "detail": "Campaign report could not be read or validated.",
            }
    if database_report_path:
        try:
            db = json.loads(Path(database_report_path).read_text(encoding="utf-8"))
            verified = (
                db.get("status") == "VERIFIED"
                and db.get("schema") == "VERIFIED"
                and db.get("journal_round_trip") == "VERIFIED"
            )
            checks["hosted_schema_and_round_trip"] = {
                "status": "PASS — VERIFIED" if verified else "BLOCKED",
                "checked_at": db.get("checked_at", ""),
                "schema": db.get("schema", "NOT_CHECKED"),
                "journal_round_trip": db.get("journal_round_trip", "NOT_CHECKED"),
                "migration_history": "NOT INSPECTED",
                "migration_applied_by_verifier": db.get("migrations_applied", False),
                "audit_record_id": db.get("audit_record_id", ""),
                "reason": db.get("reason", ""),
            }
            if verified and not local_backend:
                checks["journal_liveness"] = {
                    "status": "PASS — VERIFIED",
                    "backend": backend,
                    "durable": True,
                    "detail": "Hosted Journal schema and idempotent write/read round trip verified.",
                }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            checks["hosted_schema_and_round_trip"] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "detail": "Database verification report could not be read.",
            }
    required = (
        "configuration",
        "risk_safety",
        "journal_liveness",
        "hosted_schema_and_round_trip",
        "historical_dataset",
        "historical_scope",
        "research_campaign",
        "historical_backtest",
        "chronological_holdout",
        "walk_forward",
        "cost_stress",
        "robustness",
        "paper_replay",
        "forward_paper_testing",
    )
    ready = all(checks[key]["status"].startswith("PASS") for key in required)
    return {
        "overall": "READY_FOR_MT5_DEMO_INTEGRATION"
        if ready
        else "NOT_READY_FOR_MT5_DEMO_INTEGRATION",
        "checked_at": datetime.now(UTC),
        "checks": checks,
        "profitability": "UNPROVEN",
    }
