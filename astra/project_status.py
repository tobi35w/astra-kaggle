"""Conservative project assessment from saved local research evidence.

This is a view of recorded evidence, not a new market-data validation, hosted
database check, model qualification, or estimate of project completion.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_invalid_number
        )
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _invalid_number(value):
    raise ValueError("Non-finite number in saved evidence")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _count(value):
    number = _number(value)
    return (
        int(number)
        if number is not None and number >= 0 and number == int(number)
        else None
    )


def _cost_summary(report: dict) -> dict:
    """Weight per-group mean returns by runs; never add USD and JPY cash P&L."""
    groups = report.get("summary", {})
    if not isinstance(groups, dict):
        return {}
    summaries = {}
    for scenario in ("base", "double_cost"):
        runs = trades = 0
        weighted_return = 0.0
        for key, group in groups.items():
            if not str(key).endswith("|" + scenario) or not isinstance(group, dict):
                continue
            count = _count(group.get("window_runs"))
            mean_return = _number(group.get("mean_return_pct"))
            trade_count = _count(group.get("total_closed_trades"))
            if not count or mean_return is None or trade_count is None:
                return {}
            runs += count
            trades += trade_count
            weighted_return += mean_return * count
        if runs:
            if _number(weighted_return) is None:
                return {}
            summaries[scenario] = {
                "runs": runs,
                "trades": trades,
                "mean_return_pct": weighted_return / runs,
            }
    return summaries


def project_assessment(root: Path) -> dict:
    """Summarize the research stage without treating file existence as success."""
    data = root / "data"
    curated = data / "training-arena" / "curated-history-h1-h4-v1"
    report = _read(curated / "evaluation-report.json")
    diagnostics = _read(curated / "loss-diagnostics.json")
    research = _read(data / "astra-training-four-pairs" / "research-progress.json")
    acquisition = _read(
        data / "astra-continuous-2026-tail" / "acquisition-summary.json"
    )
    costs = _cost_summary(report)
    report_count = _count(report.get("test_count"))
    summarized_runs = sum(value["runs"] for value in costs.values())
    complete = (
        report.get("status") == "COMPLETED"
        and report_count is not None
        and report_count > 0
        and summarized_runs == report_count
        and set(costs) == {"base", "double_cost"}
    )
    matching_diagnostics = (
        complete
        and bool(report.get("campaign_id"))
        and diagnostics.get("campaign_id") == report.get("campaign_id")
        and _count(diagnostics.get("result_count")) == report_count
    )
    warnings = []
    if report and not complete:
        warnings.append(
            "The saved curated report is incomplete or its result counts do not reconcile."
        )
    if diagnostics and not matching_diagnostics:
        warnings.append(
            "The saved loss diagnosis does not match a complete curated report."
        )
    completed = _count(research.get("completed"))
    total = _count(research.get("total"))
    if completed is not None and total is not None and completed > total:
        warnings.append(
            f"Research recorded {completed:,} completed against {total:,} planned. "
            "The display caps progress; the original evidence remains unchanged."
        )

    adverse = complete and any(
        value["mean_return_pct"] <= 0 for value in costs.values()
    )
    rejected = (
        matching_diagnostics and diagnostics.get("promotion_status") == "NOT_QUALIFIED"
    )
    promotion = "NOT_QUALIFIED" if adverse or rejected else "NOT_ASSESSED"
    stage = "RESEARCH_VALIDATION" if complete else "EVIDENCE_COLLECTION"
    next_action = (
        "Reserve a fresh, contiguous, quality-checked holdout before tuning the next candidate."
        if complete
        else "Complete and reconcile quality-checked research evidence before selecting a candidate."
    )
    evidence = [
        {
            "id": "curated",
            "label": "Curated historical validation",
            "status": "RECORDED" if complete else "INCOMPLETE" if report else "MISSING",
            "detail": (
                f"{report_count:,} saved runs across base and doubled costs. "
                "Historical windows are consumed evidence."
                if complete
                else "A complete, internally consistent evaluation report is required."
            ),
            "recorded_at": report.get("completed_at"),
        },
        {
            "id": "candidate",
            "label": "Candidate qualification",
            "status": promotion,
            "detail": (
                "The recorded baseline does not establish a repeatable edge after costs."
                if promotion == "NOT_QUALIFIED"
                else "Completed tests alone do not qualify a model for use."
            ),
            "recorded_at": diagnostics.get("created_at")
            if matching_diagnostics
            else None,
        },
        {
            "id": "acquisition",
            "label": "Fresh historical data",
            "status": "NEEDS_REVIEW" if acquisition else "MISSING",
            "detail": (
                f"Last saved acquisition status: {acquisition.get('status', 'UNKNOWN')}. "
                "Download counts alone do not verify a continuous research window."
            ),
            "recorded_at": acquisition.get("acquired_at"),
        },
        {
            "id": "confirmation",
            "label": "Independent confirmation",
            "status": "NOT_ESTABLISHED",
            "detail": "These saved artifacts do not establish a reserved, untouched confirmation period or ongoing forward paper results.",
            "recorded_at": None,
        },
    ]
    return {
        "assessed_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "stage_label": "Research validation" if complete else "Evidence collection",
        "headline": "Validate an edge before advancing"
        if complete
        else "Build a trustworthy research baseline",
        "summary": (
            "Astra has a working research and paper engine. Saved evaluations are complete, "
            "but the baseline has not qualified for promotion."
            if promotion == "NOT_QUALIFIED"
            else "Astra's next development decision depends on consistent historical evidence and independent validation."
        ),
        "next_action": next_action,
        "profitability": "UNPROVEN",
        "promotion": promotion,
        "metrics": {
            "curated_runs": report_count if complete else None,
            "trade_records": sum(value["trades"] for value in costs.values())
            if complete
            else None,
            "base_mean_return_pct": costs.get("base", {}).get("mean_return_pct")
            if complete
            else None,
            "double_cost_mean_return_pct": costs.get("double_cost", {}).get(
                "mean_return_pct"
            )
            if complete
            else None,
        },
        "evidence": evidence,
        "next_steps": [
            {
                "title": "Verify the software",
                "detail": "Run the full offline suite with coverage, Ruff lint, and formatting checks in this desk.",
                "status": "NEXT",
            },
            {
                "title": "Reserve fresh evidence",
                "detail": "Validate a contiguous period, record its checksums and chronological split, and exclude previously evaluated bars before tuning.",
                "status": "PENDING",
            },
            {
                "title": "Freeze and evaluate one candidate",
                "detail": "Freeze strategy, features, costs, and acceptance criteria; purge overlapping labels and evaluate once at base and doubled costs.",
                "status": "PENDING",
            },
            {
                "title": "Review the result",
                "detail": "Check sufficient trades, consistency across windows, calibration, and risk/P&L reconciliation before any paper-model promotion.",
                "status": "PENDING",
            },
        ],
        "warnings": warnings,
    }
