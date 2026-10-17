"""Synthetic saved-report fixtures; no market evaluation is performed here."""

import json

import pytest

from astra.project_status import project_assessment


def save(root, path, value):
    target = root / "data" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding="utf-8")


REPORT = "training-arena/curated-history-h1-h4-v1/evaluation-report.json"


def fixture_report():
    return {
        "campaign_id": "synthetic-test",
        "status": "COMPLETED",
        "test_count": 8,
        "summary": {
            "a|EURUSD|H4|base": {
                "window_runs": 1,
                "mean_return_pct": 4,
                "total_closed_trades": 5,
            },
            "a|USDJPY|H4|base": {
                "window_runs": 3,
                "mean_return_pct": -2,
                "total_closed_trades": 15,
            },
            "a|EURUSD|H4|double_cost": {
                "window_runs": 1,
                "mean_return_pct": 2,
                "total_closed_trades": 5,
            },
            "a|USDJPY|H4|double_cost": {
                "window_runs": 3,
                "mean_return_pct": -3,
                "total_closed_trades": 15,
            },
        },
    }


def test_missing_evidence_never_claims_completed_or_profitable(tmp_path):
    state = project_assessment(tmp_path)
    assert state["stage"] == "EVIDENCE_COLLECTION"
    assert state["metrics"]["curated_runs"] is None
    assert state["promotion"] == "NOT_ASSESSED"
    assert state["profitability"] == "UNPROVEN"


def test_assessment_weights_returns_by_runs_without_summing_quote_currency(tmp_path):
    save(tmp_path, REPORT, fixture_report())
    state = project_assessment(tmp_path)
    assert state["stage"] == "RESEARCH_VALIDATION"
    assert state["metrics"]["base_mean_return_pct"] == pytest.approx(-0.5)
    assert state["metrics"]["double_cost_mean_return_pct"] == pytest.approx(-1.75)
    assert state["metrics"]["trade_records"] == 40
    assert state["promotion"] == "NOT_QUALIFIED"
    assert "fresh" in state["next_action"]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"status": "COMPLETED"},
        {"status": "COMPLETED", "test_count": True},
        {"summary": None},
    ],
)
def test_malformed_report_cannot_complete_gate(tmp_path, value):
    save(tmp_path, REPORT, value)
    assert project_assessment(tmp_path)["stage"] == "EVIDENCE_COLLECTION"


def test_inconsistent_and_nonfinite_summary_cannot_complete_gate(tmp_path):
    report = fixture_report()
    report["test_count"] = 9
    save(tmp_path, REPORT, report)
    assert project_assessment(tmp_path)["stage"] == "EVIDENCE_COLLECTION"
    report["test_count"] = 8
    report["summary"]["a|USDJPY|H4|base"]["mean_return_pct"] = float("nan")
    save(tmp_path, REPORT, report)
    assert project_assessment(tmp_path)["stage"] == "EVIDENCE_COLLECTION"


def test_positive_saved_results_do_not_automatically_promote_model(tmp_path):
    report = fixture_report()
    for group in report["summary"].values():
        group["mean_return_pct"] = 3
    save(tmp_path, REPORT, report)
    assert project_assessment(tmp_path)["promotion"] == "NOT_ASSESSED"


def test_stale_diagnosis_and_overcomplete_progress_are_explicit(tmp_path):
    save(tmp_path, REPORT, fixture_report())
    save(
        tmp_path,
        "astra-training-four-pairs/research-progress.json",
        {"completed": 12, "total": 10},
    )
    save(
        tmp_path,
        "training-arena/curated-history-h1-h4-v1/loss-diagnostics.json",
        {"campaign_id": "other", "result_count": 8},
    )
    state = project_assessment(tmp_path)
    assert len(state["warnings"]) == 2
    assert "12 completed against 10 planned" in state["warnings"][1]
