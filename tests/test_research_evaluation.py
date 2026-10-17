from pathlib import Path

import pytest

import scripts.evaluate_curated_history as evaluation


def test_snapshot_save_retries_a_transient_windows_file_lock(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    original_replace = Path.replace
    calls = 0

    def replace_with_one_lock(path, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("simulated transient file lock")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", replace_with_one_lock)
    monkeypatch.setattr(evaluation.time, "sleep", lambda _: None)

    evaluation.save(target, {"status": "RUNNING", "completed": 42})

    assert calls == 2
    assert '"completed": 42' in target.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("progress.json.*.tmp"))


@pytest.mark.parametrize("status", ["FAILED", "RUNNING"])
def test_resume_validates_scope_and_keeps_checkpointed_results(status):
    previous = {
        "status": status,
        "scope": {"pairs": ["EURUSD"], "timeframes": ["M5"]},
        "assumptions": {"spread_pips": 1.0},
        "items": [{"kind": "BACKTEST"}, {"kind": "OOS_HOLDOUT"}],
        "errors": [
            {"type": "PermissionError", "detail": "temporary file lock"},
            {"pair": "EURUSD", "reason": "DataValidationError"},
        ],
    }

    resumed, completed = evaluation.prepare_resume(
        previous,
        {"completed": 1},
        {"pairs": ("EURUSD",), "timeframes": ("M5",)},
        {"spread_pips": 1.0},
    )

    assert resumed["status"] == "RUNNING"
    assert completed == 2
    assert len(resumed["items"]) == 2
    assert resumed["errors"] == [{"pair": "EURUSD", "reason": "DataValidationError"}]
    assert (
        resumed["resume_history"][0]["previous_failure"][0]["type"] == "PermissionError"
    )
    with pytest.raises(ValueError, match="scope or assumptions"):
        evaluation.prepare_resume(
            {
                **previous,
                "status": "FAILED",
                "scope": {"pairs": ["GBPUSD"]},
            },
            {"completed": 1},
            {"pairs": ("EURUSD",)},
            {"spread_pips": 1.0},
        )
