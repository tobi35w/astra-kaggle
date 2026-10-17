import json

from astra import training_arena


def test_arena_state_exposes_curated_history_progress(tmp_path, monkeypatch):
    arena_root = tmp_path / "training-arena"
    history_root = arena_root / "curated-history-h1-h4-v1"
    history_root.mkdir(parents=True)
    (history_root / "progress.json").write_text(
        json.dumps(
            {
                "campaign_id": "history-test",
                "status": "RUNNING",
                "completed_tests": 7,
                "total_tests": 22,
                "current_pair": "EURUSD",
                "current_timeframe": "H1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_arena, "ARENA_ROOT", arena_root)
    monkeypatch.setattr(training_arena, "CURATED_HISTORY_ROOT", history_root)
    monkeypatch.setattr(training_arena, "RESEARCH_ROOT", tmp_path / "research")
    monkeypatch.setattr(training_arena, "DATASET_ROOT", tmp_path / "dataset")

    state = training_arena.arena_state()

    assert state["curated_history"]["status"] == "RUNNING"
    assert state["curated_history"]["completed_tests"] == 7
    assert state["curated_history"]["current_pair"] == "EURUSD"
