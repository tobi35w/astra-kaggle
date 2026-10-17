import json

import pytest

import scripts.launch_research_sandbox as launcher


def test_launcher_resumes_unfinished_report_in_detached_process(tmp_path, monkeypatch):
    results = tmp_path / "research-results.json"
    results.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    monkeypatch.setattr(launcher, "RESULTS", results)
    monkeypatch.setattr(launcher, "DATA_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    message = launcher.start_or_resume_evaluation()

    assert "started or resumed" in message
    assert calls[0][0][-1] == "--resume"
    assert calls[0][1]["stdin"] is launcher.subprocess.DEVNULL
    assert calls[0][1]["stdout"].name.endswith("research-run.stdout.log")


def test_launcher_does_not_restart_completed_evaluation(tmp_path, monkeypatch):
    results = tmp_path / "research-results.json"
    results.write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    monkeypatch.setattr(launcher, "RESULTS", results)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_a, **_k: pytest.fail())

    assert "already complete" in launcher.start_or_resume_evaluation()
