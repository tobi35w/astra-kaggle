"""Launcher behavior uses injected local responses and never starts real jobs."""

from scripts import launch_progress_desk as launcher


def test_reopen_reuses_server_without_starting_workers(monkeypatch):
    opened = []
    monkeypatch.setattr(launcher, "desk_available", lambda: True)
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        launcher,
        "start_server",
        lambda: (_ for _ in ()).throw(AssertionError("duplicate server")),
    )
    assert launcher.main() == 0
    assert opened == [launcher.URL]


def test_launcher_waits_for_verified_server_before_opening(monkeypatch):
    availability = iter([False, False, True])
    actions = []
    monkeypatch.setattr(launcher, "desk_available", lambda: next(availability))
    monkeypatch.setattr(launcher, "start_server", lambda: actions.append("start"))
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(launcher.webbrowser, "open", actions.append)
    assert launcher.main() == 0
    assert actions == ["start", launcher.URL]


def test_launcher_rejects_unrelated_service(monkeypatch):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args):
            pass

        def getresponse(self):
            return self

        status = 200

        def getheader(self, *args):
            return "UnrelatedService/1"

        def close(self):
            pass

    monkeypatch.setattr(launcher.http.client, "HTTPConnection", Connection)
    assert launcher.desk_available() is False
