import json
import runpy
import sys
import threading
import time
from datetime import UTC, datetime
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from astra.sandbox import (
    MAX_BODY,
    Sandbox,
    SandboxError,
    SandboxHandler,
    SandboxServer,
    number,
)


@pytest.fixture
def sandbox_server(tmp_path):
    app = Sandbox(tmp_path / "sandbox")
    server = SimpleNamespace(server_port=18787, application=app)
    base = f"http://127.0.0.1:{server.server_port}"
    _IN_PROCESS_SERVERS[base] = server
    yield app, server, base
    _IN_PROCESS_SERVERS.pop(base, None)
    app.close()


@pytest.fixture
def socket_sandbox_server(tmp_path):
    app = Sandbox(tmp_path / "socket-sandbox")
    try:
        server = SandboxServer(0, app)
    except OSError as error:
        app.close()
        if getattr(error, "winerror", None) == 10013 or error.errno in (1, 13):
            pytest.skip("loopback sockets are blocked by this environment")
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield app, server, base
    finally:
        server.shutdown()
        server.server_close()
        app.close()
        thread.join(timeout=2)


class InProcessResponse:
    def __init__(self):
        self.status = 0
        self.headers = {}
        self.body = b""

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def request(base, path, method="GET", body=None, token=None, extra_headers=None):
    _, _, port = base.partition("://")
    host = port.split(":", 1)[-1]
    header_values = {
        "Host": f"127.0.0.1:{host}",
        "Origin": base,
        "Content-Length": "0",
    }
    if body is not None:
        payload = json.dumps(body).encode()
        header_values["Content-Type"] = "application/json"
        header_values["Content-Length"] = str(len(payload))
    else:
        payload = b""
    if token:
        header_values["X-Astra-Token"] = token
    header_values.update(extra_headers or {})
    headers = Message()
    for key, value in header_values.items():
        headers[key] = value

    response = InProcessResponse()
    handler = SandboxHandler.__new__(SandboxHandler)
    handler.server = _IN_PROCESS_SERVERS[base]
    handler.path = path
    handler.command = method
    handler.headers = headers
    handler.rfile = BytesIO(payload)
    handler.wfile = BytesIO()
    handler.send_response = lambda status, *_args: setattr(response, "status", status)
    handler.send_header = lambda key, value: response.headers.__setitem__(key, value)
    handler.end_headers = lambda: None
    handler.do_POST() if method == "POST" else handler.do_GET()
    response.body = handler.wfile.getvalue()
    return response


_IN_PROCESS_SERVERS = {}


def sample_csv(market_data):
    data = market_data(70, oscillate=True)
    rows = ["timestamp,open,high,low,close,volume"]
    rows.extend(
        f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
        for c in data.candles
    )
    return "\n".join(rows)


def test_sandbox_serves_local_dashboard_assets(sandbox_server):
    _, _, base = sandbox_server
    with request(base, "/") as response:
        html = response.read().decode()
        assert response.status == 200
        assert "Astra Research Desk" in html
        assert "paper research" in html.lower()
    for asset in ("/app.js", "/styles.css"):
        with request(base, asset) as response:
            assert response.status == 200
            assert response.read()


def test_sandbox_serves_curated_research_dashboard(sandbox_server):
    _, _, base = sandbox_server
    with request(base, "/research") as response:
        html = response.read().decode()
        assert response.status == 200
        assert "Training <span>&amp;</span> test results" in html
        assert "Overall test P&amp;L" in html
        assert "PROFIT / LOSS" in html
        assert "TEST RESULTS" in html
    for asset in ("/research.js", "/research.css"):
        with request(base, asset) as response:
            assert response.status == 200
            contents = response.read().decode()
            assert contents
            if asset.endswith(".js"):
                assert "function renderPnl()" in contents
                assert "setInterval(refresh, 5000)" in contents
                assert "showDetails(button.dataset.id)" in contents
            else:
                assert "--green:#b8ff35" in contents
                assert "--gold:#f5c653" in contents


def test_curated_research_api_summarizes_runs_and_returns_trade_detail(
    sandbox_server, tmp_path, monkeypatch
):
    app, _, base = sandbox_server
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    monkeypatch.setattr(app, "curated_research_dir", lambda: research_dir)
    progress = {"status": "RUNNING", "completed": 1, "total": 3}
    (research_dir / "research-progress.json").write_text(json.dumps(progress))
    report = {
        "status": "RUNNING",
        "profitability": "UNPROVEN",
        "items": [
            {
                "kind": "BACKTEST",
                "pair": "EURUSD",
                "segment": "2024-window",
                "timeframe": "H4",
                "strategy": "ema_trend",
                "candles": 100,
                "metrics": {"net_pnl": 2.5},
                "trades": [{"pnl": 2.5}],
            }
        ],
    }
    (research_dir / "research-results.json").write_text(json.dumps(report))
    from astra.adapters import SQLiteRepository
    from astra.repositories import Entity, Journal

    store = SQLiteRepository(str(research_dir / "research-journal.sqlite3"))
    journal = Journal(store)
    record = journal.record(
        Entity.BACKTEST,
        datetime.now(UTC),
        {
            "experiment_key": "GBPUSD:2025-window:H1:momentum",
            "source": "BACKTEST",
            "metrics": {"net_pnl": -3.0},
            "trade_count": 2,
        },
        pair="GBPUSD",
        timeframe="H1",
        strategy="momentum",
    )
    store.close()
    with request(base, "/api/curated-research/status") as response:
        assert json.loads(response.read())["completed"] == 1
    with request(base, "/api/curated-research/results") as response:
        summary = json.loads(response.read())
        assert summary["profitability"] == "UNPROVEN"
        assert summary["items"][0]["metrics"]["net_pnl"] == 2.5
        assert summary["items"][1]["journal_only"] is True
        assert summary["items"][1]["metrics"]["net_pnl"] == -3.0
    with request(
        base, "/api/curated-research/trades/j-" + record.record_id
    ) as response:
        assert json.loads(response.read())["journal_only"] is True
    with request(base, "/api/curated-research/trades/0") as response:
        assert json.loads(response.read())["trades"] == [{"pnl": 2.5}]


def test_curated_research_control_pauses_and_resumes_local_run(
    sandbox_server, tmp_path, monkeypatch
):
    app, _, base = sandbox_server
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    monkeypatch.setattr(app, "curated_research_dir", lambda: research_dir)
    (research_dir / "research-progress.json").write_text(
        json.dumps({"status": "RUNNING", "completed": 12, "total": 100})
    )
    with request(base, "/api/state") as response:
        token = json.loads(response.read())["token"]

    with request(
        base,
        "/api/curated-research/control",
        "POST",
        {"action": "pause"},
        token,
    ) as response:
        assert response.status == 202
        assert json.loads(response.read()) == {"action": "pause", "requested": True}
    control = research_dir / "research-control.json"
    assert json.loads(control.read_text()) == {"action": "pause"}
    assert app.curated_research_status()["status"] == "PAUSE_REQUESTED"

    with request(
        base,
        "/api/curated-research/control",
        "POST",
        {"action": "resume"},
        token,
    ) as response:
        assert response.status == 202
    assert not control.exists()
    assert app.curated_research_status()["status"] == "RUNNING"

    with request(
        base,
        "/api/curated-research/control",
        "POST",
        {"action": "stop"},
        token,
    ) as response:
        assert response.status == 400


def test_curated_research_status_reports_resumed_active_evaluation(
    sandbox_server, tmp_path, monkeypatch
):
    app, _, _ = sandbox_server
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "research-progress.json").write_text(
        json.dumps({"status": "PAUSED", "completed": 12, "total": 100})
    )
    (research_dir / "research-control.json").write_text('{"action":"pause"}')
    monkeypatch.setattr(app, "curated_research_dir", lambda: research_dir)
    monkeypatch.setattr(app, "_evaluation_is_active", lambda _path: True)

    assert app.curated_research_status()["status"] == "PAUSED"
    (research_dir / "research-control.json").unlink()
    assert app.curated_research_status()["status"] == "RUNNING"


@pytest.mark.socket
def test_sandbox_loopback_http_smoke(socket_sandbox_server):
    _, _, base = socket_sandbox_server
    try:
        with urlopen(base, timeout=2) as response:
            assert response.status == 200
            assert "Astra Research Desk" in response.read().decode()
    except URLError as error:
        reason = error.reason
        if isinstance(reason, OSError) and (
            getattr(reason, "winerror", None) == 10013 or reason.errno in (1, 13)
        ):
            pytest.skip("loopback sockets are blocked by this environment")
        raise
    except OSError as error:
        if getattr(error, "winerror", None) == 10013 or error.errno in (1, 13):
            pytest.skip("loopback sockets are blocked by this environment")
        raise


def test_sandbox_rejects_mutations_without_local_token(sandbox_server):
    _, _, base = sandbox_server
    response = request(base, "/api/datasets", "POST", {"csv": "bad"})
    assert response.status == 403


def test_sandbox_imports_runs_and_persists_paper_research(sandbox_server, market_data):
    app, _, base = sandbox_server
    with request(base, "/api/state") as response:
        state = json.loads(response.read())
    token = state["token"]
    with request(
        base,
        "/api/datasets",
        "POST",
        {
            "name": "synthetic-test.csv",
            "pair": "EURUSD",
            "timeframe": "M5",
            "csv": sample_csv(market_data),
        },
        token,
    ) as response:
        dataset = json.loads(response.read())
        assert response.status == 201
    with request(base, f"/api/datasets/{dataset['id']}") as response:
        preview = json.loads(response.read())
        assert preview["summary"]["count"] == 70
    with request(
        base,
        "/api/runs",
        "POST",
        {
            "dataset_id": dataset["id"],
            "mode": "backtest",
            "strategies": ["ema_trend"],
            "settings": {},
        },
        token,
    ) as response:
        job_id = json.loads(response.read())["id"]
        assert response.status == 202
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with request(base, f"/api/jobs/{job_id}") as response:
            job = json.loads(response.read())
        if job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "complete", job
    run_id = job["result"]["id"]
    with request(base, f"/api/runs/{run_id}") as response:
        saved = json.loads(response.read())
    assert saved["mode"] == "backtest"
    assert saved["dataset"]["name"] == "synthetic-test.csv"
    assert (app.run_dir / f"{run_id}.json").is_file()


def test_sandbox_validates_numeric_settings_and_modes(tmp_path):
    assert number({"risk": "0.5"}, "risk", 0, 0, 1) == 0.5
    with pytest.raises(SandboxError, match="must be a number"):
        number({"risk": "bad"}, "risk", 0, 0, 1)
    with pytest.raises(SandboxError, match="between"):
        number({"risk": float("inf")}, "risk", 0, 0, 1)
    app = Sandbox(tmp_path)
    try:
        with pytest.raises(SandboxError, match="supported experiment"):
            app.submit({"mode": "unknown", "dataset_id": "missing"})
        with pytest.raises(SandboxError, match="ten strategies"):
            app.options({"mode": "analyze", "strategies": ["unknown"]})
        with pytest.raises(SandboxError, match="settings must be an object"):
            app.options({"mode": "analyze", "settings": []})
        with pytest.raises(SandboxError, match="must be between"):
            app.options({"mode": "analyze", "settings": {"equity": 0}})
    finally:
        app.close()


def test_sandbox_upload_cleanup_restore_and_saved_run_validation(tmp_path, market_data):
    root = tmp_path / "workspace"
    app = Sandbox(root)
    with pytest.raises(SandboxError, match="Choose a CSV"):
        app.upload({"csv": ""})
    with pytest.raises(SandboxError, match="too large"):
        app.upload({"pair": "EURUSD", "timeframe": "M5", "csv": "x" * (MAX_BODY + 1)})
    with pytest.raises(ValueError):
        app.upload({"pair": "EURUSD", "timeframe": "M5", "csv": "not,csv"})
    assert not list(app.data_dir.glob("*.csv"))

    summary = app.upload(
        {"pair": "EURUSD", "timeframe": "M5", "csv": sample_csv(market_data)}
    )
    run_id = "a" * 32
    expected = {
        "id": run_id,
        "mode": "backtest",
        "created_at": "2025-01-01T00:00:00+00:00",
        "dataset": summary,
        "settings": {},
        "headline": "fixture",
    }
    (app.run_dir / f"{run_id}.json").write_text(json.dumps(expected), encoding="utf-8")
    (app.run_dir / "broken.json").write_text("{", encoding="utf-8")
    (app.run_dir / "missing-fields.json").write_text("{}", encoding="utf-8")
    (app.data_dir / "bad.json").write_text("{", encoding="utf-8")
    (app.data_dir / "missing-csv.json").write_text(
        json.dumps({"id": "b" * 24}), encoding="utf-8"
    )
    app.close()

    restored = Sandbox(root)
    try:
        assert summary["id"] in restored.datasets
        assert len(restored.history()) == 1
        assert restored.result(run_id)["headline"] == "fixture"
        for invalid in ("short", "g" * 32, "c" * 32):
            with pytest.raises(SandboxError, match="could not be found"):
                restored.result(invalid)
        with pytest.raises(SandboxError, match="imported dataset"):
            restored.data("missing")
        with pytest.raises(SandboxError, match="no longer active"):
            restored.job("missing")
    finally:
        restored.close()


def test_sandbox_upload_rejects_too_many_candles_and_cleans_file(
    tmp_path, market_data, monkeypatch
):
    import astra.sandbox as sandbox_module

    app = Sandbox(tmp_path)
    try:
        monkeypatch.setattr(sandbox_module, "MAX_CANDLES", 1)
        with pytest.raises(SandboxError, match="up to 1 candles"):
            app.upload(
                {"pair": "EURUSD", "timeframe": "M5", "csv": sample_csv(market_data)}
            )
        assert not list(app.data_dir.glob("*.csv"))
    finally:
        app.close()


def test_sandbox_submit_blocks_duplicate_jobs_and_run_failures(
    tmp_path, market_data, monkeypatch
):
    app = Sandbox(tmp_path)
    try:
        dataset = app.upload(
            {"pair": "EURUSD", "timeframe": "M5", "csv": sample_csv(market_data)}
        )
        body = {
            "dataset_id": dataset["id"],
            "mode": "backtest",
            "strategies": ["ema_trend"],
            "settings": {},
        }
        app.jobs["existing"] = {"id": "existing", "status": "running"}
        with pytest.raises(SandboxError, match="already running"):
            app.submit(body)
        ok_id = "a" * 32
        monkeypatch.setitem(app.jobs, ok_id, {"id": ok_id, "status": "queued"})
        monkeypatch.setattr(
            app, "execute", lambda body, identity: {"id": identity, "mode": "backtest"}
        )
        app._run_job(ok_id, body)
        assert app.job(ok_id)["status"] == "complete"
        assert app.result(ok_id)["id"] == ok_id

        bad_id = "b" * 32
        monkeypatch.setitem(app.jobs, bad_id, {"id": bad_id, "status": "queued"})
        monkeypatch.setattr(
            app, "execute", lambda *_: (_ for _ in ()).throw(SandboxError("bad input"))
        )
        app._run_job(bad_id, body)
        assert app.job(bad_id)["message"] == "bad input"

        crash_id = "c" * 32
        monkeypatch.setitem(app.jobs, crash_id, {"id": crash_id, "status": "queued"})
        monkeypatch.setattr(
            app,
            "execute",
            lambda *_: (_ for _ in ()).throw(RuntimeError("private path")),
        )
        app._run_job(crash_id, body)
        assert "private path" not in app.job(crash_id)["message"]
    finally:
        app.close()


def test_sandbox_http_rejects_wrong_host_origin_and_unknown_paths(sandbox_server):
    app, server, base = sandbox_server
    origin = f"http://127.0.0.1:{server.server_port}"
    wrong_origin = request(
        base, "/api/state", extra_headers={"Origin": "http://evil.invalid"}
    )
    assert wrong_origin.status == 403
    wrong_host = request(base, "/api/state", extra_headers={"Host": "localhost:1"})
    assert wrong_host.status == 403
    assert request(base, "/unknown").status == 404
    assert request(base, "/api/datasets/missing").status == 404
    job_id = "j" * 32
    app.jobs[job_id] = {"id": job_id, "status": "complete"}
    with request(base, f"/api/jobs/{job_id}") as response:
        assert json.loads(response.read())["status"] == "complete"
    run_id = "f" * 32
    saved = {
        "id": run_id,
        "mode": "backtest",
        "created_at": "now",
        "dataset": {},
        "settings": {},
        "headline": "fixture",
    }
    (app.run_dir / f"{run_id}.json").write_text(json.dumps(saved), encoding="utf-8")
    with request(base, f"/api/runs/{run_id}") as response:
        assert json.loads(response.read())["headline"] == "fixture"
    assert origin in base


def test_sandbox_execute_modes_use_existing_research_engines(
    tmp_path, market_data, monkeypatch
):
    from astra.models import Action, Decision
    from astra.robustness import RobustnessReport

    app = Sandbox(tmp_path)
    try:
        m5 = app.upload(
            {"pair": "EURUSD", "timeframe": "M5", "csv": sample_csv(market_data)}
        )
        m15_data = market_data(70, timeframe="M15", oscillate=True)
        m15_csv = "timestamp,open,high,low,close,volume\n" + "\n".join(
            f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},{c.close},{c.volume}"
            for c in m15_data.candles
        )
        m15 = app.upload({"pair": "EURUSD", "timeframe": "M15", "csv": m15_csv})

        with pytest.raises(SandboxError, match="supported experiment"):
            app.execute({"dataset_id": m5["id"], "mode": "invalid"}, "0" * 32)

        decision = Decision(
            datetime(2025, 1, 1, tzinfo=UTC), "EURUSD", "M5", Action.WAIT, "fixture"
        )
        monkeypatch.setattr("astra.decision.decide", lambda *a, **kw: decision)
        monkeypatch.setattr("astra.intelligence.detect_regime", lambda *a: "RANGING")
        monkeypatch.setattr("astra.market.structure", lambda *a: {"state": "RANGE"})
        monkeypatch.setattr("astra.market.momentum", lambda *a: {"score": 0})
        monkeypatch.setattr("astra.market.multi_timeframe", lambda *a, **kw: "aligned")
        analyzed = app.execute(
            {
                "dataset_id": m5["id"],
                "mode": "analyze",
                "timeframes": {"M15": m15["id"]},
            },
            "d" * 32,
        )
        assert analyzed["headline"] == "WAIT"
        assert analyzed["result"]["timeframes"] == "aligned"
        single = app.execute({"dataset_id": m5["id"], "mode": "analyze"}, "e" * 32)
        assert single["result"]["timeframes"] is None

        monkeypatch.setattr(
            "astra.arena.compete", lambda *a, **kw: ({"strategy": "ema_trend"},)
        )
        arena = app.execute({"dataset_id": m5["id"], "mode": "arena"}, "f" * 32)
        assert arena["headline"] == "1 strategies compared"

        monkeypatch.setattr(
            "astra.robustness.evaluate_walk_forward", lambda *a, **kw: ({"fold": 1},)
        )
        walked = app.execute({"dataset_id": m5["id"], "mode": "walk_forward"}, "1" * 32)
        assert walked["headline"] == "1 chronological folds"
        monkeypatch.setattr(
            "astra.robustness.evaluate_walk_forward", lambda *a, **kw: ()
        )
        with pytest.raises(SandboxError, match="shorter than"):
            app.execute({"dataset_id": m5["id"], "mode": "walk_forward"}, "2" * 32)

        monkeypatch.setattr(
            "astra.robustness.robustness_report",
            lambda *a, **kw: RobustnessReport("ROBUST", (), {}, (), {}, 0.0),
        )
        robust = app.execute({"dataset_id": m5["id"], "mode": "robustness"}, "3" * 32)
        assert robust["headline"] == "ROBUST"

        monkeypatch.setattr(
            "astra.pipeline.replay_paper",
            lambda *a, **kw: {"trades": [1, 2], "positions": [3]},
        )
        paper = app.execute({"dataset_id": m5["id"], "mode": "paper"}, "4" * 32)
        assert paper["headline"] == "2 closed · 1 open"
    finally:
        app.close()


def test_sandbox_http_post_validation_and_sanitized_errors(sandbox_server, monkeypatch):
    app, _, base = sandbox_server
    token = app.token

    def raw_post(path, data, content_type="application/json", length=None):
        headers = {"Content-Type": content_type, "X-Astra-Token": token}
        if length is not None:
            headers["Content-Length"] = str(length)
        if data is None:
            return request(base, path, "POST", token=token, extra_headers=headers)
        return _raw_post(base, path, data, headers)

    def _raw_post(base, path, data, headers):
        response = InProcessResponse()
        handler = SandboxHandler.__new__(SandboxHandler)
        handler.server = _IN_PROCESS_SERVERS[base]
        handler.path = path
        handler.command = "POST"
        header_values = {
            "Host": f"127.0.0.1:{handler.server.server_port}",
            "Origin": base,
        }
        header_values.update(headers)
        if "Content-Length" not in header_values:
            header_values["Content-Length"] = str(len(data))
        request_headers = Message()
        for key, value in header_values.items():
            request_headers[key] = value
        handler.headers = request_headers
        handler.rfile = BytesIO(data)
        handler.wfile = BytesIO()
        handler.send_response = lambda status, *_args: setattr(
            response, "status", status
        )
        handler.send_header = lambda key, value: response.headers.__setitem__(
            key, value
        )
        handler.end_headers = lambda: None
        handler.do_POST()
        response.body = handler.wfile.getvalue()
        return response

    assert raw_post("/api/datasets", None, length=0).status == 413
    assert raw_post("/api/datasets", None, length=MAX_BODY + 1).status == 413
    assert raw_post("/api/datasets", b"{}", content_type="text/plain").status == 415
    assert raw_post("/api/datasets", b"{").status == 400
    assert raw_post("/api/datasets", b"[]").status == 400
    assert raw_post("/api/unknown", b"{}").status == 404

    monkeypatch.setattr(
        app,
        "upload",
        lambda _body: (_ for _ in ()).throw(RuntimeError("private disk path")),
    )
    internal = raw_post("/api/datasets", b"{}")
    assert internal.status == 500
    assert "private disk path" not in internal.read().decode()


def test_sandbox_serve_and_module_entrypoint_cleanup(tmp_path, monkeypatch, capsys):
    import http.server

    import astra.sandbox as sandbox_module

    closed = []

    class FakeApplication:
        def __init__(self, workspace):
            self.workspace = workspace

        def close(self):
            closed.append("application")

    class FakeServer:
        server_port = 9123

        def __init__(self, port, app):
            self.app = app

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            closed.append("server")

    monkeypatch.setattr(sandbox_module, "Sandbox", FakeApplication)
    monkeypatch.setattr(sandbox_module, "SandboxServer", FakeServer)
    monkeypatch.setattr(
        sandbox_module.webbrowser, "open", lambda url: closed.append(url)
    )
    sandbox_module.serve(9123, tmp_path, open_browser=True)
    assert closed == ["http://127.0.0.1:9123", "server", "application"]
    assert "Astra sandbox stopped" in capsys.readouterr().out

    class FakeBaseServer:
        def __init__(self, address, handler):
            self.server_port = address[1]

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            closed.append("entrypoint-server")

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", FakeBaseServer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "astra-sandbox",
            "--port",
            "9124",
            "--workspace",
            str(tmp_path / "entry"),
            "--no-browser",
        ],
    )
    monkeypatch.delitem(sys.modules, "astra.sandbox")
    runpy.run_module("astra.sandbox", run_name="__main__")
    assert "entrypoint-server" in closed
