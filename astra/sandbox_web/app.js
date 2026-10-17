(() => {
  const $ = (selector) => document.querySelector(selector);
  const state = { token: "", datasets: [], strategies: [], selected: "", mode: "backtest", file: null, poller: null };
  const modes = [
    ["analyze", "Decision analysis", "One decision at the latest candle"],
    ["backtest", "Backtest", "One strategy with paper fills"],
    ["arena", "Strategy arena", "Compare selected strategies"],
    ["walk_forward", "Walk-forward", "Chronological train/test folds"],
    ["robustness", "Robustness", "Stress costs and parameters"],
    ["paper", "Paper replay", "Replay the paper portfolio"]
  ];
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (value) => typeof value === "number" ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value) : String(value ?? "—");
  async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body) { headers["Content-Type"] = "application/json"; headers["X-Astra-Token"] = state.token; }
    const response = await fetch(path, { ...options, headers, cache: "no-store" });
    const value = response.headers.get("content-type")?.includes("json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(value.error || "The local request could not be completed.");
    return value;
  }
  function notice(message, ok = false) { const node = $("#notice"); node.textContent = message; node.className = `notice${ok ? " ok" : ""}`; }
  function selectedStrategies() { return [...document.querySelectorAll(".strategy-option input:checked")].map((input) => input.value); }
  function renderModes() {
    $("#mode-grid").innerHTML = modes.map(([key, title, hint]) => `<button type="button" class="mode-card${state.mode === key ? " active" : ""}" data-mode="${key}"><strong>${title}</strong><small>${hint}</small></button>`).join("");
    document.querySelectorAll(".mode-card").forEach((button) => button.addEventListener("click", () => { state.mode = button.dataset.mode; renderModes(); }));
  }
  function renderStrategies() {
    $("#strategies").innerHTML = state.strategies.map((strategy, index) => `<label class="strategy-option"><input type="checkbox" value="${esc(strategy.name)}" ${index === 0 ? "checked" : ""}><span><strong>${esc(strategy.name.replaceAll("_", " "))}</strong><small>${esc(strategy.description || strategy.family || "Astra strategy")}</small></span></label>`).join("");
  }
  function renderDatasets() {
    $("#datasets").innerHTML = state.datasets.length ? state.datasets.map((item) => `<button type="button" class="dataset${state.selected === item.id ? " selected" : ""}" data-id="${esc(item.id)}"><span class="dataset-icon">▤</span><span class="dataset-info"><strong>${esc(item.name)}</strong><small>${esc(item.pair)} · ${esc(item.timeframe)} · ${fmt(item.count)} bars</small></span><span class="dataset-check">${state.selected === item.id ? "✓" : ""}</span></button>`).join("") : "";
    document.querySelectorAll(".dataset").forEach((button) => button.addEventListener("click", () => selectDataset(button.dataset.id)));
    $("#run-button").disabled = !state.selected;
  }
  async function selectDataset(id) {
    state.selected = id; renderDatasets();
    try {
      const value = await api(`/api/datasets/${encodeURIComponent(id)}`);
      const summary = value.summary;
      $("#preview").innerHTML = [["PAIR / FRAME", `${summary.pair} · ${summary.timeframe}`], ["AVAILABLE BARS", fmt(summary.count)], ["FIRST CLOSE TIME", summary.start], ["LAST CLOSE TIME", summary.end], ["REPORTED GAPS", fmt(summary.gaps)]]
        .map(([label, val]) => `<div>${label}<strong>${esc(val)}</strong></div>`).join("");
      $("#preview").classList.remove("hidden");
    } catch (error) { notice(error.message); }
  }
  async function loadHistory() {
    const value = await api("/api/state");
    state.token = value.token; state.datasets = value.datasets; state.strategies = value.strategies;
    $("#pair").innerHTML = value.pairs.map((pair) => `<option>${esc(pair)}</option>`).join("");
    $("#timeframe").innerHTML = value.timeframes.map((timeframe) => `<option>${esc(timeframe)}</option>`).join("");
    renderModes(); renderStrategies(); renderDatasets(); renderRuns(value.runs);
    if (state.datasets.length) await selectDataset(state.datasets[0].id);
  }
  async function importFile() {
    const file = state.file || $("#file-input").files[0];
    if (!file) { notice("Choose a candle CSV first."); return; }
    $("#import-button").disabled = true; notice("Reading and validating candle data…");
    try {
      const value = await api("/api/datasets", { method: "POST", body: JSON.stringify({ name: file.name, pair: $("#pair").value, timeframe: $("#timeframe").value, csv: await file.text() }) });
      state.file = null; $("#file-input").value = "";
      const latest = await api("/api/state"); state.datasets = latest.datasets;
      renderDatasets(); await selectDataset(value.id); notice("Dataset imported and validated on this machine.", true);
    } catch (error) { notice(error.message); }
    finally { $("#import-button").disabled = false; }
  }
  function formSettings() {
    const values = {};
    document.querySelectorAll(".settings-grid input").forEach((input) => { values[input.name] = input.type === "checkbox" ? input.checked : Number(input.value); });
    return values;
  }
  async function runExperiment() {
    const strategies = selectedStrategies();
    if (!state.selected || !strategies.length) { notice("Select a dataset and at least one strategy."); return; }
    const button = $("#run-button"); button.disabled = true; $("#job-status").textContent = "Queuing experiment…"; $("#job-status").className = "job-status running"; $("#job-status").classList.remove("hidden");
    try {
      const job = await api("/api/runs", { method: "POST", body: JSON.stringify({ dataset_id: state.selected, mode: state.mode, strategies, settings: formSettings() }) });
      pollJob(job.id);
    } catch (error) { $("#job-status").textContent = error.message; $("#job-status").className = "job-status"; button.disabled = false; }
  }
  function pollJob(id) {
    clearTimeout(state.poller);
    const check = async () => {
      try {
        const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
        $("#job-status").textContent = job.message;
        if (job.status === "complete") {
          $("#job-status").className = "job-status"; $("#run-button").disabled = !state.selected;
          const history = await api("/api/state"); renderRuns(history.runs); showResult(job.result); return;
        }
        if (job.status === "failed") { $("#job-status").className = "job-status"; $("#run-button").disabled = !state.selected; return; }
      } catch (error) { $("#job-status").textContent = error.message; $("#run-button").disabled = !state.selected; return; }
      state.poller = setTimeout(check, 650);
    };
    check();
  }
  function metricsFor(run) {
    const data = run.result || {};
    const candidates = data.metrics || data.decision?.context?.risk || data;
    if (Array.isArray(candidates)) return candidates.slice(0, 3).map((x) => [x.strategy || "Result", x.return_pct ?? x.score ?? "—"]);
    return Object.entries(candidates).filter(([, value]) => ["string", "number", "boolean"].includes(typeof value)).slice(0, 3);
  }
  function renderRuns(runs) {
    $("#run-count").textContent = runs.length ? `${runs.length} recent run${runs.length === 1 ? "" : "s"}` : "No runs yet";
    if (!runs.length) { $("#results").innerHTML = '<div class="empty-state"><span>◎</span><p>Your saved experiments will appear here.</p><small>Import candles to begin.</small></div>'; return; }
    $("#results").innerHTML = `<table class="run-table"><thead><tr><th>Experiment</th><th>Mode</th><th>Dataset</th><th>Saved</th></tr></thead><tbody>${runs.map((run) => `<tr class="run-row" data-id="${esc(run.id)}"><td><strong>${esc(run.headline)}</strong><small>${esc((run.strategies || []).join(", "))}</small></td><td><span class="tag">${esc(run.mode.replaceAll("_", " "))}</span></td><td>${esc(run.dataset.pair)} · ${esc(run.dataset.timeframe)}</td><td>${esc(new Date(run.created_at).toLocaleString())}</td></tr>`).join("")}</tbody></table>`;
    document.querySelectorAll(".run-row").forEach((row) => row.addEventListener("click", async () => { try { showResult(await api(`/api/runs/${encodeURIComponent(row.dataset.id)}`)); } catch (error) { notice(error.message); } }));
  }
  function showResult(run) {
    const metrics = metricsFor(run);
    $("#result-detail").innerHTML = `<h2 class="result-title">${esc(run.headline)}</h2><div class="result-subtitle">${esc(run.mode.replaceAll("_", " ").toUpperCase())} · ${esc(run.dataset.pair)} ${esc(run.dataset.timeframe)} · ${esc(new Date(run.created_at).toLocaleString())}</div><div class="result-metrics">${metrics.map(([key, value]) => `<div class="metric"><small>${esc(key.replaceAll("_", " "))}</small><strong>${esc(fmt(value))}</strong></div>`).join("")}</div><details><summary class="field-label">Full structured result</summary><pre class="json-view">${esc(JSON.stringify(run.result, null, 2))}</pre></details><p class="limitation">${esc(run.limitations)}</p>`;
    $("#result-dialog").showModal();
  }
  function setupDrop() {
    const zone = $("#drop-zone"), input = $("#file-input");
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); input.click(); } });
    input.addEventListener("change", () => { state.file = input.files[0] || null; if (state.file) notice(`${state.file.name} ready to import.`); });
    zone.addEventListener("dragover", (event) => { event.preventDefault(); zone.classList.add("drag"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
    zone.addEventListener("drop", (event) => { event.preventDefault(); zone.classList.remove("drag"); state.file = event.dataTransfer.files[0] || null; if (state.file) notice(`${state.file.name} ready to import.`); });
  }
  setupDrop();
  $("#import-button").addEventListener("click", importFile);
  $("#run-button").addEventListener("click", runExperiment);
  loadHistory().catch((error) => notice(error.message));
})();
