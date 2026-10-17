/* Local Evaluation Desk. Saved artifacts describe recorded evidence, not process liveness. */
"use strict";
const REFRESH_MS = 3000;
const REQUEST_TIMEOUT_MS = 12000;
const $ = (id) => document.getElementById(id);
let state = null;
let token = "";
let refreshPending = false;
let controlPending = false;
let connected = false;
let lastReceived = null;
let noticeTimer = null;
const VIEWS = {
  overview: ["Evaluation overview", "One place to follow the tests, the evidence, and what comes next."],
  tests: ["Software verification", "Run the complete local checks and follow every stage as it finishes."],
  research: ["Historical research", "Understand the recorded experiments and the limits of their evidence."],
  training: ["Training & validation", "Track candidate experiments, curated validation, and cost sensitivity."],
  data: ["Data quality", "See what was acquired, what failed, and which source checks were recorded."],
  paper: ["Paper replay", "Review execution outcomes from the saved historical rehearsal."],
};
function object(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
function list(value) { return Array.isArray(value) ? value : []; }
function numeric(value) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function integer(value) { const n = numeric(value); return n === null ? "—" : n.toLocaleString("en-US", { maximumFractionDigits: 0 }); }
function decimal(value, digits = 2) { const n = numeric(value); return n === null ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }); }
function percent(value, digits = 2) { return numeric(value) === null ? "—" : `${decimal(value, digits)}%`; }
function signedPercent(value) { return numeric(value) === null ? "—" : `${value > 0 ? "+" : ""}${percent(value)}`; }
function returnClass(value) { return numeric(value) === null || value === 0 ? "" : value > 0 ? "positive" : "negative"; }
function escape(value) { return String(value ?? "—").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]); }
function humanize(value) { return value ? String(value).replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase()) : "Not recorded"; }
function parsedDate(value) {
  if (!value) return null;
  const compact = String(value).match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
  const stamp = compact ? `${compact[1]}-${compact[2]}-${compact[3]}T${compact[4]}:${compact[5]}:${compact[6]}Z` : value;
  const date = new Date(stamp);
  return Number.isNaN(date.getTime()) ? null : date;
}
function stamp(value, short = false) {
  const date = parsedDate(value);
  if (!date) return "Time not recorded";
  return date.toLocaleString("en-GB", short ? { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false } : { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}
function statusLabel(value) {
  return ({ NOT_STARTED: "Not started", MISSING: "No evidence", NOT_ASSESSED: "Not assessed", NOT_QUALIFIED: "Not qualified", NOT_ESTABLISHED: "Not established", RECORDED_ACTIVE: "Saved activity", STALE: "Stale record", UNKNOWN: "Unknown", NEEDS_REVIEW: "Needs review", INCOMPLETE: "Incomplete", INTERRUPTED: "Interrupted", RUNNING: "Running", PASSED: "Passed", FAILED: "Failed", COMPLETED: "Completed", RECORDED: "Recorded", CLEAN: "Clean", ISSUES: "Issues found", IN_PROGRESS: "Recorded in progress" })[value] || humanize(value);
}
function statusColor(value) {
  if (["PASSED", "PASS", "CLEAN"].includes(value)) return "green";
  if (["FAILED", "ERROR", "ISSUES"].includes(value)) return "red";
  if (["NOT_QUALIFIED", "STALE", "INTERRUPTED", "NEEDS_REVIEW", "INCOMPLETE", "RECORDED_ACTIVE", "IN_PROGRESS"].includes(value)) return "amber";
  return value === "RUNNING" ? "blue" : "neutral";
}
function badge(value, label) { return `<span class="badge ${statusColor(value)}">${escape(label || statusLabel(value))}</span>`; }
function setBadge(id, value) { $(id).className = `badge ${statusColor(value)}`; $(id).textContent = statusLabel(value); }
function metric(label, value, detail = "", className = "") { return `<article class="metric"><span class="metric-label">${escape(label)}</span><b class="metric-value ${className}">${escape(value)}</b><span class="metric-detail">${escape(detail)}</span></article>`; }
function miniMetric(label, value) { return `<div class="mini-metric"><span>${escape(label)}</span><b>${escape(value)}</b></div>`; }
function emptyRow(columns, message) { return `<tr><td colspan="${columns}" class="empty-state">${escape(message)}</td></tr>`; }
function recordTime(record) { return object(record.freshness).updated_at || record.updated_at || record.completed_at || record.manifest_updated_at; }
function savedTime(record) { const when = recordTime(record); return when ? `Saved ${stamp(when, true)}` : "Timestamp not recorded"; }
function freshDescription(record) {
  const freshness = object(record.freshness);
  if (freshness.state === "MISSING" || record.status === "NOT_STARTED") return "No saved evidence was found for this activity.";
  const when = recordTime(record);
  const prefix = when ? `Last recorded update: ${stamp(when)}.` : "The artifact does not record an update time.";
  if (record.status === "STALE") return `${prefix} The previous activity record is stale; current process activity is unknown.`;
  if (record.status === "RECORDED_ACTIVE") return `${prefix} The artifact recently reported activity; a running process has not been verified.`;
  return `${prefix} This is a saved evidence snapshot.`;
}
function progressMarkup(record, name) {
  const total = numeric(record.total), completed = numeric(record.completed);
  if (total === null || total <= 0 || completed === null) return '<div class="progress-empty">No planned run total is recorded yet.</div>';
  const fraction = Math.max(0, Math.min(1, numeric(record.progress_fraction) ?? completed / total));
  const warning = record.count_inconsistent || completed > total;
  const current = object(record.current);
  const context = [current.stage && humanize(current.stage), current.pair, current.timeframe, current.strategy && humanize(current.strategy)].filter(Boolean).join(" · ");
  return `<div class="progress-meta"><span>${integer(completed)} recorded / ${integer(total)} planned</span><b>${percent(fraction * 100, 0)}</b></div><progress max="1" value="${fraction}" aria-label="${escape(name)} recorded progress">${percent(fraction * 100, 0)}</progress>${context ? `<p class="caption">Last recorded activity: ${escape(context)}</p>` : ""}${warning ? '<p class="caption caution">Recorded and planned counts disagree. Progress is capped at 100%; the source counts are shown above.</p>' : ""}`;
}
function renderList(id, values, fallback) {
  const rows = list(values).filter((item) => typeof item === "string" && item.trim());
  $(id).innerHTML = rows.length ? rows.map((item) => `<li>${escape(item)}</li>`).join("") : `<li>${escape(fallback)}</li>`;
}
function switchView() {
  const aliases = { acquisition: "data", sources: "data" };
  const requested = location.hash.slice(1), candidate = aliases[requested] || requested;
  const view = Object.hasOwn(VIEWS, candidate) ? candidate : "overview";
  document.querySelectorAll(".nav-item").forEach((item) => {
    const active = item.dataset.view === view;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "page"); else item.removeAttribute("aria-current");
  });
  document.querySelectorAll(".view").forEach((section) => { const active = section.id === `view-${view}`; section.hidden = !active; section.classList.toggle("active", active); });
  $("breadcrumb").textContent = view === "data" ? "DATA QUALITY" : view.toUpperCase();
  $("pageTitle").textContent = VIEWS[view][0];
  $("pageDescription").textContent = VIEWS[view][1];
  document.title = `${VIEWS[view][0]} · Astra`;
  if (view === "tests" && $("followLog").checked) $("testLog").scrollTop = $("testLog").scrollHeight;
}
function runStatus(run) { return run.running ? "RUNNING" : run.status || (run.exit_code === 0 ? "PASSED" : run.exit_code != null ? "FAILED" : "NOT_STARTED"); }
function card({ label, status, value, unit, detail, time, view }) {
  return `<article class="evidence-card"><div class="card-head"><span class="card-label">${escape(label)}</span>${badge(status)}</div><h3>${escape(value)}${unit ? `<small>${escape(unit)}</small>` : ""}</h3><p>${escape(detail)}</p><div class="card-footer"><span>${escape(time)}</span><a href="#${view}" aria-label="View ${escape(label.toLowerCase())}">Details</a></div></article>`;
}
function renderOverview(next) {
  const assessment = object(next.assessment), run = object(next.test_run), research = object(next.research);
  const training = object(next.training), curated = object(next.curated_history), acquisition = object(next.acquisition), replay = object(next.rehearsal);
  const trainingSummary = object(training.summary), curatedSummary = object(curated.summary);
  const testStatus = runStatus(run), counts = object(run.counts);
  $("stageTitle").textContent = assessment.headline || "Build a trustworthy research baseline";
  $("stageSummary").textContent = assessment.summary || "The local evidence assessment is not available. Review the recorded checks below before choosing the next experiment.";
  $("stageLabel").textContent = assessment.stage_label || "Evidence collection";
  $("nextAction").textContent = assessment.next_action || "Verify the software and reconcile the available historical results.";
  $("profitabilityTag").textContent = `Profitability ${String(assessment.profitability || "UNPROVEN").toLowerCase()}`;
  const verified = testStatus === "PASSED";
  const cards = [
    { label: "Software verification", status: testStatus, value: numeric(counts.passed) !== null ? integer(counts.passed) : run.running ? "Checking…" : verified ? "Passed" : testStatus === "FAILED" ? "Needs attention" : "Ready to run", unit: numeric(counts.passed) !== null ? "tests passed" : "", detail: run.running ? `${run.phase || "Checks"} in progress. Output is available in the console.` : verified ? `Coverage ${percent(run.coverage_pct, 1)} · pytest, lint & formatting recorded.` : testStatus === "FAILED" ? "A verification stage failed. Open the console to review its output." : testStatus === "INTERRUPTED" ? "The last run ended without a verified completion. Run the checks again." : "Run pytest with coverage, Ruff lint, and formatting checks.", time: run.finished_at ? `Finished ${stamp(run.finished_at, true)}` : run.started_at ? `Started ${stamp(run.started_at, true)}` : "No run recorded", view: "tests" },
    { label: "Historical research", status: research.status || "MISSING", value: integer(research.experiments), unit: "saved experiments", detail: research.count_inconsistent ? "Saved experiment and planned counts need reconciliation." : "Historical strategy evaluations and cost-stress evidence.", time: savedTime(research), view: "research" },
    { label: "Training Arena", status: training.status || "MISSING", value: integer(training.completed), unit: "recorded tests", detail: `${integer(trainingSummary.trade_count)} trade records · ${integer(trainingSummary.model_count)} saved models.`, time: savedTime(training), view: "training" },
    { label: "Curated validation", status: curated.status || "MISSING", value: integer(curated.completed), unit: "recorded runs", detail: `${integer(curatedSummary.trade_count)} trade records across independent windows and cost scenarios.`, time: savedTime(curated), view: "training" },
    { label: "Data acquisition", status: acquisition.status || acquisition.summary_status || "MISSING", value: integer(acquisition.units), unit: "manifest records", detail: "Downloads still require continuity and quality validation before research use.", time: savedTime(acquisition), view: "data" },
    { label: "Historical paper replay", status: replay.available ? "RECORDED" : "MISSING", value: replay.available ? integer(replay.trades) : "No replay", unit: replay.available ? "closed trades" : "", detail: replay.available ? "Finite replay on recorded prices. Forward observation is a separate step." : "A saved paper execution rehearsal will appear here.", time: replay.available ? (recordTime(replay) ? savedTime(replay) : "Saved historical artifact") : "No replay recorded", view: "paper" },
  ];
  const cardMarkup = cards.map(card).join("");
  // Preserve keyboard focus on a card link during unchanged polling updates.
  if ($("overviewCards").innerHTML !== cardMarkup) $("overviewCards").innerHTML = cardMarkup;
  $("overviewCards").setAttribute("aria-busy", "false");
  const steps = list(assessment.next_steps);
  $("nextSteps").innerHTML = steps.length ? steps.map((step) => `<li><b>${escape(step.title)}</b><p>${escape(step.detail)}</p></li>`).join("") : '<li><b>Verify the software</b><p>Run software checks, then review the available data and validation results.</p></li>';
  const gates = list(assessment.evidence);
  $("evidenceGates").innerHTML = gates.length ? gates.map((gate) => `<div class="gate"><div class="gate-title"><span>${escape(gate.label)}</span>${badge(gate.status)}</div><p>${escape(gate.detail)}</p>${gate.recorded_at ? `<p>${escape(stamp(gate.recorded_at))}</p>` : ""}</div>`).join("") : '<p class="empty-state">The evidence assessment is unavailable.</p>';
  const warnings = list(assessment.warnings).filter((value) => typeof value === "string");
  $("assessmentWarnings").textContent = warnings.join(" ");
  $("assessmentWarnings").hidden = !warnings.length;
}
function renderTests(run, history) {
  const counts = object(run.counts), status = runStatus(run);
  $("testNavDot").hidden = !run.running;
  $("testMetrics").innerHTML = [
    metric("Verification result", statusLabel(status), run.running ? run.phase || "Preparing checks" : "All three stages must pass", status === "PASSED" ? "positive" : status === "FAILED" ? "negative" : ""),
    metric("Tests passed", integer(counts.passed), numeric(counts.tests) !== null ? `${integer(counts.tests)} total · ${integer(counts.skipped)} skipped` : "Reported when pytest finishes"),
    metric("Failures & errors", numeric(counts.failed) !== null && numeric(counts.errors) !== null ? integer(counts.failed + counts.errors) : "—", "From the recorded pytest result", counts.failed > 0 || counts.errors > 0 ? "negative" : ""),
    metric("Code coverage", percent(run.coverage_pct, 1), "Measured by pytest-cov"),
  ].join("");
  setBadge("testRunBadge", status);
  $("testRunHeading").textContent = run.running ? run.phase || "Starting verification" : status === "PASSED" ? "Software verification passed" : status === "FAILED" ? "Review the failed stage" : status === "INTERRUPTED" ? "The last run was interrupted" : "Ready when you are";
  const phases = list(run.phases);
  const displayed = phases.length ? phases : ["Tests and coverage", "Ruff lint", "Ruff format"].map((name) => ({ name, status: "PENDING" }));
  $("testPhases").innerHTML = displayed.map((phase, index) => `<div class="phase ${statusColor(phase.status)}"><span class="phase-icon" aria-hidden="true">${phase.status === "PASSED" ? "✓" : phase.status === "FAILED" ? "×" : index + 1}</span><div><b>${escape(phase.name)}</b><small>${escape(statusLabel(phase.status))}${phase.exit_code != null ? ` · exit ${escape(phase.exit_code)}` : ""}</small></div></div>`).join("");
  $("testTiming").textContent = [run.started_at ? `Started ${stamp(run.started_at)}` : "Run software checks to execute pytest with coverage, Ruff lint, and Ruff format checks.", run.finished_at ? `Finished ${stamp(run.finished_at)}` : "", run.storage_error ? "Verification results could not be saved. Check the console before retrying." : ""].filter(Boolean).join(" · ");
  const output = typeof run.log === "string" && run.log ? run.log : "No output recorded. Select “Run software checks” to begin.";
  if ($("testLog").textContent !== output) {
    $("testLog").textContent = output;
    if ($("followLog").checked) $("testLog").scrollTop = $("testLog").scrollHeight;
  }
  $("testLogMeta").textContent = run.running ? "Receiving current run output" : run.log ? `Saved run output · ${integer(run.log.split("\n").length)} visible lines` : "Waiting for a run";
  $("testHistoryBody").innerHTML = history.length ? history.slice(0, 10).map((item) => {
    const itemCounts = object(item.counts);
    return `<tr><td>${escape(stamp(item.started_at))}</td><td>${badge(item.status || (item.exit_code === 0 ? "PASSED" : item.exit_code != null ? "FAILED" : "UNKNOWN"))}</td><td>${integer(itemCounts.passed)} passed${numeric(itemCounts.failed) !== null ? `<small>${integer(itemCounts.failed)} failed · ${integer(itemCounts.skipped)} skipped</small>` : ""}</td><td>${percent(item.coverage_pct, 1)}</td><td>${item.finished_at ? escape(stamp(item.finished_at, true)) : "—"}</td></tr>`;
  }).join("") : emptyRow(5, "No saved verification runs yet. Your next run will appear here.");
}
function renderResearch(research) {
  $("researchMetrics").innerHTML = [metric("Saved experiments", integer(research.experiments), "Records in the results artifact"), metric("Completed counter", integer(research.completed), `${integer(research.total)} originally planned`), metric("Recorded errors", integer(research.errors), "Errors reported by the saved campaign", research.errors > 0 ? "negative" : ""), metric("Profitability", humanize(research.profitability || "UNPROVEN"), "A separate research qualification", "caution")].join("");
  setBadge("researchBadge", research.status || "MISSING");
  $("researchProgress").innerHTML = progressMarkup(research, "Historical research");
  $("researchFreshness").textContent = freshDescription(research);
  $("researchNote").textContent = research.status_note || "Saved experiments provide historical evidence. A current worker is not tracked by this record.";
  const kinds = Object.entries(object(research.experiment_kinds));
  $("researchKinds").innerHTML = kinds.length ? kinds.map(([name, count]) => `<span class="chip">${escape(humanize(name))}<b>${integer(count)}</b></span>`).join("") : '<span class="chip">No experiment categories recorded</span>';
  renderList("researchLimitations", research.limitations, "No campaign-specific limitations were recorded. Review the source evaluation before relying on a result.");
}
function renderTraining(training, curated) {
  for (const [prefix, record, label] of [["training", training, "Training Arena"], ["curated", curated, "Curated validation"]]) {
    const summary = object(record.summary);
    setBadge(`${prefix}Badge`, record.status || "MISSING");
    $(`${prefix}Progress`).innerHTML = progressMarkup(record, label);
    $(`${prefix}Metrics`).innerHTML = [miniMetric("Saved results", integer(record.saved_results)), miniMetric("Trade records", integer(summary.trade_count)), miniMetric("Saved models", integer(summary.model_count))].join("");
    $(`${prefix}Freshness`).textContent = freshDescription(record);
    $(`${prefix}Note`).textContent = record.status_note || "No saved campaign status is available.";
  }
  const scenarios = object(object(curated.summary).scenarios);
  $("scenarioCards").innerHTML = [["base", "Base costs"], ["double_cost", "Doubled costs"]].map(([key, label]) => {
    const record = object(scenarios[key]);
    return `<div class="scenario-card"><div class="scenario-head"><span>${label}</span><span>${integer(record.tests)} tests</span></div><div class="scenario-value ${returnClass(record.mean_return_pct)}">${signedPercent(record.mean_return_pct)}</div><div class="scenario-subtitle">Mean net return per independent test</div><div class="scenario-facts"><span><b>${integer(record.positive_tests)}</b> positive</span><span><b>${integer(record.negative_tests)}</b> negative</span><span><b>${integer(record.trade_count)}</b> trade records</span></div></div>`;
  }).join("");
  $("scenarioUnits").textContent = object(curated.summary).units_note || "These are independent test windows. The mean return is a comparison across tests, not a combined portfolio return; cash P&L across different quote currencies is not added.";
  renderCuratedTable(curated);
  const limitations = [...list(training.limitations), ...list(curated.limitations)];
  renderList("trainingLimitations", [...new Set(limitations)], "No additional campaign limitations were recorded. An untouched chronological holdout and cost checks are needed before promotion.");
}
function renderCuratedTable(curated) {
  const filter = $("scenarioFilter").value;
  const groups = list(object(curated.summary).groups).filter((item) => filter === "all" || item.scenario === filter);
  $("curatedBody").innerHTML = groups.length ? groups.map((item) => `<tr><td>${escape(humanize(item.strategy))}</td><td>${escape(item.pair)}</td><td>${escape(item.timeframe)}</td><td>${item.scenario === "double_cost" ? "Doubled" : item.scenario === "base" ? "Base" : escape(humanize(item.scenario))}</td><td>${integer(item.positive_windows)} / ${integer(item.window_runs)}</td><td class="${returnClass(item.median_return_pct)}">${signedPercent(item.median_return_pct)}</td><td>${integer(item.total_closed_trades)}</td></tr>`).join("") : emptyRow(7, "No grouped validation results are available for these costs.");
}
function acquisitionTotals(acquisition) {
  const totals = { downloaded: 0, failed: 0, other: 0 };
  for (const bucket of Object.values(object(acquisition.by_symbol))) {
    for (const [key, value] of Object.entries(object(bucket))) {
      if (numeric(value) !== null) totals[key === "downloaded" ? "downloaded" : key === "failed" ? "failed" : "other"] += value;
    }
  }
  return totals;
}
function renderData(acquisition, qleap) {
  const totals = acquisitionTotals(acquisition), rows = Object.entries(object(acquisition.by_symbol));
  $("acquisitionMetrics").innerHTML = [metric("Manifest records", integer(acquisition.units), `${integer(rows.length)} currency pairs represented`), metric("Downloaded units", rows.length ? integer(totals.downloaded) : "—", "Saved downloads; continuity still needs validation"), metric("Failed units", rows.length ? integer(totals.failed) : "—", "Recorded acquisition failures", totals.failed > 0 ? "negative" : "")].join("");
  setBadge("acquisitionBadge", acquisition.status || acquisition.summary_status || "MISSING");
  $("acquisitionFreshness").textContent = freshDescription(acquisition);
  $("acquisitionNote").textContent = acquisition.status_note || "A manifest records acquisition outcomes. It cannot confirm that an acquisition process is currently running.";
  $("acquisitionBody").innerHTML = rows.length ? rows.map(([symbol, value]) => {
    const counts = object(value), other = Object.entries(counts).filter(([key]) => !["downloaded", "failed"].includes(key)).reduce((sum, [, count]) => sum + (numeric(count) ?? 0), 0);
    return `<tr><td>${escape(symbol)}</td><td>${integer(counts.downloaded ?? 0)}</td><td class="${counts.failed > 0 ? "negative" : ""}">${integer(counts.failed ?? 0)}</td><td>${integer(other)}</td><td>${escape(object(acquisition.latest_days)[symbol] || "Not recorded")}</td></tr>`;
  }).join("") : emptyRow(5, "No acquisition records found. Validated new data will need a saved manifest.");
  $("qleapAudited").textContent = qleap.available ? stamp(qleap.audited_at, true) : "No audit recorded";
  const pairs = Object.entries(object(qleap.pairs));
  $("qleapBody").innerHTML = qleap.available && pairs.length ? pairs.map(([symbol, item]) => {
    const range = list(item.range).map((value) => value ? String(value).slice(0, 10) : "Unknown").join(" → ");
    let structuralStatus = item.structural_status, issues = numeric(item.structural_issues);
    if (!structuralStatus) {
      const values = [item.non_monotonic_files, item.duplicate_timestamp_files, item.bad_prices, item.bad_ohlc, item.negative_volume];
      issues = values.every((value) => numeric(value) !== null) ? values.reduce((sum, value) => sum + value, 0) : null;
      structuralStatus = issues === null ? "UNKNOWN" : issues > 0 ? "ISSUES" : "CLEAN";
    }
    return `<tr><td>${escape(symbol)}</td><td>${integer(item.rows)}</td><td>${escape(range || "Not recorded")}</td><td>${badge(structuralStatus, structuralStatus === "ISSUES" ? `${integer(issues)} issues` : null)}</td><td>${integer(item.closed_session_bars)}</td></tr>`;
  }).join("") : emptyRow(5, "No source audit found. Source approval requires its own validation record.");
  $("qleapVerdict").textContent = qleap.available ? `Converted-source status: ${humanize(qleap.converted_status)}. This source is separate from the curated dataset; structural checks alone do not approve it for evaluation.` : "";
}
function renderReplay(replay) {
  const available = Boolean(replay.available), pnl = available ? numeric(replay.net_pnl) : null;
  $("rehearsalMetrics").innerHTML = [metric("Closed trades", available ? integer(replay.trades) : "—", "Historical replay record"), metric("Winning trades", available ? integer(replay.wins) : "—", "Recorded winning outcomes"), metric("Net P&L", pnl === null ? "—" : `${pnl > 0 ? "+" : ""}${decimal(pnl)}`, "Total in the replay account’s units", returnClass(pnl)), metric("Final equity", available ? decimal(replay.final_equity) : "—", "Replay account units")].join("");
  $("rehearsalDataset").textContent = available ? replay.dataset || "Recorded replay" : "No replay recorded";
  const details = available ? [["Decisions", integer(replay.decisions)], ["Starting equity", decimal(replay.starting_equity)], ["Filled orders", integer(object(replay.orders).FILLED)], ["Rejected orders", integer(object(replay.orders).REJECTED)], ["Trade decisions", integer(replay.trade_decisions)], ["Strategies used", list(replay.strategies_used).map(humanize).join(", ") || "Not recorded"]] : [];
  $("rehearsalDetails").innerHTML = details.map(([label, value]) => `<div class="detail"><span>${escape(label)}</span><b>${escape(value)}</b></div>`).join("");
  $("rehearsalUpdated").textContent = available ? recordTime(replay) ? savedTime(replay) : "The saved replay summary does not include a completion timestamp." : "No saved paper replay evidence is available.";
  $("rehearsalCaveat").textContent = [replay.caveat || "Historical replay outcomes depend on the recorded data and execution assumptions.", "Forward paper observation requires a separate, continuously updated record."].join(" ");
}
function updateControls() {
  const running = Boolean(object(state?.test_run).running);
  document.querySelectorAll("[data-run-tests]").forEach((button) => {
    button.disabled = !connected || !token || running || controlPending;
    button.textContent = controlPending ? "Starting checks…" : running ? "Checks running…" : "▷  Run software checks";
  });
}
function render(next) {
  state = next;
  token = typeof next.control_token === "string" ? next.control_token : "";
  renderOverview(next);
  renderTests(object(next.test_run), list(next.test_history));
  renderResearch(object(next.research));
  renderTraining(object(next.training), object(next.curated_history));
  renderData(object(next.acquisition), object(next.qleap));
  renderReplay(object(next.rehearsal));
  updateControls();
}
function notice(message, error = false) {
  clearTimeout(noticeTimer);
  $("actionNotice").textContent = message;
  $("actionNotice").className = `action-notice${error ? " error" : ""}`;
  $("actionNotice").hidden = false;
  if (!error) noticeTimer = setTimeout(() => { $("actionNotice").hidden = true; }, 9000);
}
async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { cache: "no-store", ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error("The local desk returned an unreadable response."); }
    if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : `The local desk returned HTTP ${response.status}.`);
    if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("The local desk returned an invalid state.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The local desk did not respond within 12 seconds.");
    if (error instanceof TypeError) throw new Error("The local desk could not be reached.");
    throw error;
  } finally { clearTimeout(timeout); }
}
async function refresh() {
  if (refreshPending) return;
  refreshPending = true;
  $("refreshBtn").disabled = true;
  try {
    const next = await request("/api/state", { headers: { Accept: "application/json" } });
    connected = true;
    render(next);
    lastReceived = new Date();
    const generated = parsedDate(next.generated_at);
    const delayed = generated && Date.now() - generated.getTime() > 30000;
    $("connectionState").textContent = delayed ? "Delayed state" : "Desk connected";
    $("connectionPill").className = `connection-pill ${delayed ? "offline" : "connected"}`;
    $("connectionError").hidden = !delayed;
    if (delayed) $("connectionError").textContent = `The desk responded with an older state generated ${stamp(next.generated_at)}. Displayed activity may be out of date.`;
    $("syncStamp").textContent = `Synced ${stamp(next.generated_at || lastReceived.toISOString(), true)} · refreshes every 3s`;
  } catch (error) {
    connected = false;
    $("connectionState").textContent = "Connection lost";
    $("connectionPill").className = "connection-pill offline";
    $("connectionError").textContent = `${error.message} ${lastReceived ? `Showing the last snapshot received ${stamp(lastReceived.toISOString())}; activity may have changed.` : "Start the local Evaluation Desk and refresh this page."} Retrying automatically.`;
    $("connectionError").hidden = false;
    $("syncStamp").textContent = lastReceived ? `Last received ${stamp(lastReceived.toISOString(), true)} · reconnecting` : "No evidence received yet · reconnecting";
  } finally {
    refreshPending = false;
    $("refreshBtn").disabled = false;
    updateControls();
  }
}
async function runTests() {
  if (!token || !connected || controlPending || object(state?.test_run).running) return;
  controlPending = true;
  updateControls();
  try {
    const result = await request("/api/control", { method: "POST", headers: { "Content-Type": "application/json", "X-Astra-Token": token }, body: JSON.stringify({ action: "run-tests" }) });
    notice(result.started ? "Software verification started. Follow pytest, coverage, and Ruff in the console." : "A verification run is already active. Its progress appears below.");
    location.hash = "tests";
    switchView();
  } catch (error) { notice(`Could not confirm that verification started. ${error.message} Refresh the state before trying again.`, true); }
  finally { controlPending = false; await refresh(); updateControls(); }
}
window.addEventListener("hashchange", () => { switchView(); $("main").focus({ preventScroll: true }); window.scrollTo({ top: 0, behavior: "instant" }); });
$("refreshBtn").addEventListener("click", refresh);
document.querySelectorAll("[data-run-tests]").forEach((button) => button.addEventListener("click", runTests));
$("scenarioFilter").addEventListener("change", () => renderCuratedTable(object(state?.curated_history)));
$("followLog").addEventListener("change", () => { if ($("followLog").checked) $("testLog").scrollTop = $("testLog").scrollHeight; });
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
switchView();
refresh();
setInterval(refresh, REFRESH_MS);
