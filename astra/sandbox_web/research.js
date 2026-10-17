(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "—").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  let report = {items: []};
  const money = (v) => v == null ? "—" : Number(v).toFixed(2);
  const stamp = (v) => v ? new Date(v).toLocaleString() : "—";
  const pnlTypes = [
    ["BACKTEST", "Backtest"],
    ["OOS_HOLDOUT", "Out-of-sample holdout"],
    ["WALK_FORWARD", "Walk-forward folds"],
    ["COST_STRESS", "Cost stress"],
  ];
  async function api(path) {
    const response = await fetch(path, {cache:"no-store"});
    if (!response.ok) throw new Error("Research view could not load.");
    return response.json();
  }
  async function control(action) {
    const state = await api("/api/state");
    const response = await fetch("/api/curated-research/control", {
      method:"POST",
      headers:{"Content-Type":"application/json", "X-Astra-Token":state.token},
      body:JSON.stringify({action}),
      cache:"no-store"
    });
    if (!response.ok) throw new Error("Could not change the research run state.");
    return response.json();
  }
  function renderRows() {
    const filters = {pair:$("pair").value, timeframe:$("timeframe").value, kind:$("kind").value};
    const values = report.items.filter((item) => (!filters.pair || item.pair === filters.pair) && (!filters.timeframe || item.timeframe === filters.timeframe) && (!filters.kind || item.kind === filters.kind));
    $("summary").textContent = `${values.length.toLocaleString()} results shown`;
    $("rows").innerHTML = values.map((item) => {
      const metrics = item.metrics || {};
      const metricKind = ["BACKTEST","OOS_HOLDOUT","COST_STRESS"].includes(item.kind);
      const pnl = metricKind ? money(metrics.net_pnl) : item.kind === "WALK_FORWARD" ? `${item.folds?.length || 0} folds` : item.status;
      const factor = metricKind ? money(metrics.profit_factor) : item.kind === "WALK_FORWARD" ? (item.folds || []).reduce((s, f) => s + (f.trade_count || 0), 0) + " trades" : item.sample_count?.toLocaleString() || "—";
      const count = item.kind === "BACKTEST" ? `${item.candles ? `${item.candles.toLocaleString()} bars · ` : ""}${item.trade_count || 0} trades` : metricKind ? `${item.trade_count || 0} trades` : item.kind === "WALK_FORWARD" ? `${(item.folds || []).reduce((s, f) => s + (f.trade_count || 0), 0)} OOS trades` : `${item.sample_count?.toLocaleString() || 0} labels`;
      const label = item.strategy ? `${item.segment} · ${item.strategy}` : item.segment || item.reason || "";
      return `<tr><td><span class="pill">${esc(item.kind)}</span></td><td><b>${esc(item.pair)}</b><small>${esc(item.timeframe)}</small></td><td>${esc(label)}</td><td>${esc(count)}</td><td>${metricKind ? `<b class="${Number(metrics.net_pnl) >= 0 ? "positive" : "negative"}">${esc(pnl)}</b>` : esc(pnl)}</td><td>${esc(factor)}</td><td><button data-id="${item.id}" class="view">Details</button></td></tr>`;
    }).join("") || `<tr><td colspan="7" class="empty">No results match these filters yet.</td></tr>`;
    document.querySelectorAll(".view").forEach((button) => button.addEventListener("click", () => showDetails(button.dataset.id)));
  }
  function renderPnl() {
    const totals = Object.fromEntries(pnlTypes.map(([kind]) => [kind, {reports:0, samples:0, profit:0, loss:0, net:0}]));
    for (const item of report.items) {
      const total = totals[item.kind];
      if (!total) continue;
      total.reports += 1;
      const metrics = item.kind === "WALK_FORWARD" ? (item.folds || []).map((fold) => fold.metrics) : [item.metrics];
      for (const metric of metrics) {
        const value = metric?.net_pnl;
        if (value == null || !Number.isFinite(Number(value))) continue;
        const pnl = Number(value);
        total.samples += 1;
        total.net += pnl;
        if (pnl > 0) total.profit += pnl;
        if (pnl < 0) total.loss += pnl;
      }
    }
    const combined = Object.values(totals).reduce((sum, row) => ({
      reports:sum.reports + row.reports,
      samples:sum.samples + row.samples,
      profit:sum.profit + row.profit,
      loss:sum.loss + row.loss,
      net:sum.net + row.net,
    }), {reports:0,samples:0,profit:0,loss:0,net:0});
    $("pnl-profit").textContent = money(combined.profit);
    $("pnl-loss").textContent = money(combined.loss);
    $("pnl-net").textContent = money(combined.net);
    $("pnl-net").className = combined.net >= 0 ? "positive" : "negative";
    $("pnl-count").textContent = combined.reports.toLocaleString();
    $("pnl-count-detail").textContent = `${combined.samples.toLocaleString()} P&L measurements`;
    $("pnl-updated").textContent = `SYNCED ${new Date().toLocaleTimeString()}`;
    $("pnl-rows").innerHTML = pnlTypes.map(([kind, label]) => {
      const value = totals[kind];
      const signClass = value.net >= 0 ? "positive" : "negative";
      return `<tr><td><span class="pill">${esc(label)}</span></td><td>${value.reports.toLocaleString()}</td><td>${value.samples.toLocaleString()}</td><td class="positive">${esc(money(value.profit))}</td><td class="negative">${esc(money(value.loss))}</td><td class="${signClass}"><b>${esc(money(value.net))}</b></td></tr>`;
    }).join("") + `<tr class="total-row"><td>Combined report total</td><td>${combined.reports.toLocaleString()}</td><td>${combined.samples.toLocaleString()}</td><td class="positive">${esc(money(combined.profit))}</td><td class="negative">${esc(money(combined.loss))}</td><td class="${combined.net >= 0 ? "positive" : "negative"}"><b>${esc(money(combined.net))}</b></td></tr>`;
  }
  async function showDetails(id) {
    const item = await api(`/api/curated-research/trades/${id}`);
    const trades = item.trades || (item.folds || []).flatMap((fold) => (fold.trades || []).map((trade) => ({...trade, fold:fold.fold})));
    const metrics = item.metrics || {};
    const metricGrid = item.metrics ? `<div class="metrics">${[["Net P&L",metrics.net_pnl],["Return",metrics.total_return_pct == null ? null : `${money(metrics.total_return_pct)}%`],["Profit factor",metrics.profit_factor],["Win rate",metrics.win_rate == null ? null : `${money(metrics.win_rate*100)}%`],["Max drawdown",metrics.max_drawdown_pct == null ? null : `${money(metrics.max_drawdown_pct)}%`],["Trades",metrics.trade_count]].map(([k,v]) => `<div><small>${esc(k)}</small><b>${esc(v ?? "—")}</b></div>`).join("")}</div>` : "";
    const folds = (item.folds || []).map((fold) => `<tr><td>${fold.fold}</td><td>${esc(stamp(fold.test_start))} → ${esc(stamp(fold.test_end))}</td><td>${esc(money(fold.metrics?.net_pnl))}</td><td>${fold.trade_count || 0}</td></tr>`).join("");
    const tradeRows = trades.map((t) => { const pnl = t.pnl ?? (Number(t.pnl_per_unit) * Number(t.size)); return `<tr><td>${esc(t.fold ?? "Full")}</td><td>${esc(stamp(t.opened_at))}</td><td>${esc(stamp(t.closed_at))}</td><td>${esc(t.side)}</td><td>${esc(money(t.entry))}</td><td>${esc(money(t.exit))}</td><td class="${Number(pnl) >= 0 ? "positive" : "negative"}">${esc(money(pnl))}</td><td>${esc(t.reason)}</td></tr>`; }).join("");
    const tradeNotice = item.journal_only ? `<p>Metrics are already saved in the research Journal. Trade-level details will appear after this window is written to the results file.</p>` : "";
    $("detail-content").innerHTML = `<h2>${esc(item.kind)} · ${esc(item.pair)} ${esc(item.timeframe)}</h2><p>${esc(item.segment)} ${item.strategy ? `· ${esc(item.strategy)}` : ""}</p>${item.kind === "ML_TRAINING" ? `<p><b>${esc(item.status)}</b> · ${esc(item.sample_count)} labels</p><pre>${esc(JSON.stringify(item.model || item.reason, null, 2))}</pre>` : metricGrid}${folds ? `<h3>Chronological out-of-sample folds</h3><table><thead><tr><th>Fold</th><th>Test period</th><th>Net P&amp;L</th><th>Trades</th></tr></thead><tbody>${folds}</tbody></table>` : ""}${tradeNotice}<h3>Closed trades (${trades.length})</h3><div class="trade-table"><table><thead><tr><th>Fold</th><th>Opened</th><th>Closed</th><th>Side</th><th>Entry</th><th>Exit</th><th>Total P&amp;L</th><th>Exit reason</th></tr></thead><tbody>${tradeRows || `<tr><td colspan="8">${item.journal_only ? "Trade rows are waiting for the saved result batch." : "No closed trades."}</td></tr>`}</tbody></table></div>`;
    $("detail").showModal();
  }
  async function refresh() {
    try {
      const [status, next] = await Promise.all([api("/api/curated-research/status"), api("/api/curated-research/results")]);
      report = next;
      renderPnl();
      const pct = status.total ? Math.round(100 * status.completed / status.total) : 0;
      $("progress-status").innerHTML = `<div class="progress-head"><b>${esc(status.status)}</b><span>${status.completed || 0} / ${status.total || 0} execution steps${status.total ? ` · ${pct}%` : ""} · ${report.items.length.toLocaleString()} unique saved results</span></div><progress max="${status.total || 1}" value="${status.completed || 0}"></progress><small>${esc([status.stage,status.pair,status.segment,status.timeframe,status.strategy].filter(Boolean).join(" · ") || "Start the research job from the project terminal to begin.")} · Repeated work can increase execution steps without adding a new result.</small>`;
      const button = $("run-control");
      button.hidden = !["RUNNING","PAUSE_REQUESTED","PAUSED"].includes(status.status);
      button.disabled = false;
      button.dataset.action = status.status === "PAUSED" || status.status === "PAUSE_REQUESTED" ? "resume" : "pause";
      button.textContent = status.status === "PAUSED" ? "▶ Play / Resume" : status.status === "PAUSE_REQUESTED" ? "▶ Cancel pause" : "Ⅱ Pause";
      for (const [id, key] of [["pair","pair"],["timeframe","timeframe"]]) {
        const select = $(id), old = select.value, vals = [...new Set(report.items.map((i) => i[key]).filter(Boolean))].sort();
        select.innerHTML = `<option value="">All ${key === "pair" ? "pairs" : "timeframes"}</option>` + vals.map((v) => `<option>${esc(v)}</option>`).join(""); select.value = old;
      }
      $("assumptions").innerHTML = Object.entries(report.assumptions || {}).map(([k,v]) => `<span><b>${esc(k.replaceAll("_"," "))}:</b> ${esc(Array.isArray(v) ? v.join(", ") : typeof v === "object" ? JSON.stringify(v) : v)}</span>`).join("");
      $("limits").innerHTML = (report.limitations || []).map((line) => `<li>${esc(line)}</li>`).join("");
      renderRows();
    } catch (error) { $("progress-status").textContent = error.message; }
  }
  $("run-control").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const action = button.dataset.action;
    button.disabled = true;
    try {
      await control(action);
      button.textContent = action === "pause" ? "Pause requested…" : "Resume requested…";
      await refresh();
    } catch (error) {
      $("progress-status").textContent = error.message;
      button.disabled = false;
    }
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
    document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.id !== tab.dataset.panel; });
  }));
  ["pair","timeframe","kind"].forEach((id) => $(id).addEventListener("change", renderRows));
  refresh(); setInterval(refresh, 5000);
})();
