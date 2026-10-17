const $ = (id) => document.getElementById(id);
let snapshot = null;
let view = "overview";
let toastTimer = null;
let tradeDetails = new Map();
let nextTradeDetailId = 0;

const money = (n) => `${Number(n || 0) < 0 ? "−" : ""}$${Math.abs(Number(n || 0)).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
const number = (n, d=0) => Number(n || 0).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const pct = (n, d=1) => `${(Number(n || 0) * 100).toFixed(d)}%`;
const esc = (value) => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const shortTime = (value) => value ? new Date(value).toLocaleString(undefined,{month:"short",day:"2-digit",hour:"2-digit",minute:"2-digit",timeZone:"UTC"})+" UTC" : "—";
const pnlClass = (n) => Number(n || 0) >= 0 ? "result-positive" : "result-negative";
const signedMoney = (n) => `${Number(n || 0) > 0 ? "+" : ""}${money(n)}`;
const tradePnl = (trade) => trade?.pnl !== undefined && trade?.pnl !== null && Number.isFinite(Number(trade.pnl))
  ? Number(trade.pnl)
  : Number(trade?.pnl_per_unit || 0) * Number(trade?.size || 0);
const detailButton = (trade) => {
  const id = `trade-${++nextTradeDetailId}`;
  tradeDetails.set(id, trade);
  return `<button class="detail-button" type="button" data-trade-detail="${id}">Details</button>`;
};

function showTradeDetails(trade) {
  if (!trade) return;
  const pnl = tradePnl(trade), risk = Number(trade.risk_per_unit || 0) * Number(trade.size || 0);
  const fields = [
    ["Pair / timeframe", `${trade.pair || "—"} · ${trade.timeframe || "—"}`],
    ["Strategy", trade.strategy], ["Side", trade.side], ["Position size", `${number(trade.size, 0)} units`],
    ["Entry price", number(trade.entry, 5)], ["Exit price", number(trade.exit, 5)],
    ["Stop loss", number(trade.stop_loss, 5)], ["Take profit", number(trade.take_profit, 5)],
    ["Total P&L", signedMoney(pnl)], ["P&L per unit", number(trade.pnl_per_unit, 8)],
    ["Risk amount", money(risk)], ["Result in R", `${number(risk ? pnl / risk : 0, 2)}R`],
    ["Opened (UTC)", shortTime(trade.opened_at)], ["Closed (UTC)", shortTime(trade.closed_at)],
    ["Exit reason", trade.reason], ["Market regime", trade.regime], ["Signal confidence", pct(trade.confidence || 0)],
    ["Evidence account", trade.test_id]
  ];
  $("tradeModalBody").innerHTML = fields.map(([label, value]) => `<div class="trade-detail"><small>${esc(label)}</small><b>${esc(value ?? "—")}</b></div>`).join("");
  $("tradeModal").hidden = false;
  $("tradeModalClose").focus();
}

function setView(next) {
  view = next;
  document.querySelectorAll(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === next));
  document.querySelectorAll(".view").forEach(section => section.classList.toggle("active", section.id === `view-${next}`));
  const titles = {overview:"Training Arena",accounts:"Test accounts",trades:"Trade log",models:"Model lab",evidence:"Data & evidence"};
  $("pageTitle").textContent = titles[next] || "Training Arena";
  if (next === "trades") renderTrades();
}

function showToast(message) {
  const element = $("toast"); element.textContent = message; element.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => element.classList.remove("show"), 2300);
}

function drawEquity(tests) {
  const canvas = $("equityChart"), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  if (!rect.width) return;
  canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio,ratio);
  const w=rect.width,h=rect.height, values=[0]; let total=0;
  for (const test of tests.slice().reverse()) { total += Number(test.net_pnl||0); values.push(total); }
  if(values.length<2){ctx.strokeStyle="#4dffb5";ctx.globalAlpha=.5;ctx.beginPath();ctx.moveTo(0,h*.72);ctx.lineTo(w,h*.72);ctx.stroke();return;}
  const lo=Math.min(0,...values),hi=Math.max(0,...values),span=hi-lo||1;
  const pts=values.map((v,i)=>[i/(values.length-1)*w,h-((v-lo)/span)*(h*.78)-h*.1]);
  const grad=ctx.createLinearGradient(0,0,w,0);grad.addColorStop(0,"#4dffb5");grad.addColorStop(1,"#ffc95d");
  ctx.beginPath();pts.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle=grad;ctx.lineWidth=2;ctx.shadowColor="#4dffb5";ctx.shadowBlur=8;ctx.stroke();
}

function updateOverview(data) {
  const a=data.arena||{}, c=data.campaign||{}, s=c.summary||{}, tests=c.tests||[], active=a.active_test||{};
  const history=data.curated_history||{};
  const historyPanel=$("curatedHistoryPanel");
  if(historyPanel && history.status && history.status!=="NOT_STARTED") {
    historyPanel.hidden=false;
    const done=Number(history.completed_tests||0), total=Number(history.total_tests||0);
    const percent=total?Math.min(100,done/total*100):0;
    $("curatedHistoryStatus").textContent=String(history.status).replaceAll("_"," ");
    $("curatedHistoryProgress").textContent=`${number(done)} / ${number(total)} evaluations`;
    $("curatedHistoryPercent").textContent=`${percent.toFixed(0)}%`;
    $("curatedHistoryProgressBar").style.width=`${percent}%`;
    const identity=[history.current_pair,history.current_timeframe,history.current_window].filter(Boolean).join(" · ");
    $("curatedHistoryStage").textContent=history.status==="COMPLETED"?`Completed historical holdout. Report: ${history.report?.campaign_id||history.campaign_id||"available locally"}.`:history.status==="FAILED"?"Evaluation stopped with an error; completed checkpoints are preserved.":identity?`Current window: ${identity}`:"Preparing curated historical windows.";
    const latest=history.current_experiment?`${history.current_experiment} · latest ${number(history.latest_trade_count)} trades · ${number(history.latest_net_return_pct,3)}%`:"Progress refreshes automatically every two seconds.";
    $("curatedHistoryLatest").textContent=latest;
  }
  const status=a.status||c.status||"WAITING_FOR_RESEARCH";
  const sync=data.storage_sync||{};
  $("connectionState").textContent=sync.status==="CONNECTED"?"SUPABASE SYNCED":sync.status==="RETRYING"?"LOCAL · SYNC QUEUED":"LOCAL STORAGE";
  $("liveBadge").textContent=status==="RUNNING"?"REPLAYING":status==="PAUSED"?"PAUSED":status==="COMPLETED"?"CAMPAIGN DONE":status==="FAILED"?"NEEDS ATTENTION":"STANDBY";
  $("phaseHeadline").textContent=status==="WAITING_FOR_RESEARCH"?"NEXT PHASE ARMED":status==="RUNNING"?"TRAINING & PAPER REPLAY ACTIVE":status==="PAUSED"?"REPLAY PAUSED":status==="COMPLETED"?"CAMPAIGN COMPLETE":status==="FAILED"?"CAMPAIGN STOPPED SAFELY":"TRAINING ARENA READY";
  const research=data.research||{};
  $("phaseSubline").textContent=status==="WAITING_FOR_RESEARCH"?`Research Sandbox: ${research.completed||0} / ${research.total||0} steps · this arena starts when it is complete`:status==="RUNNING"?`${a.stage||"Running offline paper research"} · local data, no broker`:status==="PAUSED"?"Paused at a safe checkpoint; resume when ready.":status==="COMPLETED"?"All model and out-of-sample paper simulation runs are saved locally.":a.last_error?`Stopped safely (${a.last_error}).`:"The local FX dataset and campaign runner are prepared.";
  const ready=status==="RUNNING"||status==="PAUSED";
  $("pauseBtn").disabled=!ready;
  $("pauseBtn").textContent=status==="PAUSED"?"▶  RESUME":"Ⅱ  PAUSE";
  $("phaseStepDot").className=(status==="RUNNING"||status==="PAUSED"||status==="COMPLETED")?"done":"";
  const count=Number(a.completed_tests??tests.length), total=Number(a.total_tests||68), percent=total?Math.min(100,count/total*100):0;
  $("testProgressText").textContent=`${count} / ${total} test accounts`;
  $("testPercent").textContent=`${percent.toFixed(0)}%`;
  $("testProgressBar").style.width=`${percent}%`;
  const activePair=active.pair||a.current_pair, activeTimeframe=active.timeframe||a.current_timeframe;
  $("activePair").textContent=activePair?`${activePair} · ${activeTimeframe||""}`:"—";
  $("activeTestName").textContent=active.strategy?`${active.strategy.replaceAll("_"," ")} · ${active.id||""}`:a.current_strategy||"Waiting on Research completion";
  $("activeStage").textContent=a.stage||"The arena will automatically unlock when the current evaluation is complete.";
  $("barProgress").textContent=active.total_bars?`${number(active.bar)} / ${number(active.total_bars)}`:"—";
  $("activeBalance").textContent=active.equity?money(active.equity):"—";
  $("activeTrades").textContent=active.closed_trades==null?"—":number(active.closed_trades);
  const positions=active.open_positions||[];
  $("openPositions").textContent=number(positions.length);
  $("positionStrip").innerHTML=positions.length?positions.map(p=>`<span class="position-chip">${esc(p.pair)} ${esc(p.side)} · ${number(p.size,0)} units · entry ${number(p.entry,5)} · SL ${number(p.stop_loss,5)} · TP ${number(p.take_profit,5)} · ${signedMoney(p.unrealized_pnl)}</span>`).join(""):"<span class=\"muted\">No simulated positions open</span>";
  const starting=Number(s.aggregate_starting_balance||0), net=Number(s.aggregate_net_pnl||0);
  $("combinedEquity").textContent=money(starting+net); $("netPnl").textContent=signedMoney(net); $("netPnl").className=`metric-number ${pnlClass(net)}`;
  $("aggregateBalance").textContent=`${money(starting)} test capital`;
  $("paperTrades").textContent=number(s.trade_count); $("winLoss").textContent=`${number(s.wins)} wins · ${number(s.losses)} losses`;
  const models=c.models||[]; const trained=models.filter(m=>m.status==="TRAINED");
  $("modelsReady").innerHTML=`${trained.length}<span class="unit"> / ${number(a.total_models||20)}</span>`;
  $("modelStatus").textContent=trained.length?`Mean validation accuracy ${pct(trained.reduce((sum,m)=>sum+(m.validation?.validation_accuracy||0),0)/trained.length)}`:"Chronological, purged split";
  $("tradeCountNav").textContent=number(s.trade_count);
  const rank=Math.floor(count/5)+1, rankNames=["Signal Scout","Pattern Runner","Risk Navigator","Model Tactician","Arena Architect"];
  $("levelLabel").textContent=`LEVEL ${String(rank).padStart(2,"0")}`; $("rankName").textContent=rankNames[Math.min(rank-1,rankNames.length-1)];
  $("xpText").textContent=`${count%5} / 5 completed test accounts to next rank`; $("xpBar").style.width=`${count%5*20}%`;
  $("priorEvaluation").textContent=research.status==="COMPLETED"?`${number(research.prior_phase_result_count)} source-separated results linked`:research.status==="RUNNING"?`${number(Math.min(research.completed||0,research.total||0))} / ${number(research.total)} steps running`:"Waiting for Research Sandbox";
  $("datasetSummary").textContent=`${number(data.dataset?.rows)} bars · ${data.dataset?.structural_pass_files||0}/20 structure checks`;
  $("datasetState").textContent=data.dataset?.offline_ready?"READY":"CHECK";
  $("modelSequenceState").textContent=trained.length?`${trained.length}/20 TRAINED`:"QUEUED";
  $("simSequenceState").textContent=tests.length?`${tests.length}/${total} TESTED`:"QUEUED";
  const liveClosed=active.last_closed_trade?{...active.last_closed_trade,test_id:active.id,pair:activePair,timeframe:activeTimeframe,strategy:active.last_closed_trade.strategy}:null;
  const seenTrades = new Map();
  if (liveClosed) {
    const key = `${liveClosed.test_id}|${liveClosed.closed_at}|${liveClosed.entry}|${liveClosed.side}`;
    seenTrades.set(key, liveClosed);
  }
  // Preserve the API's evaluation order (newest completed test first). Sorting
  // by candle date made the feed look frozen when a new test covered older data.
  for (const trade of c.recent_trades || []) {
    const key = `${trade.test_id}|${trade.closed_at}|${trade.entry}|${trade.side}`;
    if (!seenTrades.has(key)) seenTrades.set(key, trade);
  }
  for (const test of tests) for (const trade of test.trades || []) {
    const enriched = { ...trade, test_id: trade.test_id || test.id, pair: trade.pair || test.pair, timeframe: trade.timeframe || test.timeframe, strategy: trade.strategy || test.strategy };
    const key = `${enriched.test_id}|${enriched.closed_at}|${enriched.entry}|${enriched.side}`;
    if (!seenTrades.has(key)) seenTrades.set(key, enriched);
  }
  const allClosedTrades = [...seenTrades.values()];
  const recent = allClosedTrades;
  const wins = allClosedTrades.filter(t => tradePnl(t) > 0).length;
  const losses = allClosedTrades.filter(t => tradePnl(t) < 0).length;
  tradeDetails = new Map(); nextTradeDetailId = 0;
  snapshot.campaign.recent_trades=recent;
  snapshot.campaign._all_trades=allClosedTrades;
  $("paperTrades").textContent=number(s.trade_count ?? allClosedTrades.length);
  $("winLoss").textContent=`${number(wins)} wins · ${number(losses)} losses`;
  renderRecent(recent); renderAccounts(tests); renderTrades(); renderModels(models);
  renderEvidence(data); drawEquity(tests);
  $("lastUpdate").textContent=`UPDATED ${shortTime(a.updated_at||c.updated_at)}`;
}

function renderRecent(trades) {
  const body=$("recentTrades");
  if(!trades.length){body.innerHTML='<tr><td colspan="8" class="empty-state">The first OOS trade will appear here when the arena starts.</td></tr>';return;}
  body.innerHTML=trades.slice(0,8).map(t=>{const pnl=tradePnl(t);return `<tr><td><b>${esc(t.pair)}</b> · ${esc(t.timeframe)}</td><td>${esc(t.strategy)}</td><td><span class="side-pill ${String(t.side).toLowerCase()}">${esc(t.side)}</span></td><td>${number(t.entry,5)}</td><td>${number(t.exit,5)}</td><td class="${pnlClass(pnl)}">${signedMoney(pnl)}</td><td>${shortTime(t.closed_at)}</td><td>${detailButton(t)}</td></tr>`}).join("");
}

function renderAccounts(tests) {
  $("accountsCount").textContent=`${tests.length} / 68 ACCOUNTS`;
  const grid=$("accountGrid");
  if(!tests.length){grid.innerHTML='<div class="empty-card">Accounts populate as each out-of-sample simulation completes. Every card represents an independent $10,000 paper ledger.</div>';return;}
  grid.innerHTML=tests.slice().reverse().map(t=>`<article class="account-card" data-test="${esc(t.id)}"><div class="account-card-top"><span class="pair-chip">${esc(t.pair)} · ${esc(t.timeframe)}</span><span class="tag">${t.kind==="ML_PAPER"?"MODEL OOS":"BASELINE"}</span></div><h3>${esc(t.strategy.replaceAll("_"," "))}</h3><small>Independent simulated account · ${esc(t.evidence)}</small><div class="account-balance">${money(t.ending_equity)}</div><div class="account-foot"><span class="${pnlClass(t.net_pnl)}">${signedMoney(t.net_pnl)}</span><span>${number(t.metrics.trade_count)} trades · ${pct(t.metrics.win_rate)}</span></div><div class="account-foot" style="margin-top:8px"><span>Max DD ${number(t.metrics.max_drawdown_pct,2)}%</span><span>PF ${t.metrics.profit_factor==null?"—":number(t.metrics.profit_factor,2)}</span></div><div class="mini-progress"><i style="width:${Math.min(100,Math.max(1,(Number(t.metrics.trade_count)||0)*2))}%"></i></div></article>`).join("");
}

function allTrades() { return snapshot?.campaign?._all_trades || snapshot?.campaign?.recent_trades || []; }
function renderTrades() {
  if(!snapshot)return;
  const pair=$("pairFilter").value, timeframe=$("timeframeFilter").value;
  const trades=allTrades().filter(t=>(!pair||t.pair===pair)&&(!timeframe||t.timeframe===timeframe));
  $("tradeRowsCount").textContent=`Showing ${number(Math.min(trades.length,200))} newest of ${number(trades.length)} trades`;
  const body=$("allTrades");
  if(!trades.length){body.innerHTML='<tr><td colspan="14" class="empty-state">No trades match these filters yet.</td></tr>';return;}
  body.innerHTML=trades.slice(0,200).map(t=>{const pnl=tradePnl(t),risk=Number(t.risk_per_unit||0)*Number(t.size||0);return `<tr><td>${esc(t.pair)}</td><td>${esc(t.timeframe)}</td><td>${esc(t.strategy)}</td><td><span class="side-pill ${String(t.side).toLowerCase()}">${esc(t.side)}</span></td><td>${number(t.size,0)}</td><td>${number(t.entry,5)}</td><td>${number(t.stop_loss,5)}</td><td>${number(t.take_profit,5)}</td><td>${number(t.exit,5)}</td><td class="${pnlClass(pnl)}">${signedMoney(pnl)}</td><td>${number(risk?Math.abs(pnl/risk):0,2)}R</td><td>${esc(t.reason)}</td><td>${shortTime(t.closed_at)}</td><td>${detailButton(t)}</td></tr>`}).join("");
}

function renderModels(models) {
  const trained=models.filter(m=>m.status==="TRAINED"), total=models.length;
  $("modelCount").textContent=`${trained.length} / 20`;
  $("meanAccuracy").textContent=trained.length?pct(trained.reduce((s,m)=>s+(m.validation?.validation_accuracy||0),0)/trained.length):"—";
  $("meanBrier").textContent=trained.length?number(trained.reduce((s,m)=>s+(m.validation?.validation_brier||0),0)/trained.length,3):"—";
  $("modelGrid").innerHTML=total?models.slice().reverse().map(m=>{
    const v=m.validation||{};
    return `<article class="model-card"><div class="model-card-top"><span>${esc(m.pair)} · ${esc(m.timeframe)}</span><span>${esc(m.status)}</span></div><h3>Direction model · ${esc(String(m.id).split(":").pop())}</h3><p>${esc(m.reason||"Chronological validation")}</p><div class="model-scores"><div><small>HOLDOUT ACCURACY</small><b>${v.validation_accuracy==null?"—":pct(v.validation_accuracy)}</b></div><div><small>BRIER SCORE</small><b>${v.validation_brier==null?"—":number(v.validation_brier,3)}</b></div><div><small>LABELS</small><b>${number(m.sample_count)}</b></div></div><p style="margin-top:10px">Model available ${shortTime(v.trained_at)} · OOS begins ${shortTime(m.oos_start)}</p></article>`;
  }).join(""):'<div class="empty-card">Model runs start automatically after the Research evaluation finishes.</div>';
}

function renderEvidence(data) {
  const r=data.research||{}, d=data.dataset||{}, q=d.limitations||[];
  $("researchResults").textContent=number(r.prior_phase_result_count);
  const sync=data.storage_sync||{};
  $("researchStatus").innerHTML=`<div class="data-row"><span>Evaluation</span><b>${esc(r.status)}</b></div><div class="data-row"><span>Prior profitability</span><b>${esc(r.prior_phase_profitability||"UNPROVEN")}</b></div><div class="data-row"><span>Evidence treatment</span><b>Context only; kept separate</b></div><div class="data-row"><span>Supabase mirror</span><b>${esc(sync.status||"LOCAL SPOOL READY")}</b></div>`;
  $("datasetRows").textContent=`${number(d.rows)} bars`;
  $("datasetDetails").textContent=`${d.source||"Yllvar/fx-historical-data"} · ${d.structural_pass_files||0} of 20 files passed structural checks. Timestamp convention is corrected from bar-open to bar-close availability before training and simulation.`;
  $("datasetPills").innerHTML=[...(d.pairs||[]),...(d.timeframes||[])].map(x=>`<span class="pill">${esc(x)}</span>`).join("");
  $("limitations").innerHTML=(q.length?q:["No quality audit is available yet."]).map(item=>`<div class="warning-item">${esc(item)}</div>`).join("");
}

async function refresh() {
  try {
    const response=await fetch("/api/state",{cache:"no-store"}); if(!response.ok)throw new Error("status unavailable");
    snapshot=await response.json(); updateOverview(snapshot);
  } catch(error) { $("connectionState").textContent="RECONNECTING"; }
}

async function control(action) {
  if(!snapshot?.control_token)return;
  try {
    const response=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json","X-Astra-Token":snapshot.control_token},body:JSON.stringify({action})});
    if(!response.ok)throw new Error(); showToast(action==="pause"?"Pause requested; it will stop at the next safe checkpoint.":"Simulation resumed."); await refresh();
  } catch { showToast("Control request did not reach the local arena."); }
}

document.querySelectorAll(".nav-item").forEach(button=>button.addEventListener("click",()=>setView(button.dataset.view)));
document.querySelectorAll("[data-jump]").forEach(button=>button.addEventListener("click",()=>setView(button.dataset.jump)));
$("refreshBtn").addEventListener("click",refresh);
$("pauseBtn").addEventListener("click",()=>control(snapshot?.arena?.status==="PAUSED"?"resume":"pause"));
$("pairFilter").addEventListener("change",renderTrades); $("timeframeFilter").addEventListener("change",renderTrades);
document.addEventListener("click", event => {
  const button = event.target.closest("[data-trade-detail]");
  if (button) showTradeDetails(tradeDetails.get(button.dataset.tradeDetail));
  if (event.target === $("tradeModal") || event.target.closest("#tradeModalClose")) $("tradeModal").hidden = true;
});
document.addEventListener("keydown", event => { if (event.key === "Escape") $("tradeModal").hidden = true; });
window.addEventListener("resize",()=>snapshot&&drawEquity(snapshot.campaign.tests||[]));
refresh(); setInterval(refresh,2000);
