// Aegis console. Security note: every value that can contain attacker-
// controlled text (memos, rule names, LLM output) is written with
// textContent / createElement, never innerHTML, so it can never become markup.
"use strict";

const $ = (id) => document.getElementById(id);
const store = {
  get() { try { return sessionStorage.getItem("aegis-key"); } catch { return null; } },
  set(v) { try { sessionStorage.setItem("aegis-key", v); } catch { /* private mode */ } },
  clear() { try { sessionStorage.removeItem("aegis-key"); } catch { /* ignore */ } },
};
let apiKey = store.get();
let activeAlert = null;
let ws = null;
let alertRefresh = null;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    // CSSOM assignment, not a style="" attribute: the strict CSP blocks the latter.
    else if (k === "style") Object.assign(node.style, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { "X-API-Key": apiKey, "Content-Type": "application/json", ...(opts.headers || {}) } });
  if (res.status === 401) { signOut("That key was rejected."); throw new Error("unauthorised"); }
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

const usd = (v) => v.toLocaleString(undefined, { style: "currency", currency: "USD" });
const time = (ms) => new Date(ms).toISOString().slice(5, 16).replace("T", " ");

// ---------------------------------------------------------------- gate
function signOut(message = "") {
  store.clear();
  apiKey = null;
  if (ws) ws.close();
  $("gate").hidden = false;
  $("gate-error").textContent = message;
  $("gate-key").focus();
}

$("gate-form").addEventListener("submit", (e) => {
  e.preventDefault();
  apiKey = $("gate-key").value.trim();
  store.set(apiKey);
  start();
});
$("logout").addEventListener("click", () => signOut());

// ---------------------------------------------------------------- header + stats
async function loadMeta() {
  const m = await api("/v1/meta");
  $("m-model").textContent = m.model_fingerprint.slice(0, 12);
  $("m-model").title = `SHA-256 ${m.model_fingerprint}`;
  $("m-review").textContent = `risk ≥ ${m.review_threshold.toFixed(3)}`;
  $("m-block").textContent = `risk ≥ ${m.block_threshold.toFixed(3)}`;
  $("m-investigator").textContent = m.investigator;
}

async function loadStats() {
  const s = await api("/v1/stats");
  $("k-processed").textContent = s.processed.toLocaleString();
  $("k-reviews").textContent = s.reviews.toLocaleString();
  $("k-blocks").textContent = s.blocks.toLocaleString();
  $("k-p50").textContent = `${s.latency.p50_us.toFixed(1)} µs`;
  $("k-p99").textContent = `${s.latency.p99_us.toFixed(1)} µs`;
}

async function loadAudit() {
  const a = await api("/v1/audit/verify");
  const dd = $("m-audit");
  dd.textContent = a.ok ? `✓ verified · ${a.entries} entries` : `✗ BROKEN at #${a.first_bad_seq}`;
  dd.className = a.ok ? "ok" : "bad";
  dd.title = a.ok ? `head ${a.head}` : a.reason;
}

// ---------------------------------------------------------------- live stream
function addTxn(e) {
  const body = $("stream");
  const bar = el("div", { class: "riskbar" }, el("i", { class: e.decision, style: { width: `${Math.round(e.risk * 100)}%` } }));
  const row = el("tr", { class: e.decision !== "ALLOW" ? "flash" : "" },
    el("td", { text: time(e.timestamp_ms) }),
    el("td", { text: String(e.account_id) }),
    el("td", { class: "num", text: usd(e.amount_usd) }),
    el("td", { text: e.category }),
    el("td", { text: e.channel }),
    el("td", {}, bar),
    el("td", {}, el("span", { class: `pill ${e.decision}`, text: e.decision })));
  body.prepend(row);
  while (body.children.length > 60) body.lastChild.remove();
  if (e.alert_id) { clearTimeout(alertRefresh); alertRefresh = setTimeout(loadAlerts, 400); }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/v1/stream`);
  ws.onopen = () => ws.send(JSON.stringify({ api_key: apiKey }));
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "hello") { $("k-conn").textContent = "live"; $("k-conn").className = "ok"; }
    if (e.type === "txn") addTxn(e);
  };
  ws.onclose = () => {
    $("k-conn").textContent = "offline"; $("k-conn").className = "bad";
    if (apiKey) setTimeout(connect, 2000);
  };
}

// ---------------------------------------------------------------- alerts
async function loadAlerts() {
  const list = await api("/v1/alerts?limit=60");
  const ul = $("alerts");
  ul.replaceChildren(...list.map((a) => {
    const chips = el("div");
    for (const r of a.result.rules) chips.append(el("span", { class: `chip ${a.result.hard_rule ? "hard" : ""}`, text: r }));
    const li = el("li", { tabindex: "0", role: "button", class: a.alert_id === activeAlert ? "active" : "",
        onclick: () => openCase(a.alert_id), onkeydown: (ev) => { if (ev.key === "Enter") openCase(a.alert_id); } },
      el("div", { class: "row1" },
        el("strong", { text: `${usd(a.summary.amount_usd)} · ${a.summary.category}` }),
        el("span", { class: `pill ${a.result.decision}`, text: a.result.decision })),
      el("div", { class: "row2", text: `${a.alert_id} · ${a.summary.channel} · risk ${a.result.risk.toFixed(2)}${a.investigated ? ` · ${a.status.replaceAll("_", " ")}` : ""}` }),
      chips);
    return li;
  }));
}

// ---------------------------------------------------------------- case view
function kv(pairs) {
  const dl = el("dl");
  for (const [k, v] of pairs) dl.append(el("dt", { text: k }), el("dd", { text: String(v) }));
  return dl;
}

async function openCase(id) {
  activeAlert = id;
  document.querySelectorAll(".alerts li").forEach((li) => li.classList.remove("active"));
  const a = await api(`/v1/alerts/${encodeURIComponent(id)}`);
  const t = a.transaction;
  const box = el("div", { class: "case" });
  box.append(
    el("div", { class: "row1" }, el("strong", { text: a.alert_id }), el("span", { class: `pill ${a.result.decision}`, text: a.result.decision })),
    el("h3", { text: "Transaction" }),
    kv([["Account", t.account_id], ["Amount", usd(t.amount_cents / 100)], ["Category", a.summary.category],
        ["Channel", a.summary.channel], ["Country", t.country], ["Device", t.device_id], ["Time (UTC)", time(t.timestamp_ms)],
        ["Risk", `${a.result.risk.toFixed(3)} (model ${a.result.model_score.toFixed(3)})`]]),
  );
  if (a.result.rules.length) {
    const chips = el("div");
    for (const r of a.result.rules) chips.append(el("span", { class: `chip ${a.result.hard_rule ? "hard" : ""}`, text: r }));
    box.append(el("h3", { text: "Rules fired" }), chips);
  }
  if (a.explanation.length) {
    box.append(el("h3", { text: "Why the model flagged it" }));
    const max = Math.max(...a.explanation.map((f) => f.contribution));
    for (const f of a.explanation) {
      box.append(el("div", { class: "factor" }, el("span", { text: f.description }),
        el("div", { class: "bar", style: { width: `${Math.max(6, (f.contribution / max) * 90)}px` }, title: `+${f.contribution}` })));
    }
  }
  box.append(el("h3", { text: "Memo" }), el("div", { class: "untrusted", text: t.memo || "(empty)" }));

  const reportBox = el("div");
  const btn = el("button", { type: "button", text: a.investigation ? "Re-run investigation" : "Investigate with AI" });
  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Investigating…";
    try {
      renderReport(reportBox, await api(`/v1/alerts/${encodeURIComponent(id)}/investigate?force=${!!a.investigation}`, { method: "POST" }));
      loadAlerts(); loadAudit();
    } catch (err) {
      reportBox.replaceChildren(el("p", { class: "error", text: err.message }));
    } finally { btn.disabled = false; btn.textContent = "Re-run investigation"; }
  });
  box.append(el("h3", { text: "Investigation" }), btn, reportBox);
  if (a.investigation) renderReport(reportBox, a.investigation);
  $("detail").className = "";
  $("detail").replaceChildren(box);
}

function renderReport(box, r) {
  const list = (items) => el("ul", {}, ...items.map((i) => el("li", { text: i })));
  const out = el("div", { class: "report" },
    el("div", { class: `verdict ${r.final_verdict}`, text: `${r.final_verdict.replaceAll("_", " ")}` }),
    el("p", { class: "muted", text: `${r.typology.replaceAll("_", " ")} · confidence ${(r.confidence * 100).toFixed(0)}% · by ${r.investigator} · ${r.telemetry.seconds}s` }));
  for (const o of r.policy_overrides) out.append(el("div", { class: "override", text: `Policy override (model said "${r.verdict}"): ${o}` }));
  if (r.injection.detected) out.append(el("div", { class: "override", text: `Prompt injection detected in memo: ${r.injection.indicators.join(", ")}` }));
  out.append(el("p", { text: r.summary }), el("strong", { text: "Evidence" }), list(r.evidence),
    el("strong", { text: "Recommended actions" }), list(r.recommended_actions));
  if (r.sar_narrative) out.append(el("strong", { text: "Draft SAR narrative" }), el("p", { class: "sar", text: r.sar_narrative }));
  if (r.error) out.append(el("p", { class: "muted", text: `LLM unavailable, deterministic fallback used (${r.error})` }));
  box.replaceChildren(out);
}

// ---------------------------------------------------------------- boot
async function start() {
  try {
    await loadMeta();
  } catch { return; }
  $("gate").hidden = true;
  connect();
  loadStats(); loadAlerts(); loadAudit();
  setInterval(() => apiKey && loadStats().catch(() => {}), 2000);
  setInterval(() => apiKey && loadAudit().catch(() => {}), 10000);
}

if (apiKey) start(); else signOut();
