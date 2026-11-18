/* ClauseGuard popup: read the current page, suggest its type, analyse it, highlight flags. */
"use strict";

const core = globalThis.ClauseGuardCore;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const state = { tab: null, page: null, text: "", source: "", apiBase: core.DEFAULT_API, result: null };

function setStatus(message, isError = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", isError);
}

async function settings() {
  const stored = await chrome.storage.sync.get({ apiUrl: core.DEFAULT_API });
  return stored.apiUrl || core.DEFAULT_API;
}

/** Inject the content script (only when the user opened the popup) and ask it something. */
async function askPage(message) {
  await chrome.scripting.executeScript({
    target: { tabId: state.tab.id },
    files: ["lib/core.js", "content.js"],
  });
  return chrome.tabs.sendMessage(state.tab.id, message);
}

async function readPage() {
  [state.tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!state.tab || !/^https?:|^file:/.test(state.tab.url || "")) {
    $("page-info").textContent = "Open a web page (terms, privacy policy or contract) to analyse it.";
    return false;
  }
  try {
    state.page = await askPage({ type: "cg:text" });
  } catch (err) {
    $("page-info").textContent = "ClauseGuard can't read this page (browser pages and the "
      + "Chrome Web Store are protected). For a PDF, use the web app.";
    return false;
  }
  const chosen = core.chooseText(state.page);
  state.text = chosen.text;
  state.source = chosen.source;
  const words = core.wordCount(state.text);
  const what = { selection: "the selected text", main: "this page's main content", page: "this page" }[chosen.source];
  $("page-info").textContent = `${state.page.title || state.tab.url} · ${words.toLocaleString("en")} words (${what})`;
  if (words < core.MIN_WORDS) {
    setStatus(`There's too little text here (${words} words; at least ${core.MIN_WORDS} needed). `
      + "Select the document's text, or open the full document.", true);
    return false;
  }
  return true;
}

async function suggestType() {
  try {
    const d = await core.api.detect(state.apiBase, state.text);
    $("detect-hint").textContent = core.detectionText(d);
    if (d.confident && core.TYPE_NAMES[d.document_type] && !$("doctype").value) {
      $("doctype").value = d.document_type;
      $("detect-hint").textContent += " Change it if that's wrong.";
    }
  } catch (err) {
    $("detect-hint").textContent = err.code === "unreachable" ? err.message : "Choose the document type.";
  }
  updateButton();
}

function updateButton() {
  $("analyze").disabled = !(state.text && $("doctype").value) || state.busy;
}

function renderResult(r, highlight) {
  $("start").hidden = true;
  $("result").hidden = false;
  const score = $("score");
  score.className = `score band-${r.risk_band}`;
  score.replaceChildren(el("strong", null, String(Math.round(r.risk_score))), el("span", null, `${r.risk_band} risk`));
  const counts = core.severityCounts(r.findings).map(([s, n]) => `${n} ${core.SEVERITY_LABEL[s].toLowerCase()}`);
  $("summary-line").textContent = `${r.document_type_name} · ${r.findings.length} flagged`
    + (counts.length ? ` (${counts.join(", ")})` : "");
  $("highlight-line").textContent = highlight && !highlight.error
    ? `${highlight.placed} of ${highlight.total} highlighted on the page. Hover over a highlight for its explanation.`
    : "Highlights could not be placed on this page; see the full report.";
  $("warnings").replaceChildren(...r.warnings.map((w) => el("li", null, w)));
  const list = $("findings");
  list.replaceChildren();
  for (const { f, i } of core.sortedFindings(r.findings).slice(0, 8)) {
    const item = el("li", `finding sev-${f.severity}`);
    const head = el("button", "finding-head");
    head.type = "button";
    head.append(el("span", `badge sev-${f.severity}`, core.SEVERITY_LABEL[f.severity]), el("span", null, f.labels[0].name));
    head.title = "Show on the page";
    head.addEventListener("click", () => chrome.tabs.sendMessage(state.tab.id, { type: "cg:focus", index: i }));
    item.append(head, el("p", "explanation", f.explanation));
    list.append(item);
  }
  if (r.findings.length > 8) list.append(el("li", "muted small", `…and ${r.findings.length - 8} more in the full report.`));
  if (!r.findings.length) list.append(el("li", "muted", "No risky clauses found."));
}

async function analyze() {
  state.busy = true;
  updateButton();
  setStatus("Analysing… this can take 10–30 seconds on a free server.");
  try {
    const result = await core.api.analyze(state.apiBase, state.text, $("doctype").value);
    state.result = result;
    let highlight = null;
    try {
      highlight = await chrome.tabs.sendMessage(state.tab.id, { type: "cg:highlight", findings: result.findings });
    } catch (err) {
      highlight = { error: String(err) };
    }
    await chrome.storage.session.set({
      lastReport: { result, url: state.tab.url, title: state.page.title, source: state.source, at: Date.now() },
    });
    setStatus("");
    renderResult(result, highlight);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    state.busy = false;
    updateButton();
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  $("open-options").addEventListener("click", (e) => { e.preventDefault(); chrome.runtime.openOptionsPage(); });
  $("doctype").addEventListener("change", updateButton);
  $("analyze").addEventListener("click", analyze);
  $("report").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("report.html") }));
  $("clear").addEventListener("click", () => chrome.tabs.sendMessage(state.tab.id, { type: "cg:clear" }));
  $("again").addEventListener("click", () => { $("result").hidden = true; $("start").hidden = false; });
  state.apiBase = await settings();
  if (await readPage()) await suggestType();
});
