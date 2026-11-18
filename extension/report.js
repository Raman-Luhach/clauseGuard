/* ClauseGuard full report: the last analysis (from the popup or the right-click menu). */
"use strict";

const core = globalThis.ClauseGuardCore;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function render({ result: r, url, title, source }) {
  $("report").hidden = false;
  const where = { selection: "selected text", main: "main content", page: "whole page" }[source] || source;
  $("source").textContent = `${title || url || ""}${where ? ` · ${where}` : ""}`;
  const score = $("score");
  score.className = `score band-${r.risk_band}`;
  score.replaceChildren(el("strong", null, String(Math.round(r.risk_score))), el("span", null, `${r.risk_band} risk`));
  const counts = core.severityCounts(r.findings).map(([s, n]) => `${n} ${core.SEVERITY_LABEL[s].toLowerCase()}`);
  $("summary-line").textContent = `${r.document_type_name} · ${r.n_words.toLocaleString("en")} words · `
    + `${r.findings.length} flagged${counts.length ? ` (${counts.join(", ")})` : ""}`;
  const chosen = r.type_source === "router" ? "type detected automatically" : "type chosen by you";
  $("model-line").textContent = `Model ${r.model.model} · ${chosen}`;
  $("warnings").replaceChildren(...r.warnings.map((w) => el("li", null, w)));
  const show = r.protections_found.length + r.protections_missing.length > 0;
  $("protections").hidden = !show;
  $("protections-list").replaceChildren(
    ...r.protections_found.map((p) => el("li", null, `✓ ${p}`)),
    ...r.protections_missing.map((p) => el("li", null, `✗ ${p}`)),
  );
  const list = $("findings");
  list.replaceChildren();
  for (const { f } of core.sortedFindings(r.findings)) {
    const item = el("li", `finding sev-${f.severity}`);
    const head = el("div", "finding-head");
    head.append(el("span", `badge sev-${f.severity}`, core.SEVERITY_LABEL[f.severity]),
      el("span", null, f.labels.map((l) => l.name).join(", ")),
      el("span", "muted small", `${f.risk_group_name} · ${Math.round(f.labels[0].score * 100)}%`));
    item.append(head, el("p", "quote", f.text), el("p", "explanation", f.explanation));
    list.append(item);
  }
  if (!r.findings.length) list.append(el("li", "muted", "No risky clauses found."));
  $("disclaimer").textContent = r.disclaimer;
}

document.addEventListener("DOMContentLoaded", async () => {
  $("open-options").addEventListener("click", (e) => { e.preventDefault(); chrome.runtime.openOptionsPage(); });
  const { lastReport, lastError } = await chrome.storage.session.get(["lastReport", "lastError"]);
  if (lastError && (!lastReport || lastError.at > (lastReport.at || 0))) {
    $("error").hidden = false;
    $("error").textContent = lastError.message;
  } else if (lastReport) {
    render(lastReport);
  } else {
    $("empty").hidden = false;
  }
});
