/* ClauseGuard web UI. Plain JavaScript, no build step.
 *
 * The pure helpers at the top are also loaded by the Node tests (tests/js/app.test.mjs).
 * User text is only ever inserted with textContent / text nodes, never as HTML.
 */
"use strict";

const SEVERITY_ORDER = ["high", "medium", "low", "protective", "info"];
const SEVERITY_LABEL = {
  high: "High", medium: "Medium", low: "Low", protective: "Protective", info: "Info",
};

/** Split `text` into consecutive pieces: [{text, finding}] where finding is an index or null.
 *  Findings are ordered by position; one that overlaps an earlier one is skipped. */
function segments(text, findings) {
  const order = findings
    .map((f, i) => ({ start: f.char_start, end: f.char_end, i }))
    .filter((f) => f.start >= 0 && f.end <= text.length && f.end > f.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const out = [];
  let pos = 0;
  for (const f of order) {
    if (f.start < pos) continue; // overlap: keep the earlier one
    if (f.start > pos) out.push({ text: text.slice(pos, f.start), finding: null });
    out.push({ text: text.slice(f.start, f.end), finding: f.i });
    pos = f.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), finding: null });
  return out;
}

/** Counts per severity, in display order, skipping severities with no findings. */
function severityCounts(findings) {
  const counts = {};
  for (const f of findings) counts[f.severity] = (counts[f.severity] || 0) + 1;
  return SEVERITY_ORDER.filter((s) => counts[s]).map((s) => [s, counts[s]]);
}

function summaryLine(result) {
  const parts = [
    `${result.document_type_name}`,
    `${result.n_words.toLocaleString("en")} words`,
  ];
  if (result.n_pages) parts.push(`${result.n_pages} page${result.n_pages === 1 ? "" : "s"}`);
  const n = result.findings.length;
  parts.push(`${n} flagged clause${n === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

if (typeof module !== "undefined") {
  module.exports = { segments, severityCounts, summaryLine, SEVERITY_ORDER };
}

/* ---------------------------------------------------------------- page */

if (typeof document !== "undefined") {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const state = { type: null, mode: "file", file: null, result: null, hidden: new Set() };

  function updateButton() {
    const hasInput = state.mode === "file" ? Boolean(state.file) : $("text").value.trim().length > 0;
    $("analyze").disabled = !(state.type && hasInput);
  }

  function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
  }

  async function loadTypes() {
    const box = $("types");
    try {
      const types = await (await fetch("/document-types")).json();
      box.replaceChildren();
      for (const t of types) {
        const label = el("label", "type" + (t.ready ? "" : " disabled"));
        const input = el("input");
        Object.assign(input, { type: "radio", name: "doctype", value: t.id, disabled: !t.ready });
        input.addEventListener("change", () => { state.type = t.id; updateButton(); });
        label.append(input, el("strong", null, t.name),
          el("small", null, t.ready ? t.description : "Not available yet: no trained model."));
        box.append(label);
      }
    } catch (err) {
      box.replaceChildren(el("p", "muted", "Could not reach the server."));
    }
  }

  function selectTab(mode) {
    state.mode = mode;
    $("tab-file").setAttribute("aria-selected", String(mode === "file"));
    $("tab-text").setAttribute("aria-selected", String(mode === "text"));
    $("pane-file").hidden = mode !== "file";
    $("pane-text").hidden = mode !== "text";
    updateButton();
  }

  function chooseFile(file) {
    state.file = file || null;
    $("drop-label").textContent = file ? `${file.name} (${Math.round(file.size / 1024)} KB)`
      : "Drop a PDF or text file here, or choose a file";
    updateButton();
  }

  async function analyze() {
    const form = new FormData();
    form.append("document_type", state.type);
    if (state.mode === "file") form.append("file", state.file);
    else form.append("text", $("text").value);
    $("analyze").disabled = true;
    setStatus("Analysing…");
    try {
      const response = await fetch("/analyze", { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) {
        setStatus(body.error ? body.error.message : "Something went wrong.", true);
        return;
      }
      state.result = body;
      state.hidden.clear();
      setStatus(`Done in ${(body.timings_ms.total / 1000).toFixed(1)} s.`);
      render();
    } catch (err) {
      setStatus("Could not reach the server.", true);
    } finally {
      updateButton();
    }
  }

  function focusFinding(i, scrollDocument) {
    document.querySelectorAll(".active").forEach((n) => n.classList.remove("active"));
    const mark = $(`hl-${i}`);
    const card = $(`finding-${i}`);
    if (mark) mark.classList.add("active");
    if (card) card.classList.add("active");
    const target = scrollDocument ? mark : card;
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function sendFeedback(finding, agree, foot) {
    const r = state.result;
    try {
      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_type: r.document_type, model_run: r.model.run_id,
          label: finding.labels[0].label, agree }),
      });
      foot.replaceChildren(el("span", "meta", "Thanks for the feedback."));
    } catch (err) {
      foot.append(el("span", "meta", " Could not send feedback."));
    }
  }

  function renderScore(r) {
    const score = $("score");
    score.className = `score band-${r.risk_band}`;
    score.replaceChildren(el("strong", null, String(Math.round(r.risk_score))),
      el("span", null, `${r.risk_band} risk`));
    score.title = "0–100: how risky this document is compared with typical documents of its type";
  }

  function renderProtections(r) {
    const show = r.protections_found.length + r.protections_missing.length > 0;
    $("protections").hidden = !show;
    const list = $("protections-list");
    list.replaceChildren(
      ...r.protections_found.map((p) => el("li", "ok", p)),
      ...r.protections_missing.map((p) => el("li", "missing", p)),
    );
  }

  function renderFilters(r) {
    const box = $("filters");
    box.replaceChildren();
    for (const [severity, count] of severityCounts(r.findings)) {
      const chip = el("button", "chip", `${SEVERITY_LABEL[severity]} (${count})`);
      chip.type = "button";
      chip.setAttribute("aria-pressed", String(!state.hidden.has(severity)));
      chip.addEventListener("click", () => {
        if (state.hidden.has(severity)) state.hidden.delete(severity);
        else state.hidden.add(severity);
        render();
      });
      box.append(chip);
    }
  }

  function renderDocument(r) {
    const box = $("document");
    const nodes = segments(r.text, r.findings).map((s) => {
      if (s.finding === null) return document.createTextNode(s.text);
      const f = r.findings[s.finding];
      const hidden = state.hidden.has(f.severity);
      const mark = el("mark", hidden ? "hidden-sev" : `sev-${f.severity}`, s.text);
      mark.id = `hl-${s.finding}`;
      if (!hidden) {
        mark.title = f.labels.map((l) => l.name).join(", ");
        mark.addEventListener("click", () => focusFinding(s.finding, false));
      }
      return mark;
    });
    box.replaceChildren(...nodes);
  }

  function renderFindings(r) {
    const list = $("findings");
    list.replaceChildren();
    const order = r.findings.map((f, i) => i).sort((a, b) =>
      SEVERITY_ORDER.indexOf(r.findings[a].severity) - SEVERITY_ORDER.indexOf(r.findings[b].severity)
      || a - b);
    for (const i of order) {
      const f = r.findings[i];
      if (state.hidden.has(f.severity)) continue;
      const item = el("li", `finding sev-${f.severity}`);
      item.id = `finding-${i}`;
      const head = el("div", "head");
      head.append(el("span", `badge sev-${f.severity}`, SEVERITY_LABEL[f.severity]),
        el("span", "label-name", f.labels[0].name),
        el("span", "meta", `${f.risk_group_name} · ${Math.round(f.labels[0].score * 100)}%`
          + (f.page ? ` · page ${f.page}` : "")));
      const quote = el("p", "quote", f.text.length > 280 ? f.text.slice(0, 277) + "…" : f.text);
      const explanation = el("p", null, f.explanation);
      const foot = el("div", "foot");
      const jump = el("button", "link", "Show in document");
      jump.type = "button";
      jump.addEventListener("click", () => focusFinding(i, true));
      const yes = el("button", "vote", "👍 Agree");
      const no = el("button", "vote", "👎 Disagree");
      yes.type = no.type = "button";
      const votes = el("span");
      votes.append(yes, " ", no);
      yes.addEventListener("click", () => sendFeedback(f, true, votes));
      no.addEventListener("click", () => sendFeedback(f, false, votes));
      foot.append(jump, votes);
      item.append(head, quote, explanation, foot);
      list.append(item);
    }
    if (!list.children.length) list.append(el("li", "muted", "No clauses to show."));
  }

  function render() {
    const r = state.result;
    $("results").hidden = false;
    $("summary-line").textContent = summaryLine(r);
    $("warnings").replaceChildren(...r.warnings.map((w) => el("li", null, w)));
    renderScore(r);
    renderProtections(r);
    renderFilters(r);
    renderDocument(r);
    renderFindings(r);
    $("results").scrollIntoView({ behavior: "smooth" });
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadTypes();
    $("tab-file").addEventListener("click", () => selectTab("file"));
    $("tab-text").addEventListener("click", () => selectTab("text"));
    $("file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
    $("text").addEventListener("input", updateButton);
    $("analyze").addEventListener("click", analyze);
    const drop = $("drop");
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      chooseFile(e.dataTransfer.files[0]);
    });
  });
}
