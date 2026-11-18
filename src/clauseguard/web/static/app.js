/* ClauseGuard analyser page. Plain JavaScript, no build step.
 *
 * The pure helpers at the top are also loaded by the tests (tests/js/app.test.mjs).
 * User text is only ever inserted with textContent / text nodes, never as HTML.
 */
"use strict";

const SEVERITY_ORDER = ["high", "medium", "low", "protective", "info"];
const SEVERITY_LABEL = {
  high: "High", medium: "Medium", low: "Low", protective: "Protective", info: "Info",
};
const RISKY = ["high", "medium", "low"];

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

/** Risky findings per risk group: [[group name, {high, medium, low, total}]], largest first. */
function groupBreakdown(findings) {
  const groups = new Map();
  for (const f of findings) {
    if (!RISKY.includes(f.severity)) continue;
    const g = groups.get(f.risk_group_name) || { high: 0, medium: 0, low: 0, total: 0 };
    g[f.severity] += 1;
    g.total += 1;
    groups.set(f.risk_group_name, g);
  }
  return [...groups.entries()].sort((a, b) => b[1].total - a[1].total || b[1].high - a[1].high);
}

/** One or two sentences that sum up the report. */
function verdictText(result) {
  const risky = result.findings.filter((f) => RISKY.includes(f.severity));
  if (!risky.length) {
    return "ClauseGuard didn't flag any risky clauses in this document. That is a good sign, "
      + "but it is not a guarantee: read anything important yourself.";
  }
  const high = risky.filter((f) => f.severity === "high").length;
  const top = groupBreakdown(result.findings)[0];
  const focus = top ? ` Most of the flags concern ${top[0].toLowerCase()}.` : "";
  if (result.risk_band === "high") {
    return `More aggressive than most documents of its kind, with ${high} high-severity `
      + `clause${high === 1 ? "" : "s"}.${focus} Read those before you agree.`;
  }
  if (result.risk_band === "medium") {
    return `About as demanding as a typical document of its kind.${focus} `
      + (high ? `The ${high} high-severity flag${high === 1 ? " is" : "s are"} worth reading first.`
        : "None of the flags is high-severity.");
  }
  return `Milder than most documents of its kind.${focus} A few clauses are still worth knowing about.`;
}

/** A plain-text summary of the report (for the clipboard). */
function reportText(result, limit = 10) {
  const lines = [
    `ClauseGuard risk report: ${result.document_type_name}`,
    `Risk score ${Math.round(result.risk_score ?? 0)}/100 (${result.risk_band || "n/a"}) · ${summaryLine(result)}`,
    "",
  ];
  const order = result.findings
    .filter((f) => RISKY.includes(f.severity))
    .sort((a, b) => SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity));
  order.slice(0, limit).forEach((f, i) => {
    const quote = f.text.length > 160 ? `${f.text.slice(0, 157)}…` : f.text;
    lines.push(`${i + 1}. [${SEVERITY_LABEL[f.severity]}] ${f.labels[0].name}: "${quote}"`);
    lines.push(`   ${f.explanation}`);
  });
  if (order.length > limit) lines.push(`…and ${order.length - limit} more.`);
  lines.push("", "Not legal advice.");
  return lines.join("\n");
}

/** The hint shown under the type choice for a /detect answer, and the type to pre-select
 *  (null when the router is unsure or the text looks unrelated). `names` maps id -> name. */
function detectionHint(d, names) {
  if (!d) return { text: "", select: null };
  if (d.unrelated) {
    return { text: "This doesn't look like a terms of service, a privacy policy or a contract. "
      + "If it is one, choose its type above.", select: null };
  }
  if (!d.confident || !names[d.document_type]) {
    const guess = Object.entries(d.probabilities || {})
      .filter(([k]) => names[k]).sort((a, b) => b[1] - a[1])[0];
    const maybe = guess ? ` It might be a ${names[guess[0]]} (${Math.round(guess[1] * 100)}%).` : "";
    return { text: `ClauseGuard isn't sure what kind of document this is.${maybe} Please choose.`,
      select: null };
  }
  return { text: `Looks like a ${names[d.document_type]} (${Math.round(d.confidence * 100)}% sure).`,
    select: d.document_type };
}

function wordCount(text) {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

if (typeof module !== "undefined") {
  module.exports = {
    segments, severityCounts, summaryLine, detectionHint, groupBreakdown, verdictText,
    reportText, wordCount, SEVERITY_ORDER,
  };
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
  const SVG = "http://www.w3.org/2000/svg";
  const icon = (name) => {
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  };
  const toast = (m) => (window.cgToast ? window.cgToast(m) : undefined);
  const TYPE_ICON = { tos: "tos", privacy: "privacy", contract: "contract" };
  const DROP_DEFAULT = $("drop-label").cloneNode(true);

  const state = {
    type: null, mode: "file", file: null, result: null, hidden: new Set(), query: "",
    userChoseType: false, names: {}, models: {}, detectSeq: 0, busy: false,
  };

  function hasInput() {
    return state.mode === "file" ? Boolean(state.file) : $("text").value.trim().length > 0;
  }

  function updateButton() {
    $("analyze").disabled = !hasInput() || state.busy;
    $("reset").hidden = !(hasInput() || state.result);
    document.querySelectorAll("#examples button").forEach((b) => { b.disabled = state.busy; });
  }

  function setHint(text, kind) {
    const hint = $("detect-hint");
    hint.className = `detect-hint${kind ? ` ${kind}` : ""}`;
    hint.replaceChildren(icon(kind === "ok" ? "check-circle" : kind === "warn" ? "alert" : "info"),
      el("span", null, text));
  }

  function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
  }

  /* ---- document types ---- */
  async function loadTypes() {
    const box = $("types");
    try {
      const types = await (await fetch("/document-types")).json();
      box.replaceChildren();
      for (const t of types) {
        const label = el("label", `type${t.ready ? "" : " disabled"}`);
        const input = el("input");
        Object.assign(input, { type: "radio", name: "doctype", value: t.id, disabled: !t.ready });
        input.addEventListener("change", () => {
          state.type = t.id; state.model = t.model; state.userChoseType = true; updateButton();
        });
        if (t.ready) { state.names[t.id] = t.name; state.models[t.id] = t.model; }
        const iconBox = el("span", "doc-icon");
        iconBox.append(icon(TYPE_ICON[t.id] || "file"));
        const body = el("span");
        body.append(el("strong", null, t.name),
          el("small", null, t.ready ? t.description : "Not available yet: no trained model."));
        if (t.ready && t.model) body.append(el("span", "model", `Model: ${t.model}`));
        const check = el("span", "check");
        check.append(icon("check"));
        label.append(input, iconBox, body, check);
        box.append(label);
      }
    } catch (err) {
      box.replaceChildren(el("p", "muted", "Could not reach the server. Is it running?"));
    }
  }

  function setType(id, byUser = false) {
    const input = document.querySelector(`input[name="doctype"][value="${id}"]`);
    if (!input || input.disabled) return false;
    input.checked = true;
    state.type = id;
    state.model = state.models[id];
    if (byUser) state.userChoseType = true;
    updateButton();
    return true;
  }

  async function suggestType() {
    const form = new FormData();
    if (state.mode === "file") {
      if (!state.file) return;
      form.append("file", state.file);
    } else {
      const text = $("text").value;
      if (wordCount(text) < 50) return;  // too short to judge
      form.append("text", text);
    }
    const seq = ++state.detectSeq;
    setHint("Working out what kind of document this is…");
    try {
      const response = await fetch("/detect", { method: "POST", body: form });
      if (seq !== state.detectSeq) return;  // a newer request is on its way
      if (!response.ok) { setHint("Choose the document type above."); return; }
      const hint = detectionHint(await response.json(), state.names);
      let kind = hint.select ? "ok" : "warn";
      if (hint.select && !state.userChoseType) {
        setType(hint.select);
        hint.text += " Selected for you; change it if that's wrong.";
      } else if (hint.select && state.type !== hint.select) {
        hint.text += ` You chose ${state.names[state.type] || "another type"}.`;
        kind = "warn";
      }
      setHint(hint.text, kind);
    } catch (err) { /* detection is optional */ }
  }

  let suggestTimer = null;
  function suggestSoon() {
    clearTimeout(suggestTimer);
    suggestTimer = setTimeout(suggestType, 600);
  }

  /* ---- input ---- */
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
    const drop = $("drop");
    drop.classList.toggle("has-file", Boolean(file));
    const label = $("drop-label");
    if (file) {
      const chip = el("span", "file-chip");
      const kb = file.size / 1024;
      chip.append(icon("file"), el("span", null,
        `${file.name} · ${kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(kb))} KB`}`));
      label.replaceChildren(chip, el("br"), el("small", "muted", "Click to choose a different file"));
    } else {
      label.replaceChildren(...DROP_DEFAULT.cloneNode(true).childNodes);
    }
    updateButton();
    suggestType();
  }

  function updateCount() {
    const n = wordCount($("text").value);
    $("word-count").textContent = `${n.toLocaleString("en")} word${n === 1 ? "" : "s"}`;
  }

  async function loadExample(type) {
    try {
      const response = await fetch(`/examples/${type}`);
      if (!response.ok) throw new Error();
      selectTab("text");
      $("text").value = await response.text();
      updateCount();
      state.userChoseType = false;
      setType(type);
      setStatus("Example loaded. Press Analyse.");
      suggestType();
      updateButton();
      $("analyze").focus({ preventScroll: true });
    } catch (err) {
      setStatus("Could not load the example.", true);
    }
  }

  function reset() {
    $("text").value = "";
    updateCount();
    $("file").value = "";
    chooseFile(null);
    state.result = null;
    state.userChoseType = false;
    state.type = null;
    document.querySelectorAll('input[name="doctype"]').forEach((i) => { i.checked = false; });
    setHint("Add the document below and ClauseGuard will suggest its type. You can always change it.");
    $("results").hidden = true;
    $("workflow").hidden = true;
    setStatus("");
    updateButton();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ---- progress: shows each stage while the server works, then real numbers ---- */
  const STAGES = [
    ["read", "Read"],
    ["clean", "Clean text"],
    ["split", "Split"],
    ["classify", "Classify"],
    ["score", "Score"],
    ["explain", "Explain"],
  ];
  let pipelineTimer = null;

  function setStage(id, status, detail) {
    const node = $(`stage-${id}`);
    node.className = `stage ${status}`;
    if (detail !== undefined) node.querySelector(".detail").textContent = detail;
  }

  function startPipeline() {
    $("workflow").hidden = false;
    $("pipeline").replaceChildren(...STAGES.map(([id, name]) => {
      const item = el("li", "stage");
      item.id = `stage-${id}`;
      const title = el("div", "name");
      const dot = el("span", "dot");
      dot.append(icon("check"));
      title.append(dot, name);
      item.append(title, el("div", "detail", "waiting"));
      return item;
    }));
    let i = 0;
    setStage(STAGES[0][0], "active", "working…");
    clearInterval(pipelineTimer);
    pipelineTimer = setInterval(() => {  // the server reports timings only at the end
      if (i >= 3) return;  // stay on "Classify" (the slow step) until the answer arrives
      setStage(STAGES[i][0], "done", "done");
      i += 1;
      setStage(STAGES[i][0], "active", i === 3 && state.model ? `${state.model}…` : "working…");
    }, 700);
  }

  function finishPipeline(r) {
    clearInterval(pipelineTimer);
    const t = r.timings_ms, d = r.diagnostics || {};
    const unit = r.findings[0] ? r.findings[0].unit_type : (r.document_type === "privacy" ? "paragraph" : "sentence");
    const s = (ms) => `${((ms || 0) / 1000).toFixed(1)} s`;
    setStage("read", "done", r.n_pages ? `PDF, ${r.n_pages} pages` : "text");
    setStage("clean", "done", `${r.n_words.toLocaleString("en")} words`);
    setStage("split", "done", `${r.n_units} ${unit}s · ${s(t.segmentation)}`);
    const read = d.units_read_by_model;
    const model = state.models[r.document_type] || r.model.model;
    setStage("classify", "done", `${model} · ${read !== undefined ? `${read} read · ` : ""}${s(t.classification)}`);
    setStage("score", "done", `${Math.round(r.risk_score ?? 0)}/100 · ${r.findings.length} flags`);
    const llm = r.findings.some((f) => f.explanation_source === "llm");
    setStage("explain", "done", `${llm ? "Claude" : "plain-language"} · total ${s(t.total)}`);
  }

  function failPipeline(message) {
    clearInterval(pipelineTimer);
    const active = document.querySelector(".stage.active") || $(`stage-${STAGES[0][0]}`);
    if (active) { active.className = "stage failed"; active.querySelector(".detail").textContent = message; }
  }

  async function analyze() {
    if (!hasInput() || state.busy) return;
    const form = new FormData();
    form.append("document_type", state.type || "auto");
    if (state.mode === "file") form.append("file", state.file);
    else form.append("text", $("text").value);
    state.busy = true;
    updateButton();
    $("results").hidden = true;
    setStatus("Analysing… this takes a few seconds.");
    startPipeline();
    try {
      const response = await fetch("/analyze", { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) {
        const message = body.error ? body.error.message : "Something went wrong.";
        failPipeline("stopped");
        setStatus(message, true);
        if (body.error && ["uncertain_document_type", "unrelated_document"].includes(body.error.code)) {
          setHint("Choose the document type above, then press Analyse again.", "warn");
          $("types").scrollIntoView({ behavior: "smooth", block: "center" });
        }
        return;
      }
      state.result = body;
      state.hidden.clear();
      state.query = "";
      $("search").value = "";
      if (body.type_source === "router") setType(body.document_type);
      finishPipeline(body);
      setStatus(`Done in ${(body.timings_ms.total / 1000).toFixed(1)} s.`);
      render(true);
    } catch (err) {
      failPipeline("server not reachable");
      setStatus("Could not reach the server.", true);
    } finally {
      state.busy = false;
      updateButton();
    }
  }

  /* ---- results ---- */
  function focusFinding(i, scrollDocument) {
    document.querySelectorAll("mark.active, .finding.active").forEach((n) => n.classList.remove("active"));
    const mark = $(`hl-${i}`);
    const card = $(`finding-${i}`);
    if (mark) mark.classList.add("active");
    if (card) card.classList.add("active");
    const target = scrollDocument ? mark : card;
    if (!target) return;
    const pane = target.closest(".doc-text, .finding-list");
    if (pane && pane.scrollHeight > pane.clientHeight) {
      const top = target.offsetTop - pane.offsetTop - pane.clientHeight / 3;
      pane.scrollTo({ top, behavior: "smooth" });
    } else {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  async function sendFeedback(finding, agree, box) {
    const r = state.result;
    try {
      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_type: r.document_type, model_run: r.model.run_id,
          label: finding.labels[0].label, agree }),
      });
      box.replaceChildren(el("span", null, "Thanks for the feedback."));
    } catch (err) {
      box.append(el("span", null, " Could not send feedback."));
    }
  }

  function renderHead(r) {
    const score = Math.round(r.risk_score ?? 0);
    const band = r.risk_band || "low";
    $("gauge").className = `gauge band-${band}`;
    $("score-value").textContent = String(score);
    $("score-band").textContent = `${band} risk`;
    const circumference = 2 * Math.PI * 62;
    const bar = $("gauge-bar");
    bar.style.strokeDashoffset = String(circumference);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bar.style.strokeDashoffset = String(circumference * (1 - score / 100));
    }));
    $("report-type").textContent = r.type_source === "router"
      ? `${r.document_type_name} · detected automatically` : r.document_type_name;
    $("results-title").textContent = `${band[0].toUpperCase()}${band.slice(1)} risk`;
    const timing = r.timings_ms && r.timings_ms.total ? ` · ${(r.timings_ms.total / 1000).toFixed(1)} s` : "";
    $("summary-line").textContent = `${summaryLine(r)} · ${state.models[r.document_type] || r.model.model}${timing}`;
    $("verdict").textContent = verdictText(r);
    $("warnings").replaceChildren(...r.warnings.map((w) => el("li", null, w)));
  }

  function renderTiles(r) {
    const counts = Object.fromEntries(severityCounts(r.findings));
    const shown = ["high", "medium", "low"];
    if (counts.protective) shown.push("protective");
    if (counts.info) shown.push("info");
    $("tiles").replaceChildren(...shown.map((s) => {
      const tile = el("div", `tile sev-${s}`);
      tile.append(el("strong", null, String(counts[s] || 0)),
        el("span", null, s === "protective" ? "protective clauses" : s === "info" ? "for information"
          : `${SEVERITY_LABEL[s].toLowerCase()}-severity flags`));
      return tile;
    }));
  }

  function renderGroups(r) {
    const groups = groupBreakdown(r.findings);
    const box = $("group-bars");
    if (!groups.length) { box.replaceChildren(el("p", "empty-note", "No risky clauses were flagged.")); return; }
    const max = Math.max(...groups.map(([, g]) => g.total));
    box.replaceChildren(...groups.map(([name, g]) => {
      const row = el("div", "row");
      const track = el("div", "track");
      track.title = RISKY.filter((s) => g[s]).map((s) => `${g[s]} ${s}`).join(", ");
      for (const s of RISKY) {
        if (!g[s]) continue;
        const part = el("i", `sev-${s}`);
        part.style.width = `${(100 * g[s]) / max}%`;
        track.append(part);
      }
      row.append(el("span", null, name), track, el("b", null, String(g.total)));
      return row;
    }));
  }

  function renderSide(r) {
    const list = $("protections-list");
    const hasProtections = r.protections_found.length + r.protections_missing.length > 0;
    if (hasProtections) {
      $("protections-title").textContent = "Protections in this policy";
      list.replaceChildren(
        ...r.protections_found.map((p) => el("li", "ok", p)),
        ...r.protections_missing.map((p) => el("li", "missing", p)),
      );
      return;
    }
    $("protections-title").textContent = "Read these first";
    // one line per category, most severe first; clicking jumps to its most confident clause
    const byLabel = new Map();
    r.findings.forEach((f, i) => {
      if (!RISKY.includes(f.severity)) return;
      const name = f.labels[0].name;
      const entry = byLabel.get(name);
      if (!entry) byLabel.set(name, { f, i, n: 1 });
      else {
        entry.n += 1;
        if (SEVERITY_ORDER.indexOf(f.severity) < SEVERITY_ORDER.indexOf(entry.f.severity)
          || (f.severity === entry.f.severity && f.labels[0].score > entry.f.labels[0].score)) {
          entry.f = f; entry.i = i;
        }
      }
    });
    const top = [...byLabel.entries()].sort((a, b) =>
      SEVERITY_ORDER.indexOf(a[1].f.severity) - SEVERITY_ORDER.indexOf(b[1].f.severity) || b[1].n - a[1].n)
      .slice(0, 5);
    if (!top.length) { list.replaceChildren(el("li", "ok", "Nothing stood out in this document.")); return; }
    list.replaceChildren(...top.map(([name, { f, i, n }]) => {
      const item = el("li");
      item.style.display = "block";
      const link = el("button", "link", name);
      link.type = "button";
      link.style.margin = "0 8px";
      link.addEventListener("click", () => { focusFinding(i, true); });
      item.append(el("span", `badge sev-${f.severity}`, SEVERITY_LABEL[f.severity]), link,
        el("span", "muted small", n === 1 ? "1 clause" : `${n} clauses`));
      return item;
    }));
  }

  function renderFilters(r) {
    const box = $("filters");
    box.replaceChildren();
    for (const [severity, count] of severityCounts(r.findings)) {
      const chip = el("button", `chip sev-${severity}`);
      chip.type = "button";
      chip.append(el("i"), `${SEVERITY_LABEL[severity]} ${count}`);
      chip.setAttribute("aria-pressed", String(!state.hidden.has(severity)));
      chip.addEventListener("click", () => {
        if (state.hidden.has(severity)) state.hidden.delete(severity);
        else state.hidden.add(severity);
        render(false);
      });
      box.append(chip);
    }
  }

  function matches(f) {
    if (state.hidden.has(f.severity)) return false;
    if (!state.query) return true;
    const q = state.query.toLowerCase();
    return [f.text, f.explanation, f.risk_group_name, ...f.labels.map((l) => l.name)]
      .some((s) => s && s.toLowerCase().includes(q));
  }

  function renderDocument(r) {
    const nodes = segments(r.text, r.findings).map((s) => {
      if (s.finding === null) return document.createTextNode(s.text);
      const f = r.findings[s.finding];
      const shown = matches(f);
      const mark = el("mark", shown ? `sev-${f.severity}` : "hidden-sev", s.text);
      mark.id = `hl-${s.finding}`;
      if (shown) {
        mark.title = `${SEVERITY_LABEL[f.severity]}: ${f.labels.map((l) => l.name).join(", ")}`;
        mark.addEventListener("click", () => focusFinding(s.finding, false));
      }
      return mark;
    });
    $("document").replaceChildren(...nodes);
  }

  function renderFindings(r) {
    const list = $("findings");
    list.replaceChildren();
    const order = r.findings.map((f, i) => i).sort((a, b) =>
      SEVERITY_ORDER.indexOf(r.findings[a].severity) - SEVERITY_ORDER.indexOf(r.findings[b].severity)
      || a - b);
    let shown = 0;
    for (const i of order) {
      const f = r.findings[i];
      if (!matches(f)) continue;
      shown += 1;
      const item = el("li", `finding sev-${f.severity}`);
      item.id = `finding-${i}`;
      item.tabIndex = 0;
      const head = el("div", "head");
      head.append(el("span", `badge sev-${f.severity}`, SEVERITY_LABEL[f.severity]),
        el("span", "label-name", f.labels[0].name),
        el("span", "meta", `${Math.round(f.labels[0].score * 100)}% sure${f.page ? ` · p. ${f.page}` : ""}`));
      const quote = el("p", "quote", f.text.length > 300 ? `${f.text.slice(0, 297)}…` : f.text);
      const explanation = el("p", "explanation", f.explanation);
      const foot = el("div", "foot");
      const group = el("span", null, f.risk_group_name);
      const jump = el("button", "link", "Show in document");
      jump.type = "button";
      jump.addEventListener("click", (e) => { e.stopPropagation(); focusFinding(i, true); });
      const votes = el("span");
      const yes = el("button", "vote", "Correct");
      const no = el("button", "vote", "Not correct");
      yes.type = no.type = "button";
      yes.addEventListener("click", (e) => { e.stopPropagation(); sendFeedback(f, true, votes); });
      no.addEventListener("click", (e) => { e.stopPropagation(); sendFeedback(f, false, votes); });
      votes.append(yes, " ", no);
      foot.append(group, el("span", "spacer"), jump, votes);
      item.append(head, quote, explanation, foot);
      item.addEventListener("click", () => focusFinding(i, true));
      item.addEventListener("keydown", (e) => { if (e.key === "Enter") focusFinding(i, true); });
      list.append(item);
    }
    $("findings-count").textContent = `${shown} of ${r.findings.length} shown`;
    if (!shown) {
      list.append(el("li", "none", r.findings.length ? "No flags match these filters."
        : "No clauses were flagged in this document."));
    }
  }

  function render(first) {
    const r = state.result;
    $("results").hidden = false;
    if (first) {
      renderHead(r);
      renderTiles(r);
      renderGroups(r);
      renderSide(r);
    }
    renderFilters(r);
    renderDocument(r);
    renderFindings(r);
    if (first) $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---- report actions ---- */
  async function copyReport() {
    try {
      await navigator.clipboard.writeText(reportText(state.result));
      toast("Summary copied to the clipboard");
    } catch (err) {
      toast("Could not copy: your browser blocked the clipboard");
    }
  }

  function downloadReport() {
    const r = state.result;
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
    const a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = `clauseguard-${r.document_type}-report.json`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  /* ---- start ---- */
  document.addEventListener("DOMContentLoaded", async () => {
    $("tab-file").addEventListener("click", () => selectTab("file"));
    $("tab-text").addEventListener("click", () => selectTab("text"));
    $("file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
    $("text").addEventListener("input", () => { updateCount(); updateButton(); suggestSoon(); });
    $("analyze").addEventListener("click", analyze);
    $("reset").addEventListener("click", reset);
    $("copy-report").addEventListener("click", copyReport);
    $("download-report").addEventListener("click", downloadReport);
    $("print-report").addEventListener("click", () => window.print());
    $("search").addEventListener("input", (e) => { state.query = e.target.value.trim(); render(false); });
    document.querySelectorAll("#examples button").forEach((b) => {
      b.addEventListener("click", () => loadExample(b.dataset.example));
    });
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); analyze(); }
    });
    const drop = $("drop");
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]);
    });

    await loadTypes();
    const params = new URLSearchParams(location.search);
    const example = params.get("example");
    const type = params.get("type");
    if (example) loadExample(example);
    else if (type && setType(type, true)) setHint(`${state.names[type]} selected. Add the document below.`, "ok");
    updateButton();
  });
}
