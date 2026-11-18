/* ClauseGuard content script. Injected on demand (when the user clicks the toolbar button),
 * never on every page. It reads the page's text and highlights flagged clauses in place.
 * Page text is only ever read; findings are inserted with textContent, never as HTML.
 */
"use strict";

(function () {
  if (window.__clauseguardContent) return; // already injected into this page
  window.__clauseguardContent = true;
  const core = globalThis.ClauseGuardCore;
  const SKIP = "script, style, noscript, textarea, input, select, svg, [contenteditable=''], "
    + "[contenteditable='true'], .clauseguard-tip";
  const BLOCKS = "p, div, li, dd, dt, td, th, tr, table, ul, ol, dl, h1, h2, h3, h4, h5, h6, "
    + "section, article, main, header, footer, nav, aside, blockquote, pre, figure, form, body";

  function pageText() {
    const selection = String(window.getSelection ? window.getSelection() : "");
    const mainEl = document.querySelector("main, article, [role='main']");
    return {
      selection,
      main: mainEl ? mainEl.innerText : "",
      body: document.body ? document.body.innerText : "",
      title: document.title,
      url: location.href,
      lang: document.documentElement.lang || "",
    };
  }

  /* ---- text nodes of the page, joined with spaces, and where each one starts ---- */
  function textNodes() {
    const nodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const parent = node.parentElement;
        if (!parent || parent.closest(SKIP)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    while (walker.nextNode()) nodes.push(walker.currentNode);
    // Nodes in the same block ("We may <b>terminate</b> your") join without a gap, like the
    // page's visible text; nodes in different blocks, or after a <br>, get a space.
    let text = "";
    const starts = [];
    let lastBlock = null;
    for (const node of nodes) {
      const block = node.parentElement.closest(BLOCKS);
      const afterBreak = node.previousSibling && node.previousSibling.nodeName === "BR";
      if (text && (block !== lastBlock || afterBreak)) text += " ";
      starts.push(text.length);
      text += node.nodeValue;
      lastBlock = block;
    }
    return { nodes, starts, text };
  }

  function ensureStyle() {
    if (document.getElementById("clauseguard-style")) return;
    const style = document.createElement("style");
    style.id = "clauseguard-style";
    style.textContent = `
      mark.clauseguard-mark { background: #fde68a; color: inherit; border-radius: 2px;
        box-shadow: 0 0 0 1px rgba(0,0,0,.08); cursor: help; padding: 0 1px; }
      mark.clauseguard-mark.cg-high { background: #fecaca; }
      mark.clauseguard-mark.cg-medium { background: #fed7aa; }
      mark.clauseguard-mark.cg-low { background: #fef08a; }
      mark.clauseguard-mark.cg-protective { background: #bbf7d0; }
      mark.clauseguard-mark.cg-info { background: #bae6fd; }
      mark.clauseguard-mark.cg-focus { outline: 2px solid #1d4ed8; }
      .clauseguard-tip { position: absolute; z-index: 2147483647; max-width: 360px;
        background: #111827; color: #f9fafb; font: 13px/1.45 system-ui, sans-serif;
        padding: 10px 12px; border-radius: 8px; box-shadow: 0 6px 24px rgba(0,0,0,.25); }
      .clauseguard-tip strong { display: block; margin-bottom: 4px; font-size: 13px; }
      .clauseguard-tip small { display: block; margin-top: 6px; color: #9ca3af; }`;
    document.documentElement.appendChild(style);
  }

  let tip = null;
  function showTip(mark, finding) {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "clauseguard-tip";
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    const severity = core.SEVERITY_LABEL[finding.severity] || finding.severity;
    const title = document.createElement("strong");
    title.textContent = `${severity}: ${finding.labels.map((l) => l.name).join(", ")}`;
    const body = document.createElement("div");
    body.textContent = finding.explanation;
    const foot = document.createElement("small");
    foot.textContent = "ClauseGuard · not legal advice";
    tip.replaceChildren(title, body, foot);
    const rect = mark.getBoundingClientRect();
    tip.style.left = `${Math.max(8, window.scrollX + rect.left)}px`;
    tip.style.top = `${window.scrollY + rect.bottom + 6}px`;
    tip.hidden = false;
  }
  function hideTip() { if (tip) tip.hidden = true; }

  function clear() {
    hideTip();
    for (const mark of document.querySelectorAll("mark.clauseguard-mark")) {
      const parent = mark.parentNode;
      while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
      parent.removeChild(mark);
      parent.normalize();
    }
  }

  /** Wrap [start, end) of the joined text in <mark>s, one per text node it touches. */
  function wrap(index, finding, start, end, nodes, starts) {
    const pieces = [];
    for (let k = 0; k < nodes.length; k++) {
      const nodeStart = starts[k];
      const nodeEnd = nodeStart + nodes[k].nodeValue.length;
      if (nodeEnd <= start) continue;
      if (nodeStart >= end) break;
      pieces.push([nodes[k], Math.max(0, start - nodeStart), Math.min(nodes[k].nodeValue.length, end - nodeStart)]);
    }
    const marks = [];
    for (const [node, a, b] of pieces.reverse()) { // right to left keeps offsets valid
      if (b <= a) continue;
      const range = document.createRange();
      range.setStart(node, a);
      range.setEnd(node, b);
      const mark = document.createElement("mark");
      mark.className = `clauseguard-mark cg-${finding.severity}`;
      mark.dataset.clauseguard = String(index);
      range.surroundContents(mark);
      mark.addEventListener("mouseenter", () => showTip(mark, finding));
      mark.addEventListener("mouseleave", hideTip);
      marks.unshift(mark);
    }
    return marks;
  }

  function highlight(findings) {
    clear();
    ensureStyle();
    const { nodes, starts, text } = textNodes();
    const placed = core.locateFindings(text, findings);
    let first = null;
    for (const p of placed.slice().reverse()) { // later ranges first: earlier offsets stay valid
      const marks = wrap(p.index, findings[p.index], p.start, p.end, nodes, starts);
      if (marks.length && findings[p.index].severity === "high") first = marks[0];
    }
    return { placed: placed.length, total: findings.length, scrolled: Boolean(first) };
  }

  function focus(index) {
    document.querySelectorAll("mark.cg-focus").forEach((m) => m.classList.remove("cg-focus"));
    const marks = document.querySelectorAll(`mark.clauseguard-mark[data-clauseguard="${index}"]`);
    marks.forEach((m) => m.classList.add("cg-focus"));
    if (marks[0]) marks[0].scrollIntoView({ behavior: "smooth", block: "center" });
    return { found: marks.length > 0 };
  }

  chrome.runtime.onMessage.addListener((message, _sender, reply) => {
    try {
      if (message.type === "cg:text") reply(pageText());
      else if (message.type === "cg:highlight") reply(highlight(message.findings || []));
      else if (message.type === "cg:focus") reply(focus(message.index));
      else if (message.type === "cg:clear") { clear(); reply({ ok: true }); }
    } catch (err) {
      reply({ error: String(err && err.message ? err.message : err) });
    }
    return false;
  });
})();
