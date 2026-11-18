/* ClauseGuard extension: pure helpers shared by the popup, the report page, the background
 * worker and the content script. No DOM access here, so they can be tested outside a browser
 * (tests/js/extension.test.mjs). Loaded as a classic script: it defines globalThis.ClauseGuardCore.
 */
"use strict";

(function (root) {
  const DEFAULT_API = "http://127.0.0.1:8000";
  const MIN_WORDS = 50; // the server rejects shorter documents
  const SEVERITY_ORDER = ["high", "medium", "low", "protective", "info"];
  const SEVERITY_LABEL = {
    high: "High", medium: "Medium", low: "Low", protective: "Protective", info: "Info",
  };
  const TYPE_NAMES = { tos: "Terms of Service", privacy: "Privacy Policy", contract: "Contract" };

  /* ---------------------------------------------------------------- text matching */

  // Characters the server's cleaning also normalises: curly quotes, dashes, odd spaces.
  const CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
  };

  function normChar(c) {
    if (c in CHAR_MAP) return CHAR_MAP[c];
    return /\s/.test(c) ? " " : c.toLowerCase();
  }

  /** Normalise `text` for matching: lowercase, unified quotes/dashes, single spaces.
   *  Returns {text, map} where map[i] is the index in the original of normalised char i. */
  function normalize(text) {
    let out = "";
    const map = [];
    let lastSpace = true; // drop leading whitespace
    for (let i = 0; i < text.length; i++) {
      const c = normChar(text[i]);
      if (c === "") continue;
      if (",.;:!?)]".includes(c) && out.endsWith(" ")) { // "word , next" -> "word, next"
        out = out.slice(0, -1);
        map.pop();
      }
      if (c === " ") {
        if (lastSpace) continue;
        lastSpace = true;
      } else {
        lastSpace = false;
      }
      out += c;
      map.push(i);
    }
    if (out.endsWith(" ")) { out = out.slice(0, -1); map.pop(); }
    return { text: out, map };
  }

  /** Where `needle` occurs in the normalised `haystack` ({text}), searching from `from`
   *  first and then from the start. Long clauses that don't match whole (the page text may
   *  differ a little from the server's cleaned text) are found by their first and last
   *  60 characters. Returns {start, end} in normalised coordinates, or null. */
  function findClause(haystack, needle, from = 0) {
    const n = normalize(needle).text;
    if (n.length < 12) return null; // too short to place reliably
    const find = (s, pos) => {
      const i = haystack.text.indexOf(s, pos);
      return i >= 0 ? i : (pos > 0 ? haystack.text.indexOf(s) : -1);
    };
    const whole = find(n, from);
    if (whole >= 0) return { start: whole, end: whole + n.length };
    if (n.length < 140) return null;
    const head = n.slice(0, 60);
    const tail = n.slice(-60);
    const start = find(head, from);
    if (start < 0) return null;
    const tailAt = haystack.text.indexOf(tail, start + head.length);
    // accept only if the span has about the expected length
    if (tailAt < 0 || tailAt + tail.length - start > n.length * 1.3) {
      return { start, end: start + head.length };
    }
    return { start, end: tailAt + tail.length };
  }

  /** Locate every finding in the page text. Returns [{finding, start, end}] in original-text
   *  coordinates of `pageText`, skipping findings that can't be placed or that overlap. */
  function locateFindings(pageText, findings) {
    const hay = normalize(pageText);
    const placed = [];
    let cursor = 0;
    findings.forEach((f, index) => {
      const hit = findClause(hay, f.text, cursor);
      if (!hit) return;
      cursor = hit.end;
      placed.push({ index, start: hay.map[hit.start], end: hay.map[hit.end - 1] + 1 });
    });
    placed.sort((a, b) => a.start - b.start);
    const out = [];
    for (const p of placed) {
      if (out.length && p.start < out[out.length - 1].end) continue;
      out.push(p);
    }
    return out;
  }

  /* ---------------------------------------------------------------- page text */

  function wordCount(text) {
    const t = (text || "").trim();
    return t ? t.split(/\s+/).length : 0;
  }

  /** Which text to analyse: the selection when it is long enough, else the page's main
   *  content when it holds most of the page, else the whole page. */
  function chooseText({ selection, main, body }) {
    if (wordCount(selection) >= MIN_WORDS) return { text: selection, source: "selection" };
    const bodyWords = wordCount(body);
    if (wordCount(main) >= Math.max(MIN_WORDS, 0.5 * bodyWords)) return { text: main, source: "main" };
    return { text: body || "", source: "page" };
  }

  /* ---------------------------------------------------------------- results */

  function severityCounts(findings) {
    const counts = {};
    for (const f of findings) counts[f.severity] = (counts[f.severity] || 0) + 1;
    return SEVERITY_ORDER.filter((s) => counts[s]).map((s) => [s, counts[s]]);
  }

  /** Findings sorted for display: most severe first, then document order. */
  function sortedFindings(findings) {
    return findings.map((f, i) => ({ f, i })).sort((a, b) =>
      SEVERITY_ORDER.indexOf(a.f.severity) - SEVERITY_ORDER.indexOf(b.f.severity) || a.i - b.i);
  }

  function detectionText(d) {
    if (!d) return "";
    if (d.unrelated) return "This page doesn't look like terms of service, a privacy policy or a contract.";
    if (!d.confident || !TYPE_NAMES[d.document_type]) return "Not sure what kind of document this is. Please choose.";
    return `Looks like a ${TYPE_NAMES[d.document_type]} (${Math.round(d.confidence * 100)}% sure).`;
  }

  /* ---------------------------------------------------------------- API */

  function apiUrl(base, path) {
    return `${(base || DEFAULT_API).replace(/\/+$/, "")}${path}`;
  }

  /** POST a form to the ClauseGuard API. Resolves to the JSON body; rejects with an Error
   *  whose message is fit to show the user (and .code / .details from the server). */
  async function post(base, path, fields, fetchImpl = root.fetch) {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null) form.append(k, v);
    let response;
    try {
      response = await fetchImpl(apiUrl(base, path), { method: "POST", body: form });
    } catch (err) {
      const e = new Error(`Can't reach the ClauseGuard server at ${base || DEFAULT_API}. `
        + "Check the server address in the extension's options.");
      e.code = "unreachable";
      throw e;
    }
    let body = null;
    try { body = await response.json(); } catch (err) { body = null; }
    if (!response.ok) {
      const info = (body && body.error) || {};
      const e = new Error(info.message || `The server answered ${response.status}.`);
      e.code = info.code || `http_${response.status}`;
      e.details = info;
      throw e;
    }
    return body;
  }

  const api = {
    detect: (base, text, fetchImpl) => post(base, "/detect", { text }, fetchImpl),
    analyze: (base, text, documentType, fetchImpl) =>
      post(base, "/analyze", { text, document_type: documentType || "auto" }, fetchImpl),
    async health(base, fetchImpl = root.fetch) {
      const response = await fetchImpl(apiUrl(base, "/health"));
      return response.json();
    },
  };

  const core = {
    DEFAULT_API, MIN_WORDS, SEVERITY_ORDER, SEVERITY_LABEL, TYPE_NAMES,
    normalize, findClause, locateFindings, wordCount, chooseText,
    severityCounts, sortedFindings, detectionText, apiUrl, post, api,
  };
  root.ClauseGuardCore = core;
  if (typeof module !== "undefined" && module.exports) module.exports = core;
})(typeof globalThis !== "undefined" ? globalThis : this);
