// The browser extension's pure helpers (extension/lib/core.js): node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const core = require("../../extension/lib/core.js");

test("normalize lowercases, unifies quotes and collapses whitespace, keeping a map", () => {
  const { text, map } = core.normalize("  We’LL  share\n\tYour “data”. ");
  assert.equal(text, "we'll share your \"data\".");
  assert.equal(map.length, text.length);
  assert.equal(map[0], 2); // "W" in the original
});

test("locateFindings places clauses despite case and whitespace differences", () => {
  const page = "Welcome!\n\nWe may   TERMINATE your account at any time without notice.\nOther text. "
    + "You agree to binding arbitration of all disputes.";
  const findings = [
    { text: "We may terminate your account at any time without notice." },
    { text: "You agree to binding arbitration of all disputes." },
    { text: "This sentence is not on the page at all, anywhere." },
  ];
  const placed = core.locateFindings(page, findings);
  assert.deepEqual(placed.map((p) => p.index), [0, 1]);
  assert.equal(page.slice(placed[0].start, placed[0].end),
    "We may   TERMINATE your account at any time without notice.");
  assert.equal(page.slice(placed[1].start, placed[1].end),
    "You agree to binding arbitration of all disputes.");
});

test("long clauses that differ slightly are found by their first and last words", () => {
  const clause = "Company may share personal information with advertising partners and analytics "
    + "providers for marketing purposes, including targeted advertising across websites and apps "
    + "that you use, unless you opt out in your account settings.";
  const page = `Intro. ${clause.replace("analytics providers", "analytics  service providers")} Next.`;
  const placed = core.locateFindings(page, [{ text: clause }]);
  assert.equal(placed.length, 1);
  assert.ok(page.slice(placed[0].start, placed[0].end).startsWith("Company may share"));
  assert.ok(page.slice(placed[0].start, placed[0].end).endsWith("account settings."));
});

test("chooseText prefers a long selection, then the main content, then the page", () => {
  const words = (n) => Array.from({ length: n }, (_, i) => `w${i}`).join(" ");
  assert.equal(core.chooseText({ selection: words(60), main: words(500), body: words(900) }).source, "selection");
  assert.equal(core.chooseText({ selection: words(5), main: words(500), body: words(900) }).source, "main");
  assert.equal(core.chooseText({ selection: "", main: words(100), body: words(900) }).source, "page");
});

test("detectionText explains the router's answer", () => {
  assert.match(core.detectionText({ confident: true, document_type: "tos", confidence: 0.93 }),
    /Terms of Service \(93% sure\)/);
  assert.match(core.detectionText({ confident: false, unrelated: true }), /doesn't look like/);
  assert.match(core.detectionText({ confident: false, unrelated: false }), /Not sure/);
});

test("post turns server errors into readable errors", async () => {
  const fakeFetch = async () => ({
    ok: false,
    status: 422,
    json: async () => ({ error: { code: "too_short", message: "The document has only 3 words." } }),
  });
  await assert.rejects(core.api.analyze("http://x", "short", "tos", fakeFetch),
    (err) => err.code === "too_short" && /only 3 words/.test(err.message));
  const down = async () => { throw new TypeError("Failed to fetch"); };
  await assert.rejects(core.api.detect("http://x", "text", down), (err) => err.code === "unreachable");
});
