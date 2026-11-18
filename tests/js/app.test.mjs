// Node's built-in test runner: node --test tests/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { segments, severityCounts, summaryLine } = require("../../src/clauseguard/web/static/app.js");

const f = (start, end, severity = "high") => ({ char_start: start, char_end: end, severity });

test("segments cover the whole text in order", () => {
  const text = "aaa BBB ccc DDD eee";
  const parts = segments(text, [f(12, 15), f(4, 7)]);
  assert.equal(parts.map((p) => p.text).join(""), text);
  assert.deepEqual(parts.map((p) => p.finding), [null, 1, null, 0, null]);
});

test("overlapping and invalid findings are skipped", () => {
  const parts = segments("0123456789", [f(2, 6), f(4, 8), f(-1, 3), f(5, 50)]);
  assert.deepEqual(parts.filter((p) => p.finding !== null).map((p) => p.finding), [0]);
  assert.equal(parts.map((p) => p.text).join(""), "0123456789");
});

test("text that looks like HTML stays plain text", () => {
  const text = "<script>alert(1)</script> We may terminate.";
  const parts = segments(text, [f(26, 43)]);
  assert.equal(parts[0].text, "<script>alert(1)</script> ");
});

test("severity counts follow display order", () => {
  assert.deepEqual(severityCounts([f(0, 1, "low"), f(0, 1, "high"), f(0, 1, "low")]),
    [["high", 1], ["low", 2]]);
});

test("summary line", () => {
  const line = summaryLine({ document_type_name: "Contract", n_words: 12345, n_pages: 1,
    findings: [{}, {}] });
  assert.equal(line, "Contract · 12,345 words · 1 page · 2 flagged clauses");
});

const { detectionHint } = require("../../src/clauseguard/web/static/app.js");
const names = { tos: "Terms of Service", privacy: "Privacy Policy", contract: "Contract" };

test("a confident detection pre-selects its type", () => {
  const hint = detectionHint({ document_type: "privacy", confidence: 0.91, confident: true,
    unrelated: false, probabilities: { privacy: 0.91, tos: 0.05 } }, names);
  assert.equal(hint.select, "privacy");
  assert.match(hint.text, /Privacy Policy \(91% sure\)/);
});

test("an unsure or unrelated detection selects nothing", () => {
  const unsure = detectionHint({ document_type: null, confidence: 0.5, confident: false,
    unrelated: false, probabilities: { tos: 0.5, privacy: 0.45, other: 0.05 } }, names);
  assert.equal(unsure.select, null);
  assert.match(unsure.text, /might be a Terms of Service \(50%\)/);
  const unrelated = detectionHint({ document_type: null, confidence: 0.9, confident: false,
    unrelated: true, probabilities: { other: 0.9 } }, names);
  assert.equal(unrelated.select, null);
  assert.match(unrelated.text, /doesn't look like/);
});

const { groupBreakdown, verdictText, reportText, wordCount } = require("../../src/clauseguard/web/static/app.js");

const finding = (severity, group, name = "Arbitration") => ({
  severity, risk_group_name: group, text: "We may terminate.", explanation: "Plain words.",
  labels: [{ name, score: 0.9 }],
});

test("risk groups count only risky flags, largest first", () => {
  const groups = groupBreakdown([finding("high", "Legal risk"), finding("low", "Legal risk"),
    finding("medium", "Personal data"), finding("protective", "Protective")]);
  assert.deepEqual(groups.map(([name, g]) => [name, g.total, g.high]),
    [["Legal risk", 2, 1], ["Personal data", 1, 0]]);
});

test("the verdict reflects the band and the main risk group", () => {
  const high = verdictText({ risk_band: "high", findings: [finding("high", "Legal risk")] });
  assert.match(high, /More aggressive than most/);
  assert.match(high, /1 high-severity clause\./);
  assert.match(high, /legal risk/);
  assert.match(verdictText({ risk_band: "low", findings: [] }), /didn't flag any risky clauses/);
});

test("the copied summary lists the most severe flags first", () => {
  const text = reportText({ document_type_name: "Terms of Service", risk_score: 61.4, risk_band: "medium",
    n_words: 1200, findings: [finding("low", "Legal risk", "Choice of law"), finding("high", "Legal risk")] });
  assert.match(text, /Risk score 61\/100 \(medium\)/);
  assert.ok(text.indexOf("[High] Arbitration") < text.indexOf("[Low] Choice of law"));
});

test("word count", () => {
  assert.equal(wordCount("  one two\nthree  "), 3);
  assert.equal(wordCount("   "), 0);
});
