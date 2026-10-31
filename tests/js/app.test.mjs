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
