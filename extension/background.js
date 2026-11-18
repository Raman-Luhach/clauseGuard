/* ClauseGuard background worker: the right-click "Analyse selected text" menu item. */
"use strict";

importScripts("lib/core.js");
const core = globalThis.ClauseGuardCore;
const MENU_ID = "clauseguard-analyse-selection";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: "Analyse selected text with ClauseGuard",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID) return;
  const reportUrl = chrome.runtime.getURL("report.html");
  const text = info.selectionText || "";
  const { apiUrl } = await chrome.storage.sync.get({ apiUrl: core.DEFAULT_API });
  try {
    if (core.wordCount(text) < core.MIN_WORDS) {
      throw new Error(`Select more text: at least ${core.MIN_WORDS} words of the document.`);
    }
    // The menu's selection text has no line breaks, which is fine for sentence-level analysis.
    const result = await core.api.analyze(apiUrl, text, "auto");
    await chrome.storage.session.set({
      lastReport: { result, url: tab && tab.url, title: tab && tab.title, source: "selection", at: Date.now() },
    });
  } catch (err) {
    const message = err.code === "uncertain_document_type" || err.code === "unrelated_document"
      ? `${err.message} Open the ClauseGuard button in the toolbar to choose the type.`
      : err.message;
    await chrome.storage.session.set({ lastError: { message, at: Date.now() } });
  }
  chrome.tabs.create({ url: reportUrl });
});
