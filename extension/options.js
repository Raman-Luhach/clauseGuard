/* ClauseGuard settings: which server to use. */
"use strict";

const core = globalThis.ClauseGuardCore;
const $ = (id) => document.getElementById(id);

function setStatus(message, isError = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", isError);
}

function cleanUrl(value) {
  const url = new URL(value.trim() || core.DEFAULT_API);
  if (!/^https?:$/.test(url.protocol)) throw new Error("Use an http:// or https:// address.");
  return url.origin + url.pathname.replace(/\/+$/, "");
}

async function testConnection() {
  setStatus("Checking…");
  try {
    const health = await core.api.health(cleanUrl($("api").value));
    const ready = Object.entries(health.modules || {}).filter(([, m]) => m.ready).map(([, m]) => m.name);
    setStatus(ready.length
      ? `Connected. Ready: ${ready.join(", ")}${health.router ? "; automatic type detection on" : ""}.`
      : "Connected, but no models are ready on that server.", !ready.length);
  } catch (err) {
    setStatus(`Can't reach a ClauseGuard server there (${err.message}).`, true);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const { apiUrl } = await chrome.storage.sync.get({ apiUrl: core.DEFAULT_API });
  $("api").value = apiUrl;
  $("save").addEventListener("click", async () => {
    try {
      const url = cleanUrl($("api").value);
      await chrome.storage.sync.set({ apiUrl: url });
      $("api").value = url;
      setStatus("Saved.");
    } catch (err) {
      setStatus(err.message, true);
    }
  });
  $("test").addEventListener("click", testConnection);
});
