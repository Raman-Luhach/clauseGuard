/* Content pages: the category browser (How it works) and live model status (Models). */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const SEV = { high: "High", medium: "Medium", low: "Low", protective: "Protective", info: "Info" };
  const ORDER = ["high", "medium", "low", "protective", "info"];

  /* ---- categories ---- */
  const tabs = $("cat-tabs");
  const cats = $("cats");
  if (tabs && cats) {
    fetch("/taxonomy").then((r) => r.json()).then((tax) => {
      const modules = Object.entries(tax.modules);
      function show(id) {
        tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.id === id)));
        const labels = [...tax.modules[id].labels]
          .sort((a, b) => ORDER.indexOf(a.severity) - ORDER.indexOf(b.severity) || a.name.localeCompare(b.name));
        cats.replaceChildren(...labels.map((l) => {
          const card = el("article", "cat");
          const head = el("header");
          head.append(el("strong", null, l.name), el("span", `badge sev-${l.severity}`, SEV[l.severity] || l.severity));
          card.append(head, el("p", null, l.definition), el("p", "group", l.risk_group_name));
          return card;
        }));
      }
      tabs.replaceChildren(...modules.map(([id, m]) => {
        const b = el("button", null, `${m.name} · ${m.labels.length}`);
        b.type = "button";
        b.setAttribute("role", "tab");
        b.dataset.id = id;
        b.addEventListener("click", () => show(id));
        return b;
      }));
      if (modules.length) show(modules[0][0]);
    }).catch(() => { cats.replaceChildren(el("p", "muted", "Could not load the categories.")); });
  }

  /* ---- live model status ---- */
  window.addEventListener("cg:health", (e) => {
    const mods = e.detail.modules || {};
    document.querySelectorAll(".metric-card[data-module]").forEach((card) => {
      const m = mods[card.dataset.module];
      const dot = card.querySelector("[data-status]");
      if (!dot) return;
      dot.classList.remove("ok", "bad");
      if (m && m.ready) {
        dot.textContent = "Live";
        dot.classList.add("ok");
        dot.title = `Served run: ${m.run}`;
        const name = card.querySelector("[data-model]");
        if (name && m.model) name.textContent = `${m.model} · ${m.unit} level`;
      } else {
        dot.textContent = "Offline";
        dot.classList.add("bad");
      }
    });
  });
})();
