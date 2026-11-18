/* Shared behaviour for every page: theme, navigation, model status, scroll effects, toasts. */
"use strict";

(function () {
  const root = document.documentElement;
  const $ = (id) => document.getElementById(id);

  /* ---- theme: follows the system until the user picks one ---- */
  function currentTheme() {
    if (root.dataset.theme) return root.dataset.theme;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("cg-theme", next); } catch (e) { /* private mode */ }
    });
  }

  /* ---- navigation ---- */
  const nav = $("nav");
  const onScroll = () => nav && nav.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const path = location.pathname.replace(/\/+$/, "") || "/";
  document.querySelectorAll(".nav-links a").forEach((a) => {
    if (a.getAttribute("href") === path) a.setAttribute("aria-current", "page");
  });

  const menu = $("menu-btn");
  if (menu && nav) {
    menu.addEventListener("click", () => {
      const open = !nav.classList.contains("open");
      nav.classList.toggle("open", open);
      menu.setAttribute("aria-expanded", String(open));
    });
  }

  /* ---- model status in the footer ---- */
  const dot = $("health-dot");
  if (dot) {
    fetch("/health").then((r) => r.json()).then((h) => {
      const mods = Object.values(h.modules || {});
      const ready = mods.filter((m) => m.ready).length;
      dot.textContent = ready === mods.length && ready
        ? `All ${ready} models ready${h.router ? " · type detection on" : ""}`
        : `${ready} of ${mods.length} models ready`;
      dot.classList.add(ready ? "ok" : "bad");
      window.dispatchEvent(new CustomEvent("cg:health", { detail: h }));
    }).catch(() => { dot.textContent = "Server not reachable"; dot.classList.add("bad"); });
  }

  /* ---- reveal on scroll ---- */
  const reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && reveals.length) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }, { rootMargin: "0px 0px -8% 0px" });
    reveals.forEach((n) => io.observe(n));
  } else {
    reveals.forEach((n) => n.classList.add("in"));
  }

  /* ---- table of contents: highlight the section in view ---- */
  const tocLinks = [...document.querySelectorAll(".toc a[href^='#']")];
  if (tocLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(tocLinks.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        tocLinks.forEach((a) => a.classList.remove("on"));
        const link = byId.get(e.target.id);
        if (link) link.classList.add("on");
      }
    }, { rootMargin: "-20% 0px -70% 0px" });
    byId.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  }

  /* ---- toast ---- */
  let toastTimer = null;
  window.cgToast = function (message) {
    const t = $("toast");
    if (!t) return;
    t.textContent = message;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
  };
})();
