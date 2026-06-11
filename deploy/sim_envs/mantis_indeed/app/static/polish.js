/* mantis-polish.js — universal interaction polish for all envs. */

(function () {
  "use strict";

  // ── Close <details> on outside click + Esc ───────────────────
  document.addEventListener("click", function (e) {
    document.querySelectorAll("details[open]").forEach(function (d) {
      if (!d.contains(e.target)) d.removeAttribute("open");
    });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      document.querySelectorAll("details[open]").forEach(function (d) {
        d.removeAttribute("open");
      });
    }
  });

  // ── Toast helper ────────────────────────────────────────────
  function toast(message, variant) {
    var t = document.createElement("div");
    t.className = "mantis-toast" + (variant ? " mantis-toast-" + variant : "");
    t.setAttribute("role", "status");
    t.setAttribute("aria-live", "polite");
    t.textContent = message;
    document.body.appendChild(t);
    setTimeout(function () {
      t.style.opacity = "0";
      setTimeout(function () { t.remove(); }, 240);
    }, 2400);
  }
  window.mantisToast = toast;

  // ── URL-driven toasts (POST-redirect-GET) ───────────────────
  var url = new URL(location.href);
  var created = url.searchParams.get("created");
  if (created) {
    var labels = {
      order: "Order created.", product: "Product saved.",
      customer: "Customer saved.", discount: "Discount created.",
      lead: "Lead submitted.", ticket: "Ticket submitted.",
      campaign: "Campaign created.", post: "Post published.",
    };
    toast(labels[created] || "Saved.", "success");
  }
  ["fulfilled", "refunded", "archived", "published", "saved",
   "deleted", "applied", "sent"].forEach(function (k) {
    if (url.searchParams.get(k)) {
      var t = k.charAt(0).toUpperCase() + k.slice(1) + ".";
      toast(t, k === "deleted" ? "critical" : "success");
    }
  });

  // ── Keyboard shortcuts ──────────────────────────────────────
  var chord = null;
  var chordTimer = null;
  document.addEventListener("keydown", function (e) {
    if (e.target.matches("input, textarea, select, [contenteditable]")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    // `/` focuses the search bar
    if (e.key === "/") {
      var search = document.querySelector(
        'input[type=search]:not([disabled]), ' +
        'input[name=q]:not([disabled]), ' +
        '[data-testid$="search"]:not([disabled])'
      );
      if (search) {
        e.preventDefault();
        search.focus();
        search.select && search.select();
      }
      return;
    }

    // g + X chord navigation
    if (e.key === "g" && !chord) {
      chord = "g";
      clearTimeout(chordTimer);
      chordTimer = setTimeout(function () { chord = null; }, 1000);
      return;
    }
    if (chord === "g") {
      chord = null;
      clearTimeout(chordTimer);
      var m = location.pathname.match(/^(\/store\/[^/]+\/admin)/);
      var adminMap = {
        h: "", o: "/orders", p: "/products", c: "/customers",
        a: "/analytics", m: "/marketing", d: "/discounts", s: "/settings",
      };
      if (m && adminMap[e.key] !== undefined) {
        e.preventDefault();
        location.href = m[1] + adminMap[e.key];
        return;
      }
      var rootMap = {
        h: "/", j: "/jobs", o: "/orders", f: "/feed",
        m: "/messaging", n: "/mynetwork", e: "/explore",
      };
      if (rootMap[e.key]) {
        e.preventDefault();
        location.href = rootMap[e.key];
      }
    }
  });

  // ── Show chord-hint helper on `?` ───────────────────────────
  document.addEventListener("keydown", function (e) {
    if (e.target.matches("input, textarea, select, [contenteditable]")) return;
    if (e.key === "?" && e.shiftKey) {
      e.preventDefault();
      toast(
        "g+h home, g+o orders, g+p products, g+c customers, / search",
        "success",
      );
    }
  });

  // ── Auto-emit ARIA role on tabs lacking it ──────────────────
  document.querySelectorAll(
    ".sp-tab, .adm-tab, .tab, [data-testid$=-tab], [data-testid^=tab-]",
  ).forEach(function (t) {
    if (!t.getAttribute("role")) t.setAttribute("role", "tab");
    if (t.classList.contains("is-active")) {
      t.setAttribute("aria-current", "page");
    }
  });
})();
