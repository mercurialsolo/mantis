/* mantis-shopify admin shell — interaction polish to push toward
   Polaris fidelity: close-on-outside-click + Esc, toast on success
   redirect, bulk action bar when checkboxes selected. */

(function () {
  "use strict";

  // ── Close <details> dropdowns on outside click / Esc ─────────
  document.addEventListener("click", (e) => {
    document.querySelectorAll("details[open]").forEach((d) => {
      if (!d.contains(e.target)) d.removeAttribute("open");
    });
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll("details[open]").forEach((d) =>
        d.removeAttribute("open"),
      );
    }
  });

  // ── Toast notification system ────────────────────────────────
  function showToast(message, variant) {
    const t = document.createElement("div");
    t.className = "adm-toast" + (variant ? " adm-toast-" + variant : "");
    t.setAttribute("role", "status");
    t.textContent = message;
    document.body.appendChild(t);
    setTimeout(() => {
      t.style.opacity = "0";
      setTimeout(() => t.remove(), 240);
    }, 2400);
  }

  // Surface a toast for ?created=order/product/customer/discount or
  // ?fulfilled=1 / ?refunded=1 in the URL (set by POST-redirect-GET).
  const url = new URL(location.href);
  const created = url.searchParams.get("created");
  if (created) {
    const labels = {
      order: "Order created.",
      product: "Product saved.",
      customer: "Customer saved.",
      discount: "Discount code created.",
    };
    showToast(labels[created] || "Saved.", "success");
  }
  if (url.searchParams.get("fulfilled")) showToast("Order fulfilled.", "success");
  if (url.searchParams.get("refunded")) showToast("Refund issued.", "success");
  if (url.searchParams.get("archived")) showToast("Archived.", "success");
  if (url.searchParams.get("published")) showToast("Published.", "success");

  // Expose for ad-hoc use
  window.admToast = showToast;

  // ── Bulk action bar when checkboxes selected ─────────────────
  document.querySelectorAll("table.adm-table").forEach((tbl) => {
    const allCb = tbl.querySelector('thead input[type="checkbox"]');
    const rowCbs = tbl.querySelectorAll('tbody input[type="checkbox"]');
    if (!rowCbs.length) return;

    // Create the bulk bar (one per table, inserted before <table>)
    const bar = document.createElement("div");
    bar.className = "adm-bulkbar";
    bar.setAttribute("data-testid", "adm-bulkbar");
    bar.innerHTML =
      '<span class="adm-bulkbar-count" data-testid="adm-bulkbar-count">0 selected</span>' +
      '<span class="adm-bulkbar-spacer"></span>' +
      '<a href="#tag" class="adm-btn adm-btn-plain">Tag</a>' +
      '<a href="#edit" class="adm-btn adm-btn-plain">Edit</a>' +
      '<a href="#archive" class="adm-btn adm-btn-plain">Archive</a>' +
      '<a href="#delete" class="adm-btn adm-btn-plain adm-btn-critical-text">Delete</a>';
    bar.style.display = "none";
    tbl.parentElement.insertBefore(bar, tbl);

    function refresh() {
      const sel = Array.from(rowCbs).filter((c) => c.checked).length;
      bar.querySelector(".adm-bulkbar-count").textContent = sel + " selected";
      bar.style.display = sel ? "flex" : "none";
      // Sync header checkbox indeterminate state
      if (allCb) {
        allCb.checked = sel > 0 && sel === rowCbs.length;
        allCb.indeterminate = sel > 0 && sel < rowCbs.length;
      }
    }

    rowCbs.forEach((c) => c.addEventListener("change", refresh));
    if (allCb) {
      allCb.addEventListener("change", () => {
        rowCbs.forEach((c) => (c.checked = allCb.checked));
        refresh();
      });
    }
  });
})();
