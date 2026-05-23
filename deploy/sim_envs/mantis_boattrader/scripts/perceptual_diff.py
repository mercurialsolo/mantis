"""Perceptual-diff harness for mantis-boattrader vs real boattrader.com.

Opens both sites in headed Chrome (CF challenges block headless), scrolls
each to the named region, captures matched-rectangle screenshots, and
runs the structural probe (`getBoundingClientRect` / `getComputedStyle`)
on a curated set of elements. Prints a side-by-side delta table and
saves a side-by-side composite image.

Usage:

    .venv/bin/python deploy/sim_envs/mantis_boattrader/scripts/perceptual_diff.py \\
        --sandbox https://8080-<id>.daytonaproxy01.net \\
        --token <preview-token> \\
        --region filter-panel

Regions (each maps to a selector + post-scroll y offset):

    filter-panel  → `.filters-form` / `.search-alerts` on real BT
    srp-search    → `.ai-search-v2__form`
    listing-card  → first `.listing-card`
    bdp-gallery   → `.bdp-gallery` / real BT carousel

Output:
    /tmp/bt-fidelity/<region>/{real.png, sand.png, diff.json}

The diff.json contains the structural delta table — one row per measured
element with real/sand columns and a `delta` field flagging mismatches.

Requires: playwright (`uv pip install playwright && playwright install chromium`).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any


REGIONS: dict[str, dict[str, Any]] = {
    "filter-panel": {
        "real_url": "https://www.boattrader.com/boats/",
        "real_scroll_selector": "[class*='search-alerts']",
        "sand_path": "/boats/",
        "sand_scroll_selector": ".filters-form",
        "rect": (40, 0, 380, 700),
        "probes": [
            # Each probe is (real_selector_or_finder, sand_selector_or_finder, key).
            ("form_card", "[class*='switcher-wrapper']", ".filters-form"),
            ("save_btn", "button.search-alerts-button", ".search-alerts-button"),
            ("switcher", "[class*='switcher-wrapper']", ".zip-toggle"),
            ("use_loc", "div.search-user-location", ".zip-use-location"),
        ],
    },
}

PROBE_JS = r"""
(sel) => {
  const el = typeof sel === 'string' ? document.querySelector(sel) : sel;
  if (!el) return null;
  const r = el.getBoundingClientRect();
  const cs = getComputedStyle(el);
  return {
    w: r.width|0, h: r.height|0, x: r.x|0, y: r.y|0,
    fs: cs.fontSize, fw: cs.fontWeight,
    color: cs.color, bg: cs.backgroundColor,
    border: cs.border, br: cs.borderRadius,
    padding: cs.padding, margin: cs.margin,
    boxShadow: cs.boxShadow.slice(0, 80),
    textDecoration: cs.textDecoration.slice(0, 40),
    textAlign: cs.textAlign,
  };
}
"""


def _diff_rows(real: dict, sand: dict) -> list[dict]:
    rows = []
    if not real or not sand:
        return [{"key": "(missing)", "real": real, "sand": sand}]
    for k in sorted(set(real) | set(sand)):
        rv, sv = real.get(k), sand.get(k)
        if rv != sv:
            rows.append({"key": k, "real": rv, "sand": sv})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", required=True, help="Sandbox base URL")
    ap.add_argument("--token", required=True, help="Preview token")
    ap.add_argument("--region", default="filter-panel", choices=list(REGIONS))
    ap.add_argument("--out", default="/tmp/bt-fidelity")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("error: playwright not installed. Run: uv pip install playwright && playwright install chromium")

    region = REGIONS[args.region]
    out_dir = pathlib.Path(args.out) / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})

        real_page = ctx.new_page()
        real_page.goto(region["real_url"], wait_until="networkidle", timeout=30_000)
        time.sleep(1.5)
        real_page.evaluate(f"""
            const el = document.querySelector({json.dumps(region['real_scroll_selector'])});
            if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 20);
        """)
        time.sleep(0.5)

        sand_page = ctx.new_page()
        sand_url = f"{args.sandbox.rstrip('/')}{region['sand_path']}?daytona_preview_token={args.token}"
        sand_page.goto(sand_url, wait_until="networkidle", timeout=30_000)
        time.sleep(1.0)
        sand_page.evaluate(f"""
            const el = document.querySelector({json.dumps(region['sand_scroll_selector'])});
            if (el) window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 20);
        """)
        time.sleep(0.5)

        x0, y0, x1, y1 = region["rect"]
        clip = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
        real_path = out_dir / "real.png"
        sand_path = out_dir / "sand.png"
        real_page.screenshot(path=str(real_path), clip=clip)
        sand_page.screenshot(path=str(sand_path), clip=clip)

        # Run structural probes side-by-side.
        diffs = {}
        for key, real_sel, sand_sel in region["probes"]:
            real_data = real_page.evaluate(PROBE_JS, real_sel)
            sand_data = sand_page.evaluate(PROBE_JS, sand_sel)
            diffs[key] = {
                "real": real_data,
                "sand": sand_data,
                "delta": _diff_rows(real_data, sand_data),
            }

        (out_dir / "diff.json").write_text(json.dumps(diffs, indent=2))

        # Print summary.
        print(f"region: {args.region}")
        print(f"  real:    {real_path}")
        print(f"  sand:    {sand_path}")
        print(f"  diff:    {out_dir / 'diff.json'}")
        any_mismatch = False
        for key, d in diffs.items():
            print(f"\n  [{key}]")
            for row in d["delta"]:
                print(f"    {row['key']}: real={row['real']!r} sand={row['sand']!r}")
                any_mismatch = True
        if not any_mismatch:
            print("\n  ✓ no structural deltas — fidelity passes")
        else:
            print("\n  ✗ structural deltas above; rerun after CSS edit + _daytona_patch.py")

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
