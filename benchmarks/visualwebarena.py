"""VisualWebArena benchmark — Playwright-driven Chromium, no QEMU.

Phase 2 scaffold for EXP-26. This file runs our Gemma 4 agent loop against
the VWA task suite using a direct Playwright+Chromium pipeline instead of
the OSWorld QEMU Ubuntu VM. Because VWA tasks are all browser-only, we
skip the entire desktop-OS stack and talk to a headless Chromium directly
from the Modal A100 container.

Architecture:

    Modal A100 container
    ├── llama.cpp server (Gemma 4 26B Q4_K_M, port 8080)
    ├── Playwright + Chromium (headless, 1280x720)
    ├── VWA env adapter (translates our click/type/key actions to
    │                    Playwright page operations)
    └── Agent loop (same one as modal_osworld_direct.py, modulo the
                    screenshot/action layer)

The 5 VWA sidecar Docker services (Classifieds, Shopping, Reddit,
Wikipedia, Homepage) run OUTSIDE Modal on a persistent host, because
Magento-Shopping state persists across tasks and spinning them up
per-run would be too slow. The Modal container connects to them via
their public hostnames (passed via env vars at launch).

Status: SCAFFOLD — this file runs the single-task smoke path (task 0).
Full 910-task evaluation, adapter tuning, and curriculum integration
come in follow-up commits.

Run (once sidecars are up):
    uv run modal run --detach benchmarks/visualwebarena.py --tasks 1
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

# Reuse the base llama-server startup helpers but NOT the QEMU/OSWorld stack.
from modal_osworld_direct import (
    GEMMA4_MODEL,
    GGUF_CONFIGS,
    download_model,
    start_llama_server,
    vol,
)


# ── Image: Playwright + Chromium, no QEMU/VNC ────────────────────────────────
vwa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential", "curl", "git", "wget",
        # Chromium runtime deps (pulled in by playwright install --with-deps)
    )
    .pip_install(
        "requests", "pillow", "numpy",
        "playwright", "beautifulsoup4", "lxml",
        "openai",  # for OpenAI-compatible llama.cpp client
    )
    .run_commands(
        # Install llama.cpp (prebuilt binary) — same approach as OSWorld image
        "curl -L https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-b4000-bin-ubuntu-x64.zip -o /tmp/llama.zip || true",
        # Playwright + Chromium runtime
        "playwright install --with-deps chromium",
    )
    .add_local_python_source("mantis_agent")
    .add_local_python_source("modal_osworld_direct")
)


# ── VWA sidecar hostnames (override via Modal env vars at deploy time) ───────
VWA_ENDPOINTS = {
    "classifieds": os.environ.get("VWA_CLASSIFIEDS_URL", "http://vwa-classifieds:9980"),
    "shopping":    os.environ.get("VWA_SHOPPING_URL",    "http://vwa-shopping:7770"),
    "reddit":      os.environ.get("VWA_REDDIT_URL",      "http://vwa-reddit:9999"),
    "wikipedia":   os.environ.get("VWA_WIKIPEDIA_URL",   "http://vwa-wikipedia:8888"),
    "homepage":    os.environ.get("VWA_HOMEPAGE_URL",    "http://vwa-homepage:4399"),
}


# ── Pixel-to-Playwright action adapter ───────────────────────────────────────
#
# Our agent emits click(x, y) / type_text(s) / key(combo) / wait(n) / run_command(...)
# in the helpers prelude. VWA's canonical action format is a Playwright DSL
# (``click [id]``, ``type [id] [text]``, ...) but it also supports pixel
# clicks via ``mouse_click_action(x, y)``. Since our agent already produces
# pixel coordinates, we translate 1:1 to Playwright page.mouse.* calls and
# page.keyboard.* calls. No need for Set-of-Mark element IDs.

class PlaywrightAdapter:
    """Translate our agent's action vocabulary to Playwright operations.

    This is the bridge layer that makes the OSWorld-style agent loop work
    against a real browser without changing the agent code. The adapter
    owns the Playwright Page object, provides a screenshot method, and
    exposes ``execute_action(code_str)`` which the agent loop calls on
    every step.
    """

    def __init__(self, start_url: str, viewport_width: int = 1280, viewport_height: int = 720):
        from playwright.sync_api import sync_playwright
        self._pw_ctx = sync_playwright().start()
        self._browser = self._pw_ctx.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
        )
        self.page = self._context.new_page()
        self.start_url = start_url
        if start_url:
            self.page.goto(start_url, wait_until="domcontentloaded")

    # ── Observation ───────────────────────────────────────────────────────
    def screenshot(self) -> bytes:
        """Return a PNG screenshot of the current page as bytes."""
        return self.page.screenshot(type="png")

    # ── Action primitives — mirror mantis_agent.tools.helpers names ──────
    def click(self, x: int, y: int, button: str = "left") -> None:
        self.page.mouse.click(x, y, button=button)

    def double_click(self, x: int, y: int) -> None:
        self.page.mouse.dblclick(x, y)

    def type_text(self, text: str) -> None:
        self.page.keyboard.type(text)

    def key(self, combo: str) -> None:
        """Accept 'ctrl+l', 'Return', 'Escape', etc. — normalize to Playwright."""
        combo = combo.strip()
        # Playwright uses + for combos ("Control+L"), and capitalizes
        # modifier names. Normalize.
        parts = [p.strip() for p in combo.split("+")]
        normalized: list[str] = []
        for p in parts:
            low = p.lower()
            if low in ("ctrl", "control"):
                normalized.append("Control")
            elif low in ("alt", "option"):
                normalized.append("Alt")
            elif low == "shift":
                normalized.append("Shift")
            elif low in ("meta", "cmd", "command", "win", "super"):
                normalized.append("Meta")
            elif low in ("return", "enter"):
                normalized.append("Enter")
            elif low == "escape" or low == "esc":
                normalized.append("Escape")
            elif low in ("tab", "backspace", "delete", "home", "end",
                         "pageup", "pagedown", "up", "down", "left", "right"):
                normalized.append(low.capitalize())
            else:
                # Single character — keep as-is
                normalized.append(p)
        self.page.keyboard.press("+".join(normalized))

    def scroll(self, x: int, y: int, clicks: int = -3) -> None:
        # Playwright expects delta in pixels; treat each "click" as 100 px.
        self.page.mouse.move(x, y)
        self.page.mouse.wheel(0, clicks * -100)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def run_command(self, cmd: str) -> None:
        """No-op in VWA — we do not have a shell inside the browser.

        The agent curriculum should steer it away from run_command on
        chrome/VWA tasks; if it still tries, we print a polite refusal
        so the captured-output feedback can redirect.
        """
        print(f"[VWA] run_command ignored (no shell in browser context): {cmd[:80]}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    def close(self) -> None:
        try:
            self._context.close()
        except Exception:
            pass
        try:
            self._browser.close()
        except Exception:
            pass
        try:
            self._pw_ctx.stop()
        except Exception:
            pass


# ── VWA task loading ─────────────────────────────────────────────────────────

def load_vwa_task_configs(config_dir: Optional[Path] = None) -> list[dict]:
    """Load VWA task config JSONs. Expects config_files/vwa/ layout.

    Phase 2 scaffold: returns an empty list until we mount the VWA repo
    into the Modal image or fetch configs from the volume. Callers should
    handle the empty case gracefully.
    """
    if config_dir is None:
        config_dir = Path("/data/vwa/config_files/vwa")
    if not config_dir.exists():
        return []
    configs: list[dict] = []
    for sub in ("test_classifieds", "test_shopping", "test_reddit"):
        sub_dir = config_dir / sub
        if not sub_dir.exists():
            continue
        for f in sorted(sub_dir.glob("*.json")):
            try:
                configs.append(json.loads(f.read_text()))
            except Exception:
                continue
    return configs


# ── Modal app + function ─────────────────────────────────────────────────────

vwa_app = modal.App("gemma4-vwa")


@vwa_app.function(
    gpu="A100-80GB",
    image=vwa_image,
    volumes={"/data": vol},
    timeout=86400,
    memory=32768,
    cpu=8,
)
def run_vwa(max_tasks: int = 1, max_steps: int = 30):
    """Run the Gemma 4 agent against VisualWebArena tasks.

    Phase 2 scaffold: loads task configs (if present), drives Playwright
    for a single task, proves the adapter/loop integration works
    end-to-end. Scoring, retries, and full distillation loop are
    deferred to the next commit.
    """
    # 1. Start llama-server (Gemma 4) inside the container
    cfg = GGUF_CONFIGS[GEMMA4_MODEL]
    model_path = download_model("/data")
    llama_proc = start_llama_server(model_path)
    print(f"llama-server started for VWA run")

    # 2. Load tasks (will be empty until we populate /data/vwa)
    tasks = load_vwa_task_configs()
    if not tasks:
        print("WARNING: no VWA task configs found under /data/vwa — smoke-testing adapter only")
        tasks = [
            {
                "task_id": "smoke-test-0",
                "intent": "Open the homepage and report the page title.",
                "start_url": VWA_ENDPOINTS["homepage"],
                "sites": ["homepage"],
            }
        ]

    # 3. Run the agent loop for up to max_tasks tasks
    results: list[dict] = []
    for task in tasks[:max_tasks]:
        print(f"\n=== VWA task {task.get('task_id')} ===")
        print(f"  intent: {task.get('intent', '')[:120]}")
        print(f"  start_url: {task.get('start_url')}")

        adapter = None
        try:
            adapter = PlaywrightAdapter(start_url=task.get("start_url", "about:blank"))

            # SMOKE: take one screenshot and report size, proving the adapter works.
            # Real agent loop integration comes in the follow-up commit.
            png = adapter.screenshot()
            title = adapter.page.title()
            print(f"  screenshot: {len(png)} bytes")
            print(f"  page title: {title!r}")

            results.append({
                "task_id": task.get("task_id"),
                "status": "smoke_ok",
                "screenshot_bytes": len(png),
                "page_title": title,
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({
                "task_id": task.get("task_id"),
                "status": "error",
                "error": str(e)[:200],
            })
        finally:
            if adapter:
                adapter.close()

    # 4. Save results
    os.makedirs("/data/results", exist_ok=True)
    out_path = f"/data/results/vwa_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_id": time.strftime("%Y%m%d_%H%M%S"),
                "benchmark": "visualwebarena",
                "tasks_run": len(results),
                "task_results": results,
            },
            f,
            indent=2,
        )
    vol.commit()

    llama_proc.terminate()
    return {"tasks_run": len(results), "results": results, "output": out_path}


@vwa_app.local_entrypoint()
def main(tasks: int = 1, steps: int = 30):
    """Phase 2 smoke entrypoint. tasks=1 runs a single adapter smoke test."""
    print("Mantis — VisualWebArena Benchmark (Phase 2 scaffold)")
    print(f"  Tasks: {tasks}")
    print(f"  Steps: {steps}")
    print()
    print("NOTE: Phase 2 scaffold only drives the Playwright adapter.")
    print("      Full agent integration comes in the follow-up commit.")
    print()
    result = run_vwa.remote(max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
