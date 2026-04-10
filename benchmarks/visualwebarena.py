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


# ── Agent loop primitives ────────────────────────────────────────────────────
#
# These run inside the Modal container. They bypass OSWorld's PromptAgent
# entirely and talk directly to the llama-server OpenAI-compatible endpoint.
# Same llama.cpp server is started via ``start_llama_server`` from the shared
# module, so we get the exact same Gemma 4 26B vision model as the OS run.

VWA_SYSTEM_PROMPT = """\
You are a computer-use agent controlling a headless web browser. You see a screenshot of the current page each step and output Python code to perform ONE action.

The viewport is 1280x720 pixels. Coordinates must match what you see in the screenshot.

# Available helpers (already imported — just call them)

- `click(x, y)` — left click at (x, y)
- `click(x, y, button="right")` — right click
- `double_click(x, y)` — double click
- `type_text(text)` — type text into the focused field. Handles all special characters correctly.
- `key(combo)` — press a key or combo. Examples: `key("Return")`, `key("ctrl+l")` (address bar), `key("Tab")`, `key("Escape")`, `key("Page_Down")`.
- `wait(seconds=2)` — pause for N seconds (page loads, AJAX).
- `scroll(x, y, clicks=-3)` — scroll at (x, y). Negative clicks = scroll down.

# Important — you are in a browser, not a desktop OS

- There is NO shell, NO terminal, NO sudo. Do not call `run_command`.
- All task progress happens through clicks, keystrokes, and page navigation.
- After clicking a link, submitting a form, or pressing Enter in the address bar, **always `wait(2)`** before the next action — pages load asynchronously and clicking stale UI wastes steps.

# Finishing a task

When you have the answer or the task is complete, output EXACTLY this and nothing else:

```
DONE: <your final answer or confirmation>
```

When the task is impossible from this page, output:

```
FAIL
```

When you need to pause and let the page settle, output:

```
WAIT
```

# Approach

On your FIRST response, write a brief numbered plan, then execute the first action.

First reflect on what you see in the screenshot, then output your code.\
"""


def _llama_infer(
    screenshot_png: bytes,
    instruction: str,
    hint_text: str,
    trajectory: list[dict],
    model_file: str,
    port: int = 8080,
    timeout: int = 120,
) -> str:
    """POST screenshot + instruction to llama-server and return the model's text.

    Uses the OpenAI-compatible chat completions endpoint that llama.cpp
    exposes. The image is base64-encoded as a data URL.
    """
    import base64
    import requests as _req

    img_b64 = base64.b64encode(screenshot_png).decode("ascii")

    # Build a compact user message: instruction, hint (curriculum), last few
    # actions as history, then the current screenshot.
    user_parts: list[dict] = [
        {"type": "text", "text": f"TASK: {instruction}"},
    ]
    if hint_text:
        user_parts.append({"type": "text", "text": hint_text})
    if trajectory:
        history_text = "\n".join(
            f"Step {i + 1}: {t['action'][:160]}"
            + (f"\n  feedback: {t['feedback'][:160]}" if t.get("feedback") else "")
            for i, t in enumerate(trajectory[-5:])
        )
        user_parts.append({"type": "text", "text": f"PREVIOUS STEPS:\n{history_text}"})
    user_parts.append(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    )

    messages = [
        {"role": "system", "content": VWA_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]

    resp = _req.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": model_file,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_code(response: str) -> str:
    """Pull the first python code block out of the model's response."""
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if no fenced block, the whole response might be code
    return response.strip()


def _parse_terminal(code: str) -> Optional[str]:
    """Return 'DONE', 'WAIT', 'FAIL', or None if the code block is a normal action."""
    stripped = code.strip()
    first_line = stripped.split("\n", 1)[0].strip()
    if first_line.startswith("DONE"):
        return "DONE"
    if first_line == "FAIL":
        return "FAIL"
    if first_line == "WAIT":
        return "WAIT"
    return None


def _execute_action(code: str, adapter: "PlaywrightAdapter") -> str:
    """Exec model-generated Python code with adapter-bound helpers.

    Returns captured stdout (feedback for the next step). Exceptions are
    converted to string feedback so the loop continues.
    """
    import contextlib
    import io

    ns = {
        "click": adapter.click,
        "double_click": adapter.double_click,
        "type_text": adapter.type_text,
        "key": adapter.key,
        "scroll": adapter.scroll,
        "wait": adapter.wait,
        "run_command": adapter.run_command,
    }
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception as e:
        buf.write(f"ERROR: {type(e).__name__}: {e}\n")
    return buf.getvalue().strip()


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

    Phase 2: full agent loop (screenshot → llama-server → parse → execute
    via PlaywrightAdapter → feedback → next step). Works end-to-end on any
    public URL as a fallback when the 5 VWA sidecars are not yet deployed,
    so the pipeline can be smoke-tested without the Docker stack.

    Scoring (string_match / url_match / page_image_query) is a follow-up
    — this commit just records trajectories so we can inspect whether the
    model actually navigates sensibly.
    """
    # 1. Start llama-server (Gemma 4) inside the container
    cfg = GGUF_CONFIGS[GEMMA4_MODEL]
    model_file = cfg["model_file"]
    model_path = download_model("/data")
    llama_proc = start_llama_server(model_path)
    print(f"llama-server started for VWA run (model={model_file})")

    # Lazy import of curriculum so we pick up relevant chrome techniques
    try:
        from mantis_agent.curriculum import select_techniques
    except Exception as _e:
        print(f"  curriculum load failed (non-fatal): {_e}")
        select_techniques = None  # type: ignore

    # 2. Load tasks. Fallback: use a public URL so the whole pipeline can
    # smoke-test without the 5 VWA sidecars.
    tasks = load_vwa_task_configs()
    if not tasks:
        print("NOTE: no VWA task configs under /data/vwa — using a public-URL")
        print("      fallback so the agent loop can smoke-test end-to-end.")
        tasks = [
            {
                "task_id": "smoke-example-com",
                "intent": (
                    "You are on example.com. Read the page and tell me what the "
                    "main heading says and what the page is for. Then output DONE "
                    "with your answer."
                ),
                "start_url": "https://example.com",
                "sites": ["public"],
            }
        ]

    # 3. Run the agent loop for each task
    results: list[dict] = []
    for task in tasks[:max_tasks]:
        task_id = task.get("task_id", "?")
        instruction = task.get("intent", "")
        start_url = task.get("start_url", "about:blank")
        print(f"\n=== VWA task {task_id} ===")
        print(f"  intent:    {instruction[:140]}")
        print(f"  start_url: {start_url}")

        # Derive curriculum hint (chrome techniques for browser tasks)
        hint_text = ""
        if select_techniques is not None:
            try:
                hint_text = select_techniques(
                    instruction, hint_text="", domain="chrome", max_topics=3
                )
            except Exception as _e:
                print(f"  curriculum injection failed: {_e}")
        if hint_text:
            print(f"  curriculum: {len(hint_text)} chars injected")

        adapter: Optional[PlaywrightAdapter] = None
        trajectory: list[dict] = []
        final_status = "unknown"
        final_answer: Optional[str] = None
        t_start = time.time()

        try:
            adapter = PlaywrightAdapter(start_url=start_url)
            # Give the page a moment to finish rendering before the first shot.
            adapter.wait(2)

            for step in range(max_steps):
                # 3a. Observe
                try:
                    png = adapter.screenshot()
                except Exception as _e:
                    print(f"  step {step + 1}: screenshot failed: {_e}")
                    break

                # 3b. Infer
                try:
                    response = _llama_infer(
                        screenshot_png=png,
                        instruction=instruction,
                        hint_text=hint_text,
                        trajectory=trajectory,
                        model_file=model_file,
                    )
                except Exception as _e:
                    print(f"  step {step + 1}: inference failed: {_e}")
                    break

                code = _extract_code(response)
                terminal = _parse_terminal(code)

                # 3c. Handle terminal states
                if terminal == "DONE":
                    final_status = "done"
                    # Extract the answer after "DONE:"
                    first_line = code.strip().split("\n", 1)[0]
                    if ":" in first_line:
                        final_answer = first_line.split(":", 1)[1].strip()
                    print(f"  step {step + 1}: DONE — {final_answer or '(no answer)'}")
                    trajectory.append({"action": code, "feedback": "", "terminal": "DONE"})
                    break
                if terminal == "FAIL":
                    final_status = "failed_by_agent"
                    print(f"  step {step + 1}: FAIL (agent gave up)")
                    trajectory.append({"action": code, "feedback": "", "terminal": "FAIL"})
                    break
                if terminal == "WAIT":
                    adapter.wait(2)
                    trajectory.append({"action": "WAIT", "feedback": "(waited 2s)"})
                    continue

                # 3d. Execute the action
                feedback = _execute_action(code, adapter)
                action_brief = code.split("\n", 1)[0][:140]
                fb_brief = feedback[:180] if feedback else ""
                print(f"  step {step + 1}: {action_brief}")
                if fb_brief:
                    print(f"            ↳ {fb_brief}")
                trajectory.append({"action": code, "feedback": feedback})
            else:
                final_status = "step_limit"
                print(f"  reached step limit ({max_steps})")

            results.append(
                {
                    "task_id": task_id,
                    "status": final_status,
                    "final_answer": final_answer,
                    "final_url": adapter.page.url if adapter else None,
                    "final_title": adapter.page.title() if adapter else None,
                    "steps": len(trajectory),
                    "duration_s": round(time.time() - t_start, 1),
                    "trajectory": [
                        {
                            "action": t["action"][:400],
                            "feedback": (t.get("feedback") or "")[:400],
                        }
                        for t in trajectory
                    ],
                }
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append(
                {
                    "task_id": task_id,
                    "status": "error",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "steps": len(trajectory),
                    "duration_s": round(time.time() - t_start, 1),
                }
            )
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
