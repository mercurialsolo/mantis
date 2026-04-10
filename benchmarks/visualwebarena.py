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

# Reuse the OSWorld image directly — it already has llama.cpp (with the
# Gemma 4 vision mmproj), Playwright + Chromium, the mantis_agent package,
# and every Python dep the agent loop needs. Building a separate VWA image
# would just duplicate the llama.cpp compile step and add cold-start time.
# The QEMU/VM code paths in that image are dormant unless we invoke them.
from modal_osworld_direct import (
    GEMMA4_MODEL,
    GGUF_CONFIGS,
    download_model,
    start_llama_server,
    image as vwa_image,  # reuse OSWorld image as-is
    vol,
)


# ── VWA sidecar hostnames ─────────────────────────────────────────────────────
# Override via env vars, or they auto-resolve from Modal deployed sidecars.
# The sidecars are deployed via: modal deploy benchmarks/vwa_sidecars.py
# which gives each service a URL like https://user--vwa-sidecars-vwa-homepage.modal.run
#
# To find your deployed URLs:
#   modal app list | grep vwa-sidecars
#   modal app logs <app-id>
VWA_ENDPOINTS = {
    "classifieds": os.environ.get("VWA_CLASSIFIEDS_URL", "https://getmason--vwa-sidecars-vwa-classifieds.modal.run"),
    "shopping":    os.environ.get("VWA_SHOPPING_URL",    "https://getmason--vwa-sidecars-vwa-shopping.modal.run"),
    "reddit":      os.environ.get("VWA_REDDIT_URL",      "https://getmason--vwa-sidecars-vwa-reddit.modal.run"),
    "wikipedia":   os.environ.get("VWA_WIKIPEDIA_URL",   "https://getmason--vwa-sidecars-vwa-wikipedia.modal.run"),
    "homepage":    os.environ.get("VWA_HOMEPAGE_URL",    "https://getmason--vwa-sidecars-vwa-homepage.modal.run"),
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

    # ── Set-of-Mark (SoM) element discovery ─────────────────────────────
    def _get_interactive_elements(self) -> list[dict]:
        """Query the DOM for all interactive/clickable elements with bounding boxes.

        Returns a list of dicts with keys: id, tag, text, role, bbox (x, y, w, h).
        Elements are numbered sequentially starting from 0.
        """
        js_code = """
        () => {
            const selectors = [
                'a[href]', 'button', 'input', 'select', 'textarea',
                '[role="button"]', '[role="link"]', '[role="menuitem"]',
                '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
                '[role="option"]', '[role="switch"]', '[role="combobox"]',
                '[onclick]', '[tabindex]',
                'label', 'summary',
            ];
            const seen = new Set();
            const elements = [];
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    if (seen.has(el)) continue;
                    seen.add(el);
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 5 || rect.height < 5) continue;
                    if (rect.top > window.innerHeight || rect.left > window.innerWidth) continue;
                    if (rect.bottom < 0 || rect.right < 0) continue;
                    const text = (el.textContent || el.value || el.placeholder || el.getAttribute('aria-label') || '').trim().substring(0, 80);
                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        text: text,
                        role: el.getAttribute('role') || '',
                        type: el.getAttribute('type') || '',
                        bbox: {
                            x: Math.round(rect.x),
                            y: Math.round(rect.y),
                            w: Math.round(rect.width),
                            h: Math.round(rect.height),
                        },
                    });
                }
            }
            return elements;
        }
        """
        try:
            elements = self.page.evaluate(js_code)
            return [{"id": i, **el} for i, el in enumerate(elements)]
        except Exception as e:
            print(f"[SoM] element query failed: {e}")
            return []

    def _annotate_screenshot(self, png_bytes: bytes, elements: list[dict]) -> bytes:
        """Draw numbered SoM labels on the screenshot at each element's bbox.

        Returns annotated PNG bytes. Each label is a small colored rectangle
        with the element's ID number, positioned at the top-left corner of
        the element's bounding box.
        """
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(io.BytesIO(png_bytes))
        draw = ImageDraw.Draw(img)

        # Use a small built-in font (no external font file needed)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        for el in elements:
            bbox = el["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            eid = el["id"]

            # Draw a thin border around the element
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)

            # Draw the ID label (small red box with white text)
            label = str(eid)
            label_w = len(label) * 8 + 4
            label_h = 14
            lx, ly = max(0, x), max(0, y - label_h)
            draw.rectangle([lx, ly, lx + label_w, ly + label_h], fill="red")
            draw.text((lx + 2, ly), label, fill="white", font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── Observation ───────────────────────────────────────────────────────
    def screenshot(self) -> bytes:
        """Return a PNG screenshot of the current page as bytes."""
        return self.page.screenshot(type="png")

    def screenshot_som(self) -> tuple[bytes, list[dict]]:
        """Return a SoM-annotated screenshot + element list.

        The screenshot has numbered labels drawn on every interactive element.
        The element list maps IDs to bounding boxes so ``click_element(id)``
        can compute the center coordinates.
        """
        raw_png = self.page.screenshot(type="png")
        elements = self._get_interactive_elements()
        annotated = self._annotate_screenshot(raw_png, elements)
        return annotated, elements

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
            elif low in ("tab", "backspace", "delete", "home", "end"):
                normalized.append(low.capitalize())
            elif low in ("pageup", "page_up"):
                normalized.append("PageUp")
            elif low in ("pagedown", "page_down"):
                normalized.append("PageDown")
            elif low in ("up", "arrowup"):
                normalized.append("ArrowUp")
            elif low in ("down", "arrowdown"):
                normalized.append("ArrowDown")
            elif low in ("left", "arrowleft"):
                normalized.append("ArrowLeft")
            elif low in ("right", "arrowright"):
                normalized.append("ArrowRight")
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

    def click_element(self, element_id: int) -> None:
        """Click the center of a SoM-labeled element by its ID.

        The model sees numbered labels on the screenshot and outputs
        ``click_element(3)`` instead of guessing pixel coordinates.
        This is dramatically more accurate for flat UI elements.
        """
        if not hasattr(self, '_last_elements') or not self._last_elements:
            print(f"[SoM] No elements cached — falling back to noop")
            return
        matches = [el for el in self._last_elements if el["id"] == element_id]
        if not matches:
            print(f"[SoM] Element {element_id} not found (max id: {max(e['id'] for e in self._last_elements)})")
            return
        el = matches[0]
        cx = el["bbox"]["x"] + el["bbox"]["w"] // 2
        cy = el["bbox"]["y"] + el["bbox"]["h"] // 2
        print(f"[SoM] Clicking element {element_id}: '{el['text'][:40]}' at ({cx}, {cy})")
        self.page.mouse.click(cx, cy)

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

def load_vwa_task_configs(
    config_dir: Optional[Path] = None,
    endpoints: Optional[dict] = None,
) -> list[dict]:
    """Load VWA task configs from raw JSON files (234 + 466 + 210 = 910 tasks).

    The raw configs from the VWA repo use URL placeholders like ``__CLASSIFIEDS__``,
    ``__SHOPPING__``, ``__REDDIT__``, ``__WIKIPEDIA__``, ``__HOMEPAGE__``. These get
    replaced with the actual sidecar hostnames from ``endpoints``.

    Supports two layouts:
      1. **Per-domain raw files**: ``config_dir/test_classifieds.raw.json``, etc.
         (the format in the VWA GitHub repo — one big list per domain).
      2. **Per-task split files**: ``config_dir/test_classifieds/*.json``
         (if someone already split them — one file per task).
    """
    if config_dir is None:
        config_dir = Path("/data/vwa/config_files/vwa")
    if endpoints is None:
        endpoints = VWA_ENDPOINTS

    # Placeholder → actual URL mapping
    replacements = {
        "__CLASSIFIEDS__": endpoints.get("classifieds", "http://localhost:9980"),
        "__SHOPPING__":    endpoints.get("shopping",    "http://localhost:7770"),
        "__REDDIT__":      endpoints.get("reddit",      "http://localhost:9999"),
        "__WIKIPEDIA__":   endpoints.get("wikipedia",   "http://localhost:8888"),
        "__HOMEPAGE__":    endpoints.get("homepage",     "http://localhost:4399"),
    }

    def _replace_urls(task: dict) -> dict:
        """Replace URL placeholders in all string fields."""
        task_str = json.dumps(task)
        for placeholder, url in replacements.items():
            task_str = task_str.replace(placeholder, url)
        return json.loads(task_str)

    configs: list[dict] = []
    if not config_dir.exists():
        return configs

    for domain in ("test_classifieds", "test_shopping", "test_reddit"):
        # Try raw file first (the GitHub repo format)
        raw_path = config_dir / f"{domain}.raw.json"
        if raw_path.exists():
            try:
                raw_tasks = json.loads(raw_path.read_text())
                for task in raw_tasks:
                    configs.append(_replace_urls(task))
            except Exception as e:
                print(f"  warn: failed to load {raw_path}: {e}")
            continue

        # Fall back to per-task split files
        split_dir = config_dir / domain
        if split_dir.exists():
            for f in sorted(split_dir.glob("*.json")):
                try:
                    configs.append(_replace_urls(json.loads(f.read_text())))
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

The viewport is 1280x720 pixels.

# Set-of-Mark (SoM) — IMPORTANT

The screenshot has RED NUMBERED LABELS on every interactive element (links, buttons, inputs, etc.). Each label shows the element's ID number. To click an element, use its ID:

- `click_element(3)` — click the element labeled [3] in the screenshot

This is MUCH more accurate than guessing pixel coordinates. **Always prefer click_element(id) over click(x, y)** when you can see a numbered label on the target.

# Available helpers (already imported — just call them)

- `click_element(id)` — **PREFERRED** — click element by its SoM label number
- `click(x, y)` — fallback: click at raw pixel coordinates (only if no SoM label is visible)
- `type_text(text)` — type text into the focused field
- `key(combo)` — press a key or combo. Examples: `key("Return")`, `key("ctrl+l")` (address bar), `key("Tab")`, `key("Escape")`, `key("Page_Down")`
- `wait(seconds=2)` — pause for N seconds (page loads, AJAX)
- `scroll(x, y, clicks=-3)` — scroll at (x, y). Negative clicks = scroll down
- `double_click(x, y)` — double click at coordinates

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
        "click_element": adapter.click_element,
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

    # Check if VWA sidecars are actually reachable. If not, fall back to
    # public-URL smoke tests so the pipeline can be validated end-to-end
    # without spinning up the Docker stack.
    sidecars_ok = False
    if tasks:
        import requests as _req
        test_url = tasks[0].get("start_url", "")
        try:
            _req.get(test_url, timeout=5)
            sidecars_ok = True
            print(f"  VWA sidecars reachable ({test_url[:50]})")
        except Exception:
            print(f"  VWA sidecars NOT reachable ({test_url[:50]}) — using public-URL smoke tests")

    if not tasks or not sidecars_ok:
        tasks = [
            {
                "task_id": "smoke-wikipedia",
                "intent": (
                    "You are on the Wikipedia main page. Find and click the link "
                    "to the article about 'Python (programming language)'. Then "
                    "report the first sentence of the article. Output DONE with "
                    "your answer."
                ),
                "start_url": "https://en.wikipedia.org/wiki/Main_Page",
                "sites": ["public"],
            },
            {
                "task_id": "smoke-example-com",
                "intent": (
                    "You are on example.com. Read the page and tell me what the "
                    "main heading says. Then output DONE with your answer."
                ),
                "start_url": "https://example.com",
                "sites": ["public"],
            },
        ]
        print(f"  Using {len(tasks)} public-URL smoke tasks")

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
                # 3a. Observe — SoM-annotated screenshot with numbered elements
                try:
                    png, elements = adapter.screenshot_som()
                    adapter._last_elements = elements  # cache for click_element()
                except Exception as _e:
                    print(f"  step {step + 1}: screenshot failed: {_e}")
                    break

                # Build a text summary of SoM elements for the inference
                if elements:
                    elem_summary = "INTERACTIVE ELEMENTS (click by ID using click_element(id)):\n"
                    for el in elements[:40]:  # cap at 40 to avoid token overflow
                        elem_summary += f"  [{el['id']}] {el['tag']}"
                        if el.get('text'):
                            elem_summary += f" \"{el['text'][:50]}\""
                        if el.get('type'):
                            elem_summary += f" type={el['type']}"
                        elem_summary += "\n"
                    if len(elements) > 40:
                        elem_summary += f"  ... and {len(elements) - 40} more elements\n"
                else:
                    elem_summary = ""

                # 3b. Infer
                try:
                    response = _llama_infer(
                        screenshot_png=png,
                        instruction=instruction,
                        hint_text=hint_text + "\n\n" + elem_summary if elem_summary else hint_text,
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
