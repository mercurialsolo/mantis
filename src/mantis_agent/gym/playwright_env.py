"""Playwright-based GymEnvironment for live web applications.

Drives a real browser (Chromium/Firefox) against any URL — CRMs, SaaS tools,
internal apps, public websites. No Docker container needed.

This is the environment to use when:
- The target is a live web app at a URL (not a Docker image)
- Authentication is part of the task (login flow) or pre-injected via cookies
- You need Playwright's DOM access for verification (URL check, element state)

Architecture:
    GymRunner
       ↕ step(Action) / screenshot
    PlaywrightGymEnv
       ↕ Playwright page.mouse / page.keyboard / page.screenshot
    Chromium (headless or headed)
       ↕ HTTP
    Live web app (CRM, SaaS, etc.)
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger(__name__)


class PlaywrightGymEnv(GymEnvironment):
    """Gym environment backed by a Playwright browser session.

    Supports session persistence: after a login task completes, call
    save_session() to snapshot cookies/localStorage to disk. Future tasks
    that need an authenticated user pass the saved state file as
    storage_state and skip the login flow entirely.

    Args:
        start_url: Initial URL to navigate to on reset.
        viewport: Browser viewport size as (width, height).
        headless: Run browser in headless mode.
        browser_type: "chromium", "firefox", or "webkit".
        storage_state: Path to Playwright storage state JSON for pre-auth.
            If the file exists, cookies/localStorage are restored on launch.
        session_dir: Directory to store session state files. Defaults to
            .sessions/ in the current working directory.
        settle_time: Seconds to wait after each action for page to settle.
        timeout: Page navigation timeout in milliseconds.
    """

    def __init__(
        self,
        start_url: str = "about:blank",
        viewport: tuple[int, int] = (1280, 720),
        headless: bool = True,
        browser_type: str = "chromium",
        storage_state: str | None = None,
        session_dir: str = ".sessions",
        settle_time: float = 1.0,
        timeout: int = 30000,
        proxy: dict | None = None,
        cdp_url: str | None = None,
        human_speed: bool = False,
    ):
        self._start_url = start_url
        self._viewport = viewport
        self._headless = headless
        self._browser_type = browser_type
        self._storage_state = storage_state
        self._session_dir = Path(session_dir)
        self._settle_time = settle_time
        self._timeout = timeout
        self._proxy = proxy
        self._cdp_url = cdp_url  # Connect to existing Chrome via CDP
        self._human_speed = human_speed  # Realistic human-like delays

        self._pw_ctx = None
        self._browser = None
        self._context = None
        self._page = None

    def _launch_browser(self) -> None:
        """Launch browser — supports CDP (existing Chrome), stealth, and proxy."""
        from playwright.sync_api import sync_playwright
        import random

        self._pw_ctx = sync_playwright().start()

        # Mode 1: Connect to existing Chrome via CDP (best for CF bypass)
        if self._cdp_url:
            self._browser = self._pw_ctx.chromium.connect_over_cdp(self._cdp_url)
            self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context(
                viewport={"width": self._viewport[0], "height": self._viewport[1]},
            )
            self._context.set_default_timeout(self._timeout)
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
            logger.info(f"Connected to Chrome via CDP: {self._cdp_url}")
            return

        # Mode 2: Launch new browser with stealth
        launcher = getattr(self._pw_ctx, self._browser_type)
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]
        launch_kwargs = {
            "headless": self._headless,
            "args": launch_args,
        }
        if self._proxy:
            launch_kwargs["proxy"] = self._proxy
        self._browser = launcher.launch(**launch_kwargs)

        # Stealth context
        ctx_kwargs: dict[str, Any] = {
            "viewport": {"width": self._viewport[0], "height": self._viewport[1]},
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "locale": "en-US",
            "timezone_id": "America/New_York",
        }
        if self._storage_state:
            import os
            if os.path.exists(self._storage_state):
                ctx_kwargs["storage_state"] = self._storage_state

        self._context = self._browser.new_context(**ctx_kwargs)
        self._context.set_default_timeout(self._timeout)

        self._page = self._context.new_page()

        # Apply stealth
        try:
            from playwright_stealth import stealth_sync
            stealth_sync(self._page)
            logger.info("Stealth mode applied")
        except ImportError:
            self._context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                window.chrome = {runtime: {}};
            """)
            logger.info("Stealth mode applied (fallback)")

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Launch browser and navigate to start URL.

        If the browser is already running, reuses it instead of tearing
        down and relaunching (avoids proxy tunnel re-establishment).

        Args:
            task: Task description (used by the runner, not the browser).
            **kwargs: Optional overrides:
                - start_url: Override the default start URL.
                - storage_state: Override pre-auth state.
        """
        # Allow per-task overrides
        if "start_url" in kwargs:
            self._start_url = kwargs["start_url"]
        if "storage_state" in kwargs:
            self._storage_state = kwargs["storage_state"]

        if self._page is not None and self._browser is not None:
            # Browser already running — reuse it, just navigate
            logger.info("Reusing existing browser session (skipping teardown)")
            if self._start_url and self._start_url != "about:blank":
                try:
                    # Use "commit" instead of "domcontentloaded" — faster, doesn't
                    # hang on heavy JS sites. CDP Chrome + proxy can be slow.
                    self._page.goto(self._start_url, wait_until="commit", timeout=45000)
                    time.sleep(self._settle_time + 1.0)  # Extra settle for JS
                    self.dismiss_popups()
                except Exception as e:
                    logger.warning(f"Navigation failed on reuse: {e}")
                    # Don't relaunch browser (kills CDP) — just continue with current page
                    logger.info("Continuing with current page state")
            return self._capture()

        # No browser running — launch fresh
        if self._page is not None:
            self.close()

        self._launch_browser()

        if self._start_url and self._start_url != "about:blank":
            wait_strategy = "commit" if self._cdp_url else "domcontentloaded"
            self._page.goto(self._start_url, wait_until=wait_strategy, timeout=45000)
            time.sleep(self._settle_time + (1.0 if self._cdp_url else 0))
            self.dismiss_popups()

        return self._capture()

    def step(self, action: Action) -> GymResult:
        """Execute a Mantis action via Playwright and return a screenshot."""
        if self._page is None:
            raise RuntimeError("Browser not initialized — call reset() first")

        # Human-like pre-action delay (realistic browsing pace)
        if self._human_speed:
            import random
            if action.action_type == ActionType.CLICK:
                time.sleep(random.uniform(0.3, 1.0))  # Think before clicking
            elif action.action_type == ActionType.TYPE:
                time.sleep(random.uniform(0.5, 1.5))  # Pause before typing
            elif action.action_type == ActionType.SCROLL:
                time.sleep(random.uniform(0.2, 0.6))  # Reading pace

        self._execute_action(action)

        # Post-action settle time (longer for human speed)
        if action.action_type not in (ActionType.WAIT, ActionType.DONE):
            settle = self._settle_time
            if self._human_speed:
                import random
                settle += random.uniform(0.5, 2.0)  # Variable reading time
            time.sleep(settle)

        # Auto-dismiss popups before capturing screenshot
        # Clears cookie banners, notification prompts, etc. via DOM
        self.dismiss_popups()

        obs = self._capture()

        try:
            info: dict[str, Any] = {
                "url": self._page.url,
                "title": self._page.title(),
            }
        except Exception:
            info = {"url": "", "title": ""}

        # Pure visual mode — no DOM inspection, no focus detection, no type verification
        return GymResult(
            observation=obs,
            reward=0.0,
            done=False,
            info=info,
        )

    def close(self) -> None:
        """Shut down the browser."""
        for resource in (self._context, self._browser):
            if resource:
                try:
                    resource.close()
                except Exception:
                    pass
        if self._pw_ctx:
            try:
                self._pw_ctx.stop()
            except Exception:
                pass
        self._page = None
        self._context = None
        self._browser = None
        self._pw_ctx = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    @property
    def page(self):
        """Direct access to the Playwright page for verification."""
        return self._page

    @property
    def current_url(self) -> str:
        """Current page URL."""
        if self._page:
            return self._page.url
        return ""

    # ── Session persistence ──────────────────────────────────────────────

    def session_path_for(self, name: str) -> str:
        """Return the file path where a named session would be stored.

        Args:
            name: Session name (e.g., "staffai_crm", "classifieds").
                  Used as the filename stem.
        """
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return str(self._session_dir / f"{name}_state.json")

    def save_session(self, name: str) -> str:
        """Persist current browser session (cookies + localStorage) to disk.

        Call this after a login task succeeds. The saved file can be passed
        as storage_state to skip login on subsequent tasks.

        Args:
            name: Session name for the file (e.g., "staffai_crm").

        Returns:
            Path to the saved state file.
        """
        if self._context is None:
            raise RuntimeError("No active browser context — call reset() first")

        path = self.session_path_for(name)
        state = self._context.storage_state()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Session saved: {path} ({len(state.get('cookies', []))} cookies)")
        return path

    def has_session(self, name: str) -> bool:
        """Check if a saved session exists for the given name."""
        return os.path.exists(self.session_path_for(name))

    def load_session(self, name: str) -> None:
        """Set the storage_state to a previously saved session.

        Call this before reset() to restore a logged-in session.

        Args:
            name: Session name used in save_session().
        """
        path = self.session_path_for(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved session found: {path}")
        self._storage_state = path
        logger.info(f"Session loaded: {path}")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _detect_focused_input(self) -> dict | None:
        """Check if an input/textarea element is currently focused.

        Returns a dict with element info if a form field has focus, else None.
        Used by the runner to generate form-aware nudges.
        """
        try:
            result = self._page.evaluate("""() => {
                const el = document.activeElement;
                if (!el) return null;
                const tag = el.tagName.toLowerCase();
                if (tag !== 'input' && tag !== 'textarea' && !el.isContentEditable) return null;
                return {
                    tag: tag,
                    type: el.getAttribute('type') || '',
                    name: el.getAttribute('name') || '',
                    placeholder: el.getAttribute('placeholder') || '',
                    id: el.id || '',
                    value: (el.value || '').substring(0, 100),
                    empty: !(el.value || '').trim(),
                };
            }""")
            return result
        except Exception:
            return None

    def _verify_typed_text(self, expected_text: str) -> dict | None:
        """Verify that typed text actually landed in the active element.

        Returns dict with verification result, or None if check not possible.
        """
        try:
            result = self._page.evaluate("""() => {
                const el = document.activeElement;
                if (!el) return null;
                const tag = el.tagName.toLowerCase();
                if (tag !== 'input' && tag !== 'textarea' && !el.isContentEditable) return null;
                return {
                    tag: tag,
                    type: el.getAttribute('type') || '',
                    name: el.getAttribute('name') || '',
                    value: (el.value || '').substring(0, 200),
                    placeholder: el.getAttribute('placeholder') || '',
                };
            }""")
            if result is None:
                return {"success": False, "reason": "no input field focused after typing"}
            actual = result.get("value", "")
            if expected_text in actual:
                return {"success": True, "field": result.get("name") or result.get("type"), "value": actual}
            else:
                return {
                    "success": False,
                    "reason": f"field contains '{actual}' instead of '{expected_text}'",
                    "field": result.get("name") or result.get("placeholder") or result.get("type"),
                    "actual_value": actual,
                }
        except Exception:
            return None

    def dismiss_popups(self) -> None:
        """Auto-dismiss common popups (cookies, notifications) via DOM.

        Called before each observation to clear UI obstacles that waste
        vision model steps. Uses DOM selectors — no vision needed.
        """
        if not self._page:
            return

        # Common cookie consent patterns
        selectors = [
            # By text
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("Accept Cookies")',
            'button:has-text("I Accept")',
            'button:has-text("OK")',
            'button:has-text("Got it")',
            'button:has-text("Agree")',
            # By common IDs/classes
            '#onetrust-accept-btn-handler',
            '.cookie-accept',
            '[data-testid="cookie-accept"]',
            '.cc-accept',
            '.cc-btn.cc-dismiss',
            '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
            # Close/X buttons on overlays
            'button[aria-label="Close"]',
            'button[aria-label="Dismiss"]',
        ]

        for selector in selectors:
            try:
                el = self._page.query_selector(selector)
                if el and el.is_visible():
                    el.click()
                    logger.info(f"Dismissed popup: {selector}")
                    time.sleep(0.5)
                    return  # One dismissal per call
            except Exception:
                continue

    def _capture(self, annotate_som: bool = False) -> GymObservation:
        """Take a screenshot and return as GymObservation.

        If annotate_som=True, overlays numbered red labels on all interactive
        elements (Set-of-Mark), and includes the element list in extras.
        """
        png_bytes = self._page.screenshot(type="png")

        elements = []
        element_text = ""
        if annotate_som:
            elements = self._get_interactive_elements()
            if elements:
                png_bytes = self._annotate_screenshot(png_bytes, elements)
                element_text = self._format_element_list(elements)

        screenshot = Image.open(io.BytesIO(png_bytes))

        return GymObservation(
            screenshot=screenshot,
            extras={
                "url": self._page.url,
                "title": self._page.title(),
                "elements": elements,
                "element_text": element_text,
                # No DOM state — SoM only, no DOM inspection
            },
        )

    def _get_interactive_elements(self) -> list[dict]:
        """Query DOM for all interactive elements with bounding boxes."""
        try:
            elements = self._page.evaluate("""() => {
                const selectors = [
                    'a[href]', 'button', 'input', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[role="menuitem"]',
                    '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
                    '[role="option"]', '[role="switch"]', '[role="combobox"]',
                    '[onclick]', '[tabindex]',
                    'label', 'summary',
                ];
                const seen = new Set();
                const result = [];
                for (const sel of selectors) {
                    for (const el of document.querySelectorAll(sel)) {
                        if (seen.has(el)) continue;
                        seen.add(el);
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 5 || rect.height < 5) continue;
                        if (rect.top > window.innerHeight || rect.left > window.innerWidth) continue;
                        if (rect.bottom < 0 || rect.right < 0) continue;
                        const text = (el.textContent || '').trim().substring(0, 80).replace(/\\s+/g, ' ');
                        const value = (el.value || '').substring(0, 50);
                        const placeholder = el.placeholder || el.getAttribute('placeholder') || '';
                        result.push({
                            tag: el.tagName.toLowerCase(),
                            text: text,
                            type: el.getAttribute('type') || '',
                            role: el.getAttribute('role') || '',
                            id: el.id || '',
                            name: el.getAttribute('name') || '',
                            value: value,
                            placeholder: placeholder,
                            bbox: {
                                x: Math.round(rect.x), y: Math.round(rect.y),
                                w: Math.round(rect.width), h: Math.round(rect.height),
                            },
                        });
                    }
                }
                return result;
            }""")
            return [{"id": i, **el} for i, el in enumerate(elements)]
        except Exception:
            return []

    def _annotate_screenshot(self, png_bytes: bytes, elements: list[dict]) -> bytes:
        """Draw numbered SoM labels on the screenshot."""
        from PIL import ImageDraw, ImageFont

        img = Image.open(io.BytesIO(png_bytes))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for el in elements:
            bbox = el["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            eid = el["id"]

            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)

            label = str(eid)
            label_w = len(label) * 8 + 4
            label_h = 14
            lx, ly = max(0, x), max(0, y - label_h)
            draw.rectangle([lx, ly, lx + label_w, ly + label_h], fill="red")
            draw.text((lx + 2, ly), label, fill="white", font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _format_element_list(elements: list[dict]) -> str:
        """Format elements as a numbered text list for the brain prompt."""
        lines = ["INTERACTIVE ELEMENTS (use element number to identify targets):"]
        for el in elements[:40]:
            parts = [f"[{el['id']}]"]
            tag = el["tag"]
            if tag == "input":
                parts.append(f"<input type={el.get('type', 'text')}>")
                if el.get("value"):
                    parts.append(f'value="{el["value"]}"')
                elif el.get("placeholder"):
                    parts.append(f'placeholder="{el["placeholder"]}"')
                if el.get("name"):
                    parts.append(f'name="{el["name"]}"')
            elif tag == "button":
                parts.append(f'<button> "{el["text"][:40]}"')
            elif tag == "a":
                parts.append(f'<link> "{el["text"][:40]}"')
            elif tag == "select":
                parts.append(f"<dropdown>")
                if el.get("name"):
                    parts.append(f'name="{el["name"]}"')
            else:
                parts.append(f'<{tag}> "{el["text"][:40]}"')
            lines.append("  " + " ".join(parts))
        if len(elements) > 40:
            lines.append(f"  ... ({len(elements) - 40} more)")
        return "\n".join(lines)

    def _snap_to_element(self, x: int, y: int) -> tuple[int, int]:
        """If (x,y) lands inside a SoM element's bbox, snap to its center.

        This corrects imprecise coordinate predictions from the brain.
        If the click doesn't land inside any known element, returns original coords.
        """
        try:
            elements = self._get_interactive_elements()
            for el in elements:
                bbox = el["bbox"]
                bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    cx, cy = bx + bw // 2, by + bh // 2
                    if (cx, cy) != (x, y):
                        print(f"  [snap] ({x},{y}) → ({cx},{cy}) [{el['id']}] {el.get('tag','')} '{el.get('text','')[:30]}'")
                    return cx, cy
        except Exception:
            pass
        return x, y

    def _get_dom_state(self) -> str:
        """Get a summary of current form field states visible on page."""
        try:
            fields = self._page.evaluate("""() => {
                const inputs = document.querySelectorAll('input, textarea, select');
                const result = [];
                for (const el of inputs) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 5 || rect.height < 5) continue;
                    const tag = el.tagName.toLowerCase();
                    const name = el.name || el.id || el.placeholder || el.type || tag;
                    const value = el.value || '';
                    const focused = el === document.activeElement;
                    result.push({name, value: value.substring(0, 100), type: el.type || '', focused, tag});
                }
                return result;
            }""")
            if not fields:
                return ""
            lines = ["FORM STATE:"]
            for f in fields:
                status = "FOCUSED" if f["focused"] else ""
                val = f["value"] if f["value"] else "(empty)"
                line = f"  {f['name']}: {val}"
                if status:
                    line += f" [{status}]"
                lines.append(line)
            return "\n".join(lines)
        except Exception:
            return ""

    def _execute_action(self, action: Action) -> None:
        """Translate and execute a Mantis Action via Playwright."""
        match action.action_type:
            case ActionType.CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                button = action.params.get("button", "left")
                self._page.mouse.click(x, y, button=button)

            case ActionType.DOUBLE_CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                self._page.mouse.dblclick(x, y)

            case ActionType.TYPE:
                text = action.params["text"]
                if text.startswith("http://") or text.startswith("https://"):
                    try:
                        self._page.goto(text, wait_until="domcontentloaded", timeout=15000)
                        logger.info(f"Navigated to {text}")
                    except Exception:
                        self._page.keyboard.type(text)
                elif self._human_speed:
                    # Type character by character with realistic delays
                    import random
                    for char in text:
                        self._page.keyboard.type(char)
                        time.sleep(random.uniform(0.03, 0.12))  # 30-120ms per keystroke
                else:
                    self._page.keyboard.type(text)

            case ActionType.KEY_PRESS:
                combo = action.params["keys"]
                pw_combo = self._normalize_key_combo(combo)

                # Handle tab management shortcuts that Playwright headless
                # doesn't support natively via keyboard
                combo_lower = combo.lower().replace(" ", "")
                if combo_lower in ("ctrl+t", "control+t"):
                    # Open new tab
                    new_page = self._context.new_page()
                    self._pages = getattr(self, '_pages', [self._page])
                    self._pages.append(new_page)
                    self._page = new_page
                    logger.info(f"New tab opened ({len(self._pages)} tabs)")
                elif combo_lower in ("ctrl+w", "control+w"):
                    # Close current tab, switch to previous
                    pages = getattr(self, '_pages', [self._page])
                    if len(pages) > 1:
                        closing = self._page
                        if closing in pages:
                            pages.remove(closing)
                        self._page = pages[-1]
                        try:
                            closing.close()
                        except Exception:
                            pass
                        self._page.bring_to_front()
                        logger.info(f"Tab closed ({len(pages)} tabs remaining)")
                elif combo_lower in ("ctrl+tab", "control+tab"):
                    # Switch to next tab
                    pages = getattr(self, '_pages', [self._page])
                    if len(pages) > 1:
                        idx = pages.index(self._page)
                        self._page = pages[(idx + 1) % len(pages)]
                        self._page.bring_to_front()
                        logger.info(f"Switched to tab {pages.index(self._page) + 1}/{len(pages)}")
                elif combo_lower in ("alt+left", "alt+arrowleft"):
                    # Browser back — use short timeout to avoid hanging on heavy pages
                    try:
                        self._page.go_back(wait_until="commit", timeout=15000)
                    except Exception:
                        logger.warning("go_back timed out, using keyboard shortcut")
                        self._page.keyboard.press("Alt+ArrowLeft")
                else:
                    try:
                        self._page.keyboard.press(pw_combo)
                    except Exception as e:
                        logger.warning(f"Key press failed '{pw_combo}': {e}")

            case ActionType.SCROLL:
                direction = action.params["direction"]
                amount = action.params.get("amount", 3)
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                dx, dy = 0, 0
                if direction == "up":
                    dy = -amount * 100
                elif direction == "down":
                    dy = amount * 100
                elif direction == "left":
                    dx = -amount * 100
                elif direction == "right":
                    dx = amount * 100
                self._page.mouse.move(x, y)
                self._page.mouse.wheel(dx, dy)

            case ActionType.DRAG:
                sx, sy = action.params["start_x"], action.params["start_y"]
                ex, ey = action.params["end_x"], action.params["end_y"]
                self._page.mouse.move(sx, sy)
                self._page.mouse.down()
                self._page.mouse.move(ex, ey, steps=10)
                self._page.mouse.up()

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                time.sleep(min(seconds, 10.0))

            case ActionType.DONE:
                pass  # Handled by the runner

    @staticmethod
    def _normalize_key_combo(combo: str) -> str:
        """Normalize key names to Playwright format.

        e.g. 'ctrl+c' → 'Control+c', 'enter' → 'Enter', 'command+v' → 'Meta+v'
        """
        key_map = {
            "ctrl": "Control", "control": "Control",
            "alt": "Alt", "option": "Alt",
            "shift": "Shift",
            "meta": "Meta", "cmd": "Meta", "command": "Meta", "win": "Meta", "super": "Meta",
            "return": "Enter", "enter": "Enter",
            "escape": "Escape", "esc": "Escape",
            "tab": "Tab", "backspace": "Backspace", "delete": "Delete",
            "home": "Home", "end": "End",
            "pageup": "PageUp", "page_up": "PageUp",
            "pagedown": "PageDown", "page_down": "PageDown",
            "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
            "f6": "F6", "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10",
            "f11": "F11", "f12": "F12", "space": " ",
            "up": "ArrowUp", "arrowup": "ArrowUp",
            "down": "ArrowDown", "arrowdown": "ArrowDown",
            "left": "ArrowLeft", "arrowleft": "ArrowLeft",
            "right": "ArrowRight", "arrowright": "ArrowRight",
        }
        parts = [p.strip() for p in combo.split("+")]
        normalized = [key_map.get(p.lower(), p) for p in parts]
        return "+".join(normalized)

    def __repr__(self) -> str:
        return f"PlaywrightGymEnv(url={self._start_url!r}, viewport={self._viewport})"
