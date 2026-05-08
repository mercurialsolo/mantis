"""XdotoolGymEnv — pure screen-level CUA environment using Xvfb + xdotool.

Strictly screenshot + input events. No CDP, no DOM, no JS injection.

Architecture:
    Xvfb (virtual display :99)
      └── Chrome (regular process, zero automation hooks)
            ↕ X11 events
    xdotool → mousemove, click, type, key
    mss/scrot → screenshot.png

    Brain sees screenshot → outputs action → xdotool executes

The model handles everything visually: cookies, popups, navigation.
The env just executes actions and returns screenshots.

Requirements:
    apt-get install xvfb xdotool scrot chromium-browser
    pip install mss pillow
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from typing import Any

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger(__name__)


def scale_brain_to_display(
    x_brain: int | float,
    y_brain: int | float,
    brain_size: tuple[int, int],
    display_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale (x, y) from brain-image pixel space to display pixel space.

    See docs/reference/coordinate-spaces.md for the full contract.
    Returns the rounded display-space integer (x, y).
    """
    bw, bh = brain_size
    dw, dh = display_size
    if bw <= 0 or bh <= 0:
        raise ValueError(f"brain_size must be positive: got {brain_size}")
    if dw <= 0 or dh <= 0:
        raise ValueError(f"display_size must be positive: got {display_size}")
    return round(x_brain * dw / bw), round(y_brain * dh / bh)


class XdotoolGymEnv(GymEnvironment):
    """Pure screen-level environment — Xvfb + xdotool + mss.

    No Playwright, no CDP, no DOM access. Just pixels and input events.
    Identical to a human using a mouse and keyboard.

    Args:
        start_url: URL to open in the browser on reset.
        viewport: Screen size as (width, height).
        browser: Browser command ("chromium-browser", "firefox", "google-chrome").
        display: X11 display number (e.g., ":99"). If None, starts Xvfb.
        settle_time: Seconds to wait after actions for page to update.
        human_speed: Add realistic delays between actions.
        proxy_server: HTTP proxy URL (e.g. "http://127.0.0.1:3128").
        profile_dir: Chrome user-data-dir for cookie/session persistence.
    """

    def __init__(
        self,
        start_url: str = "about:blank",
        viewport: tuple[int, int] = (1280, 720),
        browser: str = "chromium-browser",
        display: str | None = None,
        settle_time: float = 1.5,
        human_speed: bool = False,
        proxy_server: str = "",
        profile_dir: str = "/data/chrome-profile",
        save_screenshots: str = "",
        cdp_port: int = 9222,
    ):
        self._start_url = start_url
        self._viewport = viewport
        self._browser_cmd = browser
        self._display = display
        self._settle_time = settle_time
        self._human_speed = human_speed
        self._proxy_server = proxy_server
        self._profile_dir = profile_dir
        self._save_screenshots = save_screenshots  # Dir to save screenshots for replay
        self._step_counter = 0
        # CDP read-only access — used by current_url to query Chrome's
        # navigation state directly instead of relying on the screenshot
        # extractor reading the address bar pixels (issue #89 §1).
        self._cdp_port = cdp_port

        self._xvfb_proc = None
        self._browser_proc = None
        self._env = {}

    # ── Xvfb + Browser ──────────��───────────────────────────────────

    def _start_xvfb(self) -> str:
        """Start Xvfb virtual display if not already running."""
        if self._display:
            return self._display

        display = ":99"
        cmd = [
            "Xvfb", display,
            "-screen", "0", f"{self._viewport[0]}x{self._viewport[1]}x24",
            "-ac", "-nolisten", "tcp",
        ]
        self._xvfb_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        logger.info(f"Xvfb started on {display}")
        return display

    def _start_browser(self, url: str) -> None:
        """Launch browser with persistent profile, no CDP."""
        os.makedirs(self._profile_dir, exist_ok=True)

        # Clean session recovery files to prevent "Restore pages?" dialog.
        # Preserves: Cookies, Local Storage, Login Data, Preferences.
        default_dir = os.path.join(self._profile_dir, "Default")
        if os.path.isdir(default_dir):
            for stale in ["Current Session", "Current Tabs",
                          "Last Session", "Last Tabs", "Session Storage"]:
                path = os.path.join(default_dir, stale)
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            import shutil
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            os.remove(path)
                    except OSError:
                        pass

        # Remove lock files from prior unclean shutdown
        for lock in ["SingletonLock", "SingletonSocket", "SingletonCookie"]:
            path = os.path.join(self._profile_dir, lock)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        # Disable Chrome's "Save password?" / autofill prompts via the
        # profile Preferences file. The CLI flags
        # ``--disable-save-password-bubble`` and
        # ``--disable-features=PasswordManagerOnboarding,...`` don't kill
        # the bubble in current Chromium — the canonical kill switch is
        # ``credentials_enable_service: false`` plus
        # ``profile.password_manager_enabled: false`` in
        # ``Default/Preferences``. Symptom: a "Save password?" overlay
        # rendered after the login submit on staff-crm intercepted clicks
        # on the Lead Management table (run 13).
        prefs_dir = os.path.join(self._profile_dir, "Default")
        os.makedirs(prefs_dir, exist_ok=True)
        prefs_path = os.path.join(prefs_dir, "Preferences")
        prefs: dict[str, Any] = {}
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path) as f:
                    prefs = json.load(f)
            except (OSError, json.JSONDecodeError):
                prefs = {}
        prefs["credentials_enable_service"] = False
        prefs["credentials_enable_autosignin"] = False
        profile_section = prefs.setdefault("profile", {})
        profile_section["password_manager_enabled"] = False
        try:
            with open(prefs_path, "w") as f:
                json.dump(prefs, f)
        except OSError as exc:
            logger.debug(
                "could not write Preferences to suppress password bubble: %s",
                exc,
            )

        cmd = [
            self._browser_cmd,
            "--no-sandbox",
            "--test-type",  # Suppress --no-sandbox warning bar
            "--disable-gpu",
            "--no-first-run",
            "--disable-default-apps",
            "--disable-infobars",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--noerrdialogs",
            "--disable-features=InfiniteSessionRestore",
            "--disable-blink-features=AutomationControlled",  # Hide navigator.webdriver
            "--disable-dev-shm-usage",
            # Suppress Chrome's "Save password?" / autofill prompts that
            # block the CUA after login (run 033 hit this on staff-crm).
            "--password-store=basic",
            "--disable-save-password-bubble",
            "--disable-features=PasswordManagerOnboarding,AutofillEnableAccountWalletStorage,AutofillServerCommunication",
            # CDP for current_url + Input.insertText (run 032 found Chrome
            # rejects WS connections with 403 Forbidden without explicit
            # --remote-allow-origins). Bound to localhost only — wildcard
            # origin is safe because the port itself is unreachable from
            # outside the container.
            f"--remote-debugging-port={self._cdp_port}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-allow-origins=*",
            f"--window-size={self._viewport[0]},{self._viewport[1]}",
            "--start-maximized",
            f"--user-data-dir={self._profile_dir}",
        ]
        if self._proxy_server:
            cmd.append(f"--proxy-server={self._proxy_server}")
        cmd.append(url)

        self._browser_proc = subprocess.Popen(
            cmd, env=self._env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        logger.info(f"Browser started: {self._browser_cmd} → {url}")

    # ── Screenshot ──────────────────────────────────────────────────

    def screenshot(self) -> Image.Image:
        """Public: capture current screenshot as PIL Image."""
        return self._screenshot()

    def _screenshot(self) -> Image.Image:
        """Capture screenshot via mss (fast) or scrot (fallback)."""
        try:
            import mss
            with mss.mss(display=self._env.get("DISPLAY", ":99")) as sct:
                monitor = sct.monitors[0]
                img = sct.grab(monitor)
                return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        except Exception:
            pass

        try:
            tmp = "/tmp/mantis_screenshot.png"
            subprocess.run(
                ["scrot", "-o", tmp],
                env=self._env, capture_output=True, timeout=5,
            )
            return Image.open(tmp)
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return Image.new("RGB", self._viewport, "gray")

    # ── xdotool ───��─────────────────────────────────────────────��───

    def _xdotool(self, *args: str) -> None:
        """Run xdotool command."""
        subprocess.run(
            ["xdotool"] + list(args),
            env=self._env, capture_output=True, timeout=5,
        )

    def _cdp_insert_text(self, text: str) -> bool:
        """Type text via Chrome DevTools Protocol's Input.insertText.

        Returns True on success, False on any failure (caller falls back).
        Uses the same `--remote-debugging-port` Chrome was launched with
        (`current_url` already uses the same port). Dispatches a native
        input event, which React's controlled-input onChange registers
        cleanly — same outcome as Playwright's ``el.type()``.

        This is the preferred typing path because xdotool's keypress
        events don't reliably round-trip through React's value binding
        (run 020-031, b3b4364 commit history). Click/scroll/screenshot
        stay xdotool; only the type-text execution moves to CDP.
        """
        try:
            import json as _json
            import urllib.request
            try:
                import websocket  # websocket-client package
            except ImportError:
                return False
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
            ws_url = None
            for tab in tabs:
                if tab.get("type") != "page":
                    continue
                url = str(tab.get("url") or "")
                if not url or url.startswith("chrome://") or url.startswith("about:"):
                    continue
                ws_url = tab.get("webSocketDebuggerUrl")
                if ws_url:
                    break
            if not ws_url:
                return False
            ws = websocket.create_connection(ws_url, timeout=3)
            try:
                req_id = int(time.time() * 1000) % 1_000_000
                ws.send(_json.dumps({
                    "id": req_id,
                    "method": "Input.insertText",
                    "params": {"text": text},
                }))
                ws.settimeout(3)
                for _ in range(8):
                    raw = ws.recv()
                    if not raw:
                        continue
                    resp = _json.loads(raw)
                    if resp.get("id") != req_id:
                        continue
                    if resp.get("error"):
                        return False
                    return True
                return False
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("CDP insertText failed: %s", exc)
            return False

    def _xdotool_type(self, text: str) -> None:
        """Type text via clipboard-paste (preferred) or xdotool fallback.

        Diagnosed empirically across runs 020-029 + the claude-in-chrome
        MCP comparison: xdotool's ``type`` emits X-level keypress events
        that Chrome translates to JS ``KeyboardEvent``s, but React's
        controlled inputs frequently fail to flush those into component
        state — the typed text shows visually in the field but the form's
        internal state stays empty, so submit produces ``AUTH_FAIL_001``.

        Commit b3b4364 (Apr 2026) documented this exact issue on the
        Playwright path and fixed it via ``el.type()``. When xdotool
        replaced Playwright for "pure screen-level CUA", the React
        compatibility was lost.

        Clipboard-paste sidesteps the issue: xclip writes the text to
        the X clipboard and xdotool sends ``ctrl+v``. The browser fires
        a proper ``paste`` event which React registers, populating state
        correctly. Falls back to xdotool ``type`` for environments
        without xclip or when human-speed mode is requested.
        """
        if not text:
            return

        # Preferred path: CDP Input.insertText fires a synthesized input
        # event that React's onChange picks up reliably — same mechanism
        # MCP form_input / Playwright el.type() use. Falls through to
        # clipboard-paste then xdotool type if CDP is unreachable.
        if (
            not self._human_speed
            and os.environ.get("MANTIS_DISABLE_CDP_TYPE", "") != "1"
        ):
            if self._cdp_insert_text(text):
                return
            logger.info("CDP insertText unavailable; trying clipboard paste")

        # Prefer clipboard-paste for React/Vue/framework compatibility.
        # Disabled by setting MANTIS_DISABLE_PASTE_TYPE=1 or in human-speed mode
        # (which intentionally simulates per-keystroke typing).
        use_paste = (
            not self._human_speed
            and os.environ.get("MANTIS_DISABLE_PASTE_TYPE", "") != "1"
        )
        if use_paste:
            try:
                # xclip stays alive as the X-selection owner until a paste
                # reads from it — using subprocess.run waits for that exit
                # and times out (run 030). Popen + close stdin lets xclip
                # background itself; -loops 1 makes it exit after one read.
                xclip_proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard", "-loops", "1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env,
                )
                xclip_proc.stdin.write(text.encode())
                xclip_proc.stdin.close()
                # Tiny pause so xclip is registered as selection owner
                # before xdotool requests the paste.
                time.sleep(0.05)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    env=self._env, timeout=5, check=True,
                    capture_output=True,
                )
                # xclip should self-exit after the paste consumed the
                # selection (-loops 1). Reap it; if it overstays, kill.
                try:
                    xclip_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    xclip_proc.terminate()
                    try:
                        xclip_proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        xclip_proc.kill()
                return
            except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "xclip paste failed (%s), falling back to xdotool type", exc,
                )

        # Fallback: xdotool type with configurable delay. The default 60ms
        # is what landed when paste was unavailable (run 027/029 era).
        delay_ms = int(os.environ.get("MANTIS_TYPE_DELAY_MS", "60"))
        if self._human_speed:
            for char in text:
                subprocess.run(
                    ["xdotool", "type", "--delay", str(random.randint(30, 120)), char],
                    env=self._env, capture_output=True, timeout=5,
                )
        else:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), text],
                env=self._env, capture_output=True, timeout=20,
            )

    # ── GymEnvironment interface ─────────────────────────────────────

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Start Xvfb + browser, navigate to URL."""
        url = kwargs.get("start_url", "")  # Only navigate if explicitly passed

        # Browser already running
        if self._browser_proc and self._browser_proc.poll() is None:
            if url and url != "about:blank":
                # Navigate to the specified URL
                logger.info(f"Navigating to {url[:60]}")
                self._xdotool("key", "ctrl+l")
                time.sleep(0.3)
                self._xdotool("key", "ctrl+a")
                time.sleep(0.2)
                self._xdotool_type(url)
                time.sleep(0.3)
                self._xdotool("key", "Return")
                time.sleep(self._settle_time + 2)
            else:
                # No URL — just capture current page state (for sub-plan micro-steps)
                logger.info("Reusing browser (no navigation)")
            return self._capture()

        # Fresh start
        if self._browser_proc:
            self.close()

        display = self._start_xvfb()
        self._env = {**os.environ, "DISPLAY": display}

        self._start_browser(url or self._start_url)
        time.sleep(self._settle_time + 2)

        return self._capture()

    def step(self, action: Action) -> GymResult:
        """Execute action via xdotool and return screenshot."""
        if self._human_speed:
            if action.action_type == ActionType.CLICK:
                time.sleep(random.uniform(0.3, 1.0))
            elif action.action_type == ActionType.TYPE:
                time.sleep(random.uniform(0.5, 1.5))

        self._execute_action(action)

        # Post-action settle
        if action.action_type not in (ActionType.WAIT, ActionType.DONE):
            settle = self._settle_time
            if self._human_speed:
                settle += random.uniform(0.5, 2.0)
            time.sleep(settle)

        obs = self._capture()
        return GymResult(observation=obs, reward=0.0, done=False, info={})

    def close(self) -> None:
        """Kill browser and Xvfb."""
        if self._browser_proc:
            self._browser_proc.terminate()
            try:
                self._browser_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._browser_proc.kill()
            self._browser_proc = None

        if self._xvfb_proc:
            self._xvfb_proc.terminate()
            self._xvfb_proc = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    @property
    def current_url(self) -> str:
        """Read the active tab's URL via Chrome DevTools Protocol.

        Issue #89 §1: the runner used to read the URL from the address-bar
        pixels through ClaudeExtractor.extract — fragile when the page is
        mid-render or the address bar is offscreen, surfacing as the
        well-known ``(url=)`` empty-string in click-verify logs.

        CDP's ``GET /json/list`` returns every Chrome tab; the active page
        is the first entry of type ``page`` whose ``url`` is non-empty.
        Bound to 127.0.0.1 only — never reachable from outside the container.

        Returns ``""`` if CDP is unreachable so the runner can fall back to
        screenshot extraction (preserves backward compatibility).
        """
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
        except Exception:
            return ""
        for tab in tabs:
            if tab.get("type") == "page" and tab.get("url"):
                url = str(tab["url"])
                # Filter chrome:// internal URLs — not what callers want.
                if url.startswith("chrome://") or url.startswith("about:"):
                    continue
                return url
        return ""

    def has_session(self, name: str) -> bool:
        return self._browser_proc is not None and self._browser_proc.poll() is None

    def save_session(self, name: str) -> None:
        pass

    def load_session(self, name: str) -> None:
        pass

    # ��─ Internal ──���──────────────────────��───────────────────────────

    def _capture(self) -> GymObservation:
        """Take screenshot and return as observation.

        If save_screenshots is set, saves each screenshot for replay testing.
        """
        screenshot = self._screenshot()

        if self._save_screenshots:
            from .replay_env import save_screenshot
            save_screenshot(screenshot, self._save_screenshots, self._step_counter)
            self._step_counter += 1

        return GymObservation(screenshot=screenshot, extras={})

    @staticmethod
    def _to_int(val) -> int:
        """Safely extract an integer from a possibly malformed value."""
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip()
        # Extract first number from strings like "143, 417]" or "0, y>\n0..."
        m = __import__("re").match(r'-?\d+', s)
        return int(m.group(0)) if m else 0

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to viewport bounds."""
        x, y = self._to_int(x), self._to_int(y)
        return max(0, min(x, self._viewport[0] - 1)), max(0, min(y, self._viewport[1] - 1))

    def _execute_action(self, action: Action) -> None:
        """Translate Mantis Action to xdotool commands."""
        match action.action_type:
            case ActionType.CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                button = action.params.get("button", "left")
                btn_num = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
                self._xdotool("mousemove", str(x), str(y))
                if self._human_speed:
                    time.sleep(random.uniform(0.05, 0.15))
                self._xdotool("click", btn_num)

            case ActionType.DOUBLE_CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                self._xdotool("mousemove", str(x), str(y))
                self._xdotool("click", "--repeat", "2", "1")

            case ActionType.TYPE:
                text = action.params.get("text") or action.params.get("content") or ""
                if not text:
                    logger.warning(f"type_text missing text: {action.params}")
                    return
                text = str(text)
                if text.startswith("http://") or text.startswith("https://"):
                    self._xdotool("key", "ctrl+l")
                    time.sleep(0.5)
                    self._xdotool("key", "ctrl+a")
                    time.sleep(0.2)
                    self._xdotool_type(text)
                    time.sleep(0.3)
                    self._xdotool("key", "Return")
                else:
                    self._xdotool_type(text)

            case ActionType.KEY_PRESS:
                keys = action.params.get("keys") or action.params.get("key") or ""
                if not keys:
                    logger.warning(f"key_press missing keys: {action.params}")
                    return
                keys = str(keys)
                key_map = {
                    "enter": "Return", "tab": "Tab", "escape": "Escape",
                    "backspace": "BackSpace", "delete": "Delete",
                    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
                    "home": "Home", "end": "End",
                    "pageup": "Page_Up", "pagedown": "Page_Down",
                    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
                    "f11": "F11", "f12": "F12",
                    "space": "space",
                }
                parts = keys.split("+")
                mapped = [key_map.get(p.strip().lower(), p.strip()) for p in parts]
                self._xdotool("key", "+".join(mapped))

            case ActionType.SCROLL:
                direction = action.params.get("direction", "down")
                amount = action.params.get("amount", 3)
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                self._xdotool("mousemove", str(x), str(y))
                btn = "4" if direction == "up" else "5"
                for _ in range(amount):
                    self._xdotool("click", btn)
                    if self._human_speed:
                        time.sleep(random.uniform(0.05, 0.15))

            case ActionType.DRAG:
                sx, sy = action.params.get("start_x", 0), action.params.get("start_y", 0)
                ex, ey = action.params["end_x"], action.params["end_y"]
                sx, sy = self._clamp(sx, sy)
                ex, ey = self._clamp(ex, ey)
                self._xdotool("mousemove", str(sx), str(sy))
                self._xdotool("mousedown", "1")
                time.sleep(0.1)
                self._xdotool("mousemove", str(ex), str(ey))
                self._xdotool("mouseup", "1")

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                time.sleep(min(seconds, 10.0))

            case ActionType.DONE:
                pass

            case ActionType.LAUNCH_APP:
                self._launch_app(action.params)

    # ── App launch (issue #72) ───────────────────────────────────────
    def _launch_app(self, params: dict[str, Any]) -> None:
        """Launch a desktop binary on the env's display.

        Errors are logged + swallowed so the runner sees the action as a
        no-op rather than crashing on bad params. The caller verifies launch
        success by checking the next screenshot (same contract as a click).
        """
        name = str(params.get("name") or "").strip()
        if not name:
            logger.warning("launch_app missing 'name': %s", params)
            return
        args = params.get("args") or []
        if not isinstance(args, (list, tuple)):
            logger.warning("launch_app args must be a list: %s", params)
            args = []
        extra_env = params.get("env") or {}
        if not isinstance(extra_env, dict):
            logger.warning("launch_app env must be a dict: %s", params)
            extra_env = {}

        # Compose env: env's DISPLAY + caller's overrides.
        proc_env = dict(self._env)
        proc_env.update({str(k): str(v) for k, v in extra_env.items()})

        try:
            subprocess.Popen(
                [name, *[str(a) for a in args]],
                env=proc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("launch_app: started %s args=%s", name, list(args)[:6])
        except FileNotFoundError:
            logger.error("launch_app: binary not found on PATH: %s", name)
        except Exception as exc:  # noqa: BLE001 — surface as no-op
            logger.error("launch_app: failed to start %s: %s", name, exc)
