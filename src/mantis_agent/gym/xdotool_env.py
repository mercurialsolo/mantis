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

    def _xdotool_type(self, text: str) -> None:
        """Type text via xdotool."""
        if self._human_speed:
            for char in text:
                subprocess.run(
                    ["xdotool", "type", "--delay", str(random.randint(30, 120)), char],
                    env=self._env, capture_output=True, timeout=5,
                )
        else:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", text],
                env=self._env, capture_output=True, timeout=10,
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
        # No CDP — model reads URL from the address bar in the screenshot
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
