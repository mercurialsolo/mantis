"""XdotoolGymEnv — pure screen-level CUA environment using Xvfb + xdotool.

No automation framework. Just:
- Screenshots via scrot/mss (pixel capture)
- Clicks via xdotool (mouse events)
- Typing via xdotool (keyboard events)
- Browser launched as a regular process

The CUA model sees only screenshots. Actions are physical input events
injected at the X11 level — identical to a human using a mouse/keyboard.

Architecture:
    Xvfb (virtual display :99)
      └── Chrome/Firefox (regular process, not automated)
            ↕ X11 events
    xdotool → mouse_move, click, type
    scrot/mss → screenshot.png

    CUA Brain sees screenshot → outputs (x, y, text) → xdotool executes

Requirements:
    apt-get install xvfb xdotool scrot chromium-browser
    pip install mss pillow
"""

from __future__ import annotations

import io
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


class XdotoolGymEnv(GymEnvironment):
    """Pure screen-level environment — Xvfb + xdotool + scrot.

    No Playwright, no CDP, no DOM access. Just pixels and input events.

    Args:
        start_url: URL to open in the browser on reset.
        viewport: Screen size as (width, height).
        browser: Browser command ("chromium-browser", "firefox", "google-chrome").
        display: X11 display number (e.g., ":99"). If None, starts Xvfb.
        settle_time: Seconds to wait after actions.
        human_speed: Add realistic delays between actions.
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
    ):
        self._start_url = start_url
        self._viewport = viewport
        self._browser_cmd = browser
        self._display = display
        self._settle_time = settle_time
        self._human_speed = human_speed
        self._proxy_server = proxy_server  # e.g. "http://127.0.0.1:3128"

        self._xvfb_proc = None
        self._browser_proc = None
        self._env = {}

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
        """Launch browser as a regular process on the virtual display."""
        cmd = [
            self._browser_cmd,
            "--no-sandbox",
            "--disable-gpu",
            "--no-first-run",
            "--disable-default-apps",
            "--disable-infobars",
            f"--window-size={self._viewport[0]},{self._viewport[1]}",
            "--start-maximized",
        ]
        # Proxy support
        if self._proxy_server:
            cmd.append(f"--proxy-server={self._proxy_server}")
        cmd.append(url)

        self._browser_proc = subprocess.Popen(
            cmd, env=self._env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Wait for browser to open
        logger.info(f"Browser started: {self._browser_cmd} → {url}")

    def _screenshot(self) -> Image.Image:
        """Capture screenshot via mss (fast) or scrot (fallback)."""
        try:
            import mss
            with mss.mss(display=self._env.get("DISPLAY", ":99")) as sct:
                monitor = sct.monitors[0]  # Full screen
                img = sct.grab(monitor)
                return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        except Exception:
            pass

        # Fallback: scrot
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

    def _xdotool(self, *args: str) -> None:
        """Run xdotool command."""
        subprocess.run(
            ["xdotool"] + list(args),
            env=self._env, capture_output=True, timeout=5,
        )

    def _xdotool_type(self, text: str) -> None:
        """Type text via xdotool with optional human-like delays."""
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
        url = kwargs.get("start_url", self._start_url)

        # Browser already running — navigate via address bar (no restart)
        if self._browser_proc and self._browser_proc.poll() is None:
            logger.info("Reusing existing browser (xdotool navigate)")
            if url and url != "about:blank":
                # Navigate: Ctrl+L → select all → type URL → Enter
                self._xdotool("key", "ctrl+l")
                time.sleep(0.3)
                self._xdotool("key", "ctrl+a")
                time.sleep(0.2)
                self._xdotool_type(url)
                time.sleep(0.3)
                self._xdotool("key", "Return")
                time.sleep(self._settle_time + 2)
            return self._capture()

        # Fresh start
        if self._browser_proc:
            self.close()

        display = self._start_xvfb()
        self._env = {**os.environ, "DISPLAY": display}

        self._start_browser(url)
        time.sleep(self._settle_time + 2)

        return self._capture()

    def step(self, action: Action) -> GymResult:
        """Execute action via xdotool and return screenshot."""
        # Human-like pre-action delay
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
        # Can't read URL without DOM — return empty
        # The model reads it from the address bar in the screenshot
        return ""

    # ── Internal ─────────────────────────────────────────────────────

    def _capture(self) -> GymObservation:
        """Take screenshot and return as observation."""
        screenshot = self._screenshot()
        return GymObservation(screenshot=screenshot, extras={})

    def _execute_action(self, action: Action) -> None:
        """Translate Mantis Action to xdotool commands."""
        match action.action_type:
            case ActionType.CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                button = action.params.get("button", "left")
                btn_num = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
                # Move mouse smoothly, then click
                if self._human_speed:
                    self._xdotool("mousemove", "--sync", str(x), str(y))
                    time.sleep(random.uniform(0.05, 0.15))
                else:
                    self._xdotool("mousemove", str(x), str(y))
                self._xdotool("click", btn_num)

            case ActionType.DOUBLE_CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                self._xdotool("mousemove", str(x), str(y))
                self._xdotool("click", "--repeat", "2", "1")

            case ActionType.TYPE:
                text = action.params["text"]
                if text.startswith("http://") or text.startswith("https://"):
                    # URL: focus address bar first, then type
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
                keys = action.params["keys"]
                # Normalize to xdotool format: ctrl+c → ctrl+c (same)
                # But xdotool uses "Return" not "enter"
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
                mapped = []
                for p in parts:
                    p_lower = p.strip().lower()
                    mapped.append(key_map.get(p_lower, p.strip()))
                self._xdotool("key", "+".join(mapped))

            case ActionType.SCROLL:
                direction = action.params["direction"]
                amount = action.params.get("amount", 3)
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                self._xdotool("mousemove", str(x), str(y))
                # xdotool: button 4=scroll up, 5=scroll down
                btn = "4" if direction == "up" else "5"
                for _ in range(amount):
                    self._xdotool("click", btn)
                    if self._human_speed:
                        time.sleep(random.uniform(0.05, 0.15))

            case ActionType.DRAG:
                sx, sy = action.params["start_x"], action.params["start_y"]
                ex, ey = action.params["end_x"], action.params["end_y"]
                self._xdotool("mousemove", str(sx), str(sy))
                self._xdotool("mousedown", "1")
                time.sleep(0.1)
                self._xdotool("mousemove", "--sync", str(ex), str(ey))
                self._xdotool("mouseup", "1")

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                time.sleep(min(seconds, 10.0))

            case ActionType.DONE:
                pass
