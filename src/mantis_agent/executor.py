"""ActionExecutor — translates model actions into OS-level input events.

Handles the physical execution of actions on the computer. Includes safety
bounds checking and action history for debugging.

``pyautogui`` is in the ``[local-cua]`` extras and only needed at execute
time. The bare-package install (and CI) can import :class:`ExecutionResult`
and the module itself; instantiating :class:`ActionExecutor` is what
requires the dependency to be present.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .actions import Action, ActionType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import pyautogui as _pyautogui_t  # noqa: F401


def _load_pyautogui() -> Any:
    """Import pyautogui on demand and apply our default safety knobs.

    Raised import error includes a hint so users know which extras to install.
    """
    try:
        import pyautogui  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ActionExecutor requires pyautogui. Install with: pip install -e .[local-cua]"
        ) from exc
    pyautogui.PAUSE = 0.05
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    return pyautogui


@dataclass
class ExecutionResult:
    """Result of executing an action."""

    action: Action
    success: bool
    duration: float = 0.0
    error: str = ""


class ActionExecutor:
    """Executes structured actions via pyautogui.

    Args:
        screen_bounds: (width, height) of the screen. Actions outside bounds
                      are clipped with a warning.
        safe_mode: If True, adds delays between actions and logs everything.
        type_interval: Delay between keystrokes when typing (seconds).
    """

    def __init__(
        self,
        screen_bounds: tuple[int, int] = (1920, 1080),
        safe_mode: bool = True,
        type_interval: float = 0.02,
    ):
        self.screen_bounds = screen_bounds
        self.safe_mode = safe_mode
        self.type_interval = type_interval
        self.history: list[ExecutionResult] = []
        self._pg = _load_pyautogui()

    def execute(self, action: Action) -> ExecutionResult:
        """Execute a single action and return the result."""
        t0 = time.monotonic()

        if self.safe_mode:
            logger.info(f"Executing: {action}")

        try:
            match action.action_type:
                case ActionType.CLICK:
                    self._click(action)
                case ActionType.DOUBLE_CLICK:
                    self._double_click(action)
                case ActionType.TYPE:
                    self._type_text(action)
                case ActionType.KEY_PRESS:
                    self._key_press(action)
                case ActionType.SCROLL:
                    self._scroll(action)
                case ActionType.DRAG:
                    self._drag(action)
                case ActionType.WAIT:
                    self._wait(action)
                case ActionType.DONE:
                    pass  # Handled by the agent loop

            result = ExecutionResult(
                action=action, success=True, duration=time.monotonic() - t0
            )
        except Exception as e:
            logger.error(f"Action failed: {e}")
            result = ExecutionResult(
                action=action, success=False, duration=time.monotonic() - t0, error=str(e)
            )

        self.history.append(result)
        return result

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to screen bounds."""
        w, h = self.screen_bounds
        cx = max(0, min(x, w - 1))
        cy = max(0, min(y, h - 1))
        if cx != x or cy != y:
            logger.warning(f"Coordinates clamped: ({x},{y}) → ({cx},{cy})")
        return cx, cy

    def _click(self, action: Action) -> None:
        x, y = self._clamp(action.params["x"], action.params["y"])
        button = action.params.get("button", "left")
        self._pg.click(x, y, button=button)

    def _double_click(self, action: Action) -> None:
        x, y = self._clamp(action.params["x"], action.params["y"])
        self._pg.doubleClick(x, y)

    def _type_text(self, action: Action) -> None:
        text = action.params["text"]
        # Use write for ASCII, typewrite doesn't handle unicode well
        # For unicode, we use the platform clipboard as fallback
        try:
            self._pg.write(text, interval=self.type_interval)
        except Exception:
            # Fallback: use clipboard for non-ASCII text
            self._type_via_clipboard(text)

    def _type_via_clipboard(self, text: str) -> None:
        """Type text by pasting from clipboard (handles unicode)."""
        import subprocess

        if platform.system() == "Darwin":
            subprocess.run(
                ["pbcopy"], input=text.encode("utf-8"), check=True
            )
            self._pg.hotkey("command", "v")
        elif platform.system() == "Linux":
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
            )
            self._pg.hotkey("ctrl", "v")
        else:
            # Windows
            subprocess.run(
                ["clip"], input=text.encode("utf-16le"), check=True
            )
            self._pg.hotkey("ctrl", "v")

    def _key_press(self, action: Action) -> None:
        keys_str = action.params["keys"]
        # Handle combinations like "command+c", "ctrl+shift+a"
        parts = [k.strip() for k in keys_str.split("+")]
        if len(parts) == 1:
            self._pg.press(parts[0])
        else:
            self._pg.hotkey(*parts)

    def _scroll(self, action: Action) -> None:
        direction = action.params["direction"]
        amount = action.params.get("amount", 3)
        x = action.params.get("x")
        y = action.params.get("y")

        # self._pg.scroll: positive = up, negative = down
        scroll_map = {"up": amount, "down": -amount, "left": -amount, "right": amount}
        clicks = scroll_map.get(direction, 0)

        if x is not None and y is not None:
            x, y = self._clamp(x, y)
            if direction in ("left", "right"):
                self._pg.hscroll(clicks, x=x, y=y)
            else:
                self._pg.scroll(clicks, x=x, y=y)
        else:
            if direction in ("left", "right"):
                self._pg.hscroll(clicks)
            else:
                self._pg.scroll(clicks)

    def _drag(self, action: Action) -> None:
        sx, sy = self._clamp(action.params["start_x"], action.params["start_y"])
        ex, ey = self._clamp(action.params["end_x"], action.params["end_y"])
        self._pg.moveTo(sx, sy)
        self._pg.drag(ex - sx, ey - sy, duration=0.5)

    def _wait(self, action: Action) -> None:
        seconds = action.params.get("seconds", 1.0)
        time.sleep(min(seconds, 5.0))  # Cap wait at 5 seconds
