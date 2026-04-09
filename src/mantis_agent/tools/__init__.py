"""Tools provided to the Mantis agent as Python helpers.

The agent's prompt describes high-level helpers (click, type_text, key, ...)
that hide pyautogui/xdotool implementation details. The helpers are defined as
a Python source string in :mod:`helpers` and prepended to every action the
model emits before execution.

This means the model never has to know about pyautogui's quirks (special-char
mangling, character-by-character typing, coordinate scaling) — those concerns
live in one place where they can be tested and fixed independently of any
prompt.
"""

from __future__ import annotations

from .helpers import HELPERS_PRELUDE, wrap_action

__all__ = ["HELPERS_PRELUDE", "wrap_action"]
