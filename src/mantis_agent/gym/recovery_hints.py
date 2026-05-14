"""Per-step recovery hints — accumulate across failed attempts and
surface into the next attempt's prompt.

The hint protocol was introduced for form steps under issue #224's
follow-up: when the agentic recovery loop returns ``mode=add_hint``,
the hint string is appended to ``runner._recovery_hints[step_index]``
and the next form-handler attempt prepends every accumulated hint
into ``find_form_target``'s search prompt. Hints come from Claude
analysing the failure screenshot — much more specific than the
generic "avoid these coords" feedback the snapshot-diff path
produces.

This module generalizes the *consumption* side (epic #377 Phase A.3)
so any handler that builds a Claude prompt from ``step.intent`` can
splice the accumulated hints in with one helper call — no more
hand-rolling the same string-building boilerplate inside each
handler.

The producer side stays in :mod:`~.step_recovery` (which decides
when to call :func:`add_hint`).
"""

from __future__ import annotations

from typing import Any


_HINT_BLOCK_HEADER = "\n\nRECOVERY HINTS from previous failed attempts:\n"


def get_hint_block(runner: Any, step_index: int) -> str:
    """Return a multi-line hint block to splice into a handler's
    prompt, or ``""`` if no hints are stored for this step.

    Defensive on every access: tests / hosts that hand the runner a
    ``MagicMock`` auto-create the ``_recovery_hints`` attribute as a
    Mock, not a dict. Returning an empty string for any non-dict /
    non-list value keeps stubbed runners from leaking ``Mock``-stringified
    junk into a production prompt.
    """
    hints_dict = getattr(runner, "_recovery_hints", None)
    if not isinstance(hints_dict, dict):
        return ""
    stored = hints_dict.get(step_index, [])
    if not isinstance(stored, list):
        return ""
    hints = [str(h) for h in stored if h]
    if not hints:
        return ""
    return _HINT_BLOCK_HEADER + "\n".join(f"  - {h}" for h in hints)


def has_hints(runner: Any, step_index: int) -> bool:
    """Cheap presence check — useful when the caller wants to log
    ``"applying N recovery hint(s)"`` before splicing."""
    hints_dict = getattr(runner, "_recovery_hints", None)
    if not isinstance(hints_dict, dict):
        return False
    stored = hints_dict.get(step_index, [])
    return isinstance(stored, list) and bool([h for h in stored if h])


def count(runner: Any, step_index: int) -> int:
    """How many hints are accumulated. Same input validation."""
    hints_dict = getattr(runner, "_recovery_hints", None)
    if not isinstance(hints_dict, dict):
        return 0
    stored = hints_dict.get(step_index, [])
    if not isinstance(stored, list):
        return 0
    return sum(1 for h in stored if h)


__all__ = ["get_hint_block", "has_hints", "count"]
