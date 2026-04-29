"""Reusable reward primitives.

These are pure functions — given an action / info / state slice, return a
single named contribution to the reward. `RewardFn` implementations compose
them with weights tuned per environment. None of these call out over the
network: they read whatever the adapter has already populated in
`gym_result.info`.

Signals available today (any may be missing, depending on adapter):
  info["url"]              — current page URL (playwright)
  info["title"]            — current page title
  info["backtracked"]      — adapter detected off-site nav and rolled back
  info["warning"]          — human-readable reason for backtrack
  info["focused_input"]    — dict describing currently focused form field
  info["type_verified"]    — {"success": bool, "field": str, "reason": str}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..actions import Action


VALID_ACTION_NAMES = {
    "click", "double_click", "type_text",
    "key_press", "scroll", "drag", "wait", "done",
}


def format_reward(action: "Action", value: float = 0.1) -> float:
    """+`value` if the action is well-formed, else 0.

    "Well-formed" = an enum action_type the executor knows about, with the
    minimum required params present (e.g. CLICK has x and y). This is the
    cheapest possible shaping term and discourages garbage tool calls.
    """
    from ..actions import ActionType

    if action.action_type.value not in VALID_ACTION_NAMES:
        return 0.0

    required = {
        ActionType.CLICK: ("x", "y"),
        ActionType.DOUBLE_CLICK: ("x", "y"),
        ActionType.TYPE: ("text",),
        ActionType.KEY_PRESS: ("keys",),
        ActionType.SCROLL: ("direction",),
        ActionType.DRAG: ("start_x", "start_y", "end_x", "end_y"),
    }.get(action.action_type, ())

    if not all(k in action.params for k in required):
        return 0.0
    return value


def off_site_penalty(
    info: dict[str, Any],
    allowed_domains: tuple[str, ...] = (),
    penalty: float = -0.5,
) -> float:
    """Negative reward when the adapter signals an off-site navigation.

    Two trigger paths:
      1. `info["backtracked"]` is true (adapter explicitly rolled back).
      2. `allowed_domains` is non-empty and `info["url"]` host is not in it.

    Use case: prevent the policy from clicking social-media icons or external
    links in extraction tasks. Without an explicit penalty, GRPO finds these
    as a way to "escape" hard pages and short-circuit the episode.
    """
    if info.get("backtracked"):
        return penalty
    if allowed_domains:
        url = info.get("url", "")
        if url:
            host = urlparse(url).netloc.lower()
            if host and not any(host.endswith(d.lower()) for d in allowed_domains):
                return penalty
    return 0.0


def loop_penalty(
    action_history: list["Action"],
    window: int = 3,
    penalty: float = -0.2,
) -> float:
    """Negative reward when the last `window` actions are identical.

    Mirrors `GymRunner._detect_repeat`. Identical = same action_type AND same
    params dict. Discourages the policy from spamming the same click/scroll
    when nothing changes.
    """
    if len(action_history) < window:
        return 0.0
    recent = action_history[-window:]
    first = recent[0]
    same = all(
        a.action_type == first.action_type and a.params == first.params
        for a in recent[1:]
    )
    return penalty if same else 0.0


def type_verified_reward(info: dict[str, Any], value: float = 0.1) -> float:
    """+`value` if a TYPE action was verified to land in a real field."""
    tv = info.get("type_verified")
    if tv and tv.get("success"):
        return value
    return 0.0


def url_progress_reward(
    info: dict[str, Any],
    last_url: str,
    value: float = 0.05,
) -> float:
    """+`value` when the URL changes (any forward navigation).

    Cheap proxy for "something happened" — useful when a domain-specific
    success signal is too sparse and we need shaping to bootstrap learning.
    Watch for hacking: the policy may navigate aimlessly to farm this.
    Pair with off_site_penalty.
    """
    new_url = info.get("url", "")
    if new_url and new_url != last_url:
        return value
    return 0.0


def task_success_reward(
    summary: str | None,
    success: bool,
    value: float = 1.0,
    failure_value: float = 0.0,
) -> float:
    """Terminal reward from a `done(success=..., summary=...)` action.

    Trusts the policy's own self-report. For tasks with an external grader
    (OSWorld, VWA, BoatTrader gate), prefer those over this.
    """
    if success:
        return value
    return failure_value
