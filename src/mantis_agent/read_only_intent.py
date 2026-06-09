"""Detect read-only / no-click intent in plan_text (#831).

User feedback on HN extraction surfaced the gap: even with prompts
like *"do not click story links / More"*, the decomposer was still
emitting ``click`` / ``paginate`` / ``loop`` steps, and the runner
duly clicked the "More" link, landed on a pagination URL, and
returned ``?p=2`` as the extracted URL.

The leverage isn't fixing the click — it's *not emitting it in the
first place*. This module pulls the read-only signal out of the
plan_text as a small regex-vocabulary helper. When the signal fires,
the decomposer:

1. Prepends a hard ``READ-ONLY MODE`` constraint to its system
   prompt so the model emits ``navigate + extract`` only.
2. Validates the decomposed plan post-Claude — if any ``click`` /
   ``paginate`` / ``loop`` step survived, the decomposer raises so
   the run fails fast instead of misbehaving.

Operators / SDK callers can also set the explicit ``read_only: true``
flag on the request body, which bypasses the regex detection (always
on regardless of phrasing) — useful when the prose itself doesn't
include a negative phrasing but the workflow is read-only by design
(e.g. a daily front-page snapshot).

Detection vocabulary leans **inclusive** rather than precise: we'd
rather have one false positive (run as read-only when the user wanted
clicks) than a false negative (return pagination URLs when the user
explicitly forbade them). Add new phrasings here when feedback
surfaces them.
"""

from __future__ import annotations

import re

# Compiled patterns. Each one is a "this plan is read-only" signal.
# Lowercased before matching so the patterns themselves are
# case-insensitive without the ``re.IGNORECASE`` cost on every call.
_READ_ONLY_PATTERNS = tuple(
    re.compile(p) for p in (
        # Explicit "do not click / navigate"
        r"\bdo not click\b",
        r"\bdon'?t click\b",
        r"\bdo not navigate\b",
        r"\bdon'?t navigate\b",
        r"\bdo not follow\b",
        r"\bdon'?t follow\b",
        r"\bdo not paginate\b",
        r"\bdon'?t paginate\b",
        # Positive read-only intent
        r"\bread[- ]only\b",
        r"\bstay on (?:this|the) page\b",
        r"\bremain on (?:this|the) page\b",
        r"\bjust read\b",
        r"\bread (?:and|only)\b",
        # No-link / inline phrasing
        r"\bwithout clicking\b",
        r"\bwithout navigating\b",
        r"\bwithout opening\b",
        r"\bwithout following\b",
        # Visible-only phrasing
        r"\bvisible on the (?:current )?page\b",
        r"\bshown on the (?:current )?page\b",
        r"\binline\b.*\bonly\b",
    )
)


def is_read_only(plan_text: str) -> bool:
    """Return True when the plan_text expresses read-only intent.

    Pattern set is intentionally permissive — false-positive cost
    (running as read-only when clicks would be useful) is low; false-
    negative cost (clicking when forbidden, returning garbage URLs)
    is high.
    """
    if not plan_text:
        return False
    haystack = plan_text.lower()
    return any(p.search(haystack) for p in _READ_ONLY_PATTERNS)


# Step types that mutate page state / navigate / open new contexts.
# Read-only plans MUST NOT contain any of these — that's the post-
# decomposition validation contract.
_FORBIDDEN_READ_ONLY_STEP_TYPES: frozenset[str] = frozenset({
    "click", "paginate", "loop", "submit", "fill_field",
    "select_option", "right_click",
})


READ_ONLY_PROMPT_CONSTRAINT = """\
READ-ONLY MODE — HARD CONSTRAINT (#831):

The source plan EXPLICITLY asked for read-only extraction (no clicks,
no pagination, no navigation beyond the initial URL). Your decomposed
plan MUST satisfy:

- ONLY these step types are allowed: navigate, extract_data, extract_url,
  detect_visible, scroll (scrolling within the same page is fine —
  it does not navigate away). For a typical read-only plan the shape
  is just: navigate → extract_data (or extract_url).
- FORBIDDEN step types in this mode: click, paginate, loop, submit,
  fill_field, select_option, right_click.
- The extract step MUST use ``claude_only: true`` (vision-only read,
  no Holo3 click loop).
- The extract step's ``extract`` block SHOULD set ``max_items`` to N
  when the source asks for top-N rows; the multi-row branch returns
  all N in one Claude call without per-row navigation.

If the source plan is genuinely ambiguous (e.g. "look at the top stories"
without an explicit "do not click"), still emit only navigate + extract.
A reader expects to be on one page and read it — anything more is a
regression in this mode.
"""


def validate_read_only_plan(steps: list, *, plan_text: str = "") -> str:
    """Return an empty string when ``steps`` satisfies read-only contract,
    or a non-empty error message when it doesn't.

    Caller raises / rejects on a non-empty return. Pure function — no
    side effects.

    ``steps`` is a list of dicts (decomposed JSON) or :class:`MicroIntent`
    instances. Both shapes carry a ``type`` field.
    """
    bad: list[tuple[int, str]] = []
    for i, step in enumerate(steps):
        step_type = (
            step.get("type", "") if isinstance(step, dict)
            else getattr(step, "type", "")
        )
        if step_type in _FORBIDDEN_READ_ONLY_STEP_TYPES:
            bad.append((i, str(step_type)))
    if not bad:
        return ""
    parts = [f"step {i} ({t!r})" for i, t in bad]
    return (
        f"plan_text expressed read-only intent but the decomposed plan "
        f"contains forbidden step type(s): {', '.join(parts)}. "
        f"Source: {plan_text[:100]!r}"
    )


__all__ = [
    "READ_ONLY_PROMPT_CONSTRAINT",
    "is_read_only",
    "validate_read_only_plan",
]
