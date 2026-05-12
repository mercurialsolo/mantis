"""Predicate grammar + evaluators for #291.

Brains emit structured predictions either as a JSON block:

    {"expected": ["url_contains:/checkout", "title_changed", "field_focused:email"]}

or as a back-compat ``Predicted: <prose>`` line (the #120 Holo3 surface).
The runner parses the predictions, evaluates each predicate against the
post-action observation, writes per-predicate booleans into the trajectory
step, and derives a ``world_model_error`` reward contribution (fraction
wrong of evaluated predicates).

Predicate grammar — deterministic, observable from gym signals today
(``gym_result.info`` plus a perceptual frame hash):

  url_contains:<substr>       observed URL contains substr
  url_equals:<url>            observed URL == url exactly
  url_changed                 URL differs from previous step
  url_unchanged               URL identical to previous step
  title_contains:<substr>     page title contains substr
  title_changed               title differs from previous step
  field_focused[:<name>]      a field is focused (and matches name if given)
  field_unfocused             no field is focused
  frame_changed               frame hash differs from previous step
  frame_stable                frame hash identical to previous step

Best-effort kinds — recognised by the grammar so the brain can emit them
and trajectories round-trip the predictions, but evaluators return ``None``
("not measured") on any adapter that doesn't expose the DOM/OCR signal:

  element_appears:<text>
  element_disappears:<text>
  modal_opens
  modal_closes

A predicate evaluating to ``None`` is excluded from the accuracy
denominator — it didn't fail, it just wasn't measurable. That keeps the
``world_model_error`` metric stable across adapter capability levels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Predicate:
    """One parsed prediction token, e.g. ``url_contains:/checkout``."""

    kind: str
    arg: str | None = None
    raw: str = ""

    def __str__(self) -> str:
        if self.raw:
            return self.raw
        if self.arg is not None:
            return f"{self.kind}:{self.arg}"
        return self.kind


@dataclass
class PredicateResult:
    """Outcome of evaluating one predicate. ``result is None`` => unevaluable."""

    predicate: str
    result: bool | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "result": self.result,
            "reason": self.reason,
        }


@dataclass
class ObservationContext:
    """Signals the runner hands to predicate evaluators after each step."""

    url: str = ""
    title: str = ""
    focused_input: dict | None = None
    frame_hash: str = ""
    prev_url: str = ""
    prev_title: str = ""
    prev_frame_hash: str = ""


# Recognised predicate kinds — anything else is dropped during parsing so
# free-form prose ("clicking the link") doesn't pollute the predicate stream.
_KNOWN_KINDS: frozenset[str] = frozenset({
    "url_contains", "url_equals", "url_changed", "url_unchanged",
    "title_contains", "title_changed",
    "field_focused", "field_unfocused",
    "frame_changed", "frame_stable",
    # Best-effort kinds — evaluators return None when no DOM/OCR signal is
    # available. Listed so trajectories round-trip them and adopting envs
    # can light them up later without a grammar change.
    "element_appears", "element_disappears",
    "modal_opens", "modal_closes",
})


# ``{"expected": [...]}`` may live inside a markdown fence; the regex
# tolerates the fence prefix and matches the first such object.
_JSON_BLOCK_RE = re.compile(
    r'\{[^{}]*"expected"\s*:\s*\[[^\]]*\][^{}]*\}',
    re.DOTALL,
)

_PREDICTED_LINE_RE = re.compile(
    r"^\s*Predicted\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_FREEFORM_DELIMS_RE = re.compile(r"[,;]+")


# Cap to keep trajectories bounded — matches brain_holo3's 240-char cap on
# the back-compat free-form line, but lifted to 500 to fit a small JSON
# block of ~5 predicates.
_PREDICTED_OUTCOME_MAX_CHARS: int = 500


def extract_predicted_outcome(text: str) -> str:
    """Pull a brain's predicted-outcome string out of its raw response.

    Looks for, in order:
    1. A ``{"expected": [...]}`` JSON block (the #291 structured contract).
    2. A trailing ``Predicted: <prose>`` line (the #120 Holo3 surface).

    Returns ``""`` when neither form is present so trajectories from brains
    that don't emit predictions stay clean (no spurious empty predicates).

    The result is the literal text the brain emitted (capped at
    :data:`_PREDICTED_OUTCOME_MAX_CHARS`) — :func:`parse_predicates` does
    the actual grammar work downstream. Keeping the raw form lets traces
    round-trip what the model actually said for distillation.
    """
    if not text:
        return ""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(0)[:_PREDICTED_OUTCOME_MAX_CHARS]
    pm = _PREDICTED_LINE_RE.search(text)
    if pm:
        line = pm.group(1).strip().strip('"').strip("'").strip()
        return line[:_PREDICTED_OUTCOME_MAX_CHARS]
    return ""


def _coerce_predicate(token: str) -> Predicate | None:
    """Turn ``'url_contains:/checkout'`` into a :class:`Predicate`.

    Returns ``None`` when the token's kind isn't in :data:`_KNOWN_KINDS` —
    that's what lets ``parse_predicates`` accept free-form prose alongside
    structured tokens (everything that isn't a real predicate is dropped).
    """
    token = token.strip().strip('"').strip("'").rstrip(".")
    if not token:
        return None
    if ":" in token:
        kind, _, arg = token.partition(":")
        kind = kind.strip().lower()
        arg = arg.strip()
        if kind in _KNOWN_KINDS:
            raw = f"{kind}:{arg}" if arg else kind
            return Predicate(kind=kind, arg=arg or None, raw=raw)
        return None
    kind = token.strip().lower()
    if kind in _KNOWN_KINDS:
        return Predicate(kind=kind, arg=None, raw=kind)
    return None


def parse_predicates(text: str) -> list[Predicate]:
    """Extract a predicate list from a brain's ``predicted_outcome`` string.

    Two surface forms are supported:

    1. **Structured JSON** — the #291 contract::

           {"expected": ["url_contains:/checkout", "title_changed"]}

    2. **Back-compat free-form** — the #120 Holo3 surface::

           Predicted: clicking will navigate to /checkout, title changes.

       Free-form prose is split on commas/semicolons and each token is
       matched against the grammar. Tokens that don't match a known kind
       are silently dropped, which lets brains adopt structured emissions
       incrementally without breaking existing prompts.

    Returns ``[]`` when nothing parses — the caller treats that as
    "no prediction this step" (no reward contribution).
    """
    if not text:
        return []
    out: list[Predicate] = []

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            for tok in obj.get("expected", []) or []:
                p = _coerce_predicate(str(tok))
                if p is not None:
                    out.append(p)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        if out:
            return out

    pm = _PREDICTED_LINE_RE.search(text)
    if pm:
        line = pm.group(1)
        for tok in _FREEFORM_DELIMS_RE.split(line):
            p = _coerce_predicate(tok)
            if p is not None:
                out.append(p)

    return out


def _focus_matches(focused: dict | None, name: str) -> bool:
    """``focused_input`` matches ``name`` if any identifying field contains
    the substring (case-insensitive).

    Mantis adapters populate ``focused_input`` with varying keys depending on
    available DOM access (id, name, label, selector, placeholder); checking
    each one keeps the predicate robust across env capability levels.
    """
    if not focused or not isinstance(focused, dict):
        return False
    needle = name.lower().lstrip("#.")
    for k in ("name", "id", "label", "selector", "placeholder"):
        v = focused.get(k)
        if v and needle in str(v).lower():
            return True
    return False


def evaluate_predicate(p: Predicate, ctx: ObservationContext) -> PredicateResult:
    """Evaluate one predicate against the post-action observation.

    Returns ``result=None`` when the env didn't expose the signal needed
    (e.g. ``element_appears`` on a screenshot-only adapter, or
    ``url_changed`` when the env doesn't surface URLs at all).
    """
    raw = p.raw or str(p)
    kind = p.kind
    arg = p.arg

    if kind == "url_contains":
        if not arg:
            return PredicateResult(raw, False, "url_contains requires arg")
        if not ctx.url:
            return PredicateResult(raw, None, "no url in observation")
        return PredicateResult(raw, arg in ctx.url, ctx.url)

    if kind == "url_equals":
        if not arg:
            return PredicateResult(raw, False, "url_equals requires arg")
        if not ctx.url:
            return PredicateResult(raw, None, "no url in observation")
        return PredicateResult(raw, ctx.url == arg, ctx.url)

    if kind == "url_changed":
        if not ctx.url and not ctx.prev_url:
            return PredicateResult(raw, None, "no url before/after")
        return PredicateResult(
            raw, ctx.url != ctx.prev_url, f"{ctx.prev_url} -> {ctx.url}",
        )

    if kind == "url_unchanged":
        if not ctx.url and not ctx.prev_url:
            return PredicateResult(raw, None, "no url before/after")
        return PredicateResult(
            raw, ctx.url == ctx.prev_url, f"{ctx.prev_url} -> {ctx.url}",
        )

    if kind == "title_contains":
        if not arg:
            return PredicateResult(raw, False, "title_contains requires arg")
        if not ctx.title:
            return PredicateResult(raw, None, "no title in observation")
        return PredicateResult(raw, arg in ctx.title, ctx.title)

    if kind == "title_changed":
        if not ctx.title and not ctx.prev_title:
            return PredicateResult(raw, None, "no title before/after")
        return PredicateResult(
            raw, ctx.title != ctx.prev_title, f"{ctx.prev_title} -> {ctx.title}",
        )

    if kind == "field_focused":
        if ctx.focused_input is None:
            return PredicateResult(raw, False, "no focused field")
        if arg:
            return PredicateResult(
                raw, _focus_matches(ctx.focused_input, arg), str(ctx.focused_input),
            )
        return PredicateResult(raw, True, str(ctx.focused_input))

    if kind == "field_unfocused":
        return PredicateResult(raw, ctx.focused_input is None, str(ctx.focused_input))

    if kind == "frame_changed":
        if not ctx.frame_hash or not ctx.prev_frame_hash:
            return PredicateResult(raw, None, "no frame hash before/after")
        return PredicateResult(
            raw,
            ctx.frame_hash != ctx.prev_frame_hash,
            f"{ctx.prev_frame_hash} -> {ctx.frame_hash}",
        )

    if kind == "frame_stable":
        if not ctx.frame_hash or not ctx.prev_frame_hash:
            return PredicateResult(raw, None, "no frame hash before/after")
        return PredicateResult(
            raw,
            ctx.frame_hash == ctx.prev_frame_hash,
            f"{ctx.prev_frame_hash} -> {ctx.frame_hash}",
        )

    if kind in {"element_appears", "element_disappears", "modal_opens", "modal_closes"}:
        # No DOM/OCR bridge in observed_state today — skip rather than fail.
        return PredicateResult(raw, None, "no DOM/OCR signal")

    return PredicateResult(raw, None, f"unknown predicate kind: {kind}")


def evaluate_all(
    predicates: list[Predicate], ctx: ObservationContext,
) -> list[PredicateResult]:
    return [evaluate_predicate(p, ctx) for p in predicates]


def world_model_error(results: list[PredicateResult]) -> float | None:
    """Fraction of evaluated predicates the brain got wrong, in ``[0.0, 1.0]``.

    Returns ``None`` when no predicates were evaluable — the caller should
    skip the reward contribution for that step (don't conflate "no signal"
    with "perfect prediction"). Lower is better.
    """
    evaluated = [r for r in results if r.result is not None]
    if not evaluated:
        return None
    wrong = sum(1 for r in evaluated if not r.result)
    return wrong / len(evaluated)
