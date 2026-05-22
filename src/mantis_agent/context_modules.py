"""Context-aware composition of brain prompts + recovery hints.

Each :class:`ContextModule` declares when it applies (a predicate over a
small ``ctx`` dict) and what it contributes:

* ``prompt_section`` — text spliced into the brain's system prompt for
  this dispatch (e.g. BLOCKING UI guidance, SCROLLING guidance).
* ``step_hint`` — text appended to the per-step recovery-hint block the
  brain's task prompt picks up via ``recovery_hints.get_hint_block``.
* ``handler_hints`` — structured key/value pairs handlers can read
  before dispatching (e.g. ``prefer_keyboard=True`` for the scroll
  handler to skip click-based scrolling).

The registry is iterated on every brain dispatch. Adding a behaviour
for a newly observed failure pattern means adding ONE module — no
edits to the monolithic system prompt file, no per-handler if/else
chains.

Context shape (all keys optional, modules degrade gracefully on miss):
* ``step_type``     — the MicroIntent.type for the dispatching step
* ``step_section``  — setup / extraction / pagination
* ``step_intent``   — the intent prose (sliced for predicate matches)
* ``failure_class`` — the last attempt's failure_class (for reactive
                       modules that should only fire after a failure)
* ``url``           — current page URL (for site-specific modules)

Modules use the context var :data:`current_context` to read what the
dispatching caller pushed via :func:`push_step_context` — no signature
plumbing through GymRunner / brain layers is required.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


# ── Context plumbing (context-var, no signature changes needed) ───────


_CURRENT_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("mantis_step_context", default=None)
)


@contextlib.contextmanager
def push_step_context(ctx: dict[str, Any]) -> Iterator[None]:
    """Publish ``ctx`` as the active dispatch context for the duration
    of the ``with`` block. Nested pushes are supported; the prior value
    restores on exit.
    """
    token = _CURRENT_CONTEXT.set(dict(ctx))
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_context() -> dict[str, Any]:
    """Return the active dispatch context (or an empty dict if none
    has been pushed). Always returns a fresh shallow copy so callers
    can mutate freely without affecting the var.
    """
    val = _CURRENT_CONTEXT.get()
    return dict(val) if val else {}


# ── Module shape ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextModule:
    """One context-aware contribution to brain prompts / hints / handlers.

    All fields are immutable so modules can be safely shared across
    threads / coroutines without copying.
    """

    name: str
    applies_when: Callable[[dict[str, Any]], bool]
    prompt_section: str = ""
    step_hint: str = ""
    handler_hints: dict[str, Any] = field(default_factory=dict)
    description: str = ""  # human-readable; surfaced in logs


# ── Predicates ────────────────────────────────────────────────────────


def _always(_ctx: dict[str, Any]) -> bool:
    return True


def _is_scroll(ctx: dict[str, Any]) -> bool:
    return str(ctx.get("step_type") or "") == "scroll"


def _is_extraction_scroll(ctx: dict[str, Any]) -> bool:
    return (
        str(ctx.get("step_type") or "") == "scroll"
        and str(ctx.get("step_section") or "") == "extraction"
    )


def _is_extraction_phase(ctx: dict[str, Any]) -> bool:
    """Steps where blocking overlays actually matter — extraction
    primitives reading from detail pages, plus the scroll that
    precedes them. Excludes navigate / setup / pagination steps
    where the brain doesn't need overlay-scanning deliberation
    (those pages haven't loaded extractable content yet).

    Tuned after observing that always-on BLOCKING_OVERLAY makes the
    brain ~3× slower per step (extra deliberation on overlay-scan
    for every action), regressing total step throughput. Scoping to
    extraction-flavor steps keeps the overlay-dismissal benefit
    where it pays (carousel-on-detail-page) without taxing
    navigate / paginate / loop steps.
    """
    step_type = str(ctx.get("step_type") or "")
    step_section = str(ctx.get("step_section") or "")
    if step_section == "extraction":
        return True
    if step_type in ("extract_data", "extract_url"):
        return True
    return False


# ── Module content ────────────────────────────────────────────────────


_BLOCKING_OVERLAY_SECTION = """\
BLOCKING UI — CRITICAL (applies to EVERY action, not just on retry):
Before each action, scan the viewport for elements that BLOCK the planned content. Common shapes (site-agnostic):
- Fullscreen image / media lightbox — full-viewport image with a "N of M" caption near the top and an "X" / "Close" button at top-right. Opens when you click a hero / thumbnail / gallery image. Blocks all underlying content. Cannot be scrolled past.
- Cookie / GDPR consent banner — bottom or top of viewport, "Accept" / "Reject" / "Manage" buttons, often with a dark scrim across the page.
- Newsletter / email signup modal — centered popup with input + "Subscribe" / "No thanks".
- "Sign in to continue" / paywall / login wall — centered modal blocking the underlying content; the URL DID load the right page but the overlay covers it.
- Notification toast / chat widget — small overlay at a corner; only blocks if it covers the target action area.

When you see any of these, DISMISS IT FIRST, then proceed with the planned action:
1. Try key_press(keys="Escape") — works for most modals and lightboxes.
2. If Escape doesn't work, look for "Close" / "X" / "Dismiss" / "No thanks" / "Reject all" / "Got it" / "Continue" and click that.
3. If the overlay is a content-blocking ad with no close affordance, scroll past it ONLY if scrolling visibly moves the underlying page.
4. NEVER click into the overlay's interactive content unless the plan explicitly directs it.

After dismissal, the viewport should show the underlying content the plan expects. If the same overlay reappears repeatedly, mention it in your reasoning so recovery can route to a different strategy.\
"""


_KEYBOARD_SCROLL_SECTION = """\
SCROLLING — CRITICAL (this step is a scroll action):
- Use scroll(direction="down", amount=10) to advance through long pages in one action. Larger amounts cover more of the page per step and conserve your action budget — prefer amount=10+ on tall pages.
- key_press(keys="Page_Down") / key_press(keys="End") are good alternatives when scroll() doesn't move the page (sticky-positioned content, scroll-locked overlays).
- NEVER click on images, photos, thumbnails, video previews, or any media element during a scroll step. Many sites turn the hero image into a fullscreen carousel trigger; one stray click opens a "N of M" lightbox overlay that blocks all underlying content and is hard to recover from. If you need to scroll past an image area, use scroll() / Page_Down — don't click.
- NEVER click on within-page navigation links (Next / Previous / "go to next item") during a scroll step — they navigate to a different page, breaking the workflow position the plan expects to scroll within.\
"""


# ── Module registry ───────────────────────────────────────────────────


BLOCKING_OVERLAY = ContextModule(
    name="blocking_overlay",
    applies_when=_is_extraction_phase,
    prompt_section=_BLOCKING_OVERLAY_SECTION,
    handler_hints={"dismiss_overlay_first": True},
    description=(
        "Active on extraction-phase steps (extract_data / extract_url / "
        "any step in the extraction section): instructs the brain to "
        "scan for and dismiss blocking overlays (carousels, cookie "
        "banners, paywalls, signup modals) before acting. Scoped to "
        "extraction because that's where overlays actually block the "
        "target content; running it always-on costs ~3× per-step "
        "deliberation time on navigate / setup / pagination steps "
        "where overlay-scanning is wasted budget."
    ),
)


KEYBOARD_SCROLL = ContextModule(
    name="keyboard_scroll",
    applies_when=_is_scroll,
    prompt_section=_KEYBOARD_SCROLL_SECTION,
    handler_hints={"avoid_image_click": True},
    description=(
        "Active on scroll steps: directs the brain to use scroll() "
        "with large amounts (covers tall pages in few actions) and "
        "forbids clicking images / in-page nav links during scrolling. "
        "Earlier version mandated keyboard-only, but Page_Down × 6 "
        "exhausted the per-step brain budget on tall pages — run "
        "20260522_173945_263f2bef halted at brain_loop_exhausted on "
        "the scroll step under that policy. Now: scroll() preferred, "
        "Page_Down as fallback, no image clicks."
    ),
)


# FORM FILLING and NAVIGATION guidance stays in the base
# ``holo3_system.txt`` prompt (always-on) because the brain depends on
# them for routine click / form interactions across most plans. Earlier
# attempt to move them into per-step modules regressed throughput
# significantly (runs 20260522_160850_4484731f and _162958_370b9a1a
# showed the brain making worse decisions on click / navigate steps
# without those defaults). Only NEW guidance — BLOCKING_OVERLAY
# (carousel/modal class) and KEYBOARD_SCROLL (image-misclick defense
# during scroll-to-read) — lives in the module registry. Future
# additions should follow the same rule: new behaviour as modules,
# pre-existing battle-tested defaults stay in the base.


MODULES: list[ContextModule] = [
    BLOCKING_OVERLAY,
    KEYBOARD_SCROLL,
]


# ── Composition ───────────────────────────────────────────────────────


def applicable_modules(
    ctx: dict[str, Any] | None = None,
    modules: list[ContextModule] | None = None,
) -> list[ContextModule]:
    """Return the modules whose ``applies_when`` predicate is true for
    ``ctx``. Exceptions inside a module's predicate are swallowed —
    one broken module shouldn't take down the whole composition.
    """
    if ctx is None:
        ctx = current_context()
    pool = modules if modules is not None else MODULES
    out: list[ContextModule] = []
    for mod in pool:
        try:
            if mod.applies_when(ctx):
                out.append(mod)
        except Exception:  # noqa: BLE001 — predicate failure is non-fatal
            continue
    return out


def compose_system_prompt(
    base: str,
    ctx: dict[str, Any] | None = None,
    modules: list[ContextModule] | None = None,
) -> str:
    """Return the brain's system prompt with applicable module sections
    spliced in after the base. Order matches registry order so callers
    can tune priority by reordering.
    """
    if ctx is None:
        ctx = current_context()
    parts = [base.rstrip()]
    for mod in applicable_modules(ctx, modules=modules):
        if mod.prompt_section:
            parts.append("\n\n" + mod.prompt_section.strip())
    return "\n".join(parts).strip()


def applicable_step_hints(
    ctx: dict[str, Any] | None = None,
    modules: list[ContextModule] | None = None,
) -> list[str]:
    """Hint strings from applicable modules. Caller decides what to do
    with them (typically: append to ``runner._recovery_hints[step_index]``
    so the brain's per-step hint block picks them up).
    """
    if ctx is None:
        ctx = current_context()
    return [
        mod.step_hint
        for mod in applicable_modules(ctx, modules=modules)
        if mod.step_hint
    ]


def applicable_handler_hints(
    ctx: dict[str, Any] | None = None,
    modules: list[ContextModule] | None = None,
) -> dict[str, Any]:
    """Merged structured hints from applicable modules. Later-registered
    modules override earlier ones on key collision (consistent with
    the prompt-composition order semantics).
    """
    if ctx is None:
        ctx = current_context()
    merged: dict[str, Any] = {}
    for mod in applicable_modules(ctx, modules=modules):
        if mod.handler_hints:
            merged.update(mod.handler_hints)
    return merged


__all__ = [
    "ContextModule",
    "MODULES",
    "BLOCKING_OVERLAY",
    "KEYBOARD_SCROLL",
    "push_step_context",
    "current_context",
    "applicable_modules",
    "compose_system_prompt",
    "applicable_step_hints",
    "applicable_handler_hints",
]
