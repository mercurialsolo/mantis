"""Persisted state for MicroPlanRunner — extracted from micro_runner.py (#115).

Contains the pure data types and persistence primitives. They have no
dependency on the runner's execution loop, so they can be imported from
host integration code without pulling in the full xdotool/grounding stack.

Anything in this module is part of the **persisted contract** with disk:

* :class:`StepResult` — outcome of one micro-intent (round-trips through JSON via ``_PERSISTED``).
* :class:`RunCheckpoint` — full logical run state for cross-session resume.
* :class:`PauseState` — serializable snapshot when a tool handler raised :class:`PauseRequested`.
* :class:`PauseRequested` — exception used by host tools to request runner pause.
* :class:`RunnerResult` — public return type from :meth:`MicroPlanRunner.run` / ``resume``.
* :data:`REVERSE_ACTIONS` — fallback recovery presses for each step type.

This file MUST stay backward-compatible: changing field names or
dropping fields breaks resume of in-flight runs from old checkpoints.

The legacy import path ``mantis_agent.gym.micro_runner`` re-exports every
name here, so existing callers keep working unchanged.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar

from ..actions import Action

if TYPE_CHECKING:
    from ..cua_contracts.types import RecoveryDecision, Verdict
    from .preview_gate import PreviewResult


# ── Step result ──────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Outcome of executing one micro-intent.

    Persistent fields (`step_index`, `intent`, `success`, ...) round-trip through
    the checkpoint JSON. ``screenshot_png`` and ``last_action`` are observability
    extras populated by the runner — they are deliberately excluded from
    ``to_dict()`` so the checkpoint stays small and JSON-clean.
    """
    step_index: int
    intent: str
    success: bool
    data: str = ""
    steps_used: int = 0
    duration: float = 0.0
    reversed: bool = False

    # Issue #246: recipe-rejection skip envelope. ``skip=True`` is the
    # runner's signal to a hosting orchestrator that a recipe correctly
    # excluded this row (dealer / spam / similar) and the host should
    # advance past it, *not* retry the same step. ``skip_reason`` is
    # the recipe-author key (``"dealer"``, ``"incomplete_required"``,
    # …) the host can branch on. Both default off so legacy callers
    # see no change; recipes opt in via ``ExtractionSchema.rejection_intents``.
    skip: bool = False
    skip_reason: str | None = None

    # Observability extras — populated by MicroPlanRunner; not persisted.
    screenshot_png: bytes | None = field(default=None, repr=False, compare=False)
    last_action: Action | None = field(default=None, repr=False, compare=False)

    # #419 audit-triple: the brain's articulated reasoning ("thinking" /
    # extended-thinking blocks) that produced THIS step's action. Stamped
    # by handlers that drive a brain (Holo3StepHandler pulls from the
    # trajectory's final ``thinking`` field); handlers that don't run a
    # brain (deterministic navigate / paginate / form fill / gate) leave
    # it empty. Surfaced on every step in ``result.json`` so post-mortems
    # see the *chain* of thought, not just the last step's. Deliberately
    # not persisted in the checkpoint (resume doesn't need it) and not
    # used in equality (so dataclass-replace round-trips ignore it).
    reasoning: str = field(default="", repr=False, compare=False)

    # #480 mandatory verdict: a typed :class:`~..cua_contracts.types.Verdict`
    # recording the runner's decision about this step (ok / recoverable /
    # non_recoverable + reason + evidence). Populated by the executor
    # AFTER the handler returns and BEFORE the cursor advances; the
    # runner refuses to advance if this field is None. Existing
    # ``success`` / ``failure_class`` continue to drive runner control
    # flow for now — the typed verdict is the structural contract
    # downstream consumers (#478 emit, #481 preview gate, #483
    # normalised recovery decisions) read against. Not persisted in
    # the checkpoint — resume re-derives it from the legacy fields.
    verdict: Verdict | None = field(default=None, repr=False, compare=False)

    # #483 normalised recovery decision. Typed
    # :class:`~..cua_contracts.types.RecoveryDecision` (advance /
    # retry / replan / rollback / terminate) derived from the verdict
    # + attempt index + step.required. Stamped alongside the verdict
    # in run_executor; the existing retry / halt branches keep
    # driving control flow off ``success`` + ``failure_class`` for
    # back-compat, but metrics / dashboards / future planner layers
    # read this typed slot to group failures by next-action without
    # re-deriving from prose. Not persisted (resume re-derives).
    recovery_decision: RecoveryDecision | None = field(
        default=None, repr=False, compare=False,
    )

    # #482 pre-execution gate result. When the reversibility gate
    # runs (env-opted-in + action is IRREVERSIBLE + a verifier is
    # wired), the gate's :class:`PreviewResult` lands here so
    # downstream consumers (result.json, canonical events, future
    # HITL approval flows) see the gate's decision + evidence.
    # ``None`` means the gate was skipped (default in production
    # until #482 rolls out) or wasn't applicable to this step type.
    preview_result: PreviewResult | None = field(
        default=None, repr=False, compare=False,
    )

    # #300 follow-up: which dispatch path executed the *primary* click /
    # input action for this step. ``"som"`` = CDP-anchored
    # :meth:`~.xdotool_env.XdotoolGymEnv.cdp_click_at_point` (Set-of-Mark
    # routing — bypasses xdotool's mouse pipeline so SPA row clicks fire
    # the right DOM handlers). ``"vision"`` = legacy xdotool click. ``""``
    # = step type didn't dispatch a routable action (extract_data, gate,
    # navigate, etc.). Aggregate counts surface on the run result.
    executor_backend: str = ""

    # Failure diagnostics — populated by the executor on ``success=False``.
    # ``failure_class`` is one of the keys documented in
    # :mod:`~.failure_class` (cf_challenge / nav_timeout / http_4xx /
    # http_5xx / selector_miss / extractor_error / budget_exceeded /
    # unknown). ``final_url`` and ``page_title`` snapshot the browser at
    # the moment of failure so post-mortems land in result.json instead
    # of Modal logs. All three are empty on success.
    failure_class: str = ""
    final_url: str = ""
    page_title: str = ""

    _PERSISTED: ClassVar[tuple[str, ...]] = (
        "step_index", "intent", "success", "data", "steps_used", "duration", "reversed",
        "skip", "skip_reason", "executor_backend",
        "failure_class", "final_url", "page_title",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (omits screenshot_png + last_action)."""
        return {name: getattr(self, name) for name in self._PERSISTED}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepResult":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in allowed})


# ── Run checkpoint ───────────────────────────────────────────────────────


@dataclass
class RunCheckpoint:
    """Persistent logical run state for cross-session resume."""
    version: int = 2
    run_key: str = ""
    plan_signature: str = ""
    session_name: str = ""
    status: str = "running"
    halt_reason: str = ""
    step_index: int = 0
    page: int = 1
    current_url: str = ""
    reentry_url: str = ""
    seen_urls: list = field(default_factory=list)
    extracted_leads: list = field(default_factory=list)
    step_results: list = field(default_factory=list)
    loop_counters: dict = field(default_factory=dict)
    listings_on_page: int = 0
    extracted_titles: list = field(default_factory=list)
    page_listings: list = field(default_factory=list)
    page_listing_index: int = 0
    viewport_stage: int = 0
    current_page: int = 1
    results_base_url: str = ""
    required_filter_tokens: list = field(default_factory=list)
    scroll_state: dict = field(default_factory=dict)
    last_extracted: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    dynamic_coverage: dict = field(default_factory=dict)
    prompt_versions: dict = field(default_factory=dict)  # {name: short_sha} for #127
    timestamp: float = 0.0

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = asdict(self)
        payload["timestamp"] = time.time()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RunCheckpoint | None":
        try:
            with open(path) as f:
                d = json.load(f)
            allowed = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in d.items() if k in allowed})
        except Exception:
            return None


# ── Reverse-action presets ───────────────────────────────────────────────


# Reverse actions for each step type
REVERSE_ACTIONS: dict[str, list[tuple[str, str]]] = {
    "click": [("key_press", "Escape"), ("key_press", "alt+Left")],
    "scroll": [("key_press", "Home")],
    "navigate": [("key_press", "alt+Left")],
    "navigate_back": [],  # Already going back
    "filter": [("key_press", "alt+Left")],
    "paginate": [("key_press", "alt+Left")],
}


# ── Pause / tool channel ─────────────────────────────────────────────────


class _PauseRequested(Exception):
    """Raised by a registered tool handler to request runner pause (#73).

    Hosts call ``raise PauseRequested(prompt=...)`` from inside a
    ``request_user_input`` (or similar) handler. The runner catches it in
    ``_invoke_tool`` and returns a serializable :class:`PauseState`.
    """

    def __init__(self, reason: str = "", prompt: str = "", **extras: Any):
        super().__init__(reason or prompt or "pause requested")
        self.reason = reason or "user_input"
        self.prompt = prompt
        self.extras = dict(extras)


# Public alias so hosts don't depend on a leading underscore.
PauseRequested = _PauseRequested


@dataclass(frozen=True)
class FormFieldValue:
    """One captured form field — text input, select, checkbox, radio,
    or contenteditable — keyed by a stable selector in
    :attr:`BrowserState.form_state` (epic #358 Phase B).

    Password fields are captured as ``masked=True`` with an empty
    ``value``: the selector survives so the resume path knows
    *which* field needs re-prompting, but the secret never lands in
    the JSON. Opt out of masking via ``MANTIS_PAUSE_CAPTURE_PASSWORDS=1``
    (test / debug only).
    """

    kind: str        # "text" | "select" | "checkbox" | "radio" | "contenteditable"
    value: str = ""  # for text/select/contenteditable; "true"/"false" for checkbox/radio
    masked: bool = False


@dataclass
class BrowserState:
    """Browser-runtime snapshot captured at pause time (epic #358).

    Phase A: URL + scroll + viewport — the smallest unit of "where
    in the page was the agent?" that lets a resumed run restore
    visual context instead of starting from page-top. Captured via
    CDP just before :class:`PauseRequested` raises; replayed during
    ``runner.resume``.

    Phase B: ``form_state`` — unsubmitted form input (text inputs,
    selects, checkboxes, radios, contenteditable elements) keyed by
    a stable selector. Half-filled forms paused for OTP / 2FA /
    manual review come back populated. Password fields are captured
    as ``masked=True`` with empty values (caller re-prompts).

    Defaults to all-zero / empty so a snapshot from a runner that
    hasn't initialised the browser (or an env adapter that doesn't
    support CDP capture) still round-trips cleanly through JSON —
    the resume code branches on ``bool(url)`` to decide whether to
    apply any restoration at all.
    """
    url: str = ""
    scroll_x: int = 0
    scroll_y: int = 0
    viewport_w: int = 0
    viewport_h: int = 0
    captured_at: float = 0.0
    form_state: dict[str, FormFieldValue] = field(default_factory=dict)


@dataclass
class PauseState:
    """Serializable snapshot of a paused MicroPlanRunner or GymRunner (#73, #285).

    Round-trips through JSON so host can store it on
    ``plan.agent_data["host_state"]``. Resume by calling
    ``runner.resume(state, user_input=...)``.

    Most fields are MicroPlanRunner-specific (``step_results``,
    ``loop_counters``, ``listings_on_page``). GymRunner fills
    ``trajectory_steps`` + ``task`` + ``task_id`` instead — the other
    fields stay empty and the runner that wrote the snapshot is the
    runner that consumes it, so cross-pollution doesn't matter.

    ``browser_state`` (epic #358 Phase A) captures URL + scroll +
    viewport so resumed plans pick up at the exact pixel — see
    :class:`BrowserState`.
    """
    version: int = 1
    run_key: str = ""
    plan_signature: str = ""
    session_name: str = ""
    step_index: int = 0
    pending_tool: str = ""
    pending_arguments: dict[str, Any] = field(default_factory=dict)
    pending_reason: str = "user_input"
    prompt: str = ""
    step_results: list[dict[str, Any]] = field(default_factory=list)
    loop_counters: dict[str, int] = field(default_factory=dict)
    listings_on_page: int = 0
    checkpoint_path: str = ""
    timestamp: float = 0.0

    # Browser-runtime snapshot — epic #358 Phase A.
    browser_state: BrowserState = field(default_factory=BrowserState)

    # GymRunner extensions (#285). Empty when the snapshot came from
    # MicroPlanRunner.
    trajectory_steps: list[dict[str, Any]] = field(default_factory=list)
    task: str = ""
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PauseState":
        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in payload.items() if k in allowed}
        # Re-hydrate the nested BrowserState dataclass from its dict
        # form. Tolerates legacy snapshots that don't carry the
        # field — falls back to the default empty BrowserState.
        bs_raw = filtered.get("browser_state")
        if isinstance(bs_raw, dict):
            bs_allowed = {f.name for f in fields(BrowserState)}
            bs_kwargs = {k: v for k, v in bs_raw.items() if k in bs_allowed}
            # Phase B: ``form_state`` is dict[selector, FormFieldValue].
            # asdict turned each value into a dict; reverse here.
            form_raw = bs_kwargs.get("form_state")
            if isinstance(form_raw, dict):
                ffv_allowed = {f.name for f in fields(FormFieldValue)}
                bs_kwargs["form_state"] = {
                    str(sel): FormFieldValue(
                        **{k: v for k, v in entry.items() if k in ffv_allowed}
                    )
                    for sel, entry in form_raw.items()
                    if isinstance(entry, dict)
                }
            filtered["browser_state"] = BrowserState(**bs_kwargs)
        elif bs_raw is None:
            filtered.pop("browser_state", None)
        return cls(**filtered)


# ── Public runner result ─────────────────────────────────────────────────


@dataclass
class RunnerResult:
    """Public result of a MicroPlanRunner.run() / resume() call.

    Carries cancellation / pause state alongside the step list so hosts wiring
    the host backend don't have to read ``self._final_status``.
    """
    steps: list[StepResult]
    status: str = "completed"  # completed | halted | cancelled | paused
    cancelled: bool = False
    paused: bool = False
    pause_state: PauseState | None = None
    halt_reason: str = ""


__all__ = [
    "StepResult",
    "RunCheckpoint",
    "REVERSE_ACTIONS",
    "PauseRequested",
    "PauseState",
    "BrowserState",
    "FormFieldValue",
    "RunnerResult",
]
