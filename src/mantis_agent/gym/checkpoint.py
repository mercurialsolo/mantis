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
from typing import Any, ClassVar

from ..actions import Action


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

    # #300 follow-up: which dispatch path executed the *primary* click /
    # input action for this step. ``"som"`` = CDP-anchored
    # :meth:`~.xdotool_env.XdotoolGymEnv.cdp_click_at_point` (Set-of-Mark
    # routing — bypasses xdotool's mouse pipeline so SPA row clicks fire
    # the right DOM handlers). ``"vision"`` = legacy xdotool click. ``""``
    # = step type didn't dispatch a routable action (extract_data, gate,
    # navigate, etc.). Aggregate counts surface on the run result.
    executor_backend: str = ""

    _PERSISTED: ClassVar[tuple[str, ...]] = (
        "step_index", "intent", "success", "data", "steps_used", "duration", "reversed",
        "skip", "skip_reason", "executor_backend",
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
        return cls(**{k: v for k, v in payload.items() if k in allowed})


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
    "RunnerResult",
]
