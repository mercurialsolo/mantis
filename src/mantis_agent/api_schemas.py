"""Pydantic request/response models + server-side caps for /predict.

Tier 1 multi-tenant hardening: replaces `dict[str, Any]` plumbing with typed
models, enforces hard caps that callers cannot override, and centralizes the
plan-shape polymorphism (`task_suite` vs `task_file` vs `micro` vs `plan_text`)
so the FastAPI handler is small and reviewable.

Caps are deliberately conservative for v1; tune via env vars if your tenant
needs more rope.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

try:
    from pydantic import BaseModel, Field, model_validator
except ImportError as exc:  # pragma: no cover - container-only deps
    raise ImportError(
        "mantis_agent.api_schemas requires pydantic. "
        "Install via: pip install -e '.[server]'"
    ) from exc


# ── Server-side caps (hard limits, not overridable by payload) ──────────────
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


MAX_STEPS_PER_PLAN = _env_int("MANTIS_MAX_STEPS_PER_PLAN", 200)
MAX_LOOP_ITERATIONS = _env_int("MANTIS_MAX_LOOP_ITERATIONS", 50)
MAX_RUNTIME_MINUTES = _env_int("MANTIS_MAX_RUNTIME_MINUTES", 60)
MAX_COST_USD = _env_float("MANTIS_MAX_COST_USD", 25.0)


# ── Request payloads ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Top-level /predict request.

    Plan input is one of (priority order): task_suite, task_file_contents,
    task_file, micro. plan_text triggers Claude-side decomposition.

    Caps (max_cost, max_time_minutes) requested here are clamped server-side
    against MAX_COST_USD / MAX_RUNTIME_MINUTES.
    """

    model_config = {"extra": "allow"}  # forward-compat for new fields

    # Plan shape (one of these is required for run mode)
    task_suite: Optional[dict[str, Any]] = None
    task_file_contents: Optional[str] = None
    task_file: Optional[str] = None
    micro: Optional[str] = None
    micro_path: Optional[str] = None  # alias for micro
    plan_text: Optional[str] = None

    # Action mode (status / result / logs / cancel for an existing run)
    action: Optional[Literal["status", "result", "logs", "cancel"]] = None
    run_id: Optional[str] = None
    tail: Optional[int] = Field(default=None, ge=1, le=10000)

    # Run options
    detached: bool = True
    state_key: Optional[str] = None
    resume_state: bool = False
    max_cost: float = Field(default=MAX_COST_USD, gt=0)
    max_time_minutes: int = Field(default=MAX_RUNTIME_MINUTES, gt=0)

    # Optional proxy override (subject to tenant allowlist)
    proxy_city: Optional[str] = None
    proxy_state: Optional[str] = None

    # Forward-compat for graph-learning + objective
    objective: Optional[str] = None
    graph_learn: bool = False
    graph_learn_only: bool = False

    @model_validator(mode="after")
    def _clamp_caps_and_validate_action(self) -> "PredictRequest":
        # Hard caps — caller cannot exceed
        if self.max_cost > MAX_COST_USD:
            self.max_cost = MAX_COST_USD
        if self.max_time_minutes > MAX_RUNTIME_MINUTES:
            self.max_time_minutes = MAX_RUNTIME_MINUTES

        if self.action is not None:
            if not self.run_id:
                raise ValueError(f"action={self.action!r} requires run_id")
            return self  # poll/cancel modes don't need a plan

        # Run mode: at least one plan-shape must be present
        plan_provided = any(
            v is not None and v != ""
            for v in (
                self.task_suite,
                self.task_file_contents,
                self.task_file,
                self.micro,
                self.micro_path,
                self.plan_text,
            )
        )
        if not plan_provided:
            raise ValueError(
                "request must provide one of: task_suite, task_file_contents, "
                "task_file, micro, plan_text"
            )
        return self


def validate_micro_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a parsed micro-plan against MAX_STEPS / MAX_LOOP_ITERATIONS.

    Returns the (possibly clamped) step list. Raises ValueError on hard
    violations (e.g., step count exceeds MAX_STEPS_PER_PLAN).
    """
    if not isinstance(steps, list):
        raise ValueError("micro plan must be a JSON array of step objects")
    if len(steps) > MAX_STEPS_PER_PLAN:
        raise ValueError(
            f"micro plan has {len(steps)} steps; server cap is "
            f"{MAX_STEPS_PER_PLAN} (set MANTIS_MAX_STEPS_PER_PLAN to override)"
        )
    clamped: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {i} is not a JSON object")
        if not step.get("intent"):
            raise ValueError(f"step {i} missing required field 'intent'")
        if not step.get("type"):
            raise ValueError(f"step {i} missing required field 'type'")
        # Clamp loop_count to MAX_LOOP_ITERATIONS
        if step.get("type") == "loop" and isinstance(step.get("loop_count"), int):
            if step["loop_count"] > MAX_LOOP_ITERATIONS:
                step = dict(step)
                step["loop_count"] = MAX_LOOP_ITERATIONS
        clamped.append(step)
    return clamped


# ── Response payloads ───────────────────────────────────────────────────────
class DetachedRunHandle(BaseModel):
    """Returned from /predict when detached=True."""

    status: Literal["queued"]
    created_at: str
    model: str
    mode: Literal["detached"]
    run_id: str
    payload: dict[str, Any]
    updated_at: str
    status_path: str
    result_path: str
    csv_path: str
    events_path: str


class RunStatus(BaseModel):
    """Detached-run status snapshot."""

    model_config = {"extra": "allow"}

    status: str
    run_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    error: Optional[str] = None
