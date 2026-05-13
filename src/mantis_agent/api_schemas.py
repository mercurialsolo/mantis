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

    # Action mode (status / result / logs / cancel / resume for an existing run).
    # ``resume`` (#344) requires ``user_input`` and a paused run; the server
    # rehydrates the stored PauseState and continues the runner from where
    # the host tool raised PauseRequested.
    action: Optional[Literal["status", "result", "logs", "cancel", "resume"]] = None
    run_id: Optional[str] = None
    tail: Optional[int] = Field(default=None, ge=1, le=10000)
    # Caller-supplied value handed to ``runner.consume_pause_input(...)`` on
    # resume (OTP code, 2FA token, free-text confirmation, etc.). Opaque
    # to the server. Required on ``action="resume"``; ignored otherwise.
    user_input: Optional[Any] = None

    # Run options
    detached: bool = True
    # Identity (#341).
    # ``state_key`` is the legacy single field; when set alone the server
    # routes it to both ``profile_id`` and ``workflow_id`` for back-compat.
    # New callers should set ``profile_id`` (Chrome user-data-dir identity,
    # sticky across plan revisions) and ``workflow_id`` (checkpoint identity,
    # rotated when the plan definition changes) independently.
    state_key: Optional[str] = None
    profile_id: Optional[str] = None
    workflow_id: Optional[str] = None
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

    # When False, free-text plans (`plan_text` or a `.txt` `micro` path) skip
    # PlanDecomposer / heuristic-objective rewriting and run the raw text as a
    # single-intent task_suite. Lets benchmarks measure the brain's intrinsic
    # plan-following without Claude pre-chunking. No effect on `task_suite` /
    # `.json` `micro` shapes (they never decompose anyway).
    decompose: bool = True

    # ── Extraction cache (saves Claude tokens on previously-seen URLs) ──
    # When cache_read is true, the runner peeks env.current_url BEFORE the
    # deep-extract Claude call; on hit, the cached lead is emitted and the
    # extract is skipped (~$0.04/item saved). When cache_write is true,
    # every viable extraction is persisted to a per-tenant file. The cache
    # is keyed by (tenant_id, cache_key); cache_key defaults to state_key.
    # TTL controls staleness — entries older than ttl_seconds are treated
    # as misses and re-extracted.
    cache_read: bool = False
    cache_write: bool = False
    cache_ttl_seconds: int = Field(default=86400, ge=0, le=2592000)  # 0..30d
    cache_key: Optional[str] = None

    # Screencast recording (Tier 2 follow-up).
    # When record_video=True, the runtime spawns ffmpeg x11grab against the
    # Xvfb display while the agent loop runs and saves the output under
    # the per-tenant per-run dir. Fetch via GET /v1/runs/{run_id}/video.
    record_video: bool = False
    video_format: Literal["mp4", "webm", "gif"] = "mp4"
    video_fps: int = Field(default=5, ge=1, le=30)

    # #300 follow-up: per-request override for
    # :attr:`RoutingPolicy.som_for_unstructured_clicks`. ``None`` defers
    # to the deployment env (``MANTIS_ROUTE_SOM_CLICKS``). ``True`` /
    # ``False`` forces the toggle on / off for this run, mirroring the
    # ablation pattern documented in ``scripts/ablate_v1_cua.py`` (the
    # ``perceptual_verify`` / ``loop_recovery`` / ``done_gate`` toggles).
    route_som_clicks: Optional[bool] = None

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
            if self.action == "resume" and self.user_input is None:
                raise ValueError("action='resume' requires user_input")
            return self  # poll/cancel/resume modes don't need a plan

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


class PureCUARequest(BaseModel):
    """``POST /v1/cua`` — pure CUA loop: brain ↔ XdotoolGymEnv, no Claude.

    Bypasses ``PlanDecomposer``, ``ClaudeGrounding`` and ``ClaudeExtractor``
    entirely. The configured brain (Holo3 / Gemma4) emits screen-space
    actions (``click`` / ``double_click`` / ``type_text`` / ``key_press`` /
    ``scroll`` / ``drag`` / ``wait`` / ``done``) which the gym env executes
    directly via xdotool against the headed Chrome inside Xvfb.

    Use when you want to measure the brain's intrinsic plan-following and
    grounding accuracy on a single instruction, with zero Claude assist.
    """

    model_config = {"extra": "allow"}

    # Single free-text instruction handed verbatim to the brain.
    instruction: str = Field(..., min_length=1)

    # Browser bootstrap.
    start_url: str = ""
    settle_time: float = Field(default=2.0, ge=0.0, le=30.0)

    # GymRunner knobs.
    max_steps: int = Field(default=30, ge=1, le=MAX_STEPS_PER_PLAN)
    frames_per_inference: int = Field(default=1, ge=1, le=8)

    # Run options (mirrors PredictRequest shape so tenant caps clamp the
    # same way). ``state_key`` / ``profile_id`` / ``workflow_id`` semantics
    # match PredictRequest (#341).
    detached: bool = False
    state_key: Optional[str] = None
    profile_id: Optional[str] = None
    workflow_id: Optional[str] = None
    max_cost: float = Field(default=MAX_COST_USD, gt=0)
    max_time_minutes: int = Field(default=MAX_RUNTIME_MINUTES, gt=0)

    # Proxy controls.
    proxy_city: Optional[str] = None
    proxy_state: Optional[str] = None
    proxy_disabled: bool = False

    # Screencast.
    record_video: bool = False
    video_format: Literal["mp4", "webm", "gif"] = "mp4"
    video_fps: int = Field(default=5, ge=1, le=30)

    # #300 follow-up: per-request override for
    # :attr:`RoutingPolicy.som_for_unstructured_clicks`. ``None``
    # defers to the deployment env (``MANTIS_ROUTE_SOM_CLICKS``).
    # ``True`` / ``False`` forces the toggle on / off for this run,
    # mirroring the per-request ablation pattern used for
    # ``perceptual_verify`` / ``loop_recovery`` / ``done_gate`` so the
    # ablation harness can A/B without redeploying.
    route_som_clicks: Optional[bool] = None

    @model_validator(mode="after")
    def _clamp_caps(self) -> "PureCUARequest":
        if self.max_cost > MAX_COST_USD:
            self.max_cost = MAX_COST_USD
        if self.max_time_minutes > MAX_RUNTIME_MINUTES:
            self.max_time_minutes = MAX_RUNTIME_MINUTES
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


_URL_RE = __import__("re").compile(r"https?://([^/\s\"'`)>]+)")


def extract_navigate_hosts(plan: list[dict[str, Any]] | dict[str, Any]) -> list[str]:
    """Pull all unique hostnames referenced by ``navigate``-type steps.

    Used to enforce the per-tenant allowlist before the plan starts.
    Inspects micro-plan step lists ({intent, type, ...}) and task-suite
    dicts ({tasks: [{intent, start_url, ...}]}). Returns a sorted, unique
    list of lowercase hosts.
    """
    hosts: set[str] = set()

    def _add_url(url: str | None) -> None:
        if not url:
            return
        m = _URL_RE.search(url)
        if m:
            hosts.add(m.group(1).lower())

    def _scan_intent(text: str | None) -> None:
        if not text:
            return
        for m in _URL_RE.finditer(text):
            hosts.add(m.group(1).lower())

    if isinstance(plan, dict):
        # task-suite shape
        _add_url(plan.get("base_url"))
        for task in plan.get("tasks") or []:
            if isinstance(task, dict):
                _add_url(task.get("start_url"))
                _scan_intent(task.get("intent"))
    elif isinstance(plan, list):
        # micro-plan shape
        for step in plan:
            if not isinstance(step, dict):
                continue
            _scan_intent(step.get("intent"))
            _add_url(step.get("url"))

    return sorted(hosts)


def assert_hosts_allowed(
    hosts: list[str],
    allowed_predicate,
) -> None:
    """Raise ``PermissionError`` if any host fails ``allowed_predicate``.

    ``allowed_predicate`` is typically ``tenant.is_domain_allowed``. Empty
    ``hosts`` list is a no-op (e.g. a plan with no navigate URLs).
    """
    rejected = [h for h in hosts if not allowed_predicate(h)]
    if rejected:
        raise PermissionError(
            f"plan references host(s) not in tenant allowlist: {', '.join(rejected)}"
        )


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
    # #344: paused-run surface. When ``status == "paused"``, ``prompt`` is
    # the message the registered tool passed to PauseRequested (e.g. "Enter
    # the 6-digit code"), ``reason`` is the PauseRequested.reason
    # ("user_input" by default), and ``pause_state`` is an opaque dict
    # the caller hands back verbatim on ``action="resume"``.
    prompt: Optional[str] = None
    reason: Optional[str] = None
    pause_state: Optional[dict[str, Any]] = None
