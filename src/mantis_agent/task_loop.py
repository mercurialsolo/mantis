"""Shared task loop for Mantis CUA executors.

Eliminates copy-pasted task iteration logic across Modal and Baseten
executor functions. Each executor configures a TaskLoopConfig with its
brain, env, and optional callbacks for executor-specific behavior.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .server_utils import build_proxy_config, resolve_proxy_server

if TYPE_CHECKING:
    from .gym.computer_client import ComputerPlaneConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskLoopConfig:
    """Configuration for the shared task loop.

    Required fields describe the runtime. Optional callbacks inject
    executor-specific behavior (filter validation, trajectory saving, etc.)
    without polluting the shared loop code.
    """

    # ── Identity ──
    run_id: str
    session_name: str
    model_name: str
    results_prefix: str  # e.g. "holo3", "cua", "gemma4cua", "claude"

    # ── Runtime ──
    brain: Any
    env: Any
    grounding: Any = None
    extractor: Any = None
    fallback_brain: Any = None
    fallback_label: str = "fallback"
    fallback_micro_retries: int = 2
    fallback_micro_max_steps: int = 6
    stop_on_task_failure: bool = True

    # ── Execution params ──
    max_steps: int = 30
    standard_task_max_steps: int = 15
    frames_per_inference: int = 5
    use_sub_plan: bool = True
    site_config: Any = None  # SiteConfig for URL patterns (passed to callbacks)

    # ── Viewer ──
    viewer_event_bus: Any = None

    # ── Callbacks (executor-specific hooks) ──

    # Called after GymRunner.run() for standard tasks, before recording scores.
    # Signature: (task_config, task_id, result, env, brain, config) -> result
    # Use for: min_steps retry, hybrid verify, filter validation (Holo3).
    on_task_result: Callable | None = None

    # Called after a standard task completes and scores are recorded.
    # Signature: (task_id, intent, result) -> None
    # Use for: trajectory saving (Claude).
    on_task_complete: Callable | None = None

    # Called after a loop task completes (all iterations done).
    # Signature: () -> None
    # Use for: vol.commit() after learning store writes.
    on_loop_complete: Callable | None = None

    # ── Persistence ──
    results_dir: str = "/data/results"
    volume_commit: Callable | None = None

    # ── Summary extras ──
    # Extra fields merged into the save_progress summary dict.
    # e.g. {"estimated_api_cost_usd": 0.0} for Claude.
    summary_extras: dict = field(default_factory=dict)


def diagnose_proxy_egress(proxy_server: str, *, timeout: float = 8.0) -> dict[str, Any]:
    """Hit ipinfo.io through ``proxy_server`` and return the egress IP +
    geo info (#viewer-proxy-diag). Best-effort — never raises.

    Returns one of:
      * ``{"disabled": True}`` when no proxy is configured
      * ``{"ip", "city", "region", "country", "org"}`` on success
      * ``{"error": "..."}`` when the probe fails (network, timeout, 5xx)

    Surfaced to the live-viewer overlay so operators see at a glance
    which exit IP / geo the run is actually using — bare ``proxy_provider``
    in the runtime block hides whether the geo modifier landed (e.g.
    PrivateProxy returning a Romanian IP despite ``city=miami``).
    """
    if not proxy_server:
        return {"disabled": True}
    try:
        import requests as _r
        r = _r.get(
            "https://ipinfo.io/json",
            proxies={"http": proxy_server, "https": proxy_server},
            timeout=timeout,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        d = r.json()
        return {
            "ip": str(d.get("ip", "") or ""),
            "city": str(d.get("city", "") or ""),
            "region": str(d.get("region", "") or ""),
            "country": str(d.get("country", "") or ""),
            "org": str(d.get("org", "") or "")[:80],
        }
    except Exception as exc:  # noqa: BLE001 — diag, never fatal
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def _computer_plane_config_from_env(
    executor_name: str | None = None,
) -> "ComputerPlaneConfig":
    """Build a `ComputerPlaneConfig` from `MANTIS_COMPUTER_PLANE_*` env vars.

    Defaults match production today: `backend="local"`. Lets the brain
    container flip a single executor onto the Phase-1 remote plane via
    a one-secret edit without touching call sites:

    * `MANTIS_COMPUTER_PLANE_BACKEND` — `local` (default) | `modal`
    * `MANTIS_COMPUTER_PLANE_URL` — remote base URL (required for
      `modal`); typically resolved at executor boot via
      `modal.Function.from_name("mantis-cua-server", "computer_plane").get_web_url()`.
    * `MANTIS_COMPUTER_PLANE_TOKEN` — `Authorization: Bearer ...`
    * `MANTIS_COMPUTER_PLANE_ENABLE_CDP` — `1`/`true` to opt in
      (off by default per the wire-contract CUA-purity constraint).

    Phase 1.5 (#846) — session-routed mode:

    * `MANTIS_SESSION_ROUTER_URL` — router base URL (typically the
      ``api()`` ASGI URL). When set, brain mints a dedicated
      per-session computer-plane container via
      ``POST /v1/computer_sessions`` instead of sharing the pinned
      ``computer_plane()`` ASGI app.
    * `MANTIS_SESSION_ROUTER_TOKEN` — tenant token forwarded to the
      router's ``require_run_scope`` middleware.
    * `MANTIS_SESSION_TTL_SECONDS` — per-session lifetime (seconds);
      clamped against the deployment cap server-side.
    """
    from .gym.computer_client import ComputerPlaneConfig

    backend_env = (os.environ.get("MANTIS_COMPUTER_PLANE_BACKEND") or "local").strip().lower()
    if backend_env not in ("local", "modal", "e2b", "daytona"):
        backend_env = "local"
    enable_cdp = (os.environ.get("MANTIS_COMPUTER_PLANE_ENABLE_CDP") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    try:
        session_ttl = int(os.environ.get("MANTIS_SESSION_TTL_SECONDS") or "3600")
    except ValueError:
        session_ttl = 3600
    cfg = ComputerPlaneConfig(
        backend=backend_env,  # type: ignore[arg-type]
        remote_base_url=(os.environ.get("MANTIS_COMPUTER_PLANE_URL") or "").strip() or None,
        remote_auth_token=(os.environ.get("MANTIS_COMPUTER_PLANE_TOKEN") or "").strip() or None,
        enable_cdp=enable_cdp,
        session_router_url=(os.environ.get("MANTIS_SESSION_ROUTER_URL") or "").strip() or None,
        session_router_auth_token=(os.environ.get("MANTIS_SESSION_ROUTER_TOKEN") or "").strip() or None,
        session_ttl_seconds=session_ttl,
    )
    return cfg.resolve_for_executor(executor_name)


def setup_env(
    *,
    base_url: str,
    run_id: str,
    session_name: str,
    settle_time: float = 2.0,
    proxy_city: str = "miami",
    proxy_state: str = "",
    proxy_provider: str = "",
    proxy_country: str = "",
    proxy_disabled: bool = False,
    display: str | None = None,
    start_xvfb: bool = False,
    browser: str = "google-chrome",
    viewport: tuple[int, int] = (1280, 720),
    profile_dir: str = "",
    save_screenshots_dir: str = "/data/screenshots",
    reuse_session: bool = False,
    extra_http_headers: dict[str, str] | None = None,
    computer_plane_config: "ComputerPlaneConfig | None" = None,
    executor_name: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Set up proxy + computer-plane env.

    Routes through ``make_computer_client(cfg)`` so the seam introduced
    in #697 (Computer Plane Phase 0) is the only construction path. The
    returned env is always a ``ComputerClient`` — today that resolves to
    ``LocalXdotoolImpl`` (a ``XdotoolGymEnv`` subclass with latency
    instrumentation) by default; Phase 1 swaps in ``RemoteComputerImpl``
    when the config flips to ``backend="modal"``.

    Args:
        proxy_disabled: When True, skip the upstream proxy entirely. Use for
            test/internal sites that don't need bot-detection bypass and
            shouldn't depend on the residential proxy's availability.
        computer_plane_config: Explicit config. Default reads
            ``MANTIS_COMPUTER_PLANE_*`` env vars via
            :func:`_computer_plane_config_from_env`. The ``backend="modal"``
            path skips the local Xvfb spawn (the remote container owns
            Xvfb + Chrome).
        executor_name: When set, the env-derived config is consulted
            for a per-executor override before resolving the backend
            (e.g. ``run_claude_cua`` flips to ``modal`` while GPU
            executors stay ``local``).

    Returns ``(env, proxy_proc_or_None, proxy_diag)`` where ``proxy_diag``
    is the ipinfo egress probe (``{"ip", "city", "region", ...}`` on
    success, ``{"disabled": True}`` when proxy is off, ``{"error": ...}``
    when the probe failed).
    """
    from .gym.computer_client import make_computer_client

    cfg = computer_plane_config or _computer_plane_config_from_env(executor_name)

    if proxy_disabled:
        proxy = None
        proxy_server, proxy_proc = "", None
        print("  Proxy: DISABLED (proxy_disabled=true)")
    else:
        proxy = build_proxy_config(
            city=proxy_city,
            state=proxy_state,
            session_id=f"mantis{run_id.replace('_', '')}",
            provider=proxy_provider,
            country=proxy_country,
        )
        proxy_server, proxy_proc = resolve_proxy_server(proxy)
        if proxy:
            provider = proxy_provider or os.environ.get("MANTIS_PROXY_PROVIDER") or "iproyal"
            print(f"  Proxy: {provider} via {proxy.get('server', '')}")

    # #viewer-proxy-diag: best-effort egress probe so the live-viewer
    # overlay shows the actual IP / city / region instead of just the
    # provider name. Hits ipinfo.io through the same proxy chain Chrome
    # uses, so we surface the IP Cloudflare sees.
    proxy_diag = diagnose_proxy_egress(proxy_server)

    # #825 — Timezone + locale consistency with proxy exit geo.
    # The Modal CUA image bakes ``TZ=America/New_York``, but a US
    # residential proxy can land in any state. Mismatch between
    # ``navigator.language`` / ``Intl.DateTimeFormat().resolvedOptions().timeZone``
    # and the IP geo is a detectable signal for CF / DataDome scoring.
    #
    # Resolve the proxy's geo → (tz, lang), then update ``TZ`` and
    # ``LANG`` in the process env BEFORE Chrome starts so Chrome
    # inherits them. The xdotool env's ``_env`` dict is built from
    # ``os.environ`` at constructor time, so this write lands before
    # the browser process is forked. CDP-level
    # ``Emulation.setTimezoneOverride`` / ``setLocaleOverride`` calls
    # happen post-launch (out of scope here — wired in cdp_stealth).
    #
    # Opt-out: set ``MANTIS_GEO_CONSISTENCY=0`` to preserve the baked-
    # in image defaults (useful for CI / replay).
    if os.environ.get("MANTIS_GEO_CONSISTENCY", "").strip().lower() not in {"0", "false", "no", "off"}:
        from .gym.geo_consistency import resolve_tz_and_lang
        resolved_tz, resolved_lang = resolve_tz_and_lang(proxy_diag)
        if resolved_tz:
            os.environ["TZ"] = resolved_tz
            os.environ["MANTIS_RESOLVED_TZ"] = resolved_tz
        if resolved_lang:
            os.environ["LANG"] = f"{resolved_lang.replace('-', '_')}.UTF-8"
            os.environ["MANTIS_RESOLVED_LANG"] = resolved_lang
        try:
            print(
                f"  Geo consistency: tz={resolved_tz} lang={resolved_lang} "
                f"(proxy={proxy_diag.get('country', '?')}/{proxy_diag.get('region', '?')})"
            )
        except Exception:
            pass

    # Local Xvfb only when the env actually runs in-process. The remote
    # computer plane owns its own Xvfb; spawning a brain-side one would
    # both waste a CPU and break DISPLAY clobbering on shared executors.
    if cfg.backend == "local" and start_xvfb and display:
        subprocess.Popen(
            ["Xvfb", display, "-screen", "0", f"{viewport[0]}x{viewport[1]}x24", "-ac", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = display
        time.sleep(1)

    env_kwargs: dict[str, Any] = {
        "start_url": base_url,
        "viewport": viewport,
        "browser": browser,
        "settle_time": settle_time,
        "human_speed": False,
        "proxy_server": proxy_server,
        "save_screenshots": f"{save_screenshots_dir}/{session_name}_{run_id}",
        "reuse_session": reuse_session,
    }
    if display and cfg.backend == "local":
        env_kwargs["display"] = display
    if profile_dir:
        env_kwargs["profile_dir"] = profile_dir
    if extra_http_headers:
        env_kwargs["extra_http_headers"] = extra_http_headers
    if cfg.backend != "local":
        # `RemoteComputerImpl` binds its session to (tenant, profile,
        # run); surface the brain-side identifiers so the remote can
        # honor profile reuse + the per-profile lock in Phase 2.
        env_kwargs.setdefault("tenant_id", session_name)
        env_kwargs.setdefault("profile_id", session_name)
        env_kwargs.setdefault("run_id", run_id)

    env = make_computer_client(cfg, **env_kwargs)
    return env, proxy_proc, proxy_diag


def setup_viewer(
    enabled: bool,
    *,
    proxy_diag: dict[str, Any] | None = None,
    api_run_id: str | None = None,
    api_tenant_id: str | None = None,
) -> tuple[Any, Any, str | None]:
    """Set up the Modal viewer if enabled.

    Returns ``(viewer_ctx, viewer_event_bus, viewer_url)``. All three
    are ``None`` when ``enabled`` is False or setup fails.

    ``proxy_diag`` is the ipinfo egress probe from :func:`setup_env`
    (passed through to the viewer's ``/api/proxy_info`` endpoint so
    the live-viewer header can display the IP / city / region).

    ``api_run_id`` / ``api_tenant_id`` thread through so the viewer's
    Pause/Resume buttons can POST to the cua-server API on behalf
    of the user (the cua-server API token stays inside the Modal
    container — never exposed to the browser).
    """
    if not enabled:
        return None, None, None
    try:
        from .viewer_modal import modal_viewer

        viewer_ctx = modal_viewer(
            proxy_diag=proxy_diag or {},
            api_run_id=api_run_id or "",
            api_tenant_id=api_tenant_id or "",
        )
        viewer_event_bus, viewer_url = viewer_ctx.__enter__()
        return viewer_ctx, viewer_event_bus, viewer_url
    except Exception as e:
        print(f"  Viewer failed to start: {e}")
        return None, None, None


# ── Internal helpers ──────────────────────────────────────────────


def _make_save_progress(
    config: TaskLoopConfig,
    tasks: list[dict],
    scores: list[float],
    task_details: list[dict],
    t0: float,
    started_at: str,
) -> Callable:
    """Build a save_progress closure for the task loop."""

    def save_progress():
        completed_at = (
            datetime.now(timezone.utc).isoformat() if len(scores) == len(tasks) else ""
        )
        summary = {
            "run_id": config.run_id,
            "session_name": config.session_name,
            "model": config.model_name,
            "tasks_run": len(tasks),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_time_s": round(time.time() - t0),
            **config.summary_extras,
            "scores": list(scores),
            "task_details": list(task_details),
        }
        results_path = os.path.join(
            config.results_dir,
            f"{config.results_prefix}_results_{config.session_name}_{config.run_id}.json",
        )
        os.makedirs(config.results_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        if config.volume_commit:
            config.volume_commit()

    return save_progress


def _build_loop_final_detail(
    task_id: str,
    results: list[Any],
    task_start: float,
    include_failure_breakdown: bool = False,
) -> dict[str, Any]:
    """Build the final detail dict after a loop task completes."""
    viable = sum(1 for r in results if r.success)
    total = len(results)
    total_parse_failures = sum(
        getattr(r, "parse_failures", 0) for r in results
    )
    real_iterations = sum(
        1
        for r in results
        if getattr(r, "parse_failures", 0) < max(r.steps // 2, 1)
    )
    detail: dict[str, Any] = {
        "task_id": task_id,
        "success": viable > 0,
        "steps": sum(r.steps for r in results),
        "duration_s": round(time.time() - task_start),
        "termination_reason": "loop_complete",
        "iterations": total,
        "viable": viable,
        "real_iterations": real_iterations,
        "parse_failures": total_parse_failures,
        "data": [r.data[:200] for r in results if r.data],
    }
    if include_failure_breakdown:
        from collections import Counter

        fail_counts = dict(
            Counter(
                getattr(r, "failure_category", "")
                for r in results
                if getattr(r, "failure_category", "")
            )
        )
        if fail_counts:
            detail["failure_breakdown"] = fail_counts
    return detail


def _redacted_action(action: Any) -> str:
    """Summarize an action without leaking typed values."""
    action_type = getattr(action, "action_type", "")
    kind = getattr(action_type, "value", str(action_type))
    params = dict(getattr(action, "params", {}) or {})
    if kind == "type_text":
        text = str(params.get("text", ""))
        params = {"text": f"<{len(text)} chars>"}
    return f"{kind}({params})"


def _recent_failure_context(result: Any, max_steps: int = 6) -> str:
    trajectory = list(getattr(result, "trajectory", []) or [])[-max_steps:]
    if not trajectory:
        return "No recent action context was recorded."

    lines: list[str] = []
    for item in trajectory:
        step = getattr(item, "step", "?")
        action = _redacted_action(getattr(item, "action", None))
        feedback = str(getattr(item, "feedback", "") or "")
        if feedback:
            feedback = feedback.replace("\n", " ")[:180]
            lines.append(f"- step {step}: {action} -> {feedback}")
        else:
            lines.append(f"- step {step}: {action}")
    return "\n".join(lines)


def _build_micro_fallback_intent(
    *,
    task_id: str,
    intent: str,
    result: Any,
    attempt: int,
) -> str:
    """Build a generic stuck-step prompt for Claude.

    The fallback is deliberately scoped to a micro-step. Claude should repair
    the current browser state and stop; the primary executor then resumes the
    original section from that state.
    """
    termination = str(getattr(result, "termination_reason", "unknown"))
    recent = _recent_failure_context(result)
    return (
        "You are handling a single stuck browser-control micro-step inside a "
        "larger workflow. Do not execute the full workflow, future sections, "
        "or broad multi-step plan.\n\n"
        f"SECTION ID: {task_id}\n"
        f"ORIGINAL SECTION GOAL:\n{intent[:2400]}\n\n"
        f"PRIMARY EXECUTOR FAILURE: {termination}\n"
        f"RECENT ACTIONS BEFORE FAILURE:\n{recent}\n\n"
        f"RECOVERY ATTEMPT: {attempt}\n\n"
        "Your job now:\n"
        "1. Look at the current browser screen.\n"
        "2. Execute only the smallest visible action or short action sequence "
        "needed to unstick this section and make concrete progress toward the "
        "original section goal.\n"
        "3. Stop immediately after that micro-step with done(success=true). "
        "The primary executor will resume the original section afterward.\n\n"
        "Constraints:\n"
        "- Do not continue into later workflow sections.\n"
        "- Do not complete the whole original section unless the only needed "
        "micro-step itself completes it.\n"
        "- Do not edit unrelated fields or clear already-correct values.\n"
        "- Operate inside the web page content. Do not click the browser "
        "toolbar, address bar, tab strip, or extension area unless the original "
        "section goal is explicit navigation.\n"
        "- If the section goal is to submit, continue, save, search, or move "
        "forward, look for an in-page button/control with that meaning; if it "
        "is not visible, use a small scroll to reveal it.\n"
        "- Prefer click, scroll, key_press, or wait. Type only when the current "
        "micro-step is clearly a focused field that requires typing.\n"
        "- If no useful recovery action is visible, finish with "
        "done(success=false) and explain the blocker briefly."
    )


def _standard_task_max_steps(task_config: dict[str, Any], config: TaskLoopConfig) -> int:
    if "max_steps" in task_config:
        return int(task_config["max_steps"])
    return min(int(config.max_steps), int(config.standard_task_max_steps))


# ── Main task loop ────────────────────────────────────────────────


def run_task_loop(
    tasks: list[dict[str, Any]],
    config: TaskLoopConfig,
    *,
    t0: float | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Execute a list of tasks using the shared loop pattern.

    Returns (scores, task_details).
    """
    from .gym.cost_meter import make_initial_costs
    from .gym.runner import GymRunner

    t0 = t0 or time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    scores: list[float] = []
    task_details: list[dict[str, Any]] = []
    save_progress = _make_save_progress(
        config, tasks, scores, task_details, t0, started_at
    )

    on_step = config.viewer_event_bus.emit if config.viewer_event_bus else None

    for i, task_config in enumerate(tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]

        print(f"\nTask {i + 1}/{len(tasks)}: {task_id}")
        task_start = time.time()
        # #350: per-task cost attribution. Each standard task creates
        # its own GymRunner with a fresh CostMeter; primary + micro
        # fallback + resume runners each accumulate independently.
        # We merge their counters here so detail["costs"] reflects
        # the whole task's spend.
        task_cost_acc: dict[str, Any] = make_initial_costs()
        last_cost_meter: Any = None

        def _merge_runner_costs(_runner: Any) -> None:
            nonlocal last_cost_meter
            meter = getattr(_runner, "cost_meter", None)
            if meter is None:
                return
            last_cost_meter = meter
            for k, v in (meter.costs or {}).items():
                try:
                    task_cost_acc[k] = task_cost_acc.get(k, 0) + v
                except TypeError:
                    pass

        try:
            if task_config.get("require_session") and config.env.has_session(
                config.session_name
            ):
                config.env.load_session(config.session_name)

            # ── Dynamic loop task ──
            if task_config.get("loop"):
                from .gym.workflow_runner import LoopConfig, WorkflowRunner

                def _make_on_loop_iteration(
                    _task_id, _task_start, _scores, _task_details, _save_progress
                ):
                    def on_loop_iteration(iter_num, iter_result, all_results):
                        viable = sum(1 for r in all_results if r.success)
                        total = len(all_results)
                        elapsed = time.time() - _task_start
                        total_parse_failures = sum(
                            getattr(r, "parse_failures", 0) for r in all_results
                        )
                        real_iterations = sum(
                            1
                            for r in all_results
                            if getattr(r, "parse_failures", 0)
                            < max(r.steps // 2, 1)
                        )
                        detail = {
                            "task_id": _task_id,
                            "success": viable > 0,
                            "steps": sum(r.steps for r in all_results),
                            "duration_s": round(elapsed),
                            "termination_reason": "loop_in_progress",
                            "iterations": total,
                            "viable": viable,
                            "real_iterations": real_iterations,
                            "parse_failures": total_parse_failures,
                            "data": [
                                r.data[:200] for r in all_results if r.data
                            ],
                        }
                        if (
                            _task_details
                            and _task_details[-1].get("task_id") == _task_id
                        ):
                            _task_details[-1] = detail
                        else:
                            _scores.append(0.0)
                            _task_details.append(detail)
                        _save_progress()
                        status = "VIABLE" if iter_result.success else "SKIP"
                        print(
                            f"  [{iter_num}] {status} — {viable}/{total} viable ({elapsed:.0f}s)"
                        )

                    return on_loop_iteration

                loop_cfg = LoopConfig(
                    iteration_intent=intent,
                    pagination_intent=task_config["loop"].get(
                        "pagination_intent",
                        "Scroll to bottom, click Next page. If no next, terminate('failure').",
                    ),
                    max_iterations=task_config["loop"].get("max_iterations", 50),
                    max_pages=task_config["loop"].get("max_pages", 10),
                    max_steps_per_iteration=task_config["loop"].get(
                        "max_steps_per_iteration", config.max_steps
                    ),
                    max_retries_per_iteration=task_config["loop"].get(
                        "max_retries_per_iteration", 2
                    ),
                    max_steps_pagination=task_config["loop"].get(
                        "max_steps_pagination", 20
                    ),
                )
                wf_kwargs: dict[str, Any] = {
                    "brain": config.brain,
                    "env": config.env,
                    "loop_config": loop_cfg,
                    "on_iteration": _make_on_loop_iteration(
                        task_id, task_start, scores, task_details, save_progress
                    ),
                    "start_url": task_config.get("start_url", ""),
                    "grounding": config.grounding,
                    "on_step": on_step,
                    "use_sub_plan": config.use_sub_plan,
                    "fallback_brain": config.fallback_brain,
                    "fallback_label": config.fallback_label,
                    "fallback_micro_retries": task_config.get(
                        "fallback_micro_retries",
                        config.fallback_micro_retries,
                    ),
                    "fallback_micro_max_steps": task_config.get(
                        "fallback_max_steps",
                        config.fallback_micro_max_steps,
                    ),
                }
                if config.extractor:
                    wf_kwargs["extractor"] = config.extractor
                wf_runner = WorkflowRunner(**wf_kwargs)
                results = wf_runner.run_loop()

                final = _build_loop_final_detail(task_id, results, task_start)
                viable = final["viable"]
                success = final["success"]

                if task_details and task_details[-1].get("task_id") == task_id:
                    task_details[-1] = final
                    scores[-1] = 1.0 if success else 0.0
                else:
                    scores.append(1.0 if success else 0.0)
                    task_details.append(final)
                print(
                    f"  Loop: {viable}/{final['iterations']} viable "
                    f"({final['real_iterations']} real, {final['parse_failures']} parse failures)"
                )
                save_progress()
                if config.on_loop_complete:
                    config.on_loop_complete()
                if (
                    not success
                    and config.stop_on_task_failure
                    and not task_config.get("continue_on_failure", False)
                ):
                    print(f"  Stopping task loop after failed task '{task_id}'")
                    break
                continue

            # ── Standard task ──
            runner = GymRunner(
                brain=config.brain,
                env=config.env,
                max_steps=_standard_task_max_steps(task_config, config),
                frames_per_inference=config.frames_per_inference,
                grounding=config.grounding,
                on_step=on_step,
            )
            result = runner.run(
                task=intent,
                task_id=task_id,
                start_url=task_config.get("start_url", ""),
            )
            _merge_runner_costs(runner)

            if not result.success and config.fallback_brain is not None:
                fallback_retries = int(
                    task_config.get(
                        "fallback_micro_retries",
                        config.fallback_micro_retries,
                    )
                )
                fallback_max_steps = int(
                    task_config.get(
                        "fallback_max_steps",
                        config.fallback_micro_max_steps,
                    )
                )
                resume_max_steps = int(
                    task_config.get(
                        "resume_max_steps",
                        _standard_task_max_steps(task_config, config),
                    )
                )

                for attempt in range(1, max(fallback_retries, 0) + 1):
                    fallback_intent = task_config.get("fallback_intent")
                    if not fallback_intent:
                        fallback_intent = _build_micro_fallback_intent(
                            task_id=task_id,
                            intent=intent,
                            result=result,
                            attempt=attempt,
                        )
                    print(
                        f"  {config.fallback_label} micro-fallback: "
                        f"recovering stuck section '{task_id}' "
                        f"(attempt {attempt}/{fallback_retries})"
                    )
                    fallback_runner = GymRunner(
                        brain=config.fallback_brain,
                        env=config.env,
                        max_steps=fallback_max_steps,
                        frames_per_inference=config.frames_per_inference,
                        grounding=config.grounding,
                        on_step=on_step,
                    )
                    micro_result = fallback_runner.run(
                        task=fallback_intent,
                        task_id=(
                            f"{task_id}_{config.fallback_label}"
                            f"_micro_fallback_{attempt}"
                        ),
                        start_url=task_config.get("fallback_start_url", ""),
                    )
                    _merge_runner_costs(fallback_runner)
                    try:
                        setattr(micro_result, "fallback_used", config.fallback_label)
                    except (AttributeError, TypeError):
                        pass

                    if not micro_result.success:
                        result = micro_result
                        continue

                    print(
                        f"  Primary resume: retrying section '{task_id}' "
                        "after micro-fallback"
                    )
                    resume_runner = GymRunner(
                        brain=config.brain,
                        env=config.env,
                        max_steps=resume_max_steps,
                        frames_per_inference=config.frames_per_inference,
                        grounding=config.grounding,
                        on_step=on_step,
                    )
                    result = resume_runner.run(
                        task=intent,
                        task_id=(
                            f"{task_id}_resume_after_{config.fallback_label}"
                            f"_{attempt}"
                        ),
                        start_url="",
                    )
                    _merge_runner_costs(resume_runner)
                    try:
                        setattr(result, "fallback_used", config.fallback_label)
                    except (AttributeError, TypeError):
                        pass
                    if result.success:
                        break

            # Executor-specific post-processing (filter validation, retries, etc.)
            if config.on_task_result:
                result = config.on_task_result(
                    task_config, task_id, result, config.env, config.brain, config
                )

            if task_config.get("save_session"):
                if result.success or (
                    "login" not in config.env.current_url.lower()
                ):
                    config.env.save_session(config.session_name)

            success = result.success
            scores.append(1.0 if success else 0.0)
            detail_entry: dict[str, Any] = {
                "task_id": task_id,
                "success": success,
                "steps": result.total_steps,
                "duration_s": round(time.time() - task_start),
                "termination_reason": result.termination_reason,
                "final_url": config.env.current_url,
            }
            if last_cost_meter is not None:
                detail_entry["costs"] = last_cost_meter.totals_from(
                    task_cost_acc
                )
            task_details.append(detail_entry)
            print(f"  {'PASS' if success else 'FAIL'} ({result.total_steps} steps)")

            if (
                not success
                and task_config.get("continue_on_failure")
                and ("setup" in task_id or "filter" in task_id)
            ):
                print(
                    f"  WARNING: Setup '{task_id}' failed — continuing with current page state"
                )

            # Executor-specific side effects (trajectory saving, etc.)
            if config.on_task_complete:
                config.on_task_complete(task_id, intent, result)

        except Exception as e:
            traceback.print_exc()
            print(f"  ERROR: {e}")
            scores.append(0.0)
            error_detail: dict[str, Any] = {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "duration_s": round(time.time() - task_start),
            }
            # #350: even on exception, attribute whatever spend happened
            # before the throw — partial runs still cost real money.
            if last_cost_meter is not None:
                error_detail["costs"] = last_cost_meter.totals_from(
                    task_cost_acc
                )
            task_details.append(error_detail)

        save_progress()
        if (
            task_details
            and task_details[-1].get("task_id") == task_id
            and not task_details[-1].get("success", False)
            and config.stop_on_task_failure
            and not task_config.get("continue_on_failure", False)
        ):
            print(f"  Stopping task loop after failed task '{task_id}'")
            break

    return scores, task_details


# ── Executor lifecycle ────────────────────────────────────────────


def run_executor_lifecycle(
    task_suite: dict[str, Any],
    config: TaskLoopConfig,
    *,
    server_proc: Any | None = None,
    proxy_proc: Any | None = None,
    viewer_ctx: Any | None = None,
    t0: float | None = None,
) -> dict[str, Any]:
    """Run the full executor lifecycle: banner + task loop + cleanup + result.

    Returns a standardized result dict.
    """
    t0 = t0 or time.time()
    tasks = task_suite.get("tasks", [])

    print(f"\n{'=' * 60}")
    print(f"Mantis CUA Server — {config.model_name}")
    print(f"  Session:  {config.session_name}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"{'=' * 60}")

    try:
        scores, task_details = run_task_loop(tasks, config, t0=t0)
    finally:
        config.env.close()
        if server_proc:
            server_proc.terminate()
        if proxy_proc:
            proxy_proc.terminate()
        if viewer_ctx:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass

    passed = sum(1 for s in scores if s > 0)
    avg = sum(scores) / len(scores) * 100 if scores else 0
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {passed}/{len(scores)} ({avg:.1f}%)")
    print(f"Time: {elapsed / 60:.0f} min")
    print(f"{'=' * 60}")

    return {"passed": passed, "total": len(scores), "score": avg}
