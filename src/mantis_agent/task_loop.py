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
from typing import Any, Callable

from .server_utils import build_proxy_config, resolve_proxy_server

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


def setup_env(
    *,
    base_url: str,
    run_id: str,
    session_name: str,
    settle_time: float = 2.0,
    proxy_city: str = "miami",
    proxy_state: str = "",
    proxy_provider: str = "",
    proxy_disabled: bool = False,
    display: str | None = None,
    start_xvfb: bool = False,
    browser: str = "google-chrome",
    viewport: tuple[int, int] = (1280, 720),
    profile_dir: str = "",
    save_screenshots_dir: str = "/data/screenshots",
) -> tuple[Any, Any]:
    """Set up proxy + XdotoolGymEnv.

    Args:
        proxy_disabled: When True, skip the upstream proxy entirely. Use for
            test/internal sites that don't need bot-detection bypass and
            shouldn't depend on the residential proxy's availability.

    Returns (env, proxy_proc_or_None).
    """
    from .gym.xdotool_env import XdotoolGymEnv

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
        )
        proxy_server, proxy_proc = resolve_proxy_server(proxy)
        if proxy:
            provider = proxy_provider or os.environ.get("MANTIS_PROXY_PROVIDER") or "iproyal"
            print(f"  Proxy: {provider} via {proxy.get('server', '')}")

    if start_xvfb and display:
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
    }
    if display:
        env_kwargs["display"] = display
    if profile_dir:
        env_kwargs["profile_dir"] = profile_dir

    env = XdotoolGymEnv(**env_kwargs)
    return env, proxy_proc


def setup_viewer(enabled: bool) -> tuple[Any, Any]:
    """Set up the Modal viewer if enabled.

    Returns (viewer_ctx, viewer_event_bus). Both None if disabled or failed.
    """
    if not enabled:
        return None, None
    try:
        from .viewer_modal import modal_viewer

        viewer_ctx = modal_viewer()
        viewer_event_bus, _viewer_url = viewer_ctx.__enter__()
        return viewer_ctx, viewer_event_bus
    except Exception as e:
        print(f"  Viewer failed to start: {e}")
        return None, None


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
                max_steps=task_config.get("max_steps", config.max_steps),
                frames_per_inference=config.frames_per_inference,
                grounding=config.grounding,
                on_step=on_step,
            )
            result = runner.run(
                task=intent,
                task_id=task_id,
                start_url=task_config.get("start_url", ""),
            )

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
                        task_config.get("max_steps", config.max_steps),
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
            task_details.append(
                {
                    "task_id": task_id,
                    "success": success,
                    "steps": result.total_steps,
                    "duration_s": round(time.time() - task_start),
                    "termination_reason": result.termination_reason,
                    "final_url": config.env.current_url,
                }
            )
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
            task_details.append(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "duration_s": round(time.time() - task_start),
                }
            )

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
