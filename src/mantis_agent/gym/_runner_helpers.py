"""Module-level helpers extracted from MicroPlanRunner — final phase of EPIC #161 #162.

Every function takes the runner as its first arg and is reachable via
``MicroPlanRunner.__getattr__`` / direct import. Behavior is unchanged
from the in-runner originals; this split is purely for LOC and
testability.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

from ..actions import Action, ActionType
from ..plan_decomposer import MicroIntent
from . import step_snapshot
from .checkpoint import REVERSE_ACTIONS, RunnerResult, StepResult, PauseState
from .run_reporter import RunReporter

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan
    from .micro_runner import MicroPlanRunner

logger = logging.getLogger("mantis_agent.gym.micro_runner")


# ── Delegation tables — back-compat surface for MicroPlanRunner ────

SCANNER_FIELDS: dict[str, str] = {
    "_seen_urls": "seen_urls",
    "_extracted_titles": "extracted_titles",
    "_page_listings": "page_listings",
    "_page_listing_index": "page_listing_index",
    "_viewport_stage": "viewport_stage",
    "_max_viewport_stages": "max_viewport_stages",
    "_results_base_url": "results_base_url",
    "_required_filter_tokens": "required_filter_tokens",
    "_listings_on_page": "listings_attempted",
}

COLLAB_METHODS: dict[str, tuple[str, str]] = {
    "_record_step_costs": ("cost_meter", "record_step"),
    "_cost_totals": ("cost_meter", "totals"),
    "_emit_cost_gauges": ("cost_meter", "emit_inflight_gauges"),
    "_persist_checkpoint": ("checkpoint_manager", "persist"),
    "_restore_from_checkpoint": ("checkpoint_manager", "restore"),
    "_checkpoint_active_progress": ("checkpoint_manager", "save_active_progress"),
    "_current_results_page_url": ("browser_state", "current_results_page_url"),
    "_reentry_url_for_step": ("browser_state", "reentry_url_for_step"),
    "_set_scroll_state": ("browser_state", "set_scroll_state"),
    "_update_scroll_state_from_trajectory": (
        "browser_state", "update_scroll_state_from_trajectory",
    ),
    "_restore_scroll_position": ("browser_state", "restore_scroll_position"),
    "_resume_browser_state": ("browser_state", "resume_browser_state"),
    "register_tool": ("tool_channel", "register"),
    "list_tools": ("tool_channel", "list"),
    "call_tool": ("tool_channel", "call"),
    "_invoke_tool": ("tool_channel", "invoke"),
}

HELPER_METHODS: dict[str, str] = {
    "_capture_screenshot_bytes": "capture_screenshot_bytes",
    "_enforce_screenshot_cap": "enforce_screenshot_cap",
    "_invoke_step_callback": "invoke_step_callback",
    "_is_cancelled": "is_cancelled",
    "_log_progress": "log_progress",
    "_log_step_diff": "log_step_diff",
    "_best_effort_current_url": "best_effort_current_url",
    "_safe_screenshot": "safe_screenshot",
    "_dump_debug_screenshot": "dump_debug_screenshot",
    "_adaptive_submit_settle": "adaptive_submit_settle",
    "_read_current_url": "read_current_url",
    "_return_to_results_page": "return_to_results_page",
    "_execute_close_detail_tab": "execute_close_detail_tab",
    "_url_has_required_filters": "url_has_required_filters",
    "_reset_results_scan_state": "reset_results_scan_state",
    "_ensure_results_filters": "ensure_results_filters",
    "_check_verify": "check_verify",
    "_current_item_label": "current_item_label",
    "_reverse_step": "reverse_step",
    "_plan_aware_reverse_actions": "plan_aware_reverse_actions",
    "_build_step_context": "build_step_context",
    "_execute_step": "execute_step",
    "_execute_navigate": "execute_navigate",
    "_execute_claude_guided_click": "execute_claude_guided_click",
    "_execute_claude_guided_filter": "execute_claude_guided_filter",
    "_execute_claude_guided_form": "execute_claude_guided_form",
    "_execute_paginate_layered": "execute_paginate_layered",
    "_execute_claude_guided_paginate": "execute_claude_guided_paginate",
    "_execute_holo3_step": "execute_holo3_step",
    "_execute_claude_step": "execute_claude_step",
    "_extract_listing_data_deep": "extract_listing_data_deep",
    "_final_summary": "final_summary",
    "_build_runner_result": "build_runner_result",
}


# ── Adaptive submit settle ─────────────────────────────────────────────

_SUBMIT_SETTLE_MAX_SECONDS: float = 8.0
_SUBMIT_SETTLE_POLL_SECONDS: float = 0.5
_SUBMIT_SETTLE_MIN_SECONDS: float = 1.0


def best_effort_current_url(runner: "MicroPlanRunner") -> str:
    try:
        return str(getattr(runner.env, "current_url", "") or "")
    except Exception as exc:
        logger.debug("current_url read failed: %s", exc)
        return ""


def safe_screenshot(runner: "MicroPlanRunner") -> Any:
    env = runner.env
    try:
        return env.screenshot() if hasattr(env, "screenshot") else None
    except Exception as exc:
        logger.debug("safe screenshot failed: %s", exc)
        return None


def dump_debug_screenshot(
    runner: "MicroPlanRunner", name_stem: str, screenshot: Any
) -> None:
    if screenshot is None:
        return
    dump_dir = os.environ.get("MANTIS_DEBUG_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    try:
        target_dir = os.path.join(
            dump_dir, runner.run_key or runner.session_name or "default",
        )
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, f"{name_stem}.png")
        screenshot.save(target, format="PNG", optimize=True)
        logger.debug("dumped debug screenshot: %s", target)
    except Exception as exc:
        logger.debug("debug screenshot dump failed: %s", exc)


def screenshot_pixel_diff_fraction(a: Any, b: Any, brightness_step: int = 8) -> float:
    """Fraction of pixels that differ between two screenshots
    (downsampled, grayscale). Issue #259 — primitive for adaptive
    content-settle on the deep-extract path.

    Cheap heuristic: downsample to 64×64, convert to grayscale,
    count pixels whose brightness diverged by more than
    ``brightness_step``. 1-step shifts (JPEG re-encode artifacts,
    cursor blinks) stay under the threshold; real content changes
    (new text appearing, image swap, modal overlay) cross it
    decisively.

    Returns 1.0 (conservative — assume changed) for any failure
    case: size mismatch, invalid input, exception during compare.
    Callers treat 1.0 as "page is still changing, keep polling".
    """
    if a is None or b is None:
        return 1.0
    try:
        if getattr(a, "size", None) != getattr(b, "size", None):
            return 1.0
        from PIL import Image  # local import — pillow may not be loaded yet
        a_small = a.convert("L").resize((64, 64), Image.NEAREST)
        b_small = b.convert("L").resize((64, 64), Image.NEAREST)
    except Exception:
        return 1.0
    try:
        a_data = list(a_small.getdata())
        b_data = list(b_small.getdata())
    except Exception:
        return 1.0
    if not a_data or len(a_data) != len(b_data):
        return 1.0
    diff = sum(1 for x, y in zip(a_data, b_data) if abs(x - y) > brightness_step)
    return diff / len(a_data)


def adaptive_content_settle(
    env: Any,
    *,
    min_seconds: float = 0.3,
    max_seconds: float = 2.0,
    poll_seconds: float = 0.3,
    diff_threshold: float = 0.02,
) -> float:
    """Poll-based settle that exits when consecutive screenshots
    stabilize. Symmetric to :func:`adaptive_submit_settle` but
    polls *screenshot stability* instead of URL change.

    Used after actions that may trigger lazy-load, animation, or
    XHR-driven DOM updates — the four fixed-sleep sites in
    ``_extract_listing_data_deep`` (issue #259). Worst-case the
    helper pays ``max_seconds`` (the original fixed-sleep value).
    Best-case (static page) it exits at
    ``min_seconds + poll_seconds`` (~0.6s) — the floor reflects
    the truth that even a stable page needs at least one render
    tick.

    Args:
        env: Object with a ``screenshot()`` method returning a
            PIL Image. Any exception during screenshot capture
            falls through to a fixed-sleep equivalent — better to
            over-pay than to halt the deep-extract.
        min_seconds: Always wait at least this long. Captures the
            initial-render reality on every page.
        max_seconds: Worst-case cap. Pages that genuinely keep
            changing (long XHR, animations) pay this and proceed.
        poll_seconds: Wall time between consecutive screenshot
            polls. ~0.2–0.3s is the right zone — small enough to
            exit promptly, large enough that the screenshot cost
            doesn't dominate.
        diff_threshold: Pixel-diff fraction below which two
            consecutive screenshots count as "settled". 0.02 (2%
            of downsampled pixels) is generous enough that
            animation tails / cursor blinks don't extend the
            settle.

    Returns:
        Actual elapsed wall-clock seconds. Callers add to
        ``runner.costs['gpu_seconds']`` for telemetry.
    """
    if diff_threshold < 0:
        raise ValueError(f"diff_threshold must be >= 0, got {diff_threshold}")
    if min_seconds > max_seconds:
        raise ValueError(
            f"min_seconds ({min_seconds}) must be <= max_seconds ({max_seconds})"
        )

    start = time.time()
    time.sleep(min_seconds)

    try:
        prev = env.screenshot()
    except Exception as exc:
        logger.debug("adaptive_content_settle: screenshot raised — fall through (%s)", exc)
        # Fall through to fixed-sleep equivalent — better to
        # over-pay than to halt.
        remaining = max_seconds - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
        return time.time() - start

    while True:
        elapsed = time.time() - start
        if elapsed >= max_seconds:
            return elapsed
        time.sleep(min(poll_seconds, max_seconds - elapsed))
        try:
            cur = env.screenshot()
        except Exception:
            break
        if screenshot_pixel_diff_fraction(prev, cur) < diff_threshold:
            return time.time() - start
        prev = cur

    # Loop broke on exception — pay remaining max_seconds budget.
    remaining = max_seconds - (time.time() - start)
    if remaining > 0:
        time.sleep(remaining)
    return time.time() - start


def adaptive_submit_settle(runner: "MicroPlanRunner", *, url_before: str) -> float:
    deadline = time.time() + _SUBMIT_SETTLE_MAX_SECONDS
    time.sleep(_SUBMIT_SETTLE_MIN_SECONDS)
    elapsed = _SUBMIT_SETTLE_MIN_SECONDS
    while time.time() < deadline:
        current = best_effort_current_url(runner)
        if current and url_before and current != url_before:
            logger.debug(
                "  [settle] url changed after %.1fs: %s → %s",
                elapsed, url_before[:40], current[:40],
            )
            return elapsed
        time.sleep(_SUBMIT_SETTLE_POLL_SECONDS)
        elapsed += _SUBMIT_SETTLE_POLL_SECONDS
    logger.debug(
        "  [settle] no url change within %.1fs (url=%s)",
        _SUBMIT_SETTLE_MAX_SECONDS, url_before[:60],
    )
    return _SUBMIT_SETTLE_MAX_SECONDS


# ── Observability ─────────────────────────────────────────────────────


def capture_screenshot_bytes(runner: "MicroPlanRunner") -> bytes | None:
    env = runner.env
    if env is None or not hasattr(env, "screenshot"):
        return None
    try:
        img = env.screenshot()
    except Exception as exc:
        logger.debug("screenshot capture failed: %s", exc)
        return None
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as exc:
        logger.debug("screenshot encode failed: %s", exc)
        return None


def enforce_screenshot_cap(
    runner: "MicroPlanRunner", results: list[StepResult]
) -> None:
    cap = runner.keep_screenshots
    if cap is None or cap < 0:
        return
    kept = 0
    for r in reversed(results):
        if r.screenshot_png is None:
            continue
        if kept >= cap:
            r.screenshot_png = None
        else:
            kept += 1


def invoke_step_callback(runner: "MicroPlanRunner", result: StepResult) -> None:
    cb = runner.step_callback
    if cb is None:
        return
    try:
        cb(result.step_index, result.intent, result.last_action, result.success)
    except Exception as exc:
        logger.warning("step_callback raised: %s", exc)


def is_cancelled(runner: "MicroPlanRunner") -> bool:
    ev = runner.cancel_event
    if ev is None:
        return False
    try:
        if callable(ev):
            return bool(ev())
        return bool(ev.is_set())
    except Exception:
        return False


def log_progress(
    runner: "MicroPlanRunner",
    step_result: StepResult,
    results: list[StepResult],
) -> None:
    gpu_cost, claude_cost, proxy_cost, total_cost = runner.cost_meter.totals()
    elapsed = time.time() - runner._run_start
    print(RunReporter.step_progress_line(
        step_index=step_result.step_index,
        success=step_result.success,
        results=results,
        gpu_cost=gpu_cost,
        claude_cost=claude_cost,
        proxy_cost=proxy_cost,
        total_cost=total_cost,
        elapsed_seconds=elapsed,
    ))
    runner.cost_meter.emit_inflight_gauges(
        gpu_cost, claude_cost, proxy_cost, total_cost
    )


def log_step_diff(
    runner: "MicroPlanRunner",
    pre_snapshot: "step_snapshot.StepStateSnapshot",
    step: MicroIntent,
    step_result: StepResult,
) -> None:
    try:
        post_snapshot = step_snapshot.capture(runner)
        delta = step_snapshot.diff(pre_snapshot, post_snapshot)
    except Exception as exc:
        logger.debug("step diff capture failed: %s", exc)
        return
    outcome = "ok" if step_result.success else "fail"
    logger.info(
        "  [diff] %s/%s step=%s: %s",
        step.type, outcome, step_result.step_index, delta.summary(),
    )


# ── URL resolver / tab handling ──────────────────────────────────────


def read_current_url(
    runner: "MicroPlanRunner", screenshot: Any = None
) -> str:
    cdp_url = ""
    cdp_error: Exception | None = None
    try:
        raw = getattr(runner.env, "current_url", "")
        if callable(raw):
            raw = raw()
        cdp_url = (raw or "").strip() if isinstance(raw, str) else ""
    except Exception as exc:
        cdp_error = exc
    if cdp_url:
        logger.info(f"  [url] cdp={cdp_url[:80]}")
        return cdp_url
    if cdp_error is not None:
        logger.info(
            f"  [url] cdp unavailable ({type(cdp_error).__name__}); "
            "falling back to OCR"
        )
    elif screenshot is not None:
        logger.info("  [url] cdp empty; falling back to OCR")
    if screenshot is not None and runner.extractor:
        try:
            verify_data = runner.extractor.extract(screenshot)
        except Exception as exc:
            logger.info(f"  [url] ocr extract failed: {exc}")
            return ""
        runner.costs["claude_extract"] += 1
        ocr_url = (verify_data.url or "") if verify_data else ""
        logger.info(f"  [url] ocr={ocr_url[:80] or '<empty>'}")
        return ocr_url
    return ""


def return_to_results_page(runner: "MicroPlanRunner") -> None:
    if runner._opened_detail_in_new_tab:
        runner.env.step(Action(
            action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+w"},
        ))
        runner._opened_detail_in_new_tab = False
    else:
        runner.env.step(Action(
            action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"},
        ))
    time.sleep(2)


def execute_close_detail_tab(
    runner: "MicroPlanRunner", step: MicroIntent, index: int
) -> StepResult:
    try:
        return_to_results_page(runner)
        if runner.extractor:
            screenshot = runner.env.screenshot()
            check = runner.extractor.extract(screenshot)
            runner.costs["claude_extract"] += 1
            url = check.url if check else ""
            if url:
                runner._last_known_url = url
            if url and runner.site_config.is_detail_page(url):
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                )
        return StepResult(
            step_index=index, intent=step.intent, success=True, steps_used=1,
        )
    except Exception as exc:
        logger.warning("  [back] Failed closing detail tab: %s", exc)
        return StepResult(step_index=index, intent=step.intent, success=False)


# ── Results-filter helpers ────────────────────────────────────────────


def extract_url_from_intent(intent: str) -> str:
    match = re.search(r'https?://[^\s"]+', intent)
    return match.group() if match else ""


def derive_filter_tokens(url: str) -> tuple[str, ...]:
    match = re.search(r'https?://[^/]+/([^?#]+)', url)
    if not match:
        return ()
    tokens = []
    for token in match.group(1).strip("/").split("/"):
        if not token or token in {"boats"} or token.startswith("page-"):
            continue
        tokens.append(token.lower())
    return tuple(tokens)


def url_has_required_filters(runner: "MicroPlanRunner", url: str) -> bool:
    url_lower = url.lower()
    is_results = (
        runner.site_config.is_results_page(url)
        if runner.site_config.results_page_pattern else bool(url_lower)
    )
    is_detail = (
        runner.site_config.is_detail_page(url)
        if runner.site_config.detail_page_pattern else False
    )
    return (
        bool(url_lower)
        and is_results
        and not is_detail
        and all(token in url_lower for token in runner._required_filter_tokens)
    )


def reset_results_scan_state(runner: "MicroPlanRunner") -> None:
    runner._page_listings = []
    runner._page_listing_index = 0
    runner._viewport_stage = 0


def ensure_results_filters(
    runner: "MicroPlanRunner", index: int, force_reload: bool = False,
) -> bool:
    if (
        not runner.extractor
        or not runner._results_base_url
        or not runner._required_filter_tokens
    ):
        return True

    url = ""
    screenshot = None
    try:
        screenshot = runner.env.screenshot()
        data = runner.extractor.extract(screenshot)
        runner.costs["claude_extract"] += 1
        url = data.url if data else ""
    except Exception as e:
        logger.warning("  [filters] URL verification failed: %s", e)

    if not force_reload and url_has_required_filters(runner, url):
        runner._last_known_url = url
        runner.dynamic_verifier.record_filter_check(
            page=runner._current_page, url=url, passed=True,
            reason="url_contains_required_filters",
        )
        return True

    if not force_reload and not url and screenshot is not None:
        gate_prefix = (
            runner.site_config.gate_verify_prompt
            or "Page is a filtered results page with these active filters: "
        )
        requirement = gate_prefix + ", ".join(runner._required_filter_tokens)
        try:
            # #523 PR B-4 — capture the verify_gate LLM call as a
            # ``verifier`` modelio record. No-op when augur is None.
            from ..observability.modelio import publish_modelio_context
            with publish_modelio_context(
                getattr(runner, "_augur", None),
                layer="verifier", step_index=index,
            ):
                passed, reason = runner.extractor.verify_gate(screenshot, requirement)
            runner.costs["claude_extract"] += 1
            if passed:
                logger.info(
                    "  [filters] Visual filter gate passed despite unreadable URL"
                )
                runner._last_known_url = (
                    runner._current_results_page_url() or runner._results_base_url
                )
                runner.dynamic_verifier.record_filter_check(
                    page=runner._current_page, url=runner._last_known_url,
                    passed=True, reason="visual_gate_passed",
                )
                return True
            logger.warning(
                "  [filters] Visual filter gate failed: %s", reason[:120]
            )
            runner.dynamic_verifier.record_filter_check(
                page=runner._current_page, url=url, passed=False,
                reason=reason[:200],
            )
        except Exception as e:
            logger.warning("  [filters] Visual filter gate errored: %s", e)

    logger.warning(
        "  [filters] Reloading canonical filtered results before step %s "
        "(current url=%s, required=%s)",
        index, url[:120], ",".join(runner._required_filter_tokens),
    )
    try:
        reload_url = (
            runner._current_results_page_url() or runner._results_base_url
        )
        runner.env.reset(task="navigate", start_url=reload_url)
        time.sleep(12)
        runner.env.step(Action(
            action_type=ActionType.KEY_PRESS, params={"keys": "Home"},
        ))
        time.sleep(2)
        reset_results_scan_state(runner)
        runner._last_known_url = reload_url
        runner._set_scroll_state(
            context="results_top", url=reload_url,
            page_downs=0, wheel_downs=0,
        )
        runner.dynamic_verifier.record_filter_check(
            page=runner._current_page, url=reload_url, passed=True,
            reason="reloaded_canonical_filtered_results",
        )
        return True
    except Exception as e:
        logger.error("  [filters] Failed to reload filtered results: %s", e)
        runner.dynamic_verifier.record_filter_check(
            page=runner._current_page, url=url, passed=False,
            reason=f"reload_failed:{e}",
        )
        return False


def check_verify(
    runner: "MicroPlanRunner",
    verify_condition: str,
    extract_data: Any,
    screenshot: Any,
) -> bool:
    v = verify_condition.lower()
    url = extract_data.url if extract_data else ""
    if "detail page" in v or "page opens" in v:
        if url and runner.site_config.is_detail_page(url):
            return True
        if (
            url
            and runner.site_config.is_results_page(url)
            and not runner.site_config.is_detail_page(url)
        ):
            logger.info("  [verify] Still on search page, not detail page")
            return False
        if not url:
            return True
    if "selected" in v or "highlighted" in v or "filter" in v:
        return True
    if "new listings" in v:
        return True
    return True


def current_item_label(runner: "MicroPlanRunner", data: Any = None) -> str:
    title = (
        runner._last_extracted.get("last_clicked_title")
        or getattr(runner, "_last_click_title", "")
    )
    if title:
        return str(title)
    if data is not None:
        year = getattr(data, "year", "") or ""
        make = getattr(data, "make", "") or ""
        model = getattr(data, "model", "") or ""
        label = " ".join(part for part in (year, make, model) if part).strip()
        if label:
            return label
        url = getattr(data, "url", "") or ""
        if url:
            return url
    return runner._last_extracted.get("last_attempted_url") or "unknown"


# ── Reverse-step (plan-aware undo) ───────────────────────────────────


def reverse_decision_from_diff(
    step: MicroIntent, delta: "step_snapshot.StepDiff",
) -> list[tuple[str, str]] | None:
    if delta.url_changed:
        return [("key_press", "alt+Left")]
    if delta.extraction_added or delta.new_urls_seen:
        return []
    if not delta.has_changes:
        return []
    if delta.focus_changed and not (
        delta.scroll_changed or delta.viewport_changed or delta.page_changed
    ):
        return [("key_press", "Escape")]
    return None


def plan_aware_reverse_actions(
    runner: "MicroPlanRunner", step: MicroIntent,
) -> list[tuple[str, str]]:
    pre = runner._pre_step_snapshot
    if pre is not None:
        try:
            post = step_snapshot.capture(runner)
            delta = step_snapshot.diff(pre, post)
        except Exception as exc:
            logger.debug("plan-aware reverse diff failed: %s", exc)
            delta = None
        if delta is not None:
            decision = reverse_decision_from_diff(step, delta)
            if decision is not None:
                logger.info(
                    "  [reverse:plan-aware] %s — diff: %s",
                    "skip" if not decision else f"{len(decision)} action(s)",
                    delta.summary(),
                )
                return decision
    actions = list(REVERSE_ACTIONS.get(step.type, []))
    if step.reverse:
        if "escape" in step.reverse.lower():
            actions = [("key_press", "Escape")] + actions
        if "alt+left" in step.reverse.lower():
            actions.append(("key_press", "alt+Left"))
    return actions


def reverse_step(runner: "MicroPlanRunner", step: MicroIntent) -> None:
    actions = plan_aware_reverse_actions(runner, step)
    for action_type, keys in actions:
        try:
            runner.env.step(Action(
                action_type=ActionType.KEY_PRESS, params={"keys": keys},
            ))
            time.sleep(0.5)
        except Exception:
            pass
    logger.info(f"  [reverse] {len(actions)} actions applied")


# ── Step dispatch + context ───────────────────────────────────────────


def build_step_context(runner: "MicroPlanRunner", index: int):
    from .step_context import StepContext
    return StepContext(
        env=runner.env,
        brain=runner.brain,
        extractor=runner.extractor,
        grounding=runner.grounding,
        cost_meter=runner.cost_meter,
        dynamic_verifier=runner.dynamic_verifier,
        scanner=runner.scanner,
        site_config=runner.site_config,
        routing_policy=getattr(runner, "routing_policy", None),
        tool_channel=runner.tool_channel,
        extraction_cache=runner.extraction_cache,
        # #406: form_target_provider lifecycle lives on the runner so a
        # single provider instance services every step (caches stay
        # warm, factory cost is paid once per run). When the runner
        # didn't wire one, ``None`` flows through and the form handler
        # falls back to the extractor's compat shims.
        form_target_provider=getattr(runner, "form_target_provider", None),
        state={"index": index},
    )


def execute_step(
    runner: "MicroPlanRunner", step: MicroIntent, index: int,
) -> StepResult:
    registry = runner._handler_registry

    # Agentic handler-escalation — issue #224 follow-up. When prior
    # failures on this step indicate the default handler is the
    # wrong tool (canonical case: text-matching submit handler keeps
    # clicking elements that match a label but produce no
    # navigation), the executor sets
    # ``runner._step_handler_override[index]`` so the next attempt
    # routes through a different handler. Currently the only escalation
    # target is ``"holo3"`` — the brain-grounded loop that operates
    # on intent prose rather than text-matching specific labels.
    # General-purpose: the override is set purely from observed
    # failure patterns, no hardcoded plan content.
    override = (
        getattr(runner, "_step_handler_override", {}).get(index)
        if hasattr(runner, "_step_handler_override") else None
    )
    if override == "holo3" and runner.brain is not None:
        # Bump the budget on escalation — the canonical case is
        # "scroll down to find an off-screen target then click it",
        # but in CRM / settings flows the target may also live on a
        # LATER page that requires pagination. 25 brain turns covers
        # scroll + paginate + multi-page search without being
        # extravagant; the original submit budget (3-5) was nowhere
        # near enough.
        escalated_budget = max(step.budget, 25)

        # Augment the task prose with failure context. Holo3 sees the
        # screenshots and reasons about page state directly, but
        # giving it the prior failure trace means it doesn't re-pick
        # the same wrong elements the form handler already tried, and
        # gets explicit licence to paginate when the visible viewport
        # has no matching target. The escalation only fires after
        # 2+ confirmed no-state-change clicks, so the failure trace
        # is already information the brain needs.
        history = (
            getattr(runner, "_step_failure_history", {}).get(index, [])
            if hasattr(runner, "_step_failure_history") else []
        )
        augmented_intent = step.intent
        if history:
            wrong_targets = [
                f"({r.get('x', '?')}, {r.get('y', '?')}) labelled "
                f"'{r.get('label', '?')}'"
                for r in history[-3:]
            ]
            augmented_intent = (
                f"{step.intent}\n\n"
                f"CONTEXT: previous attempts to satisfy this step clicked "
                f"on these targets without changing the page UI: "
                f"{', '.join(wrong_targets)}. "
                f"That means those clicks landed on text that LOOKED like "
                f"the target (status badge, filter chip, label) but is "
                f"not actually the navigable element. Possible reasons:\n"
                f"  - the real target is below the fold (scroll down to find it)\n"
                f"  - the real target is on a LATER page (use pagination "
                f"controls — Next button, page-number links, or scroll "
                f"the table to load more rows)\n"
                f"  - the visible page state genuinely has zero matching "
                f"items; check filter / sort / status indicators to "
                f"confirm before giving up\n"
                f"Pick a different element or navigation action than the "
                f"prior attempts.\n\n"
                f"SUCCESS CRITERION: the step is COMPLETE the moment the "
                f"page navigates (URL changes, or the visible content "
                f"clearly transitions from a list / form to a detail / "
                f"target view). The MOMENT that happens, emit a `done` "
                f"action with success=true and a one-line summary. Do "
                f"NOT take any further actions on the new page — that "
                f"would undo the navigation you just achieved. Subsequent "
                f"plan steps will operate on the new view; your job ends "
                f"when the navigation succeeds."
            )

        escalated_step = MicroIntent(
            intent=augmented_intent, type=step.type,
            verify=step.verify, budget=escalated_budget,
            reverse=step.reverse, grounding=step.grounding,
            section=step.section, required=step.required,
            gate=step.gate, claude_only=step.claude_only,
            loop_target=step.loop_target,
            loop_count=step.loop_count,
            params=step.params, hints=step.hints,
        )
        # WARNING level so the trace survives Modal's INFO-suppressed
        # log capture. Firing this is the canonical signal that the
        # agentic-handler-escalation path engaged on a retry attempt.
        logger.warning(
            f"  [escalation] step {index} routing via Holo3StepHandler "
            f"(brain-grounded loop, budget={escalated_budget}, "
            f"was={step.budget}, history={len(history)} prior failure(s)) "
            f"— original step.type={step.type}"
        )
        return execute_holo3_step(runner, escalated_step, index)

    if step.type == "click" and runner.extractor:
        layout_hint = (step.hints or {}).get("layout", "")
        is_listings = (
            layout_hint == "listings"
            or (not layout_hint and step.section == "extraction")
        )
        ctx = build_step_context(runner, index)
        if is_listings:
            if not ensure_results_filters(runner, index):
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="filters_not_applied",
                )
            return _stamp_backend(registry.get("click").execute(step, ctx), ctx)
        # Preserve ``kind`` and ``aliases`` from the original click step
        # so downstream form-handler logic (Tab-walk for nav_link /
        # row_link / cell_link, region inference, alias matching) sees
        # the same hints the plan author specified. Pre-fix, only
        # ``label`` was threaded — the Tab-walk fallback for row_link
        # steps couldn't fire because ``kind`` got dropped here.
        orig_params = step.params or {}
        synthesised = MicroIntent(
            intent=step.intent, type="submit",
            budget=step.budget, section=step.section,
            required=step.required,
            params={
                "label": orig_params.get("label", ""),
                "kind": orig_params.get("kind", ""),
                "aliases": orig_params.get("aliases", []),
            },
        )
        return _stamp_backend(registry.get("submit").execute(synthesised, ctx), ctx)

    if step.gate and runner.extractor:
        print(f"  [gate] Verifying: {(step.verify or step.intent)[:80]}")
        time.sleep(2)
        screenshot = runner.env.screenshot()
        # #523 PR B-4 — capture the gate verify_gate call as a
        # ``verifier`` modelio record. No-op when augur is None.
        from ..observability.modelio import publish_modelio_context
        with publish_modelio_context(
            getattr(runner, "_augur", None),
            layer="verifier", step_index=index,
        ):
            passed, reason = runner.extractor.verify_gate(
                screenshot, step.verify or step.intent,
            )
        runner.costs["claude_extract"] += 1
        print(f"  [gate] Result: {'PASS' if passed else 'FAIL'} — {reason[:80]}")
        return StepResult(
            step_index=index, intent=step.intent, success=passed,
            data=f"gate:{'PASS' if passed else 'FAIL'}:{reason[:100]}",
        )

    if step.claude_only:
        ctx = build_step_context(runner, index)
        return _stamp_backend(registry.get("extract_url").execute(step, ctx), ctx)

    if step.type == "paginate":
        if not ensure_results_filters(runner, index):
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="filters_not_applied",
            )
        ctx = build_step_context(runner, index)
        return _stamp_backend(registry.get("paginate").execute(step, ctx), ctx)

    if step.type == "navigate_back" and runner._opened_detail_in_new_tab:
        return execute_close_detail_tab(runner, step, index)

    handler = registry.get(step.type)
    if handler is not None:
        ctx = build_step_context(runner, index)
        return _stamp_backend(handler.execute(step, ctx), ctx)

    return runner._execute_holo3_step(step, index)


def _stamp_backend(result: StepResult, ctx: Any) -> StepResult:
    """Tag ``result.executor_backend`` from the handler's scratch dict.

    Handlers that dispatch a click via :func:`gym.som_dispatch.try_som_click`
    set ``ctx.state["_executor_backend"]`` to ``"som"`` (or ``"vision"``
    when the SoM path was checked and missed). This stamp moves that
    onto the StepResult so the per-step backend is visible on the
    /v1/predict aggregate without every handler touching the
    :class:`StepResult` field by hand. Handlers that don't dispatch a
    routable click leave the scratch unset; the StepResult stays with
    the default ``executor_backend=""``.
    """
    backend = ctx.state.get("_executor_backend", "") if ctx is not None else ""
    if backend and not result.executor_backend:
        result.executor_backend = backend
    return result


# ── Per-handler shims ─────────────────────────────────────────────────
#
# Tests and external callers invoke ``runner._execute_navigate`` /
# ``_execute_claude_guided_*`` etc. directly — these route to the
# registered handler with a built context.

def _handler_shim(
    runner: "MicroPlanRunner", handler_type: str, fallback_module: str,
    step: MicroIntent, index: int,
) -> StepResult:
    handler = runner._handler_registry.get(handler_type)
    if handler is None:
        from importlib import import_module
        mod = import_module(f"mantis_agent.gym.step_handlers.{fallback_module}")
        cls_name = {
            "navigate": "NavigateHandler",
            "click": "ClaudeGuidedClickHandler",
            "filter": "ClaudeGuidedFilterHandler",
            "form": "ClaudeGuidedFormHandler",
            "paginate": "PaginateHandler",
            "claude_step": "ClaudeStepHandler",
            "holo3": "Holo3StepHandler",
        }[fallback_module]
        handler = getattr(mod, cls_name)(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_navigate(runner, step, index):
    return _handler_shim(runner, "navigate", "navigate", step, index)


def execute_claude_guided_click(runner, step, index):
    from .step_handlers.click import ClaudeGuidedClickHandler
    handler = ClaudeGuidedClickHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_claude_guided_filter(runner, step, index):
    from .step_handlers.filter import ClaudeGuidedFilterHandler
    handler = ClaudeGuidedFilterHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_claude_guided_form(runner, step, index):
    from .step_handlers.form import ClaudeGuidedFormHandler
    handler = ClaudeGuidedFormHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_paginate_layered(runner, step, index):
    from .step_handlers.paginate import PaginateHandler
    handler = PaginateHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_claude_guided_paginate(runner, step, index):
    from .step_handlers.paginate import PaginateHandler
    handler = PaginateHandler(runner)
    ctx = build_step_context(runner, index)
    return handler._claude_guided_paginate(step, ctx, index)


def execute_holo3_step(runner, step, index):
    from .step_handlers.holo3 import Holo3StepHandler
    handler = Holo3StepHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def execute_claude_step(runner, step, index):
    from .step_handlers.claude_step import ClaudeStepHandler
    handler = ClaudeStepHandler(runner)
    ctx = build_step_context(runner, index)
    return handler.execute(step, ctx)


def extract_listing_data_deep(runner, initial_screenshot):
    from .step_handlers.claude_step import ClaudeStepHandler
    handler = ClaudeStepHandler(runner)
    ctx = build_step_context(runner, 0)
    return handler._extract_listing_data_deep(initial_screenshot, ctx)


# ── Run finalization ─────────────────────────────────────────────────


def final_summary(runner: "MicroPlanRunner", results: list[StepResult]) -> None:
    # #351: sync gpu_seconds from the TimeMeter's ``think`` bucket
    # at terminal time so totals() returns real brain-inference
    # wall time (not the pre-#351 ``steps × per-step`` synthetic).
    # No-op when the runner doesn't have a TimeMeter — CostMeter's
    # sync helper guards.
    _time_meter = getattr(runner, "time_meter", None)
    if _time_meter is not None:
        runner.cost_meter.sync_gpu_seconds_from_time_meter(_time_meter)
    gpu_cost, claude_cost, proxy_cost, total_cost = runner.cost_meter.totals()
    elapsed = time.time() - runner._run_start
    print()
    for line in RunReporter.final_summary_lines(
        results=results,
        gpu_cost=gpu_cost, claude_cost=claude_cost,
        proxy_cost=proxy_cost, total_cost=total_cost,
        elapsed_seconds=elapsed,
        gpu_steps=int(runner.costs.get("gpu_steps", 0)),
        claude_extract_calls=int(runner.costs.get("claude_extract", 0)),
        claude_grounding_calls=int(runner.costs.get("claude_grounding", 0)),
        proxy_mb=float(runner.costs.get("proxy_mb", 0.0)),
    ):
        print(line)
    runner._final_costs = RunReporter.final_costs_dict(
        results=results,
        gpu_cost=gpu_cost, claude_cost=claude_cost,
        proxy_cost=proxy_cost, total_cost=total_cost,
        final_status=runner._final_status,
        checkpoint_path=runner.checkpoint_path,
    )
    # #349: emit the terminal cost + duration histograms so
    # dashboards (docs/operations/cost.md PromQL examples) have
    # real series to query. Mirrors the inflight gauge's label set
    # so panels grouped by tenant_id / model / status align. Skip
    # silently when no tenant_id is set so local / script runs
    # don't pollute the registry with default-label series; same
    # discipline as the inflight gauge.
    tenant_id = str(getattr(runner, "tenant_id", "") or "")
    if tenant_id:
        try:
            from .. import metrics as _metrics
            labels = {
                "tenant_id": tenant_id,
                "model": str(getattr(runner, "model_name", "") or ""),
                "status": str(getattr(runner, "_final_status", "") or "completed"),
            }
            _metrics.RUN_COST_USD.labels(**labels).observe(total_cost)
            _metrics.RUN_DURATION_SECONDS.labels(**labels).observe(elapsed)
        except Exception as exc:  # noqa: BLE001 — observability, never fatal
            logger.debug("terminal cost/duration histogram emit failed: %s", exc)


def _capture_browser_state_safe(runner: "MicroPlanRunner"):
    """Best-effort browser-state capture for PauseState (epic #358
    Phase A). Returns an empty :class:`BrowserState` when:

    - The env doesn't expose ``capture_browser_state`` (legacy
      adapters, host-integration mocks).
    - Capture raises (CDP unreachable, page closed, JS exception).

    Empty browser_state is the explicit "no capture available"
    signal: ``restore_browser_state`` checks ``bool(url)`` and
    no-ops when empty, so a pause-without-capture resumes from
    whatever URL the env happens to have at restart time —
    same behaviour as before the field existed.
    """
    from .checkpoint import BrowserState
    capture = getattr(runner.env, "capture_browser_state", None)
    if not callable(capture):
        return BrowserState()
    try:
        return capture()
    except Exception as exc:  # noqa: BLE001 — observability path
        logger.debug("capture_browser_state raised: %s", exc)
        return BrowserState()


def build_runner_result(
    runner: "MicroPlanRunner", plan: "MicroPlan", steps: list[StepResult],
) -> RunnerResult:
    status = runner._final_status or "completed"
    cancelled = status == "cancelled"
    paused = status == "paused" and runner.tool_channel.is_paused()
    pause_state: PauseState | None = None
    if paused:
        pp = runner.tool_channel.pending_pause or {}
        pause_state = PauseState(
            run_key=runner.run_key,
            plan_signature=runner.plan_signature
            or runner._compute_plan_signature(plan),
            session_name=runner.session_name,
            step_index=getattr(runner, "_last_run_step_index", 0),
            pending_tool=str(pp.get("tool", "")),
            pending_arguments=dict(pp.get("arguments", {})),
            pending_reason=str(pp.get("reason", "user_input")),
            prompt=str(pp.get("prompt", "")),
            step_results=[s.to_dict() for s in steps],
            loop_counters={
                str(k): v
                for k, v in getattr(runner, "_last_loop_counters", {}).items()
            },
            listings_on_page=getattr(runner, "_last_listings_on_page", 0),
            checkpoint_path=runner.checkpoint_path,
            timestamp=time.time(),
            # Epic #358 Phase A — snapshot URL + scroll + viewport so
            # resume() restores the exact pixel state.
            browser_state=_capture_browser_state_safe(runner),
        )
    return RunnerResult(
        steps=steps, status=status, cancelled=cancelled, paused=paused,
        pause_state=pause_state,
        halt_reason="" if status == "completed" else status,
    )
