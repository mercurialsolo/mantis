"""ClaudeGuidedClickHandler — listings click + grounding + click probes.

Phase 2 of EPIC #161, second handler extraction. Lifts
``MicroPlanRunner._execute_claude_guided_click`` (369 LOC) verbatim
into a standalone class. The runner method becomes a delegating shim;
the dispatch in ``_execute_step`` keeps its layout-hint branching
(``listings`` vs single-element form click) so the registry doesn't
gain a "click" entry until ``FormHandler`` is also extracted in a
follow-up PR.

Behavior is identical: same Home-then-Page_Down viewport reconstruction,
same ``find_all_listings`` scan + blocked-rescan retry, same
already-extracted-title filter, same per-card click → grounding refine
→ 2-attempt verify → middle-click fallback → 5-point probe-area
fallback. Cost accounting (``costs[gpu_steps|gpu_seconds|claude_extract|
claude_grounding|proxy_mb]``) is bumped at the same call sites.

What the handler reads from :class:`StepContext`:

- ``env``                — screenshot / step / screen_size
- ``extractor``          — find_all_listings (Claude vision)
- ``grounding``          — coordinate refinement
- ``scanner``            — viewport stage / page-listing cache (read via
                           runner property delegates from #161 Phase 1.2)
- ``dynamic_verifier``   — record_viewport_scan / record_item_attempt /
                           record_item_opened / record_item_completed /
                           record_page_exhausted
- ``site_config``        — is_detail_page

What the handler reads from the runner (parent back-reference):

- ``costs``                   — cost meter dict (aliased)
- ``_current_page``           — pagination state
- ``_last_known_url``, ``_last_extracted``, ``_last_click_title``
- ``_listings_on_page``       — runner-owned property (counter)
- ``_opened_detail_in_new_tab``
- ``_blocked_retry_done``     — handler-private retry flag, set on the
                                runner so checkpoint persistence sees it
- ``_set_scroll_state``       — delegates to BrowserState (#115)
- ``_current_results_page_url`` — delegates to BrowserState
- ``_read_current_url``       — runner method (CDP → OCR fallback)

Phase 4 of EPIC #161 will lift the listings-specific bookkeeping
(``_listings_on_page``, ``_blocked_retry_done``, ``_extracted_titles``
mutations) onto :class:`~..listings_scanner.ListingsScanner` methods,
shrinking the parent back-reference surface.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..log_utils import url_for_log
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


class ClaudeGuidedClickHandler:
    """Implements :class:`~..step_context.StepHandler` for listings ``click``.

    Not registered in the default registry yet — the dispatch in
    ``MicroPlanRunner._execute_step`` keeps the layout-hint branching
    that decides between listings click (this handler) and single-element
    form click (handled by ``_execute_claude_guided_form``, a separate
    follow-up). Once FormHandler lands, both bind to the "click" type
    and the dispatch logic moves into a small router.
    """

    step_type = "click"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        grounding = ctx.grounding
        dynamic_verifier = ctx.dynamic_verifier
        site_config = ctx.site_config
        index = int(ctx.state.get("index", 0))

        # Pre-settle — page may still be loading after navigate/paginate.
        # Lifted from MicroPlanRunner._execute_step's "click" branch in EPIC #161
        # cleanup so registry-first dispatch produces identical timing.
        time.sleep(2)

        # If no cached listings, scan the page (one Claude call for ALL cards)
        if runner._page_listing_index >= len(runner._page_listings):
            # Staged per-viewport scan: scan ONE viewport, cache its cards.
            # When cache empties, advance to next viewport (Page_Down).
            # Only page_exhausted after all viewport stages return empty.
            while runner._viewport_stage < runner._max_viewport_stages:
                # Reconstruct the current viewport deterministically:
                # always start from Home, then Page_Down N times.
                try:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                    time.sleep(0.5)
                    for _ in range(runner._viewport_stage):
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                        time.sleep(0.5)
                    runner._set_scroll_state(
                        context="results_scan",
                        url=runner._current_results_page_url() or runner._results_base_url,
                        page_downs=runner._viewport_stage,
                        wheel_downs=0,
                        viewport_stage=runner._viewport_stage,
                    )
                except Exception:
                    pass

                screenshot = env.screenshot()
                scan_result = extractor.find_all_listings(screenshot)
                runner.costs["claude_extract"] += 1

                scan_status = "ok"
                if isinstance(scan_result, tuple):
                    status = scan_result[0]
                    if status == "blocked":
                        # Could be a real error/anti-bot page OR a transient
                        # proxy/CDN loading splash. Wait + re-scan the SAME
                        # viewport once before giving up — proxy-loading
                        # screens typically resolve in 5-15s. Caught when
                        # Chromium's first-paint splash misled find_all_listings
                        # into reporting blocked on a CRM that loaded fine
                        # 30s later.
                        already_retried = getattr(runner, "_blocked_retry_done", False)
                        if not already_retried:
                            logger.warning(
                                f"  [claude-click] Viewport {runner._viewport_stage}: "
                                f"blocked/error page — waiting 12s and rescanning"
                            )
                            runner._blocked_retry_done = True
                            time.sleep(12)
                            screenshot = env.screenshot()
                            scan_result = extractor.find_all_listings(screenshot)
                            runner.costs["claude_extract"] += 1
                            if isinstance(scan_result, tuple) and scan_result[0] == "blocked":
                                logger.warning(
                                    f"  [claude-click] Viewport {runner._viewport_stage}: "
                                    f"still blocked after rescan — halting"
                                )
                            else:
                                # Recovered — fall through to normal processing.
                                runner._blocked_retry_done = False
                                if isinstance(scan_result, tuple):
                                    status = scan_result[0]
                                else:
                                    status = "ok"
                        if status == "blocked":
                            runner._blocked_retry_done = False
                            logger.warning(f"  [claude-click] Viewport {runner._viewport_stage}: blocked/error page")
                            dynamic_verifier.record_viewport_scan(
                                page=runner._current_page,
                                viewport_stage=runner._viewport_stage,
                                cards=[],
                                new_cards=[],
                                status="blocked",
                                url=runner._current_results_page_url() or runner._last_known_url,
                            )
                            return StepResult(
                                step_index=index,
                                intent=step.intent,
                                success=False,
                                data="page_blocked",
                            )
                    if status == "error":
                        logger.warning(f"  [claude-click] Viewport {runner._viewport_stage}: parse/API failure")
                        dynamic_verifier.record_viewport_scan(
                            page=runner._current_page,
                            viewport_stage=runner._viewport_stage,
                            cards=[],
                            new_cards=[],
                            status="error",
                            url=runner._current_results_page_url() or runner._last_known_url,
                        )
                        return StepResult(
                            step_index=index,
                            intent=step.intent,
                            success=False,
                            data="scan_error",
                        )
                    scan_status = status
                    cards = []
                else:
                    cards = scan_result

                # Filter out already-extracted titles
                skip = set(t.lower() for t in runner._extracted_titles)
                filtered = [(x, y, t) for x, y, t in cards
                           if t.lower() not in skip and t != "unknown"]
                unknown_cards = [(x, y, t) for x, y, t in cards if t == "unknown"]
                filtered.extend(unknown_cards)
                filtered.sort(key=lambda c: c[1])
                dynamic_verifier.record_viewport_scan(
                    page=runner._current_page,
                    viewport_stage=runner._viewport_stage,
                    cards=cards,
                    new_cards=filtered,
                    status=scan_status,
                    url=runner._current_results_page_url() or runner._last_known_url,
                )

                logger.info(f"  [claude-click] Viewport {runner._viewport_stage}: {len(cards)} cards, {len(filtered)} new")

                if filtered:
                    runner._page_listings = filtered
                    runner._page_listing_index = 0
                    break  # Found cards in this viewport — click them
                else:
                    runner._viewport_stage += 1  # Try next viewport

            if not runner._page_listings or runner._page_listing_index >= len(runner._page_listings):
                logger.info(f"  [claude-click] All {runner._max_viewport_stages} viewports exhausted")
                dynamic_verifier.record_page_exhausted(
                    page=runner._current_page,
                    reason=f"all_{runner._max_viewport_stages}_viewports_exhausted",
                )
                return StepResult(step_index=index, intent=step.intent, success=False,
                                data="page_exhausted")

        # Pop next card from cache — scroll to the viewport where it was found
        x, y, title = runner._page_listings[runner._page_listing_index]
        runner._page_listing_index += 1
        title_for_verification = (
            title
            if title.strip().lower() != "unknown"
            else f"unknown@v{runner._viewport_stage}:{x},{y}"
        )
        runner._last_click_title = title_for_verification
        dynamic_verifier.record_item_attempt(
            page=runner._current_page,
            item=title_for_verification,
            viewport_stage=runner._viewport_stage,
        )

        # Scroll to the correct viewport (Home + N Page_Downs)
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(runner._viewport_stage):
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
            runner._set_scroll_state(
                context="results_click",
                url=runner._current_results_page_url() or runner._results_base_url,
                page_downs=runner._viewport_stage,
                wheel_downs=0,
                viewport_stage=runner._viewport_stage,
            )
        except Exception:
            pass

        logger.info(
            f"  [claude-click] Card {runner._page_listing_index}/{len(runner._page_listings)}: "
            f"'{title_for_verification[:40]}' at ({x}, {y}) viewport={runner._viewport_stage}"
        )

        # Delay before the final screenshot so grounding sees the frame we will actually click.
        time.sleep(random.uniform(1.5, 3.5))

        # Grounding refines — but only accept if the delta is small.
        # #181: high-risk listing-card title clicks bypass the grounding
        # cache so a stale cached coordinate doesn't pin a regression on
        # the photo region. Per-step ``hints["independent_grounding"]``
        # overrides on a single step; site_config.require_independent_grounding
        # opts whole layouts in.
        force_compute = _should_force_independent_grounding(step, site_config)
        if grounding and title.strip().lower() != "unknown":
            screenshot = env.screenshot()
            grounding_result = grounding.ground(
                screenshot, title, x, y, force_compute=force_compute,
            )
            runner.costs["claude_grounding"] += 1
            dx = abs(grounding_result.x - x)
            dy = abs(grounding_result.y - y)
            outcome = "rejected"
            if grounding_result.confidence > 0.5 and dx < 200 and dy < 200:
                x, y = grounding_result.x, grounding_result.y
                outcome = "accepted"
                logger.info(
                    f"  [grounding] refined to ({x}, {y}) delta=({dx},{dy}) "
                    f"force={force_compute}"
                )
            else:
                logger.info(
                    f"  [grounding] rejected: delta=({dx},{dy}) "
                    f"conf={grounding_result.confidence} force={force_compute}"
                )
            _emit_grounding_metrics(
                runner, dx=dx, dy=dy, outcome=outcome, force=force_compute,
            )
        elif title.strip().lower() == "unknown":
            logger.info("  [grounding] skipped for unknown-title card; using scan coordinates")

        # Click
        try:
            env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y}))
            runner.costs["gpu_steps"] += 1
            runner.costs["gpu_seconds"] += 3
            runner.costs["proxy_mb"] += 5.0
        except Exception as e:
            logger.warning(f"  [claude-click] Click failed: {e}")
            dynamic_verifier.record_item_completed(
                page=runner._current_page,
                item=getattr(runner, "_last_click_title", "") or title,
                success=False,
                reason=f"click_failed:{e}",
            )
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Verify: are we on a detail page? Retry once (page may still load)
        # Prefer CDP over screenshot URL extraction — issue #89 §1.
        for verify_attempt in range(2):
            time.sleep(3 + verify_attempt * 3)  # 3s first, 6s retry
            url = runner._read_current_url()
            if not url:
                # CDP unavailable — fall back to screenshot OCR.
                after = env.screenshot()
                url = runner._read_current_url(after)

            if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
                logger.info(f"  [claude-click] Verified on detail page: {url[:60]}")
                runner._last_known_url = url
                dynamic_verifier.record_item_opened(
                    page=runner._current_page,
                    item=getattr(runner, "_last_click_title", "") or title,
                    url=url,
                )
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_clicked_title": getattr(runner, "_last_click_title", ""),
                    "last_attempted_url": url,
                    "last_attempted_at": time.time(),
                    "last_attempted_step": index,
                }
                runner._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                runner._listings_on_page += 1
                # Store the exact title Claude found for skip list
                if hasattr(runner, '_last_click_title') and runner._last_click_title:
                    runner._extracted_titles.append(runner._last_click_title)
                return StepResult(step_index=index, intent=step.intent, success=True,
                                steps_used=1, duration=3.0 + verify_attempt * 3)

            # Auth-wall short-circuit: if the click hijacked the original tab to
            # a /login page, neither middle-click nor probe-area clicks can
            # rescue this card — they'll just re-fire on a login form. Bail
            # immediately and let StepRecoveryPolicy roll back to the results
            # URL so the next loop iteration starts clean.
            if url and _looks_like_login_redirect(url):
                logger.warning(
                    "  [claude-click] Click redirected to login (url=%s) — "
                    "bailing out for recovery to roll back",
                    url[:80],
                )
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=getattr(runner, "_last_click_title", "") or title,
                    url=url,
                    success=False,
                    reason="login_redirect",
                )
                if hasattr(runner, '_last_click_title') and runner._last_click_title:
                    runner._extracted_titles.append(runner._last_click_title)
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data="login_redirect",
                )

            if verify_attempt == 0:
                logger.info(
                    "  [claude-click] Not on detail page yet (url=%s) — retrying verify",
                    url_for_log(url),
                )

        logger.info("  [claude-click] Plain click did not navigate — trying middle-click fallback")
        try:
            env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y, "button": "middle"}))
            runner.costs["gpu_steps"] += 1
            runner.costs["gpu_seconds"] += 3
            runner.costs["proxy_mb"] += 5.0
            time.sleep(2)

            for switch_attempt in range(2):
                url = runner._read_current_url()
                if not url:
                    after = env.screenshot()
                    url = runner._read_current_url(after)
                if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
                    logger.info(f"  [claude-click] Middle-click fallback opened detail: {url[:60]}")
                    runner._opened_detail_in_new_tab = True
                    runner._last_known_url = url
                    dynamic_verifier.record_item_opened(
                        page=runner._current_page,
                        item=getattr(runner, "_last_click_title", "") or title,
                        url=url,
                    )
                    runner._last_extracted = {
                        **runner._last_extracted,
                        "last_clicked_title": getattr(runner, "_last_click_title", ""),
                        "last_attempted_url": url,
                        "last_attempted_at": time.time(),
                        "last_attempted_step": index,
                    }
                    runner._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                    runner._listings_on_page += 1
                    if hasattr(runner, '_last_click_title') and runner._last_click_title:
                        runner._extracted_titles.append(runner._last_click_title)
                    return StepResult(step_index=index, intent=step.intent, success=True,
                                    steps_used=2 + switch_attempt, duration=9.0)

                # Empty-newtab short-circuit: middle-click on a non-anchor card
                # region (e.g. whitespace) opens a blank chrome://newtab/.
                # Repeated ctrl+Tab + probe-area clicks won't recover — the
                # click target itself isn't a link. Close the empty tab,
                # mark the card tried, and let recovery move on.
                if url and _looks_like_blank_newtab(url):
                    logger.warning(
                        "  [claude-click] Middle-click landed on blank tab "
                        "(%s) — aborting fallback chain",
                        url_for_log(url),
                    )
                    try:
                        env.step(Action(
                            action_type=ActionType.KEY_PRESS,
                            params={"keys": "ctrl+w"},
                        ))
                        time.sleep(0.5)
                    except Exception as close_err:
                        logger.debug(
                            "  [claude-click] ctrl+w on blank tab failed: %s",
                            close_err,
                        )
                    dynamic_verifier.record_item_completed(
                        page=runner._current_page,
                        item=getattr(runner, "_last_click_title", "") or title,
                        url=url,
                        success=False,
                        reason="newtab_blank",
                    )
                    if hasattr(runner, '_last_click_title') and runner._last_click_title:
                        runner._extracted_titles.append(runner._last_click_title)
                    return StepResult(
                        step_index=index, intent=step.intent, success=False,
                        data="newtab_blank",
                    )

                if switch_attempt == 0:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+Tab"}))
                    time.sleep(2)
        except Exception as e:
            logger.warning(f"  [claude-click] Middle-click fallback failed: {e}")

        logger.info("  [claude-click] Middle-click did not verify — trying card-area click probes")
        probe_points = [
            ("image_center", x, y - 145),
            ("image_lower", x, y - 90),
            ("title_lower", x, y + 28),
            ("title_left", x - 90, y),
            ("title_right", x + 90, y),
        ]
        tried_points: set[tuple[int, int]] = set()
        for label, probe_x, probe_y in probe_points:
            probe_x = max(1, min(int(probe_x), env.screen_size[0] - 2))
            probe_y = max(1, min(int(probe_y), env.screen_size[1] - 2))
            if (probe_x, probe_y) in tried_points:
                continue
            tried_points.add((probe_x, probe_y))

            try:
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(0.3)
                for _ in range(runner._viewport_stage):
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                    time.sleep(0.3)
                logger.info(
                    "  [claude-click] Probe %s at (%s, %s)",
                    label,
                    probe_x,
                    probe_y,
                )
                env.step(Action(action_type=ActionType.CLICK, params={"x": probe_x, "y": probe_y}))
                runner.costs["gpu_steps"] += 1
                runner.costs["gpu_seconds"] += 3
                runner.costs["proxy_mb"] += 5.0
                time.sleep(3)

                url = runner._read_current_url()
                if not url:
                    after = env.screenshot()
                    url = runner._read_current_url(after)
                if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
                    logger.info(
                        "  [claude-click] Probe %s opened detail: %s",
                        label,
                        url[:60],
                    )
                    runner._last_known_url = url
                    dynamic_verifier.record_item_opened(
                        page=runner._current_page,
                        item=getattr(runner, "_last_click_title", "") or title,
                        url=url,
                    )
                    runner._last_extracted = {
                        **runner._last_extracted,
                        "last_clicked_title": getattr(runner, "_last_click_title", ""),
                        "last_attempted_url": url,
                        "last_attempted_at": time.time(),
                        "last_attempted_step": index,
                    }
                    runner._set_scroll_state(context="detail_top", url=url, page_downs=0, wheel_downs=0)
                    runner._listings_on_page += 1
                    if hasattr(runner, '_last_click_title') and runner._last_click_title:
                        runner._extracted_titles.append(runner._last_click_title)
                    return StepResult(step_index=index, intent=step.intent, success=True,
                                    steps_used=3, duration=12.0)
            except Exception as e:
                logger.warning(f"  [claude-click] Probe {label} failed: {e}")

        logger.warning(
            "  [claude-click] Failed verification after retries (url=%s)",
            url_for_log(url),
        )
        dynamic_verifier.record_item_completed(
            page=runner._current_page,
            item=getattr(runner, "_last_click_title", "") or title,
            url=url,
            success=False,
            reason="detail_page_not_verified",
        )
        # Mark title as tried so we don't re-attempt the same card
        if hasattr(runner, '_last_click_title') and runner._last_click_title:
            runner._extracted_titles.append(runner._last_click_title)
        return StepResult(step_index=index, intent=step.intent, success=False)


# ── click-failure URL predicates ──────────────────────────────────────


# Path tokens used to detect an auth wall hijacking a card click. Kept
# conservative — match common SaaS / CRM login routes, not search-result
# pages that happen to contain "login" as a query parameter.
_LOGIN_PATH_TOKENS: tuple[str, ...] = (
    "/login",
    "/signin",
    "/sign-in",
    "/sign_in",
    "/auth/login",
    "/oauth/authorize",
    "/users/sign_in",
    "/account/login",
)


def _looks_like_login_redirect(url: str) -> bool:
    """True when the post-click URL is a login wall, not a detail page.

    Used by the click handler to short-circuit the middle-click + probe
    fallback chain — neither will rescue a click that was hijacked into
    an auth flow. StepRecoveryPolicy receives ``data="login_redirect"``
    and rolls back to the prior results URL.
    """
    if not url:
        return False
    lowered = url.lower()
    return any(token in lowered for token in _LOGIN_PATH_TOKENS)


def _looks_like_blank_newtab(url: str) -> bool:
    """True when the middle-click landed on Chromium's empty new-tab page.

    Used by the click handler to abort the middle-click + probe chain
    when the click target wasn't a navigable link. Matches
    ``chrome://newtab/``, ``about:blank``, and the Edge/Chromium variants
    that surface in the same scenario.
    """
    if not url:
        return False
    lowered = url.lower().strip()
    return (
        lowered.startswith("chrome://newtab")
        or lowered.startswith("chrome://new-tab-page")
        or lowered.startswith("edge://newtab")
        or lowered.startswith("about:blank")
    )


# ── #181 helpers — independent grounding routing + metric emission ────


def _should_force_independent_grounding(step: Any, site_config: Any) -> bool:
    """Decide whether a click step bypasses the grounding cache.

    Two opt-in surfaces — either is sufficient:

    1. Per-step hint: ``step.hints["independent_grounding"] = True``
       (or any truthy value). Lets a plan author pin a single step.
    2. Site config: any of ``step.hints["layout"]`` / ``step.section``
       / the literal ``step.type`` matches an entry in
       ``site_config.require_independent_grounding``. Lets a recipe
       opt whole layouts in.

    Defaults to False so routine clicks still benefit from the
    GroundingCache cost-savings shipped in #117.
    """
    hints = getattr(step, "hints", None) or {}
    if hints.get("independent_grounding"):
        return True
    require = tuple(getattr(site_config, "require_independent_grounding", ()) or ())
    if not require:
        return False
    layout = str(hints.get("layout", "") or "")
    section = str(getattr(step, "section", "") or "")
    type_ = str(getattr(step, "type", "") or "")
    return any(tag in require for tag in (layout, section, type_) if tag)


def _emit_grounding_metrics(
    runner: Any, *, dx: int, dy: int, outcome: str, force: bool,
) -> None:
    """Emit ``mantis_grounding_correction_distance_pixels`` +
    ``mantis_grounding_call_total`` for one click-handler grounding
    pass. Wrapped in try/except — telemetry must never break a run."""
    try:
        from ...metrics import GROUNDING_CALL_TOTAL, GROUNDING_CORRECTION_DISTANCE
        tenant_id = getattr(runner, "tenant_id", "") or ""
        magnitude = (dx * dx + dy * dy) ** 0.5
        GROUNDING_CORRECTION_DISTANCE.labels(
            tenant_id=tenant_id, force_compute=str(bool(force)).lower(),
        ).observe(magnitude)
        GROUNDING_CALL_TOTAL.labels(
            tenant_id=tenant_id,
            outcome=outcome,
            force_compute=str(bool(force)).lower(),
        ).inc()
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks runs
        logger.debug("grounding metric emit failed: %s", exc)
