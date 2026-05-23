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
from .. import adaptive_settle
from ..checkpoint import StepResult
from ..log_utils import url_for_log
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)

# Issue #600: the deterministic Home + N Page_Down scroll-restore would
# place viewport stage 0 at the very top of the page, where a typical
# results page shows a sticky header / search bar / promo band and the
# first row of cards is only peeking in at the bottom edge (just the
# photos, no titles or prices visible yet). ``find_all_listings`` then
# returned ``title="unknown"`` for every card and the runner fell back
# to coord placeholders — which (a) made PR #597's already-clicked
# prefilter inert and (b) made the brain re-click the same screen
# coordinates next iteration, hitting the URL-dedup gate.
#
# Adding a single Page_Down before stage 0 pushes the header off the
# top so the first card row is fully visible (title + price + dealer
# tag) when the scan + click both fire. The constant applies to both
# the scan-time scroll and the click-time scroll so the brain clicks
# at the same Y the scan reported.
HEADER_PRESCROLL_PAGE_DOWNS = 1


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
        # #294 adaptive-settle: cap at 2s, exit early when frame hash settles.
        adaptive_settle.settle_after_action(env, max_seconds=2.0)

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
                    for _ in range(HEADER_PRESCROLL_PAGE_DOWNS + runner._viewport_stage):
                        env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                        time.sleep(0.5)
                    runner._set_scroll_state(
                        context="results_scan",
                        url=runner._current_results_page_url() or runner._results_base_url,
                        page_downs=HEADER_PRESCROLL_PAGE_DOWNS + runner._viewport_stage,
                        wheel_downs=0,
                        viewport_stage=runner._viewport_stage,
                    )
                except Exception:
                    pass

                screenshot = env.screenshot()
                # Issue #597: pass the runner's already-clicked title tail
                # so the scan prompt instructs the brain NOT to return
                # entries we've already processed. Shifts dedup from a
                # post-hoc title-substring filter (below) to the model's
                # classification step — saves Claude tokens and produces
                # cleaner output (fewer cards to filter). Tail-12 keeps
                # the prompt token budget bounded on long-running loops.
                already_tail = list(getattr(runner, "_extracted_titles", []) or [])[-12:]
                scan_result = extractor.find_all_listings(
                    screenshot, already_clicked=already_tail,
                )
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
                            scan_result = extractor.find_all_listings(
                                screenshot, already_clicked=already_tail,
                            )
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

        # Scroll to the correct viewport (Home + header-prescroll +
        # N Page_Downs). The header prescroll must match the scan-time
        # one (Issue #600) so the brain clicks the same Y the scan saw.
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
            for _ in range(HEADER_PRESCROLL_PAGE_DOWNS + runner._viewport_stage):
                env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                time.sleep(0.5)
            runner._set_scroll_state(
                context="results_click",
                url=runner._current_results_page_url() or runner._results_base_url,
                page_downs=HEADER_PRESCROLL_PAGE_DOWNS + runner._viewport_stage,
                wheel_downs=0,
                viewport_stage=runner._viewport_stage,
            )
        except Exception:
            pass

        # WARNING level (not INFO) so per-run navigation counts surface
        # in production logs — Modal app logs suppresses INFO. Without
        # this every per-run "how many leads were clicked through?"
        # question requires either tee-ing logs to durable storage or
        # bisecting via deploy+rerun. See
        # ``feedback_warning_level_for_modal_observability.md``.
        logger.warning(
            f"  [claude-click] NAV Card {runner._page_listing_index}/{len(runner._page_listings)}: "
            f"'{title_for_verification[:60]}' at ({x}, {y}) viewport={runner._viewport_stage}"
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

        # Capture the BEFORE-click frame for SPA-aware verification —
        # if the URL check after click fails, the framework falls back
        # to comparing this frame with the AFTER frame to decide whether
        # the click landed on a same-URL modal/overlay (the lu.ma /
        # generic-SPA pattern). Defensive: failure to capture downgrades
        # the SPA fallback to a no-op rather than the click itself.
        pre_click_screenshot: Any = None
        try:
            pre_click_screenshot = env.screenshot()
        except Exception:
            pre_click_screenshot = None

        # Capture pre-click URL so the post-click trust-gate retry has
        # a baseline to compare against. ``_best_effort_current_url``
        # returns "" on any error so the retry's gate fails closed.
        url_before_click: str = ""
        try:
            url_before_click = str(getattr(runner, "_best_effort_current_url", lambda: "")() or "")
        except Exception:
            url_before_click = ""

        # Click — #300: try SoM-anchored CDP dispatch first when the
        # routing policy promotes it AND the env exposes the CDP click
        # shim. Falls through to legacy xdotool on policy-off /
        # capability-miss / no-element-at-point. The result is recorded
        # on the StepResult's ``executor_backend`` tag (visible on the
        # /v1/predict response aggregate).
        # #582: snapshot Chrome tab count BEFORE the click so we can
        # detect new-tab opens from any click path (plain, SoM, Holo3-
        # direct, modifier, JS-driven window.open). Today only the
        # middle-click fallback sets ``_opened_detail_in_new_tab``;
        # plain-click → window.open() spawns a tab the runner never
        # knows about → subsequent navigate_back's Alt+Left fails →
        # ``navigate_back_recovered`` halt cycle.
        tabs_before_click = _safe_tab_count(env)

        # Issue #598: when the plan signals "open in new tab" intent
        # (plan_decomposer sets ``hints.open_in_new_tab=True`` on
        # extraction click steps whose source plan prose says "open
        # link in new tab" / "Ctrl+click" / "right-click"), dispatch
        # middle-click as PRIMARY. On success, fall straight through
        # the existing new-tab settle path (which sets
        # _opened_detail_in_new_tab=True so navigate_back routes via
        # Ctrl+W) and return early. On failure, fall through to the
        # legacy plain-click chain (which still has its own middle-
        # click fallback) so a missed click doesn't drop the listing.
        primary_result = _try_open_in_new_tab_primary(
            handler=self, runner=runner, env=env, ctx=ctx,
            step=step, index=index, x=x, y=y, title=title,
            site_config=site_config, dynamic_verifier=dynamic_verifier,
        )
        if primary_result is not None:
            return primary_result

        primary_click_backend = "vision"
        from ..som_dispatch import try_som_click
        try:
            if try_som_click(env, x, y, ctx.routing_policy):
                primary_click_backend = "som"
                logger.info(
                    f"  [claude-click] SoM CDP click at ({x},{y}) — "
                    f"skipped xdotool"
                )
            else:
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
        # Stash so the rest of execute() can tag the eventual StepResult.
        ctx.state["_executor_backend"] = primary_click_backend

        # PR-G: SoM ``el.click()`` returns ok=True but the page didn't
        # navigate. The synthetic click event has ``isTrusted=false`` and
        # some SPA frameworks gate on trusted gestures and silently
        # reject the synthetic chain — surfaced on staff-crm-long step
        # 8 (row Robot-Name link, run ``74add5d8``). Retry once with CDP
        # ``Input.dispatchMouseEvent`` (isTrusted=true, indistinguishable
        # from a real click). Same primitive PR #447 Fix B added to the
        # submit handler; extracted to ``pointer_retry`` so both handlers
        # share the gate logic and stabilization-window semantics.
        from ..pointer_retry import pointer_retry_if_unchanged
        pointer_retry_if_unchanged(
            env, runner, x, y,
            url_before=url_before_click,
            executor_backend=primary_click_backend,
            log_prefix="[claude-click]",
        )

        # Verify: are we on a detail page? Retry once (page may still load)
        # Prefer CDP over screenshot URL extraction — issue #89 §1.
        for verify_attempt in range(2):
            # #294: cap at the legacy 3s/6s budget per attempt; exit early
            # when the navigation lands and the frame stabilises.
            adaptive_settle.settle_after_action(
                env, max_seconds=3.0 + verify_attempt * 3.0,
            )
            url = runner._read_current_url()
            if not url:
                # CDP unavailable — fall back to screenshot OCR.
                after = env.screenshot()
                url = runner._read_current_url(after)

            if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
                # #582: tab-count diff. If the click opened a new tab
                # (plain click → site's onclick → window.open(), or a
                # modifier click that took a path other than the middle-
                # click fallback), set the flag so subsequent
                # navigate_back routes through execute_close_detail_tab
                # (Ctrl+W) instead of Alt+Left.
                tabs_after = _safe_tab_count(env)
                if tabs_after > tabs_before_click > 0:
                    runner._opened_detail_in_new_tab = True
                    logger.warning(
                        "  [claude-click] New tab detected via CDP "
                        "diff (%d → %d) — flagged for Ctrl+W on "
                        "next navigate_back",
                        tabs_before_click, tabs_after,
                    )
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

        # SPA-aware fallback (lu.ma / single-page apps that open detail
        # content in a modal without changing the URL). The URL check
        # above only succeeds when the click triggers a full-page
        # navigation; for SPA modals the URL stays put and the URL
        # check returns False even though the click succeeded. Compare
        # the BEFORE / AFTER screenshots via the extractor's tool_use
        # verifier and accept the click when navigation is confirmed
        # (URL change OR same-URL modal opened). Skipped when the
        # extractor or the pre-click frame is unavailable; falls
        # through to middle-click in that case.
        # Track the most recent verify_post_click_navigation outcome
        # so the terminal failure return can stamp ``failure_class``.
        # Epic #377 follow-up: ``wrong_target`` was leaking out as
        # ``unknown`` because the terminal failure carried no ``data``.
        last_verify_outcome: dict | None = None
        if extractor is not None and pre_click_screenshot is not None:
            try:
                post_click_screenshot = env.screenshot()
            except Exception:
                post_click_screenshot = None
            if post_click_screenshot is not None:
                nav = extractor.verify_post_click_navigation(
                    pre_click_screenshot,
                    post_click_screenshot,
                    step.intent,
                )
                runner.costs["claude_extract"] += 1
                if nav is not None:
                    last_verify_outcome = dict(nav)
                if nav and nav.get("navigated") is True:
                    kind = str(nav.get("kind", "modal"))
                    reason = str(nav.get("reason", ""))[:120]
                    logger.info(
                        "  [claude-click] SPA-aware verify accepted click "
                        "(kind=%s): %s",
                        kind, reason,
                    )
                    accepted_url = url or runner._results_base_url
                    runner._last_known_url = accepted_url
                    dynamic_verifier.record_item_opened(
                        page=runner._current_page,
                        item=getattr(runner, "_last_click_title", "") or title,
                        url=accepted_url,
                    )
                    runner._last_extracted = {
                        **runner._last_extracted,
                        "last_clicked_title": getattr(runner, "_last_click_title", ""),
                        "last_attempted_url": accepted_url,
                        "last_attempted_at": time.time(),
                        "last_attempted_step": index,
                        "last_click_verify_kind": kind,
                    }
                    runner._set_scroll_state(
                        context="detail_top",
                        url=accepted_url,
                        page_downs=0,
                        wheel_downs=0,
                    )
                    runner._listings_on_page += 1
                    if hasattr(runner, '_last_click_title') and runner._last_click_title:
                        runner._extracted_titles.append(runner._last_click_title)
                    return StepResult(
                        step_index=index, intent=step.intent, success=True,
                        steps_used=1, duration=9.0,
                    )

        logger.info("  [claude-click] Plain click did not navigate — trying middle-click fallback")
        try:
            env.step(Action(action_type=ActionType.CLICK, params={"x": x, "y": y, "button": "middle"}))
            runner.costs["gpu_steps"] += 1
            runner.costs["gpu_seconds"] += 3
            runner.costs["proxy_mb"] += 5.0
            # #294: cap at 2s for middle-click newtab settle.
            adaptive_settle.settle_after_action(env, max_seconds=2.0)

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
                    # #294: cap at 2s for tab-switch settle.
                    adaptive_settle.settle_after_action(env, max_seconds=2.0)
        except Exception as e:
            logger.warning(f"  [claude-click] Middle-click fallback failed: {e}")

        # Defense-in-depth (#209 Symptom 2 / Finding #3): before escalating
        # to probe-area clicks, switch focus back to the source (first) tab
        # and re-verify. A slow JS handler can navigate the source tab AFTER
        # our 2-attempt verify gate ran but BEFORE the middle-click loop
        # started, leaving us with the right state on the wrong-focused tab.
        # ctrl+1 is the Chromium shortcut for "first tab" — a no-op when
        # focus is already there, so it's safe even when middle-click
        # didn't actually open a new tab.
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+1"}))
            # #294: cap at 1s for tab-focus settle.
            adaptive_settle.settle_after_action(env, max_seconds=1.0)
            url = runner._read_current_url()
            if not url:
                after = env.screenshot()
                url = runner._read_current_url(after)
            if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
                logger.info(
                    "  [claude-click] Source tab navigated after middle-click "
                    "(url=%s) — accepting late navigation",
                    url_for_log(url),
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
                runner._set_scroll_state(
                    context="detail_top", url=url, page_downs=0, wheel_downs=0,
                )
                runner._listings_on_page += 1
                if hasattr(runner, '_last_click_title') and runner._last_click_title:
                    runner._extracted_titles.append(runner._last_click_title)
                return StepResult(
                    step_index=index, intent=step.intent, success=True,
                    steps_used=3, duration=11.0,
                )
        except Exception as recheck_err:
            logger.debug(
                "  [claude-click] Source-tab recheck failed (non-fatal): %s",
                recheck_err,
            )

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
                # Match the scan-time header prescroll (Issue #600).
                for _ in range(HEADER_PRESCROLL_PAGE_DOWNS + runner._viewport_stage):
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
                # #294: cap at 3s for probe-area click navigation settle.
                adaptive_settle.settle_after_action(env, max_seconds=3.0)

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
        # Epic #377 follow-up: surface the SPA-aware verifier's
        # ``kind`` on the terminal failure so the critic / dashboard
        # can route on a stable signal. ``wrong_target`` (Claude saw
        # the click hit a category card / login wall / ad — the wrong
        # destination) maps to ``failure_class=wrong_target``.
        # ``no_change`` maps to ``no_state_change`` (matches the
        # demotion class from Phase A.1). Anything else falls through
        # to ``unknown``.
        verify_kind = (
            str(last_verify_outcome.get("kind", ""))
            if last_verify_outcome else ""
        )
        verify_reason = (
            str(last_verify_outcome.get("reason", ""))[:120]
            if last_verify_outcome else ""
        )
        if verify_kind == "wrong_target":
            failure_class = "wrong_target"
            data = f"click_no_nav:wrong_target:{verify_reason}"
        elif verify_kind == "no_change":
            failure_class = "no_state_change"
            # ``no_state_change`` substring also lets the classifier
            # fallback (failure_class.py) map a legacy result.json
            # to the same class even without the handler stamp.
            data = f"click_no_nav:no_state_change:{verify_reason}"
        else:
            failure_class = ""
            data = "click_no_nav:detail_page_not_verified"
        return StepResult(
            step_index=index, intent=step.intent, success=False,
            data=data, failure_class=failure_class,
        )


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


def _try_open_in_new_tab_primary(
    *,
    handler: Any,
    runner: Any,
    env: Any,
    ctx: Any,
    step: Any,
    index: int,
    x: int,
    y: int,
    title: str,
    site_config: Any,
    dynamic_verifier: Any,
):
    """Issue #598: when ``step.hints.open_in_new_tab is True``, try
    middle-click as the PRIMARY click dispatch (skip the plain-click +
    2-attempt verify dance).

    Mirrors the success contract of the existing middle-click FALLBACK
    path inside ``execute()`` (currently lines ~571-606): dispatch
    middle-click, settle, verify URL is a detail page (on either the
    source or the new tab), set ``_opened_detail_in_new_tab=True`` so
    ``navigate_back`` routes via Ctrl+W (existing
    ``execute_close_detail_tab`` handler), and return a success
    ``StepResult``.

    Screenshot correctness (the #598 ask): after middle-click and
    ``ctrl+Tab`` to switch to the new tab, ``env.screenshot()``
    captures the focused window — which IS the new tab. The same
    primitive the existing fallback uses; the only thing this helper
    changes is the dispatch ORDER (primary instead of fallback).

    Returns:
      - ``StepResult`` (success or blank-newtab failure) when the
        hint is set and middle-click succeeded or definitively failed
        — caller returns this verbatim.
      - ``None`` when the hint is unset OR middle-click didn't open a
        new tab — caller falls through to the legacy plain-click chain
        so a missed middle-click doesn't drop the listing.
    """
    from ..checkpoint import StepResult  # local import — avoid TYPE_CHECKING cycle

    hints = getattr(step, "hints", None) or {}
    if not hints.get("open_in_new_tab"):
        return None

    logger.warning(
        "  [claude-click] PRIMARY middle-click (hints.open_in_new_tab=True) "
        "at (%d, %d) title=%r",
        x, y, (title or "")[:60],
    )

    # Snapshot tab count BEFORE middle-click. Some sites convert
    # middle-click to plain navigation (preventDefault on auxclick),
    # which would put the detail page on the SOURCE tab without
    # opening a new one. If we set ``_opened_detail_in_new_tab=True``
    # unconditionally on URL match, the subsequent ``navigate_back``
    # would route via ``execute_close_detail_tab`` (Ctrl+W) and close
    # the SOURCE tab — losing the runner's session. Gate the flag on
    # a real tab-count delta (#582 pattern used in the plain-click
    # path).
    tabs_before = _safe_tab_count(env)

    try:
        env.step(Action(action_type=ActionType.CLICK, params={
            "x": x, "y": y, "button": "middle",
        }))
        runner.costs["gpu_steps"] += 1
        runner.costs["gpu_seconds"] += 3
        runner.costs["proxy_mb"] += 5.0
        # Same cap the fallback uses — 2s for new-tab settle.
        adaptive_settle.settle_after_action(env, max_seconds=2.0)
    except Exception as exc:  # noqa: BLE001 — fall through on dispatch failure
        logger.warning(
            "  [claude-click] PRIMARY middle-click dispatch failed: %s "
            "— falling through to plain-click chain",
            exc,
        )
        return None

    # Verify (2 attempts, ctrl+Tab between) — mirror the fallback contract.
    for switch_attempt in range(2):
        url = runner._read_current_url()
        if not url:
            after = env.screenshot()
            url = runner._read_current_url(after)
        if url and site_config.is_detail_page(url, base_url=runner._results_base_url):
            tabs_after = _safe_tab_count(env)
            opened_new_tab = tabs_after > tabs_before > 0
            logger.warning(
                "  [claude-click] PRIMARY middle-click opened detail: %s "
                "(tabs %d → %d, new_tab=%s)",
                url[:80], tabs_before, tabs_after, opened_new_tab,
            )
            # Only set the new-tab flag when a real tab was opened.
            # When the click navigated the source tab in-place (some
            # sites preventDefault on auxclick), keeping the flag False
            # routes ``navigate_back`` via the existing CDP-back path
            # (#609) instead of Ctrl+W — which would otherwise close
            # the source tab and leave the runner without a browser.
            if opened_new_tab:
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
            runner._set_scroll_state(
                context="detail_top", url=url, page_downs=0, wheel_downs=0,
            )
            runner._listings_on_page += 1
            if getattr(runner, "_last_click_title", ""):
                runner._extracted_titles.append(runner._last_click_title)
            ctx.state["_executor_backend"] = "middle_primary"
            return StepResult(
                step_index=index, intent=step.intent, success=True,
                steps_used=1 + switch_attempt, duration=9.0,
            )
        if url and _looks_like_blank_newtab(url):
            # Middle-click landed on an empty newtab page — the card
            # region isn't a real link. Close the tab and let recovery
            # move on; do NOT fall through to plain-click (plain-click
            # on the same coords would just sit there with no nav).
            logger.warning(
                "  [claude-click] PRIMARY middle-click landed on blank "
                "tab (%s) — aborting",
                url_for_log(url),
            )
            try:
                env.step(Action(
                    action_type=ActionType.KEY_PRESS,
                    params={"keys": "ctrl+w"},
                ))
                time.sleep(0.5)
            except Exception:  # noqa: BLE001
                pass
            dynamic_verifier.record_item_completed(
                page=runner._current_page,
                item=getattr(runner, "_last_click_title", "") or title,
                url=url,
                success=False,
                reason="newtab_blank",
            )
            if getattr(runner, "_last_click_title", ""):
                runner._extracted_titles.append(runner._last_click_title)
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="newtab_blank",
            )
        if switch_attempt == 0:
            # Try the new tab if focus is still on the source.
            env.step(Action(
                action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+Tab"},
            ))
            adaptive_settle.settle_after_action(env, max_seconds=2.0)

    # Middle-click fired but neither tab landed on a detail page.
    # Fall through to legacy plain-click chain (which has its own
    # fallback) so we don't lose the listing — middle-click may have
    # been off-target and a fresh plain-click on the refined grounding
    # coords could still work.
    logger.info(
        "  [claude-click] PRIMARY middle-click didn't open detail "
        "— falling through to plain-click chain"
    )
    return None


def _safe_tab_count(env: Any) -> int:
    """#582: defensively read ``env.cdp_count_pages()`` as an int.

    Returns 0 when the env doesn't expose the method, the call raises,
    or the result isn't coercible to int. Tests use ``MagicMock`` envs
    that auto-create attributes returning ``MagicMock`` objects — those
    can't be compared with ``>``, so we coerce + swallow defensively.
    Production XdotoolGymEnv returns a real int; this is purely a
    test-shape compatibility shim.
    """
    fn = getattr(env, "cdp_count_pages", None)
    if not callable(fn):
        return 0
    try:
        return int(fn())
    except (TypeError, ValueError):
        return 0
    except Exception:  # noqa: BLE001 — never break the click path
        return 0


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
