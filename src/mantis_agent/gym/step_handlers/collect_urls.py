"""CollectUrlsHandler — multi-viewport listing URL harvest (#615, #638).

The extraction-loop body today is ``click → extract_url → scroll →
extract_data → navigate_back``. The ``click → extract_url`` chain costs
~3-5 seconds per iteration just to learn the URL of the listing the
brain just clicked. For the fan-out runner (#617) to work, each worker
needs to ``navigate(url)`` directly to its slice — which means the URL
list has to be known up front, not discovered iteratively.

This handler walks the page through multiple viewport stages, calling
``ClaudeExtractor.find_all_listings`` at each stop and resolving each
card's anchor href via CDP ``Runtime.evaluate`` on
``document.elementFromPoint(vx, vy)`` — same screen-to-viewport
coordinate translation the click handler's ``cdp_click_at_point`` uses
(``xdotool_env.py:610``). Mirrors the existing viewport-stage scanner
in ``ClaudeGuidedClickHandler`` (gym/step_handlers/click.py:120).

#638: the original single-viewport implementation harvested only 3
URLs vs the ~25 listings a typical results page actually carries (the
rest sit below the fold). Multi-viewport scan brings throughput to the
full page complement at the cost of N Claude calls instead of 1.

Provenance is CUA-pure under ``feedback_cua_cdp_post_action_verify.md``:
vision picks the target (find_all_listings), CDP only reads the
resolved attribute (.href). No DOM-derived targeting.

Output is stashed on ``runner._collected_urls`` for the fan-out runner
(#616, #617) to read. The handler also returns the URL count in
``StepResult.data`` for trace-level visibility.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from .._runner_helpers import adaptive_content_settle
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)

# Default cap on viewport stages — matches ClaudeGuidedClickHandler's
# ``_max_viewport_stages`` (4). Bounds the per-Phase-1 Claude scan cost
# at ~$0.02 × 4 = ~$0.08 in the worst case. Plans that need more can
# override via ``step.hints['max_viewport_stages']``.
DEFAULT_MAX_VIEWPORT_STAGES = 4

# How many Page_Down keypresses per viewport stage. 2 page-downs ≈
# 1.5 viewport heights, which gives ~30-40% overlap between adjacent
# stages — keeps each card visible to find_all_listings in at least
# one stage even when its centroid sits near a stage boundary.
PAGE_DOWNS_PER_STAGE = 2


# JS payload: takes (sx, sy) in screen-space, translates to viewport
# coords using the same chromeH offset cdp_click_at_point applies,
# walks the element tree at that point to find a navigable URL. Returns
# the resolved URL string or null when no candidate is found.
#
# Resolution priority (matches what production listing cards use across
# common marketplace sites):
#   1. ``<a href>`` ancestor — BoatTrader's listing card wraps in an
#      anchor. ``el.closest('a[href]')`` finds it cross-browser.
#   2. ``[href]`` ancestor — anchor-less elements that nevertheless
#      carry an href attribute (rare but seen on some SPAs).
#   3. ``<a href>`` descendant of the topmost ancestor at the point —
#      catches cards where the click target is a wrapping ``<div>``
#      with the navigable ``<a>`` rendered INSIDE (BoatTrader's
#      sponsored-listing variant does this).
#   4. ``data-href`` / ``data-url`` on any ancestor — common React
#      pattern for click-handled divs that synthesise navigation in JS.
#
# Returns null only when none of the above yield a value, indicating
# the card is genuinely non-navigable (e.g. an in-page modal trigger).
_HREF_LOOKUP_JS = """
(() => {
  const oh = window.outerHeight;
  const ih = window.innerHeight;
  const chromeH = Math.max(0, oh - ih);
  const sx = {sx};
  const sy = {sy};
  const vx = sx;
  const vy = sy - chromeH;
  const el = document.elementFromPoint(vx, vy);
  if (!el) return null;
  // 1. anchor ancestor (most common)
  const anchor = el.closest('a[href]');
  if (anchor && anchor.href) return anchor.href;
  // 2. any [href] ancestor
  const hrefAncestor = el.closest('[href]');
  if (hrefAncestor && hrefAncestor.href) return hrefAncestor.href;
  // 3. find an anchor inside the topmost "card-like" ancestor at the
  //    point — walk up to the nearest element with role="link",
  //    data-test*="listing", or a class hint, then look for an <a>
  //    descendant. Cap at 5 levels to bound search.
  let card = el;
  for (let i = 0; i < 5 && card; i++) {
    const aDesc = card.querySelector && card.querySelector('a[href]');
    if (aDesc && aDesc.href) return aDesc.href;
    card = card.parentElement;
  }
  // 4. data-href / data-url ancestor (React click-handled cards)
  const dataAncestor = el.closest('[data-href], [data-url]');
  if (dataAncestor) {
    const v = dataAncestor.getAttribute('data-href')
              || dataAncestor.getAttribute('data-url');
    if (v) return v;
  }
  return null;
})()
"""


class CollectUrlsHandler:
    """Implements :class:`~..step_context.StepHandler` for ``collect_urls``.

    Acceptance gate (issue #615): if URL resolution returns fewer than
    80% of the cards ``find_all_listings`` located, the handler still
    reports success but logs a WARNING — the fan-out runner (#616)
    consults this fraction to decide whether to fall back to the
    sequential extraction loop. Modal suppresses INFO logs in
    production (``feedback_warning_level_for_modal_observability.md``)
    so this signal MUST be WARNING-level to be debuggable from logs.
    """

    step_type = "collect_urls"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        index = int(ctx.state.get("index", 0))

        if not extractor:
            logger.warning("  [collect_urls] no extractor wired — returning empty")
            runner._collected_urls = []
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="no_extractor",
            )

        # Settle for the cold-load case. BoatTrader on a proxied
        # cold-cache fetch can take 3-5s before listings render; the
        # 1s pre-extract pause is the right floor for warm pages but
        # leaves Phase-1 fan-out scanning a half-rendered page (cards
        # not yet hydrated). Settle up to 4s before the first viewport
        # scan; subsequent viewports use the shorter intra-loop wait.
        adaptive_content_settle(env, min_seconds=1.0, max_seconds=4.0)

        # #638: per-step override for the viewport-stage cap. Plans that
        # know the catalog is paginated and small can pin a tighter cap;
        # plans on infinite-scroll feeds can lift it. Falls back to the
        # module default (4) which matches the click handler's own
        # ``_max_viewport_stages``.
        max_stages = int(
            (step.hints or {}).get(
                "max_viewport_stages", DEFAULT_MAX_VIEWPORT_STAGES,
            )
        )

        # #638 axis 2: accumulate across calls so multi-page Phase-1
        # plans (navigate(page-1) → collect_urls → navigate(page-2) →
        # collect_urls → ...) carry forward URLs from prior pages.
        # ``urls`` and ``seen`` are seeded from ``runner._collected_urls``
        # so this invocation's dedup catches both within-this-page
        # duplicates (viewport overlap) AND cross-page duplicates
        # (featured listings rendered on multiple pages). On a single-
        # page invocation the seed is empty and behaviour is unchanged.
        urls: list[str] = list(getattr(runner, "_collected_urls", []) or [])
        seen: set[str] = set(urls)
        new_urls_this_call = 0  # tracked for the per-call log line
        total_cards = 0
        total_unresolved = 0
        first_signal: str | None = None
        consecutive_empty_stages = 0

        # Reset to the top of the results page once, then advance one
        # viewport per stage. Mirrors ClaudeGuidedClickHandler's
        # deterministic Home → Page_Down chain so each viewport is
        # reproducible across retries.
        self._scroll_to_top(env)

        for stage in range(max_stages):
            if stage > 0:
                self._page_down_to_stage(env, stage)
                # Short settle between stages — lazy-loaded marketplaces
                # often render the next-viewport cards within ~1s of the
                # scroll event firing.
                adaptive_content_settle(env, min_seconds=0.5, max_seconds=2.0)

            screenshot = env.screenshot()
            scan = extractor.find_all_listings(screenshot)
            runner.costs["claude_extract"] += 1

            # find_all_listings returns either list[(x, y, title)] or a
            # signal tuple (("empty",) / ("blocked",) / ("error",)).
            # Blocked / error at stage 0 is fatal (whole page is gone);
            # blocked at later stages may just mean we scrolled off the
            # bottom — treat as empty and let the early-exit fire.
            if not isinstance(scan, list):
                signal = scan[0] if scan else "unknown"
                if stage == 0:
                    if first_signal is None:
                        first_signal = signal
                    logger.warning(
                        "  [collect_urls] stage 0: signal=%s — empty harvest",
                        signal,
                    )
                    break
                logger.warning(
                    "  [collect_urls] stage %d: signal=%s (treating as empty)",
                    stage, signal,
                )
                consecutive_empty_stages += 1
                if consecutive_empty_stages >= 2:
                    break
                continue

            card_count = len(scan)
            total_cards += card_count
            new_this_stage = 0
            unresolved_this_stage = 0
            for x, y, _title in scan:
                href = self._resolve_href(env, int(x), int(y))
                if not href:
                    unresolved_this_stage += 1
                    continue
                # Cross-stage dedup: the same card often appears in
                # multiple adjacent viewports because PAGE_DOWNS_PER_STAGE
                # is < 1 full viewport — that overlap is intentional
                # (keeps cards near stage boundaries discoverable), and
                # the seen set absorbs the redundancy.
                if href in seen:
                    continue
                seen.add(href)
                urls.append(href)
                new_this_stage += 1
                new_urls_this_call += 1
            total_unresolved += unresolved_this_stage

            logger.warning(
                "  [collect_urls] stage %d: scan=%d cards, "
                "new_urls=%d, unresolved=%d, total_so_far=%d",
                stage, card_count, new_this_stage,
                unresolved_this_stage, len(urls),
            )

            # Early exit: two consecutive stages added zero new URLs
            # means we've either run off the bottom or the page only
            # had what we already have. Caps wasted Claude calls on
            # short results pages without forcing operators to tune
            # max_viewport_stages per plan.
            if new_this_stage == 0:
                consecutive_empty_stages += 1
                if consecutive_empty_stages >= 2:
                    logger.warning(
                        "  [collect_urls] early exit at stage %d "
                        "(2 consecutive empty stages)", stage,
                    )
                    break
            else:
                consecutive_empty_stages = 0

        runner._collected_urls = urls
        runner._last_known_url = runner._last_known_url or ""
        resolved = len(urls)

        # Stage-0 signal path — no scan succeeded, advance past the step.
        if first_signal is not None and resolved == 0:
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"scan_signal:{first_signal}",
                skip=True,
                skip_reason=f"collect_urls_signal_{first_signal}",
            )

        # All-unresolved path — scans succeeded but every CDP href
        # lookup returned null. Same skip envelope as before so the
        # runner advances past the step.
        if total_cards > 0 and resolved == 0:
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"urls:0/{total_cards}:all_unresolved",
                skip=True, skip_reason="collect_urls_all_unresolved",
            )

        coverage = resolved / max(total_cards, 1)
        # The CUMULATIVE total (resolved) is what the orchestrator reads;
        # the NEW count (new_urls_this_call) shows what this specific
        # invocation contributed. Same value on single-page Phase-1, but
        # diverges on multi-page (#638 axis 2) where each subsequent
        # navigate+collect_urls pair adds to the running total.
        if coverage < 0.8:
            logger.warning(
                "  [collect_urls] FINAL: %d new URLs this call, "
                "%d cumulative from %d cards across viewport stages "
                "(%.0f%% coverage, %d unresolved)",
                new_urls_this_call, resolved, total_cards,
                coverage * 100, total_unresolved,
            )
        else:
            logger.warning(
                "  [collect_urls] FINAL: %d new URLs this call, "
                "%d cumulative from %d cards across viewport stages "
                "(%.0f%% coverage)",
                new_urls_this_call, resolved, total_cards, coverage * 100,
            )

        return StepResult(
            step_index=index, intent=step.intent,
            success=bool(urls),
            data=f"urls:{resolved}/{total_cards}",
        )

    @staticmethod
    def _scroll_to_top(env) -> None:
        """Reset the page to the top before stage 0. Mirrors the
        Home-key reset in ClaudeGuidedClickHandler's viewport scanner.
        Best-effort — never raises."""
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001 — never break the scan
            logger.debug("collect_urls scroll-to-top failed: %s", exc)

    @staticmethod
    def _page_down_to_stage(env, stage: int) -> None:
        """Advance ``PAGE_DOWNS_PER_STAGE`` viewport positions for the
        given stage. Stage 0 is the top (no Page_Down), stage N issues
        ``N × PAGE_DOWNS_PER_STAGE`` Page_Down presses cumulatively —
        but since each call only advances ONE stage's worth (not
        cumulative from top), this is called per-stage."""
        try:
            for _ in range(PAGE_DOWNS_PER_STAGE):
                env.step(Action(
                    action_type=ActionType.KEY_PRESS,
                    params={"keys": "Page_Down"},
                ))
                time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "collect_urls page_down stage %d failed: %s",
                stage, exc,
            )

    @staticmethod
    def _resolve_href(env, x: int, y: int) -> str:
        """Return the href of the nearest ``<a>`` ancestor of the element
        at screen (x, y), or empty string on any failure.

        Defensive against environments without CDP (returns ""); the
        caller skips that card silently.
        """
        cdp = getattr(env, "cdp_evaluate", None)
        if cdp is None:
            return ""
        try:
            js = _HREF_LOOKUP_JS.replace("{sx}", str(x)).replace("{sy}", str(y))
            result = cdp(js)
        except Exception as exc:  # noqa: BLE001 — never break the loop
            logger.debug("collect_urls href lookup raised: %s", exc)
            return ""
        if not isinstance(result, str):
            return ""
        return result
