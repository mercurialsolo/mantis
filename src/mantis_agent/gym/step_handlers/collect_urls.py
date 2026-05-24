"""CollectUrlsHandler — single-pass listing URL harvest (#615).

The extraction-loop body today is ``click → extract_url → scroll →
extract_data → navigate_back``. The ``click → extract_url`` chain costs
~3-5 seconds per iteration just to learn the URL of the listing the
brain just clicked. For the fan-out runner (#617) to work, each worker
needs to ``navigate(url)`` directly to its slice — which means the URL
list has to be known up front, not discovered iteratively.

This handler runs ONE Claude pass via ``ClaudeExtractor.find_all_listings``
to locate every visible card on the results page, then resolves each
card's anchor href via CDP ``Runtime.evaluate`` on
``document.elementFromPoint(vx, vy)`` — same screen-to-viewport
coordinate translation the click handler's ``cdp_click_at_point`` uses
(``xdotool_env.py:610``).

Provenance is CUA-pure under ``feedback_cua_cdp_post_action_verify.md``:
vision picks the target (find_all_listings), CDP only reads the
resolved attribute (.href). No DOM-derived targeting.

Output is stashed on ``runner._collected_urls`` for the fan-out runner
(#616, #617) to read. The handler also returns the URL count in
``StepResult.data`` for trace-level visibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .._runner_helpers import adaptive_content_settle
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


# JS payload: takes (sx, sy) in screen-space, translates to viewport
# coords using the same chromeH offset cdp_click_at_point applies,
# walks up the DOM from elementFromPoint to find the nearest ``<a>``
# ancestor, returns its absolute href. Returns None when no anchor is
# found above the point — caller treats that as a card without a
# resolvable URL and silently drops it from the partition.
_HREF_LOOKUP_JS = """
(() => {
  const oh = window.outerHeight;
  const ih = window.innerHeight;
  const chromeH = Math.max(0, oh - ih);
  const sx = {sx};
  const sy = {sy};
  const vx = sx;
  const vy = sy - chromeH;
  let el = document.elementFromPoint(vx, vy);
  if (!el) return null;
  while (el && el.tagName !== 'A') {
    el = el.parentElement;
  }
  if (!el) return null;
  return el.href || null;
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
        # not yet hydrated). Settle up to 4s here — collect_urls runs
        # ONCE per Phase-1 dispatch, so the cost is bounded.
        adaptive_content_settle(env, min_seconds=1.0, max_seconds=4.0)

        screenshot = env.screenshot()
        scan = extractor.find_all_listings(screenshot)
        # find_all_listings returns either list[(x, y, title)] or a
        # signal tuple (("empty",) / ("blocked",) / ("error",)) — drop
        # the signal cases as empty harvests rather than crashing.
        if not isinstance(scan, list):
            signal = scan[0] if scan else "unknown"
            logger.warning(
                "  [collect_urls] find_all_listings returned signal=%s — empty harvest",
                signal,
            )
            runner._collected_urls = []
            # Skip envelope (success=False, skip=True) so the runner's
            # retry policy advances past this step instead of burning
            # ~9 retries (each costs another Claude scan call) on the
            # same blocked / empty page state.
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"scan_signal:{signal}",
                skip=True, skip_reason=f"collect_urls_signal_{signal}",
            )

        runner.costs["claude_extract"] += 1
        card_count = len(scan)
        # Always-on WARNING so operators (and the orchestrator's local
        # CLI grep) see the scan outcome even when Modal trims worker
        # log tails. Format: ``[collect_urls] scan returned N cards``.
        logger.warning(
            "  [collect_urls] scan returned %d card(s)", card_count,
        )
        urls: list[str] = []
        seen: set[str] = set()
        unresolved = 0
        for x, y, _title in scan:
            href = self._resolve_href(env, int(x), int(y))
            if not href:
                unresolved += 1
                continue
            # Dedup: lazy-loaded results pages occasionally render the
            # same card twice with slightly different y; the URL is the
            # right primary key anyway.
            if href in seen:
                continue
            seen.add(href)
            urls.append(href)

        runner._collected_urls = urls
        runner._last_known_url = runner._last_known_url or ""
        resolved = len(urls)
        coverage = resolved / max(card_count, 1)
        # Always-on WARNING so the resolved-vs-found ratio is visible.
        # When unresolved == card_count the CDP href lookup is failing
        # universally (elementFromPoint miss, wrong chromeH calc, page
        # not yet hydrated) — operator needs this signal to diagnose.
        if coverage < 0.8:
            logger.warning(
                "  [collect_urls] coverage degraded: %d/%d cards resolved hrefs "
                "(%.0f%%, %d unresolved) — fan-out runner should consider "
                "sequential fallback",
                resolved, card_count, coverage * 100, unresolved,
            )
        else:
            logger.warning(
                "  [collect_urls] resolved %d/%d card hrefs (%.0f%% coverage)",
                resolved, card_count, coverage * 100,
            )

        # Fail-fast skip envelope when EVERY card failed CDP resolution.
        # Triggers the same runner-side advance as the scan-signal path
        # — no point retrying because the cdp_evaluate JS will return
        # the same null on every retry against the same screenshot.
        if card_count > 0 and resolved == 0:
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"urls:0/{card_count}:all_unresolved",
                skip=True, skip_reason="collect_urls_all_unresolved",
            )

        return StepResult(
            step_index=index, intent=step.intent,
            success=bool(urls),
            data=f"urls:{resolved}/{card_count}",
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
