"""ClaudeStepHandler — extract_url + extract_data + deep-extract subroutine.

Phase 2 of EPIC #161, fourth handler extraction. Lifts
``MicroPlanRunner._execute_claude_step`` (200 LOC) plus the
``_extract_listing_data_deep`` subroutine (137 LOC) — together the
single biggest concentration of extraction logic on the runner —
into one cohesive handler module.

Two step types share this body:

- ``extract_url``: single screenshot → ``ClaudeExtractor.extract`` → URL
  field; dedup against ``self._seen_urls``; return DUPLICATE if seen.
- ``extract_data``: multi-viewport deep extract that visits up to 6
  viewports, clicks safe reveal controls (``find_listing_content_control``),
  then asks Claude to consolidate fields across all captured screenshots
  (``extract_multi``). Includes the cache-aware short-circuit added in
  PR #166: peeks ``env.current_url`` BEFORE the deep call; on a fresh
  cache hit emits the cached lead summary and skips the entire Claude
  pipeline (~$0.04/item saved).

Behavior is identical to the in-runner version. The runner method
becomes a delegating shim. ``_current_item_label`` stays on the runner
for now; it's a 17-LOC helper that ``_extract_listing_data_deep`` and
the dispatch in ``run()`` both consume.

What the handler reads from :class:`StepContext`:

- ``env``               — screenshot / step / current_url
- ``extractor``         — extract / extract_multi /
                          find_listing_content_control / verify_gate
- ``dynamic_verifier``  — record_item_completed
- ``extraction_cache``  — get / put (per-request opt-in from PR #166)

What it reads from the runner via parent back-reference:

- ``costs``                       — cost meter aliased dict
- ``_seen_urls``                  — listings dedup set
- ``_current_page``, ``_last_known_url``, ``_last_extracted``
- ``_set_scroll_state``           — delegates to BrowserState
- ``_best_effort_current_url``    — pre-extract URL peek
- ``_current_item_label``         — 17-LOC helper, runs against either the
                                    last clicked title or a year/make/model
                                    fallback
- ``_lead_key``, ``_lead_has_phone`` — backward-compat shims to ListingDedup
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


def _resolve_extraction_context(runner: "MicroPlanRunner", data: object | None):
    """Classify the extracted page as SEARCH_TILE / DETAIL_PAGE / UNKNOWN.

    Issue #236: ``ExtractionResult.missing_required_reason`` accepts a
    context arg to choose between strict (detail-page) and loose
    (search-tile) required-field contracts. The cleanest signal is the
    URL the extractor was reading, classified via the runner's
    ``SiteConfig.is_detail_page`` / ``is_results_page`` (those already
    encode the per-domain regex authoring).

    Returns ``ExtractionContext.UNKNOWN`` when no SiteConfig is
    configured or its patterns are empty — preserves legacy behavior
    for runs without a configured site.
    """
    from ...extraction import ExtractionContext

    site_config = getattr(runner, "site_config", None)
    if site_config is None:
        return ExtractionContext.UNKNOWN
    url = (
        getattr(data, "url", "")
        or getattr(runner, "_last_known_url", "")
        or ""
    )
    if not url:
        return ExtractionContext.UNKNOWN
    try:
        if site_config.is_detail_page(url):
            return ExtractionContext.DETAIL_PAGE
        if site_config.is_results_page(url):
            return ExtractionContext.SEARCH_TILE
    except Exception:  # noqa: BLE001 — never break the extraction path
        return ExtractionContext.UNKNOWN
    return ExtractionContext.UNKNOWN


def _check_already_seen(runner, ctx: StepContext) -> str | None:
    """Consult the host's seen-URL predicate at the top of
    ``extract_data``.

    Returns the matched URL when the runner should short-circuit
    with ``skip_reason='already_seen'``; returns ``None`` to
    proceed with the normal extract path.

    Mantis owns the timing window (post-navigate / pre-extract)
    and the applicability gate (detail-page only via
    :meth:`SiteConfig.is_detail_page`). The host owns the
    predicate's policy entirely — URL match, content hash, CRM
    lookup, anything. Issue #255.

    A buggy predicate (raises on a malformed URL, hits a network
    timeout, …) must NOT halt the run. Over-extracting is the
    safe failure mode; under-extracting (silently skipping a real
    new lead) is the dangerous one. So any predicate exception is
    swallowed and we fall through to the normal extract path.
    """
    predicate = getattr(runner, "seen_url_predicate", None)
    if predicate is None:
        return None
    try:
        current_url = runner._best_effort_current_url()
    except Exception:  # noqa: BLE001 — never break extract on CDP glitch
        return None
    if not current_url:
        return None
    # Applicability gate: only fire on canonical detail pages so
    # search / results URLs aren't accidentally deduped.
    site_config = ctx.site_config or getattr(runner, "site_config", None)
    if site_config is not None:
        try:
            if not site_config.is_detail_page(current_url):
                return None
        except Exception:  # noqa: BLE001
            return None
    try:
        if predicate(current_url):
            return current_url
    except Exception as exc:  # noqa: BLE001
        logger.debug("seen_url_predicate raised: %s", exc)
        return None
    return None


def _resolve_skip_envelope(extractor, *, rejection_key: str) -> tuple[bool, str | None]:
    """Look up a rejection key in ``extractor.schema.rejection_intents``
    and return the StepResult skip envelope (``skip``, ``skip_reason``).

    Issue #246: when a recipe annotates a rejection key as
    ``"skip"``, the rejection is terminal-for-this-row and a host
    orchestrator should advance past it without retrying. The
    runner surfaces that intent via ``StepResult.skip=True``;
    ``skip_reason`` carries the recipe-author key (``"dealer"``,
    ``"incomplete_required"``, …) so hosts can branch on the
    specific kind. Any other intent (``"extract_more"``,
    ``"retry"``) or a missing entry leaves ``skip=False`` —
    matches today's behavior, host-side retry logic continues to
    apply.
    """
    schema = getattr(extractor, "schema", None)
    if schema is None:
        return False, None
    intent = schema.rejection_intents.get(rejection_key, "")
    if intent == "skip":
        return True, rejection_key
    return False, None


class ClaudeStepHandler:
    """Implements :class:`~..step_context.StepHandler` for Claude-only steps.

    Bound to ``extract_url``. The ``extract_data`` type is handled by
    the same body but takes a different code path; the runner's
    ``_execute_step`` continues to dispatch on ``step.claude_only``
    rather than the registry for now, so the handler is reachable via
    the runner shim. The dispatch swap happens in the cleanup PR.
    """

    step_type = "extract_url"

    def __init__(self, runner: "MicroPlanRunner") -> None:
        self.parent = runner

    def execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        dynamic_verifier = ctx.dynamic_verifier
        extraction_cache = ctx.extraction_cache
        index = int(ctx.state.get("index", 0))

        if not extractor:
            return StepResult(step_index=index, intent=step.intent, success=False)

        # Pre-settle — page may still be rendering after scroll.
        # Issue #259: adaptive — exits early when consecutive
        # screenshots stabilize (typical ~0.3-0.4s on static pages),
        # caps at the original 1.0s on pages that keep changing.
        from .._runner_helpers import adaptive_content_settle
        adaptive_content_settle(env, min_seconds=0.2, max_seconds=1.0)

        screenshot = env.screenshot()

        if step.type == "extract_url":
            data = extractor.extract(screenshot)
            url = data.url if data else ""

            # #585: validate URL matches the recipe's detail-page pattern
            # when we're in the extraction section. Catches the wrong-page
            # case where a click landed on a marketing CTA (``/boat-loans/``,
            # ``/financing/`` etc.) and extract_url would otherwise accept
            # it as a "listing" → downstream extract_data runs on a
            # marketing page → UNKNOWN-filled junk lead.
            # Gated on (1) section=="extraction" so non-marketplace plans
            # aren't affected, and (2) detail_page_pattern present so
            # plans without the pattern (analysis-stage-only configs)
            # don't break.
            site_config = getattr(runner, "site_config", None)
            if (
                url
                and step.section == "extraction"
                and site_config is not None
                and getattr(site_config, "detail_page_pattern", "")
                and not site_config.is_detail_page(url)
            ):
                logger.warning(
                    "  [extract_url] WRONG_PAGE — url=%s doesn't match "
                    "detail_page_pattern=%r — failing step for recovery",
                    url[:120], site_config.detail_page_pattern,
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=(
                        f"WRONG_PAGE|{url}|"
                        f"expected={site_config.detail_page_pattern}"
                    ),
                    failure_class="wrong_target",
                )

            # Dedup check (Phase 4: scanner owns the predicate now).
            if runner.scanner.is_duplicate(url):
                # WARNING-level so the dedup decision shows up in Modal
                # logs (INFO is suppressed in production). Without this
                # the "0 leads, halt=duplicate_listing" diagnosis is
                # un-debuggable from logs alone (#600 follow-up).
                logger.warning(
                    "  [dedup] extract_url already seen: %s (step=%d)",
                    url[:80], index,
                )
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=runner._current_item_label(data),
                    url=url,
                    success=True,
                    reason="duplicate_url_skipped",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False, data=f"DUPLICATE|{url}",
                )
            # #627: cross-worker dedup. The per-container scanner caught
            # within-this-container repeats above; the shared seen-set
            # catches the case where a sibling fan-out worker already
            # extracted this URL. Short-circuit so we don't pay the
            # extract_data Claude cost (~$0.20) on a confirmed duplicate.
            # NullSharedSeenSet.contains() returns False → unchanged
            # behaviour for single-worker / non-fanout runs.
            shared = getattr(runner, "_shared_seen_set", None)
            if url and shared is not None and shared.contains(url):
                logger.warning(
                    "  [shared-seen] cross-worker dedup hit: %s "
                    "(step=%d, shared set size=%d)",
                    url[:80], index, shared.size(),
                )
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=runner._current_item_label(data),
                    url=url,
                    success=True,
                    reason="cross_worker_dedup_skipped",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False, data=f"DUPLICATE|cross_worker|{url}",
                )
            if url:
                # Issue #603: do NOT mark_seen here. extract_url is a
                # PRE-extraction probe; the URL hasn't been deep-extracted
                # yet, only navigated to. Marking seen here causes the
                # next plan step (``extract_data``) to fire DUPLICATE
                # against the same URL we're about to extract, dropping
                # the lead on every first iteration. extract_data's
                # success path (below) is the sole mark-seen authority
                # — that way ``mark_seen`` semantically means "lead
                # data captured", and cross-iteration dedup still works
                # because iteration N+1's extract_url sees iteration N's
                # extract_data mark.
                runner._last_known_url = url
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_attempted_url": url,
                    "last_attempted_step": index,
                    "last_attempted_at": time.time(),
                }

            return StepResult(
                step_index=index, intent=step.intent,
                success=bool(url), data=f"URL:{url}" if url else "",
            )

        elif step.type == "extract_data":
            # Issue #255: cross-session already-seen short-circuit.
            # Runs BEFORE the in-session cache check — the cache is
            # an intra-session optimization, the predicate is host-
            # state (master.csv, CRM, …) crossing sessions. Mantis
            # owns the timing window only; the predicate's policy
            # (URL match / content hash / CRM lookup) is host-owned.
            # Applicability gate: only fire on canonical detail
            # pages (#236 ``is_detail_page``) so search/results URLs
            # aren't accidentally deduped if a host's predicate
            # would match them.
            already_seen = _check_already_seen(runner, ctx)
            if already_seen is not None:
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False,
                    skip=True, skip_reason="already_seen",
                    data=f"already_seen|url={already_seen[:160]}",
                )

            # Cache short-circuit BEFORE the expensive deep-extract Claude
            # call. We peek the browser's current URL (cheap CDP read, no
            # tokens). If it's a fresh cache hit, emit the cached lead and
            # skip the Claude work entirely (~$0.04/item saved). When the
            # URL isn't yet known (e.g. agent hasn't navigated into a card
            # yet) we fall through to the normal deep-extract path.
            if extraction_cache is not None and extraction_cache.read_enabled:
                pre_url = runner._best_effort_current_url()
                cached = extraction_cache.get(pre_url) if pre_url else None
                if cached is not None:
                    logger.info("  [cache] hit for %s — skipping deep-extract", pre_url[:80])
                    runner.scanner.mark_seen(pre_url)
                    runner._last_known_url = pre_url
                    runner._last_extracted = {
                        **runner._last_extracted,
                        "last_completed_url": pre_url,
                        "last_completed_summary": cached.summary,
                        "last_completed_step": index,
                        "last_completed_at": time.time(),
                    }
                    dynamic_verifier.record_item_completed(
                        page=runner._current_page,
                        item=cached.item_label or runner._current_item_label(None),
                        url=pre_url,
                        success=True,
                        reason="cache_hit",
                    )
                    # #508: cache stores `fields` alongside summary, so
                    # cache-hit short-circuits still surface structured
                    # rows. Older caches without fields → empty dict.
                    return StepResult(
                        step_index=index, intent=step.intent,
                        success=True, data=cached.summary,
                        extracted_fields=dict(
                            getattr(cached, "fields", {}) or {}
                        ),
                    )

            data, _actions_used = self._extract_listing_data_deep(screenshot, ctx)
            item_label = runner._current_item_label(data)

            # Dedup: short-circuit if this URL was extracted in a prior loop
            # iteration. Mirrors the extract_url dedup above so loop_count
            # iterations always advance to a fresh item instead of re-clicking
            # the same card. Without this the runner can re-extract the same
            # listing and produce duplicate entries in the lead set.
            extracted_url = getattr(data, "url", "") if data else ""
            if runner.scanner.is_duplicate(extracted_url):
                # WARNING-level (#600 follow-up): the "0 leads, halt=
                # duplicate_listing" outcome turned out to be this gate
                # firing immediately after extract_url marked the SAME
                # URL seen one step earlier. We need the per-step URL
                # in production logs to verify the mark→check race.
                logger.warning(
                    "  [dedup] extract_data already seen: %s (step=%d)",
                    extracted_url[:80], index,
                )
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=item_label,
                    url=extracted_url,
                    success=True,
                    reason="duplicate_url_skipped",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False, data=f"DUPLICATE|{extracted_url}",
                )

            if data and getattr(data, "url", ""):
                runner._last_known_url = data.url
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_attempted_url": data.url,
                    "last_attempted_step": index,
                    "last_attempted_at": time.time(),
                }
            if data and data.is_viable():
                runner.scanner.mark_seen(extracted_url)
                # #627: also write to the cross-worker shared seen-set
                # so sibling fan-out workers can short-circuit on this
                # URL. NullSharedSeenSet.add() is a no-op for non-fanout
                # runs — preserves single-worker behaviour.
                shared = getattr(runner, "_shared_seen_set", None)
                if shared is not None and extracted_url:
                    try:
                        shared.add(extracted_url)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("[shared-seen] add raised: %s", exc)
                summary = data.to_summary()
                # Persist to cache so subsequent runs (or loop iterations)
                # can short-circuit. No-op when cache_write is disabled.
                if extraction_cache is not None and extracted_url:
                    try:
                        extraction_cache.put(
                            extracted_url,
                            summary,
                            fields=dict(getattr(data, "extracted_fields", {}) or {}),
                            item_label=item_label,
                        )
                    except Exception as exc:  # noqa: BLE001 — cache is best-effort
                        logger.warning("extraction cache put failed: %s", exc)
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_completed_url": data.url,
                    "last_completed_key": runner._lead_key(summary),
                    "last_completed_summary": summary,
                    "last_completed_has_phone": runner._lead_has_phone(summary),
                    "last_completed_step": index,
                    "last_completed_at": time.time(),
                }
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason="viable_lead",
                )
                # #508: pass the schema-keyed dict through verbatim so the
                # aggregator can emit structured rows. ``data=summary`` is
                # the legacy pipe-delimited string (kept for back-compat);
                # ``extracted_fields`` is the canonical structured copy.
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=True, data=summary,
                    extracted_fields=dict(
                        getattr(data, "extracted_fields", {}) or {}
                    ),
                )
            if data and data.dealer_reason():
                reason = data.dealer_reason()
                # WARNING-level (#600 follow-up) for production visibility.
                logger.warning(
                    "  [extract] Rejected non-private listing: %s url=%s",
                    reason, (data.url or "")[:80],
                )
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_rejected_url": data.url,
                    "last_rejected_reason": f"dealer:{reason}",
                    "last_rejected_step": index,
                    "last_rejected_at": time.time(),
                }
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason=f"rejected_dealer:{reason}",
                )
                skip, skip_reason = _resolve_skip_envelope(
                    extractor, rejection_key="dealer",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False,
                    data=f"REJECTED_DEALER|{reason}|{data.to_summary()[:160]}",
                    skip=skip, skip_reason=skip_reason,
                )
            # Issue #236: pick the right required-field contract based
            # on which kind of page the extractor read. ``DETAIL_PAGE``
            # enforces the strict canonical set (e.g. year + make for
            # marketplace listings); ``SEARCH_TILE`` enforces the
            # looser tile_required_fields set when defined (typically
            # just ``url``) so the runner keeps the row to drive a
            # follow-up navigate-into-detail. Source of truth: the
            # extracted URL classified via SiteConfig. Default
            # ``UNKNOWN`` when no SiteConfig is configured —
            # preserves legacy behavior for non-listings sites.
            extraction_context = _resolve_extraction_context(runner, data)
            if data and data.missing_required_reason(extraction_context):
                reason = data.missing_required_reason(extraction_context)
                # WARNING-level (#600 follow-up). Include URL + the
                # extraction_context so we can tell DETAIL_PAGE-strict
                # vs SEARCH_TILE-loose contracts apart in production.
                logger.warning(
                    "  [extract] Rejected incomplete lead: %s url=%s ctx=%s",
                    reason, (data.url or "")[:80], extraction_context,
                )
                runner._last_extracted = {
                    **runner._last_extracted,
                    "last_rejected_url": data.url,
                    "last_rejected_reason": f"incomplete:{reason}",
                    "last_rejected_step": index,
                    "last_rejected_at": time.time(),
                }
                dynamic_verifier.record_item_completed(
                    page=runner._current_page,
                    item=item_label,
                    url=data.url,
                    success=True,
                    reason=f"rejected_incomplete:{reason}",
                )
                skip, skip_reason = _resolve_skip_envelope(
                    extractor, rejection_key="incomplete_required",
                )
                return StepResult(
                    step_index=index, intent=step.intent,
                    success=False,
                    data=f"REJECTED_INCOMPLETE|{reason}|{data.to_summary()[:160]}",
                    skip=skip, skip_reason=skip_reason,
                )
            # Catch-all: extract returned but didn't pass viable / dealer
            # / missing-required gates. Previously silent; promote to
            # WARNING (#600 follow-up) so the "0 leads, halt=…" outcome
            # can be triaged from logs alone.
            logger.warning(
                "  [extract] Catch-all reject (not viable, no dealer flag, no required-miss): "
                "url=%s data_present=%s",
                (getattr(data, "url", "") or "")[:80],
                bool(data),
            )
            dynamic_verifier.record_item_completed(
                page=runner._current_page,
                item=item_label,
                url=getattr(data, "url", "") if data else "",
                success=False,
                reason="extract_data_incomplete",
            )
            return StepResult(
                step_index=index, intent=step.intent,
                success=False, data=data.raw_response[:100] if data else "",
            )

        return StepResult(step_index=index, intent=step.intent, success=False)

    def _extract_listing_data_deep(self, initial_screenshot, ctx: StepContext):
        """Capture top, expanded description, and lower detail viewports.

        Private-seller phones often appear inside seller-written
        descriptions, and those descriptions can be collapsed. This routine is
        the execution-time policy for dynamic pages: inspect each viewport,
        click only safe reveal controls, then ask Claude to extract from the
        complete screenshot set.
        """
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor

        screenshots = []
        labels = []
        controls_clicked = 0
        clicked_keys: set[str] = set()
        max_screenshots = 12
        max_viewports = 6

        def capture(label: str):
            if len(screenshots) >= max_screenshots:
                return None
            try:
                shot = env.screenshot()
                screenshots.append(shot)
                labels.append(label)
                return shot
            except Exception as e:
                logger.warning(f"  [deep-extract] screenshot failed: {e}")
                return None

        if initial_screenshot is not None:
            screenshots.append(initial_screenshot)
            labels.append("initial extraction viewport")

        # Start from the top so the final prompt sees title, price, seller card,
        # and any safe contact/phone reveal controls before scanning details.
        # Issue #259: adaptive — browsers re-layout when jumping to top
        # (sticky nav, hero image, JS widgets). Static pages exit ~0.5s;
        # JS-heavy pages still pay the full 1.5s cap.
        try:
            env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            from .._runner_helpers import adaptive_content_settle
            adaptive_content_settle(env, min_seconds=0.3, max_seconds=1.5)
        except Exception:
            pass
        runner._set_scroll_state(
            context="detail_extract",
            url=runner._last_known_url,
            page_downs=0,
            wheel_downs=0,
            label="top/contact area",
            flush=True,
        )
        top_shot = capture("top/contact area")

        for viewport in range(max_viewports):
            runner._set_scroll_state(
                context="detail_extract",
                url=runner._last_known_url,
                page_downs=viewport,
                wheel_downs=0,
                viewport_stage=viewport,
                label=f"detail viewport {viewport + 1}",
                flush=True,
            )
            if viewport == 0 and top_shot is not None:
                shot = top_shot
            else:
                shot = capture(f"detail viewport {viewport + 1}")
            if shot is None:
                break

            target = extractor.find_listing_content_control(shot)
            runner.costs["claude_extract"] += 1

            if target:
                key = (
                    f"{target.get('action', '')}:{target.get('label', '').lower()}:"
                    f"{target['x'] // 25}:{target['y'] // 25}"
                )
                if key not in clicked_keys:
                    clicked_keys.add(key)
                    try:
                        env.step(Action(
                            action_type=ActionType.CLICK,
                            params={"x": target["x"], "y": target["y"]},
                        ))
                        controls_clicked += 1
                        # Issue #259: reveal click triggers XHR +
                        # DOM update + reveal animation. JS-backed
                        # reveals (BoatTrader Show Phone, etc) need
                        # the full 2s; static-content reveals
                        # ("Show more" expanding existing DOM) exit
                        # at ~0.6s.
                        from .._runner_helpers import adaptive_content_settle
                        adaptive_content_settle(env, min_seconds=0.5, max_seconds=2.0)
                        capture(
                            f"after {target.get('action', 'expand')} "
                            f"{target.get('label', '')[:40]}"
                        )
                        runner._set_scroll_state(
                            context="detail_extract",
                            url=runner._last_known_url,
                            page_downs=viewport,
                            wheel_downs=0,
                            viewport_stage=viewport,
                            label=(
                                f"after {target.get('action', 'expand')} "
                                f"{target.get('label', '')[:40]}"
                            ),
                            flush=True,
                        )
                        logger.info(
                            "  [deep-extract] clicked %s '%s'",
                            target.get("action", ""),
                            target.get("label", "")[:60],
                        )
                    except Exception as e:
                        logger.warning(f"  [deep-extract] reveal click failed: {e}")

            if viewport < max_viewports - 1:
                try:
                    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                    # Issue #259: lazy-load below-the-fold content
                    # typically completes in <500ms; pages with
                    # heavy virtualized lists or image lazy-load
                    # still get the full 1s cap.
                    from .._runner_helpers import adaptive_content_settle
                    adaptive_content_settle(env, min_seconds=0.2, max_seconds=1.0)
                except Exception:
                    break

        data = extractor.extract_multi(screenshots, labels=labels)
        runner._set_scroll_state(
            context="detail_extract_complete",
            url=runner._last_known_url,
            page_downs=max(0, min(len(labels), max_viewports) - 1),
            wheel_downs=0,
            viewport_stage=max(0, min(len(labels), max_viewports) - 1),
            label=f"captured {len(labels)} screenshots, controls_clicked={controls_clicked}",
            flush=True,
        )
        if data and data.is_viable():
            return data, controls_clicked

        # Fallback to legacy single-screenshot extraction if the multi-shot JSON
        # parse fails or somehow loses the core listing identity.
        fallback_shot = screenshots[-1] if screenshots else initial_screenshot
        if fallback_shot is not None:
            fallback = extractor.extract(fallback_shot)
            runner.costs["claude_extract"] += 1
            return fallback, controls_clicked

        return data, controls_clicked
