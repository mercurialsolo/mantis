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
from typing import TYPE_CHECKING, Any

from ...actions import Action, ActionType
from ..checkpoint import StepResult
from ..step_context import StepContext

if TYPE_CHECKING:
    from ..micro_runner import MicroPlanRunner
    from ...plan_decomposer import MicroIntent

logger = logging.getLogger(__name__)


# Loop-extract-dedupe knobs. 8 passes is enough to cover ~70 cards
# on a 1280x720 viewport at YC's grid density without burning Claude
# budget on infinite-scroll feeds. The runner's max_cost still
# bounds spend; this is a sanity ceiling.
_MAX_SCROLL_PASSES = 8


def _pick_dedup_key(schema) -> str:
    """Pick the schema's primary key for cross-pass dedup.

    Heuristic — use the FIRST ``required=True`` field. For most
    listings shapes that's the natural identifier: ``rank`` for HN,
    ``name`` for YC, ``id`` / ``url`` for marketplaces. Falls back to
    the first field name if no required fields are declared (which
    would surface in unit tests as a misconfig).
    """
    fields = getattr(schema, "fields", None) or []
    required = getattr(schema, "required_fields", None) or []
    if required:
        return str(required[0])
    if fields:
        return str(fields[0].get("name") if isinstance(fields[0], dict)
                   else getattr(fields[0], "name", ""))
    return ""


def _filter_new_rows(
    rows, seen_keys: set, dedup_key: str,
) -> list:
    """Return only rows whose ``dedup_key`` value hasn't been seen."""
    if not dedup_key:
        return [dict(r) for r in rows]
    fresh = []
    for r in rows:
        key = str(r.get(dedup_key, "") or "").strip().lower()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        fresh.append(dict(r))
    return fresh


# #880: advance the ACTIVE scroll container by ~0.85 viewport. The plain
# ``window.scrollBy`` the loop used pre-#880 is a no-op on pages whose
# results live in an inner ``overflow:auto`` container — YC's virtualized
# Algolia directory, SPA results panels — so the extract loop saw the same
# viewport every pass and bailed at the first fold (observed: YC W26 run
# captured 5/10, then 6th pass onward ``new=0``). This payload tries the
# window / document scroller first; when that doesn't move, it finds the
# largest scrollable element and advances it, firing a synthetic ``scroll``
# event so virtualized lists re-render. It returns whether anything
# actually advanced so the loop can detect true exhaustion instead of
# burning two empty passes. Reading ``scrollHeight``/``scrollTop`` here is
# action-mechanics (which scroller to drive + did it move), NOT extraction-
# content derivation — same provenance the mechanical scroll handler's
# ``scrollY`` readback already relies on (feedback_cua_cdp_post_action_verify).
_INNER_SCROLLER_JS = (
    "(function(){"
    "  var frac=0.85;"
    "  function wy(){return window.scrollY||document.documentElement.scrollTop||0;}"
    "  var amt=Math.round(window.innerHeight*frac);"
    "  var before=wy();"
    "  window.scrollBy(0,amt);"
    "  if(document.scrollingElement)document.scrollingElement.scrollBy(0,amt);"
    "  if(Math.abs(wy()-before)>=2)return true;"
    "  var best=null,bestArea=0;"
    "  var els=document.querySelectorAll("
    "    'div,main,section,ul,ol,[role=list],[role=feed],[role=main]');"
    "  for(var i=0;i<els.length;i++){"
    "    var e=els[i];"
    "    if(e.scrollHeight-e.clientHeight<=40)continue;"
    "    var cs=getComputedStyle(e);"
    "    if(cs.overflowY!=='auto'&&cs.overflowY!=='scroll')continue;"
    "    var area=e.clientWidth*e.clientHeight;"
    "    if(area>bestArea){bestArea=area;best=e;}"
    "  }"
    "  if(best){"
    "    var t=best.scrollTop;"
    "    best.scrollTop=t+Math.round(best.clientHeight*frac);"
    "    best.dispatchEvent(new Event('scroll',{bubbles:true}));"
    "    return (best.scrollTop-t)>=2;"
    "  }"
    "  return false;"
    "})()"
)


def _cdp_scroll_once(env) -> bool:
    """Advance the active scroll container by ~0.85 viewport via CDP.

    Returns False when the env doesn't expose ``cdp_evaluate`` (test
    envs, replay envs) OR when no scroller actually moved (page at the
    bottom / nothing scrollable) so the caller can stop the loop
    cleanly. Vision-driven scroll is deliberately NOT a fallback — the
    whole point of this path is to skip the Holo3
    ``brain_loop_exhausted`` failure mode.

    #880: drives the dominant inner ``overflow:auto`` container when the
    window scroller is pinned (virtualized lists / SPA panels). See
    :data:`_INNER_SCROLLER_JS`.
    """
    cdp_evaluate = getattr(env, "cdp_evaluate", None)
    if cdp_evaluate is None:
        return False
    try:
        moved = cdp_evaluate(_INNER_SCROLLER_JS)
        # Brief settle so the next screenshot reflects the scroll.
        time.sleep(0.7)
        # Real envs return an explicit bool: False means no scroller
        # advanced (genuinely exhausted) → stop now instead of wasting
        # two empty extract passes. Test/replay envs and older CDP shims
        # return None — preserve the legacy "scroll issued" contract so
        # they keep looping until the empty-pass guard fires.
        return moved is not False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[claude_step] CDP scroll raised: %s — stopping the "
            "extract-loop", exc,
        )
        return False


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
        # Per-step extraction schema (#785 follow-up). When the plan
        # author declares ``step.extract.fields`` inline, swap a
        # transient :class:`ExtractionSchema` onto the extractor for
        # the duration of this step — so the validator enforces *the
        # plan's* required_fields instead of the recipe's (or, with
        # no recipe, ``no_schema_configured``). Restored in
        # ``finally`` so the swap doesn't leak across steps.
        extractor_obj = ctx.extractor
        step_extract = getattr(step, "extract", None) or {}
        transient_schema = None
        if extractor_obj is not None and step_extract.get("fields"):
            from ...extraction import ExtractionSchema

            try:
                transient_schema = ExtractionSchema.from_dict(step_extract)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[claude_step] step.extract malformed; "
                    "falling back to recipe schema: %s",
                    exc,
                )
                transient_schema = None
        original_schema = (
            getattr(extractor_obj, "schema", None)
            if extractor_obj is not None
            else None
        )
        # Diagnostic when the step is an extraction step but no schema
        # is available from any source (no inline `extract`, no
        # recipe-bound `extractor.schema`). Without this, the only
        # signal a plan author gets is the eventual
        # ``no_schema_configured`` rejection in the trace — which is
        # accurate but easy to miss. Surfaces the misconfig at the
        # entry to the step instead. WARNING-level so it survives
        # Modal's filter (`feedback_warning_level_for_modal_observability`).
        if (
            transient_schema is None
            and original_schema is None
            and step.type in ("extract_data", "extract_url")
            and (step.claude_only or step.type == "extract_url")
        ):
            logger.warning(
                "[claude_step] %s step has no extraction schema "
                "(no `extract` block on the step, no recipe-bound "
                "`extractor.schema`). The validator will reject every "
                "extracted row with `no_schema_configured`. Either add "
                "an inline `extract` block to this step (see "
                "docs/client/plans.md#inline-extraction-schema) or "
                "configure a recipe at executor startup.",
                step.type,
            )

        if transient_schema is not None:
            extractor_obj.schema = transient_schema
            # WARNING-level so it survives Modal's log filter (see
            # `feedback_warning_level_for_modal_observability.md`).
            # One line per step that opts in; quiet for legacy plans.
            logger.warning(
                "[claude_step] inline-schema swap: required=%s (step.type=%s)",
                transient_schema.required_fields,
                step.type,
            )

        try:
            # Multi-row branch (#785 follow-up: HN top-N pattern).
            # Triggered when (a) the step type is ``extract_rows``
            # explicitly, or (b) the (transient or recipe) schema has
            # ``max_items > 1`` set on a ``extract_data`` step.
            # Otherwise fall through to the single-row pipeline.
            effective_schema = (
                transient_schema
                if transient_schema is not None
                else original_schema
            )
            # Coerce defensively: existing tests mock the schema with
            # MagicMock and `mock.max_items` returns a Mock that can't
            # be compared with int. ``int(... or 0)`` collapses both
            # missing attr and Mock-shaped values to 0 cleanly.
            try:
                max_items_for_schema = int(
                    getattr(effective_schema, "max_items", 0) or 0
                )
            except (TypeError, ValueError):
                max_items_for_schema = 0
            wants_multi = (
                step.type == "extract_rows"
                or (
                    step.type == "extract_data"
                    and effective_schema is not None
                    and max_items_for_schema > 1
                )
            )
            if wants_multi and extractor_obj is not None and effective_schema is not None:
                return self._execute_rows(step, ctx, effective_schema)
            return self._execute(step, ctx)
        finally:
            if transient_schema is not None and extractor_obj is not None:
                extractor_obj.schema = original_schema

    def _execute_rows(
        self, step: "MicroIntent", ctx: StepContext, schema,
    ) -> StepResult:
        """Multi-row extraction with loop-extract-dedupe (#785 follow-up).

        Strategy: extract → if collected < ``max_items``, CDP-scroll →
        extract again → dedupe by the schema's first required field
        (typically the primary key — ``rank`` / ``name`` / ``id``).
        Loop until collected reaches ``max_items``, OR ``_MAX_SCROLL_
        PASSES`` (8) iterations, OR two consecutive passes yield zero
        new rows (page exhausted).

        Pre-fix the handler did a single extract pass and returned
        whatever Claude saw on the initial viewport — for the YC
        directory grid that's typically 5/10, missing the cards below
        the fold. The loop bridges the gap.

        Each row still lands on the synthesized ``StepResult.
        extracted_rows`` list; the artifact aggregator unpacks them
        into ``leads.csv`` / ``extracted_rows.csv`` /
        ``extracted_rows.json`` automatically.
        """
        runner = self.parent
        env = ctx.env
        extractor = ctx.extractor
        index = int(ctx.state.get("index", 0))

        if not env or not extractor:
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data="extract_rows:no_env_or_extractor",
            )

        max_items = max(int(getattr(schema, "max_items", 0) or 0), 1)
        dedup_key = _pick_dedup_key(schema)
        seen_keys: set[str] = set()
        all_rows: list[dict[str, Any]] = []
        consecutive_empty_passes = 0
        for pass_idx in range(_MAX_SCROLL_PASSES):
            screenshot = env.screenshot()
            rows = extractor.extract_rows(screenshot, max_items)
            runner.costs["claude_extract"] = (
                runner.costs.get("claude_extract", 0) + 1
            )
            new_rows = _filter_new_rows(rows, seen_keys, dedup_key)
            logger.warning(
                "[claude_step] extract_rows pass=%d: %d/%d captured "
                "(new=%d, total=%d/%d)",
                pass_idx + 1, len(rows), max_items,
                len(new_rows), len(all_rows) + len(new_rows), max_items,
            )
            all_rows.extend(new_rows)
            if len(all_rows) >= max_items:
                all_rows = all_rows[:max_items]
                break
            if not new_rows:
                consecutive_empty_passes += 1
                if consecutive_empty_passes >= 2:
                    logger.warning(
                        "[claude_step] extract_rows: 2 consecutive "
                        "empty passes — page exhausted at "
                        "%d/%d", len(all_rows), max_items,
                    )
                    break
            else:
                consecutive_empty_passes = 0
            # CDP scroll for the next pass. Vision-driven scroll
            # would re-introduce the brain_loop_exhausted footgun
            # the loop-extract pattern is designed to avoid.
            if not _cdp_scroll_once(env):
                logger.warning(
                    "[claude_step] extract_rows: CDP scroll "
                    "unavailable — stopping at %d/%d",
                    len(all_rows), max_items,
                )
                break

        if not all_rows:
            logger.warning(
                "[claude_step] extract_rows returned 0 rows after "
                "%d passes (schema=%s, max_items=%d) — page may not "
                "have loaded, or vision couldn't parse the list shape",
                pass_idx + 1,
                getattr(schema, "entity_name", "?"), max_items,
            )
            return StepResult(
                step_index=index, intent=step.intent, success=False,
                data=f"extract_rows:0/{max_items}:no_visible_rows",
            )

        # Legacy single-row consumers see the first row via
        # ``extracted_fields``; the full list is the new
        # ``extracted_rows`` field.
        return StepResult(
            step_index=index, intent=step.intent, success=True,
            data=f"extract_rows:{len(all_rows)}/{max_items}",
            extracted_fields=dict(all_rows[0]),
            extracted_rows=[dict(r) for r in all_rows],
        )

    def _execute(self, step: "MicroIntent", ctx: StepContext) -> StepResult:
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
            # CUA rule (feedback_cua_no_dom_access.md): the runner must
            # be screenshot-grounded only. Reading ``env.current_url``
            # to derive the listing URL would use CDP to *derive* state
            # — banned. Vision extraction is the only authorized path
            # for extract_url, even at the cost of occasional
            # transcription typos (which a downstream post-processor
            # in workflow_runner._build_viable_row normalizes).
            data = extractor.extract(screenshot)
            url = data.url if data else ""

            # #833: pagination URL filter. When the extracted URL is
            # structurally a pagination URL (``?p=2``, ``/page/3``,
            # ``?next=...``), it's almost never what the caller wanted
            # — they asked for a story/item URL. Reject the step so the
            # recovery layer (or the read-only constraint from #831)
            # can react. Plan-author can opt back in via
            # ``extract.allow_pagination_urls: true`` on the step.
            from ...extraction.pagination_filter import (
                is_allowed_pagination,
                is_pagination_url,
            )
            step_extract = getattr(step, "extract", None) or {}
            if url and is_pagination_url(url) and not is_allowed_pagination(step_extract):
                logger.warning(
                    "  [extract_url] PAGINATION_URL — url=%s looks like a "
                    "pagination link; rejecting (opt-in via "
                    "extract.allow_pagination_urls=true)",
                    url[:120],
                )
                return StepResult(
                    step_index=index, intent=step.intent, success=False,
                    data=f"PAGINATION_URL|{url}",
                    failure_class="wrong_target",
                )

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
            #
            # ``contains(url) is True`` rejects MagicMock fakes whose
            # auto-generated .contains returns a truthy Mock — without
            # that the test_listing_card_grounding fakes would trip
            # this gate and silently turn every URL into a DUPLICATE.
            shared = getattr(runner, "_shared_seen_set", None)
            if url and shared is not None and shared.contains(url) is True:
                # #631 follow-up: increment the per-worker hit counter
                # so the orchestrator can aggregate cross-worker dedup
                # savings without grepping container logs. Modal trims
                # stopped-ephemeral container log tails, so log-only
                # signals aren't durable evidence.
                try:
                    runner._shared_seen_hits = int(
                        getattr(runner, "_shared_seen_hits", 0) or 0,
                    ) + 1
                except Exception:  # noqa: BLE001 — never break extract
                    pass
                logger.warning(
                    "  [shared-seen] cross-worker dedup hit: %s "
                    "(step=%d, shared set size=%d, worker hits=%d)",
                    url[:80], index, shared.size(),
                    getattr(runner, "_shared_seen_hits", 0),
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
