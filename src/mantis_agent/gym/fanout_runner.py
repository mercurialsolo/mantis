"""Fan-out runners — parallelize loop bodies across workers (#616, #617).

Two transports share the same plan-rewriter (``rewrite_for_fanout``):

  * :class:`LocalFanoutRunner` — :class:`concurrent.futures.ProcessPoolExecutor`
    spawning N processes on the host. Each worker gets its own Xvfb +
    Chrome via the same ``MicroPlanRunner`` the sequential path uses.
    Gated behind ``MANTIS_FANOUT=local`` (off by default) — issue #616.
  * :class:`ModalFanoutRunner` (in ``deploy/modal/modal_cua_server.py``)
    — replaces the legacy ``_make_page_task`` path. Drives
    ``Function.spawn`` with the same partitioned sub-plans. Issue
    #617.

Inputs:

  - A :class:`~..plan_decomposer.MicroPlan` with at least one
    ``parallelizable_*`` :class:`~..plan_decomposer.LoopGroup`
    (populated by ``PlanDecomposer._classify_loop_groups``).
  - A URL list — typically harvested by the ``collect_urls`` primitive
    (#615) running once on the source plan before fan-out kicks in.
  - ``workers``: how many partitions to split the URL list into.

Output: merged ``list[StepResult]`` from every worker, in worker-id
order. The caller treats this as the run's full output — callers that
need the original loop-step trace pre-rewrite are out of scope for
the first version (issue #618 follow-up).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..observability.augur import open_orchestrator_session
from ..plan_decomposer import LoopGroup, MicroIntent, MicroPlan
from .checkpoint import StepResult

logger = logging.getLogger(__name__)


# ── #627: shared seen-URL set across fan-out workers ────────────────────


class SharedSeenSet(Protocol):
    """Cross-worker seen-URL store.

    Each fan-out worker reads from / writes to the same logical set so
    a listing already extracted by one worker won't be re-clicked +
    re-extracted by a sibling. Per-container dedup
    (``ListingsScanner.is_duplicate``) catches within-container repeats;
    this protocol catches cross-container repeats that featured /
    sponsored listings cause when they appear on multiple result pages.
    """

    def contains(self, url: str) -> bool: ...
    def add(self, url: str) -> None: ...
    def size(self) -> int: ...


class NullSharedSeenSet:
    """Default implementation — every workload starts with this until a
    real backend (Modal Dict) is wired. ``contains`` always returns
    False, ``add`` is a no-op, ``size`` is always 0. Lets the dedup
    plumbing exist on every code path without forcing every test
    fixture to mock a real backend.
    """

    def contains(self, url: str) -> bool:  # noqa: ARG002
        return False

    def add(self, url: str) -> None:
        pass

    def size(self) -> int:
        return 0


class _InMemorySharedSeenSet:
    """In-process backend — for tests and the local fan-out runner.
    Backed by a plain ``set[str]``; no IPC, no thread safety.
    """

    def __init__(self) -> None:
        self._urls: set[str] = set()

    def contains(self, url: str) -> bool:
        return _normalize_listing_url(url) in self._urls

    def add(self, url: str) -> None:
        normalized = _normalize_listing_url(url)
        if normalized:
            self._urls.add(normalized)

    def size(self) -> int:
        return len(self._urls)


class _ModalDictSharedSeenSet:
    """Modal Dict backend — used in production fan-out. Wraps
    :class:`modal.Dict.from_name` keyed by a per-run dict name so all
    spawned workers attach to the same logical set.

    URL normalization matches :func:`_normalize_listing_url` so keys
    line up with the orchestrator's :func:`dedup_leads_by_url` pass.

    Modal Dicts are atomic on write but reads/writes are network calls;
    cost per check is ~1-2 ms. Worth it because the alternative is a
    full Claude extract (~$0.20, 30-60s) on a duplicate.
    """

    def __init__(self, dict_name: str) -> None:
        import modal
        self._name = dict_name
        self._dict = modal.Dict.from_name(dict_name, create_if_missing=True)

    def contains(self, url: str) -> bool:
        normalized = _normalize_listing_url(url)
        if not normalized:
            return False
        try:
            return normalized in self._dict
        except Exception as exc:  # noqa: BLE001 — never break a run
            logger.warning("[shared-seen] contains() raised: %s", exc)
            return False

    def add(self, url: str) -> None:
        normalized = _normalize_listing_url(url)
        if not normalized:
            return
        try:
            # ``put`` is atomic; value isn't read anywhere — using a
            # tiny sentinel (1) keeps wire size minimal vs storing the
            # URL again as both key and value.
            self._dict.put(normalized, 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[shared-seen] add() raised: %s", exc)

    def size(self) -> int:
        # modal.Dict doesn't implement __len__; use the SDK's .len()
        # method (returns int) and fall back to counting keys() if even
        # that's missing on older SDKs.
        try:
            if hasattr(self._dict, "len"):
                return int(self._dict.len())
            return sum(1 for _ in self._dict.keys())
        except Exception as exc:  # noqa: BLE001
            logger.debug("[shared-seen] size() raised: %s", exc)
            return 0


# ── #631: augur branch_context per fan-out worker ──────────────────────


def build_fanout_branch_context(suite_dict: dict) -> dict | None:
    """Construct the ``branch_context`` payload for one fan-out worker.

    Reads orchestrator-injected metadata from the suite:

      * ``_fanout_parent_run_id``  — shared across all workers
      * ``_fanout_phase``          — ``"phase1_collect"`` / ``"phase2_extract"``
                                     / ``"pagination_partition"`` (or empty
                                     for the legacy pagination path without
                                     a phase tag).
      * ``_fanout_branch_id``      — per-worker id (orchestrator sets this).
      * ``_fanout_url_count``      — # URLs this worker is assigned (Phase 2).

    Returns the dict that gets forwarded to ``AugurAdapter(branch_context=)``,
    which in turn passes it to ``DebugSession``. Per augur-sdk 0.2.1, mantis
    fan-out uses ``mutated_axis="action"`` (different URL per worker = action
    mutation); the SDK's auto-mode resolves to ``sandbox`` (no replay
    prefix; execute fresh).

    Returns ``None`` for non-fanout / single-worker runs (no
    ``_fanout_parent_run_id`` in the suite) so AugurAdapter opens the
    session without a branch label, preserving today's shape.
    """
    parent_run_id = suite_dict.get("_fanout_parent_run_id", "")
    if not parent_run_id:
        return None
    branch_id = (
        suite_dict.get("_fanout_branch_id")
        or f"{parent_run_id}:{suite_dict.get('session_name', 'worker')}"
    )
    phase = suite_dict.get("_fanout_phase", "")
    mutation: dict[str, Any] = {}
    if phase:
        mutation["phase"] = phase
    url_count = suite_dict.get("_fanout_url_count")
    if url_count is not None:
        mutation["url_count"] = int(url_count)
    return {
        "parent_run_id": str(parent_run_id),
        "branch_point_step_index": 0,
        # ``action`` axis = different URL / partition target per worker.
        # SDK auto-resolves mode to ``sandbox`` for this axis (no prefix
        # replay; each worker executes fresh against its assigned URL).
        "mutated_axis": "action",
        "mutation": mutation,
        "branch_id": str(branch_id),
    }


def build_shared_seen_set(suite_dict: dict) -> SharedSeenSet:
    """Construct the right backend for a worker based on suite metadata.

    Resolution order:

      * ``_fanout_seen_dict_name`` in suite + ``modal`` importable →
        :class:`_ModalDictSharedSeenSet` keyed by that name.
      * Otherwise (no fan-out, no Modal, tests) → :class:`NullSharedSeenSet`.

    The Modal orchestrator generates the dict name once per run (UUID
    or run_id) and writes it into every spawned worker's task_suite
    BEFORE spawning. Workers re-attach via ``from_name`` so all see
    the same logical set.
    """
    dict_name = suite_dict.get("_fanout_seen_dict_name", "")
    if not dict_name:
        return NullSharedSeenSet()
    try:
        return _ModalDictSharedSeenSet(str(dict_name))
    except Exception as exc:  # noqa: BLE001 — Modal not importable / not on Modal
        logger.warning(
            "[shared-seen] modal.Dict.from_name(%s) failed (%s) — "
            "falling back to NullSharedSeenSet",
            dict_name, exc,
        )
        return NullSharedSeenSet()


def fanout_enabled() -> bool:
    """#616 opt-in gate. ``MANTIS_FANOUT=local`` enables local fan-out;
    anything else (including unset) leaves the sequential path
    unchanged. Read once per call site so per-request overrides via
    ``os.environ`` take effect mid-process (tests use this)."""
    raw = os.environ.get("MANTIS_FANOUT", "").strip().lower()
    return raw == "local"


# ── Plan rewriter ──────────────────────────────────────────────────────


def rewrite_for_fanout(plan: MicroPlan, group: LoopGroup) -> MicroPlan:
    """Produce a per-worker sub-plan from a parallelizable loop group.

    The loop body's first step (``click`` on a listing card) and its
    URL-probe (``extract_url``) are replaced by a single ``navigate``
    step whose URL is filled in per-worker at dispatch time. The
    ``navigate_back`` at the body's tail is dropped — workers always
    operate on a fresh tab with its own navigation history.

    Steps OUTSIDE the group's body are preserved verbatim, including:

      - The pre-loop setup chain (navigate to results page, filter,
        collect_urls, …).
      - Any post-loop steps (pagination loop, completion report).

    The original ``loop`` step itself is removed — each worker's
    sub-plan is one-shot. The fan-out runner orchestrates iteration
    via partition slicing instead.

    Returns a new :class:`MicroPlan` (does not mutate ``plan``). The
    returned plan's first body step is a ``navigate`` step whose
    ``params["url"]`` is the empty string sentinel; the fan-out runner
    fills it in per worker before dispatching.

    Refuses to rewrite (returns the original plan unchanged) if the
    group's shape is ``sequential`` — the classifier (#614) ruled it
    unsafe.
    """
    if group.shape == "sequential":
        return plan

    body_start, body_end = group.body_range
    new_steps: list[MicroIntent] = []

    # Preserve everything before the body.
    new_steps.extend(_clone_steps(plan.steps[:body_start]))

    # Body rewrite: drop the click+extract_url anchor, drop the loop's
    # navigate_back tail. Replace with a navigate-to-url step. Keep
    # scroll + extract_data as the per-worker body.
    body = plan.steps[body_start:body_end]
    new_body: list[MicroIntent] = [
        MicroIntent(
            intent="Navigate to this worker's listing URL",
            type="navigate",
            section="extraction",
            budget=3,
            # URL is filled in per worker by the fan-out runner.
            params={"url": "", "wait_after_load_seconds": 6},
        )
    ]
    for step in body:
        # Skip the body's entry click — replaced by navigate above.
        if step.type == "click" and step.section == "extraction":
            continue
        # Skip extract_url — the URL came from collect_urls upstream.
        if step.type == "extract_url":
            continue
        # Skip navigate_back — each worker has its own tab.
        if step.type == "navigate_back":
            continue
        new_body.append(_clone_step(step))
    new_steps.extend(new_body)

    # Drop the loop step itself; everything AFTER it is preserved.
    new_steps.extend(_clone_steps(plan.steps[group.loop_step_idx + 1:]))

    rewritten = MicroPlan(
        source_plan=plan.source_plan,
        domain=plan.domain,
    )
    rewritten.steps = new_steps
    rewritten.shapes = list(plan.shapes)
    # Don't carry loop_groups onto the rewritten plan — the loop is gone.
    rewritten.loop_groups = []
    return rewritten


def _clone_steps(steps: list[MicroIntent]) -> list[MicroIntent]:
    return [_clone_step(s) for s in steps]


def _clone_step(step: MicroIntent) -> MicroIntent:
    return MicroIntent(
        intent=step.intent,
        type=step.type,
        verify=step.verify,
        budget=step.budget,
        reverse=step.reverse,
        grounding=step.grounding,
        section=step.section,
        required=step.required,
        gate=step.gate,
        claude_only=step.claude_only,
        loop_target=step.loop_target,
        loop_count=step.loop_count,
        params=dict(step.params or {}),
        hints=dict(step.hints or {}),
    )


# ── URL partitioning ───────────────────────────────────────────────────


def partition_urls(urls: list[str], workers: int) -> list[list[str]]:
    """Round-robin slice of ``urls`` into ``workers`` chunks.

    Round-robin (vs contiguous) keeps each worker's chunk diverse in
    position-on-page — useful when results pages render slower listings
    near the bottom, so contiguous slicing would concentrate slow work
    on one worker.

    Returns ``workers`` chunks even when ``len(urls) < workers``
    (extras are empty). Callers should filter out the empty chunks
    before spawning workers; passing them through is harmless but
    wastes a process slot.
    """
    if workers < 1:
        workers = 1
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, url in enumerate(urls):
        chunks[i % workers].append(url)
    return chunks


def build_worker_subplan(
    rewritten: MicroPlan, urls: list[str],
) -> list[MicroPlan]:
    """Produce one sub-plan per URL by cloning ``rewritten`` and filling
    in the per-worker navigate URL.

    The fan-out runner used to take a single plan and loop URLs inside
    one runner instance, but that breaks the runner's "one plan, one
    state" invariant (extraction cache, scanner state, dedup set all
    accumulate across iterations in a way that interacts badly with
    parallel mutation). One sub-plan per URL keeps the runner state
    clean per execution and gives the executor a natural unit of work.
    """
    plans: list[MicroPlan] = []
    for url in urls:
        sub = MicroPlan(
            source_plan=rewritten.source_plan,
            domain=rewritten.domain,
        )
        sub.steps = _clone_steps(rewritten.steps)
        # Find the first navigate step with empty url — the rewriter's
        # sentinel — and fill it. This is more robust than indexing by
        # position because future rewriters may move the navigate
        # around (e.g. wrapping it in a setup chain).
        for step in sub.steps:
            if step.type == "navigate" and not (step.params or {}).get("url"):
                step.params = {**(step.params or {}), "url": url}
                step.intent = f"Navigate to {url}"
                break
        plans.append(sub)
    return plans


# ── Local transport ────────────────────────────────────────────────────


@dataclass
class LocalFanoutResult:
    """Merged output from one :meth:`LocalFanoutRunner.run` invocation."""

    results: list[StepResult] = field(default_factory=list)
    per_worker_results: list[list[StepResult]] = field(default_factory=list)
    workers: int = 0
    urls_dispatched: int = 0
    failures: int = 0


class LocalFanoutRunner:
    """Spawn worker processes via :class:`ProcessPoolExecutor` and merge
    their :class:`StepResult` outputs.

    The factory callable ``runner_factory`` builds a fresh
    :class:`MicroPlanRunner` (or a stub for tests) per worker. The
    runner factory is passed via callable injection rather than
    imported here directly so unit tests can swap a stub without
    spinning up Xvfb + Chrome.

    Per-worker isolation comes from process boundaries: each worker
    gets its own copy of every module-level state (extraction cache,
    seen_urls scanner, augur adapter). The merge layer (#618 follow-up)
    can de-dup across workers if needed.
    """

    def __init__(
        self, runner_factory: Callable[..., Any], workers: int = 4,
    ) -> None:
        self.runner_factory = runner_factory
        self.workers = max(1, workers)

    def run(
        self, plan: MicroPlan, urls: list[str], *, group: LoopGroup,
    ) -> LocalFanoutResult:
        rewritten = rewrite_for_fanout(plan, group)
        sub_plans = build_worker_subplan(rewritten, urls)
        active = min(self.workers, len(sub_plans)) or 1
        logger.warning(
            "  [fanout/local] dispatching %d URL(s) across %d worker(s)",
            len(urls), active,
        )

        per_worker: list[list[StepResult]] = []
        failures = 0
        # ``max_workers`` clamps to one even when sub_plans is empty so
        # the executor doesn't raise on construction.
        with ProcessPoolExecutor(max_workers=active) as pool:
            futures = [
                pool.submit(_run_one_subplan, self.runner_factory, sp)
                for sp in sub_plans
            ]
            for fut in futures:
                try:
                    per_worker.append(fut.result())
                except Exception as exc:  # noqa: BLE001 — worker crashed
                    failures += 1
                    logger.warning(
                        "  [fanout/local] worker raised: %s", exc,
                    )
                    per_worker.append([])

        merged: list[StepResult] = []
        for chunk in per_worker:
            merged.extend(chunk)
        return LocalFanoutResult(
            results=merged,
            per_worker_results=per_worker,
            workers=active,
            urls_dispatched=len(urls),
            failures=failures,
        )


def _run_one_subplan(
    runner_factory: Callable[..., Any], plan: MicroPlan,
) -> list[StepResult]:
    """Worker-process entry point. Must be importable at module level
    (ProcessPoolExecutor pickles the function reference)."""
    runner = runner_factory()
    return runner.run(plan)


# ── #623: orchestrator-side worker result reader ──────────────────────


def _normalize_listing_url(url: str) -> str:
    """Canonical form for cross-partition lead-dedup keys (#621).

    Rules:
      - Strip surrounding whitespace.
      - Lowercase (most marketplaces use case-insensitive paths).
      - Drop a trailing slash (BoatTrader emits both ``/boat/<slug>``
        and ``/boat/<slug>/`` interchangeably across pages).

    No URL-parse / query-string dance — that's per-domain policy. The
    listing URL primary key is opaque enough that simple normalization
    catches the common duplication signals (path-case drift, trailing
    slash) without overreaching.
    """
    if not url:
        return ""
    s = str(url).strip().lower()
    if s.endswith("/"):
        s = s[:-1]
    return s


def _lead_url(lead: Any) -> str:
    """Pull the listing URL from a lead row, regardless of its shape.

    Two formats coexist in the codebase — the verification run for
    #621 surfaced this when the dict-only first pass treated all 87
    string-shaped leads as non-dict and dropped them:

      * **String** (the actual production shape from
        ``build_micro_result``): ``"VIABLE | Year: ... | URL: ..."``.
        Parsed with ``ListingDedup.lead_key`` — the same helper
        per-container dedup uses, so cross-partition keys match
        per-container keys.
      * **Dict** (defensive — host integrations / future structured
        paths): ``{"listing_url": "..."}`` or ``{"url": "..."}``.

    Returns the empty string when neither shape yields a URL. The
    dedup pass treats such leads as "no key" and passes them through
    unchanged.
    """
    if isinstance(lead, str):
        from .listing_dedup import ListingDedup
        # lead_key falls back to the row's first 100 chars when no
        # URL regex match — that fallback isn't a URL and would
        # collide all no-URL rows under one bucket. Gate on the
        # ``URL:`` token explicitly.
        if "URL:" not in lead:
            return ""
        return ListingDedup.lead_key(lead)
    if isinstance(lead, dict):
        return str(lead.get("listing_url") or lead.get("url") or "")
    return ""


def dedup_leads_by_url(
    per_partition_leads: list[list[Any]],
) -> tuple[list[Any], int, int]:
    """Cross-partition lead-list merge with URL dedup (#621).

    Each worker returns its own ``leads`` list — concatenated naively
    that gives a ``raw`` total, but featured / sponsored listings can
    repeat across pages and pagination drift mid-run can shift the
    same listing onto two adjacent pages. The dedup pass collapses
    duplicates by normalized listing URL, preserving first-seen
    partition order.

    Returns ``(deduped, raw_count, deduped_count)``.

    Lead rows are heterogeneous (str from ``build_micro_result``;
    dict from host paths). URL extraction is delegated to
    :func:`_lead_url` which handles both shapes; rows that yield no
    URL pass through unchanged.
    """
    seen: set[str] = set()
    deduped: list[Any] = []
    raw = 0
    for chunk in per_partition_leads:
        for lead in chunk:
            raw += 1
            url = _normalize_listing_url(_lead_url(lead))
            if not url:
                deduped.append(lead)
                continue
            if url in seen:
                continue
            seen.add(url)
            deduped.append(lead)
    return deduped, raw, len(deduped)


def read_partition_result(result: dict | None) -> dict:
    """Extract the lead-count fields from a worker's return dict.

    The Modal worker's return shape is whatever ``build_micro_result``
    in ``server_utils`` produces — keys ``viable`` (count),
    ``leads_with_phone`` (count), and ``leads`` (the list).

    The original #617 orchestrator read ``leads_count`` / ``score``
    instead — keys that no MicroPlanRunner-backed executor returns.
    Both lookups silently defaulted to 0, so ``Total leads`` always
    printed 0 even when every partition successfully extracted leads.

    Returns a normalized dict with:

      * ``viable``: total leads in the partition (int, defaults to 0).
      * ``with_phone``: subset that carry a phone (int, defaults to 0).
      * ``leads``: the raw lead rows list (used by #621's cross-partition
        dedup pass; absent in worker shapes that elide the rows for
        bandwidth — defaults to an empty list).

    Tolerant of ``None`` input (when ``handle.get()`` itself failed
    upstream) — returns the zero shape rather than raising.
    """
    if not isinstance(result, dict):
        return {
            "viable": 0, "with_phone": 0, "leads": [],
            "collected_urls": [], "shared_seen_hits": 0,
        }
    viable = int(result.get("viable", 0) or 0)
    with_phone = int(result.get("leads_with_phone", 0) or 0)
    leads_raw = result.get("leads") or []
    leads = list(leads_raw) if isinstance(leads_raw, list) else []
    # #628: Phase-1 workers return harvested URLs via ``collected_urls``
    # (added to build_micro_result). Phase-2 / sequential workers carry
    # an empty list. The orchestrator reads this after Phase-1 finishes
    # to drive Phase-2 partitioning.
    urls_raw = result.get("collected_urls") or []
    collected_urls = (
        [str(u) for u in urls_raw if u] if isinstance(urls_raw, list) else []
    )
    # #631 follow-up: per-worker cross-worker dedup hit count for the
    # orchestrator's aggregate metric. Always 0 for non-fanout runs.
    shared_seen_hits = int(result.get("shared_seen_hits", 0) or 0)
    return {
        "viable": viable, "with_phone": with_phone,
        "leads": leads, "collected_urls": collected_urls,
        "shared_seen_hits": shared_seen_hits,
    }


# ── Modal transport — partition prep (#617) ────────────────────────────


DEFAULT_PAGINATION_URL_TEMPLATE = "{base}/page-{n}/"
"""Per-page URL synthesis template. The boattrader convention is
``{base}/page-{n}/`` — Zillow uses ``{base}?page={n}``, Facebook uses
cursor-based pagination. Plans override via ``paginate.params['url_template']``
in their decomposer output.

The template is consumed by :func:`partition_urls_for_pagination`,
which substitutes ``{base}`` (the setup-navigate URL stripped of any
trailing slash) and ``{n}`` (the 1-indexed page number).
"""


def partition_urls_for_pagination(
    base_url: str, max_pages: int,
    *, template: str = DEFAULT_PAGINATION_URL_TEMPLATE,
) -> list[str]:
    """Synthesize one URL per page partition from a setup base URL.

    Page 1 is the bare ``base_url`` (no template substitution) —
    BoatTrader's first page is ``/by-owner/radius-25/``, not
    ``/by-owner/radius-25/page-1/``. Pages 2..N use the template.

    Returns a list of length ``max(1, max_pages)``. ``max_pages == 0``
    is clamped to ``1`` so a misconfigured loop_count doesn't return
    an empty partition list (which would silently produce zero
    workers).
    """
    n = max(1, max_pages)
    base = base_url.rstrip("/")
    urls = [base_url]
    for page in range(2, n + 1):
        urls.append(template.format(base=base, n=page))
    return urls


# ── #628: Phase-1/Phase-2 fan-out for parallelizable_url_collect ──────


def _build_plan_from_suite(suite_dict: dict) -> MicroPlan:
    """Materialize a :class:`MicroPlan` from a task_suite ``_micro_plan``
    list, populating ``loop_groups`` either from the cached
    ``_loop_groups`` payload or by recomputing the classifier."""
    from ..plan_decomposer import PlanDecomposer

    steps_raw = suite_dict.get("_micro_plan") or []
    plan = MicroPlan(domain=str(suite_dict.get("session_name", "")))
    for s in steps_raw:
        plan.steps.append(PlanDecomposer._build_intent(s))

    groups_raw = suite_dict.get("_loop_groups")
    if groups_raw:
        plan.loop_groups = [
            LoopGroup(
                loop_step_idx=int(g.get("loop_step_idx", -1)),
                body_range=tuple(g.get("body_range", (0, 0))),
                shape=str(g.get("shape", "sequential")),
            )
            for g in groups_raw
            if isinstance(g, dict)
        ]
    else:
        PlanDecomposer._classify_loop_groups(plan)
    # #629: plan-level pagination URL template carried alongside the
    # micro_plan / loop_groups payload. Empty when the decomposer
    # didn't set one (which is most plans today).
    plan.pagination_url_template = str(
        suite_dict.get("_pagination_url_template", "") or ""
    )
    return plan


def find_url_collect_group(suite_dict: dict) -> LoopGroup | None:
    """Return the first ``parallelizable_url_collect`` group on the plan
    or ``None`` if the plan has none.

    Used by the Modal orchestrator to decide between the Phase-1/Phase-2
    path (#628) and the existing pagination-only path (#617). Returns
    None when the plan has no ``_micro_plan`` or when the classifier
    found no extraction-loop body to fan out.
    """
    if not suite_dict.get("_micro_plan"):
        return None
    plan = _build_plan_from_suite(suite_dict)
    for g in plan.loop_groups:
        if g.shape == "parallelizable_url_collect":
            return g
    return None


def resolve_phase1_max_pages(suite_dict: dict) -> tuple[int, str]:
    """Decide how many pages Phase-1 should walk + the URL template.

    Resolution order for ``max_pages``:

      1. Explicit ``_fanout_phase1_max_pages`` override on the suite
         (operator/test escape hatch).
      2. ``loop_count`` of the plan's ``parallelizable_pagination``
         group, clamped to :data:`DEFAULT_PHASE1_MAX_PAGES` (so a
         50-page pagination plan doesn't burn $5 in Phase-1).
      3. Default of 1 (single page — backward compatible with the
         original #628 Phase-1 shape).

    Resolution order for ``url_template``:

      1. Plan-level ``_pagination_url_template`` (set by #629).
      2. ``params['url_template']`` on the ``paginate`` step inside
         the pagination group (back-compat).
      3. :data:`DEFAULT_PAGINATION_URL_TEMPLATE`.

    Returns ``(max_pages, url_template)``. Callers pass both into
    :func:`prepare_phase1_suite` as keyword args.
    """
    if not suite_dict.get("_micro_plan"):
        return 1, ""
    plan = _build_plan_from_suite(suite_dict)
    explicit = suite_dict.get("_fanout_phase1_max_pages")
    pagination_group = next(
        (g for g in plan.loop_groups if g.shape == "parallelizable_pagination"),
        None,
    )

    template = DEFAULT_PAGINATION_URL_TEMPLATE
    plan_tpl = (
        getattr(plan, "pagination_url_template", "") or ""
    ).strip()
    if plan_tpl:
        template = plan_tpl
    elif pagination_group is not None:
        body_start, body_end = pagination_group.body_range
        paginate_step = next(
            (s for s in plan.steps[body_start:body_end] if s.type == "paginate"),
            None,
        )
        if paginate_step is not None:
            custom = (paginate_step.params or {}).get("url_template")
            if custom:
                template = str(custom)

    if explicit is not None:
        try:
            max_pages = max(1, int(explicit))
        except (TypeError, ValueError):
            max_pages = 1
    elif pagination_group is not None:
        # Hand-authored plans frequently omit ``loop_count`` on the
        # pagination loop (the per-page partition path defaults this to
        # ``5``). Treat absent loop_count as "walk up to
        # DEFAULT_PHASE1_MAX_PAGES" — the existence of a pagination group
        # is enough signal that Phase-1 should harvest beyond page 1.
        loop_count = int(
            plan.steps[pagination_group.loop_step_idx].loop_count
            or DEFAULT_PHASE1_MAX_PAGES
        )
        max_pages = max(1, min(loop_count, DEFAULT_PHASE1_MAX_PAGES))
    else:
        max_pages = 1
    return max_pages, template


def _step_dict_clone(step_dict: dict) -> dict:
    """Shallow-clone a step dict, deep-copying ``params`` + ``hints`` so
    sub-suite mutations don't leak into the source plan."""
    cloned = dict(step_dict)
    cloned["params"] = dict(step_dict.get("params") or {})
    cloned["hints"] = dict(step_dict.get("hints") or {})
    return cloned


#: Default cap on Phase-1 multi-page harvest. Each page costs one
#: ``collect_urls`` invocation (~$0.05-0.10 with multi-viewport scan)
#: plus one navigate (~$0.01) — capping at 3 keeps Phase-1 cost
#: bounded at ~$0.30 in the worst case. Plans that need broader
#: coverage can lift via ``_fanout_phase1_max_pages`` in the suite,
#: or via ``paginate.params['url_template']`` + the orchestrator's
#: pagination loop_count.
DEFAULT_PHASE1_MAX_PAGES = 3


def partition_pages(max_pages: int, n_workers: int) -> list[list[int]]:
    """Round-robin slice the page set ``[1..max_pages]`` across N workers.

    Returns a list of length ``min(n_workers, max_pages)`` so empty
    chunks are never emitted (an empty chunk would mean spawning an
    idle Phase-1 worker). Round-robin (vs contiguous) keeps each
    worker's chunk diverse in seller-card-density distribution — many
    real domains render featured listings up front and the long tail
    at the bottom, so contiguous slicing would concentrate slow scans
    on the last worker.

    Worker w_i (0-indexed) walks pages ``[i+1, i+1+n_workers, i+1+2*n_workers, ...]``
    truncated at ``max_pages``. ``n_workers <= 1`` falls back to a
    single chunk of all pages (the current serial behaviour).
    """
    n = max(1, int(max_pages))
    w = max(1, int(n_workers))
    if w >= n:
        w = n  # don't over-shard
    chunks: list[list[int]] = [[] for _ in range(w)]
    for page in range(1, n + 1):
        # 1-indexed page → 0-indexed slot via (page-1) % w.
        chunks[(page - 1) % w].append(page)
    return chunks


def _phase1_collect_step() -> dict:
    """Stamp the canonical ``collect_urls`` step shape used by Phase-1
    workers. Shared between ``prepare_phase1_suite`` (single-worker) and
    ``prepare_phase1_partitions`` (multi-worker)."""
    return {
        "intent": "Collect every listing URL visible on this results page",
        "type": "collect_urls",
        "section": "extraction",
        "claude_only": True,
        "budget": 0,
        # required=False (not True) so the runner's step-recovery
        # policy doesn't trigger agentic_recovery + add_hint retries
        # on collect_urls failure. The orchestrator already handles
        # empty collect_urls by falling through to the pagination
        # path — burning ~$0.20 on 3 doomed Claude scan retries +
        # critic-row-link recovery attempts is wasted budget.
        "required": False,
        "params": {},
        "hints": {},
        "gate": False,
        "loop_target": -1,
        "loop_count": 0,
        "grounding": False,
        "verify": "",
        "reverse": "",
    }


def _phase1_navigate_step(url: str) -> dict:
    """Stamp the canonical ``navigate`` step shape used between page
    transitions inside a Phase-1 worker."""
    return {
        "intent": f"Navigate to {url}",
        "type": "navigate",
        "section": "extraction",
        "budget": 3,
        "required": False,
        "params": {"url": url, "wait_after_load_seconds": 6},
        "hints": {},
        "claude_only": False,
        "gate": False,
        "loop_target": -1,
        "loop_count": 0,
        "grounding": False,
        "verify": "",
        "reverse": "",
    }


def _setup_steps_base_url(setup_steps: list[dict]) -> str:
    """Find the first ``navigate`` step's URL in ``setup_steps`` — the
    base URL that all subsequent per-page navigations key off."""
    for s in setup_steps:
        if s.get("type") == "navigate":
            url = (s.get("params") or {}).get("url", "") or ""
            if url:
                return url
    return ""


def _phase1_plan_for_pages(
    setup_steps: list[dict],
    pages: list[int],
    pagination_url_template: str,
) -> list[dict]:
    """Build the Phase-1 plan-steps list for a worker assigned ``pages``.

    Each worker shares the setup chain (cookie banners, login, base
    navigate). Then:
      * If page 1 is in the worker's set: collect_urls runs on the
        base URL the setup just navigated to.
      * For every page p > 1 in the worker's set:
        navigate(page-p URL) + collect_urls.
      * If page 1 is NOT in the worker's set: the worker still runs
        setup (navigating to the base URL), then jumps straight to its
        first assigned page>1. The base URL's listings are not
        harvested by this worker — a different worker handles them.

    Returns the flat list of step dicts the worker will execute.
    Falls back silently to a single-collect (page 1 only) when the
    template is malformed or the base URL is missing — better fewer
    URLs than a crash.
    """
    steps: list[dict] = list(setup_steps)
    sorted_pages = sorted(set(int(p) for p in pages))
    if 1 in sorted_pages:
        steps.append(_phase1_collect_step())
    if any(p > 1 for p in sorted_pages):
        base_url = _setup_steps_base_url(setup_steps)
        template = pagination_url_template or DEFAULT_PAGINATION_URL_TEMPLATE
        if (
            base_url
            and "{base}" in template
            and "{n}" in template
        ):
            base = base_url.rstrip("/")
            for p in sorted_pages:
                if p == 1:
                    continue
                page_url = template.format(base=base, n=p)
                steps.append(_phase1_navigate_step(page_url))
                steps.append(_phase1_collect_step())
    return steps


def prepare_phase1_suite(
    suite_dict: dict, group: LoopGroup,
    *,
    max_pages: int = 1,
    pagination_url_template: str = "",
    pages: list[int] | None = None,
) -> dict:
    """Build a one-container Phase-1 sub-suite that harvests listing URLs.

    Phase-1 runs the plan's setup chain (everything BEFORE the
    extraction loop body), then walks ``max_pages`` paginated pages
    accumulating URLs via the ``collect_urls`` handler. The worker
    returns its harvested URL list to the orchestrator via the result
    envelope's ``collected_urls`` field (added by
    :func:`build_micro_result`).

    On ``max_pages=1`` (default) the sub-suite is:
        setup_steps + collect_urls
    On ``max_pages=N > 1`` the sub-suite is (#638 axis 2):
        setup_steps + collect_urls
                    + navigate(page-2) + collect_urls
                    + navigate(page-3) + collect_urls
                    + ...
    The collect_urls handler appends-with-dedup to
    ``runner._collected_urls``, so URLs from all pages accumulate
    into a single list that crosses to the orchestrator on return.

    Phase-1 is a serial bottleneck — one Modal container, no fan-out
    inside. Each page costs ~$0.05-0.10 (multi-viewport scan) + ~$0.01
    (navigate); ``max_pages=3`` caps the worst-case Phase-1 cost at
    ~$0.30.

    ``pagination_url_template`` defaults to the same
    :data:`DEFAULT_PAGINATION_URL_TEMPLATE` the per-page fan-out uses
    so the URL synthesis is consistent across both paths. Set to a
    custom template (e.g. ``"{base}?page={n}"``) to override.
    """
    body_start, _body_end = group.body_range
    loop_idx = group.loop_step_idx
    steps_raw = suite_dict.get("_micro_plan") or []
    setup_steps = [
        _step_dict_clone(s) for s in steps_raw[:body_start]
    ]

    # #644: explicit ``pages`` overrides ``max_pages``. The legacy
    # path (no ``pages`` arg) walks pages 1..max_pages sequentially,
    # which matches the pre-#644 behaviour byte-for-byte.
    if pages is None:
        pages = list(range(1, max(1, int(max_pages)) + 1))
    plan_steps = _phase1_plan_for_pages(
        setup_steps, pages, pagination_url_template,
    )

    sub_suite = dict(suite_dict)
    sub_suite["_micro_plan"] = plan_steps
    sub_suite["session_name"] = (
        f"{suite_dict.get('session_name', 'fanout')}_phase1"
    )
    # Phase-1 has no loop — drop the cached group dump so the child
    # container doesn't accidentally route through any fanout path.
    sub_suite.pop("_loop_groups", None)
    # Surface that this is the collection phase so the worker can
    # apply phase-specific behaviour (e.g. tightening max_cost).
    sub_suite["_fanout_phase"] = "phase1_collect"
    # Loop_idx is unused but documenting the source body span helps
    # operators when reading the suite payload in dispatch logs.
    sub_suite["_fanout_source_loop_step"] = loop_idx
    # Record the worker's assigned page set (and total page count) so
    # the operator can see (in the dispatch log) what each worker was
    # asked to walk. Legacy callers passing only ``max_pages`` see the
    # same shape they always did (``_fanout_phase1_pages == max_pages``)
    # because ``pages = [1..max_pages]`` in that case.
    sub_suite["_fanout_phase1_pages"] = len(pages)
    sub_suite["_fanout_phase1_page_set"] = list(pages)
    return sub_suite


def prepare_phase1_partitions(
    suite_dict: dict, group: LoopGroup,
    *,
    n_workers: int,
    max_pages: int,
    pagination_url_template: str = "",
) -> list[dict]:
    """#644 — build M Phase-1 sub-suites each scanning a page slice.

    The orchestrator spawns one Modal worker per returned sub-suite in
    parallel, then merges + dedups the per-worker ``collected_urls``
    on its side (orchestrator-side merge — option 2 in the issue;
    see also #627's ``shared_seen_set`` for a future early-skip
    optimization).

    Round-robin page assignment (via :func:`partition_pages`) keeps
    each worker's chunk diverse in seller-density distribution.
    Workers whose chunk is empty are dropped — passing
    ``n_workers > max_pages`` returns ``max_pages`` sub-suites instead
    of ``n_workers``.

    When ``n_workers <= 1`` this returns a single sub-suite equivalent
    to :func:`prepare_phase1_suite` — the orchestrator's serial path
    is preserved when fan-out is disabled.

    Each sub-suite has the standard Phase-1 envelope plus a
    per-worker ``_fanout_phase1_worker_index`` (0-indexed) that the
    operator can read in dispatch logs.
    """
    chunks = partition_pages(max_pages, n_workers)
    sub_suites: list[dict] = []
    for i, page_chunk in enumerate(chunks):
        if not page_chunk:
            continue
        sub = prepare_phase1_suite(
            suite_dict, group,
            pagination_url_template=pagination_url_template,
            pages=page_chunk,
        )
        # Per-worker session name + index — tells the operator which
        # worker owns which page slice in the dispatch log + Augur Runs.
        sub["session_name"] = (
            f"{suite_dict.get('session_name', 'fanout')}_phase1_w{i + 1}"
        )
        sub["_fanout_phase1_worker_index"] = i
        sub["_fanout_phase1_worker_count"] = len(chunks)
        sub_suites.append(sub)
    return sub_suites


def dedup_urls_across_workers(per_worker_urls: list[list[str]]) -> list[str]:
    """Concatenate + dedup the per-worker harvest preserving first-seen
    order across workers in their list-order.

    Worker w_1's URLs come first (in the order they were harvested),
    then w_2's URLs not already seen, and so on. This keeps the
    deterministic order useful for cache-keyed Phase-2 partitioning
    while not stamping any "winner-takes-all" structure on the chunks
    themselves.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for chunk in per_worker_urls:
        for url in chunk:
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            merged.append(url)
    return merged


def prepare_phase2_suites(
    suite_dict: dict, collected_urls: list[str], group: LoopGroup,
    workers: int,
) -> list[dict]:
    """Build per-worker Phase-2 sub-suites — one navigate+scroll+extract
    plan per URL slice.

    Splits ``collected_urls`` round-robin across ``workers`` chunks via
    :func:`partition_urls`, then for each chunk builds a sub-suite whose
    ``_micro_plan`` is a flat sequence of (navigate, scroll, extract_data)
    triples — one per URL in the chunk.

    No inner loop, no extraction iteration, no dedup state. Each URL is
    its own one-shot atomic unit of work. The phase rewriter (issue #621)
    is unnecessary here because Phase-1 already produced unique URLs.

    Returns one sub-suite per chunk; chunks with zero URLs are dropped
    (the caller doesn't need to spawn idle workers).
    """
    if not collected_urls:
        return []

    chunks = partition_urls(collected_urls, max(1, workers))
    chunks = [c for c in chunks if c]  # drop empty slices

    # Resolve any setup-section steps to carry forward (cookie banner
    # dismissals, etc). For now we omit them — each Phase-2 worker
    # navigates directly to a listing URL, no results-page touch.
    steps_raw = suite_dict.get("_micro_plan") or []

    body_dicts = steps_raw[group.body_range[0]: group.body_range[1]]

    # Per-URL extraction sequence: everything in the loop body from the
    # first ``scroll`` step onwards, minus ``navigate_back`` (loop-iter
    # mechanics — Phase-2 already isolates each URL into its own
    # navigate). Captures any in-page interactions the plan author
    # placed BETWEEN scroll and extract_data (e.g. ``click(Show More)``
    # to expand a truncated description so phone numbers buried in the
    # freeform text become extractable).
    #
    # Steps BEFORE the first scroll (typically ``click(listing_card)``
    # + ``extract_url``) are loop-navigation: they're how the original
    # plan transitions from results-page → detail-page. Phase-2 skips
    # them because each worker navigates directly to a known URL.
    #
    # Fallback: when the body has no ``scroll`` (degenerate plans) we
    # synthesize the legacy ``(scroll, extract_data)`` pair so existing
    # tests + production plans built before this rewrite continue to
    # work unchanged.
    first_scroll_idx = next(
        (i for i, s in enumerate(body_dicts) if s.get("type") == "scroll"),
        None,
    )
    if first_scroll_idx is not None:
        per_url_templates = [
            _step_dict_clone(s)
            for s in body_dicts[first_scroll_idx:]
            if s.get("type") != "navigate_back"
        ]
    else:
        per_url_templates = [
            {
                "intent": "Scroll the listing page to surface description + more details",
                "type": "scroll",
                "section": "extraction",
                "budget": 10,
                "required": False,
                "params": {},
                "hints": {},
                "claude_only": False,
                "gate": False,
                "loop_target": -1,
                "loop_count": 0,
                "grounding": False,
                "verify": "",
                "reverse": "",
            },
        ]
        extract_from_body = next(
            (_step_dict_clone(s) for s in body_dicts if s.get("type") == "extract_data"),
            None,
        )
        per_url_templates.append(extract_from_body or {
            "intent": "Extract structured fields from the listing detail page",
            "type": "extract_data",
            "section": "extraction",
            "claude_only": True,
            "budget": 0,
            "required": False,
            "params": {},
            "hints": {},
            "gate": False,
            "loop_target": -1,
            "loop_count": 0,
            "grounding": False,
            "verify": "",
            "reverse": "",
        })

    def _navigate_step(url: str) -> dict:
        return {
            "intent": f"Navigate to {url}",
            "type": "navigate",
            "section": "extraction",
            "budget": 3,
            "required": True,
            "params": {"url": url, "wait_after_load_seconds": 6},
            "hints": {},
            "claude_only": False,
            "gate": False,
            "loop_target": -1,
            "loop_count": 0,
            "grounding": False,
            "verify": "",
            "reverse": "",
        }

    sub_suites: list[dict] = []
    for i, chunk in enumerate(chunks):
        plan_steps: list[dict] = []
        for url in chunk:
            plan_steps.append(_navigate_step(url))
            for tmpl in per_url_templates:
                plan_steps.append(_step_dict_clone(tmpl))
        sub_suite = dict(suite_dict)
        sub_suite["_micro_plan"] = plan_steps
        # Drop loop_groups — the rewritten plan has no loops.
        sub_suite.pop("_loop_groups", None)
        sub_suite["session_name"] = (
            f"{suite_dict.get('session_name', 'fanout')}_phase2_w{i + 1}"
        )
        sub_suite["_fanout_phase"] = "phase2_extract"
        sub_suite["_fanout_url_count"] = len(chunk)
        sub_suites.append(sub_suite)

    logger.warning(
        "  [fanout/phase2] partitioned %d URL(s) across %d worker(s) "
        "(avg %.1f URLs/worker)",
        len(collected_urls), len(sub_suites),
        len(collected_urls) / max(len(sub_suites), 1),
    )
    return sub_suites


def prepare_modal_partitions(
    suite_dict: dict, workers: int,
    *, max_pages: int | None = None,
) -> list[dict]:
    """Build per-partition task_suite dicts for the Modal fan-out path.

    Reads:
      - ``_micro_plan``: list of step dicts (PlanDecomposer.MicroIntent
        form) — required.
      - ``_loop_groups``: list of LoopGroup dicts (#614). Optional;
        recomputed via the classifier when absent.

    Behavior:
      - If the plan has no parallelizable loop group, returns an
        empty list (caller falls through to single-worker dispatch).
      - For ``parallelizable_pagination``, synthesizes per-page URLs,
        produces one sub-suite per page where the setup navigate URL
        is rewritten and the outer pagination loop is dropped.
      - For ``parallelizable_url_collect``, returns an empty list with
        a WARNING — that shape needs the collect_urls → fan-out
        bridge wired in a follow-up (see #618 follow-up).

    ``max_pages`` overrides the loop step's ``loop_count`` when set
    (operator can cap the parallelism without re-decomposing the plan).
    """
    steps = suite_dict.get("_micro_plan") or []
    if not steps:
        return []

    # #629: use the shared plan-builder so the plan-level
    # ``pagination_url_template`` is picked up here too (was a real bug:
    # the older copy-pasted inline build didn't read the field, so
    # plan-level templates silently fell back to the default).
    plan = _build_plan_from_suite(suite_dict)

    pagination_group = next(
        (g for g in plan.loop_groups if g.shape == "parallelizable_pagination"),
        None,
    )
    if pagination_group is None:
        url_collect = next(
            (g for g in plan.loop_groups if g.shape == "parallelizable_url_collect"),
            None,
        )
        if url_collect is not None:
            logger.warning(
                "  [fanout/modal] plan has parallelizable_url_collect group but "
                "no pagination group — url-collect fan-out not yet wired into "
                "the Modal transport (follow-up issue). Falling through to "
                "single-worker dispatch."
            )
        return []

    # Resolve the setup navigate URL (the first navigate step's URL).
    base_url = ""
    for step in plan.steps:
        if step.type == "navigate":
            base_url = (step.params or {}).get("url", "")
            if base_url:
                break
    if not base_url:
        logger.warning(
            "  [fanout/modal] no setup navigate URL found — can't synthesize "
            "partition URLs. Falling through to single-worker dispatch."
        )
        return []

    # #629: template resolution order — plan-level field first, then
    # paginate-step params (back-compat), then default. The plan-level
    # field is the right home: it's a property of the plan as a whole,
    # not one step. Paginate-step params stay supported for plans
    # authored before #629 landed.
    template = DEFAULT_PAGINATION_URL_TEMPLATE
    plan_level_template = str(
        getattr(plan, "pagination_url_template", "") or ""
    )
    if (
        plan_level_template
        and "{base}" in plan_level_template
        and "{n}" in plan_level_template
    ):
        template = plan_level_template
    else:
        # The paginate step lives anywhere in the body range — the
        # body's first step is the extraction click (per
        # _fix_loop_targets which rewinds both inner + outer loop
        # targets to the same click). Scan the body to find it.
        body_start, body_end = pagination_group.body_range
        paginate_step = next(
            (s for s in plan.steps[body_start:body_end] if s.type == "paginate"),
            None,
        )
        if paginate_step is not None:
            custom = (paginate_step.params or {}).get("url_template")
            if isinstance(custom, str) and "{base}" in custom and "{n}" in custom:
                template = custom

    pages = max_pages if max_pages else (
        plan.steps[pagination_group.loop_step_idx].loop_count or 5
    )
    partition_urls = partition_urls_for_pagination(
        base_url, pages, template=template,
    )

    # Build per-partition sub-suites. Each sub-suite is a deep-ish copy
    # of the parent — only ``_micro_plan`` is rewritten.
    #
    # Two surgical edits to the per-worker sub-plan:
    #
    #   1. Drop the outer pagination loop step itself — each worker
    #      processes a single assigned page, so the outer loop is
    #      replaced by partition slicing.
    #   2. Drop any ``paginate`` step that sits between body_start
    #      and the outer loop — pagination is done by the orchestrator
    #      via the partition URL list, the worker must not click
    #      "Next page".
    #
    # The INNER extraction loop body (click → extract_url → scroll →
    # extract_data → navigate_back → loop) is preserved verbatim so
    # each worker iterates over the ~25 listings on its assigned page.
    # The earlier version dropped the entire body_range which left
    # workers with only the setup-navigate step — verified empirically
    # by the first deploy producing 0 leads × 5 partitions.
    sub_suites: list[dict] = []
    body_start, _body_end = pagination_group.body_range
    loop_idx = pagination_group.loop_step_idx
    for partition_url in partition_urls:
        sub_plan_steps: list[dict] = []
        for i, step_dict in enumerate(steps):
            # Drop the outer pagination loop step itself.
            if i == loop_idx:
                continue
            # Drop ``paginate`` steps that live inside the pagination
            # body — the orchestrator owns pagination via partition
            # URLs.
            if (
                body_start <= i < loop_idx
                and step_dict.get("type") == "paginate"
            ):
                continue
            cloned = dict(step_dict)
            # Rewrite the first navigate's URL to this partition.
            if (
                cloned.get("type") == "navigate"
                and cloned.get("section") in ("setup", "", None)
                and cloned.get("params", {}).get("url") == base_url
            ):
                cloned_params = dict(cloned.get("params", {}))
                cloned_params["url"] = partition_url
                cloned["params"] = cloned_params
                cloned["intent"] = f"Navigate to {partition_url}"
            sub_plan_steps.append(cloned)
        sub_suite = dict(suite_dict)
        sub_suite["_micro_plan"] = sub_plan_steps
        # The sub-suites are single-page now; drop loop_groups so the
        # child container doesn't try to re-fan-out.
        sub_suite.pop("_loop_groups", None)
        sub_suite["session_name"] = (
            f"{suite_dict.get('session_name', 'fanout')}_p{len(sub_suites) + 1}"
        )
        sub_suites.append(sub_suite)

    if workers > 0:
        # The caller may use fewer workers than partitions; we still
        # return one sub-suite per partition (the Modal driver assigns
        # them to workers as they free up).
        logger.warning(
            "  [fanout/modal] prepared %d partition(s) for %d worker(s) "
            "(template=%s base=%s)",
            len(sub_suites), workers, template, base_url[:80],
        )
    return sub_suites


# ── #673: shared fanout dispatcher (CLI + HTTP path) ────────────────────


def run_fanout_dispatch(
    task_suite: dict,
    *,
    executor_fn: Any,
    model: str,
    claude_model: str,
    max_steps: int,
    workers: int,
    fanout_parent_run_id: str,
    json_dumps: Callable[[Any], str] | None = None,
    shared_seen_printer: Callable[[dict, int], None] | None = None,
) -> dict | None:
    """Run Phase-1/Phase-2 fan-out dispatch and return the aggregate result.

    Returns
    -------
    dict
        ``{"viable": int, "leads_with_phone": int, "leads": list,
        "collected_urls": list, "shared_seen_hits": int}`` shaped like
        :func:`mantis_agent.server_utils.build_micro_result` when the
        fan-out completed.
    None
        When the suite is not fan-out eligible (no
        ``parallelizable_url_collect`` group) OR Phase-1 harvested zero
        URLs — caller should fall through to the legacy per-page
        partition path (or to single-runner execution).

    The body is the verbatim lift of the Phase-1/Phase-2 spawn block
    that lived in :func:`deploy.modal.modal_cua_server.main` lines
    3060-3270 before #673. Same control flow, same print statements,
    same per-partition ``_fanout_branch_id`` labels — the only thing
    that changes is the call site (CLI + HTTP `/v1/predict` now both
    call this single helper).

    Recursion guard: callers that route through this helper must check
    ``task_suite.get("_fanout_phase")`` first. Sub-suites produced by
    :func:`prepare_phase1_suite` / :func:`prepare_phase2_suites` carry
    that key set; a sub-worker re-entering the dispatcher would spawn
    grand-children indefinitely. The single-runner path handles them.

    Why this is a free function not a method: the CLI entrypoint and
    the HTTP-path Modal executor both spawn workers via the same
    ``executor_fn`` (a ``modal.Function`` ref). Passing ``executor_fn``
    in keeps the helper Modal-agnostic — tests can pass a MagicMock.
    """
    _json = json_dumps if json_dumps is not None else _default_json_dumps

    url_collect_group = find_url_collect_group(task_suite)
    if url_collect_group is None:
        return None

    phase1_max_pages, phase1_template = resolve_phase1_max_pages(task_suite)
    phase1_workers_req = task_suite.get("_fanout_phase1_workers")
    try:
        phase1_workers = max(1, int(phase1_workers_req or 1))
    except (TypeError, ValueError):
        phase1_workers = 1
    phase1_workers = min(phase1_workers, phase1_max_pages)

    # augur-sdk 0.4.0 (#38) — open a parent-only DebugSession that
    # carries the fan-out's aggregate metadata (phase counts, fan-out
    # pattern, plan signature) so the Augur viewer renders one parent
    # row with N children grouped under it via the children's
    # ``branch_context.parent_run_id`` (mercurialsolo/augur#138).
    # The helper is a no-op when augur-sdk is unavailable / disabled /
    # predates 0.4.0 — the server still synthesizes a parent row in
    # those cases; the orchestrator session is purely for the
    # aggregate tags the children can't supply on their own.
    plan_signature = str(task_suite.get("_plan_signature") or "")
    orchestrator_tags: dict[str, str] = {
        "phase1_workers": str(phase1_workers),
        "phase2_workers_configured": str(workers),
        "fanout_pattern": "phase1_collect_phase2_extract",
        "phase1_max_pages": str(phase1_max_pages),
    }
    if plan_signature:
        orchestrator_tags["plan_signature"] = plan_signature[:64]
    session_name = (
        f"fanout-{plan_signature[:12]}" if plan_signature else "fanout"
    )
    # augur-sdk 0.6.0 (#680): compose a TaskSpec from suite metadata
    # and propagate ``group_id`` to every child session via the suite
    # envelope. Both are passthrough kwargs on
    # ``open_orchestrator_session`` — silently ignored by older SDKs.
    task_spec = _build_task_spec_from_suite(
        task_suite,
        phase1_workers=phase1_workers,
        phase1_max_pages=phase1_max_pages,
        phase2_workers=workers,
        max_steps=max_steps,
    )
    # Children read this on AugurAdapter init and forward to
    # ``DebugSession(group_id=...)``; the GRPO/RL sibling correlation
    # in the Augur viewer keys on the same field that already drives
    # branch_context.parent_run_id grouping.
    task_suite["_fanout_group_id"] = fanout_parent_run_id
    with open_orchestrator_session(
        run_id=fanout_parent_run_id,
        session_name=session_name,
        tags=orchestrator_tags,
        task_spec=task_spec,
        group_id=fanout_parent_run_id,
    ):
        return _dispatch_phases(
            task_suite,
            _json=_json,
            url_collect_group=url_collect_group,
            phase1_max_pages=phase1_max_pages,
            phase1_template=phase1_template,
            phase1_workers=phase1_workers,
            executor_fn=executor_fn,
            model=model,
            claude_model=claude_model,
            max_steps=max_steps,
            workers=workers,
            fanout_parent_run_id=fanout_parent_run_id,
            shared_seen_printer=shared_seen_printer,
        )


def _dispatch_phases(
    task_suite: dict,
    *,
    _json: Callable[[Any], str],
    url_collect_group: LoopGroup,
    phase1_max_pages: int,
    phase1_template: str | None,
    phase1_workers: int,
    executor_fn: Any,
    model: str,
    claude_model: str,
    max_steps: int,
    workers: int,
    fanout_parent_run_id: str,
    shared_seen_printer: Callable[[dict, int], None] | None,
) -> dict | None:
    """Inner Phase-1/Phase-2 spawn body — extracted so the outer
    :func:`run_fanout_dispatch` can wrap it in the orchestrator-session
    context manager without indenting the original 200-line block.
    """
    if phase1_workers <= 1:
        # Serial path — exactly the #638 axis-2 behaviour.
        print(
            f"\n  ═══ PHASE-1 (#628): URL collection on 1 container "
            f"(max_pages={phase1_max_pages}, "
            f"template={phase1_template!r}) ═══"
        )
        phase1_suite = prepare_phase1_suite(
            task_suite, url_collect_group,
            max_pages=phase1_max_pages,
            pagination_url_template=phase1_template,
        )
        phase1_suite["_fanout_branch_id"] = (
            f"{fanout_parent_run_id}:phase1"
        )
        spawn_kwargs = {
            "task_file_contents": _json(phase1_suite),
            "max_steps": max_steps,
        }
        if model == "claude":
            spawn_kwargs["claude_model"] = claude_model
        phase1_handle = executor_fn.spawn(**spawn_kwargs)
        print("    [phase1] worker spawned")
        try:
            _phase1_raw = phase1_handle.get()
            if isinstance(_phase1_raw, dict):
                _urls_in_result = _phase1_raw.get(
                    "collected_urls", "MISSING"
                )
                _urls_type = type(_urls_in_result).__name__
                _urls_len = (
                    len(_urls_in_result)
                    if isinstance(_urls_in_result, list) else "n/a"
                )
                print(
                    f"    [phase1] raw result: "
                    f"viable={_phase1_raw.get('viable', 'MISSING')} "
                    f"collected_urls_type={_urls_type} "
                    f"collected_urls_len={_urls_len}"
                )
            phase1_summary = read_partition_result(_phase1_raw)
            collected_urls = phase1_summary["collected_urls"]
        except Exception as exc:  # noqa: BLE001
            print(f"    [phase1] ERROR: {exc} — aborting fan-out")
            collected_urls = []
        print(
            f"    [phase1] harvested {len(collected_urls)} "
            f"unique URL(s)"
        )
    else:
        # #644: M parallel Phase-1 workers each scanning a page-slice.
        phase1_sub_suites = prepare_phase1_partitions(
            task_suite, url_collect_group,
            n_workers=phase1_workers,
            max_pages=phase1_max_pages,
            pagination_url_template=phase1_template,
        )
        print(
            f"\n  ═══ PHASE-1 (#644 axis 3): URL collection "
            f"on {len(phase1_sub_suites)} containers "
            f"(max_pages={phase1_max_pages}, "
            f"template={phase1_template!r}) ═══"
        )
        phase1_handles: list = []
        for i, sub in enumerate(phase1_sub_suites):
            sub["_fanout_branch_id"] = (
                f"{fanout_parent_run_id}:phase1_w{i + 1}"
            )
            spawn_kwargs = {
                "task_file_contents": _json(sub),
                "max_steps": max_steps,
            }
            if model == "claude":
                spawn_kwargs["claude_model"] = claude_model
            handle = executor_fn.spawn(**spawn_kwargs)
            phase1_handles.append((i, handle))
            print(
                f"    [phase1] worker {i + 1}/"
                f"{len(phase1_sub_suites)} spawned "
                f"(pages={sub['_fanout_phase1_page_set']})"
            )
        per_worker_urls: list[list[str]] = []
        for i, handle in phase1_handles:
            try:
                summary = read_partition_result(handle.get())
                urls = summary["collected_urls"]
                per_worker_urls.append(urls)
                print(
                    f"    [phase1] worker {i + 1}: "
                    f"collected_urls={len(urls)}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [phase1] worker {i + 1} ERROR: {exc}")
                per_worker_urls.append([])
        collected_urls = dedup_urls_across_workers(per_worker_urls)
        raw_total = sum(len(c) for c in per_worker_urls)
        if raw_total > len(collected_urls):
            print(
                f"    [phase1] cross-worker dedup: "
                f"{raw_total} → {len(collected_urls)} unique URL(s)"
            )
        else:
            print(
                f"    [phase1] harvested {len(collected_urls)} "
                f"unique URL(s) (no cross-worker duplicates)"
            )

    if not collected_urls:
        print(
            "    [phase1] no URLs harvested — caller should fall "
            "through to pagination-partition path"
        )
        return None

    phase2_suites = prepare_phase2_suites(
        task_suite, collected_urls, url_collect_group, workers,
    )
    print(
        f"\n  ═══ PHASE-2 (#628): "
        f"{len(phase2_suites)} worker(s) × "
        f"{len(collected_urls)} URL(s) ═══"
    )
    phase2_handles: list = []
    for i, sub_suite in enumerate(phase2_suites):
        sub_suite["_fanout_branch_id"] = (
            f"{fanout_parent_run_id}:phase2_w{i + 1}"
        )
        kwargs = {
            "task_file_contents": _json(sub_suite),
            "max_steps": max_steps,
        }
        if model == "claude":
            kwargs["claude_model"] = claude_model
        handle = executor_fn.spawn(**kwargs)
        phase2_handles.append((i, handle))
        print(
            f"    [phase2] worker {i + 1}/{len(phase2_suites)} "
            f"spawned ({sub_suite.get('_fanout_url_count', '?')} URLs)"
        )

    merged_phone = 0
    merged_shared_seen_hits = 0
    per_worker_leads: list[list[dict]] = []
    for i, handle in phase2_handles:
        try:
            summary = read_partition_result(handle.get())
            merged_phone += summary["with_phone"]
            merged_shared_seen_hits += summary["shared_seen_hits"]
            per_worker_leads.append(summary["leads"])
            print(
                f"    [phase2] worker {i + 1}: "
                f"viable={summary['viable']} "
                f"phone={summary['with_phone']} "
                f"shared_seen_hits={summary['shared_seen_hits']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [phase2] worker {i + 1}: ERROR — {exc}")
            per_worker_leads.append([])

    leads, raw_total, dedup_total = dedup_leads_by_url(per_worker_leads)
    print("\n  ═══ PHASE-1/PHASE-2 RESULTS (#628) ═══")
    print(f"  URLs collected (Phase 1):  {len(collected_urls)}")
    print(f"  Workers (Phase 2):         {len(phase2_suites)}")
    print(f"  Total leads (raw):         {raw_total}")
    print(f"  Total leads (deduped):     {dedup_total}")
    if raw_total > dedup_total:
        print(
            f"  Duplicates collapsed:      {raw_total - dedup_total} "
            f"(unexpected — Phase 1 URLs were already unique)"
        )
    print(f"  With phone:                {merged_phone}")
    if shared_seen_printer is not None:
        try:
            shared_seen_printer(task_suite, merged_shared_seen_hits)
        except Exception as exc:  # noqa: BLE001 — telemetry never breaks runs
            print(f"  [shared-seen] printer raised: {exc}")

    return {
        "viable": dedup_total,
        "leads_with_phone": merged_phone,
        "leads": leads,
        "collected_urls": collected_urls,
        "shared_seen_hits": merged_shared_seen_hits,
    }


def _default_json_dumps(obj: Any) -> str:
    """Default JSON serializer used by :func:`run_fanout_dispatch`.

    Standard library only — no torch / pydantic transitive deps —
    so the helper stays importable in the slim ``orchestrator``
    extras footprint the HTTP API container ships with.
    """
    import json
    return json.dumps(obj)


def _build_task_spec_from_suite(
    task_suite: dict,
    *,
    phase1_workers: int,
    phase1_max_pages: int,
    phase2_workers: int,
    max_steps: int,
) -> dict[str, Any]:
    """Compose an ``augur_sdk.TaskSpec`` dict from a fan-out suite.

    Surfaces in the Augur viewer on the orchestrator (parent) row, so
    downstream RL trainers can read the canonical task definition
    instead of reconstructing it from worker bundles. Every field is
    optional in the 0.6.0 schema; we only emit keys we have.

    Field mapping (suite → task_spec):

    * ``_plan_name`` + domain → ``task_spec_id`` (``<domain>.<plan>.v1``)
    * ``_plan_name`` → ``instruction`` fallback when the plan didn't
      ship a top-level natural-language brief
    * domain from ``_site_config.url_patterns[0]`` → ``task_class``
    * ``max_steps`` arg → ``max_steps`` (planner-level budget, not the
      per-step LLM token budget)
    * Fan-out shape (Phase-1 max-pages, worker counts) → ``env_id``
      annotation so siblings produced by the same dispatch share an
      env identity in the trainer's eyes.
    """
    plan_name = str(task_suite.get("_plan_name") or "").strip()
    site_config = task_suite.get("_site_config") or {}
    domain = ""
    if isinstance(site_config, dict):
        domain = str(site_config.get("domain") or "").strip()
    if not domain:
        # Fallback: try to read off the first URL pattern in the suite.
        url_patterns = (
            site_config.get("url_patterns") if isinstance(site_config, dict)
            else None
        )
        if isinstance(url_patterns, list) and url_patterns:
            first = str(url_patterns[0])
            if "//" in first:
                domain = first.split("//", 1)[1].split("/", 1)[0]
    task_class = domain or "unknown"
    task_spec_id = (
        f"{task_class}.{plan_name}.v1" if plan_name and task_class != "unknown"
        else (f"{task_class}.fanout.v1" if task_class != "unknown" else "")
    )
    spec: dict[str, Any] = {}
    if task_spec_id:
        spec["task_spec_id"] = task_spec_id
    if plan_name:
        spec["instruction"] = plan_name.replace("_", " ")
    if task_class != "unknown":
        spec["task_class"] = task_class
    if max_steps and max_steps > 0:
        spec["max_steps"] = int(max_steps)
    # ``env_id`` is the trainer-visible identity of the env the
    # rollout exercised. Stamp the Modal app + fan-out shape so
    # sibling rollouts (Phase-1 workers, Phase-2 workers) read as
    # variants of one env, not independent envs.
    spec["env_id"] = (
        f"modal:mantis-cua-server:fanout"
        f"(p1={phase1_workers}xp{phase1_max_pages},p2={phase2_workers})"
    )
    return spec
