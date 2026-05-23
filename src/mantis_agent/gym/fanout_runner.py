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
        try:
            return len(self._dict)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[shared-seen] size() raised: %s", exc)
            return 0


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
        return {"viable": 0, "with_phone": 0, "leads": [], "collected_urls": []}
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
    return {
        "viable": viable, "with_phone": with_phone,
        "leads": leads, "collected_urls": collected_urls,
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


def _step_dict_clone(step_dict: dict) -> dict:
    """Shallow-clone a step dict, deep-copying ``params`` + ``hints`` so
    sub-suite mutations don't leak into the source plan."""
    cloned = dict(step_dict)
    cloned["params"] = dict(step_dict.get("params") or {})
    cloned["hints"] = dict(step_dict.get("hints") or {})
    return cloned


def prepare_phase1_suite(
    suite_dict: dict, group: LoopGroup,
) -> dict:
    """Build a one-container Phase-1 sub-suite that harvests listing URLs.

    The Phase-1 worker runs the plan's setup chain (everything BEFORE
    the extraction loop body) and then a synthesized ``collect_urls``
    step (#615). The worker returns its harvested URL list to the
    orchestrator via the result envelope's ``collected_urls`` field
    (added by :func:`build_micro_result`).

    Phase-1 is a serial bottleneck — one Modal container, no fan-out —
    so its cost is bounded by the cost of one navigate + a single
    Claude scan call (~$0.02-0.05). For a 5-page plan this typically
    runs in ~30-60 seconds vs the ~3-5 min the pagination-fanout's
    setup chain takes per partition.

    Steps in the returned sub-suite's ``_micro_plan``:

      - All steps BEFORE the loop body (setup navigate, filters,
        verification gates).
      - One injected ``collect_urls`` step (claude_only=True,
        section=extraction) that runs ``CollectUrlsHandler``.
      - No loop body, no pagination — that's Phase 2's job.
    """
    body_start, _body_end = group.body_range
    loop_idx = group.loop_step_idx
    steps_raw = suite_dict.get("_micro_plan") or []
    setup_steps = [
        _step_dict_clone(s) for s in steps_raw[:body_start]
    ]
    collect_step = {
        "intent": "Collect every listing URL visible on this results page",
        "type": "collect_urls",
        "section": "extraction",
        "claude_only": True,
        "budget": 0,
        "required": True,
        "params": {},
        "hints": {},
        "gate": False,
        "loop_target": -1,
        "loop_count": 0,
        "grounding": False,
        "verify": "",
        "reverse": "",
    }
    sub_suite = dict(suite_dict)
    sub_suite["_micro_plan"] = setup_steps + [collect_step]
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
    return sub_suite


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

    # Find the extract_data body step to clone — that's the per-listing
    # extraction operator the Phase-1/Phase-2 split is built around.
    # Fall back to a minimal hand-built extract_data if absent.
    body_dicts = steps_raw[group.body_range[0]: group.body_range[1]]
    extract_template = next(
        (_step_dict_clone(s) for s in body_dicts if s.get("type") == "extract_data"),
        None,
    )
    if extract_template is None:
        extract_template = {
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
        }
    scroll_template = {
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
    }

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
            plan_steps.append(dict(scroll_template))
            plan_steps.append(dict(extract_template))
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
    from ..plan_decomposer import LoopGroup as _LG  # noqa: F401 — local for clarity

    steps = suite_dict.get("_micro_plan") or []
    if not steps:
        return []

    plan = MicroPlan(domain=str(suite_dict.get("session_name", "")))
    from ..plan_decomposer import PlanDecomposer
    for s in steps:
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

    paginate_step = plan.steps[pagination_group.body_range[0]] if pagination_group.body_range[0] < len(plan.steps) else None
    template = DEFAULT_PAGINATION_URL_TEMPLATE
    if paginate_step is not None and paginate_step.type == "paginate":
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
