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
from typing import Any, Callable

from ..plan_decomposer import LoopGroup, MicroIntent, MicroPlan
from .checkpoint import StepResult

logger = logging.getLogger(__name__)


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
