"""Plan-evolution persistence + promotion gate — Phase 2 (#706).

Persists step rewrites learned during agentic recovery so the **next**
run of the same plan starts smarter. JSON-on-volume, scope-keyed,
promotion-gated.

Storage layout:

    /data/plan_evolution/workflow/<workflow_id>/<plan_hash>.json

(tenant- and site-scopes land in Phase 3 — #707.)

Public surface used by the rest of Mantis:

* :func:`apply_plan_overlay` — pre-flight: load promoted rewrites and
  return a MicroPlan with them applied. Called by the dispatcher before
  ``build_micro_suite``; idempotent + no-op when no store exists.
* :func:`record_rewrite_candidate` — runtime: stamp every rewrite the
  recovery loop applies as a `candidate`. Called from
  :mod:`mantis_agent.agentic_recovery` when a `rewrite_url` decision is
  built.
* :func:`record_run_outcome` — terminal: for every rewrite that
  participated in this run, increment success / failure counters and
  apply the promotion / demotion gates.

Promotion semantics (per spec):

* `candidate` → `promoted` after **3 consecutive successful runs** of
  the same rewrite (matched by step_index + new_url within ±10% URL
  diff for tracker params).
* `promoted` → `demoted` after **2 consecutive failed runs**.
* Rewrites idle for **30 days** flip to `cold`; cold rewrites must
  re-promote (3 fresh successes) before re-applying.

Concurrency: atomic writes via tmpfile + rename. Concurrent runs of the
same plan race on counter increments; the gate uses *consecutive* runs
so a missed increment delays promotion by one cycle (acceptable for
Phase 2). Phase 3 may upgrade to SQLite if multi-region writes become
common.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..plan_decomposer import MicroPlan

logger = logging.getLogger(__name__)


RewriteScope = Literal["workflow", "tenant", "site"]
RewriteStatus = Literal["candidate", "promoted", "demoted", "cold"]
RewriteSource = Literal[
    "pattern_transform", "page_links", "web_search",
    "brain_proposal", "manual",
]

# Spec defaults — exposed as module-level constants so tests + future
# operator overrides can tune them without touching the class body.
PROMOTION_THRESHOLD = 3       # consecutive successes → promoted
DEMOTION_THRESHOLD = 2        # consecutive failures while promoted → demoted
COLD_AGE_SECONDS = 30 * 86400  # 30 days unused → cold


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _root_dir() -> str:
    """Where the store lives on disk.

    Volume-backed in production (`/data/plan_evolution`), env-overridable
    for tests + local CLI runs (`MANTIS_PLAN_EVOLUTION_DIR`).
    """
    return os.environ.get(
        "MANTIS_PLAN_EVOLUTION_DIR",
        os.path.join(
            os.environ.get("MANTIS_DATA_DIR", "/data"), "plan_evolution",
        ),
    )


def _file_path(scope: RewriteScope, scope_id: str, plan_hash: str) -> str:
    return os.path.join(_root_dir(), scope, _sanitize(scope_id), f"{plan_hash}.json")


def _sanitize(s: str) -> str:
    """Filesystem-safe scope id — same convention server_utils.safe_state_key uses."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)[:120] or "_"


# ── records ──────────────────────────────────────────────────────────


@dataclass
class StepRewrite:
    step_index: int
    original: dict             # step body as authored
    rewritten: dict            # in-place replacement (intent + params)
    source: RewriteSource
    confidence: float
    scope: RewriteScope = "workflow"
    status: RewriteStatus = "candidate"
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    successful_runs: int = 0
    failed_runs: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    demotion_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "original": self.original,
            "rewritten": self.rewritten,
            "source": self.source,
            "confidence": self.confidence,
            "scope": self.scope,
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "demotion_reason": self.demotion_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepRewrite":
        return cls(
            step_index=int(d.get("step_index", 0)),
            original=dict(d.get("original", {}) or {}),
            rewritten=dict(d.get("rewritten", {}) or {}),
            source=d.get("source", "pattern_transform"),
            confidence=float(d.get("confidence", 0.0)),
            scope=d.get("scope", "workflow"),
            status=d.get("status", "candidate"),
            first_seen=d.get("first_seen", _now_iso()),
            last_seen=d.get("last_seen", _now_iso()),
            successful_runs=int(d.get("successful_runs", 0)),
            failed_runs=int(d.get("failed_runs", 0)),
            consecutive_successes=int(d.get("consecutive_successes", 0)),
            consecutive_failures=int(d.get("consecutive_failures", 0)),
            demotion_reason=d.get("demotion_reason"),
        )


@dataclass
class PlanEvolution:
    plan_hash: str
    scope: RewriteScope
    scope_id: str            # workflow_id (Phase 2); tenant_id / site (Phase 3)
    rewrites: list[StepRewrite] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "rewrites": [r.to_dict() for r in self.rewrites],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanEvolution":
        return cls(
            plan_hash=str(d.get("plan_hash", "")),
            scope=d.get("scope", "workflow"),
            scope_id=str(d.get("scope_id", "")),
            rewrites=[StepRewrite.from_dict(r) for r in d.get("rewrites", [])],
        )

    def find_rewrite(
        self, *, step_index: int, new_url: str,
    ) -> StepRewrite | None:
        """Match a rewrite by step + URL (with tracker-param tolerance)."""
        for r in self.rewrites:
            if r.step_index != step_index:
                continue
            if _urls_match(_url_in_step(r.rewritten), new_url):
                return r
        return None


# ── store I/O ────────────────────────────────────────────────────────


def _load(scope: RewriteScope, scope_id: str, plan_hash: str) -> PlanEvolution:
    path = _file_path(scope, scope_id, plan_hash)
    try:
        with open(path) as f:
            data = json.load(f)
        return PlanEvolution.from_dict(data)
    except FileNotFoundError:
        return PlanEvolution(plan_hash=plan_hash, scope=scope, scope_id=scope_id)
    except Exception as exc:  # noqa: BLE001 — corrupt store → start fresh
        logger.warning(
            "plan_evolution: failed to load %s (%s); starting fresh", path, exc,
        )
        return PlanEvolution(plan_hash=plan_hash, scope=scope, scope_id=scope_id)


def _save(evolution: PlanEvolution) -> None:
    """Atomic write via tmpfile + rename."""
    path = _file_path(evolution.scope, evolution.scope_id, evolution.plan_hash)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
    with open(tmp_path, "w") as f:
        json.dump(evolution.to_dict(), f, indent=2, sort_keys=False)
    os.replace(tmp_path, path)


# ── apply overlay (pre-flight) ───────────────────────────────────────


def apply_plan_overlay(
    plan: "MicroPlan",
    *,
    plan_hash: str,
    workflow_id: str,
    include_candidates: bool = False,
) -> tuple["MicroPlan", list[StepRewrite]]:
    """Return the plan with learned rewrites applied + the list of
    rewrites that were applied.

    Pre-flight: called BEFORE the executor runs so the rewritten URLs land
    in the steps the executor receives. Idempotent + no-op when no store
    exists or nothing is applicable.

    ``promoted`` rewrites are always applied. With ``include_candidates``
    (exploration, #894), not-yet-promoted ``candidate`` rewrites are ALSO
    applied so they accumulate the consecutive wins promotion requires —
    at the cost of applying an unproven rewrite to a live run. The
    cold-idle demotion applies to ``promoted`` rewrites only (candidates
    have no "promoted-but-stale" state to expire).

    Phase 2 ships ``scope='workflow'`` only — tenant + site scopes added
    in Phase 3.
    """
    if not plan_hash or not workflow_id:
        return plan, []

    evo = _load("workflow", workflow_id, plan_hash)
    if not evo.rewrites:
        return plan, []

    applicable = {"promoted", "candidate"} if include_candidates else {"promoted"}
    applied: list[StepRewrite] = []
    now_ts = time.time()
    for rewrite in evo.rewrites:
        if rewrite.status not in applicable:
            continue
        # Cold-transition gate: promoted rewrites unused for ≥ COLD_AGE
        # demote to `cold` and skip application. Re-promotion requires
        # 3 fresh successful runs. (Candidates are exempt — they're being
        # explored toward promotion, not expired.)
        if rewrite.status == "promoted":
            last_ts = _parse_iso(rewrite.last_seen)
            if last_ts and (now_ts - last_ts) > COLD_AGE_SECONDS:
                rewrite.status = "cold"
                rewrite.demotion_reason = "idle_30d"
                continue
        if rewrite.step_index >= len(plan.steps):
            continue
        step = plan.steps[rewrite.step_index]
        _apply_rewrite_in_place(step, rewrite.rewritten)
        applied.append(rewrite)
        logger.warning(
            "  [plan-overlay] step %d rewritten via %s "
            "(%s, confidence=%.2f, %d/%d successes)",
            rewrite.step_index, rewrite.source, rewrite.status,
            rewrite.confidence, rewrite.successful_runs, PROMOTION_THRESHOLD,
        )

    # Persist any cold-transition demotions.
    if any(r.status == "cold" for r in evo.rewrites):
        try:
            _save(evo)
        except Exception as exc:  # noqa: BLE001
            logger.debug("plan_evolution: cold-transition save failed: %s", exc)

    return plan, applied


def _apply_rewrite_in_place(step: Any, rewritten: dict) -> None:
    """Merge `rewritten` fields into the MicroIntent."""
    if "intent" in rewritten and rewritten["intent"]:
        step.intent = str(rewritten["intent"])
    if "type" in rewritten and rewritten["type"]:
        step.type = str(rewritten["type"])
    if "params" in rewritten and isinstance(rewritten["params"], dict):
        merged = dict(getattr(step, "params", {}) or {})
        merged.update(rewritten["params"])
        step.params = merged


# ── record candidate (runtime) ────────────────────────────────────────


def record_rewrite_candidate(
    *,
    plan_hash: str,
    workflow_id: str,
    step_index: int,
    original_step: dict,
    rewritten_step: dict,
    source: RewriteSource,
    confidence: float,
) -> StepRewrite | None:
    """Record a rewrite produced by recovery as a `candidate`.

    Idempotent: when the same (step_index, new_url) already has a
    record, returns the existing one (updates `last_seen`). Returns
    None if the store write failed.

    Called from :mod:`mantis_agent.agentic_recovery` whenever the
    recovery loop applies a `rewrite_url` decision during a run.
    """
    if not plan_hash or not workflow_id:
        return None

    evo = _load("workflow", workflow_id, plan_hash)
    new_url = _url_in_step(rewritten_step)
    existing = evo.find_rewrite(step_index=step_index, new_url=new_url)
    if existing is not None:
        existing.last_seen = _now_iso()
        # Always keep the freshest source / confidence — heuristics may
        # improve between runs (e.g. pattern_transform → page_links).
        existing.source = source
        existing.confidence = max(existing.confidence, confidence)
    else:
        existing = StepRewrite(
            step_index=step_index,
            original=dict(original_step or {}),
            rewritten=dict(rewritten_step or {}),
            source=source,
            confidence=confidence,
            scope="workflow",
            status="candidate",
        )
        evo.rewrites.append(existing)

    try:
        _save(evo)
    except Exception as exc:  # noqa: BLE001 — never break the run on store fail
        logger.warning(
            "plan_evolution: failed to record candidate for %s/%s (%s)",
            workflow_id, plan_hash, exc,
        )
        return None
    return existing


# ── run outcome + promotion gate (terminal) ──────────────────────────


def record_run_outcome(
    *,
    plan_hash: str,
    workflow_id: str,
    applied_rewrites: list[StepRewrite],
    outcome: Literal["success", "failure"],
) -> None:
    """Increment success / failure counters for participating rewrites
    and apply the promotion / demotion gates.

    Called at run terminal — the caller passes the list of rewrites
    that participated (overlay-applied + recovery-applied).

    The gate semantics (per spec):

    * `candidate`: 3 consecutive successes → `promoted`; failure resets
      the consecutive counter.
    * `promoted`: 2 consecutive failures → `demoted`; success increments
      the lifetime counter.
    * `demoted`: counters keep accumulating but the rewrite is never
      applied via overlay. Phase 3 may add re-promotion logic; Phase 2
      requires manual operator intervention to flip back to candidate.
    * `cold`: must transition to `candidate` first (handled by
      :func:`apply_plan_overlay`'s idle check + a fresh candidate
      record).
    """
    if not plan_hash or not workflow_id or not applied_rewrites:
        return

    evo = _load("workflow", workflow_id, plan_hash)
    applied_keys = {
        (r.step_index, _url_in_step(r.rewritten)) for r in applied_rewrites
    }
    changed = False
    for stored in evo.rewrites:
        key = (stored.step_index, _url_in_step(stored.rewritten))
        if key not in applied_keys:
            continue
        stored.last_seen = _now_iso()
        if outcome == "success":
            stored.successful_runs += 1
            stored.consecutive_successes += 1
            stored.consecutive_failures = 0
            if (
                stored.status == "candidate"
                and stored.consecutive_successes >= PROMOTION_THRESHOLD
            ):
                stored.status = "promoted"
                stored.demotion_reason = None
                logger.warning(
                    "  [plan-overlay] step %d PROMOTED after %d successes "
                    "(workflow=%s)",
                    stored.step_index, stored.consecutive_successes, workflow_id,
                )
        else:
            stored.failed_runs += 1
            stored.consecutive_failures += 1
            stored.consecutive_successes = 0
            if (
                stored.status == "promoted"
                and stored.consecutive_failures >= DEMOTION_THRESHOLD
            ):
                stored.status = "demoted"
                stored.demotion_reason = f"{DEMOTION_THRESHOLD}_consecutive_failures"
                logger.warning(
                    "  [plan-overlay] step %d DEMOTED after %d failures "
                    "(workflow=%s)",
                    stored.step_index, stored.consecutive_failures, workflow_id,
                )
        changed = True

    if changed:
        try:
            _save(evo)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "plan_evolution: failed to record outcome for %s/%s (%s)",
                workflow_id, plan_hash, exc,
            )


# ── inspector (CLI + tests) ──────────────────────────────────────────


def finalize_run_outcomes(
    *,
    plan_hash: str,
    workflow_id: str,
    applied_rewrites: list[StepRewrite],
    step_results: list[Any],
) -> None:
    """Determine per-rewrite outcome from step results + record + apply
    the promotion / demotion gates.

    Called from the runner's terminal path. For each rewrite that
    participated in this run, look up the StepResult at its `step_index`;
    if the result was successful, count as `success` (advances candidate
    toward promotion); otherwise count as `failure` (advances promoted
    toward demotion).

    Failures here are non-fatal — store errors degrade silently so the
    run's terminal status isn't polluted by store I/O.
    """
    if not applied_rewrites or not plan_hash or not workflow_id:
        return
    successes: list[StepRewrite] = []
    failures: list[StepRewrite] = []
    for rewrite in applied_rewrites:
        idx = rewrite.step_index
        if idx < 0 or idx >= len(step_results):
            continue
        result = step_results[idx]
        if bool(getattr(result, "success", False)):
            successes.append(rewrite)
        else:
            failures.append(rewrite)
    if successes:
        record_run_outcome(
            plan_hash=plan_hash, workflow_id=workflow_id,
            applied_rewrites=successes, outcome="success",
        )
    if failures:
        record_run_outcome(
            plan_hash=plan_hash, workflow_id=workflow_id,
            applied_rewrites=failures, outcome="failure",
        )


def load_for_inspection(
    *, plan_hash: str, workflow_id: str, scope: RewriteScope = "workflow",
) -> PlanEvolution:
    """Read-only load for the CLI inspector + tests."""
    return _load(scope, workflow_id, plan_hash)


def list_plans(*, workflow_id: str, scope: RewriteScope = "workflow") -> list[str]:
    """List all plan_hashes with stored evolutions for a workflow."""
    dir_path = os.path.join(_root_dir(), scope, _sanitize(workflow_id))
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        f[:-5] for f in os.listdir(dir_path) if f.endswith(".json")
    )


# ── helpers ──────────────────────────────────────────────────────────


def _url_in_step(step_body: dict) -> str:
    """Extract the URL from a step body (intent prose or params.url)."""
    params = (step_body or {}).get("params") or {}
    url = str(params.get("url", "") or "").strip()
    if url:
        return url
    intent = str((step_body or {}).get("intent", "") or "")
    import re
    m = re.search(r'https?://[^\s"]+', intent)
    return m.group(0) if m else ""


def _urls_match(a: str, b: str) -> bool:
    """Compare URLs with tracker-param tolerance per spec (±10% query diff).

    Phase 2 implementation: exact match on scheme+netloc+path; ignore
    query string differences entirely. Phase 3 may add fuzzy query
    matching when the same plan starts producing tracker variants.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    pa, pb = urlparse(a), urlparse(b)
    return (
        pa.scheme == pb.scheme
        and pa.netloc == pb.netloc
        and pa.path == pb.path
    )


def _parse_iso(s: str) -> float:
    """ISO timestamp → unix seconds; 0 on parse failure."""
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return 0.0
