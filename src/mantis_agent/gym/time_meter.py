"""Per-run wall-time accounting for MicroPlanRunner — sibling to
:class:`~.cost_meter.CostMeter`.

Where CostMeter answers "where did the dollars go?", TimeMeter answers
"where did the wall-time go?". Both are bookkeeping primitives the
runner mutates in-place; both surface on the run result envelope
(Phase B / #365) and feed Prometheus.

Buckets are stable strings (the :data:`BUCKETS` constant). They are
mutually exclusive — a wrapped block of code is credited to exactly
one bucket. Sum of bucket totals tracks ``elapsed_seconds()`` within a
few percent; the residual lives in ``overhead`` (computed lazily at
read time, not stored).

Use the :meth:`measure` context manager at the call site:

>>> meter = TimeMeter(tenant_id="acme")
>>> with meter.measure("think", step_idx=3):
...     result = brain.think(frames=[...], task=...)
>>> meter.totals["think"]
0.482

The Prometheus histogram ``STEP_LATENCY_SECONDS`` (declared in
``metrics.py`` but never observed before this module) fires on every
``measure()`` exit, labelled by ``(tenant_id, phase)``. Skipped when
``tenant_id`` is empty so local / script runs don't leak a default
series into the registry.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


BUCKETS: tuple[str, ...] = (
    "perceive",       # env.screenshot() capture, viewport sync
    "think",          # brain inference (Holo3 / Gemma4 / Claude action emission)
    "act",            # env.step(action) — keystroke / mouse / scroll / CDP click
    "settle",         # post-action wait (fixed or adaptive)
    "claude_ground",  # ClaudeGrounding.refine_click(...) coordinate refinement
    "claude_extract", # ClaudeExtractor.find_* — extraction calls
    "claude_verify",  # legacy verifier path — multi-screenshot click verification
    # #421 split: ``verify_gate`` defaults to Haiku 4.5 and escalates
    # to Opus on FAIL. Tracking the two as separate buckets lets cost
    # reports show the actual savings + escalation hit-rate per run.
    "claude_verify_haiku",            # gate verification (cheap default)
    "claude_verify_opus_escalation",  # gate re-ask after Haiku FAIL
    "load",           # env.reset(url) page-load + Cloudflare wait + proxy CONNECT
    "overhead",       # residual: runner orchestration, dispatch, Python (computed lazily)
)


def _make_initial_totals() -> dict[str, float]:
    return {b: 0.0 for b in BUCKETS}


def _make_initial_step_record() -> dict[str, float]:
    return {b: 0.0 for b in BUCKETS}


@dataclass
class TimeMeter:
    """Owns wall-time counters per bucket + per-step timing records.

    Mirrors :class:`CostMeter`'s pattern: the runner records into
    ``self.totals`` via the :meth:`measure` context manager, and
    ``self.per_step[idx]`` keeps a list of per-step bucket breakdowns
    for the final result envelope (Phase B / #365).

    ``run_start`` uses :func:`time.monotonic` so :meth:`elapsed_seconds`
    is unaffected by wall-clock jumps (NTP, daylight-saving). Individual
    measurements also use monotonic time — bucket totals always advance,
    never go backwards.
    """

    tenant_id: str = ""
    totals: dict[str, float] = field(default_factory=_make_initial_totals)
    per_step: list[dict[str, float]] = field(default_factory=list)
    run_start: float = field(default_factory=time.monotonic)

    # ── Measurement primitives ──────────────────────────────────────────

    @contextmanager
    def measure(
        self, bucket: str, *, step_idx: int | None = None,
    ) -> Iterator[None]:
        """Time the wrapped block and credit ``bucket``.

        When ``step_idx`` is provided, the elapsed time is *also*
        credited to ``self.per_step[step_idx][bucket]``, growing the
        ``per_step`` list as needed so callers can address future
        steps before they've been dispatched.

        Unknown buckets raise ``KeyError`` — callers should use a
        constant from :data:`BUCKETS` to keep the vocabulary closed.
        """
        if bucket not in self.totals:
            raise KeyError(
                f"unknown TimeMeter bucket: {bucket!r}. "
                f"Allowed: {sorted(self.totals)}"
            )
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            self.totals[bucket] += elapsed
            if step_idx is not None:
                while len(self.per_step) <= step_idx:
                    self.per_step.append(_make_initial_step_record())
                self.per_step[step_idx][bucket] += elapsed
            self._emit_prometheus(bucket, elapsed)

    def record(
        self, bucket: str, seconds: float, *, step_idx: int | None = None,
    ) -> None:
        """Direct credit without a context manager.

        Useful when the elapsed time is computed elsewhere (e.g., a
        helper that already times its own work and returns the
        duration). Same accounting + Prometheus emission as
        :meth:`measure`.
        """
        if bucket not in self.totals:
            raise KeyError(
                f"unknown TimeMeter bucket: {bucket!r}. "
                f"Allowed: {sorted(self.totals)}"
            )
        if seconds < 0:
            return
        self.totals[bucket] += seconds
        if step_idx is not None:
            while len(self.per_step) <= step_idx:
                self.per_step.append(_make_initial_step_record())
            self.per_step[step_idx][bucket] += seconds
        self._emit_prometheus(bucket, seconds)

    # ── Aggregate read accessors ────────────────────────────────────────

    def elapsed_seconds(self) -> float:
        """Wall-clock since the meter was constructed."""
        return time.monotonic() - self.run_start

    def breakdown(self) -> dict[str, float]:
        """Snapshot of totals with ``overhead`` filled in as the residual.

        ``overhead`` is the difference between wall-clock elapsed and the
        sum of explicitly-measured buckets — captures runner orchestration,
        dispatch, Python work that isn't wrapped in a :meth:`measure`
        block. Floored at 0 so overlapping measurements never produce a
        negative residual.
        """
        out = dict(self.totals)
        residual = self.elapsed_seconds() - sum(
            v for k, v in self.totals.items() if k != "overhead"
        )
        # ``overhead`` may already have explicit contributions; the
        # residual adds whatever the wall clock says is unaccounted.
        out["overhead"] = max(0.0, out["overhead"] + residual)
        return out

    def step_breakdown(self, step_idx: int) -> dict[str, float]:
        """Per-step bucket record, or an all-zeros dict if the step hasn't
        been measured yet. Stable for callers that read by index.
        """
        if 0 <= step_idx < len(self.per_step):
            return dict(self.per_step[step_idx])
        return _make_initial_step_record()

    # ── Prometheus emission ─────────────────────────────────────────────

    def _emit_prometheus(self, bucket: str, seconds: float) -> None:
        """Observe one measurement into ``STEP_LATENCY_SECONDS``.

        Skipped when ``tenant_id`` is empty so local / script runs
        don't pollute the registry with a default-label series.
        Best-effort: a metric emission failure must never break a run.
        """
        if not self.tenant_id:
            return
        try:
            from .. import metrics as _metrics
            _metrics.STEP_LATENCY_SECONDS.labels(
                tenant_id=self.tenant_id, phase=bucket,
            ).observe(seconds)
        except Exception as exc:  # noqa: BLE001 — metrics are observability, never fatal
            logger.debug("STEP_LATENCY_SECONDS emit failed (bucket=%s): %s", bucket, exc)


# ── Dispatch context (contextvars) ──────────────────────────────────────


# Published by the executor for the duration of a step dispatch so deep
# helpers (adaptive_settle, brain wrappers, Claude callers) can record
# into the runner's TimeMeter without having to be passed it as an
# argument. The contract is (meter, step_idx) — both required so the
# helper can route to the right per-step bucket.
#
# PEP 567 contextvars make this scope-correct for asyncio + threaded
# callers (each task / thread sees its own publication; reset() rolls
# back deterministically on context exit). The previous PR threaded
# the meter through call-site arguments; this is the equivalent
# without touching every adaptive_settle.* invocation.
_CURRENT_DISPATCH: contextvars.ContextVar[
    "tuple[TimeMeter, int | None] | None"
] = contextvars.ContextVar("mantis_time_meter_dispatch", default=None)


def current_dispatch() -> "tuple[TimeMeter, int | None] | None":
    """Return ``(meter, step_idx)`` if a runner has published a dispatch
    context, else ``None``. Helpers that opportunistically time
    themselves should bail out cleanly when this returns ``None``.
    """
    return _CURRENT_DISPATCH.get()


@contextmanager
def publish_dispatch(
    meter: "TimeMeter | None", step_idx: int | None,
) -> Iterator[None]:
    """Publish ``(meter, step_idx)`` for the wrapped block.

    ``meter=None`` clears the context — useful in tests that exercise
    helpers in isolation. Always resets on exit, including on
    exceptions, so a crashed dispatch can't leak context into the
    next one.
    """
    token = _CURRENT_DISPATCH.set((meter, step_idx) if meter is not None else None)
    try:
        yield
    finally:
        _CURRENT_DISPATCH.reset(token)


def record_to_current(bucket: str, seconds: float) -> None:
    """Credit ``seconds`` to ``bucket`` on the currently-published meter.

    No-op when no dispatch context is published (e.g., script runs,
    standalone helper tests). Helpers call this instead of demanding
    the meter as an arg — keeps their public signature free of
    bookkeeping plumbing.
    """
    pub = _CURRENT_DISPATCH.get()
    if pub is None:
        return
    meter, step_idx = pub
    meter.record(bucket, seconds, step_idx=step_idx)


__all__ = [
    "BUCKETS", "TimeMeter",
    "current_dispatch", "publish_dispatch", "record_to_current",
]
