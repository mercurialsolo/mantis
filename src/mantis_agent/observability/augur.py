"""Augur SDK adapter — per-run debug bundles + optional live streaming (#509).

When ``augur-sdk>=0.1.0`` is installed and ``MANTIS_AUGUR_DISABLED`` is
not set, every gym run produces a path-stable, schema-validated bundle
at ``${MANTIS_DATA_DIR}/augur/<run_id>/``. When ``AUGUR_DSN`` is also
exported, the SDK streams the same records to the configured Augur
workspace (Sentry-style DSN; 15-s heartbeat; connection-status badge).

Spec compliance (https://mercurialsolo.github.io/augur-sdk/spec/):

* §4.3 — Emission failures are non-fatal: every public method on
  :class:`AugurAdapter` swallows exceptions and demotes to a debug log.
  A misconfigured DSN, a transient network blip, or an SDK-side schema
  mismatch can never break a Mantis run.
* §4.5 — Never reads from Augur during a run. The bundle on disk is the
  source of truth; streaming is observe-only on top of it.

Coexists with :class:`mantis_agent.gym.trace_exporter.TraceExporter` —
that writer continues to emit one JSON per run alongside this adapter's
finer-grained bundle. Neither replaces the other.

This module is import-safe with or without the ``augur-sdk`` package on
disk: when the import fails, :func:`is_enabled` returns ``False`` and
every :class:`AugurAdapter` instance is a no-op.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp matching the Augur schema's
    ``YYYY-MM-DDThh:mm:ssZ`` shape. Used as the default ``ts`` on
    DecisionEvents the SDK requires the field on every event."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

# Lazy import — module must load even without augur-sdk installed.
try:
    from augur_sdk import CaptureMode, DebugSession
    _AUGUR_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import-time failure → disabled
    CaptureMode = None  # type: ignore[assignment]
    DebugSession = None  # type: ignore[assignment]
    _AUGUR_AVAILABLE = False


# ── Mantis → Augur vocabulary maps ───────────────────────────────────

# Mantis ``_healing_events`` carry the per-step reasoning trail with
# layer names that pre-date the Augur DecisionLayer enum. Remap to the
# nine Augur literals (planner/grounding/model/dispatch/verifier/
# step_recovery/routing/runner/adapter); unknown layers fall through
# to "runner" as the catch-all.
_LAYER_MAP: dict[str, str] = {
    "critic-frontier": "verifier",
    "critic": "verifier",
    "gate-decision": "verifier",
    "preview-gate": "verifier",
    "agentic-recovery": "step_recovery",
    "recovery": "step_recovery",
    "step_recovery": "step_recovery",
    "som-click": "grounding",
    "grounding": "grounding",
    "planner": "planner",
    "model": "model",
    "dispatch": "dispatch",
    "routing": "routing",
    "runner": "runner",
    "adapter": "adapter",
}

# Mantis healing-event ``kind`` field (fire/skip/decision/result/...)
# remapped to Augur DecisionKind literals (decision/observation/error/
# info/metric).
_KIND_MAP: dict[str, str] = {
    "fire": "decision",
    "skip": "decision",
    "decision": "decision",
    "result": "observation",
    "observation": "observation",
    "info": "info",
    "error": "error",
    "metric": "metric",
}

# Mantis Verdict.status → Augur VerdictStatus.
_VERDICT_STATUS_MAP: dict[str, str] = {
    "ok": "passed",
    "passed": "passed",
    "recoverable": "recoverable",
    "non_recoverable": "failed",
    "failed": "failed",
    "skipped": "skipped",
}

# Mantis runner status → Augur RunStatus literal. Mantis uses more
# granular terminal labels (``completed`` / ``completed_with_failures``
# / ``budget_exceeded`` / ``time_exceeded``); collapse them onto the
# Augur five-state enum.
_RUN_STATUS_MAP: dict[str, str] = {
    "running": "running",
    "completed": "succeeded",
    "completed_with_failures": "succeeded",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "halted": "halted",
    "budget_exceeded": "halted",
    "time_exceeded": "halted",
    "paused": "running",
}


# Mantis RecoveryDecision.type → Augur RecoveryType.
_RECOVERY_TYPE_MAP: dict[str, str] = {
    "advance": "none",
    "none": "none",
    "retry": "retry",
    "replan": "replan",
    "rollback": "alternate_grounding",
    "alternate_grounding": "alternate_grounding",
    "terminate": "halt",
    "halt": "halt",
    "skip": "skip",
    "human_handoff": "human_handoff",
}


def is_enabled() -> bool:
    """Return ``True`` when the adapter should attach to runs.

    Active when the ``augur-sdk`` package is importable AND
    ``MANTIS_AUGUR_DISABLED`` is not set to a truthy value. The DSN is
    NOT required — without it the adapter still writes the on-disk
    bundle (the source of truth per the spec).
    """
    if not _AUGUR_AVAILABLE:
        return False
    flag = os.environ.get("MANTIS_AUGUR_DISABLED", "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def is_verbose() -> bool:
    """Whether to surface adapter diagnostics at WARN (else DEBUG).

    Gated by ``MANTIS_AUGUR_VERBOSE`` — opt-in because:

    * Modal suppresses INFO/DEBUG. WARN is the only level reliably
      visible in ``modal app logs``. For deploy verification we WANT
      the per-step emission noise elevated.
    * For steady-state production we don't want a WARN line per step.

    Set ``MANTIS_AUGUR_VERBOSE=1`` on the runtime to enable. Gates
    the adapter's ``__init__`` / ``_emit_augur_step`` / ``record_step``
    exception-path WARN lines.
    """
    flag = os.environ.get("MANTIS_AUGUR_VERBOSE", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def default_out_dir(run_id: str) -> Path:
    """Resolve the per-run bundle directory.

    Honors ``MANTIS_AUGUR_DIR`` for explicit overrides, otherwise nests
    under ``MANTIS_DATA_DIR/augur/<run_id>/`` (with a final fallback of
    ``./data`` when ``MANTIS_DATA_DIR`` is unset).
    """
    override = os.environ.get("MANTIS_AUGUR_DIR", "").strip()
    if override:
        return Path(override) / run_id
    root = os.environ.get("MANTIS_DATA_DIR", "./data").strip() or "./data"
    return Path(root) / "augur" / run_id


def _map_step_status(step_result: Any) -> str:
    """Mantis StepResult outcome → Augur StepStatus literal."""
    if getattr(step_result, "skip", False):
        return "skipped"
    if getattr(step_result, "success", False):
        return "succeeded"
    if getattr(step_result, "reversed", False):
        return "recovered"
    return "failed"


# Mantis executor backend → Augur Grounding.provider chip text.
# ``provider`` is a free-form string the workspace surfaces in the
# GROUNDING panel; pick names that read well in the UI.
_GROUNDING_PROVIDER_MAP: dict[str, str] = {
    "som": "mantis-som-cdp",
    "vision": "mantis-xdotool-vision",
    "plan": "mantis-plan-executor",
}


def _build_grounding(
    sr: Any,
    last_action: Any,
    action_type: str,
    action_params: dict[str, Any],
) -> dict[str, Any] | None:
    """Derive an Augur Grounding dict from a Mantis StepResult (#509 fu).

    Mantis is screenshot-grounded by design — every dispatched click /
    type / scroll resolves either through the SoM CDP pipeline
    (``executor_backend="som"``) or the brain's vision grounding
    (``executor_backend="vision"`` / xdotool). Surface that on the
    trace so the per-step ``GROUNDING`` panel populates instead of
    saying "No grounding for this step."

    The Mantis runner doesn't currently stamp ``last_action`` on
    StepResult (the field exists with ``default=None`` for resume
    handling but no handler populates it). So we key grounding
    emission off ``executor_backend`` — the field that IS reliably
    set when a dispatch happens — and treat coordinates as optional.

    Returns ``None`` for steps that didn't dispatch a grounded action
    (navigate, verify, gate, extract_data, ...): keeping the Grounding
    field absent matches the SDK's ``total=False`` semantics so the
    schema validator stays happy.
    """
    backend = (getattr(sr, "executor_backend", "") or "").lower()
    # ``som`` / ``vision`` are the two backends that actually anchor a
    # coordinate on the screenshot. Empty backend = no dispatch =
    # nothing to ground (extract_data, verify, navigate, ...).
    if backend not in {"som", "vision", "plan"}:
        return None
    provider = _GROUNDING_PROVIDER_MAP.get(backend, "mantis")
    # SoM-anchored clicks resolve through a CDP element check, so they
    # carry near-certainty by construction (the element existed and
    # accepted the synthetic event); vision-grounded coordinates are
    # the brain's best guess and warrant lower confidence. Plan
    # executor (deterministic Playwright) → 1.0.
    confidence = {"som": 0.99, "vision": 0.7, "plan": 1.0}[backend]
    target_label = str(getattr(sr, "intent", "") or "")[:120] or "(no intent)"
    grounding: dict[str, Any] = {
        "provider": provider,
        "target_label": target_label,
        "confidence": confidence,
        "evidence": f"backend={backend}",
        "provenance": "screenshot",
    }
    # Coordinates are best-effort. Mantis's StepResult.last_action is
    # rarely populated today; when it is (e.g. tests, future runner
    # work) we include them so the workspace can render the click
    # overlay. Omit cleanly when absent — ``coordinates`` is optional
    # on the Augur Grounding TypedDict (total=False).
    x = action_params.get("x")
    y = action_params.get("y")
    if x is not None and y is not None:
        grounding["coordinates"] = {"x": float(x), "y": float(y)}
    return grounding


def _resolve_capture_mode(value: str | None) -> Any:
    """Coerce a string env value to a ``CaptureMode`` (or None)."""
    if not _AUGUR_AVAILABLE:
        return None
    candidate = (value or os.environ.get("AUGUR_CAPTURE_MODE") or "screenshots").strip().lower()
    try:
        return CaptureMode(candidate)
    except Exception:  # noqa: BLE001
        return CaptureMode("screenshots")


class AugurAdapter:
    """Wrap an Augur :class:`DebugSession` with non-fatal emission.

    One instance per Mantis run. Lifecycle:

    1. Constructed at the top of ``RunExecutor.execute`` once
       ``runner.run_key`` is known.
    2. :meth:`attach_observation` + :meth:`record_step` called after
       each handler returns; :meth:`drain_healing_events` flushes any
       reasoning entries the handler accumulated on
       ``runner._healing_events``.
    3. :meth:`close` called from ``RunExecutor._finalize`` with the
       runner's final status.

    When the adapter is disabled (import failed, env opt-out, or
    constructor raised), every public method is a no-op so callers
    don't need to branch on availability.
    """

    def __init__(
        self,
        *,
        run_id: str,
        tenant_id: str = "",
        session_name: str = "",
        out_dir: str | Path | None = None,
        capture_mode: str | None = None,
        dsn: str | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        self._session: Any = None
        self._emitted_event_count: int = 0
        # Verbose diagnostic — gated by ``MANTIS_AUGUR_VERBOSE``. WARN
        # is the only level that survives Modal's INFO/DEBUG suppression,
        # so when verifying a deploy operators flip the flag and get one
        # line per adapter open + per-step emission.
        _verbose = is_verbose()
        if _verbose:
            dsn_env = os.environ.get("AUGUR_DSN", "")
            logger.warning(
                "AugurAdapter init: sdk_available=%s disabled_env=%r dsn_set=%s run_id=%s",
                _AUGUR_AVAILABLE,
                os.environ.get("MANTIS_AUGUR_DISABLED", ""),
                bool(dsn_env),
                run_id,
            )
        if not is_enabled():
            if _verbose:
                logger.warning("AugurAdapter init: disabled — adapter is a no-op")
            return
        try:
            tags = {"tenant": tenant_id or "", "session": session_name or ""}
            if extra_tags:
                tags.update({str(k): str(v) for k, v in extra_tags.items()})
            target_dir = Path(out_dir) if out_dir is not None else default_out_dir(run_id)
            session = DebugSession(
                run_id=run_id,
                client_name="mantis",
                client_version=os.environ.get("MANTIS_VERSION", "") or None,
                client_git_sha=os.environ.get("MANTIS_GIT_SHA", "") or None,
                capture_mode=_resolve_capture_mode(capture_mode),
                out_dir=str(target_dir),
                dsn=dsn if dsn is not None else (os.environ.get("AUGUR_DSN") or None),
                tags=tags,
            )
            # DebugSession is designed as a context manager — ``__enter__``
            # flips the open flag and starts the streaming sink (if any).
            # We drive that explicitly because the Mantis run lifecycle
            # straddles ``RunExecutor.execute`` (open) and ``_finalize``
            # (close), neither of which is a ``with`` block.
            session.__enter__()
            self._session = session
            if _verbose:
                logger.warning(
                    "AugurAdapter init: opened successfully streaming=%s out_dir=%s",
                    session._stream is not None, target_dir,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AugurAdapter: failed to open DebugSession: %s", exc)
            self._session = None

    @property
    def active(self) -> bool:
        return self._session is not None

    @property
    def out_dir(self) -> Path | None:
        if not self.active:
            return None
        try:
            return Path(self._session.out_dir)
        except Exception:  # noqa: BLE001
            return None

    # ── Per-step emission ────────────────────────────────────────────

    def attach_observation(
        self,
        *,
        step_index: int,
        kind: str,
        png: bytes | None,
    ) -> str | None:
        """Stage a screenshot. Returns the bundle-relative path or None.

        ``kind`` is ``"pre"`` or ``"post"`` per the SDK convention.
        Callers pass Mantis's 0-based ``StepResult.step_index``; the
        adapter bumps to 1-based on the way into the SDK so the path
        / URL match :meth:`_build_step_trace`'s 1-based numbering
        (Augur's schema requires ``step_index >= 1``). Returns the
        relative path the SDK assigned (which the caller passes back
        as ``observation_pre`` / ``observation_post`` on the StepTrace)
        or None on any failure / no PNG.
        """
        if not self.active or not png:
            return None
        try:
            return self._session.attach_observation(
                step_index=step_index + 1, kind=kind, png_bytes=png,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("AugurAdapter.attach_observation failed: %s", exc)
            return None

    def record_step(
        self,
        *,
        step_result: Any,
        started_at: str = "",
        ended_at: str = "",
        observation_pre: str | None = None,
        observation_post: str | None = None,
        step_type: str = "",
        costs: dict[str, Any] | None = None,
        latency: dict[str, Any] | None = None,
    ) -> None:
        """Emit a StepTrace for a completed step.

        ``step_type`` is the originating ``MicroIntent.type`` (the kind
        of step the plan dispatched — ``click``, ``navigate``,
        ``extract_data``, ...). Surfacing it as ``action.type`` keeps
        the workspace's ACTION DISPATCHED panel meaningful even when
        the runner doesn't stamp ``StepResult.last_action`` (which is
        the common case today — Mantis's runner sets ``last_action``
        only on a few code paths).

        ``costs`` (#518) populates the augur-sdk 0.1.6+
        :ref:`StepTrace.costs` field — a typed object the workspace
        uses to render the per-step COST in the step inspector.
        Schema keys: ``total_usd``, ``model_usd``, ``gpu_usd``,
        ``proxy_usd``, ``tokens_in``, ``tokens_out``,
        ``cache_hit_tokens``. Pass ``None`` (or omit) when the
        step didn't track cost.

        ``latency`` (also augur-sdk 0.1.6+) populates
        :ref:`StepTrace.latency` — per-layer ms breakdown
        (``planner_ms`` / ``grounding_ms`` / ``dispatch_ms`` /
        ``verifier_ms`` / ``recovery_ms`` / ``total_ms``) the
        workspace renders alongside costs. Same conventions:
        pass ``None`` to omit; zero-value keys are stripped.
        """
        if not self.active:
            return
        try:
            trace = self._build_step_trace(
                step_result, started_at, ended_at,
                observation_pre, observation_post,
                step_type=step_type,
                costs=costs,
                latency=latency,
            )
            self._session.record_step(trace)
        except Exception as exc:  # noqa: BLE001
            # Verbose deploys want this visible; production stays at debug.
            if is_verbose():
                logger.warning("AugurAdapter.record_step failed: %r", exc)
            else:
                logger.debug("AugurAdapter.record_step failed: %s", exc)

    def add_tag(self, key: str, value: str) -> None:
        """Attach a tag to the open DebugSession (#509).

        The Augur workspace's Runs-list ``MODEL`` column and similar
        derived chips read from session tags. Idempotent + safe to
        call multiple times — last write wins per key.
        """
        if not self.active or not key:
            return
        try:
            self._session.add_tag(str(key), str(value))
        except Exception as exc:  # noqa: BLE001
            if is_verbose():
                logger.warning("AugurAdapter.add_tag(%s) failed: %r", key, exc)
            else:
                logger.debug("AugurAdapter.add_tag(%s) failed: %s", key, exc)

    def append_log(
        self,
        text: str,
        *,
        step_index: int | None = None,
        name: str = "run",
    ) -> None:
        """Stream a log chunk to the workspace's ``logs/`` panel.

        Thin wrapper over ``DebugSession.append_log`` (augur-sdk 0.1.3+):
        when streaming is enabled, POSTs to ``/api/v1/runs/<id>/logs``
        which the server appends to ``logs/<name>.log`` (or
        ``logs/step-<idx>.log`` when ``step_index`` is set). When the
        SDK installed is older than 0.1.3, OR when streaming is off
        (no ``AUGUR_DSN``), this is a clean no-op — silently dropping
        the log chunk so callers never have to feature-check.

        Use for runner / handler progress lines you'd want to scrub
        through in the Augur viewer alongside the per-step trace.
        Adapter-side ``logger.debug`` already covers internal events
        — don't double-emit.
        """
        if not self.active or not text:
            return
        # SDK 0.1.2 doesn't have ``append_log``; guard so a stale
        # install in the executor image doesn't crash the wedge.
        append = getattr(self._session, "append_log", None)
        if append is None:
            return
        try:
            append(text, step_index=step_index, name=name)
        except Exception as exc:  # noqa: BLE001
            if is_verbose():
                logger.warning("AugurAdapter.append_log failed: %r", exc)
            else:
                logger.debug("AugurAdapter.append_log failed: %s", exc)

    def record_cost_metric(
        self,
        *,
        name: str,
        value: float,
        detail: dict[str, Any] | None = None,
        step_index: int | None = None,
    ) -> None:
        """Emit a metric DecisionEvent on the open session (#509 / #518).

        Mantis's :class:`gym.cost_meter.CostMeter` tracks per-bucket
        and total USD spend. Surfacing them as Augur ``kind="metric"``
        events lets the workspace's COST column populate and gives
        cost-vs-time analyses something to query against. Layer is
        ``runner`` (the Mantis runner owns the meter).

        ``step_index=None`` (default) treats the metric as run-level
        and parks it at the minimum allowed index (``1``). Pass an
        explicit 0-based Mantis step index for per-step cost deltas
        (#518) — the adapter bumps to 1-based for the Augur schema
        the same way :meth:`_build_step_trace` does.
        """
        if not self.active:
            return
        payload = {"name": str(name), "value": float(value)}
        if detail:
            payload.update(detail)
        augur_step_index = (
            int(step_index) + 1 if step_index is not None else 1
        )
        event: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "step_index": augur_step_index,
            "layer": "runner",
            "kind": "metric",
            "summary": f"{name}={value}",
            "detail": payload,
        }
        try:
            self._session.record_event(event)
        except Exception as exc:  # noqa: BLE001
            if is_verbose():
                logger.warning("AugurAdapter.record_cost_metric failed: %r", exc)
            else:
                logger.debug("AugurAdapter.record_cost_metric failed: %s", exc)

    def record_planner_reasoning(
        self, *, step_index: int, reasoning: str,
    ) -> None:
        """Emit the brain's reasoning for one step as a planner-layer
        DecisionEvent (#509).

        Mantis's :attr:`StepResult.reasoning` carries the planner /
        brain text that produced this step. The Augur workspace shows
        it in the per-step "PLANNER REASONING" panel — without this
        method it falls back to demo placeholder text. The ``summary``
        is a short prefix (≤200 chars); the full text lands on
        ``detail.text`` so the workspace can render long reasoning
        without truncation.
        """
        if not self.active or not reasoning.strip():
            return
        text = reasoning.strip()
        event: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "step_index": step_index + 1,
            "layer": "planner",
            "kind": "info",
            "summary": text[:200],
            "detail": {"text": text},
        }
        try:
            self._session.record_event(event)
        except Exception as exc:  # noqa: BLE001
            if is_verbose():
                logger.warning("AugurAdapter.record_planner_reasoning failed: %r", exc)
            else:
                logger.debug("AugurAdapter.record_planner_reasoning failed: %s", exc)

    def drain_healing_events(self, healing_events: list[dict[str, Any]]) -> None:
        """Emit any healing events accumulated past the last cursor.

        The runner appends to ``_healing_events`` from many sites
        (critic, recovery, som-click); this method emits each entry as
        a DecisionEvent. The cursor lives on the adapter so a single
        call per step picks up exactly the new entries.
        """
        if not self.active or not healing_events:
            return
        try:
            new_slice = healing_events[self._emitted_event_count:]
            for ev in new_slice:
                self._record_decision_event(ev)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AugurAdapter.drain_healing_events failed: %s", exc)
        # Advance the cursor even on partial failure so we don't busy-
        # retry the same entries on the next step.
        self._emitted_event_count = len(healing_events)

    def set_live_endpoints(
        self,
        *,
        video_url: str | None = None,
        status_url: str | None = None,
        reasoning_url: str | None = None,
    ) -> None:
        if not self.active:
            return
        try:
            self._session.set_live_endpoints(
                status_url=status_url,
                video_url=video_url,
                reasoning_url=reasoning_url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("AugurAdapter.set_live_endpoints failed: %s", exc)

    def close(self, status: str | None = None) -> Any:
        """Flush the bundle. Returns the BundleManifest or None.

        Mirrors :meth:`DebugSession.__exit__` so the explicit
        ``__enter__`` we ran in :meth:`__init__` gets its matching
        teardown. Setting ``status`` overrides the SDK's inferred
        run status (otherwise inferred as ``succeeded`` if nothing
        called :meth:`DebugSession.set_status`).
        """
        if not self.active:
            return None
        try:
            if status is not None:
                self._session.set_status(_RUN_STATUS_MAP.get(status, "halted"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("AugurAdapter.close set_status failed: %s", exc)
        try:
            manifest = self._session.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("AugurAdapter.close failed: %s", exc)
            manifest = None
        self._session = None
        return manifest

    # ── Mappers ──────────────────────────────────────────────────────

    def _build_step_trace(
        self,
        sr: Any,
        started_at: str,
        ended_at: str,
        observation_pre: str | None,
        observation_post: str | None,
        *,
        step_type: str = "",
        costs: dict[str, Any] | None = None,
        latency: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_action = getattr(sr, "last_action", None)
        action_type = ""
        action_params: dict[str, Any] = {}
        if last_action is not None:
            at = getattr(last_action, "action_type", None)
            action_type = (
                at.value if hasattr(at, "value")
                else (str(at) if at is not None else "")
            )
            # Mantis's :class:`actions.Action` is a dataclass with a
            # ``params`` dict — click coords / type text / scroll
            # deltas live inside ``params``, NOT as top-level attrs.
            # Earlier the wedge used ``getattr(last_action, "x", None)``
            # which always returned None, so grounding never fired
            # because no x/y reached the workspace. Read params first;
            # fall back to top-level attrs for compatibility with any
            # adapter test fixture that uses a plain SimpleNamespace.
            la_params = getattr(last_action, "params", None) or {}
            for attr in ("text", "selector", "key", "x", "y", "dx", "dy", "url"):
                if isinstance(la_params, dict) and la_params.get(attr) not in (None, ""):
                    action_params[attr] = la_params[attr]
                    continue
                v = getattr(last_action, attr, None)
                if v not in (None, ""):
                    action_params[attr] = v

        # Prefer the MicroIntent's ``step.type`` when ``last_action``
        # isn't stamped (the common case) so the workspace's ACTION
        # DISPATCHED panel shows ``click`` / ``navigate`` / etc.
        # instead of ``unknown``.
        action: dict[str, Any] = {
            "type": action_type or step_type or "unknown",
            "params": action_params,
            "coordinate_space": "screenshot_px",
            "dispatch_backend": getattr(sr, "executor_backend", "") or "",
        }

        grounding = _build_grounding(sr, last_action, action_type or step_type, action_params)

        # Verdict: prefer the typed slot, fall back to success boolean.
        v_obj = getattr(sr, "verdict", None)
        if v_obj is not None:
            v_status = getattr(v_obj, "status", "")
            v_status = (
                v_status.value if hasattr(v_status, "value") else str(v_status)
            )
            verdict: dict[str, Any] = {
                "status": _VERDICT_STATUS_MAP.get(v_status, "unknown"),
                "reason": getattr(v_obj, "reason", "") or "",
                "evidence_refs": [],
            }
        else:
            verdict = {
                "status": "passed" if getattr(sr, "success", False) else "failed",
                "reason": getattr(sr, "failure_class", "") or "",
                "evidence_refs": [],
            }

        rd_obj = getattr(sr, "recovery_decision", None)
        recovery_decision: dict[str, Any] | None = None
        if rd_obj is not None:
            rd_type = getattr(rd_obj, "type", "")
            rd_type = (
                rd_type.value if hasattr(rd_type, "value") else str(rd_type)
            )
            # Augur's schema requires ``attempt >= 1``. Mantis's
            # RecoveryDecision.attempt convention isn't strictly
            # 0-based or 1-based across handlers — clamp to >=1 on
            # the way out so server-side validation passes without
            # us guessing semantics. The server's error message for
            # a violation is the misleading
            # ``"step: 0 is less than the minimum of 1"`` (it reports
            # the field VALUE, not the field PATH), which is what
            # actually surfaced during the #509 verification cycle.
            recovery_decision = {
                "type": _RECOVERY_TYPE_MAP.get(rd_type, "none"),
                "reason": getattr(rd_obj, "reason", "") or "",
                "attempt": max(1, int(getattr(rd_obj, "attempt", 0) or 0)),
            }

        # All optional TypedDict fields (total=False) must be either
        # absent or non-empty for the schema validator. Build the
        # required-only base, then conditionally add the optionals.
        #
        # Augur's step_index schema requires ``>= 1`` (1-based), but
        # Mantis's StepResult.step_index is 0-based. Bump by 1 on the
        # way out so server-side validation passes — the server returns
        # 422 ``"step: 0 is less than the minimum of 1"`` otherwise,
        # silently dropping every per-step PUT. The ``step_id`` string
        # uses the 1-based index too so it lines up with the URL path.
        raw_index = int(getattr(sr, "step_index", 0))
        augur_index = raw_index + 1
        trace: dict[str, Any] = {
            "step_id": f"step-{augur_index:04d}",
            "step_index": augur_index,
            "intent": str(getattr(sr, "intent", "") or "")[:500] or "(no intent)",
            "step_type": action_type or step_type or "unknown",
            "required": True,
            "status": _map_step_status(sr),
            "started_at": started_at or "",
            "ended_at": ended_at or started_at or "",
            "duration_ms": int((getattr(sr, "duration", 0.0) or 0.0) * 1000),
            "action": action,
            "verdict": verdict,
            "events": [],
            "logs": [],
        }
        # Run the canonical classifier when the StepResult didn't carry
        # a class (or carried the placeholder "unknown"). Closes
        # mercurialsolo/augur#49 — without this, every halt streamed
        # via DSN landed as failure_class="unknown" because the SDK
        # streaming path didn't run classify() the way the file-export
        # path does.
        failure_class = str(getattr(sr, "failure_class", "") or "")
        if not failure_class or failure_class == "unknown":
            from ..gym.failure_class import classify

            page_title = str(getattr(sr, "page_title", "") or "")
            data = str(getattr(sr, "data", "") or "")
            derived = classify(data, page_title)
            if derived and derived != "unknown":
                failure_class = derived
        if failure_class and failure_class != "unknown":
            trace["failure_class"] = failure_class
        if grounding is not None:
            trace["grounding"] = grounding
        if observation_pre:
            trace["observation_pre"] = observation_pre
        if observation_post:
            trace["observation_post"] = observation_post
        if recovery_decision is not None:
            trace["recovery_decision"] = recovery_decision
        # #518 — populate the augur-sdk 0.1.6+ ``StepTrace.costs`` field
        # so the workspace step-inspector renders per-step USD spend
        # next to its intent + verdict. Only emit when the producer
        # captured non-zero numbers (avoid noisy zero-cost entries
        # for deterministic navigate / verify steps).
        if costs and any(float(v or 0) > 0 for v in costs.values()):
            trace["costs"] = {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in costs.items()
                if v not in (None, 0, 0.0)
            }
        # #518 — same shape as ``costs`` but for the 0.1.6+
        # ``StepTrace.latency`` field. Per-layer ms breakdown derived
        # from Mantis's TimeMeter buckets.
        if latency and any(float(v or 0) > 0 for v in latency.values()):
            trace["latency"] = {
                k: int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in latency.items()
                if v not in (None, 0, 0.0)
            }
        return trace

    def _record_decision_event(self, ev: dict[str, Any]) -> None:
        layer_in = str(ev.get("layer") or "runner")
        layer = _LAYER_MAP.get(layer_in, "runner")
        kind_in = str(ev.get("kind") or "decision")
        kind = _KIND_MAP.get(kind_in, "decision")
        detail = ev.get("detail")
        if not isinstance(detail, dict):
            detail = {"raw": detail} if detail is not None else {}
        # Match the 1-based ``step_index`` convention used by
        # :meth:`_build_step_trace` so DecisionEvents line up with
        # StepTraces in the workspace timeline. ``ts`` is required by
        # the SDK's EventRecorder; default to UTC-now if the upstream
        # Mantis healing event didn't carry a timestamp.
        raw_index = int(ev.get("step_index") or 0)
        event: dict[str, Any] = {
            "ts": str(ev.get("ts") or "") or _utc_now_iso(),
            "step_index": raw_index + 1,
            "layer": layer,
            "kind": kind,
            "summary": str(ev.get("summary") or "")[:200],
            "detail": detail,
        }
        self._session.record_event(event)
