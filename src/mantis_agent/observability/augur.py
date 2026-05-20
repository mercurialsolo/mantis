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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — module must load even without augur-sdk installed.
try:
    from augur_sdk import CaptureMode, DebugSession
    _AUGUR_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import-time failure → disabled
    CaptureMode = None  # type: ignore[assignment]
    DebugSession = None  # type: ignore[assignment]
    _AUGUR_AVAILABLE = False


def _patch_streaming_for_visibility() -> None:
    """Temporary diagnostic — elevate SDK streaming errors to WARN.

    augur-sdk's ``StreamingSink._post_json`` / ``_post_multipart`` /
    ``_safe_call`` log at DEBUG when the server returns >=400 or when a
    background-thread HTTP call raises. Modal suppresses DEBUG, so a
    server rejection on per-step PUT (the exact #509 verification gap)
    is invisible in ``modal app logs``. Patch the three methods to
    emit at WARN instead. Remove once #509 is operationally stable.
    """
    if not _AUGUR_AVAILABLE:
        return
    try:
        import augur_sdk.streaming as _stream  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return
    if getattr(_stream, "_mantis_patched", False):
        return
    sink_cls = _stream.StreamingSink

    def _verbose_post_json(self, path, payload, *, method="POST"):  # noqa: ANN001
        import json
        url = self.dsn.base_url + path
        body = json.dumps(payload).encode("utf-8")
        resp = self._http.request(
            method, url, body=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.dsn.token}",
            },
        )
        if resp.status >= 400:
            logger.warning(
                "augur stream %s %s -> %s %s",
                method, path, resp.status, resp.data[:200],
            )

    def _verbose_post_multipart(self, path, payload):  # noqa: ANN001
        url = self.dsn.base_url + path
        resp = self._http.request(
            "POST", url,
            fields={"image": ("shot.png", payload, "image/png")},
            headers={"Authorization": f"Bearer {self.dsn.token}"},
        )
        if resp.status >= 400:
            logger.warning(
                "augur stream upload %s -> %s %s",
                path, resp.status, resp.data[:200],
            )

    def _verbose_safe_call(self, fn):  # noqa: ANN001
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("augur stream background error: %r", exc)

    sink_cls._post_json = _verbose_post_json
    sink_cls._post_multipart = _verbose_post_multipart
    sink_cls._safe_call = _verbose_safe_call
    _stream._mantis_patched = True


_patch_streaming_for_visibility()


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
        # WARNING-level so it survives Modal log suppression — one line per
        # run, gates downstream production debugging when the workspace
        # shows no client. Remove once #509 is operationally stable.
        dsn_env = os.environ.get("AUGUR_DSN", "")
        logger.warning(
            "AugurAdapter init: sdk_available=%s disabled_env=%r dsn_set=%s run_id=%s",
            _AUGUR_AVAILABLE,
            os.environ.get("MANTIS_AUGUR_DISABLED", ""),
            bool(dsn_env),
            run_id,
        )
        if not is_enabled():
            logger.warning("AugurAdapter init: disabled — adapter is a no-op")
            return
        try:
            tags = {"tenant": tenant_id or "", "session": session_name or ""}
            if extra_tags:
                tags.update({str(k): str(v) for k, v in extra_tags.items()})
            target_dir = Path(out_dir) if out_dir is not None else default_out_dir(run_id)
            logger.warning(
                "AugurAdapter init: opening DebugSession out_dir=%s dsn_host=%s",
                target_dir,
                dsn_env.split("@")[-1].split("/")[0] if "@" in dsn_env else "(no dsn)",
            )
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
            logger.warning(
                "AugurAdapter init: opened successfully streaming=%s",
                session._stream is not None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AugurAdapter init: failed to open DebugSession: %s", exc)
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
    ) -> None:
        """Emit a StepTrace for a completed step."""
        if not self.active:
            return
        try:
            trace = self._build_step_trace(
                step_result, started_at, ended_at,
                observation_pre, observation_post,
            )
            self._session.record_step(trace)
        except Exception as exc:  # noqa: BLE001
            # Temporarily elevated from debug → warning to surface
            # silent put_step failures in Modal logs while #509 is
            # being verified end-to-end. Revert to ``debug`` once
            # the workspace timeline reliably populates.
            logger.warning("AugurAdapter.record_step failed: %r", exc)

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
            for attr in ("text", "selector", "key", "x", "y", "dx", "dy", "url"):
                v = getattr(last_action, attr, None)
                if v not in (None, ""):
                    action_params[attr] = v

        action: dict[str, Any] = {
            "type": action_type or "unknown",
            "params": action_params,
            "coordinate_space": "screenshot_px",
            "dispatch_backend": getattr(sr, "executor_backend", "") or "",
        }

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
            "step_type": action_type or "unknown",
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
        failure_class = str(getattr(sr, "failure_class", "") or "")
        if failure_class:
            trace["failure_class"] = failure_class
        if observation_pre:
            trace["observation_pre"] = observation_pre
        if observation_post:
            trace["observation_post"] = observation_post
        if recovery_decision is not None:
            trace["recovery_decision"] = recovery_decision
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
        # StepTraces in the workspace timeline.
        raw_index = int(ev.get("step_index") or 0)
        event: dict[str, Any] = {
            "ts": str(ev.get("ts") or ""),
            "step_index": raw_index + 1,
            "layer": layer,
            "kind": kind,
            "summary": str(ev.get("summary") or "")[:200],
            "detail": detail,
        }
        self._session.record_event(event)
