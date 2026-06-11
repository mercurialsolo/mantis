"""Pydantic wire models for the Session Router RPC (Phase 1.5, #846).

The session router sits in front of the Modal computer plane. Instead
of every brain pointing at one pinned ``computer_plane()`` ASGI app
(``min_containers=1, max_containers=1``), each run mints a *session*
that owns a dedicated computer-plane sandbox/container for its
lifetime. The router records that mapping cross-replica so polls and
cleanups stay coherent.

This module defines the wire contract; the orchestrator (PR 2) and the
brain-side resolver (PR 3) import from it.

Layering vs. ``computer_wire.py``:

* ``computer_wire`` — what the *brain → computer plane* RPC looks like
  (screenshot, xdotool, cdp). Per-step. Unchanged by Phase 1.5.
* ``session_wire`` — what the *brain → session router → orchestrator*
  RPC looks like (create-session, close-session). Once per run.

The two contracts deliberately don't share fields so a router-side
change can't ripple into the per-step wire.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Create session ────────────────────────────────────────────────────


class SessionCreateRequest(BaseModel):
    """Ask the router to allocate a new computer-plane session.

    The brain calls this once at the start of a run. The router
    spawns/leases a dedicated computer-plane container, waits for it
    to be reachable, and returns the per-session base URL the brain
    uses for the rest of the run.

    Identity (``tenant_id`` + ``profile_id`` + ``run_id``) is required
    so the router can:

    1. Stamp the per-session lock against the same ``(tenant, profile)``
       the API-side ``acquire_profile_lock`` would use.
    2. Look up an existing session by ``run_id`` for idempotent
       create-on-retry semantics.
    3. Surface the session to the reaper keyed by ``run_id`` (so a
       crashed brain can be cleaned up without the router holding
       its own out-of-band registry).

    Everything else mirrors the ``computer_wire.SessionInitRequest``
    shape so the orchestrator can forward fields straight through to
    ``ComputerAgent.session/init`` without translation.
    """

    tenant_id: str = Field(..., min_length=1, max_length=128)
    profile_id: str = Field(..., min_length=1, max_length=128)
    run_id: str = Field(..., min_length=1, max_length=128)

    # Forwarded to ComputerAgent.session/init verbatim.
    start_url: str = Field(default="about:blank", max_length=2048)
    viewport: tuple[int, int] = Field(default=(1280, 720))
    proxy_server: str = Field(default="", max_length=512)
    profile_dir: Optional[str] = Field(default=None, max_length=512)
    extra_http_headers: Optional[dict[str, str]] = None
    enable_cdp: bool = False

    # Router-only. TTL after which the reaper considers this session
    # orphaned and terminates the underlying container. Brain-supplied
    # so a long-running plan can opt into a longer TTL; clamped server-
    # side against a deployment cap.
    ttl_seconds: int = Field(default=3600, ge=60, le=14400)


class SessionCreateResponse(BaseModel):
    """Reply once the orchestrator has a reachable session.

    Carries the per-session ``base_url`` the brain plugs into
    ``RemoteComputerImpl(base_url=...)``. ``session_token`` is
    pre-minted by the router so the brain doesn't have to do a
    separate ``ComputerAgent.session/init`` hop — the orchestrator
    has already issued the token against the bound session.

    ``sandbox_id`` is the Modal sandbox / function-call handle the
    router uses to terminate the session. Exposed so an operator can
    correlate logs; opaque to the brain.
    """

    session_id: str
    base_url: str
    session_token: str
    expires_at_ms: int
    sandbox_id: str = ""

    # Idempotent-create signal — ``true`` means the router returned
    # a pre-existing session for this ``(tenant, profile, run_id)``
    # tuple instead of spawning a new one.
    reused: bool = False


# ── Close session ─────────────────────────────────────────────────────


class SessionCloseRequest(BaseModel):
    """Tear down a session and release the underlying container."""

    session_id: str = Field(..., min_length=1, max_length=128)
    # Optional — surfaces in observability for "why did this session
    # end?" without grepping the runner's terminal status.
    reason: str = Field(default="brain_closed", max_length=64)


class SessionCloseResponse(BaseModel):
    closed: bool
    # Echo of ``reason`` plus the terminal status the router observed
    # for the underlying container — useful when the brain calls
    # close on an already-terminated session.
    terminal_state: str = ""


# ── Diagnostics: list sessions ────────────────────────────────────────


class SessionRecord(BaseModel):
    """One entry in ``GET /v1/computer_sessions`` and the persisted
    ``KIND_SESSION`` blob in :class:`RunStateStore`.

    Reaper + router both read this shape; keep additions backwards-
    compatible so a stale on-disk record still parses.
    """

    session_id: str
    tenant_id: str
    profile_id: str
    run_id: str
    base_url: str
    sandbox_id: str = ""
    session_token: str = ""
    created_at_ms: int
    expires_at_ms: int
    last_action_ms: int = 0
    status: str = "active"  # active | closed | reaped | error
    error: str = ""


class SessionListResponse(BaseModel):
    """Reply to ``GET /v1/computer_sessions`` (operator/diagnostic)."""

    sessions: list[SessionRecord]
    count: int


# ── Errors ────────────────────────────────────────────────────────────


# Distinct exception types the router raises; the FastAPI shell maps
# them to HTTP status codes. Defined here so PR 2's orchestrator and
# PR 3's brain-side resolver can both import + react to them without
# pulling in FastAPI.


class SessionRouterError(Exception):
    """Base — never raised directly; always one of the subclasses."""

    http_status: int = 500


class SessionNotFoundError(SessionRouterError):
    http_status = 404


class SessionUnreachableError(SessionRouterError):
    """Orchestrator spawned the container but it never became
    reachable before timeout."""

    http_status = 504


class SessionQuotaExceededError(SessionRouterError):
    """Tenant hit a per-tenant active-session cap. Distinct from a
    profile-lock 409 — quota is across all profiles."""

    http_status = 429


# ── Store helpers ─────────────────────────────────────────────────────


# Wire-contract namespace. The router/orchestrator + reaper both use
# these to read and write SessionRecord blobs into the cross-replica
# :class:`run_state_store.RunStateStore`. Defined here so the
# brain-side resolver can read sessions without importing the
# orchestrator (which would pull in Modal SDK).


def parse_session_record(blob: dict | None) -> SessionRecord | None:
    """Permissive parse — None or shape-mismatched blob → None.

    Reaper iterates many records; one corrupt blob shouldn't poison
    the whole sweep. The reaper falls back to the underlying string
    representation for logging.
    """
    if not isinstance(blob, dict):
        return None
    try:
        return SessionRecord.model_validate(blob)
    except Exception:  # noqa: BLE001 — store fault ≠ run fault
        return None


def serialize_session_record(record: SessionRecord) -> dict:
    """Round-trip-safe dict for storage. Always JSON-friendly."""
    return record.model_dump(mode="json")


def iter_session_records(
    store, *, tenant_id: str | None = None,
):
    """Yield ``SessionRecord`` for each persisted session.

    ``store`` is a :class:`run_state_store.RunStateStore` (or any
    backing whose ``_d.keys()`` yields the canonical key strings).
    Unparseable entries are skipped with no exception so a reaper
    sweep can't be derailed by one bad record.

    ``tenant_id`` filter is best-effort — based on the key prefix
    sanitization done at write time. ``None`` (default) yields all
    tenants.
    """
    from .run_state_store import KIND_SESSION, list_active_keys

    suffix = f"/{KIND_SESSION}"
    prefix = ""
    if tenant_id:
        # Cheap-and-cheerful prefix match — mirrors run_state_store's
        # ``_safe`` sanitization (alnum + dash/underscore preserved).
        prefix = "".join(
            c if c.isalnum() or c in {"-", "_"} else "_" for c in tenant_id
        ) + "/"

    for key in list_active_keys(store):
        if not isinstance(key, str) or not key.endswith(suffix):
            continue
        if prefix and not key.startswith(prefix):
            continue
        # Reconstruct (tenant_id, session_id, kind) from the key shape
        # ``<tenant>/<session_id>/<kind>``. ``rsplit`` is enough — the
        # tenant slot can never contain "/" after sanitization.
        try:
            tenant_seg, sid, _kind = key.rsplit("/", 2)
        except ValueError:
            continue
        blob = store.get(tenant_seg, sid, KIND_SESSION)
        record = parse_session_record(blob)
        if record is not None:
            yield record
