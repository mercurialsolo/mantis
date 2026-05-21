"""External pause + auto-pause-on-CAPTCHA for human takeover (#540 fu).

Two trigger paths for the runner to pause mid-run AND keep the browser
+ noVNC viewer alive so a human can take over via the live viewer:

1. **External pause** — the API container writes ``pause_request.json``
   to the shared run dir (via ``action=pause`` HTTP). The runner polls
   for this between steps; when present, it enters a sleep loop until
   the file disappears (``action=resume`` deletes it). Critically the
   executor process keeps running — Chrome stays up, the noVNC tunnel
   stays up, the user has full mouse/keyboard via the viewer URL.

2. **Auto-pause on CAPTCHA** — when a step fails with
   ``failure_class='cf_challenge'`` (Cloudflare Turnstile or similar)
   and ``MANTIS_PAUSE_ON_CAPTCHA`` is enabled (default), the runner
   self-triggers an external pause and waits for human takeover
   instead of looping into a recovery halt. Same sleep-loop mechanics
   as the external trigger.

Both rely on a single sentinel file location stashed in
:data:`_REQUEST_PATH` at executor startup via :func:`init_paths`. The
runner reads it via :func:`is_pause_requested` / :func:`wait_while_paused`.
The API container writes / clears it via :func:`request_pause` /
:func:`clear_pause_request` (the latter is what ``action=resume`` does).

When no sentinel path has been wired (local CLI runs, tests), all
helpers are no-ops — the runner proceeds normally.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# Module-level sentinel path, set by ``init_paths`` at executor startup.
# ``None`` → external pause disabled (local CLI / tests). Sentinel is a
# plain JSON file holding ``{"reason": str, "requested_at": ISO-8601}``.
_REQUEST_PATH: Path | None = None


def _noop_reload() -> None:
    """Default reload callback — a no-op for local CLI runs / tests
    where POSIX stat is already coherent across processes."""
    return None


# Optional callback that invalidates the host filesystem cache so
# subsequent ``stat()`` calls see writes from other processes /
# containers. On Modal, this is ``vol.reload``; on a local CLI run
# the default no-op is correct because POSIX stat is already coherent.
# Wired by ``init_paths``; called by ``is_pause_requested`` only when
# the sentinel was previously cached as "exists" so we don't pay the
# reload tax on every poll of a healthy run.
_RELOAD_CB: Callable[[], None] = _noop_reload


def init_paths(
    pause_request_path: str | os.PathLike[str],
    reload_cb: Callable[[], None] | None = None,
) -> None:
    """Wire the sentinel path. Call once at executor startup before the
    runner loop begins.

    The Modal cua-server's ``_run_holo3_executor`` (and the parallel
    Claude/Gemma4 executors) call this with
    ``_run_dir(api_tenant_id, api_run_id) / "pause_request.json"`` so
    the API container and the executor share the same sentinel.

    ``reload_cb`` lets the caller wire a function that invalidates the
    host's volume cache so the executor sees ``pause_request.json``
    deletions written by another container. On Modal this is
    ``vol.reload`` — without it the executor's snapshot of the volume
    can keep returning ``exists() == True`` after the API container
    cleared the sentinel via ``action=resume``, causing
    ``wait_while_paused`` to loop until ``max_seconds`` (default 30
    min) expires even though the user clicked Resume immediately.
    Live repro: the viewer "Resume" button stayed on "Resume" after
    click because the executor never woke up, so subsequent
    ``/api/run_state`` polls kept returning ``status=paused``.

    Default is a no-op for local CLI / unit tests where POSIX stat
    is already coherent across processes.
    """
    global _REQUEST_PATH, _RELOAD_CB
    _REQUEST_PATH = Path(pause_request_path)
    if reload_cb is not None:
        _RELOAD_CB = reload_cb


def is_pause_requested() -> bool:
    """True when the sentinel file exists. Cheap stat() — safe to call
    inside the per-step loop on every iteration.

    Volume-staleness defence: when the cached stat says the sentinel
    exists, invoke the reload callback (see ``init_paths``) and
    re-stat. Catches the case where the API container deleted the
    sentinel via ``action=resume`` but the executor's volume snapshot
    is stale and still serves a hit. The reload cost is only paid
    while we're actively in a pause loop — healthy runs never trip
    the inner branch.
    """
    if _REQUEST_PATH is None:
        return False
    try:
        if not _REQUEST_PATH.exists():
            return False
        # Cached state says "exists". Reload before trusting it so
        # we see external deletions.
        try:
            _RELOAD_CB()
        except Exception:  # noqa: BLE001 — reload failure must not block resume
            logger.debug("external_pause: reload_cb failed", exc_info=True)
        return _REQUEST_PATH.exists()
    except OSError:
        return False


def read_pause_reason() -> str:
    """Return the reason from the sentinel file, or ``""`` if absent
    or unreadable. Used by the status synthesizer to surface why a
    run is paused (``"external"`` / ``"cf_challenge"`` / etc.)."""
    if _REQUEST_PATH is None or not _REQUEST_PATH.exists():
        return ""
    try:
        data = json.loads(_REQUEST_PATH.read_text())
        return str(data.get("reason", "") or "")
    except (OSError, json.JSONDecodeError):
        return ""


def request_pause(reason: str = "external") -> bool:
    """Write the sentinel file. Returns True when written, False when
    no sentinel path was wired (local CLI). Safe to call from both
    the API container (external trigger) and the runner itself
    (auto-pause-on-captcha)."""
    if _REQUEST_PATH is None:
        return False
    try:
        _REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REQUEST_PATH.write_text(json.dumps({
            "reason": reason,
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        logger.warning("external_pause: request written (reason=%s)", reason)
        return True
    except OSError as exc:
        logger.warning("external_pause: write failed: %s", exc)
        return False


def clear_pause_request() -> bool:
    """Delete the sentinel file. Returns True when deleted (or when
    the file didn't exist — both are success). Called by ``action=resume``
    via the API; also called by the runner just before re-entering the
    step loop to make sure stale sentinels don't auto-pause the next
    step."""
    if _REQUEST_PATH is None:
        return False
    try:
        _REQUEST_PATH.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("external_pause: clear failed: %s", exc)
        return False


def wait_while_paused(*, max_seconds: int = 1800, poll_seconds: float = 2.0) -> str:
    """Block while the sentinel file exists. Returns the reason the
    pause ended:

    * ``"resumed"`` — sentinel cleared externally (``action=resume``);
      run continues from where it was paused
    * ``"timeout"`` — exceeded ``max_seconds`` (default 30 min) without
      clearance; caller should halt rather than block forever
    * ``"not_paused"`` — sentinel wasn't set when we checked (returns
      immediately; caller proceeds normally)

    The 30-minute default gives generous headroom for a human to open
    the viewer, solve the CAPTCHA, and click resume. Modal function
    calls have hour-scale timeouts so this isn't constrained by the
    platform.

    During the wait, the executor process stays alive (Chrome stays
    up, noVNC tunnel stays up), so the user has full interactive
    control via the live-viewer URL.
    """
    if not is_pause_requested():
        return "not_paused"
    start = time.time()
    initial_reason = read_pause_reason()
    logger.warning(
        "external_pause: entering wait loop (reason=%s, max_seconds=%d)",
        initial_reason, max_seconds,
    )
    while is_pause_requested():
        if time.time() - start > max_seconds:
            logger.warning(
                "external_pause: timeout after %ds (reason=%s); resuming run",
                max_seconds, initial_reason,
            )
            # Clear the sentinel ourselves so the next iteration
            # doesn't re-pause immediately on a stale file.
            clear_pause_request()
            return "timeout"
        time.sleep(poll_seconds)
    logger.warning(
        "external_pause: resumed after %.1fs (reason=%s)",
        time.time() - start, initial_reason,
    )
    return "resumed"


def is_captcha_autopause_enabled() -> bool:
    """Whether to auto-pause when a step fails with
    ``failure_class='cf_challenge'``. Default ``True``; set
    ``MANTIS_PAUSE_ON_CAPTCHA=0`` to fall back to legacy halt-on-
    cf_challenge behavior."""
    raw = os.environ.get("MANTIS_PAUSE_ON_CAPTCHA", "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


__all__ = [
    "clear_pause_request",
    "init_paths",
    "is_captcha_autopause_enabled",
    "is_pause_requested",
    "read_pause_reason",
    "request_pause",
    "wait_while_paused",
]
