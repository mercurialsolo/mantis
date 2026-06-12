"""Hot-mode snapshot helper — SIGSTOP the Chrome process, capture the
WAL-bracketed SQLite databases, SIGCONT.

This is the implementation side of spec § 3's hot-mode design. It
does NOT change the writer's main contract; it just gives
``ProfileSnapshotter.capture(mode="hot")`` something to call instead
of raising ``NotImplementedError``.

Strategy
========

1. Send ``SIGSTOP`` to the Chrome process (and its children — Chrome
   uses multi-process; the parent's children include the renderer
   processes that own most SQLite handles). Buffered writes freeze
   in flight.
2. ``fsync`` the profile directory recursively so kernel-side dirty
   pages from already-issued writes hit disk.
3. Run the standard cold-mode capture path. The tar will include the
   ``Cookies-wal`` / ``Local Storage-wal`` / etc. files alongside
   their main databases — SQLite's normal startup will replay the
   WAL into the main file on load.
4. Send ``SIGCONT`` to resume Chrome. The whole bracket typically
   takes 1-3 seconds; Chrome perceives it as a brief stall.

Inherent risk (spec § 3, § 4): a write transaction in flight at
``SIGSTOP`` time is rolled back by SQLite's WAL replay on load. The
loader may see a state slightly stale relative to what the user did
in Chrome. We document it, we don't paper over it.

Operator opt-in
===============

Hot mode is opt-in per call (``capture(mode="hot")``). The brain's
default is cold mode — see the spec's "When would we ever ship hot
mode?" section for the narrow cases where hot mode is worth its
correctness tradeoff.

The bracket is a no-op safety fallback when the Chrome PID is
unknown (``chrome_pid=0`` or PID-not-found) — capture continues as
cold mode in that case, logged at WARNING so operators see the
demotion.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


# How long to wait for SIGSTOP to actually freeze the process before
# proceeding with the capture. Chrome rarely needs more than a few ms.
_SIGSTOP_SETTLE_SECONDS = 0.2

# How long to wait between issuing SIGCONT and considering the
# bracket complete. Some test environments need a beat for the
# kernel to actually deliver the signal before subsequent operations
# (e.g. the snapshot writer's release call) succeed.
_SIGCONT_SETTLE_SECONDS = 0.1


@contextmanager
def freeze_chrome(
    chrome_pid: int,
    *,
    sigstop_settle_seconds: float = _SIGSTOP_SETTLE_SECONDS,
    sigcont_settle_seconds: float = _SIGCONT_SETTLE_SECONDS,
) -> Iterator[None]:
    """Context manager: SIGSTOP on enter, SIGCONT on exit.

    Targets Chrome's PID and every direct child. The renderer
    processes that own the SQLite handles are children of the parent
    Chrome process; freezing only the parent leaves writes in flight
    on the renderers.

    No-op when ``chrome_pid <= 0`` — the caller hasn't told us which
    process to freeze, so the safest action is to do nothing and let
    cold mode's correctness guarantee take over (the caller bracket
    fires the standard capture either way; without the freeze the
    capture is just less aggressive).

    The exit branch always runs (even if the body raises), so a
    failure inside the protected region can never leave Chrome
    frozen — that would manifest as the brain's next step hanging on
    a screenshot.
    """
    if chrome_pid <= 0:
        logger.warning(
            "chrome_hot_snapshot: SIGSTOP skipped (chrome_pid=%d) — "
            "falling back to cold-mode capture without the freeze",
            chrome_pid,
        )
        yield
        return

    # Resolve Chrome's process tree before sending the signal — if
    # the PID is dead we want to skip the bracket entirely (don't
    # SIGSTOP some unrelated process that got the PID after Chrome
    # exited).
    pids = _resolve_process_tree(chrome_pid)
    if not pids:
        logger.warning(
            "chrome_hot_snapshot: chrome_pid=%d no longer alive — "
            "falling back to cold-mode capture", chrome_pid,
        )
        yield
        return

    logger.warning(
        "chrome_hot_snapshot: SIGSTOP %d processes (root pid=%d)",
        len(pids), chrome_pid,
    )
    _signal_pids(pids, signal.SIGSTOP)
    time.sleep(sigstop_settle_seconds)
    try:
        yield
    finally:
        # Always SIGCONT — even if the captured raised, we never want
        # to leave Chrome frozen.
        try:
            _signal_pids(pids, signal.SIGCONT)
            time.sleep(sigcont_settle_seconds)
            logger.warning(
                "chrome_hot_snapshot: SIGCONT %d processes (root pid=%d)",
                len(pids), chrome_pid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "chrome_hot_snapshot: SIGCONT raised (root pid=%d): %s",
                chrome_pid, exc,
            )


def fsync_profile_dir(source_dir: Path) -> None:
    """Best-effort recursive ``fsync`` of every file under
    ``source_dir`` so dirty pages from already-issued Chrome writes
    actually land on disk before we tar.

    Skips ``Cookies-wal`` / ``Local Storage-wal`` / ``LOCK`` /
    ``SingletonLock`` etc. — those are SQLite-internal state we want
    captured AS-IS so the loader's WAL replay sees the consistent
    bracket. Best-effort: an unreadable file (transient) doesn't
    abort the capture.
    """
    if not source_dir.is_dir():
        return
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
    # Also fsync the directory entry so the dirent reflects on disk.
    try:
        fd = os.open(str(source_dir), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


# ── helpers ───────────────────────────────────────────────────────────


def _resolve_process_tree(root_pid: int) -> list[int]:
    """Return ``[root_pid, *direct_children]`` for a live PID.

    Returns ``[]`` when the root PID isn't alive. The tree shape we
    care about is one level deep — Chrome's renderers are immediate
    children of the parent Chrome process.

    Uses ``/proc`` directly so we don't carry a ``psutil`` dependency
    for one read. Linux-only (Modal / Baseten / E2B / Daytona are all
    Linux; macOS dev paths don't actually call hot mode).
    """
    if not _pid_alive(root_pid):
        return []
    pids = [root_pid]
    proc_root = Path("/proc")
    if not proc_root.exists():
        # Not Linux; just return the root pid. The bracket still
        # works for the immediate process, just not its children.
        return pids
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # /proc/<pid>/stat is space-separated; field 4 is the parent
        # PID. The first field can contain spaces inside parens (the
        # process name), so parse from the closing paren.
        try:
            paren_end = stat.rindex(")")
            tail = stat[paren_end + 2:].split()
            if len(tail) < 2:
                continue
            ppid = int(tail[1])
        except (ValueError, IndexError):
            continue
        if ppid == root_pid:
            try:
                pids.append(int(entry.name))
            except ValueError:
                continue
    return pids


def _pid_alive(pid: int) -> bool:
    """``True`` if a process with this PID exists right now."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but isn't ours. For our case that means we
        # can't signal it, so for the bracket's purposes it might as
        # well not exist.
        return False
    except OSError:
        return False


def _signal_pids(pids: Iterable[int], sig: int) -> None:
    """Send ``sig`` to every PID in ``pids``. Skip dead ones."""
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except OSError as exc:
            logger.warning(
                "chrome_hot_snapshot: kill(%d, %d) raised: %s",
                pid, sig, exc,
            )


__all__ = [
    "freeze_chrome",
    "fsync_profile_dir",
]
