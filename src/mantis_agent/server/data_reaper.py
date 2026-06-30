"""TTL pruning for accumulated ``/data`` run artifacts — inode-cap defense.

The CUA volume (``osworld-data``, mounted at ``/data``) is capped by Modal at
**500,000 inodes**, independent of byte capacity. Every run writes a per-run
screenshot directory under ``/data/screenshots/<session>_<run_id>/`` and a
per-run state directory under ``/data/tenants/<tenant>/runs/<run_id>/`` (status
+ result + csv + events). With no TTL these accumulate forever; on
2026-06-30 the volume hit 100% inodes (``tenants`` 252K + ``screenshots`` 202K
= 91% of the cap) and every run started failing with
``OSError: [Errno 28] No space left on device`` — at 0 steps, indistinguishable
from a code bug until you look at ``df -i``.

This module prunes those two per-run artifact trees by mtime. It deliberately
touches **only** ``screenshots`` and the per-tenant ``runs`` directories —
never ``models`` / ``training`` / ``vwa`` / ``Ubuntu.qcow2`` (weights, training
data, VM images) or the reused ``chrome-profile`` dirs.

The pure functions here are unit-tested against a tmp dir; the Modal scheduled
function in ``deploy/modal/modal_cua_server.py`` mounts the volume and calls
:func:`prune_run_artifacts`, then commits.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any


def prune_stale_children(
    target: Path, ttl_seconds: float, now: float | None = None,
) -> dict[str, Any]:
    """Delete immediate children of *target* whose mtime is older than the TTL.

    A child is a per-run directory (or stray file). Directories are removed
    recursively; symlinks are unlinked, never followed. Returns a summary with
    the deleted names, the kept count, and any per-child errors. A missing
    *target* is a no-op (``missing=True``) — not an error.
    """
    now = time.time() if now is None else now
    cutoff = now - ttl_seconds
    deleted: list[str] = []
    errors: list[str] = []
    kept = 0

    if not target.is_dir():
        return {"target": str(target), "missing": True,
                "deleted": deleted, "kept": kept, "errors": errors}

    for child in target.iterdir():
        try:
            if child.stat().st_mtime >= cutoff:
                kept += 1
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            deleted.append(child.name)
        except OSError as exc:
            errors.append(f"{child.name}: {exc}")

    return {"target": str(target), "missing": False,
            "deleted": deleted, "kept": kept, "errors": errors}


def prune_run_artifacts(
    data_root: Path, ttl_seconds: float, now: float | None = None,
) -> dict[str, Any]:
    """Prune per-run artifact trees under *data_root* older than the TTL.

    Targets, and ONLY these:
      * ``<data_root>/screenshots/*`` — per-run frame-capture dirs.
      * ``<data_root>/tenants/<tenant>/runs/*`` — per-run state dirs.

    Aggregates :func:`prune_stale_children` across every target. Never recurses
    into or deletes weights / training data / VM images / chrome profiles.
    """
    now = time.time() if now is None else now
    targets: list[Path] = []

    screenshots = data_root / "screenshots"
    if screenshots.is_dir():
        targets.append(screenshots)

    tenants = data_root / "tenants"
    if tenants.is_dir():
        for tenant_dir in sorted(tenants.iterdir()):
            runs = tenant_dir / "runs"
            if runs.is_dir():
                targets.append(runs)

    total_deleted = 0
    total_kept = 0
    errors: list[str] = []
    per_target: list[dict[str, Any]] = []

    for target in targets:
        result = prune_stale_children(target, ttl_seconds, now)
        total_deleted += len(result["deleted"])
        total_kept += result["kept"]
        errors.extend(result["errors"])
        per_target.append({
            "target": result["target"],
            "deleted": len(result["deleted"]),
            "kept": result["kept"],
        })

    return {
        "deleted": total_deleted,
        "kept": total_kept,
        "ttl_seconds": ttl_seconds,
        "errors": errors,
        "targets": per_target,
    }
