"""Unit tests for the /data artifact TTL reaper (inode-cap defense)."""

from __future__ import annotations

import os
from pathlib import Path

from mantis_agent.server.data_reaper import prune_run_artifacts, prune_stale_children


def _touch(path: Path, age_seconds: float, now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    t = now - age_seconds
    os.utime(path, (t, t))


def _mkdir_aged(path: Path, age_seconds: float, now: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "frame.png").write_text("x")
    t = now - age_seconds
    os.utime(path / "frame.png", (t, t))
    os.utime(path, (t, t))


def test_prune_stale_children_deletes_old_keeps_new(tmp_path: Path) -> None:
    now = 1_000_000.0
    target = tmp_path / "screenshots"
    _mkdir_aged(target / "old_run", age_seconds=5 * 86400, now=now)   # 5 days
    _mkdir_aged(target / "new_run", age_seconds=1 * 3600, now=now)    # 1 hour

    res = prune_stale_children(target, ttl_seconds=3 * 86400, now=now)

    assert res["deleted"] == ["old_run"]
    assert res["kept"] == 1
    assert not (target / "old_run").exists()
    assert (target / "new_run").exists()


def test_prune_missing_target_is_noop(tmp_path: Path) -> None:
    res = prune_stale_children(tmp_path / "nope", ttl_seconds=10, now=1.0)
    assert res["missing"] is True
    assert res["deleted"] == []


def test_prune_run_artifacts_targets_only_screenshots_and_runs(tmp_path: Path) -> None:
    now = 2_000_000.0
    # Old artifacts that SHOULD be pruned
    _mkdir_aged(tmp_path / "screenshots" / "sess_old", 10 * 86400, now)
    _mkdir_aged(tmp_path / "tenants" / "t1" / "runs" / "run_old", 10 * 86400, now)
    # Recent artifacts that should be KEPT
    _mkdir_aged(tmp_path / "tenants" / "t1" / "runs" / "run_new", 60, now)
    # Things that MUST NEVER be touched, even though old
    _mkdir_aged(tmp_path / "models" / "gguf", 30 * 86400, now)
    _mkdir_aged(tmp_path / "tenants" / "t1" / "chrome-profile" / "default", 30 * 86400, now)
    _touch(tmp_path / "Ubuntu.qcow2", 30 * 86400, now)

    res = prune_run_artifacts(tmp_path, ttl_seconds=3 * 86400, now=now)

    assert res["deleted"] == 2  # sess_old + run_old
    assert not (tmp_path / "screenshots" / "sess_old").exists()
    assert not (tmp_path / "tenants" / "t1" / "runs" / "run_old").exists()
    assert (tmp_path / "tenants" / "t1" / "runs" / "run_new").exists()
    # Protected trees untouched
    assert (tmp_path / "models" / "gguf").exists()
    assert (tmp_path / "tenants" / "t1" / "chrome-profile" / "default").exists()
    assert (tmp_path / "Ubuntu.qcow2").exists()


def test_prune_run_artifacts_handles_no_data(tmp_path: Path) -> None:
    res = prune_run_artifacts(tmp_path, ttl_seconds=10, now=1.0)
    assert res["deleted"] == 0
    assert res["targets"] == []
