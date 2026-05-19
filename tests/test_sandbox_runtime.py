"""Tests for the sandbox runtime substrate (#484).

Covers:

* SandboxRuntime Protocol: runtime_checkable shape for noop +
  local-profile implementations.
* NoopSandboxRuntime: deterministic placeholder snapshot ids;
  restore accepts any non-empty id; lifecycle methods are no-ops.
* LocalProfileSandbox (the browser-task fixture):
  - create / capture / destroy lifecycle round-trips
  - snapshot copies the profile dir to a content-keyed path
  - restore reverses the copy (live profile == prior snapshot)
  - missing snapshot_id raises SandboxRestoreError
  - missing snapshot dir raises SandboxRestoreError
  - snapshot of non-existent profile returns None
  - destroy removes the profile dir
* snapshot_id round-trips through the canonical event emitter
  (TrajectoryEvent.action_result.snapshot_id).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis_agent.cua_contracts import (
    JSONL_FILENAME,
    LocalProfileSandbox,
    NoopSandboxRuntime,
    SandboxRestoreError,
    SandboxRuntime,
    SandboxSnapshot,
    TrajectoryEmitter,
)
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.plan_decomposer import MicroIntent


# ── Protocol shape ─────────────────────────────────────────────────────


def test_noop_runtime_satisfies_protocol() -> None:
    assert isinstance(NoopSandboxRuntime(), SandboxRuntime)


def test_local_profile_sandbox_satisfies_protocol(tmp_path: Path) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "profiles"),
        snapshot_root=str(tmp_path / "snapshots"),
    )
    assert isinstance(sb, SandboxRuntime)


# ── NoopSandboxRuntime ─────────────────────────────────────────────────


def test_noop_runtime_snapshot_returns_typed_placeholder() -> None:
    rt = NoopSandboxRuntime()
    s = rt.snapshot(key="profile_x")
    assert isinstance(s, SandboxSnapshot)
    assert s.snapshot_id.startswith("noop:profile_x:")
    assert s.size_bytes == 0
    assert s.metadata == {"runtime": "noop"}


def test_noop_runtime_snapshot_ids_are_deterministically_unique() -> None:
    """Counter ensures repeated snapshots get distinct ids — readers
    can use ids as keys without collision."""
    rt = NoopSandboxRuntime()
    ids = {rt.snapshot(key="k").snapshot_id for _ in range(5)}
    assert len(ids) == 5


def test_noop_runtime_restore_accepts_any_non_empty_id() -> None:
    rt = NoopSandboxRuntime()
    rt.restore(key="k", snapshot_id="noop:k:s1")  # no exception


def test_noop_runtime_restore_rejects_empty_id() -> None:
    rt = NoopSandboxRuntime()
    with pytest.raises(SandboxRestoreError, match="snapshot_id"):
        rt.restore(key="k", snapshot_id="")


def test_noop_runtime_capture_returns_runtime_marker() -> None:
    rt = NoopSandboxRuntime()
    state = rt.capture(key="profile_x")
    assert state["runtime"] == "noop"
    assert state["key"] == "profile_x"


def test_noop_runtime_destroy_clears_state() -> None:
    rt = NoopSandboxRuntime()
    rt.create(key="profile_x")
    assert "profile_x" in rt._created_keys
    rt.destroy(key="profile_x")
    assert "profile_x" not in rt._created_keys


# ── LocalProfileSandbox — the browser-task fixture (#484 criterion) ───


def _seed_profile(sb: LocalProfileSandbox, key: str, contents: dict[str, bytes]) -> None:
    """Helper — write fake "profile" files into the sandbox key's dir."""
    import os
    sb.create(key=key)
    profile = sb._profile_path(key)
    for relpath, data in contents.items():
        full = os.path.join(profile, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)


def test_local_profile_sandbox_rejects_empty_roots() -> None:
    with pytest.raises(ValueError, match="profile_root"):
        LocalProfileSandbox(profile_root="", snapshot_root="/tmp/x")
    with pytest.raises(ValueError, match="snapshot_root"):
        LocalProfileSandbox(profile_root="/tmp/x", snapshot_root="")


def test_local_profile_sandbox_create_makes_profile_dir(tmp_path: Path) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    sb.create(key="key1")
    assert (tmp_path / "p" / "key1").is_dir()


def test_local_profile_sandbox_capture_reports_sizes(tmp_path: Path) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    _seed_profile(sb, "k", {"cookies.txt": b"=" * 200, "prefs/p.json": b"{}"})

    state = sb.capture(key="k")
    assert state["runtime"] == "local_profile"
    assert state["key"] == "k"
    assert state["file_count"] == 2
    assert state["size_bytes"] == 202


def test_local_profile_sandbox_snapshot_returns_handle(tmp_path: Path) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    _seed_profile(sb, "k", {"a.bin": b"hello"})

    snap = sb.snapshot(key="k")
    assert snap is not None
    assert snap.snapshot_id.startswith("local:")
    assert snap.size_bytes == 5
    assert snap.metadata["runtime"] == "local_profile"
    assert snap.metadata["key"] == "k"


def test_local_profile_sandbox_snapshot_of_missing_profile_returns_none(
    tmp_path: Path,
) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    # No create() called — profile dir doesn't exist.
    assert sb.snapshot(key="never-created") is None


def test_local_profile_sandbox_snapshot_then_restore_round_trips(
    tmp_path: Path,
) -> None:
    """End-to-end: seed → snapshot → mutate profile → restore →
    profile bytes match the original snapshot."""
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    _seed_profile(sb, "k", {"cookies.txt": b"original", "prefs/p.json": b"{}"})
    snap = sb.snapshot(key="k")
    assert snap is not None

    # Mutate the live profile after the snapshot.
    import os
    profile = sb._profile_path("k")
    with open(os.path.join(profile, "cookies.txt"), "wb") as f:
        f.write(b"mutated")
    with open(os.path.join(profile, "new-file.bin"), "wb") as f:
        f.write(b"appeared-after-snapshot")

    # Restore.
    sb.restore(key="k", snapshot_id=snap.snapshot_id)

    # Profile now matches the pre-mutation state.
    with open(os.path.join(profile, "cookies.txt"), "rb") as f:
        assert f.read() == b"original"
    assert not os.path.exists(os.path.join(profile, "new-file.bin"))


def test_local_profile_sandbox_restore_missing_snapshot_raises(
    tmp_path: Path,
) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    sb.create(key="k")
    with pytest.raises(SandboxRestoreError, match="not found"):
        sb.restore(key="k", snapshot_id="local:does-not-exist")


def test_local_profile_sandbox_restore_empty_snapshot_id_raises(
    tmp_path: Path,
) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    with pytest.raises(SandboxRestoreError, match="snapshot_id"):
        sb.restore(key="k", snapshot_id="")


def test_local_profile_sandbox_destroy_removes_profile(tmp_path: Path) -> None:
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    _seed_profile(sb, "k", {"a.bin": b"x"})
    assert (tmp_path / "p" / "k").is_dir()
    sb.destroy(key="k")
    assert not (tmp_path / "p" / "k").exists()


def test_local_profile_sandbox_distinct_snapshots_distinct_paths(
    tmp_path: Path,
) -> None:
    """Two snapshots of the same profile get distinct ids + distinct
    on-disk directories — caller can roll back to either."""
    sb = LocalProfileSandbox(
        profile_root=str(tmp_path / "p"),
        snapshot_root=str(tmp_path / "s"),
    )
    _seed_profile(sb, "k", {"a.bin": b"state-1"})
    snap_a = sb.snapshot(key="k")
    import os
    with open(os.path.join(sb._profile_path("k"), "a.bin"), "wb") as f:
        f.write(b"state-2")
    snap_b = sb.snapshot(key="k")
    assert snap_a.snapshot_id != snap_b.snapshot_id
    assert os.path.isdir(sb._snapshot_path("k", snap_a.snapshot_id))
    assert os.path.isdir(sb._snapshot_path("k", snap_b.snapshot_id))


# ── snapshot_id round-trips through canonical events ─────────────────


def _intent() -> MicroIntent:
    return MicroIntent(intent="x", type="submit", required=True)


def _ok_result() -> StepResult:
    return StepResult(step_index=0, intent="x", success=True, data="ok")


def test_emit_passes_snapshot_id_through_to_action_result(tmp_path: Path) -> None:
    """When the caller passes ``snapshot_id`` to emit(), it lands
    on TrajectoryEvent.action_result.snapshot_id so a downstream
    consumer can replay the failed action against the snapshot."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    ok = emitter.emit(
        _intent(), _ok_result(),
        snapshot_id="local:abcdef123",
    )
    assert ok is True
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["action_result"]["snapshot_id"] == "local:abcdef123"


def test_emit_falls_back_to_step_result_snapshot_id(tmp_path: Path) -> None:
    """When the caller doesn't pass snapshot_id but the StepResult
    has one attached (handler-stashed), the emitter picks it up."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _ok_result()
    r.snapshot_id = "noop:k:s7"  # type: ignore[attr-defined]
    emitter.emit(_intent(), r)
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["action_result"]["snapshot_id"] == "noop:k:s7"


def test_emit_omits_snapshot_id_when_none_set(tmp_path: Path) -> None:
    """No snapshot_id stamped → action_result.snapshot_id is empty
    string (default); event still validates."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    emitter.emit(_intent(), _ok_result())
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["action_result"]["snapshot_id"] == ""


def test_emit_explicit_snapshot_id_wins_over_result_attr(tmp_path: Path) -> None:
    """Caller's explicit snapshot_id beats the StepResult stash —
    same precedence rule as grounding_trace."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    r = _ok_result()
    r.snapshot_id = "stash-id"  # type: ignore[attr-defined]
    emitter.emit(_intent(), r, snapshot_id="explicit-id")
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["action_result"]["snapshot_id"] == "explicit-id"
