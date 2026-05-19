"""Tests for #487 (model-serving facade) + #485 (observation store).

Covers:

* The facade's Role / RoutingMode enums and the typed
  ModelCallResult shape.
* PassthroughFacade end-to-end: invokes the wrapped client,
  captures latency, applies version_pin override, picks up
  version_lookup stamps.
* stamp_runtime_versions threads facade results onto the
  runner's runtime_versions dict so #488's emit hook surfaces
  them on every canonical event.
* InMemoryObservationStore: round-trips bytes by content-
  addressed ref, dedups on same input, refuses empty / non-bytes
  input, applies redaction-before-storage.
* The emitter integration: when a store is configured + the
  StepResult has screenshot_png, the canonical event's
  observation.screenshot_ref carries the real store ref instead
  of the placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mantis_agent.cua_contracts import (
    DiskObservationStore,
    InMemoryObservationStore,
    JSONL_FILENAME,
    ModelCallResult,
    PassthroughFacade,
    Role,
    RoutingMode,
    TrajectoryEmitter,
    identity_redaction,
    stamp_runtime_versions,
)
from mantis_agent.gym.checkpoint import StepResult
from mantis_agent.plan_decomposer import MicroIntent


# ── #487: Role + RoutingMode enums ─────────────────────────────────────


def test_role_enum_has_canonical_members() -> None:
    assert {r.value for r in Role} == {"planner", "grounding", "actor", "verifier"}


def test_routing_mode_enum_has_canonical_members() -> None:
    assert {m.value for m in RoutingMode} == {"prod", "shadow", "canary"}


# ── #487: PassthroughFacade ────────────────────────────────────────────


def test_passthrough_facade_invokes_wrapped_callable() -> None:
    invoker = MagicMock(return_value={"text": "hi"})
    facade = PassthroughFacade(invoker)

    result = facade.invoke(role=Role.PLANNER, payload={"prompt": "x"})

    invoker.assert_called_once_with({"prompt": "x"})
    assert isinstance(result, ModelCallResult)
    assert result.role is Role.PLANNER
    assert result.routing_mode is RoutingMode.PROD
    assert result.payload == {"text": "hi"}


def test_passthrough_facade_captures_latency() -> None:
    """Latency is measured per-call so cost dashboards don't need
    callers to time the underlying client."""
    import time
    def _slow(_payload):
        time.sleep(0.005)
        return None
    facade = PassthroughFacade(_slow)
    result = facade.invoke(role=Role.VERIFIER, payload={})
    assert result.latency_seconds >= 0.005


def test_passthrough_facade_applies_version_pin_override() -> None:
    """version_pin wins over whatever the underlying client
    reports — this is the model-registry promotion knob from
    #490 in action."""
    facade = PassthroughFacade(
        lambda _p: None,
        version_lookup=lambda: ("claude-haiku-default", "prompt_v1"),
    )
    result = facade.invoke(
        role=Role.PLANNER, payload={},
        version_pin="claude-opus-4-7-pinned",
    )
    assert result.model_version == "claude-opus-4-7-pinned"
    # prompt_version still from the lookup (not overridden).
    assert result.prompt_version == "prompt_v1"


def test_passthrough_facade_swallows_version_lookup_errors() -> None:
    """A broken version_lookup must not crash a model call. Best-
    effort observability is the rule across cua_contracts."""
    def _broken_lookup() -> tuple[str, str]:
        raise RuntimeError("registry unreachable")
    facade = PassthroughFacade(lambda _p: None, version_lookup=_broken_lookup)
    result = facade.invoke(role=Role.PLANNER, payload={})
    assert result.model_version == ""
    assert result.prompt_version == ""


def test_passthrough_facade_honours_routing_mode_on_result() -> None:
    """Routing mode is recorded on the result so the canonical
    event reader can group prod-vs-shadow without re-deriving."""
    facade = PassthroughFacade(lambda _p: None)
    out = facade.invoke(role=Role.GROUNDING, payload={}, routing_mode=RoutingMode.SHADOW)
    assert out.routing_mode is RoutingMode.SHADOW


# ── #487: stamp_runtime_versions ───────────────────────────────────────


def test_stamp_runtime_versions_writes_role_keyed_entries() -> None:
    """Facade results land in runner.runtime_versions under
    <role>_model / <role>_prompt keys — matching the canonical
    key set in cua_contracts/versions.py."""
    class _Runner:
        pass
    runner = _Runner()
    result = ModelCallResult(
        role=Role.PLANNER, routing_mode=RoutingMode.PROD,
        payload=None,
        model_version="claude-opus-4-7",
        prompt_version="planner_v3",
    )
    stamp_runtime_versions(runner, result)
    assert runner.runtime_versions == {
        "planner_model": "claude-opus-4-7",
        "planner_prompt": "planner_v3",
    }


def test_stamp_runtime_versions_skips_empty_stamps() -> None:
    """Empty stamps don't land (preserves the validator's "no
    blank values" contract from #488)."""
    runner = type("R", (), {})()
    result = ModelCallResult(
        role=Role.GROUNDING, routing_mode=RoutingMode.PROD,
        payload=None,
        model_version="haiku-4-5",
        prompt_version="",  # no prompt version known
    )
    stamp_runtime_versions(runner, result)
    assert runner.runtime_versions == {"grounding_model": "haiku-4-5"}
    assert "grounding_prompt" not in runner.runtime_versions


def test_stamp_runtime_versions_accumulates_across_roles() -> None:
    """Multiple roles building up the dict — last write wins per
    role-key but other roles' entries are preserved."""
    runner = type("R", (), {})()
    for role, mv in [
        (Role.PLANNER, "claude-opus-4-7"),
        (Role.GROUNDING, "claude-haiku-4-5"),
        (Role.VERIFIER, "claude-haiku-4-5"),
    ]:
        stamp_runtime_versions(runner, ModelCallResult(
            role=role, routing_mode=RoutingMode.PROD, payload=None,
            model_version=mv,
        ))
    assert runner.runtime_versions == {
        "planner_model": "claude-opus-4-7",
        "grounding_model": "claude-haiku-4-5",
        "verifier_model": "claude-haiku-4-5",
    }


# ── #485: InMemoryObservationStore ─────────────────────────────────────


def test_observation_store_put_returns_content_addressed_ref() -> None:
    store = InMemoryObservationStore()
    ref = store.put(b"\x89PNG\r\n\x1a\n--fake-image--")
    assert ref.startswith("sha256:")
    assert len(ref) == len("sha256:") + 64  # 64 hex chars


def test_observation_store_dedupes_same_bytes() -> None:
    """Same input bytes → same ref → one entry in storage."""
    store = InMemoryObservationStore()
    ref_a = store.put(b"same-bytes")
    ref_b = store.put(b"same-bytes")
    assert ref_a == ref_b
    assert len(store) == 1


def test_observation_store_distinct_bytes_get_distinct_refs() -> None:
    store = InMemoryObservationStore()
    ref_a = store.put(b"bytes-a")
    ref_b = store.put(b"bytes-b")
    assert ref_a != ref_b


def test_observation_store_get_round_trips() -> None:
    store = InMemoryObservationStore()
    blob = b"\x89PNG\r\n\x1a\n--fake--"
    ref = store.put(blob)
    stored = store.get(ref)
    assert stored is not None
    assert stored.ref == ref
    assert stored.image_bytes == blob
    assert stored.redaction_policy_name == "identity_redaction"


def test_observation_store_get_returns_none_for_unknown_ref() -> None:
    store = InMemoryObservationStore()
    assert store.get("sha256:" + "0" * 64) is None
    assert store.exists("sha256:" + "0" * 64) is False


def test_observation_store_rejects_empty_bytes() -> None:
    store = InMemoryObservationStore()
    with pytest.raises(ValueError, match="empty"):
        store.put(b"")


def test_observation_store_rejects_non_bytes() -> None:
    store = InMemoryObservationStore()
    with pytest.raises(TypeError, match="bytes"):
        store.put("not-bytes")  # type: ignore[arg-type]


def test_observation_store_applies_redaction_before_storage() -> None:
    """The ref is keyed on POST-redaction bytes — so two callers
    that pass different inputs that redact to the same content
    get the same ref."""
    def _zero_out_first_byte(blob: bytes) -> bytes:
        return b"\x00" + blob[1:]

    store = InMemoryObservationStore(
        redaction_policy=_zero_out_first_byte,
        redaction_policy_name="zero_first_byte",
    )
    ref_a = store.put(b"abc-xyz")
    ref_b = store.put(b"qbc-xyz")  # same after redaction
    assert ref_a == ref_b
    stored = store.get(ref_a)
    assert stored.image_bytes == b"\x00bc-xyz"
    assert stored.redaction_policy_name == "zero_first_byte"


def test_observation_store_rejects_broken_redaction_policy() -> None:
    """A redaction policy that returns empty / non-bytes is a
    writer bug — fail closed at put() rather than persist garbage."""
    store = InMemoryObservationStore(redaction_policy=lambda _b: b"")
    with pytest.raises(ValueError, match="empty / non-bytes"):
        store.put(b"original")


def test_identity_redaction_passes_bytes_through_unchanged() -> None:
    assert identity_redaction(b"hello") == b"hello"


# ── #485: Emitter integration with the store ──────────────────────────


def _intent() -> MicroIntent:
    return MicroIntent(intent="x", type="click")


def _ok_result_with_screenshot(blob: bytes) -> StepResult:
    r = StepResult(step_index=0, intent="x", success=True, data="ok")
    r.screenshot_png = blob
    return r


def test_emit_uses_observation_store_ref_when_configured(tmp_path: Path) -> None:
    """When the emitter has an observation_store + the StepResult
    has screenshot_png bytes, the canonical event's
    observation.screenshot_ref carries the real store ref
    instead of the synthetic placeholder."""
    store = InMemoryObservationStore()
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        observation_store=store,
    )
    blob = b"\x89PNG\r\n\x1a\n--fake-screenshot--"
    emitter.emit(_intent(), _ok_result_with_screenshot(blob))

    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    ref = record["observation"]["screenshot_ref"]
    assert ref.startswith("sha256:")
    # And the store actually persisted those exact bytes.
    stored = store.get(ref)
    assert stored is not None
    assert stored.image_bytes == blob


def test_emit_falls_back_to_placeholder_when_no_store(tmp_path: Path) -> None:
    """No store configured → existing placeholder:// behaviour
    preserved. Legacy callers / tests unaffected."""
    emitter = TrajectoryEmitter(run_id="r1", store_dir=str(tmp_path))
    emitter.emit(_intent(), _ok_result_with_screenshot(b"some-bytes"))

    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["observation"]["screenshot_ref"].startswith("placeholder://")


def test_emit_falls_back_to_placeholder_when_screenshot_absent(tmp_path: Path) -> None:
    """Store configured but no screenshot_png bytes on the
    StepResult → placeholder (the deterministic-handler path that
    doesn't produce a screenshot stays valid)."""
    store = InMemoryObservationStore()
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        observation_store=store,
    )
    r = StepResult(step_index=0, intent="x", success=True, data="ok")
    # No screenshot_png set.
    emitter.emit(_intent(), r)
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["observation"]["screenshot_ref"].startswith("placeholder://")
    # Store stays empty.
    assert len(store) == 0


def test_emit_swallows_store_failure_and_keeps_placeholder(
    tmp_path: Path, caplog,
) -> None:
    """observation_store.put() raising must not block emit — log
    and fall back to the placeholder. Canonical events stay
    write-safe even when storage is misconfigured."""
    broken_store = MagicMock()
    broken_store.put.side_effect = RuntimeError("S3 unreachable")
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        observation_store=broken_store,
    )
    with caplog.at_level("WARNING"):
        emitter.emit(_intent(), _ok_result_with_screenshot(b"bytes"))
    assert any("observation_store.put failed" in r.message for r in caplog.records)
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["observation"]["screenshot_ref"].startswith("placeholder://")


def test_emit_explicit_screenshot_ref_wins_over_store(tmp_path: Path) -> None:
    """When the caller passes an explicit screenshot_ref, the
    store isn't consulted — the caller knows best."""
    store = InMemoryObservationStore()
    emitter = TrajectoryEmitter(
        run_id="r1", store_dir=str(tmp_path),
        observation_store=store,
    )
    emitter.emit(
        _intent(), _ok_result_with_screenshot(b"would-be-stored"),
        screenshot_ref="s3://bucket/path/step_0.png",
    )
    record = json.loads((tmp_path / JSONL_FILENAME).read_text().strip())
    assert record["observation"]["screenshot_ref"] == "s3://bucket/path/step_0.png"
    # Store stayed empty.
    assert len(store) == 0


# ── DiskObservationStore — same protocol, on-disk persistence ─────────


def test_disk_store_persists_bytes_under_sharded_path(tmp_path: Path) -> None:
    """Storage layout: <root>/<sha[:2]>/<sha>.bin so any single
    directory stays under ~256 entries. Confirm the file lands at
    the expected path AND content matches the input."""
    store = DiskObservationStore(str(tmp_path))
    blob = b"\x89PNG\r\n\x1a\n--fake-image--"
    ref = store.put(blob)
    assert ref.startswith("sha256:")
    sha_hex = ref.split(":", 1)[1]
    expected = tmp_path / sha_hex[:2] / (sha_hex + ".bin")
    assert expected.exists()
    assert expected.read_bytes() == blob


def test_disk_store_dedups_same_bytes(tmp_path: Path) -> None:
    """Same input → same ref → idempotent put (no duplicate file
    write on the second call)."""
    store = DiskObservationStore(str(tmp_path))
    ref_a = store.put(b"some-bytes")
    mtime_a = (tmp_path / ref_a.split(":", 1)[1][:2] / (ref_a.split(":", 1)[1] + ".bin")).stat().st_mtime
    # Second put with identical bytes — should NOT touch the file.
    import time as _time
    _time.sleep(0.01)
    ref_b = store.put(b"some-bytes")
    mtime_b = (tmp_path / ref_b.split(":", 1)[1][:2] / (ref_b.split(":", 1)[1] + ".bin")).stat().st_mtime
    assert ref_a == ref_b
    assert mtime_a == mtime_b  # file untouched on dedup


def test_disk_store_get_round_trips(tmp_path: Path) -> None:
    store = DiskObservationStore(str(tmp_path))
    blob = b"--fake-screenshot--"
    ref = store.put(blob)
    stored = store.get(ref)
    assert stored is not None
    assert stored.ref == ref
    assert stored.image_bytes == blob
    assert stored.redaction_policy_name == "identity_redaction"


def test_disk_store_get_returns_none_for_unknown_ref(tmp_path: Path) -> None:
    store = DiskObservationStore(str(tmp_path))
    assert store.get("sha256:" + "0" * 64) is None
    assert store.exists("sha256:" + "0" * 64) is False


def test_disk_store_creates_root_dir_eagerly(tmp_path: Path) -> None:
    """A misconfigured root should fail at construction, not first
    put(). Conversely a non-existent-yet root is created so first
    put just works."""
    fresh_root = tmp_path / "events" / "obs"
    assert not fresh_root.exists()
    DiskObservationStore(str(fresh_root))
    assert fresh_root.is_dir()


def test_disk_store_rejects_empty_root() -> None:
    with pytest.raises(ValueError, match="root_dir"):
        DiskObservationStore("")


def test_disk_store_redaction_runs_before_persistence(tmp_path: Path) -> None:
    """The ref is keyed on POST-redaction bytes; the file on disk
    contains POST-redaction bytes. Operators reading the store
    cannot recover the original input."""
    def _zero_first_byte(b: bytes) -> bytes:
        return b"\x00" + b[1:]

    store = DiskObservationStore(
        str(tmp_path),
        redaction_policy=_zero_first_byte,
        redaction_policy_name="zero_first_byte",
    )
    raw = b"abc-xyz"
    ref = store.put(raw)
    stored = store.get(ref)
    assert stored.image_bytes == b"\x00bc-xyz"
    assert stored.redaction_policy_name == "zero_first_byte"


def test_disk_store_default_max_bytes_is_one_gib(tmp_path: Path) -> None:
    """Default cap is 1 GiB so an unconfigured production deploy
    can't grow unbounded across runs."""
    store = DiskObservationStore(str(tmp_path))
    assert store.max_bytes == (1 << 30)  # 1 GiB


def test_disk_store_explicit_max_bytes_wins(tmp_path: Path) -> None:
    store = DiskObservationStore(str(tmp_path), max_bytes=1024)
    assert store.max_bytes == 1024


def test_disk_store_env_max_bytes_used_when_no_ctor_arg(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("MANTIS_OBS_STORE_MAX_BYTES", "4096")
    store = DiskObservationStore(str(tmp_path))
    assert store.max_bytes == 4096


def test_disk_store_garbage_env_falls_back_to_default(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("MANTIS_OBS_STORE_MAX_BYTES", "not-a-number")
    store = DiskObservationStore(str(tmp_path))
    assert store.max_bytes == (1 << 30)


def test_disk_store_evicts_oldest_when_cap_exceeded(tmp_path: Path) -> None:
    """LRU eviction — oldest (by mtime) files go first when the
    next put would push the store past the cap."""
    import time as _time

    # Cap of 100 bytes; each put is 30 bytes → after 3 puts the
    # 4th triggers eviction of the oldest.
    store = DiskObservationStore(str(tmp_path), max_bytes=100)
    ref_a = store.put(b"a" * 30)
    _time.sleep(0.01)  # ensure distinct mtimes
    ref_b = store.put(b"b" * 30)
    _time.sleep(0.01)
    ref_c = store.put(b"c" * 30)
    assert store.current_bytes == 90
    # All three present.
    assert store.exists(ref_a)
    assert store.exists(ref_b)
    assert store.exists(ref_c)

    # 4th put pushes past 100 → evict oldest (ref_a).
    ref_d = store.put(b"d" * 30)
    assert store.exists(ref_d)
    assert not store.exists(ref_a), "oldest should have been evicted"
    assert store.exists(ref_b)
    assert store.exists(ref_c)
    assert store.current_bytes == 90


def test_disk_store_evicts_multiple_when_needed(tmp_path: Path) -> None:
    """A large incoming put may require evicting more than one
    file to make space."""
    import time as _time

    store = DiskObservationStore(str(tmp_path), max_bytes=100)
    refs = []
    for i in range(4):
        refs.append(store.put(bytes([0x41 + i]) * 25))
        _time.sleep(0.005)
    assert store.current_bytes == 100

    # 60-byte put under cap=100 requires current ≤ 40 before
    # write. Three 25-byte files (75 → 50 → 25) must go to get
    # there; only the newest survives.
    big_ref = store.put(b"X" * 60)
    assert store.exists(big_ref)
    assert not store.exists(refs[0])
    assert not store.exists(refs[1])
    assert not store.exists(refs[2])
    assert store.exists(refs[3])
    # 25 (refs[3]) + 60 (big) = 85.
    assert store.current_bytes == 85


def test_disk_store_oversize_observation_logs_warning(
    tmp_path: Path, caplog,
) -> None:
    """A single observation bigger than the entire cap still gets
    written (we don't want to silently lose data) but emits a
    WARNING so the operator notices."""
    store = DiskObservationStore(str(tmp_path), max_bytes=50)
    with caplog.at_level("WARNING"):
        ref = store.put(b"X" * 100)
    assert store.exists(ref)
    assert any("exceeds cap" in r.message for r in caplog.records)


def test_disk_store_init_counts_existing_files_toward_cap(
    tmp_path: Path,
) -> None:
    """A resumed run with prior observations on disk must count
    those toward the cap — without the init scan a long-lived
    Modal volume's cap would only constrain the current process."""
    # Seed the dir with files via a first store.
    first = DiskObservationStore(str(tmp_path), max_bytes=10_000)
    first.put(b"x" * 100)
    first.put(b"y" * 100)
    # Fresh store sees the existing footprint.
    second = DiskObservationStore(str(tmp_path), max_bytes=10_000)
    assert second.current_bytes == 200


def test_disk_store_dedup_does_not_double_count_size(tmp_path: Path) -> None:
    """Putting the same bytes twice shouldn't bump current_bytes
    a second time — the file already exists, no new disk
    consumption."""
    store = DiskObservationStore(str(tmp_path), max_bytes=1000)
    store.put(b"abc" * 50)  # 150 bytes
    assert store.current_bytes == 150
    store.put(b"abc" * 50)  # same content → dedup
    assert store.current_bytes == 150


def test_disk_store_rescan_size_reconciles_external_deletion(
    tmp_path: Path,
) -> None:
    """If an external process prunes files, the in-memory tally
    goes stale — rescan_size() reconciles."""
    import os
    store = DiskObservationStore(str(tmp_path), max_bytes=10_000)
    ref = store.put(b"x" * 100)
    assert store.current_bytes == 100
    # External deletion.
    os.remove(store._path_for(ref))
    # Tally is still 100 (best-effort in-memory).
    assert store.current_bytes == 100
    # Reconcile.
    reconciled = store.rescan_size()
    assert reconciled == 0
    assert store.current_bytes == 0


def test_disk_store_round_trips_across_construction(tmp_path: Path) -> None:
    """Refs survive process restart — a fresh store pointed at the
    same root sees the prior writes. Confirms the on-disk format
    is the truth, not in-memory bookkeeping."""
    store_a = DiskObservationStore(str(tmp_path))
    ref = store_a.put(b"persisted-bytes")

    store_b = DiskObservationStore(str(tmp_path))
    stored = store_b.get(ref)
    assert stored is not None
    assert stored.image_bytes == b"persisted-bytes"
    assert store_b.exists(ref) is True


# ── Default-wire: emit hook picks the disk store when canonical
#    events are enabled and no runner.observation_store is set ─────────


def test_executor_hook_defaults_to_disk_store_under_events_dir(
    monkeypatch, tmp_path: Path,
) -> None:
    """When MANTIS_CANONICAL_EVENTS_DIR is set + the runner has no
    explicit observation_store, the emit hook constructs a
    :class:`DiskObservationStore` rooted at
    ``<events-base>/<run_id>/observations/``. Every event then
    lands with a real ``sha256:<hex>`` ref AND the screenshot
    bytes are on disk for an operator to pull back."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    class _Runner:
        run_id = "default_store_test"

    runner = _Runner()
    r = StepResult(step_index=0, intent="x", success=True, data="ok")
    r.screenshot_png = b"\x89PNG\r\n\x1a\n--fake--"
    _emit_canonical_trajectory_event(runner, _intent(), r)

    # Trajectory event references a real sha256 ref, not a placeholder.
    record = json.loads(
        (tmp_path / "default_store_test" / JSONL_FILENAME).read_text().strip(),
    )
    ref = record["observation"]["screenshot_ref"]
    assert ref.startswith("sha256:")

    # The file landed under <events-base>/<run_id>/observations/
    # so an operator pulling the run dir gets both events + bytes.
    sha_hex = ref.split(":", 1)[1]
    expected = (
        tmp_path / "default_store_test" / "observations" / sha_hex[:2]
        / (sha_hex + ".bin")
    )
    assert expected.exists()
    assert expected.read_bytes() == b"\x89PNG\r\n\x1a\n--fake--"


def test_executor_hook_honours_explicit_runner_store_over_default(
    monkeypatch, tmp_path: Path,
) -> None:
    """An explicit ``runner.observation_store`` always wins — the
    default is only used when nothing is configured."""
    from mantis_agent.gym.run_executor import _emit_canonical_trajectory_event

    monkeypatch.setenv("MANTIS_CANONICAL_EVENTS_DIR", str(tmp_path))

    explicit = InMemoryObservationStore()

    class _Runner:
        run_id = "explicit_store_test"
        observation_store = explicit

    runner = _Runner()
    r = StepResult(step_index=0, intent="x", success=True, data="ok")
    r.screenshot_png = b"--explicit-store-bytes--"
    _emit_canonical_trajectory_event(runner, _intent(), r)

    # The bytes landed in the explicit (in-memory) store, NOT the
    # disk-default — the explicit-runner-store wins.
    assert len(explicit) == 1
    # The default disk path was never created.
    default_obs_dir = tmp_path / "explicit_store_test" / "observations"
    assert not default_obs_dir.exists()
