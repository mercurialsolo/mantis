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
