"""Tests for trajectory hint memory Phase 1b (#670 / #643).

Covers:
- DiskHintStore: tenant isolation, atomic writes, round-trip, LRU eviction
- record_hint_if_eligible: gating (step type, success, anchor presence)
- apply_hint_overlay: stamps hints only on grounding steps, preserves operator hints
- holo3 prompt builder includes the new preferred_target field
- Recording hook in run_executor fires through to the store
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mantis_agent.gym.hint_memory import (
    DiskHintStore,
    GROUNDING_STEP_TYPES,
    HintKey,
    HintRecord,
    InMemoryHintStore,
    ModalDictHintStore,
    NullHintStore,
    apply_hint_overlay,
    build_hint_store,
    extract_anchor_from_env,
    hint_key_for,
    record_hint_if_eligible,
)


@pytest.fixture()
def temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    monkeypatch.setenv("MANTIS_HINT_MEMORY_DIR", str(tmp_path))
    return str(tmp_path)


def _step(intent: str = "click Show More", type_: str = "click", hints: dict | None = None):
    s = SimpleNamespace(intent=intent, type=type_, params={}, hints=hints or {})
    return s


# ── DiskHintStore ────────────────────────────────────────────────────


def test_disk_store_requires_tenant_id(temp_dir: str) -> None:
    with pytest.raises(ValueError, match="non-empty tenant_id"):
        DiskHintStore(tenant_id="")


def test_disk_store_round_trips_single_record(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")
    record = HintRecord(
        anchor_text="Show More", anchor_xy_offset=(120, 540),
        viewport_stage=1, confidence=0.8, source_url="https://x.com/y/123",
    )
    store.add(key, record)
    out = store.get(key)
    assert len(out) == 1
    assert out[0].anchor_text == "Show More"
    assert out[0].anchor_xy_offset == (120, 540)
    assert out[0].viewport_stage == 1
    assert out[0].confidence == 0.8


def test_disk_store_lru_evicts_oldest_at_cap(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme", max_per_key=3)
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")
    for i in range(5):
        store.add(key, HintRecord(anchor_text=f"Anchor {i}", confidence=0.5))
    out = store.get(key)
    assert len(out) == 3
    # Newest first — Anchor 4 → 3 → 2 (oldest two evicted)
    texts = [r.anchor_text for r in out]
    assert texts == ["Anchor 4", "Anchor 3", "Anchor 2"]


def test_disk_store_tenant_isolation(temp_dir: str) -> None:
    """Different tenants writing the same key must not see each other."""
    s1 = DiskHintStore(tenant_id="customerA")
    s2 = DiskHintStore(tenant_id="customerB")
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")

    s1.add(key, HintRecord(anchor_text="A-only", confidence=0.9))
    assert [r.anchor_text for r in s1.get(key)] == ["A-only"]
    # s2 sees nothing
    assert s2.get(key) == []

    s2.add(key, HintRecord(anchor_text="B-only", confidence=0.9))
    assert [r.anchor_text for r in s2.get(key)] == ["B-only"]
    # s1 still doesn't see B's
    assert [r.anchor_text for r in s1.get(key)] == ["A-only"]


def test_disk_store_keys_isolated_within_tenant(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    k1 = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/a")
    k2 = HintKey(plan_signature="sig1", intent_hash="ih2", url_pattern="x.com/a")
    store.add(k1, HintRecord(anchor_text="A"))
    store.add(k2, HintRecord(anchor_text="B"))
    assert [r.anchor_text for r in store.get(k1)] == ["A"]
    assert [r.anchor_text for r in store.get(k2)] == ["B"]


def test_disk_store_size_counts_across_buckets(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    store.add(HintKey(plan_signature="s1", intent_hash="i1", url_pattern="x"),
              HintRecord(anchor_text="a"))
    store.add(HintKey(plan_signature="s1", intent_hash="i2", url_pattern="x"),
              HintRecord(anchor_text="b"))
    store.add(HintKey(plan_signature="s2", intent_hash="i1", url_pattern="x"),
              HintRecord(anchor_text="c"))
    assert store.size() == 3


def test_disk_store_no_op_on_empty_plan_signature(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    key = HintKey(plan_signature="", intent_hash="ih1", url_pattern="x")
    store.add(key, HintRecord(anchor_text="A"))
    assert store.get(key) == []


def test_disk_store_corrupt_file_starts_fresh(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    # Manually create a malformed file
    tenant_dir = os.path.join(temp_dir, "acme")
    os.makedirs(tenant_dir, exist_ok=True)
    with open(os.path.join(tenant_dir, "sig1.json"), "w") as f:
        f.write("not json at all")
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x")
    # Should not raise; should return empty
    assert store.get(key) == []
    # And should be writable after corruption
    store.add(key, HintRecord(anchor_text="recovered"))
    assert [r.anchor_text for r in store.get(key)] == ["recovered"]


def test_disk_store_list_plan_signatures(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    store.add(HintKey(plan_signature="sigA", intent_hash="i", url_pattern="x"),
              HintRecord(anchor_text="a"))
    store.add(HintKey(plan_signature="sigB", intent_hash="i", url_pattern="x"),
              HintRecord(anchor_text="b"))
    sigs = store.list_plan_signatures()
    assert "sigA" in sigs
    assert "sigB" in sigs


def test_disk_store_iter_records(temp_dir: str) -> None:
    store = DiskHintStore(tenant_id="acme")
    k = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    store.add(k, HintRecord(anchor_text="A"))
    store.add(k, HintRecord(anchor_text="B"))
    pairs = list(store.iter_records())
    assert len(pairs) == 2
    texts = sorted(r.anchor_text for _, r in pairs)
    assert texts == ["A", "B"]


# ── ModalDictHintStore ───────────────────────────────────────────────


class _FakeModalDict:
    """Stand-in for ``modal.Dict``: dict-like (get / __setitem__ / keys)
    but raises on ``len()`` exactly as the real one does, so size() can't
    cheat with ``len()``. One instance is shared across stores to mirror
    the cross-worker / multi-tenant single-Dict deploy."""

    def __init__(self) -> None:
        self._inner: dict = {}

    def get(self, key, default=None):
        return self._inner.get(key, default)

    def __setitem__(self, key, value) -> None:
        self._inner[key] = value

    def keys(self):
        return list(self._inner.keys())

    def __len__(self):  # noqa: D105
        raise TypeError("modal.Dict has no __len__")


def test_modal_store_requires_tenant_id() -> None:
    with pytest.raises(ValueError, match="non-empty tenant_id"):
        ModalDictHintStore({}, tenant_id="")


def test_modal_store_round_trips_single_record() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")
    store.add(key, HintRecord(
        anchor_text="Show More", anchor_xy_offset=(120, 540),
        viewport_stage=1, confidence=0.8, source_url="https://x.com/y/123",
    ))
    out = store.get(key)
    assert len(out) == 1
    assert out[0].anchor_text == "Show More"
    assert out[0].anchor_xy_offset == (120, 540)
    assert out[0].viewport_stage == 1
    assert out[0].confidence == 0.8


def test_modal_store_lru_evicts_oldest_at_cap() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme", max_per_key=3)
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")
    for i in range(5):
        store.add(key, HintRecord(anchor_text=f"Anchor {i}", confidence=0.5))
    out = store.get(key)
    assert [r.anchor_text for r in out] == ["Anchor 4", "Anchor 3", "Anchor 2"]


def test_modal_store_tenant_isolation_in_one_dict() -> None:
    """Two tenants share ONE backing Dict; key-prefix keeps them apart."""
    backing = _FakeModalDict()
    s1 = ModalDictHintStore(backing, tenant_id="customerA")
    s2 = ModalDictHintStore(backing, tenant_id="customerB")
    key = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/y")

    s1.add(key, HintRecord(anchor_text="A-only", confidence=0.9))
    assert [r.anchor_text for r in s1.get(key)] == ["A-only"]
    assert s2.get(key) == []  # B sees nothing

    s2.add(key, HintRecord(anchor_text="B-only", confidence=0.9))
    assert [r.anchor_text for r in s2.get(key)] == ["B-only"]
    assert [r.anchor_text for r in s1.get(key)] == ["A-only"]  # A unaffected


def test_modal_store_keys_isolated_within_tenant() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    k1 = HintKey(plan_signature="sig1", intent_hash="ih1", url_pattern="x.com/a")
    k2 = HintKey(plan_signature="sig1", intent_hash="ih2", url_pattern="x.com/a")
    store.add(k1, HintRecord(anchor_text="A"))
    store.add(k2, HintRecord(anchor_text="B"))
    assert [r.anchor_text for r in store.get(k1)] == ["A"]
    assert [r.anchor_text for r in store.get(k2)] == ["B"]


def test_modal_store_size_iterates_keys_not_len() -> None:
    """size() must count via keys() — _FakeModalDict raises on len()."""
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    store.add(HintKey(plan_signature="s1", intent_hash="i1", url_pattern="x"),
              HintRecord(anchor_text="a"))
    store.add(HintKey(plan_signature="s1", intent_hash="i2", url_pattern="x"),
              HintRecord(anchor_text="b"))
    store.add(HintKey(plan_signature="s2", intent_hash="i1", url_pattern="x"),
              HintRecord(anchor_text="c"))
    assert store.size() == 3  # no TypeError from len()


def test_modal_store_size_excludes_other_tenants() -> None:
    backing = _FakeModalDict()
    a = ModalDictHintStore(backing, tenant_id="A")
    b = ModalDictHintStore(backing, tenant_id="B")
    a.add(HintKey(plan_signature="s", intent_hash="i", url_pattern="x"),
          HintRecord(anchor_text="a"))
    b.add(HintKey(plan_signature="s", intent_hash="i", url_pattern="x"),
          HintRecord(anchor_text="b"))
    assert a.size() == 1
    assert b.size() == 1


def test_modal_store_no_op_on_empty_plan_signature() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    key = HintKey(plan_signature="", intent_hash="ih1", url_pattern="x")
    store.add(key, HintRecord(anchor_text="X"))
    assert store.get(key) == []
    assert store.size() == 0


def test_modal_store_get_missing_returns_empty() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    key = HintKey(plan_signature="never", intent_hash="ih", url_pattern="x")
    assert store.get(key) == []


def test_modal_store_iter_records() -> None:
    store = ModalDictHintStore(_FakeModalDict(), tenant_id="acme")
    k = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    store.add(k, HintRecord(anchor_text="A"))
    store.add(k, HintRecord(anchor_text="B"))
    pairs = list(store.iter_records())
    assert sorted(r.anchor_text for _, r in pairs) == ["A", "B"]


def test_modal_store_read_fault_degrades_to_empty() -> None:
    """A backing that raises on read must yield no hints, not crash."""

    class _Exploding:
        def get(self, key, default=None):  # noqa: ARG002
            raise RuntimeError("modal.Dict down")

        def __setitem__(self, key, value) -> None:  # noqa: ARG002
            raise RuntimeError("modal.Dict down")

        def keys(self):
            raise RuntimeError("modal.Dict down")

    store = ModalDictHintStore(_Exploding(), tenant_id="acme")
    key = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    # add swallows the write fault, get/size degrade to empty — no raise.
    store.add(key, HintRecord(anchor_text="X"))
    assert store.get(key) == []
    assert store.size() == 0


# ── build_hint_store (Modal-path backend factory) ────────────────────


def _fake_dict_factory():
    """Mimic ``modal.Dict.from_name``: same name → same backing object, so
    two workers naming the same dict share state."""
    pool: dict[str, _FakeModalDict] = {}

    def _for(name: str) -> _FakeModalDict:
        return pool.setdefault(name, _FakeModalDict())

    return _for


def test_build_hint_store_default_returns_disk() -> None:
    """No allocator flags → today's production DiskHintStore."""
    store = build_hint_store({}, tenant_id="acme")
    assert isinstance(store, DiskHintStore)


def test_build_hint_store_disabled_returns_null() -> None:
    """``_hint_store_disabled`` → frozen policy (NullHintStore)."""
    store = build_hint_store({"_hint_store_disabled": True}, tenant_id="acme")
    assert isinstance(store, NullHintStore)


def test_build_hint_store_dict_name_returns_modal() -> None:
    """``_hint_store_dict_name`` → ModalDictHintStore over that named Dict."""
    store = build_hint_store(
        {"_hint_store_dict_name": "la-hints"}, tenant_id="acme",
        modal_dict_factory=_fake_dict_factory(),
    )
    assert isinstance(store, ModalDictHintStore)
    key = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    store.add(key, HintRecord(anchor_text="A"))
    assert [r.anchor_text for r in store.get(key)] == ["A"]


def test_build_hint_store_disabled_takes_precedence_over_dict_name() -> None:
    """Frozen wins when both flags are present (first match in the ladder)."""
    store = build_hint_store(
        {"_hint_store_disabled": True, "_hint_store_dict_name": "la-hints"},
        tenant_id="acme", modal_dict_factory=_fake_dict_factory(),
    )
    assert isinstance(store, NullHintStore)


def test_build_hint_store_empty_tenant_returns_null() -> None:
    """No flags + empty tenant → NullHintStore (DiskHintStore forbids it)."""
    store = build_hint_store({}, tenant_id="")
    assert isinstance(store, NullHintStore)


def test_build_hint_store_non_dict_suite_returns_disk() -> None:
    """A malformed (non-dict) suite degrades to the production default."""
    store = build_hint_store(None, tenant_id="acme")
    assert isinstance(store, DiskHintStore)


def test_build_hint_store_modal_open_failure_falls_back_to_disk() -> None:
    """A modal.Dict open fault must not break the run — fall back to disk."""

    def _boom(_name: str):
        raise RuntimeError("modal.Dict.from_name unavailable")

    store = build_hint_store(
        {"_hint_store_dict_name": "la-hints"}, tenant_id="acme",
        modal_dict_factory=_boom,
    )
    assert isinstance(store, DiskHintStore)


def test_build_hint_store_shared_dict_name_is_cross_worker() -> None:
    """Two workers naming the same Dict see each other's anchors — the
    property the live S0 rung depends on."""
    factory = _fake_dict_factory()
    s1 = build_hint_store(
        {"_hint_store_dict_name": "la-run-1"}, tenant_id="t",
        modal_dict_factory=factory,
    )
    s2 = build_hint_store(
        {"_hint_store_dict_name": "la-run-1"}, tenant_id="t",
        modal_dict_factory=factory,
    )
    key = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    s1.add(key, HintRecord(anchor_text="cross-worker"))
    assert [r.anchor_text for r in s2.get(key)] == ["cross-worker"]


# ── extract_anchor_from_env ──────────────────────────────────────────


def test_extract_anchor_returns_none_without_diag() -> None:
    env = MagicMock(spec=[])  # no _last_som_diag
    assert extract_anchor_from_env(env) is None


def test_extract_anchor_returns_none_on_empty_elv_text() -> None:
    env = SimpleNamespace(_last_som_diag={"elv_text": "", "x": 100, "y": 200})
    assert extract_anchor_from_env(env) is None


def test_extract_anchor_returns_text_and_xy() -> None:
    env = SimpleNamespace(_last_som_diag={
        "elv_text": "Show More", "x": 542, "y": 678,
    })
    out = extract_anchor_from_env(env)
    assert out == ("Show More", (542, 678))


def test_extract_anchor_truncates_long_text() -> None:
    env = SimpleNamespace(_last_som_diag={
        "elv_text": "x" * 500, "x": 0, "y": 0,
    })
    out = extract_anchor_from_env(env)
    assert out is not None
    assert len(out[0]) == 200


# ── record_hint_if_eligible ──────────────────────────────────────────


def test_record_skipped_for_null_store() -> None:
    """Cheap short-circuit when store is the NullHintStore."""
    env = SimpleNamespace(
        _last_som_diag={"elv_text": "OK", "x": 0, "y": 0},
        current_url="https://x.com/y",
    )
    record = record_hint_if_eligible(
        store=NullHintStore(), plan_signature="sig", step=_step(),
        step_type="click", success=True, env=env,
    )
    assert record is None


def test_record_skipped_on_failure() -> None:
    env = SimpleNamespace(
        _last_som_diag={"elv_text": "OK", "x": 0, "y": 0},
        current_url="https://x.com/y",
    )
    record = record_hint_if_eligible(
        store=InMemoryHintStore(), plan_signature="sig", step=_step(),
        step_type="click", success=False, env=env,
    )
    assert record is None


def test_record_skipped_for_non_grounding_step_type() -> None:
    env = SimpleNamespace(
        _last_som_diag={"elv_text": "OK", "x": 0, "y": 0},
        current_url="https://x.com/y",
    )
    record = record_hint_if_eligible(
        store=InMemoryHintStore(), plan_signature="sig", step=_step(),
        step_type="navigate", success=True, env=env,
    )
    assert record is None


def test_record_skipped_when_anchor_missing() -> None:
    """Step succeeded but no SoM anchor was captured — skip silently."""
    env = SimpleNamespace(_last_som_diag=None, current_url="https://x.com/y")
    record = record_hint_if_eligible(
        store=InMemoryHintStore(), plan_signature="sig", step=_step(),
        step_type="click", success=True, env=env,
    )
    assert record is None


def test_record_fires_for_grounding_step_with_anchor() -> None:
    store = InMemoryHintStore()
    env = SimpleNamespace(
        _last_som_diag={"elv_text": "Show More", "x": 542, "y": 678},
        current_url="https://boattrader.com/boat/1986-marine/",
    )
    record = record_hint_if_eligible(
        store=store, plan_signature="sig1", step=_step("click Show More", "click"),
        step_type="click", success=True, env=env,
    )
    assert record is not None
    assert record.anchor_text == "Show More"
    # Stored in the right key bucket
    key = hint_key_for("sig1", _step("click Show More", "click"),
                      "https://boattrader.com/boat/1986-marine/")
    out = store.get(key)
    assert len(out) == 1
    assert out[0].anchor_text == "Show More"


def test_record_handles_each_grounding_step_type() -> None:
    """All step types in GROUNDING_STEP_TYPES are eligible."""
    env = SimpleNamespace(
        _last_som_diag={"elv_text": "X", "x": 0, "y": 0},
        current_url="https://x.com/y",
    )
    for step_type in GROUNDING_STEP_TYPES:
        store = InMemoryHintStore()
        record_hint_if_eligible(
            store=store, plan_signature="sig", step=_step("x", step_type),
            step_type=step_type, success=True, env=env,
        )
        assert store.size() == 1, f"{step_type} should record"


# ── apply_hint_overlay ───────────────────────────────────────────────


def test_apply_overlay_no_op_when_store_empty() -> None:
    plan = SimpleNamespace(steps=[_step("click X", "click")])
    n = apply_hint_overlay(plan, store=InMemoryHintStore(), plan_signature="sig")
    assert n == 0
    assert plan.steps[0].hints == {}


def test_apply_overlay_skips_non_grounding_steps() -> None:
    """Stored hint exists but step type is navigate → not applied."""
    store = InMemoryHintStore()
    step = _step("navigate to X", "navigate")
    key = hint_key_for("sig", step, "https://x.com/y")
    store.add(key, HintRecord(anchor_text="Anchor X", confidence=0.9))

    plan = SimpleNamespace(steps=[step])
    n = apply_hint_overlay(
        plan, store=store, plan_signature="sig", start_url="https://x.com/y",
    )
    assert n == 0
    assert plan.steps[0].hints == {}


def test_apply_overlay_stamps_preferred_target() -> None:
    store = InMemoryHintStore()
    step = _step("click Show More", "click")
    key = hint_key_for("sig", step, "https://x.com/y")
    store.add(key, HintRecord(
        anchor_text="Show More", viewport_stage=2, confidence=0.85,
    ))

    plan = SimpleNamespace(steps=[step])
    n = apply_hint_overlay(
        plan, store=store, plan_signature="sig", start_url="https://x.com/y",
    )
    assert n == 1
    assert plan.steps[0].hints["preferred_target_description"] == "Show More"
    assert plan.steps[0].hints["preferred_target_viewport_stage"] == 2


def test_apply_overlay_preserves_operator_hints() -> None:
    """When operator already authored preferred_target_description, the
    overlay must NOT overwrite."""
    store = InMemoryHintStore()
    step = _step("click Show More", "click", hints={
        "preferred_target_description": "Operator override",
        "region": "footer",
    })
    key = hint_key_for("sig", step, "https://x.com/y")
    store.add(key, HintRecord(anchor_text="Stored anchor", confidence=0.9))

    plan = SimpleNamespace(steps=[step])
    apply_hint_overlay(
        plan, store=store, plan_signature="sig", start_url="https://x.com/y",
    )
    # Operator value wins
    assert plan.steps[0].hints["preferred_target_description"] == "Operator override"
    # Existing other hints preserved
    assert plan.steps[0].hints["region"] == "footer"


def test_apply_overlay_skips_low_confidence_records() -> None:
    store = InMemoryHintStore()
    step = _step("click X", "click")
    key = hint_key_for("sig", step, "https://x.com/y")
    store.add(key, HintRecord(anchor_text="Low conf", confidence=0.1))

    plan = SimpleNamespace(steps=[step])
    n = apply_hint_overlay(
        plan, store=store, plan_signature="sig", start_url="https://x.com/y",
    )
    assert n == 0


def test_apply_overlay_picks_highest_confidence() -> None:
    store = InMemoryHintStore()
    step = _step("click X", "click")
    key = hint_key_for("sig", step, "https://x.com/y")
    store.add(key, HintRecord(anchor_text="Old", confidence=0.5))
    store.add(key, HintRecord(anchor_text="New high conf", confidence=0.95))

    plan = SimpleNamespace(steps=[step])
    apply_hint_overlay(
        plan, store=store, plan_signature="sig", start_url="https://x.com/y",
    )
    assert plan.steps[0].hints["preferred_target_description"] == "New high conf"


def test_apply_overlay_no_op_on_empty_plan_signature() -> None:
    store = InMemoryHintStore()
    key = HintKey(plan_signature="sig", intent_hash="ih", url_pattern="x")
    store.add(key, HintRecord(anchor_text="A", confidence=0.9))
    plan = SimpleNamespace(steps=[_step("click X", "click")])
    n = apply_hint_overlay(plan, store=store, plan_signature="")
    assert n == 0


# ── Holo3 prompt integration ─────────────────────────────────────────


def test_holo3_prompt_includes_preferred_target() -> None:
    """The holo3 prompt builder must surface preferred_target_description
    in the Target hints section so the brain actually sees the stored anchor."""
    from mantis_agent.gym.step_handlers.holo3 import _build_scoped_task as build_search_prompt

    step = _step("click Show More", "click", hints={
        "preferred_target_description": "Show More button below Description",
    })
    runner = SimpleNamespace()
    prompt = build_search_prompt(step, runner, step_index=0)
    assert "preferred_target" in prompt
    assert "Show More button below Description" in prompt


def test_holo3_prompt_includes_viewport_stage() -> None:
    from mantis_agent.gym.step_handlers.holo3 import _build_scoped_task as build_search_prompt

    step = _step("click X", "click", hints={
        "preferred_target_description": "X",
        "preferred_target_viewport_stage": 2,
    })
    runner = SimpleNamespace()
    prompt = build_search_prompt(step, runner, step_index=0)
    assert "preferred_target_at_scroll_stage" in prompt
    assert "2" in prompt
