"""#643 Phase 1 — trajectory hint memory store + types.

Pins the foundation that Phase 1b's recording / injection hooks
build on. No runner integration yet.

Coverage:
    - HintRecord schema (defaults, frozen)
    - HintKey shape + ``as_tuple``
    - NullHintStore is a complete no-op
    - InMemoryHintStore: add / get / size; LRU eviction at cap;
      per-key isolation; newest-first read order
    - intent_hash_for: stable; intent prose drives the key, not
      step_index
    - url_pattern_for: registrable host + first segment; handles
      empty / malformed URLs
    - hint_key_for: end-to-end bundle
"""

from __future__ import annotations

from types import SimpleNamespace

from mantis_agent.gym.hint_memory import (
    HintKey,
    HintRecord,
    InMemoryHintStore,
    NullHintStore,
    hint_key_for,
    intent_hash_for,
    url_pattern_for,
)


# ── HintRecord ──────────────────────────────────────────────────────


def test_hint_record_minimal_construction():
    r = HintRecord(anchor_text="Show More")
    assert r.anchor_text == "Show More"
    assert r.anchor_xy_offset == (0, 0)
    assert r.viewport_stage == 0
    assert r.confidence == 1.0
    assert r.source_url == ""
    assert r.recorded_at > 0  # default_factory ran


def test_hint_record_is_frozen():
    from dataclasses import FrozenInstanceError
    r = HintRecord(anchor_text="x")
    try:
        r.anchor_text = "y"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("HintRecord should be frozen")


# ── HintKey ─────────────────────────────────────────────────────────


def test_hint_key_as_tuple_round_trips():
    k = HintKey(plan_signature="abc", intent_hash="def", url_pattern="x.com/y")
    assert k.as_tuple() == ("abc", "def", "x.com/y")


def test_hint_key_distinct_axes_produce_distinct_keys():
    a = HintKey(plan_signature="p1", intent_hash="i1", url_pattern="u1")
    b = HintKey(plan_signature="p2", intent_hash="i1", url_pattern="u1")
    assert a != b
    assert a.as_tuple() != b.as_tuple()


# ── NullHintStore ───────────────────────────────────────────────────


def test_null_store_is_complete_no_op():
    s = NullHintStore()
    k = HintKey(plan_signature="p", intent_hash="i", url_pattern="u")
    s.add(k, HintRecord(anchor_text="x"))
    assert s.get(k) == []
    assert s.size() == 0


# ── InMemoryHintStore ───────────────────────────────────────────────


def test_in_memory_store_round_trips_record():
    s = InMemoryHintStore()
    k = HintKey(plan_signature="p", intent_hash="i", url_pattern="u")
    rec = HintRecord(anchor_text="Show More", anchor_xy_offset=(10, -20))
    s.add(k, rec)
    out = s.get(k)
    assert len(out) == 1
    assert out[0].anchor_text == "Show More"
    assert out[0].anchor_xy_offset == (10, -20)


def test_in_memory_store_get_returns_newest_first():
    s = InMemoryHintStore()
    k = HintKey(plan_signature="p", intent_hash="i", url_pattern="u")
    s.add(k, HintRecord(anchor_text="first"))
    s.add(k, HintRecord(anchor_text="second"))
    s.add(k, HintRecord(anchor_text="third"))
    out = s.get(k)
    assert [r.anchor_text for r in out] == ["third", "second", "first"]


def test_in_memory_store_lru_evicts_oldest_at_cap():
    s = InMemoryHintStore(max_per_key=3)
    k = HintKey(plan_signature="p", intent_hash="i", url_pattern="u")
    for label in ["a", "b", "c", "d", "e"]:
        s.add(k, HintRecord(anchor_text=label))
    out = s.get(k)
    # Cap = 3; only the 3 newest survive.
    assert [r.anchor_text for r in out] == ["e", "d", "c"]


def test_in_memory_store_keys_isolated():
    s = InMemoryHintStore()
    k1 = HintKey(plan_signature="p1", intent_hash="i", url_pattern="u")
    k2 = HintKey(plan_signature="p2", intent_hash="i", url_pattern="u")
    s.add(k1, HintRecord(anchor_text="k1-rec"))
    s.add(k2, HintRecord(anchor_text="k2-rec"))
    assert [r.anchor_text for r in s.get(k1)] == ["k1-rec"]
    assert [r.anchor_text for r in s.get(k2)] == ["k2-rec"]
    assert s.size() == 2


def test_in_memory_store_size_counts_all_buckets():
    s = InMemoryHintStore(max_per_key=2)
    k1 = HintKey(plan_signature="p1", intent_hash="i", url_pattern="u")
    k2 = HintKey(plan_signature="p2", intent_hash="i", url_pattern="u")
    # 2 keys × 2 records each = 4.
    for k in (k1, k2):
        s.add(k, HintRecord(anchor_text="r1"))
        s.add(k, HintRecord(anchor_text="r2"))
    assert s.size() == 4


def test_in_memory_store_max_per_key_floor_is_one():
    """max_per_key=0 / negative → clamps to 1 so the store doesn't
    accidentally become a no-op when an operator misconfigures it."""
    s = InMemoryHintStore(max_per_key=0)
    k = HintKey(plan_signature="p", intent_hash="i", url_pattern="u")
    s.add(k, HintRecord(anchor_text="a"))
    s.add(k, HintRecord(anchor_text="b"))
    out = s.get(k)
    assert [r.anchor_text for r in out] == ["b"]  # only newest survives


def test_in_memory_store_iter_records():
    s = InMemoryHintStore()
    k1 = HintKey(plan_signature="p1", intent_hash="i", url_pattern="u")
    k2 = HintKey(plan_signature="p2", intent_hash="i", url_pattern="u")
    s.add(k1, HintRecord(anchor_text="a"))
    s.add(k2, HintRecord(anchor_text="b"))
    pairs = list(s.iter_records())
    keys = {p[0].plan_signature for p in pairs}
    anchors = {p[1].anchor_text for p in pairs}
    assert keys == {"p1", "p2"}
    assert anchors == {"a", "b"}


# ── intent_hash_for ─────────────────────────────────────────────────


def test_intent_hash_for_stable_across_calls():
    step = SimpleNamespace(intent="Click the Show More toggle", type="click")
    a = intent_hash_for(step)
    b = intent_hash_for(step)
    assert a == b
    assert len(a) == 12
    assert all(c in "0123456789abcdef" for c in a)


def test_intent_hash_for_varies_with_intent():
    s1 = SimpleNamespace(intent="Click Show More", type="click")
    s2 = SimpleNamespace(intent="Click Show Less", type="click")
    assert intent_hash_for(s1) != intent_hash_for(s2)


def test_intent_hash_for_varies_with_type():
    """Same prose but different step.type → different keys
    (a ``click`` of "Show More" is a different action from a
    ``detect_visible`` for the same label)."""
    s1 = SimpleNamespace(intent="Show More", type="click")
    s2 = SimpleNamespace(intent="Show More", type="detect_visible")
    assert intent_hash_for(s1) != intent_hash_for(s2)


def test_intent_hash_for_handles_missing_attrs():
    """Robust against partially-filled step dicts — empty fields
    still produce a stable non-empty hash."""
    s = SimpleNamespace()  # no intent, no type
    h = intent_hash_for(s)
    assert h
    assert len(h) == 12


def test_intent_hash_for_does_not_depend_on_step_index():
    """Plan refactors (insert / reorder) shift indices but
    rarely rewrite intent prose. Same prose ⇒ same key."""
    s1 = SimpleNamespace(intent="Show More", type="click", step_index=2)
    s2 = SimpleNamespace(intent="Show More", type="click", step_index=5)
    assert intent_hash_for(s1) == intent_hash_for(s2)


# ── url_pattern_for ─────────────────────────────────────────────────


def test_url_pattern_for_boattrader_detail():
    assert url_pattern_for(
        "https://www.boattrader.com/boat/1986-marine-trader-europa-10167773/"
    ) == "boattrader.com/boat"


def test_url_pattern_for_strips_www():
    assert url_pattern_for("https://www.lu.ma/event/abc") == "lu.ma/event"


def test_url_pattern_for_no_path_returns_host_only():
    assert url_pattern_for("https://example.com/") == "example.com"
    assert url_pattern_for("https://example.com") == "example.com"


def test_url_pattern_for_handles_query_and_fragment():
    """First-path-segment extraction stops at ``?`` and ``#`` so
    ``/contacts?status=active`` and ``/contacts#tab`` both yield
    ``example.com/contacts``."""
    assert url_pattern_for("https://example.com/contacts?status=active") == "example.com/contacts"
    assert url_pattern_for("https://example.com/contacts#section") == "example.com/contacts"


def test_url_pattern_for_empty_and_malformed():
    assert url_pattern_for("") == ""
    assert url_pattern_for("not a url") == ""
    assert url_pattern_for("ftp://example.com") == ""


def test_url_pattern_for_case_insensitive_host():
    """Hosts vary in case; the pattern is normalised to lowercase
    so ``BoatTrader.com`` and ``boattrader.com`` match the same key."""
    assert url_pattern_for(
        "https://BoatTrader.COM/Boat/1986-marine-trader/"
    ) == "boattrader.com/boat"


# ── hint_key_for end-to-end ────────────────────────────────────────


def test_hint_key_for_bundles_all_axes():
    step = SimpleNamespace(intent="Click Show More", type="click")
    k = hint_key_for(
        plan_signature="boattrader-v8",
        step=step,
        url="https://www.boattrader.com/boat/1986-marine-trader/",
    )
    assert k.plan_signature == "boattrader-v8"
    assert k.intent_hash == intent_hash_for(step)
    assert k.url_pattern == "boattrader.com/boat"


def test_hint_key_for_handles_empty_args():
    step = SimpleNamespace()
    k = hint_key_for(plan_signature="", step=step, url="")
    assert k.plan_signature == ""
    assert k.url_pattern == ""
    # intent_hash never empty (hash of empty input is deterministic).
    assert k.intent_hash
