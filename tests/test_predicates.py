"""Tests for #291 predicate grammar + evaluators (gym/predicates.py).

Covers parsing both surface forms (structured JSON, free-form Predicted:),
evaluator semantics including the "unevaluable" case (result=None), and the
world_model_error aggregate.
"""

from __future__ import annotations

import pytest

from mantis_agent.gym.predicates import (
    ObservationContext,
    Predicate,
    PredicateResult,
    evaluate_all,
    evaluate_predicate,
    parse_predicates,
    world_model_error,
)


# ── parse_predicates: structured JSON surface ──────────────────────────


def test_parse_structured_json_block() -> None:
    text = '{"expected": ["url_contains:/checkout", "title_changed"]}'
    out = parse_predicates(text)
    assert [p.kind for p in out] == ["url_contains", "title_changed"]
    assert out[0].arg == "/checkout"
    assert out[1].arg is None


def test_parse_json_inside_prose() -> None:
    text = (
        "I'll click checkout. "
        '{"expected": ["url_contains:/checkout"]} '
        "Then proceed."
    )
    out = parse_predicates(text)
    assert len(out) == 1
    assert out[0].kind == "url_contains"
    assert out[0].arg == "/checkout"


def test_parse_json_drops_unknown_kinds_silently() -> None:
    text = '{"expected": ["url_contains:/x", "lol_unknown:foo", "frame_changed"]}'
    out = parse_predicates(text)
    assert [p.kind for p in out] == ["url_contains", "frame_changed"]


def test_parse_json_handles_empty_expected_list() -> None:
    assert parse_predicates('{"expected": []}') == []


def test_parse_json_falls_through_to_freeform_on_invalid_json() -> None:
    # Malformed JSON shouldn't crash; falls through to Predicted: parser.
    text = '{"expected": [malformed} \nPredicted: url_changed'
    out = parse_predicates(text)
    assert len(out) == 1
    assert out[0].kind == "url_changed"


# ── parse_predicates: free-form Predicted: surface ─────────────────────


def test_parse_freeform_predicted_line() -> None:
    text = "click(x=1, y=1)\nPredicted: url_contains:/detail, title_changed"
    out = parse_predicates(text)
    assert [p.kind for p in out] == ["url_contains", "title_changed"]


def test_parse_freeform_drops_prose_tokens() -> None:
    # Prose like "the page navigates" doesn't match any kind — dropped.
    text = "Predicted: the page navigates, title_changed, lol_unknown:foo"
    out = parse_predicates(text)
    assert [p.kind for p in out] == ["title_changed"]


def test_parse_returns_empty_on_no_signal() -> None:
    assert parse_predicates("") == []
    assert parse_predicates("just some thinking text") == []
    assert parse_predicates("Predicted: ") == []


def test_parse_strips_quotes_from_tokens() -> None:
    text = '{"expected": ["url_contains:/x"]}'
    out = parse_predicates(text)
    assert out[0].arg == "/x"


# ── evaluate_predicate: url predicates ─────────────────────────────────


def test_url_contains_true() -> None:
    p = Predicate("url_contains", "/checkout", "url_contains:/checkout")
    ctx = ObservationContext(url="https://shop.test/checkout?step=1")
    r = evaluate_predicate(p, ctx)
    assert r.result is True


def test_url_contains_false() -> None:
    p = Predicate("url_contains", "/checkout", "url_contains:/checkout")
    ctx = ObservationContext(url="https://shop.test/cart")
    r = evaluate_predicate(p, ctx)
    assert r.result is False


def test_url_contains_unevaluable_when_no_url() -> None:
    p = Predicate("url_contains", "/checkout", "url_contains:/checkout")
    r = evaluate_predicate(p, ObservationContext())
    assert r.result is None


def test_url_changed_true() -> None:
    p = Predicate("url_changed", None, "url_changed")
    ctx = ObservationContext(url="https://x.test/b", prev_url="https://x.test/a")
    assert evaluate_predicate(p, ctx).result is True


def test_url_changed_false() -> None:
    p = Predicate("url_changed", None, "url_changed")
    ctx = ObservationContext(url="https://x.test/a", prev_url="https://x.test/a")
    assert evaluate_predicate(p, ctx).result is False


def test_url_unchanged_true() -> None:
    p = Predicate("url_unchanged", None, "url_unchanged")
    ctx = ObservationContext(url="https://x.test/a", prev_url="https://x.test/a")
    assert evaluate_predicate(p, ctx).result is True


def test_url_equals_exact_match() -> None:
    p = Predicate("url_equals", "https://x.test/a", "url_equals:https://x.test/a")
    ctx = ObservationContext(url="https://x.test/a")
    assert evaluate_predicate(p, ctx).result is True
    ctx2 = ObservationContext(url="https://x.test/a?q=1")
    assert evaluate_predicate(p, ctx2).result is False


# ── evaluate_predicate: title predicates ───────────────────────────────


def test_title_contains() -> None:
    p = Predicate("title_contains", "Checkout", "title_contains:Checkout")
    ctx = ObservationContext(title="Checkout — Shop")
    assert evaluate_predicate(p, ctx).result is True
    ctx2 = ObservationContext(title="Cart — Shop")
    assert evaluate_predicate(p, ctx2).result is False


def test_title_changed() -> None:
    p = Predicate("title_changed", None, "title_changed")
    ctx = ObservationContext(title="B", prev_title="A")
    assert evaluate_predicate(p, ctx).result is True
    ctx2 = ObservationContext(title="A", prev_title="A")
    assert evaluate_predicate(p, ctx2).result is False


# ── evaluate_predicate: focus predicates ───────────────────────────────


def test_field_focused_any_true() -> None:
    p = Predicate("field_focused", None, "field_focused")
    ctx = ObservationContext(focused_input={"name": "email"})
    assert evaluate_predicate(p, ctx).result is True


def test_field_focused_any_false() -> None:
    p = Predicate("field_focused", None, "field_focused")
    ctx = ObservationContext(focused_input=None)
    assert evaluate_predicate(p, ctx).result is False


def test_field_focused_named_match_by_name() -> None:
    p = Predicate("field_focused", "email", "field_focused:email")
    ctx = ObservationContext(focused_input={"name": "email"})
    assert evaluate_predicate(p, ctx).result is True


def test_field_focused_named_match_by_id() -> None:
    p = Predicate("field_focused", "#login-email", "field_focused:#login-email")
    ctx = ObservationContext(focused_input={"id": "login-email"})
    assert evaluate_predicate(p, ctx).result is True


def test_field_focused_named_no_match() -> None:
    p = Predicate("field_focused", "password", "field_focused:password")
    ctx = ObservationContext(focused_input={"name": "email"})
    assert evaluate_predicate(p, ctx).result is False


def test_field_unfocused() -> None:
    p = Predicate("field_unfocused", None, "field_unfocused")
    assert evaluate_predicate(p, ObservationContext()).result is True
    ctx = ObservationContext(focused_input={"name": "x"})
    assert evaluate_predicate(p, ctx).result is False


# ── evaluate_predicate: frame predicates ───────────────────────────────


def test_frame_changed_true() -> None:
    p = Predicate("frame_changed", None, "frame_changed")
    ctx = ObservationContext(frame_hash="aaaa", prev_frame_hash="bbbb")
    assert evaluate_predicate(p, ctx).result is True


def test_frame_changed_false() -> None:
    p = Predicate("frame_changed", None, "frame_changed")
    ctx = ObservationContext(frame_hash="aaaa", prev_frame_hash="aaaa")
    assert evaluate_predicate(p, ctx).result is False


def test_frame_changed_unevaluable_at_first_step() -> None:
    p = Predicate("frame_changed", None, "frame_changed")
    ctx = ObservationContext(frame_hash="aaaa", prev_frame_hash="")
    assert evaluate_predicate(p, ctx).result is None


def test_frame_stable_true() -> None:
    p = Predicate("frame_stable", None, "frame_stable")
    ctx = ObservationContext(frame_hash="aaaa", prev_frame_hash="aaaa")
    assert evaluate_predicate(p, ctx).result is True


# ── evaluate_predicate: best-effort kinds return None ──────────────────


@pytest.mark.parametrize(
    "kind,arg",
    [
        ("element_appears", "Submit"),
        ("element_disappears", "#login"),
        ("modal_opens", None),
        ("modal_closes", None),
    ],
)
def test_dom_predicates_return_none_today(kind: str, arg: str | None) -> None:
    raw = f"{kind}:{arg}" if arg else kind
    p = Predicate(kind, arg, raw)
    r = evaluate_predicate(p, ObservationContext(url="https://x.test"))
    assert r.result is None
    assert "no DOM/OCR signal" in r.reason


# ── world_model_error aggregate ────────────────────────────────────────


def test_world_model_error_zero_when_all_correct() -> None:
    results = [
        PredicateResult("url_changed", True),
        PredicateResult("title_changed", True),
    ]
    assert world_model_error(results) == 0.0


def test_world_model_error_one_when_all_wrong() -> None:
    results = [
        PredicateResult("url_changed", False),
        PredicateResult("title_changed", False),
    ]
    assert world_model_error(results) == 1.0


def test_world_model_error_excludes_unevaluable() -> None:
    # 1 right, 1 wrong, 1 unevaluable => error = 1/2 (None excluded).
    results = [
        PredicateResult("url_changed", True),
        PredicateResult("title_changed", False),
        PredicateResult("modal_opens", None),
    ]
    assert world_model_error(results) == 0.5


def test_world_model_error_none_when_nothing_evaluable() -> None:
    results = [
        PredicateResult("modal_opens", None),
        PredicateResult("element_appears:x", None),
    ]
    assert world_model_error(results) is None


def test_world_model_error_empty_input() -> None:
    assert world_model_error([]) is None


# ── evaluate_all integration ───────────────────────────────────────────


def test_evaluate_all_round_trips_predicate_strings() -> None:
    preds = parse_predicates(
        '{"expected": ["url_changed", "field_focused:email", "frame_changed"]}',
    )
    ctx = ObservationContext(
        url="https://x.test/b",
        prev_url="https://x.test/a",
        focused_input={"name": "email"},
        frame_hash="cccc",
        prev_frame_hash="bbbb",
    )
    results = evaluate_all(preds, ctx)
    assert [r.predicate for r in results] == [
        "url_changed", "field_focused:email", "frame_changed",
    ]
    assert all(r.result is True for r in results)
