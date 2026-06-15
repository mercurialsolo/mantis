"""Unit tests for the eval-version freeze tool (experiments/holdout/freeze_eval_version.py).

Covers selection logic + the two auth scopes with injected HTTP seams (no network):
read works with the producer key; freeze refuses without an operator session cookie.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "freeze_eval_version",
    Path(__file__).resolve().parent.parent / "experiments/holdout/freeze_eval_version.py",
)
fev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fev)  # type: ignore[union-attr]


CANDIDATES = [
    {"task_spec_id": "boattrader.bt01.v1", "source": "producer"},
    {"task_spec_id": "boattrader.bt01.v1", "source": "producer"},  # dup → collapses
    {"task_spec_id": "boattrader.bt02.v1", "source": "producer"},
    {"task_spec_id": "crm.t04.v1", "source": "producer"},
]


def _get_ok(_url, _headers):
    return 200, {"candidates": CANDIDATES}


# ── selection ──────────────────────────────────────────────────────


def test_select_all_dedupes_and_keeps_order():
    assert fev.select_task_spec_ids(CANDIDATES) == [
        "boattrader.bt01.v1", "boattrader.bt02.v1", "crm.t04.v1"
    ]


def test_select_prefix():
    assert fev.select_task_spec_ids(CANDIDATES, select_prefix="boattrader.") == [
        "boattrader.bt01.v1", "boattrader.bt02.v1"
    ]


def test_select_explicit_subset():
    assert fev.select_task_spec_ids(CANDIDATES, explicit=["crm.t04.v1"]) == ["crm.t04.v1"]


def test_select_explicit_missing_raises():
    with pytest.raises(ValueError, match="no candidate in the pool"):
        fev.select_task_spec_ids(CANDIDATES, explicit=["nope.v1"])


# ── read scope ─────────────────────────────────────────────────────


def test_list_candidates_uses_bearer_key():
    seen = {}

    def get(url, headers):
        seen["url"], seen["headers"] = url, headers
        return 200, {"candidates": CANDIDATES}

    out = fev.list_candidates(base="B", tenant="t", token="KEY", http_get=get)
    assert len(out) == 4
    assert seen["headers"]["Authorization"] == "Bearer KEY"
    assert "tenant=t" in seen["url"]


def test_list_candidates_raises_on_non_200():
    with pytest.raises(RuntimeError, match=r"\[401\]"):
        fev.list_candidates(base="B", tenant="t", token="K", http_get=lambda u, h: (401, "no"))


# ── write scope (freeze) ───────────────────────────────────────────


def test_freeze_requires_session_cookie():
    with pytest.raises(ValueError, match="operator session cookie"):
        fev.freeze_version(
            base="B", tenant="t", name="v", task_spec_ids=["a.v1"],
            session_cookie="", http_post=lambda *a: (200, {}),
        )


def test_freeze_refuses_empty_set():
    with pytest.raises(ValueError, match="empty eval-version"):
        fev.freeze_version(
            base="B", tenant="t", name="v", task_spec_ids=[],
            session_cookie="session=x", http_post=lambda *a: (200, {}),
        )


def test_freeze_posts_with_cookie_and_returns_body():
    captured = {}

    def post(url, headers, body):
        captured["url"], captured["headers"], captured["body"] = url, headers, body
        return 201, {"name": body["name"], "frozen": True, "count": len(body["task_spec_ids"])}

    out = fev.freeze_version(
        base="B", tenant="t", name="mantis-holdout-v1",
        task_spec_ids=["boattrader.bt01.v1", "boattrader.bt02.v1"],
        session_cookie="session=abc", http_post=post, description="first real freeze",
    )
    assert out == {"name": "mantis-holdout-v1", "frozen": True, "count": 2}
    assert captured["headers"]["Cookie"] == "session=abc"
    assert captured["body"]["task_spec_ids"] == ["boattrader.bt01.v1", "boattrader.bt02.v1"]
    assert captured["body"]["description"] == "first real freeze"


def test_freeze_raises_on_401():
    with pytest.raises(RuntimeError, match=r"\[401\]"):
        fev.freeze_version(
            base="B", tenant="t", name="v", task_spec_ids=["a.v1"],
            session_cookie="session=x", http_post=lambda *a: (401, {"detail": "not signed in"}),
        )
