"""Cover the `decompose` flag dispatch in BasetenCUARuntime._task_suite_from_payload.

We don't fully instantiate the runtime (it needs GPU env, secrets, llama.cpp);
instead we exercise the dispatch logic via the unbound method on a lightweight
stub, which is sufficient to lock the contract.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock  # noqa: F401  (used by helpers)


def _runtime_class():
    return importlib.import_module("mantis_agent.baseten_server.runtime").BasetenCUARuntime


def _make_runtime_stub(resolved_path: Path | None = None):
    """Build an instance that skips __init__ so we can exercise dispatch only."""
    runtime_cls = _runtime_class()
    inst = runtime_cls.__new__(runtime_cls)
    inst._resolve_path = MagicMock(return_value=resolved_path or Path("/nonexistent"))
    inst._micro_suite_from_text = MagicMock(return_value={"tasks": [], "_micro_plan": []})
    inst._micro_suite_from_path = MagicMock(return_value={"tasks": [], "_micro_plan": []})
    return inst


def _call_dispatch(payload: dict[str, Any], *, resolved_path: Path | None = None) -> dict[str, Any]:
    inst = _make_runtime_stub(resolved_path)
    return inst._task_suite_from_payload(payload)


def test_plan_text_with_decompose_false_skips_decomposer() -> None:
    payload = {
        "plan_text": "Step 1: Open browser. Step 2: ...",
        "decompose": False,
        "start_url": "https://example.com",
    }
    suite = _call_dispatch(payload)
    assert suite["tasks"][0]["intent"] == payload["plan_text"]
    assert suite["tasks"][0]["start_url"] == "https://example.com"
    assert "_micro_plan" not in suite


def test_plan_text_default_uses_decomposer() -> None:
    """When decompose flag is omitted (default True), plan_text routes to the
    decomposer path. We don't actually run Claude — assert the helper was called.
    """
    inst = _make_runtime_stub()
    inst._task_suite_from_payload({"plan_text": "anything"})
    inst._micro_suite_from_text.assert_called_once()


def test_task_suite_shape_ignores_decompose_flag() -> None:
    """task_suite always runs verbatim; the flag is a no-op for this shape."""
    payload = {
        "task_suite": {"tasks": [{"intent": "x", "task_id": "t"}]},
        "decompose": False,
    }
    out = _call_dispatch(payload)
    assert out["tasks"][0]["intent"] == "x"


def test_txt_micro_path_with_decompose_false_uses_raw_text(tmp_path: Path) -> None:
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("Long English plan body.")
    payload = {"micro": str(plan_file), "decompose": False}
    suite = _call_dispatch(payload, resolved_path=plan_file)
    assert suite["tasks"][0]["intent"] == "Long English plan body."
