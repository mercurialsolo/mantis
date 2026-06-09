"""Tests for decomposer Anthropic retry (#832).

User feedback: "Sometimes execution fails before useful output because
of Anthropic/API timeout during Mantis decomposition/runtime."

Pre-fix the decomposer issued a raw ``requests.post`` and ``raise
RuntimeError`` on any non-200 status — a single 529 Overloaded killed
the whole run. Post-fix it routes through
``AnthropicToolUseClient.post_messages_with_retry`` which already
handles 429 / 502 / 503 / 504 / 529 plus network timeouts.

We assert at the wiring level (decomposer uses the shared client),
not by simulating Anthropic — the retry mechanics are covered by the
existing ``_anthropic/client.py`` test suite.
"""

from __future__ import annotations

import inspect

from mantis_agent import plan_decomposer


def test_decomposer_routes_through_shared_retry_client():
    """The decomposer module must call
    ``AnthropicToolUseClient.post_messages_with_retry`` rather than
    a raw ``requests.post`` so it inherits the retry policy.

    The actual Anthropic call lives in ``decompose_text`` (with
    ``decompose`` as a thin file-loader shim that delegates to it),
    so that's where we look.
    """
    src = inspect.getsource(plan_decomposer.PlanDecomposer.decompose_text)
    assert "AnthropicToolUseClient" in src, (
        "PlanDecomposer.decompose_text must use AnthropicToolUseClient "
        "so a single Anthropic 5xx / 529 / timeout doesn't kill the run."
    )
    assert "post_messages_with_retry" in src, (
        "PlanDecomposer.decompose_text must invoke post_messages_with_retry "
        "so the existing retry policy applies."
    )


def test_decomposer_does_not_raw_post_to_anthropic():
    """Defensive: any raw ``requests.post(... anthropic.com ...)`` call
    in the decomposer would bypass the retry. The shared client OWNS
    the URL constant, so a grep for the URL outside the shared client
    should turn up nothing in the decomposer module."""
    src = inspect.getsource(plan_decomposer)
    # The shared client's import is fine; what we're guarding against
    # is a ``requests.post("https://api.anthropic.com/v1/messages", ...)``
    # invocation.
    anthropic_url = "https://api.anthropic.com/v1/messages"
    # The URL may appear in comments / docstrings — bound on context.
    matches = [
        line for line in src.splitlines()
        if anthropic_url in line and "post" in line.lower()
        and "#" not in line.split("requests.post")[0]
    ]
    # Allow the shared-client import line; ban a raw POST.
    for line in matches:
        assert "requests.post" not in line, (
            f"decomposer should not call requests.post on Anthropic "
            f"directly; bypasses retry. Line: {line.strip()!r}"
        )


def test_decomposer_handles_resp_is_none_path():
    """The retry helper returns ``None`` when every attempt raised a
    network exception. The decomposer must raise a clear error rather
    than ``AttributeError`` on ``None.status_code``."""
    src = inspect.getsource(plan_decomposer.PlanDecomposer.decompose_text)
    assert "resp is None" in src or "is None" in src, (
        "decomposer must explicitly handle the None return from "
        "post_messages_with_retry (network exhaustion)."
    )
