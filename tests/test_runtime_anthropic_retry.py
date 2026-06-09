"""Tests for runtime Anthropic call sites routing through shared retry (#836).

Companion to #832 (decomposer side, shipped in PR #834). The
high-volume runtime call sites — ``brain_claude.query``,
``brain_claude.think``, ``agentic_recovery``, ``grounding.ground`` —
now route through ``AnthropicToolUseClient.post_messages_with_retry``
so a single 5xx / 529 / read-timeout doesn't tank a run mid-step.

Source-level assertions (mirrors the pattern in
``tests/test_decomposer_anthropic_retry.py``) — the retry mechanics
themselves are exercised by the existing ``_anthropic/client.py``
test suite.
"""

from __future__ import annotations

import inspect

import mantis_agent.brain_claude as brain_claude_mod
import mantis_agent.agentic_recovery as agentic_recovery_mod
import mantis_agent.grounding as grounding_mod


# ── brain_claude ────────────────────────────────────────────────────


def test_brain_claude_query_uses_shared_client():
    """``ClaudeBrain.query`` (the cheap text-only call) must route
    through the shared retry client so a transient 5xx doesn't
    silently return ``""``."""
    src = inspect.getsource(brain_claude_mod.ClaudeBrain.query)
    assert "AnthropicToolUseClient" in src
    assert "post_messages_with_retry" in src


def test_brain_claude_think_uses_shared_client():
    """``ClaudeBrain.think`` is the per-step inference call — every
    step pays its latency. Must route through the shared retry
    client AND cap retries at 2 to avoid per-step latency blowing up."""
    src = inspect.getsource(brain_claude_mod.ClaudeBrain.think)
    assert "AnthropicToolUseClient" in src
    assert "post_messages_with_retry" in src
    # Per-step latency control — keep max_attempts low.
    assert "max_attempts=2" in src


def test_brain_claude_think_handles_resp_is_none():
    """Network exhaustion path must not crash with AttributeError on
    ``None.status_code`` — has to return a fallback InferenceResult."""
    src = inspect.getsource(brain_claude_mod.ClaudeBrain.think)
    assert "resp is None" in src or "is None" in src


# ── agentic_recovery ────────────────────────────────────────────────


def test_agentic_recovery_uses_shared_client():
    """Recovery is gated by a recovery-budget; losing one to a
    transient burns budget we can't refund. Must retry."""
    src = inspect.getsource(agentic_recovery_mod)
    assert "AnthropicToolUseClient" in src
    assert "post_messages_with_retry" in src
    assert "max_attempts=2" in src


# ── grounding ───────────────────────────────────────────────────────


def test_grounding_uses_shared_client():
    """Grounding is on the click hot path; a single transient miss
    used to drop the run into ``api_error`` fallback. Must retry
    (with low max_attempts since latency matters). The actual call
    lives in ``ClaudeGrounding._ground_remote`` which the entry-
    point ``ground`` delegates to."""
    src = inspect.getsource(grounding_mod.ClaudeGrounding._ground_remote)
    assert "AnthropicToolUseClient" in src
    assert "post_messages_with_retry" in src
    assert "max_attempts=2" in src


def test_grounding_handles_resp_is_none():
    src = inspect.getsource(grounding_mod.ClaudeGrounding._ground_remote)
    assert "resp is None" in src or "is None" in src
