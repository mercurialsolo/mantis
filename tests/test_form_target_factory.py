"""Env-var routing for :func:`build_form_target_provider` (#406).

The function picks between :class:`ClaudeFormTargetProvider` and
:class:`Holo3FormTargetProvider` based on
``MANTIS_FORM_TARGET_PROVIDER``. Tests pin every branch including the
soft-fail paths (unknown value, holo3 requested but no brain, claude
requested but no client) — these failure modes deserve coverage
because mis-configured env vars at deploy time should NOT crash the
runner.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from mantis_agent._anthropic.client import AnthropicToolUseClient
from mantis_agent.form_targeting import (
    ClaudeFormTargetProvider,
    Holo3FormTargetProvider,
)
from mantis_agent.form_targeting.factory import build_form_target_provider


def _client() -> AnthropicToolUseClient:
    return AnthropicToolUseClient(api_key="k", model="m")


def _holo3() -> MagicMock:
    """Stand-in Holo3 brain. The factory only calls the constructor
    so a MagicMock is sufficient — no need to spin up the real class
    with its torch / transformers transitive deps."""
    return MagicMock()


# ── default branch ─────────────────────────────────────────────────


def test_default_returns_claude_provider(monkeypatch) -> None:
    """Env var unset → claude. The stable default."""
    monkeypatch.delenv("MANTIS_FORM_TARGET_PROVIDER", raising=False)
    prov = build_form_target_provider(anthropic_client=_client())
    assert isinstance(prov, ClaudeFormTargetProvider)


def test_empty_string_returns_claude_provider(monkeypatch) -> None:
    """Some shells set unset vars to ``""`` — still claude."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "")
    prov = build_form_target_provider(anthropic_client=_client())
    assert isinstance(prov, ClaudeFormTargetProvider)


def test_claude_explicit(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "claude")
    prov = build_form_target_provider(anthropic_client=_client())
    assert isinstance(prov, ClaudeFormTargetProvider)


# ── holo3 branch ───────────────────────────────────────────────────


def test_holo3_requested_with_brain_returns_holo3_provider(monkeypatch) -> None:
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "holo3")
    prov = build_form_target_provider(
        anthropic_client=_client(),
        holo3_brain=_holo3(),
    )
    assert isinstance(prov, Holo3FormTargetProvider)


def test_holo3_with_claude_client_wires_verify_fallback(monkeypatch) -> None:
    """A Holo3 provider built alongside a Claude client gets a
    :class:`ClaudeFormTargetProvider` as ``verify_dropdown_value``
    fallback — Holo3 isn't tuned for prose reads."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "holo3")
    prov = build_form_target_provider(
        anthropic_client=_client(),
        holo3_brain=_holo3(),
    )
    assert isinstance(prov, Holo3FormTargetProvider)
    assert isinstance(prov._claude_fallback, ClaudeFormTargetProvider)


def test_holo3_without_client_has_no_verify_fallback(monkeypatch) -> None:
    """No anthropic client → no fallback. The Holo3 provider degrades
    ``verify_dropdown_value`` to returning ``None`` rather than
    crashing."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "holo3")
    prov = build_form_target_provider(
        anthropic_client=None,
        holo3_brain=_holo3(),
    )
    assert isinstance(prov, Holo3FormTargetProvider)
    assert prov._claude_fallback is None


# ── soft-fail branches ─────────────────────────────────────────────


def test_holo3_requested_without_brain_falls_back_to_claude(monkeypatch, caplog) -> None:
    """A common deploy slip: env var says holo3, but the brain isn't
    wired (e.g. a callsite still constructs a Claude-only runner).
    The factory falls back to Claude rather than returning ``None``
    or crashing."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "holo3")
    caplog.set_level(logging.WARNING, logger="mantis_agent.form_targeting.factory")
    prov = build_form_target_provider(
        anthropic_client=_client(),
        holo3_brain=None,
    )
    assert isinstance(prov, ClaudeFormTargetProvider)
    assert any("falling back to claude" in r.message for r in caplog.records)


def test_unknown_value_falls_back_to_claude(monkeypatch, caplog) -> None:
    """Typo in env var (``calude``, ``hollow3``) → claude, with a
    warning so operators notice."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "calude")
    caplog.set_level(logging.WARNING, logger="mantis_agent.form_targeting.factory")
    prov = build_form_target_provider(anthropic_client=_client())
    assert isinstance(prov, ClaudeFormTargetProvider)
    assert any("not recognised" in r.message for r in caplog.records)


def test_claude_without_client_returns_none(monkeypatch) -> None:
    """No anthropic client AND default selection → no provider.
    Returning ``None`` lets the form handler fall back to the
    extractor's compat shims (the legacy path before the provider
    existed)."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "claude")
    prov = build_form_target_provider(anthropic_client=None)
    assert prov is None


# ── override argument (test convenience) ───────────────────────────


def test_override_arg_bypasses_env(monkeypatch) -> None:
    """The ``override`` keyword lets tests exercise both branches
    without environ manipulation. Useful when the test process picks
    up a stale env var from the shell."""
    monkeypatch.setenv("MANTIS_FORM_TARGET_PROVIDER", "claude")
    prov = build_form_target_provider(
        anthropic_client=_client(),
        holo3_brain=_holo3(),
        override="holo3",
    )
    assert isinstance(prov, Holo3FormTargetProvider)
