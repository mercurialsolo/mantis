"""Provider factory + env-var routing (#406 Part 3 wiring).

One function — :func:`build_form_target_provider` — that the runner
(or any caller that builds a :class:`StepContext`) invokes to get the
provider matching the operator's ``MANTIS_FORM_TARGET_PROVIDER``
preference.

Selection contract (env var values):

- ``claude`` / ``""`` / unset → :class:`ClaudeFormTargetProvider`
  (default, stable).
- ``holo3`` → :class:`Holo3FormTargetProvider` with a
  :class:`ClaudeFormTargetProvider` fallback for
  ``verify_dropdown_value`` (Holo3 isn't tuned for prose reads;
  delegating keeps the verifier honest).

Unknown values log a warning and fall back to ``claude`` rather than
crashing — keeps a typo in deploy-time config from halting plans.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .base import FormTargetProvider
from .claude import ClaudeFormTargetProvider
from .holo3 import Holo3FormTargetProvider

if TYPE_CHECKING:
    from .._anthropic.client import AnthropicToolUseClient
    from ..brain_holo3 import Holo3Brain

logger = logging.getLogger(__name__)


def build_form_target_provider(
    *,
    anthropic_client: "AnthropicToolUseClient | None" = None,
    holo3_brain: "Holo3Brain | None" = None,
    override: str | None = None,
) -> FormTargetProvider | None:
    """Build the :class:`FormTargetProvider` selected by env / override.

    Args:
        anthropic_client: Used to construct the Claude provider. When
            ``None`` and ``MANTIS_FORM_TARGET_PROVIDER`` resolves to
            ``claude``, the function returns ``None`` — the caller's
            responsibility to handle the "no client wired" case (the
            form handler does this by falling back to the legacy
            extractor-attached provider).
        holo3_brain: Required when the resolved choice is ``holo3``.
            ``None`` + ``holo3`` selection logs a warning and falls
            back to Claude.
        override: When set, ignore the env var. Tests use this to
            exercise both branches without environ manipulation.

    Returns:
        A :class:`FormTargetProvider` instance, or ``None`` when no
        client is available to satisfy any selection.
    """
    # Don't auto-construct a provider when the caller passed a non-
    # AnthropicToolUseClient (a MagicMock from a test, a stub from an
    # ad-hoc script). Returning ``None`` lets the form handler fall
    # back to ``extractor.find_form_target`` directly — which the test
    # has already mocked. Without this guard the factory would wrap
    # the MagicMock in a real provider and the test's mock would
    # never see the call.
    if anthropic_client is not None:
        from .._anthropic.client import AnthropicToolUseClient
        if not isinstance(anthropic_client, AnthropicToolUseClient):
            return None

    choice = (override or os.environ.get("MANTIS_FORM_TARGET_PROVIDER") or "claude").strip().lower()

    if choice not in {"claude", "holo3"}:
        logger.warning(
            "MANTIS_FORM_TARGET_PROVIDER=%r is not recognised; "
            "falling back to claude",
            choice,
        )
        choice = "claude"

    if choice == "holo3":
        if holo3_brain is None:
            logger.warning(
                "MANTIS_FORM_TARGET_PROVIDER=holo3 selected but no Holo3 "
                "brain was wired; falling back to claude"
            )
        else:
            claude_fallback = (
                ClaudeFormTargetProvider(anthropic_client)
                if anthropic_client is not None else None
            )
            logger.info(
                "form_target_provider=holo3 (verify_dropdown fallback: %s)",
                "claude" if claude_fallback else "none",
            )
            return Holo3FormTargetProvider(
                holo3_brain, claude_fallback=claude_fallback,
            )
        choice = "claude"

    # choice == "claude"
    if anthropic_client is None:
        return None
    logger.info("form_target_provider=claude")
    return ClaudeFormTargetProvider(anthropic_client)
