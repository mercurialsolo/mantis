"""Tests for #536 — early session metadata flush so the Augur Runs
list shows the ``Model`` column live (not only after the run halts).

The fix: ``AugurAdapter.__init__`` calls
``_flush_session_metadata_to_stream()`` right after the SDK session
opens. That posts a session-only ``trace.json`` (session block +
empty steps) to the streaming sink, landing tags including ``model``
on the workspace within one poll cycle.

No-op behavior contracts to preserve:

* Streaming not configured (``AUGUR_DSN`` unset) → no put_trace call,
  no error
* Adapter disabled (``MANTIS_AUGUR_DISABLED=1``) → wrapper never runs
* SDK exception inside put_trace → swallowed at WARN/DEBUG (telemetry
  never breaks runs)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("augur_sdk")

from mantis_agent.observability.augur import AugurAdapter


def test_flush_session_metadata_called_at_init_with_active_stream(
    monkeypatch, tmp_path: Path,
):
    """When the SDK session has an active ``_stream`` (i.e.
    ``AUGUR_DSN`` is set), the adapter's init must call
    ``put_trace(session_only_payload)`` once. Tags (including
    ``model``) must be present on the session block."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="early_flush_v1", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "Holo3-35B-A3B"},
    )
    # Inject a fake stream so put_trace becomes observable.
    fake_stream = MagicMock()
    a._session._stream = fake_stream
    # Re-fire the flush — first call in __init__ ran with _stream=None
    # (the SDK only opens _stream when AUGUR_DSN is set). For testing
    # we drive the helper directly with the stream now in place.
    a._flush_session_metadata_to_stream()

    fake_stream.put_trace.assert_called_once()
    payload = fake_stream.put_trace.call_args.args[0]
    assert "session" in payload
    assert "steps" in payload
    assert payload["steps"] == [], (
        "Steps must be empty at session-open (no steps recorded yet)"
    )
    # Model tag must be on the session payload — that's the whole
    # point of the fix.
    tags = payload["session"].get("tags", {})
    assert tags.get("model") == "Holo3-35B-A3B", (
        f"model tag missing from flushed session payload — tags={tags!r}"
    )
    a.close(status="completed")


def test_flush_session_metadata_noop_when_stream_absent(
    monkeypatch, tmp_path: Path,
):
    """No streaming sink (``AUGUR_DSN`` not set) → no put_trace call,
    no error. The on-disk bundle path doesn't need an early flush
    because there's no live consumer."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.delenv("AUGUR_DSN", raising=False)
    a = AugurAdapter(
        run_id="early_flush_no_stream", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "claude-opus-4-7"},
    )
    # The SDK won't have created a _stream when AUGUR_DSN isn't set.
    assert a._session._stream is None
    # Calling the helper explicitly must still be a clean no-op.
    a._flush_session_metadata_to_stream()  # MUST NOT raise
    a.close(status="completed")


def test_flush_session_metadata_noop_when_adapter_disabled(
    monkeypatch, tmp_path: Path,
):
    """Disabled adapter → no session, no flush attempted, no error."""
    monkeypatch.setenv("MANTIS_AUGUR_DISABLED", "1")
    a = AugurAdapter(
        run_id="early_flush_disabled", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "x"},
    )
    assert not a.active
    # MUST NOT raise.
    a._flush_session_metadata_to_stream()


def test_flush_session_metadata_swallows_stream_exception(
    monkeypatch, tmp_path: Path,
):
    """If put_trace raises (e.g. server 5xx), the wrapper must swallow
    it. Telemetry never breaks runs. Per Augur spec §4.3."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="early_flush_err", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "x"},
    )
    fake_stream = MagicMock()
    fake_stream.put_trace.side_effect = RuntimeError(
        "simulated 503 from Augur server"
    )
    a._session._stream = fake_stream
    # MUST NOT raise even when the stream call blows up.
    a._flush_session_metadata_to_stream()
    a.close(status="completed")


def test_flush_session_metadata_includes_run_id_for_routing(
    monkeypatch, tmp_path: Path,
):
    """The streaming sink routes ``put_trace`` via
    ``payload['session']['run_id']`` to its
    ``PUT /runs/{id}/trace`` endpoint. Confirm the run_id makes it
    into the payload."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="route_check_xyz", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "x"},
    )
    fake_stream = MagicMock()
    a._session._stream = fake_stream
    a._flush_session_metadata_to_stream()
    payload = fake_stream.put_trace.call_args.args[0]
    assert payload["session"]["run_id"] == "route_check_xyz"


def test_flush_session_metadata_applies_redaction_policy(
    monkeypatch, tmp_path: Path,
):
    """The session-only flush must run the SDK's redaction policy
    (same as the close-time flush) so passwords / tokens in the
    session payload are masked before they hit the network."""
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    a = AugurAdapter(
        run_id="redact_check", tenant_id="t", session_name="s",
        out_dir=tmp_path,
        extra_tags={"model": "x"},
    )
    fake_stream = MagicMock()
    a._session._stream = fake_stream
    # Spy on the redaction policy's apply method.
    real_apply = a._session.redaction_policy.apply
    redaction_spy = MagicMock(side_effect=real_apply)
    a._session.redaction_policy.apply = redaction_spy
    a._flush_session_metadata_to_stream()
    # apply must have been called (the SDK's policy walks the payload
    # recursively, so we get N calls — at least one on the root dict).
    assert redaction_spy.called
    fake_stream.put_trace.assert_called_once()
