"""#687 — env-gated ``capture_logprobs=True`` on the DebugSession.

augur-sdk 0.5.0+ added ``DebugSession(capture_logprobs=True)`` and
``ModelApiAdapterBase.extract_logprobs_from_response`` for capturing
per-token vendor logprobs at runtime. Without these, future DPO /
PPO / GRPO trainers must recompute logprobs from the trajectory —
expensive and lossy.

This PR wires:

* ``MANTIS_CAPTURE_LOGPROBS=1`` env flag (off by default in
  production — capture roughly doubles vendor response payload size).
* ``AugurAdapter.__init__`` forwards ``capture_logprobs=True`` to
  ``DebugSession`` when the flag is on. Pre-0.5.0 SDKs that reject
  the kwarg get retried without it (best-effort observability).
* ``observability/modelio.record_anthropic_modelio`` extracts the
  vendor's logprobs via the SDK's canonical helper and stamps them
  on ``response.logprobs``. The session's ``capture_logprobs=True``
  flag then governs the empty-vs-absent semantics in the bundle.

Contract pinned here:

* ``should_capture_logprobs()`` reads ``MANTIS_CAPTURE_LOGPROBS`` and
  returns ``True`` for ``1/true/yes/on``; ``False`` otherwise.
* ``AugurAdapter`` forwards ``capture_logprobs=True`` when the flag
  is on; omits the kwarg when off.
* TypeError on the kwarg → retry strips it (production-path
  preservation for pre-0.5.0 pins).
* ``record_anthropic_modelio`` calls the SDK extractor and stamps
  ``response.logprobs`` when the vendor returned them; leaves the
  field absent otherwise (the session-level flag controls the
  empty-list default).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import mantis_agent.observability.augur as augur_mod
from mantis_agent.observability.augur import (
    AugurAdapter,
    should_capture_logprobs,
)
from mantis_agent.observability.modelio import (
    ModelIOContext,
    record_anthropic_modelio,
)


@pytest.fixture
def force_augur_available(monkeypatch):
    monkeypatch.delenv("MANTIS_AUGUR_DISABLED", raising=False)
    monkeypatch.setattr(augur_mod, "_AUGUR_AVAILABLE", True)
    monkeypatch.setattr(
        augur_mod, "CaptureMode", lambda v=None: f"capture_mode:{v}",
    )


# ── should_capture_logprobs env-flag parsing ───────────────────────


@pytest.mark.parametrize(
    "flag,expected",
    [
        ("1", True), ("true", True), ("TRUE", True), ("on", True), ("yes", True),
        ("0", False), ("false", False), ("", False), ("no", False), ("off", False),
    ],
)
def test_should_capture_logprobs_parses_env(monkeypatch, flag, expected):
    monkeypatch.setenv("MANTIS_CAPTURE_LOGPROBS", flag)
    assert should_capture_logprobs() is expected


def test_should_capture_logprobs_default_off(monkeypatch):
    """Production default: env unset → off (capture has cost)."""
    monkeypatch.delenv("MANTIS_CAPTURE_LOGPROBS", raising=False)
    assert should_capture_logprobs() is False


# ── AugurAdapter forwards capture_logprobs to DebugSession ─────────


def test_adapter_forwards_capture_logprobs_when_flag_on(
    force_augur_available, monkeypatch,
):
    """``MANTIS_CAPTURE_LOGPROBS=1`` → ``DebugSession(capture_logprobs=True)``."""
    monkeypatch.setenv("MANTIS_CAPTURE_LOGPROBS", "1")
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    debug_session_cls.assert_called_once()
    kwargs = debug_session_cls.call_args.kwargs
    assert kwargs["capture_logprobs"] is True


def test_adapter_omits_capture_logprobs_when_flag_off(
    force_augur_available, monkeypatch,
):
    """Env unset → kwarg not forwarded (production default)."""
    monkeypatch.delenv("MANTIS_CAPTURE_LOGPROBS", raising=False)
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(return_value=fake_session)
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    assert "capture_logprobs" not in debug_session_cls.call_args.kwargs


def test_adapter_retries_without_capture_logprobs_on_TypeError(
    force_augur_available, monkeypatch,
):
    """Pre-0.5.0 SDK rejects the kwarg → retry strips it; session
    still opens (logprob capture silently degrades but bundle works)."""
    monkeypatch.setenv("MANTIS_CAPTURE_LOGPROBS", "1")
    fake_session = MagicMock()
    fake_session._stream = None
    debug_session_cls = MagicMock(side_effect=[
        TypeError("got unexpected keyword 'capture_logprobs'"),
        fake_session,
    ])
    with patch.object(augur_mod, "DebugSession", debug_session_cls):
        a = AugurAdapter(run_id="r", tenant_id="t", session_name="s")
    assert debug_session_cls.call_count == 2
    assert "capture_logprobs" in debug_session_cls.call_args_list[0].kwargs
    assert "capture_logprobs" not in debug_session_cls.call_args_list[1].kwargs
    assert a._session is fake_session


# ── record_anthropic_modelio stamps response.logprobs ──────────────


def test_record_anthropic_modelio_stamps_logprobs_when_vendor_returns_them():
    """SDK extractor returns logprobs list → mapper stamps it on
    response.logprobs for the augur bundle."""
    fake_augur = MagicMock()
    fake_augur.active = True
    fake_augur.record_modelio = MagicMock()
    ctx = ModelIOContext(augur=fake_augur, layer="model", step_index=2)

    # Anthropic-shape response with logprobs.
    response_json = {
        "content": [
            {"type": "text", "text": "hello",
             "logprobs": [
                 {"token": "hel", "logprob": -0.5},
                 {"token": "lo", "logprob": -1.2},
             ]},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    record_anthropic_modelio(
        request_payload={"model": "claude-3-5-sonnet", "messages": []},
        response_json=response_json,
        duration_ms=100,
        ctx=ctx,
    )
    fake_augur.record_modelio.assert_called_once()
    record = fake_augur.record_modelio.call_args.args[0]
    assert "logprobs" in record["response"]
    lp = record["response"]["logprobs"]
    assert isinstance(lp, list)
    assert lp[0]["token"] == "hel"


def test_record_anthropic_modelio_omits_logprobs_when_vendor_returned_none():
    """Vendor didn't return logprobs (typical Anthropic Messages
    today) → SDK extractor returns None → field absent from the
    record. The session-level capture_logprobs flag still governs
    the empty-list default at session close."""
    fake_augur = MagicMock()
    fake_augur.active = True
    fake_augur.record_modelio = MagicMock()
    ctx = ModelIOContext(augur=fake_augur, layer="model", step_index=0)
    response_json = {
        "content": [{"type": "text", "text": "no logprobs here"}],
        "stop_reason": "end_turn",
    }
    record_anthropic_modelio(
        request_payload={"model": "claude-3-5-sonnet", "messages": []},
        response_json=response_json,
        duration_ms=100,
        ctx=ctx,
    )
    fake_augur.record_modelio.assert_called_once()
    record = fake_augur.record_modelio.call_args.args[0]
    assert "logprobs" not in record["response"]


def test_record_anthropic_modelio_swallows_extractor_exception():
    """SDK extractor raised (corrupt response shape, schema drift) →
    mapper continues without logprobs. Telemetry never breaks the
    run; the rest of the modelio record still lands."""
    fake_augur = MagicMock()
    fake_augur.active = True
    fake_augur.record_modelio = MagicMock()
    ctx = ModelIOContext(augur=fake_augur, layer="model", step_index=0)
    response_json = {"content": [{"type": "text", "text": "x"}]}

    # Patch the SDK extractor at the import site to raise.
    import augur_sdk as sdk_mod
    real_extractor = sdk_mod.ModelApiAdapterBase.extract_logprobs_from_response
    with patch.object(
        sdk_mod.ModelApiAdapterBase, "extract_logprobs_from_response",
        side_effect=RuntimeError("extractor blew up"),
    ):
        record_anthropic_modelio(
            request_payload={"model": "claude-3-5-sonnet", "messages": []},
            response_json=response_json,
            duration_ms=100,
            ctx=ctx,
        )
    # record_modelio still called; record has no logprobs key.
    fake_augur.record_modelio.assert_called_once()
    record = fake_augur.record_modelio.call_args.args[0]
    assert "logprobs" not in record["response"]
    # Restore.
    sdk_mod.ModelApiAdapterBase.extract_logprobs_from_response = real_extractor
