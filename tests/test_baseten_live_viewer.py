"""Coverage for the live-viewer wiring on the detached /v1/predict path (#416).

The viewer surface itself (``viewer_modal._start_background`` /
``modal.forward`` tunnel) is exercised separately. These tests pin the
*opt-in plumbing*:

- ``PredictRequest.live_viewer`` defaults to ``False`` and accepts ``True``.
- ``task_loop.setup_viewer`` returns the new 3-tuple shape
  ``(ctx, event_bus, url)`` — ``None`` for each when disabled.
- ``BasetenCUARuntime._maybe_start_live_viewer`` is a no-op when the
  flag is absent, calls ``ensure_display_ready`` + ``setup_viewer`` +
  ``_write_detached_status({"viewer_url": …})`` when present, and
  swallows any exception so a viewer failure can never break the run.
- ``_stop_live_viewer`` is idempotent and swallows exceptions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from mantis_agent.api_schemas import PredictRequest


# ── Schema ───────────────────────────────────────────────────────────


def test_predict_request_defaults_live_viewer_to_false() -> None:
    req = PredictRequest.model_validate({"plan_text": "x"})
    assert req.live_viewer is False


def test_predict_request_accepts_live_viewer_true() -> None:
    req = PredictRequest.model_validate({"plan_text": "x", "live_viewer": True})
    assert req.live_viewer is True


# ── setup_viewer 3-tuple shape ───────────────────────────────────────


def test_setup_viewer_disabled_returns_three_nones() -> None:
    """Existing CLI callers were destructuring 2 values; #416 expands
    to 3. Locks the disabled-path shape so any caller that still
    destructures only 2 will fail loudly during import."""
    from mantis_agent.task_loop import setup_viewer
    ctx, bus, url = setup_viewer(False)
    assert ctx is None and bus is None and url is None


# ── _maybe_start_live_viewer ─────────────────────────────────────────


def _make_runtime() -> MagicMock:
    """Stand-in for BasetenCUARuntime that owns enough surface for
    _maybe_start_live_viewer / _stop_live_viewer to function in
    isolation. Only the methods the helpers actually call are wired."""
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime
    runtime = MagicMock(spec=BasetenCUARuntime)
    runtime._write_detached_status = MagicMock()
    # Bind the real methods to the mock so they execute against our stubs.
    runtime._maybe_start_live_viewer = BasetenCUARuntime._maybe_start_live_viewer.__get__(
        runtime, BasetenCUARuntime,
    )
    runtime._stop_live_viewer = BasetenCUARuntime._stop_live_viewer.__get__(
        runtime, BasetenCUARuntime,
    )
    return runtime


def test_maybe_start_live_viewer_returns_none_when_flag_off() -> None:
    """No flag → no viewer, no status write. The fast path."""
    runtime = _make_runtime()
    out = runtime._maybe_start_live_viewer({"live_viewer": False}, "run-1", env=MagicMock())
    assert out is None
    runtime._write_detached_status.assert_not_called()


def test_maybe_start_live_viewer_missing_key_is_off() -> None:
    """``payload.get('live_viewer')`` returning falsy → no-op."""
    runtime = _make_runtime()
    out = runtime._maybe_start_live_viewer({}, "run-1", env=MagicMock())
    assert out is None
    runtime._write_detached_status.assert_not_called()


def test_maybe_start_live_viewer_starts_and_writes_url(monkeypatch) -> None:
    """Happy path: flag on → ensure_display_ready called, setup_viewer
    invoked, URL merged into status.json, viewer_ctx returned."""
    runtime = _make_runtime()

    fake_ctx = MagicMock(name="viewer_ctx")
    fake_url = "https://tunnel.modal.run/?token=abc"

    def fake_setup_viewer(enabled: bool):
        assert enabled is True
        return fake_ctx, MagicMock(name="event_bus"), fake_url

    monkeypatch.setattr(
        "mantis_agent.task_loop.setup_viewer", fake_setup_viewer,
    )

    env = MagicMock()
    env.ensure_display_ready = MagicMock(return_value=":99")

    out = runtime._maybe_start_live_viewer(
        {"live_viewer": True}, "run-42", env=env,
    )

    assert out is fake_ctx
    env.ensure_display_ready.assert_called_once()
    # The URL must be merged into status.json — runtime callers
    # depend on this so action=status surfaces the link.
    runtime._write_detached_status.assert_called_once_with(
        "run-42", {"viewer_url": fake_url},
    )


def test_maybe_start_live_viewer_swallows_setup_exception(monkeypatch) -> None:
    """A failure in setup_viewer must never break the run. Return None
    and skip the status write — caller treats as no-viewer."""
    runtime = _make_runtime()

    def boom(_enabled: bool):
        raise RuntimeError("modal.forward unavailable")

    monkeypatch.setattr("mantis_agent.task_loop.setup_viewer", boom)
    out = runtime._maybe_start_live_viewer(
        {"live_viewer": True}, "run-X", env=MagicMock(),
    )
    assert out is None
    runtime._write_detached_status.assert_not_called()


def test_maybe_start_live_viewer_handles_setup_returning_none(monkeypatch) -> None:
    """If setup_viewer returns (None, None, None) (its disabled path),
    treat as no-viewer — don't write a null URL to status."""
    runtime = _make_runtime()
    monkeypatch.setattr(
        "mantis_agent.task_loop.setup_viewer",
        lambda enabled: (None, None, None),
    )
    out = runtime._maybe_start_live_viewer(
        {"live_viewer": True}, "run-Y", env=MagicMock(),
    )
    assert out is None
    runtime._write_detached_status.assert_not_called()


def test_maybe_start_live_viewer_tolerates_env_without_ensure_display() -> None:
    """``env.ensure_display_ready`` is best-effort. Envs that don't
    expose it (Playwright path, tests) must not break the helper."""
    runtime = _make_runtime()
    env_no_display = MagicMock(spec=[])  # no ensure_display_ready

    def fake_setup_viewer(enabled: bool):
        return MagicMock(), MagicMock(), "https://x"

    import mantis_agent.task_loop as tl_module
    original = tl_module.setup_viewer
    tl_module.setup_viewer = fake_setup_viewer  # type: ignore[assignment]
    try:
        out = runtime._maybe_start_live_viewer(
            {"live_viewer": True}, "run-Z", env=env_no_display,
        )
    finally:
        tl_module.setup_viewer = original  # type: ignore[assignment]
    assert out is not None
    runtime._write_detached_status.assert_called_once()


# ── _stop_live_viewer ────────────────────────────────────────────────


def test_stop_live_viewer_no_op_on_none() -> None:
    runtime = _make_runtime()
    runtime._stop_live_viewer(None)  # must not raise


def test_stop_live_viewer_invokes_exit_with_no_exception_info() -> None:
    runtime = _make_runtime()
    ctx = MagicMock()
    runtime._stop_live_viewer(ctx)
    ctx.__exit__.assert_called_once_with(None, None, None)


def test_stop_live_viewer_swallows_exit_exception() -> None:
    """Viewer cleanup failures must not mask whatever happened during
    the run — the FastAPI capture thread occasionally races shutdown
    and raises on __exit__. Log + swallow."""
    runtime = _make_runtime()
    ctx = MagicMock()
    ctx.__exit__.side_effect = RuntimeError("capture thread already gone")
    runtime._stop_live_viewer(ctx)  # must not raise
    ctx.__exit__.assert_called_once()
