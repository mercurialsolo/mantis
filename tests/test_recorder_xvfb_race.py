"""Recorder vs Xvfb startup race — recurring integrator error.

Pins the fix for the recurring ``ffmpeg-startup-failed:[x11grab ...]
Cannot open display :99, error 1.`` error users were seeing on
``/v1/predict`` runs with ``record_video=true``.

Root cause: the baseten runtime constructs an ``XdotoolGymEnv`` (which
lazy-starts Xvfb inside ``reset()``) and then spawns the screen recorder
*before* ``reset()`` runs. ffmpeg's ``-f x11grab -i :99`` attaches to a
display that doesn't exist yet → cryptic failure surfaces in
``result.json``'s ``video.error`` field.

Two layers of defense pinned here:

1. ``XdotoolGymEnv.ensure_display_ready()`` — idempotent hook the
   runtime can call to bring Xvfb up before ffmpeg fires. Skips the
   ``_start_xvfb`` Popen when the display is already alive.
2. Recorder probe in ``ScreenRecorder.start()`` — exits with a
   precise ``x-display-not-ready:<display>`` error if the display
   still isn't reachable, so third-party callers / tests get an
   actionable envelope (covered in ``test_recorder.py``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent.gym.xdotool_env import XdotoolGymEnv


# ── env.ensure_display_ready — fast-path when Xvfb is already up ────────


def test_ensure_display_ready_fast_path_when_alive(monkeypatch):
    """When xdpyinfo reports the display is alive, ``ensure_display_ready``
    must NOT spawn another Xvfb (would Popen-collide on :99 and waste a
    process). It should just wire ``self._env["DISPLAY"]`` and return."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._display = None
    env._env = {}
    env._xvfb_proc = None
    env._viewport = (1280, 720)

    # The probe says display is alive — no need to start Xvfb.
    monkeypatch.setattr(
        XdotoolGymEnv, "_xdpyinfo_alive", staticmethod(lambda d: True),
    )
    start_xvfb_calls: list[str] = []
    monkeypatch.setattr(
        XdotoolGymEnv,
        "_start_xvfb",
        lambda self: start_xvfb_calls.append("called") or ":99",
    )

    display = env.ensure_display_ready(timeout=0.5)

    assert display == ":99"
    assert start_xvfb_calls == []  # spawn skipped — idempotency holds
    assert env._env["DISPLAY"] == ":99"


def test_ensure_display_ready_spawns_xvfb_when_dead(monkeypatch):
    """When the display is dead, ``_start_xvfb`` is invoked exactly once
    and the result is propagated onto ``self._env["DISPLAY"]``."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._display = None
    env._env = {"OTHER": "val"}
    env._xvfb_proc = None
    env._viewport = (1280, 720)

    # First probe: not alive. Subsequent probes (during the bounded
    # wait after spawn): alive — simulates Xvfb coming up.
    probe_calls = {"n": 0}

    def _probe(_d):
        probe_calls["n"] += 1
        return probe_calls["n"] > 1

    monkeypatch.setattr(
        XdotoolGymEnv, "_xdpyinfo_alive", staticmethod(_probe),
    )
    spawned = []
    monkeypatch.setattr(
        XdotoolGymEnv,
        "_start_xvfb",
        lambda self: spawned.append("called") or ":99",
    )

    display = env.ensure_display_ready(timeout=2.0)
    assert display == ":99"
    assert spawned == ["called"]
    # Preserves prior env, adds DISPLAY.
    assert env._env["OTHER"] == "val"
    assert env._env["DISPLAY"] == ":99"


def test_ensure_display_ready_returns_even_on_persistent_failure(monkeypatch):
    """When Xvfb never comes up, ``ensure_display_ready`` must NOT
    raise — the recorder treats that as a benign failure (run continues
    without recording). Raising would crash the run."""
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._display = None
    env._env = {}
    env._xvfb_proc = None
    env._viewport = (1280, 720)

    monkeypatch.setattr(
        XdotoolGymEnv, "_xdpyinfo_alive", staticmethod(lambda d: False),
    )
    monkeypatch.setattr(
        XdotoolGymEnv,
        "_start_xvfb",
        lambda self: ":99",
    )

    # Tiny timeout so the test finishes fast — the contract is "don't
    # raise", not "wait forever".
    display = env.ensure_display_ready(timeout=0.3)
    assert display == ":99"  # we still return a value the caller can use


# ── runtime._maybe_record passes env in and calls ensure_display_ready ──


def test_maybe_record_calls_ensure_display_ready_before_spawning_ffmpeg(tmp_path):
    """The runtime hook ``_maybe_record`` must invoke
    ``env.ensure_display_ready()`` before spawning the recorder so
    Xvfb is alive by the time ffmpeg attaches. Pins the wiring
    introduced for the integrator-reported ffmpeg-startup race.
    """
    # Use a real BasetenRuntime so we exercise the actual code path.
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime

    # Build a minimal env stub that records the order of calls.
    order: list[str] = []
    env_stub = MagicMock()

    def _ensure_display() -> str:
        order.append("ensure_display_ready")
        return ":99"

    env_stub.ensure_display_ready.side_effect = _ensure_display

    # Stub ScreenRecorder so start() succeeds without ffmpeg.
    fake_rec = MagicMock()
    fake_rec._started_at = 1234567890.0
    fake_rec._fmt = "mp4"
    fake_rec.result = None

    def _rec_start() -> bool:
        order.append("recorder.start")
        return True

    fake_rec.start.side_effect = _rec_start

    rt = BasetenCUARuntime.__new__(BasetenCUARuntime)
    # Minimal payload — record_video=True triggers the path under test.
    payload = {"record_video": True, "video_format": "mp4", "video_fps": 5}

    with patch(
        "mantis_agent.recorder.ScreenRecorder",
        return_value=fake_rec,
    ), patch.dict(
        "os.environ",
        {"MANTIS_TENANT_ID": "default", "MANTIS_DATA_DIR": str(tmp_path)},
    ):
        recorder, click_log = rt._maybe_record(
            payload, run_id="test_run_xvfb_race", env=env_stub,
        )

    assert recorder is fake_rec
    assert click_log is not None
    # Critical: ensure_display_ready happens BEFORE recorder.start.
    assert order == ["ensure_display_ready", "recorder.start"], (
        f"expected ensure_display_ready→recorder.start, got {order!r}"
    )


def test_maybe_record_tolerates_env_without_ensure_display_ready(tmp_path):
    """Legacy / test envs that don't implement ``ensure_display_ready``
    must not crash ``_maybe_record`` — the wiring is best-effort. Also
    used by callers that supply ``env=None``."""
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime

    fake_rec = MagicMock()
    fake_rec._started_at = 1234567890.0
    fake_rec._fmt = "mp4"
    fake_rec.start.return_value = True
    fake_rec.result = None

    rt = BasetenCUARuntime.__new__(BasetenCUARuntime)
    payload = {"record_video": True, "video_format": "mp4", "video_fps": 5}

    # env=None — most defensive callsite.
    with patch(
        "mantis_agent.recorder.ScreenRecorder",
        return_value=fake_rec,
    ), patch.dict(
        "os.environ",
        {"MANTIS_TENANT_ID": "default", "MANTIS_DATA_DIR": str(tmp_path)},
    ):
        recorder, click_log = rt._maybe_record(
            payload, run_id="test_run_no_env", env=None,
        )

    assert recorder is fake_rec  # still ran, didn't crash


def test_maybe_record_skipped_when_record_video_false(tmp_path):
    """Sanity: when ``record_video`` isn't set, no display probe / no
    recorder spawn happens. Pin the existing short-circuit so the
    new wiring doesn't accidentally fire on plain runs."""
    from mantis_agent.baseten_server.runtime import BasetenCUARuntime

    env_stub = MagicMock()
    rt = BasetenCUARuntime.__new__(BasetenCUARuntime)

    recorder, click_log = rt._maybe_record(
        {}, run_id="no_video", env=env_stub,
    )
    assert recorder is None
    assert click_log is None
    env_stub.ensure_display_ready.assert_not_called()
