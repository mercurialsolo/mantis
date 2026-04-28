"""Tests for the polished-recording machinery: PIL cards, SRT cues,
ffmpeg compose orchestration."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from PIL import Image

from mantis_agent.presentation import (
    CardConfig,
    StepCaption,
    _format_srt_ts,
    _wrap_for_srt,
    captions_from_step_timings,
    captions_to_srt,
    compose_polished_video,
    outro_card_from_summary,
    render_card,
    title_card_for_run,
    write_card,
)


# ── Card rendering ──────────────────────────────────────────────────────────
def test_render_card_returns_valid_png():
    cfg = CardConfig(title="Mantis CUA", subtitle="hello", body_lines=["line 1"])
    data = render_card(640, 360, cfg)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    img = Image.open(io.BytesIO(data))
    assert img.size == (640, 360)


def test_render_card_handles_empty_body():
    cfg = CardConfig(title="Title only")
    data = render_card(800, 600, cfg)
    img = Image.open(io.BytesIO(data))
    assert img.size == (800, 600)


def test_render_card_long_title_does_not_crash():
    cfg = CardConfig(title="A" * 200, subtitle="B" * 200)
    data = render_card(640, 360, cfg)
    assert len(data) > 0


def test_write_card_writes_to_disk(tmp_path: Path):
    out = tmp_path / "card.png"
    cfg = CardConfig(title="x")
    returned = write_card(out, 320, 180, cfg)
    assert returned == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_title_card_helper_includes_run_id():
    cfg = title_card_for_run(plan_label="boattrader", tenant_id="vc_prod", run_id="20260428_xyz")
    assert cfg.title == "Mantis CUA"
    assert "vc_prod" in " ".join(cfg.body_lines)
    assert "20260428_xyz" in " ".join(cfg.body_lines)


def test_outro_card_from_summary_renders_lead_metrics():
    cfg = outro_card_from_summary({
        "viable": 3,
        "leads_with_phone": 1,
        "steps_executed": 17,
        "total_time_s": 569,
    }, plan_label="bt", cost_total=0.42)
    body = " ".join(cfg.body_lines)
    assert "viable leads: 3" in body
    assert "with phone:   1" in body
    assert "steps:        17" in body
    assert "duration:     569s" in body
    assert "cost:         $0.42" in body


def test_outro_card_handles_missing_fields():
    cfg = outro_card_from_summary({})
    # Should still render without crashing
    data = render_card(640, 360, cfg)
    assert len(data) > 0


# ── SRT generation ──────────────────────────────────────────────────────────
def test_format_srt_ts_basic():
    assert _format_srt_ts(0) == "00:00:00,000"
    assert _format_srt_ts(1.5) == "00:00:01,500"
    assert _format_srt_ts(65.123) == "00:01:05,123"
    assert _format_srt_ts(3661.001) == "01:01:01,001"


def test_format_srt_ts_clamps_negative():
    assert _format_srt_ts(-5) == "00:00:00,000"


def test_wrap_for_srt_short_passes_through():
    assert _wrap_for_srt("Hello world", max_chars=44) == "Hello world"


def test_wrap_for_srt_two_lines():
    text = "Click only an organic private-seller listing title; skip sponsored cards"
    out = _wrap_for_srt(text, max_chars=30)
    assert "\n" in out
    lines = out.split("\n")
    assert len(lines) == 2
    assert all(len(line) <= 31 for line in lines[:-1])


def test_wrap_for_srt_truncates_with_ellipsis():
    text = " ".join(["w"] * 100)
    out = _wrap_for_srt(text, max_chars=20)
    assert out.endswith("…")
    assert out.count("\n") == 1


def test_captions_from_step_timings_basic():
    timings = [
        (0.0, "Navigate to site", "completed"),
        (5.0, "Click listing", "completed"),
        (12.0, "Extract data", "failed"),
    ]
    caps = captions_from_step_timings(timings)
    assert len(caps) == 3
    assert caps[0].start_t == 0.0
    assert caps[0].end_t == 5.0
    assert caps[0].text.startswith("[OK]")
    assert caps[2].text.startswith("[FAIL]")
    # Last cue stretches 3s past its start
    assert caps[2].end_t == pytest.approx(15.0)


def test_captions_from_step_timings_offset():
    caps = captions_from_step_timings(
        [(0.0, "step 1", "completed")], title_offset=3.0,
    )
    assert caps[0].start_t == 3.0


def test_captions_from_step_timings_empty():
    assert captions_from_step_timings([]) == []


def test_captions_to_srt_format():
    cues = [
        StepCaption(0.0, 5.0, "first"),
        StepCaption(5.0, 10.0, "second"),
    ]
    srt = captions_to_srt(cues)
    # SRT: numbered cues, timestamps in HH:MM:SS,mmm format
    assert "1\n00:00:00,000 --> 00:00:05,000\nfirst" in srt
    assert "2\n00:00:05,000 --> 00:00:10,000\nsecond" in srt


# ── compose_polished_video (mocked ffmpeg) ──────────────────────────────────
def test_compose_skips_when_ffmpeg_missing(tmp_path: Path):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"x" * 1024)
    with patch("mantis_agent.presentation._ffmpeg_available", return_value=False):
        ok = compose_polished_video(
            raw_video=raw,
            title_card=None,
            outro_card=None,
            subtitles_srt=None,
            output=tmp_path / "polished.mp4",
        )
    assert not ok


def test_compose_skips_when_raw_missing(tmp_path: Path):
    with patch("mantis_agent.presentation._ffmpeg_available", return_value=True):
        ok = compose_polished_video(
            raw_video=tmp_path / "no.mp4",
            title_card=None,
            outro_card=None,
            subtitles_srt=None,
            output=tmp_path / "out.mp4",
        )
    assert not ok


def _fake_ffmpeg_run(*args, **kwargs):
    """subprocess.run mock that emulates a successful ffmpeg compose:
    creates the output file at the path passed as the last argv element."""
    cmd = args[0] if args else kwargs.get("args", [])
    out = Path(cmd[-1])
    out.write_bytes(b"FAKEMP4" * 64)
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = b""
    proc.stderr = b""
    return proc


def test_compose_writes_output_when_ffmpeg_succeeds(tmp_path: Path):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"y" * 4096)
    title = tmp_path / "title.png"
    title.write_bytes(b"PNG bytes")
    outro = tmp_path / "outro.png"
    outro.write_bytes(b"PNG bytes")
    out = tmp_path / "polished.mp4"

    with patch("mantis_agent.presentation._ffmpeg_available", return_value=True), \
         patch("mantis_agent.presentation.subprocess.run", side_effect=_fake_ffmpeg_run):
        ok = compose_polished_video(
            raw_video=raw,
            title_card=title,
            outro_card=outro,
            subtitles_srt=None,
            output=out,
        )
    assert ok
    assert out.exists()
    assert out.stat().st_size > 0


def test_compose_passes_subtitles_to_filter_when_present(tmp_path: Path):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"x" * 1024)
    srt = tmp_path / "captions.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    out = tmp_path / "polished.mp4"

    captured: dict = {}

    def _capture(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"video")
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = b""
        return proc

    with patch("mantis_agent.presentation._ffmpeg_available", return_value=True), \
         patch("mantis_agent.presentation.subprocess.run", side_effect=_capture):
        compose_polished_video(
            raw_video=raw,
            title_card=None,
            outro_card=None,
            subtitles_srt=srt,
            output=out,
        )
    # filter_complex should reference the subtitles= filter
    fc_idx = captured["cmd"].index("-filter_complex")
    fc = captured["cmd"][fc_idx + 1]
    assert "subtitles=" in fc


def test_compose_handles_only_raw_no_title_no_outro(tmp_path: Path):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"x" * 1024)
    out = tmp_path / "polished.mp4"
    with patch("mantis_agent.presentation._ffmpeg_available", return_value=True), \
         patch("mantis_agent.presentation.subprocess.run", side_effect=_fake_ffmpeg_run):
        ok = compose_polished_video(
            raw_video=raw,
            title_card=None,
            outro_card=None,
            subtitles_srt=None,
            output=out,
        )
    assert ok
    assert out.exists()


# ── Click ripple overlay ────────────────────────────────────────────────────
def test_click_event_log_records_with_anchor():
    import time
    from mantis_agent.presentation import ClickEventLog
    log = ClickEventLog(anchor_time=time.time())
    log.record(100, 200, button="left")
    time.sleep(0.05)
    log.record(300, 400, button="right")
    events = log.events
    assert len(events) == 2
    assert events[0].x == 100 and events[0].y == 200 and events[0].button == "left"
    assert events[1].button == "right"
    # Timestamps are monotonic and elapsed-from-anchor (i.e., near-zero).
    assert 0.0 <= events[0].t_seconds < 1.0
    assert events[1].t_seconds >= events[0].t_seconds


def test_click_recording_env_intercepts_clicks():
    import time
    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.base import GymObservation, GymResult
    from mantis_agent.presentation import ClickEventLog, ClickRecordingEnv

    class FakeEnv:
        screen_size = (1280, 720)
        def reset(self, task, **kw): return GymObservation(screenshot=None)  # type: ignore
        def step(self, action):
            return GymResult(GymObservation(screenshot=None), 0.0, False, {})  # type: ignore
        def close(self): pass

    log = ClickEventLog(anchor_time=time.time())
    env = ClickRecordingEnv(FakeEnv(), log)
    env.step(Action(action_type=ActionType.CLICK, params={"x": 50, "y": 60}))
    env.step(Action(action_type=ActionType.TYPE, params={"text": "hi"}))
    env.step(Action(action_type=ActionType.CLICK, params={"x": 200, "y": 300, "button": "right"}))
    # Multi-event log records both CLICKs and the TYPE
    assert len(log.clicks) == 2
    assert len(log.types) == 1
    assert log.clicks[0].x == 50
    assert log.clicks[1].button == "right"
    assert log.types[0].text == "hi"


def test_click_recording_env_handles_bad_params():
    import time
    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.base import GymObservation, GymResult
    from mantis_agent.presentation import ClickEventLog, ClickRecordingEnv

    class FakeEnv:
        screen_size = (1280, 720)
        def reset(self, task, **kw): return GymObservation(screenshot=None)  # type: ignore
        def step(self, action):
            return GymResult(GymObservation(screenshot=None), 0.0, False, {})  # type: ignore
        def close(self): pass

    log = ClickEventLog(anchor_time=time.time())
    env = ClickRecordingEnv(FakeEnv(), log)
    # Click with non-int x/y — should not crash, should still pass through.
    env.step(Action(action_type=ActionType.CLICK, params={"x": "bad", "y": 60}))
    assert len(log) == 0  # bad coords skipped


def test_render_ripple_overlay_returns_none_when_no_clicks(tmp_path: Path):
    from mantis_agent.presentation import render_ripple_overlay_pngs
    out = render_ripple_overlay_pngs(
        tmp_path / "ripples", duration_seconds=2.0, fps=10,
        width=1280, height=720, clicks=[],
    )
    assert out is None


def test_render_ripple_overlay_writes_png_sequence(tmp_path: Path):
    from mantis_agent.presentation import (
        ClickEvent, render_ripple_overlay_pngs,
    )
    clicks = [
        ClickEvent(t_seconds=0.0, x=100, y=200),
        ClickEvent(t_seconds=1.0, x=500, y=400),
    ]
    out = render_ripple_overlay_pngs(
        tmp_path / "ripples", duration_seconds=2.0, fps=10,
        width=1280, height=720, clicks=clicks,
    )
    assert out is not None
    frames = sorted(out.glob("frame_*.png"))
    assert len(frames) == 20  # 2s * 10fps
    # Active ripple windows: [0, 0.6] and [1.0, 1.6]. Frames 0-5 and 10-15 active.
    sizes = [f.stat().st_size for f in frames]
    # Active frames are bigger than blank ones.
    blank_size = sizes[8]  # frame at t=0.8 — between ripples
    assert sizes[0] > blank_size  # first ripple visible
    assert sizes[10] > blank_size  # second ripple visible
    assert sizes[8] == blank_size  # no active ripple at t=0.8


def test_action_event_log_records_each_type():
    """ActionEventLog accepts every visually-relevant action and routes
    to the right list."""
    import time
    from mantis_agent.presentation import ActionEventLog

    log = ActionEventLog(anchor_time=time.time())
    log.record_click(10, 20, button="left")
    log.record_click(30, 40, button="right", double=True)
    log.record_key("Ctrl+S")
    log.record_key("Enter")
    log.record_type("hello world")
    log.record_scroll("down", 5)
    log.record_drag(0, 0, 100, 100)

    assert len(log.clicks) == 2
    assert log.clicks[1].double is True
    assert len(log.keys) == 2 and log.keys[0].keys == "Ctrl+S"
    assert len(log.types) == 1 and log.types[0].text == "hello world"
    assert len(log.scrolls) == 1 and log.scrolls[0].direction == "down"
    assert len(log.drags) == 1 and log.drags[0].x2 == 100
    assert log.total == 7


def test_action_recording_env_routes_each_action_type():
    import time
    from mantis_agent.actions import Action, ActionType
    from mantis_agent.gym.base import GymObservation, GymResult
    from mantis_agent.presentation import ActionEventLog, ActionRecordingEnv

    class FakeEnv:
        screen_size = (1280, 720)
        def reset(self, task, **kw): return GymObservation(screenshot=None)  # type: ignore
        def step(self, action):
            return GymResult(GymObservation(screenshot=None), 0.0, False, {})  # type: ignore
        def close(self): pass

    log = ActionEventLog(anchor_time=time.time())
    env = ActionRecordingEnv(FakeEnv(), log)
    env.step(Action(action_type=ActionType.CLICK, params={"x": 1, "y": 2}))
    env.step(Action(action_type=ActionType.DOUBLE_CLICK, params={"x": 3, "y": 4}))
    env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Tab"}))
    env.step(Action(action_type=ActionType.TYPE, params={"text": "foo"}))
    env.step(Action(action_type=ActionType.SCROLL, params={"direction": "up", "amount": 3}))
    env.step(Action(action_type=ActionType.DRAG, params={"x1": 10, "y1": 20, "x2": 30, "y2": 40}))
    env.step(Action(action_type=ActionType.WAIT, params={}))  # ignored

    assert len(log.clicks) == 2
    assert log.clicks[1].double is True
    assert log.keys[0].keys == "Tab"
    assert log.types[0].text == "foo"
    assert log.scrolls[0].direction == "up"
    assert log.drags[0].x2 == 30


def test_render_action_overlay_returns_none_when_all_empty(tmp_path: Path):
    from mantis_agent.presentation import render_action_overlay_pngs
    out = render_action_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=2.0, fps=10, width=1280, height=720,
        clicks=[], keys=[], types=[], scrolls=[], drags=[],
    )
    assert out is None


def test_render_action_overlay_with_each_event_type(tmp_path: Path):
    from mantis_agent.presentation import (
        ClickEvent, DragEvent, KeyPressEvent, ScrollEvent, TypeEvent,
        render_action_overlay_pngs,
    )
    out = render_action_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=3.0, fps=10, width=1280, height=720,
        clicks=[ClickEvent(0.0, 100, 200)],
        keys=[KeyPressEvent(0.5, "Ctrl+S")],
        types=[TypeEvent(1.0, "hello")],
        scrolls=[ScrollEvent(2.0, "down", 5)],
        drags=[DragEvent(0.2, 0, 0, 200, 200)],
    )
    assert out is not None
    frames = sorted(out.glob("frame_*.png"))
    assert len(frames) == 30
    sizes = [f.stat().st_size for f in frames]
    # At least one frame should be substantially bigger than a blank frame.
    blank_size = min(sizes)
    assert max(sizes) > blank_size * 2


def test_render_action_overlay_drag_only(tmp_path: Path):
    from mantis_agent.presentation import (
        DragEvent, render_action_overlay_pngs,
    )
    out = render_action_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=1.5, fps=20, width=640, height=480,
        drags=[DragEvent(0.2, 50, 50, 600, 400)],
    )
    assert out is not None
    frames = sorted(out.glob("frame_*.png"))
    assert len(frames) == 30
    sizes = [f.stat().st_size for f in frames]
    nonblank = sum(1 for s in sizes if s > 200)
    assert nonblank > 0


def test_render_action_overlay_keys_only(tmp_path: Path):
    """Keyboard chord badges should render even with no clicks/types."""
    from mantis_agent.presentation import KeyPressEvent, render_action_overlay_pngs
    out = render_action_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=2.0, fps=10, width=1280, height=720,
        keys=[KeyPressEvent(0.5, "Ctrl+S"), KeyPressEvent(1.0, "Enter")],
    )
    assert out is not None


def test_render_action_overlay_scroll_arrows(tmp_path: Path):
    from mantis_agent.presentation import ScrollEvent, render_action_overlay_pngs
    out = render_action_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=4.0, fps=10, width=1280, height=720,
        scrolls=[
            ScrollEvent(0.0, "down"),
            ScrollEvent(1.0, "up"),
            ScrollEvent(2.0, "left"),
            ScrollEvent(3.0, "right"),
        ],
    )
    assert out is not None


def test_render_ripple_overlay_pngs_backwards_compat(tmp_path: Path):
    """Old name still callable for any straggling callers."""
    from mantis_agent.presentation import (
        ClickEvent, render_ripple_overlay_pngs,
    )
    out = render_ripple_overlay_pngs(
        tmp_path / "ovl",
        duration_seconds=1.0, fps=10, width=640, height=360,
        clicks=[ClickEvent(0.0, 100, 100)],
    )
    assert out is not None


def test_compose_polished_video_threads_ripples_dir(tmp_path: Path):
    """When ripples_dir is provided, ffmpeg gets a -framerate / image2 input."""
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw" * 256)
    ripples = tmp_path / "ripples"
    ripples.mkdir()
    # Provide a frame_000000.png so the renderer recognises the dir.
    (ripples / "frame_000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "polished.mp4"

    captured: dict = {}
    def _capture(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"v")
        proc = MagicMock()
        proc.returncode = 0
        proc.stderr = b""
        return proc
    with patch("mantis_agent.presentation._ffmpeg_available", return_value=True), \
         patch("mantis_agent.presentation.subprocess.run", side_effect=_capture):
        ok = compose_polished_video(
            raw_video=raw,
            title_card=None,
            outro_card=None,
            subtitles_srt=None,
            ripples_dir=ripples,
            output=out,
        )
    assert ok
    cmd = captured["cmd"]
    # The PNG sequence input should appear in the input list.
    assert any("frame_%06d.png" in str(arg) for arg in cmd)
    # filter_complex should overlay the ripples onto the run footage.
    fc_idx = cmd.index("-filter_complex")
    fc = cmd[fc_idx + 1]
    assert "[v_run]" in fc
    assert "[v_ripples]" in fc
    assert "overlay=0:0" in fc
