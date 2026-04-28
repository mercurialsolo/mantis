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
    assert all(len(l) <= 31 for l in lines[:-1])


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
        proc = MagicMock(); proc.returncode = 0; proc.stderr = b""
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
