"""Content-plateau auto-settle — the generic cold-SPA-mount fix.

Augur run li_predict_fresh showed both LinkedIn login fills recovering from
form_target_not_found: grounding ran on a partially-rendered page (header
painted, form not yet), which the legacy ≥99%-white blank check waved
through. ``wait_for_content_stable`` instead waits until the visible content
stops GROWING — automatic, self-tuning per site, screenshot-only.

These tests pin the content metric + the plateau/early-return/cap behaviour
with injected clocks (no real sleeping).
"""

from __future__ import annotations

from PIL import Image

from mantis_agent.gym import adaptive_settle


# ── content metric ──────────────────────────────────────────────────────


def _blank() -> Image.Image:
    return Image.new("RGB", (96, 96), (255, 255, 255))


def _bars(n: int) -> Image.Image:
    """An image with ``n`` black horizontal bars — edge-density grows with
    ``n``. Bars are 2px tall and widely spaced so they survive the 2×
    NEAREST downsample inside ``_content_score`` (scattered single pixels or
    a period-2 checker would alias away to a solid colour)."""
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    if n <= 0:
        return img
    gap = max(3, 96 // (n + 1))
    for i in range(n):
        y0 = (i + 1) * gap
        for yy in (y0, y0 + 1):
            if yy < 96:
                for x in range(96):
                    img.putpixel((x, yy), (0, 0, 0))
    return img


def test_content_score_orders_blank_partial_full():
    s_blank = adaptive_settle._content_score(_blank())
    s_partial = adaptive_settle._content_score(_bars(4))
    s_full = adaptive_settle._content_score(_bars(16))
    assert s_blank == 0
    assert s_blank < s_partial < s_full


def test_content_score_none_is_zero():
    assert adaptive_settle._content_score(None) == 0


# ── wait_for_content_stable ─────────────────────────────────────────────


def _clock():
    """Return (sleep_fn, time_fn) where sleeping advances a fake monotonic clock."""
    now = {"t": 0.0}
    return (
        lambda s: now.__setitem__("t", now["t"] + s),
        lambda: now["t"],
    )


def _seq_capture(frames):
    """Capture that yields each frame once then repeats the last."""
    state = {"i": 0}
    def cap():
        i = min(state["i"], len(frames) - 1)
        state["i"] += 1
        return frames[i]
    return cap


def test_returns_early_when_content_not_growing():
    """An already-rendered page (content flat from the first read) settles
    after ~require_stable polls, not the full budget."""
    sleep_fn, time_fn = _clock()
    full = _bars(16)
    frame, waited = adaptive_settle.wait_for_content_stable(
        _seq_capture([full, full, full, full]),
        max_seconds=10.0, poll_interval=0.25, require_stable=2,
        sleep_fn=sleep_fn, time_fn=time_fn,
    )
    assert frame is not None
    assert waited <= 0.75  # ~2 polls, well under the 10s cap


def test_waits_through_cold_mount_then_settles():
    """blank → blank → partial → full → full: keeps waiting while content
    grows, returns once it plateaus on the full frame."""
    sleep_fn, time_fn = _clock()
    frames = [_blank(), _blank(), _bars(4), _bars(12), _bars(12), _bars(12)]
    frame, waited = adaptive_settle.wait_for_content_stable(
        _seq_capture(frames),
        max_seconds=10.0, poll_interval=0.25, require_stable=2,
        sleep_fn=sleep_fn, time_fn=time_fn,
    )
    # Settled on the high-content frame, and it took longer than the
    # already-rendered case (had to wait out the growth).
    assert adaptive_settle._content_score(frame) > 0
    assert waited >= 0.75


def test_respects_cap_when_content_keeps_growing():
    """Pathological perpetually-growing page → bounded by max_seconds."""
    sleep_fn, time_fn = _clock()
    frames = [_bars(i) for i in range(2, 16)]  # strictly denser each read
    _frame, waited = adaptive_settle.wait_for_content_stable(
        _seq_capture(frames),
        max_seconds=2.0, poll_interval=0.25, require_stable=2,
        sleep_fn=sleep_fn, time_fn=time_fn,
    )
    assert waited <= 2.0 + 0.25  # never blows past the cap by more than a poll


def test_zero_budget_returns_immediately():
    sleep_fn, time_fn = _clock()
    frame, waited = adaptive_settle.wait_for_content_stable(
        _seq_capture([_blank()]),
        max_seconds=0.0, sleep_fn=sleep_fn, time_fn=time_fn,
    )
    assert waited == 0.0 and frame is not None
