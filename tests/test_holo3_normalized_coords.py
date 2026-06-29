"""cua-issues 2026-06-29 (S09) — Holo3 normalized-coordinate parsing.

Holo3 sometimes emits clicks with NORMALIZED fractions in [0,1], e.g.
`click(x=0.324, y=0.338)`. The old `_convert_coords` did
`int(float("0.324"))` → 0, collapsing every such click to the top-left
corner (a degenerate no-op that looped the run out — the audit's S09).
`_coords_to_screen` now scales normalized fractions by the screen size and
keeps resized-space integers on the existing Qwen path.
"""

from __future__ import annotations

from mantis_agent.brain_holo3 import Holo3Brain


SCREEN = (1280, 720)


def test_normalized_fraction_scales_to_pixels():
    # 0.324 * 1280 = 414.7 → 415 ; 0.338 * 720 = 243.4 → 243
    x, y = Holo3Brain._coords_to_screen("0.324", "0.338", SCREEN)
    assert (x, y) == (415, 243)


def test_normalized_does_not_collapse_to_origin():
    # The S09 bug: int(float("0.324")) == 0 → top-left no-op. Must NOT happen.
    x, y = Holo3Brain._coords_to_screen("0.5", "0.5", SCREEN)
    assert (x, y) == (640, 360)
    assert (x, y) != (0, 0)


def test_full_extent_fraction():
    x, y = Holo3Brain._coords_to_screen("1.0", "1.0", SCREEN)
    assert (x, y) == (1280, 720)


def test_integer_coord_stays_resized_space():
    # "640" has no decimal point → treated as a resized-space coord, NOT a
    # full-width fraction. It must go through the Qwen conversion (a small
    # absolute pixel value), not 640*1280.
    x, y = Holo3Brain._coords_to_screen("640", "360", SCREEN)
    assert 0 <= x <= 1280 and 0 <= y <= 720
    assert x != 640 * 1280  # not misread as a fraction


def test_integer_one_is_not_a_fraction():
    # bare integer 1 (no dot) → resized-space, NOT full-width.
    x, y = Holo3Brain._coords_to_screen("1", "1", SCREEN)
    assert x < 1280 and y < 720
