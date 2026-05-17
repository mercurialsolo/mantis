"""Tests for region cropping (#435 item 1).

The new ``mantis_agent.form_targeting.region`` module crops screenshots
to a hint-specified region before grounding and re-projects executor-
emitted coordinates back to full-screen space. These tests pin the
crop arithmetic, the named-region table, the no-op fallback for
unknown regions, and the end-to-end ``find_form_target(..., region=...)``
contract on the ``ClaudeFormTargetProvider``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image

from mantis_agent._anthropic.client import AnthropicToolUseClient
from mantis_agent.form_targeting.claude import ClaudeFormTargetProvider
from mantis_agent.form_targeting.region import crop_to_region, reproject_coords


def _img(w: int = 1280, h: int = 800) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


# ── crop_to_region — named regions ──────────────────────────────────


def test_crop_to_region_bottom_yields_lower_third() -> None:
    cropped, offset = crop_to_region(_img(1280, 900), "bottom")
    # 900 * 2/3 = 600 → bottom rect (0, 600, 1280, 900)
    assert cropped.size == (1280, 300)
    assert offset == (0, 600)


def test_crop_to_region_top_half_yields_upper_half() -> None:
    cropped, offset = crop_to_region(_img(1280, 800), "top-half")
    assert cropped.size == (1280, 400)
    assert offset == (0, 0)


def test_crop_to_region_sidebar_list_mid_narrows_to_middle_band() -> None:
    # 14% wide × 25% tall, centred on the middle of the left rail.
    # On a 1280x800 screenshot:
    #   left   = 0
    #   top    = 800 * 0.35 = 280
    #   right  = 1280 * 0.14 = 179
    #   bottom = 800 * 0.60 = 480
    # → cropped size (179, 200), offset (0, 280)
    cropped, offset = crop_to_region(_img(1280, 800), "sidebar-list-mid")
    assert cropped.size == (179, 200)
    assert offset == (0, 280)


def test_crop_to_region_sidebar_list_top_is_above_mid() -> None:
    _, off_top = crop_to_region(_img(1280, 800), "sidebar-list-top")
    _, off_mid = crop_to_region(_img(1280, 800), "sidebar-list-mid")
    _, off_bot = crop_to_region(_img(1280, 800), "sidebar-list-bottom")
    # Strict y-ordering — the three sidebar bands tile the rail.
    assert off_top[1] < off_mid[1] < off_bot[1]
    # All start at the screen's left edge.
    assert off_top[0] == off_mid[0] == off_bot[0] == 0


def test_crop_to_region_sidebar_list_is_narrower_than_left() -> None:
    """The whole point of sidebar-list-* is to be narrower than the
    33%-wide ``"left"`` region — that's what eliminates the
    multi-section ambiguity (#447 follow-up).
    """
    cropped_sidebar, _ = crop_to_region(_img(1280, 800), "sidebar-list-mid")
    cropped_left, _ = crop_to_region(_img(1280, 800), "left")
    assert cropped_sidebar.width < cropped_left.width


def test_crop_to_region_unknown_named_region_returns_full_screenshot() -> None:
    """Defensive: a typo in a plan hint must not crash. The caller's
    behaviour should fall back to the unchanged-screenshot path.
    """
    full = _img(1280, 800)
    cropped, offset = crop_to_region(full, "BACKGROUND-MIDDLE-XYZ")
    assert cropped.size == full.size
    assert offset == (0, 0)


def test_crop_to_region_case_insensitive() -> None:
    """Operators shouldn't have to remember casing."""
    cropped_a, off_a = crop_to_region(_img(1000, 1000), "Bottom")
    cropped_b, off_b = crop_to_region(_img(1000, 1000), "bottom")
    assert cropped_a.size == cropped_b.size
    assert off_a == off_b


# ── crop_to_region — explicit rect ──────────────────────────────────


def test_crop_to_region_explicit_rect() -> None:
    cropped, offset = crop_to_region(
        _img(1280, 800), {"x": 100, "y": 200, "w": 400, "h": 150},
    )
    assert cropped.size == (400, 150)
    assert offset == (100, 200)


def test_crop_to_region_explicit_rect_clamps_to_bounds() -> None:
    """Rect extending past screen edges → clamped to viewport."""
    cropped, offset = crop_to_region(
        _img(1280, 800), {"x": 1200, "y": 700, "w": 500, "h": 500},
    )
    # Width clamped to 80 (1280 - 1200), height clamped to 100 (800 - 700)
    assert cropped.size == (80, 100)
    assert offset == (1200, 700)


def test_crop_to_region_zero_or_negative_dimensions_no_ops() -> None:
    full = _img(1000, 1000)
    cropped, offset = crop_to_region(
        full, {"x": 0, "y": 0, "w": 0, "h": 100},
    )
    assert cropped.size == full.size
    assert offset == (0, 0)


def test_crop_to_region_malformed_dict_no_ops() -> None:
    full = _img(1000, 1000)
    cropped, offset = crop_to_region(full, {"x": "oops", "y": 0})
    assert cropped.size == full.size
    assert offset == (0, 0)


def test_crop_to_region_none_or_empty_no_ops() -> None:
    for hint in [None, "", 0, {}]:
        full = _img(1000, 1000)
        cropped, offset = crop_to_region(full, hint)
        assert cropped.size == full.size
        assert offset == (0, 0)


# ── reproject_coords ────────────────────────────────────────────────


def test_reproject_coords_adds_offset() -> None:
    out = reproject_coords({"x": 50, "y": 30, "label": "foo"}, (100, 200))
    assert out == {"x": 150, "y": 230, "label": "foo"}


def test_reproject_coords_zero_offset_is_no_op() -> None:
    coords = {"x": 50, "y": 30}
    out = reproject_coords(coords, (0, 0))
    assert out is coords  # same object — no copy on no-op


def test_reproject_coords_none_passthrough() -> None:
    assert reproject_coords(None, (10, 10)) is None


# ── ClaudeFormTargetProvider — region kwarg end-to-end ───────────


def test_claude_provider_passes_region_through_and_reprojects() -> None:
    """End-to-end: the provider receives ``region="bottom"``, the
    Anthropic call sees a cropped image, and the returned ``(x, y)``
    is shifted by the crop offset before being handed back to the
    caller.
    """
    captured: dict = {}

    def _capture(*_args, **kwargs):
        captured["json"] = kwargs.get("json")
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "content": [{
                "type": "tool_use",
                "name": "report_form_target",
                "input": {
                    "x": 100, "y": 50, "action": "click",
                    "value": "", "label": "Update Lead",
                },
            }],
        }
        return resp

    client = AnthropicToolUseClient(api_key="k", model="m", log_prefix="t")
    provider = ClaudeFormTargetProvider(client)
    with patch("requests.post", side_effect=_capture):
        result = provider.find_form_target(
            _img(1280, 900), "Click Update Lead",
            target_label="Update Lead",
            region="bottom",
        )

    assert result is not None
    # bottom region of 1280x900 = (0, 600, 1280, 900); the cropped
    # image is 1280x300 with offset (0, 600). Provider returned
    # (x=100, y=50) in cropped space → reprojected to (100, 650).
    assert result["x"] == 100
    assert result["y"] == 650


def test_claude_provider_no_region_behaves_identically_to_before() -> None:
    """Regression guard: a call with ``region=None`` (the default)
    must look identical to the pre-#435 path.
    """
    def _no_target(*_args, **kwargs):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "content": [{
                "type": "tool_use",
                "name": "report_form_target",
                "input": {
                    "x": 500, "y": 400, "action": "click",
                    "value": "", "label": "Login",
                },
            }],
        }
        return resp

    client = AnthropicToolUseClient(api_key="k", model="m", log_prefix="t")
    provider = ClaudeFormTargetProvider(client)
    with patch("requests.post", side_effect=_no_target):
        result = provider.find_form_target(
            _img(), "Click Login", target_label="Login",
        )

    assert result is not None
    assert result["x"] == 500
    assert result["y"] == 400


def test_auto_region_form_finalize_submit_after_fill_field() -> None:
    """#435 follow-up: a ``(submit, button)`` step preceded by
    ``fill_field`` actions is a form-finalize submit (Update / Save /
    Submit) → auto-crop to ``"form-footer"``. The fill_field signal
    distinguishes this from filter-toolbar submits (Apply / Search)
    that share the same (type, kind) shape but live at the top of
    the page.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    plan_steps = [
        MicroIntent(intent="Open edit", type="submit"),
        MicroIntent(intent="Set Status", type="select_option"),
        MicroIntent(intent="Set notes", type="fill_field"),   # window hit
        MicroIntent(intent="Click Update Lead", type="submit",
                    params={"label": "Update Lead", "kind": "button"}),
    ]
    region = _auto_region_for_step(
        plan_steps[3], plan_steps[3].params,
        plan_steps=plan_steps, step_index=3,
    )
    assert region == "form-footer"


def test_auto_region_filter_toolbar_submit_gets_no_default() -> None:
    """The canonical regression case from staff-crm-long step 8:
    a ``(submit, button)`` step preceded only by ``select_option``
    (filter dropdown change) is NOT a form-finalize submit and must
    NOT default to ``"form-footer"``. The Apply / Search / Sort
    buttons live in a top toolbar; the prior flat-default cropped
    them out and the runner couldn't click them.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    plan_steps = [
        MicroIntent(intent="Go to leads", type="submit"),
        MicroIntent(intent="Filter status", type="submit"),
        MicroIntent(intent="Pick Critical", type="select_option"),
        MicroIntent(intent="Click Apply", type="submit",
                    params={"label": "Apply", "kind": "button"}),
    ]
    region = _auto_region_for_step(
        plan_steps[3], plan_steps[3].params,
        plan_steps=plan_steps, step_index=3,
    )
    assert region == ""


def test_auto_region_inference_no_default_for_unmapped_shapes() -> None:
    """For step shapes outside ``(submit, button)`` — ``nav_link``,
    ``row_link``, plain ``link``, missing kind, ``select_option``
    triggers — return empty string regardless of context. Avoids
    cropping out targets that legitimately live in non-form regions.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    # Even with a fill_field in recent context, non-button kinds
    # don't get the footer default.
    plan_steps = [
        MicroIntent(intent="Fill", type="fill_field"),
        MicroIntent(intent="x", type="submit"),
    ]
    for params in (
        {"kind": "nav_link"},
        {"kind": "row_link"},
        {"kind": "link"},
        {},
    ):
        plan_steps[1] = MicroIntent(intent="x", type="submit", params=params)
        assert _auto_region_for_step(
            plan_steps[1], plan_steps[1].params,
            plan_steps=plan_steps, step_index=1,
        ) == "", f"unmapped kind {params!r} should return empty string"


def test_auto_region_returns_empty_when_context_unavailable() -> None:
    """Legacy callers / unit tests that build a StepContext directly
    don't always have plan + step_index handy. Return empty rather
    than fall back to a flat default — under-cropping is safer than
    mis-cropping.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    step = MicroIntent(intent="x", type="submit",
                       params={"kind": "button", "label": "Update Lead"})
    assert _auto_region_for_step(step, step.params) == ""


def test_auto_region_window_size_limits_lookback() -> None:
    """The lookback window is bounded (default 4 steps). A fill_field
    deep in the plan history (e.g. login fields 20 steps earlier)
    must NOT pull a current submit into footer territory — that
    would over-broadly footer-crop. The signal is locality.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    # Long plan: login fill_fields at index 0-1, then many unrelated
    # steps, then a submit at index 10.
    plan_steps: list = []
    plan_steps.append(MicroIntent(intent="login user", type="fill_field"))
    plan_steps.append(MicroIntent(intent="login pass", type="fill_field"))
    for _ in range(8):
        plan_steps.append(MicroIntent(intent="navigate", type="submit"))
    submit_step = MicroIntent(
        intent="Apply filter", type="submit",
        params={"label": "Apply", "kind": "button"},
    )
    plan_steps.append(submit_step)
    region = _auto_region_for_step(
        submit_step, submit_step.params,
        plan_steps=plan_steps, step_index=10,
    )
    assert region == "", (
        "old fill_fields outside the window shouldn't apply footer-crop"
    )


def test_claude_provider_explicit_rect_region_reprojects() -> None:
    def _capture(*_args, **kwargs):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "content": [{
                "type": "tool_use",
                "name": "report_form_target",
                "input": {
                    "x": 10, "y": 20, "action": "click",
                    "value": "", "label": "X",
                },
            }],
        }
        return resp

    client = AnthropicToolUseClient(api_key="k", model="m", log_prefix="t")
    provider = ClaudeFormTargetProvider(client)
    with patch("requests.post", side_effect=_capture):
        result = provider.find_form_target(
            _img(1280, 800), "click",
            region={"x": 300, "y": 400, "w": 200, "h": 100},
        )
    assert result["x"] == 310
    assert result["y"] == 420
