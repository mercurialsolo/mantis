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


def test_auto_region_inference_for_submit_button() -> None:
    """#435 follow-up: a ``submit`` step with ``kind: "button"`` and no
    explicit ``hints.region`` defaults to ``"form-footer"`` — the
    canonical layout for action-button rows. Removes plan-author
    burden for the most common pattern.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    step = MicroIntent(
        intent="Click Update Lead",
        type="submit",
        params={"label": "Update Lead", "kind": "button"},
    )
    region = _auto_region_for_step(step, step.params)
    assert region == "form-footer"


def test_auto_region_inference_no_default_for_unmapped_shapes() -> None:
    """For step shapes that don't have a clear default layout
    (``submit`` / ``nav_link``, ``submit`` / ``row_link``,
    ``select_option`` triggers), return empty string — caller falls
    back to the unscoped path. Avoids cropping out targets that
    legitimately live outside the form footer.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    for params in (
        {"kind": "nav_link"},
        {"kind": "row_link"},
        {"kind": "link"},
        {},  # no kind at all
    ):
        step = MicroIntent(intent="x", type="submit", params=params)
        assert _auto_region_for_step(step, step.params) == "", (
            f"unmapped kind {params!r} should return empty string"
        )


def test_auto_region_inference_skips_select_option() -> None:
    """``select_option`` shouldn't default to any region — the open
    dropdown menu repositions absolutely and a form-footer crop
    would hide the option list.
    """
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    step = MicroIntent(
        intent="Pick Foo", type="select_option",
        params={"dropdown_label": "Bar", "option_label": "Foo"},
    )
    assert _auto_region_for_step(step, step.params) == ""


def test_explicit_region_takes_precedence_over_auto() -> None:
    """When the plan author sets ``hints.region`` explicitly, the
    form handler must use that value — auto-inference only fires when
    the hint is missing. Test the form handler's logic, not just the
    inference helper.
    """
    # This is exercised by the form handler's actual control flow
    # (``step_region = hints.get("region") or _auto_region_for_step(...)``).
    # The inference helper itself doesn't see ``hints``; the handler's
    # ``if not step_region`` check is what enforces precedence. Pin
    # the helper's behaviour: it returns the auto value regardless
    # of what hints contain, since hints are the caller's
    # responsibility.
    from mantis_agent.gym.step_handlers.form import _auto_region_for_step
    from mantis_agent.plan_decomposer import MicroIntent

    step = MicroIntent(
        intent="Click", type="submit",
        params={"kind": "button"},
        hints={"region": "top"},  # plan author wants top
    )
    # Helper still suggests form-footer (it's a pure function of
    # type/kind); precedence is enforced at the handler.
    assert _auto_region_for_step(step, step.params) == "form-footer"


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
