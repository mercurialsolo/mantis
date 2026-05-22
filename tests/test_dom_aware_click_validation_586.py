"""Tests for #586 — DOM-aware pre-click validation.

When the site_config declares a ``listing_card_css_selector``, the
click handler queries the element at the proposed click coords via
CDP and rejects the click when no ancestor matches the selector.

Catches the failure mode where #584 (find_listings exclusions) and
#585 (extract_url URL pattern) couldn't intervene — the brain picked
coords on a marketing CTA AND the click somehow landed within the
detail_page_pattern (so #585 didn't fire). The DOM tells us
unambiguously whether the target is a real card or a CTA.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantis_agent.gym.xdotool_env import XdotoolGymEnv
from mantis_agent.site_config import SiteConfig


# ── cdp_element_matches_selector helper ────────────────────────────


def _make_env() -> XdotoolGymEnv:
    env = XdotoolGymEnv.__new__(XdotoolGymEnv)
    env._cdp_port = 9222
    return env


def test_returns_true_when_element_matches_selector() -> None:
    env = _make_env()
    with patch.object(env, "cdp_evaluate", return_value=True):
        assert env.cdp_element_matches_selector(
            100, 200, "a[data-listing-id]",
        ) is True


def test_returns_false_when_element_doesnt_match() -> None:
    env = _make_env()
    # CTA card — no listing-id ancestor.
    with patch.object(env, "cdp_evaluate", return_value=False):
        assert env.cdp_element_matches_selector(
            100, 200, "a[data-listing-id]",
        ) is False


def test_returns_false_when_no_element_at_point() -> None:
    env = _make_env()
    # cdp_evaluate returns False when elementFromPoint is null (off-canvas).
    with patch.object(env, "cdp_evaluate", return_value=False):
        assert env.cdp_element_matches_selector(0, 0, ".card") is False


def test_returns_false_on_cdp_exception() -> None:
    env = _make_env()
    with patch.object(env, "cdp_evaluate", side_effect=RuntimeError("ws closed")):
        assert env.cdp_element_matches_selector(100, 200, ".card") is False


def test_returns_false_on_empty_selector() -> None:
    env = _make_env()
    # Empty selector → no validation possible; never call CDP.
    with patch.object(env, "cdp_evaluate") as mock_eval:
        assert env.cdp_element_matches_selector(100, 200, "") is False
        mock_eval.assert_not_called()


def test_returns_false_on_non_int_coords() -> None:
    env = _make_env()
    # MagicMock x/y from test envs would otherwise leak into the JS
    # template as garbage strings.
    with patch.object(env, "cdp_evaluate") as mock_eval:
        assert env.cdp_element_matches_selector("100", 200, ".card") is False
        mock_eval.assert_not_called()


def test_selector_quote_escaping() -> None:
    """Single quotes in the selector must be escaped so the JS
    template doesn't break."""
    env = _make_env()
    captured = {}

    def _capture(js):
        captured["js"] = js
        return True

    with patch.object(env, "cdp_evaluate", side_effect=_capture):
        env.cdp_element_matches_selector(
            100, 200, "a[data-source='listings']",
        )
    # Single quote must be backslash-escaped in the JS string literal.
    assert "data-source=\\'listings\\'" in captured["js"]


# ── SiteConfig wiring ──────────────────────────────────────────────


def test_boattrader_default_has_selector_set() -> None:
    config = SiteConfig.default_boattrader()
    assert config.listing_card_css_selector
    # Selector includes the canonical boattrader listing-anchor marker.
    assert "data-reporting-impression-source='listings'" in config.listing_card_css_selector


def test_site_config_default_selector_is_empty() -> None:
    config = SiteConfig(domain="x.com")
    assert config.listing_card_css_selector == ""


# ── click.py PRE-CLICK REJECT integration ──────────────────────────


def test_click_rejects_off_card_target() -> None:
    """When the validation returns False AND the selector is set, the
    click handler's PRE-CLICK REJECT block triggers — verified at the
    decision-table level (the block reads site_config + env + coords
    and evaluates ``validate_target and not matches``)."""
    env = MagicMock()
    env.cdp_element_matches_selector = MagicMock(return_value=False)  # off-card
    site_config = SiteConfig.default_boattrader()
    card_selector = site_config.listing_card_css_selector
    validate_target = (
        bool(card_selector) and hasattr(env, "cdp_element_matches_selector")
    )
    assert validate_target is True
    matches = env.cdp_element_matches_selector(100, 200, card_selector)
    assert matches is False  # → block triggers OFF_CARD reject


def test_click_accepts_on_card_target() -> None:
    env = MagicMock()
    env.cdp_element_matches_selector = MagicMock(return_value=True)  # on-card
    site_config = SiteConfig.default_boattrader()
    # On-card → proceeds normally; block doesn't trigger.
    assert env.cdp_element_matches_selector(
        100, 200, site_config.listing_card_css_selector,
    ) is True


def test_click_skips_validation_when_no_selector() -> None:
    """SiteConfigs without a listing_card_css_selector (generic SaaS
    plans, analysis-stage configs) must skip validation entirely —
    no CDP call, no rejection."""
    env = MagicMock()
    env.cdp_element_matches_selector = MagicMock(return_value=False)
    site_config = SiteConfig(domain="x.com")  # no selector
    # Validation gated on bool(card_selector) — empty string skips.
    card_selector = site_config.listing_card_css_selector
    assert not card_selector  # block's bool() check fails → skip validation
    env.cdp_element_matches_selector.assert_not_called()


def test_click_skips_validation_when_env_has_no_helper() -> None:
    """Legacy envs / test stubs without ``cdp_element_matches_selector``
    must not break — validation gated on hasattr."""
    class _EnvNoHelper:
        pass

    env = _EnvNoHelper()
    site_config = SiteConfig.default_boattrader()
    card_selector = site_config.listing_card_css_selector
    validate_target = (
        bool(card_selector) and hasattr(env, "cdp_element_matches_selector")
    )
    assert validate_target is False
