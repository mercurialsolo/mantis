"""Tests for honest-presentation stealth mode (#823).

The #827 fingerprint diagnostic against bot.sannysoft.com showed the
Windows UA spoof leaking — the raw ``Mozilla/5.0 (X11; Linux x86_64)``
UA bled through alongside the Windows override, and WebGL kept
reporting the real Google/SwiftShader stack despite our Intel Iris
spoof. Honest mode (default-on) flips the posture: drop the platform
deception entirely, send the Linux UA that matches the binary, let
the real Linux/Mesa WebGL strings through, keep only the automation-
tell removals (``navigator.webdriver`` undefined, plugins populated,
languages non-empty).
"""

from __future__ import annotations

import pytest

from mantis_agent.gym import cdp_stealth
from mantis_agent.gym.cdp_stealth import (
    STEALTH_JS,
    STEALTH_JS_HONEST,
    _honest_ua,
    apply_ua_override,
    inject_stealth_patches,
    is_honest_mode,
)


# ── is_honest_mode env gating ─────────────────────────────────────


def test_honest_mode_defaults_true(monkeypatch):
    monkeypatch.delenv("MANTIS_STEALTH_HONEST", raising=False)
    assert is_honest_mode() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_honest_mode_falsy_values(monkeypatch, value):
    monkeypatch.setenv("MANTIS_STEALTH_HONEST", value)
    assert is_honest_mode() is False


@pytest.mark.parametrize("value", ["1", "true", "anything"])
def test_honest_mode_truthy_values(monkeypatch, value):
    monkeypatch.setenv("MANTIS_STEALTH_HONEST", value)
    assert is_honest_mode() is True


# ── _honest_ua structure ──────────────────────────────────────────


def test_honest_ua_advertises_linux(monkeypatch):
    """The honest UA must say Linux x86_64 — no platform deception."""
    monkeypatch.setattr(cdp_stealth, "_CHROME_MAJOR_CACHE", 132)
    ua, metadata = _honest_ua()
    assert "X11; Linux x86_64" in ua
    assert "Chrome/132." in ua
    assert metadata["platform"] == "Linux"


def test_honest_ua_metadata_is_self_consistent(monkeypatch):
    """sec-ch-ua-* brands and fullVersionList must agree with the UA
    string's Chrome major. A mixed signal across surfaces is what
    we're avoiding."""
    monkeypatch.setattr(cdp_stealth, "_CHROME_MAJOR_CACHE", 132)
    ua, metadata = _honest_ua()
    chrome_brand = next(
        (b for b in metadata["brands"] if "Google Chrome" in b["brand"]),
        None,
    )
    assert chrome_brand is not None
    assert chrome_brand["version"] == "132"
    assert metadata["fullVersion"].startswith("132.")
    assert "Chrome/132." in ua


def test_honest_ua_falls_back_when_chrome_missing(monkeypatch):
    """When the chrome binary can't be probed, fall back to the
    documented baseline rather than emit Chrome/0.0.0.0."""
    monkeypatch.setattr(cdp_stealth, "_CHROME_MAJOR_CACHE", 0)
    ua, metadata = _honest_ua()
    fallback = cdp_stealth._HONEST_CHROME_MAJOR_FALLBACK
    assert f"Chrome/{fallback}." in ua
    assert metadata["fullVersion"].startswith(f"{fallback}.")


def test_honest_ua_never_mentions_windows(monkeypatch):
    monkeypatch.setattr(cdp_stealth, "_CHROME_MAJOR_CACHE", 132)
    ua, metadata = _honest_ua()
    assert "Windows" not in ua
    assert "Win" not in ua
    assert metadata["platform"].lower() == "linux"


# ── STEALTH_JS_HONEST shape ───────────────────────────────────────


def test_honest_js_hides_webdriver():
    """The one signal we still spoof in honest mode is the automation
    context tell — navigator.webdriver."""
    assert "navigator, 'webdriver'" in STEALTH_JS_HONEST
    assert "get: () => undefined" in STEALTH_JS_HONEST


def test_honest_js_populates_plugins():
    """Headless Chromium has zero plugins; real Chrome on every
    platform has the built-in PDF viewers. Populating the array is
    truthful (real Chrome users see these)."""
    assert "PDF Viewer" in STEALTH_JS_HONEST
    assert "length" in STEALTH_JS_HONEST


def test_honest_js_does_not_patch_webgl():
    """The legacy Intel Iris OpenGL Engine string is a macOS string —
    contradicted our Linux TLS stack. Honest mode lets Mesa /
    SwiftShader report truthfully."""
    assert "Intel Iris" not in STEALTH_JS_HONEST
    assert "37445" not in STEALTH_JS_HONEST  # UNMASKED_VENDOR_WEBGL constant
    assert "37446" not in STEALTH_JS_HONEST  # UNMASKED_RENDERER_WEBGL constant


def test_honest_js_does_not_patch_canvas():
    """Per-call canvas noise was itself a tell — real users don't
    perturb their canvas hash every call."""
    assert "toDataURL" not in STEALTH_JS_HONEST
    assert "getImageData" not in STEALTH_JS_HONEST


def test_honest_js_does_not_patch_audio():
    assert "getChannelData" not in STEALTH_JS_HONEST
    assert "OfflineAudioContext" not in STEALTH_JS_HONEST


def test_honest_js_does_not_spoof_platform():
    """navigator.platform = 'Win32' on a Linux binary creates a
    mismatch with the TLS stack. Honest mode skips the spoof."""
    assert "Win32" not in STEALTH_JS_HONEST
    assert "navigator.platform" not in STEALTH_JS_HONEST
    assert "userAgentData" not in STEALTH_JS_HONEST


def test_honest_js_does_not_fake_fonts():
    """Claiming Helvetica Neue on Linux contradicts
    OffscreenCanvas text-metric cross-checks."""
    assert "Helvetica Neue" not in STEALTH_JS_HONEST
    assert "document.fonts" not in STEALTH_JS_HONEST


def test_honest_js_strictly_smaller_than_deceptive():
    """The honest payload is the minimal patch set — should be
    substantially smaller than the legacy deceptive set."""
    assert len(STEALTH_JS_HONEST) < len(STEALTH_JS)


# ── inject_stealth_patches routing ────────────────────────────────


def test_inject_uses_honest_source_when_honest(monkeypatch):
    monkeypatch.delenv("MANTIS_STEALTH_HONEST", raising=False)
    captured: dict = {}

    def fake_call(method, params):
        captured["method"] = method
        captured["source"] = params["source"]
        return True, {"identifier": "x"}

    inject_stealth_patches(fake_call)
    assert captured["method"] == "Page.addScriptToEvaluateOnNewDocument"
    assert captured["source"] is STEALTH_JS_HONEST


def test_inject_uses_legacy_source_when_opted_out(monkeypatch):
    monkeypatch.setenv("MANTIS_STEALTH_HONEST", "0")
    captured: dict = {}

    def fake_call(method, params):
        captured["source"] = params["source"]
        return True, {"identifier": "x"}

    inject_stealth_patches(fake_call)
    assert captured["source"] is STEALTH_JS


def test_inject_noop_when_cdp_stealth_disabled(monkeypatch):
    """``MANTIS_CDP_STEALTH=0`` disables the patch path entirely —
    honest mode is a sub-flag, not an override."""
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    monkeypatch.setenv("MANTIS_STEALTH_HONEST", "1")
    called: list = []

    def fake_call(method, params):
        called.append((method, params))
        return True, {}

    result = inject_stealth_patches(fake_call)
    assert result is False
    assert called == []


# ── apply_ua_override routing ─────────────────────────────────────


def test_ua_override_sends_linux_when_honest(monkeypatch):
    monkeypatch.delenv("MANTIS_STEALTH_HONEST", raising=False)
    monkeypatch.setattr(cdp_stealth, "_CHROME_MAJOR_CACHE", 132)
    captured: dict = {}

    def fake_call(method, params):
        captured["params"] = params
        return True, {}

    apply_ua_override(fake_call)
    p = captured["params"]
    assert "Linux" in p["userAgent"]
    assert "Windows" not in p["userAgent"]
    assert p["platform"] == "Linux"
    assert p["userAgentMetadata"]["platform"] == "Linux"


def test_ua_override_sends_windows_when_opted_out(monkeypatch):
    """Legacy deceptive path still reachable via opt-out."""
    monkeypatch.setenv("MANTIS_STEALTH_HONEST", "0")
    captured: dict = {}

    def fake_call(method, params):
        captured["params"] = params
        return True, {}

    apply_ua_override(fake_call)
    p = captured["params"]
    assert "Windows NT" in p["userAgent"]
    assert p["platform"] == "Windows"
    assert p["userAgentMetadata"]["platform"] == "Windows"


def test_ua_override_noop_when_cdp_stealth_disabled(monkeypatch):
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    called: list = []

    def fake_call(method, params):
        called.append((method, params))
        return True, {}

    result = apply_ua_override(fake_call)
    assert result is False
    assert called == []
