"""Tests for #539 — CDP-injected browser-fingerprint stealth patches.

The stealth module exposes:

* ``STEALTH_JS`` — the JS payload (one constant, easy to grep / diff)
* ``is_enabled()`` — gated by ``MANTIS_CDP_STEALTH`` env var
* ``inject_stealth_patches(cdp_call)`` — wraps the SDK's
  ``Page.addScriptToEvaluateOnNewDocument`` call with the env-var gate
  and exception swallow

The xdotool_env's ``_start_browser`` calls inject after the 3s post-
launch settle (covered by source-level checks here; full env tests
require Xvfb which isn't available in CI).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock


from mantis_agent.gym import cdp_stealth


# ── STEALTH_JS content (each block is bisectable) ────────────────────────


def test_stealth_js_patches_navigator_webdriver():
    """The webdriver patch is the most-commonly-cited stealth need;
    must be present so reviewers can verify."""
    assert "navigator" in cdp_stealth.STEALTH_JS
    assert "webdriver" in cdp_stealth.STEALTH_JS
    # Specifically returns undefined (not "false" or similar).
    assert "() => undefined" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_permissions_query():
    """Notifications permission must align with Notification.permission
    (the headless-Chrome tell is returning 'denied' from the
    permissions API while Notification.permission is 'default')."""
    assert "permissions" in cdp_stealth.STEALTH_JS
    assert "notifications" in cdp_stealth.STEALTH_JS
    assert "Notification.permission" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_navigator_plugins():
    """Empty navigator.plugins is a headless Chrome tell."""
    assert "plugins" in cdp_stealth.STEALTH_JS
    # Must return a non-empty plugin array — check for at least one
    # canonical PDF Viewer entry.
    assert "PDF Viewer" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_navigator_languages():
    """Empty navigator.languages is a headless tell."""
    assert "languages" in cdp_stealth.STEALTH_JS
    assert "en-US" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_chrome_runtime():
    """window.chrome.runtime missing/incomplete is a tell."""
    assert "window.chrome" in cdp_stealth.STEALTH_JS
    assert "chrome.runtime" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_webgl_vendor():
    """WebGL vendor/renderer must return realistic strings (not
    SwiftShader / Google Inc.)."""
    assert "WebGL" in cdp_stealth.STEALTH_JS
    assert "getParameter" in cdp_stealth.STEALTH_JS
    # 37445/37446 are UNMASKED_VENDOR_WEBGL / UNMASKED_RENDERER_WEBGL.
    assert "37445" in cdp_stealth.STEALTH_JS
    assert "37446" in cdp_stealth.STEALTH_JS


def test_stealth_js_is_wrapped_in_iife():
    """The whole patch set runs inside an IIFE so its locals don't
    leak into the page's global namespace (a tell in itself)."""
    src = cdp_stealth.STEALTH_JS.strip()
    assert src.startswith("(() => {") or src.startswith("(function"), (
        "STEALTH_JS must be wrapped in an IIFE to avoid global-scope leaks"
    )


def test_stealth_js_each_block_is_try_catch_wrapped():
    """Individual patches MUST be try/catch wrapped so one failure
    doesn't void the rest of the patch set (some sites disable
    specific surfaces; we want partial coverage, not zero)."""
    # Loose check: each numbered patch block has a ``try`` after it.
    # 8 blocks expected at the moment, each with its own try.
    try_count = cdp_stealth.STEALTH_JS.count("try {")
    catch_count = cdp_stealth.STEALTH_JS.count("catch (e)")
    assert try_count >= 6, f"Expected >=6 try blocks, got {try_count}"
    assert catch_count >= 6, f"Expected >=6 catch blocks, got {catch_count}"


# ── is_enabled() gate ────────────────────────────────────────────────────


def test_is_enabled_default_true(monkeypatch):
    """Default-on per the issue acceptance — most production traffic
    benefits from stealth, opt-out is rare."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    assert cdp_stealth.is_enabled() is True


def test_is_enabled_explicit_zero_disables(monkeypatch):
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    assert cdp_stealth.is_enabled() is False


def test_is_enabled_falsey_strings_disable(monkeypatch):
    for v in ["false", "no", "off", "FALSE", "False"]:
        monkeypatch.setenv("MANTIS_CDP_STEALTH", v)
        assert cdp_stealth.is_enabled() is False, f"{v!r} should disable"


def test_is_enabled_truthy_strings_enable(monkeypatch):
    for v in ["1", "true", "yes", "on", "TRUE"]:
        monkeypatch.setenv("MANTIS_CDP_STEALTH", v)
        assert cdp_stealth.is_enabled() is True, f"{v!r} should enable"


def test_is_enabled_garbage_falls_through_to_enabled(monkeypatch):
    """Unknown string defaults to enabled (don't accidentally disable
    on a typo'd env var)."""
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "garbage")
    # garbage is not in the disable-set → enabled.
    assert cdp_stealth.is_enabled() is True


# ── inject_stealth_patches() ─────────────────────────────────────────────


def test_inject_calls_cdp_with_correct_method_and_source(monkeypatch):
    """The single observable side-effect: a CDP call to
    ``Page.addScriptToEvaluateOnNewDocument`` with the STEALTH_JS
    string as ``source``."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    cdp_call = MagicMock(return_value=(True, {"identifier": "id-001"}))
    ok = cdp_stealth.inject_stealth_patches(cdp_call)
    assert ok is True
    cdp_call.assert_called_once()
    method, params = cdp_call.call_args.args
    assert method == "Page.addScriptToEvaluateOnNewDocument"
    assert params == {"source": cdp_stealth.STEALTH_JS}


def test_inject_noop_when_disabled(monkeypatch):
    """Disabled gate → no CDP call, returns False, no error."""
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    cdp_call = MagicMock()
    ok = cdp_stealth.inject_stealth_patches(cdp_call)
    assert ok is False
    cdp_call.assert_not_called()


def test_inject_returns_false_on_cdp_not_ok(monkeypatch):
    """When CDP returns ``(False, ...)`` (e.g. browser not yet ready
    or CDP unreachable), inject returns False — caller can log + move
    on."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    cdp_call = MagicMock(return_value=(False, {}))
    ok = cdp_stealth.inject_stealth_patches(cdp_call)
    assert ok is False


def test_inject_swallows_cdp_exception(monkeypatch):
    """If the CDP shim raises (network blip, websocket import missing,
    timeout), inject must NOT propagate — stealth setup never breaks
    a run."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    cdp_call = MagicMock(side_effect=RuntimeError("simulated CDP failure"))
    ok = cdp_stealth.inject_stealth_patches(cdp_call)
    assert ok is False  # exception swallowed → returns False


# ── xdotool_env wiring (source-level check; full env test needs Xvfb) ────


def test_xdotool_env_start_browser_calls_inject_stealth():
    """``XdotoolGymEnv._start_browser`` must call
    ``inject_stealth_patches`` after the browser-launch settle so the
    patches are registered before any runner navigation."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    src = inspect.getsource(XdotoolGymEnv._start_browser)
    assert "inject_stealth_patches" in src, (
        "_start_browser must call inject_stealth_patches after browser launch"
    )
    # Must be inside a try/except so a CDP failure during stealth
    # setup doesn't crash the browser-start path.
    assert "try:" in src and "except" in src, (
        "_start_browser must guard inject_stealth_patches in try/except"
    )


def test_xdotool_env_import_module_clean(monkeypatch):
    """Importing the gym package must not fail when MANTIS_CDP_STEALTH
    is set or unset (catches accidental import-time evaluation)."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    import importlib
    import mantis_agent.gym.cdp_stealth as mod
    importlib.reload(mod)
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    importlib.reload(mod)


# ── Layer-2 patches (canvas/audio/font/platform) ─────────────────────────


def test_stealth_js_patches_canvas_fingerprint():
    """Canvas toDataURL is the #1 CF fingerprint surface after
    navigator.* — must inject per-pixel noise to break the hash."""
    assert "HTMLCanvasElement" in cdp_stealth.STEALTH_JS
    assert "toDataURL" in cdp_stealth.STEALTH_JS
    assert "getImageData" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_audio_fingerprint():
    """OfflineAudioContext rendering is the #2 fingerprint surface
    — patch AudioBuffer.getChannelData with imperceptible noise."""
    assert "AudioContext" in cdp_stealth.STEALTH_JS
    assert "getChannelData" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_font_enumeration():
    """document.fonts.check must return true for the canonical
    Windows/macOS fonts that CF probes — sparse availability is
    a strong Linux/headless tell."""
    assert "document.fonts" in cdp_stealth.STEALTH_JS
    # Must whitelist Segoe UI (Windows) and Helvetica Neue (macOS)
    # — the two most-probed canary fonts.
    assert "Segoe UI" in cdp_stealth.STEALTH_JS
    assert "Helvetica Neue" in cdp_stealth.STEALTH_JS


def test_stealth_js_patches_platform_spoofing():
    """navigator.platform + navigator.userAgentData must claim
    Windows so the JS-observable surface matches the request-
    layer UA override applied via CDP."""
    assert "navigator" in cdp_stealth.STEALTH_JS
    assert "platform" in cdp_stealth.STEALTH_JS
    assert "Win32" in cdp_stealth.STEALTH_JS
    assert "userAgentData" in cdp_stealth.STEALTH_JS
    assert "Windows" in cdp_stealth.STEALTH_JS


# ── UA override (sec-ch-ua headers) ──────────────────────────────────────


def test_apply_ua_override_calls_network_setuseragent(monkeypatch):
    """The single observable side-effect: a CDP call to
    ``Network.setUserAgentOverride`` with a Windows Chrome UA +
    full UA-CH metadata so both request-layer and JS-layer
    fingerprints match."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    cdp_call = MagicMock(return_value=(True, {}))
    ok = cdp_stealth.apply_ua_override(cdp_call)
    assert ok is True
    cdp_call.assert_called_once()
    method, params = cdp_call.call_args.args
    assert method == "Network.setUserAgentOverride"
    # Must spoof a Windows + Chrome 132 UA string.
    assert "Windows NT 10.0" in params["userAgent"]
    assert "Chrome/132" in params["userAgent"]
    # Must include userAgentMetadata for the sec-ch-ua-* headers.
    md = params["userAgentMetadata"]
    assert md["platform"] == "Windows"
    assert md["mobile"] is False
    assert any(b["brand"] == "Google Chrome" for b in md["brands"])


def test_apply_ua_override_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("MANTIS_CDP_STEALTH", "0")
    cdp_call = MagicMock()
    ok = cdp_stealth.apply_ua_override(cdp_call)
    assert ok is False
    cdp_call.assert_not_called()


def test_apply_ua_override_swallows_exception(monkeypatch):
    """CDP raise → swallowed; never breaks browser startup."""
    monkeypatch.delenv("MANTIS_CDP_STEALTH", raising=False)
    cdp_call = MagicMock(side_effect=RuntimeError("CDP unreachable"))
    ok = cdp_stealth.apply_ua_override(cdp_call)
    assert ok is False


def test_xdotool_env_start_browser_calls_ua_override():
    """``_start_browser`` must call both inject_stealth_patches AND
    apply_ua_override — they cover orthogonal layers (JS-side vs
    request-layer)."""
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    src = inspect.getsource(XdotoolGymEnv._start_browser)
    assert "apply_ua_override" in src, (
        "_start_browser must call apply_ua_override alongside "
        "inject_stealth_patches"
    )
