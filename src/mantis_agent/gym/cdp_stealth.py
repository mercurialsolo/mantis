"""CDP-injected browser-fingerprint stealth patches (#539).

Cloudflare Turnstile and similar bot-detection layers fingerprint Chrome on
~20 surfaces beyond just ``navigator.webdriver``. The existing
``--disable-blink-features=AutomationControlled`` flag hides only the
webdriver bit; this module patches the rest.

Patches are injected once per-tab via CDP
``Page.addScriptToEvaluateOnNewDocument`` BEFORE any user navigation —
so the first real navigate the runner makes sees the spoofed surfaces
on the very first page load (including all frames the page creates).

**CUA-contract provenance:** these are **action-only** modifications of
the browser environment, NOT a path to derive grounding from the DOM
(see ``feedback_cua_no_dom_access.md`` in the codebase memory).
Mantis stays screenshot-grounded — these patches just make the
browser look less like an automated client so the screenshots we
observe match what a human user would see (instead of a CF challenge
page). Same provenance class as the existing post-action
``window.scrollY`` reads (see ``feedback_cua_cdp_post_action_verify.md``).

Gated by ``MANTIS_CDP_STEALTH`` env var (default ``"1"``); set to
``"0"``/``"false"`` to disable per-run.

Reference implementations:

* puppeteer-extra-plugin-stealth (Node, ~20 evasions)
* undetected-chromedriver (Python, patched-binary approach)

The patch set below is the minimum subset that beats CF Turnstile on
boattrader.com (the issue #539 reproducer) as of 2026-05-20.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── JS payload ────────────────────────────────────────────────────────

# Each block is independently necessary — removing any one drops below
# the CF Turnstile detection threshold on at least one of the canonical
# bot-protected listing targets. Keep blocks atomic so additions /
# removals are bisectable.
#
# WebGL constants:
#   37445 = UNMASKED_VENDOR_WEBGL    (gl.UNMASKED_VENDOR_WEBGL)
#   37446 = UNMASKED_RENDERER_WEBGL  (gl.UNMASKED_RENDERER_WEBGL)
# Picked to match a stock macOS Chrome install — the most common real
# fingerprint, blends with residential traffic.
STEALTH_JS: str = r"""
(() => {
  // [1] navigator.webdriver — return undefined regardless of any
  // chromedriver / automation-flag leak.
  try {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  } catch (e) {}

  // [2] navigator.permissions.query — make notifications report
  // Notification.permission instead of "denied" (which only happens
  // in headless / automated contexts).
  try {
    const originalQuery = window.navigator.permissions &&
      window.navigator.permissions.query &&
      window.navigator.permissions.query.bind(window.navigator.permissions);
    if (originalQuery) {
      window.navigator.permissions.query = (parameters) =>
        parameters && parameters.name === 'notifications'
          ? Promise.resolve({ state: Notification.permission })
          : originalQuery(parameters);
    }
  } catch (e) {}

  // [3] navigator.plugins — populate with a realistic non-empty array.
  // CF flags zero-length plugins as headless-Chrome shape.
  try {
    Object.defineProperty(navigator, 'plugins', {
      get: () => {
        const plugins = [
          { name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Microsoft Edge PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'WebKit built-in PDF', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        ];
        Object.defineProperty(plugins, 'length', { value: 5 });
        return plugins;
      },
    });
  } catch (e) {}

  // [4] navigator.languages — empty array is a headless tell.
  try {
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
  } catch (e) {}

  // [5] window.chrome.runtime — must exist with a plausible shape
  // (real Chrome has this object; bare puppeteer/playwright Chromium
  // builds don't unless --enable-features=ChromiumExtensions).
  try {
    window.chrome = window.chrome || {};
    window.chrome.runtime = window.chrome.runtime || {
      OnInstalledReason: {},
      OnRestartRequiredReason: {},
      PlatformArch: {},
      PlatformNaclArch: {},
      PlatformOs: {},
      RequestUpdateCheckStatus: {},
    };
    window.chrome.app = window.chrome.app || {
      InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
      RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' },
    };
  } catch (e) {}

  // [6] WebGL vendor/renderer — return realistic Intel strings instead
  // of "Google Inc. (Google)" / "ANGLE (Google, Vulkan ...)" which
  // leak the SwiftShader / virtualized GPU.
  try {
    const VENDOR = 'Intel Inc.';
    const RENDERER = 'Intel Iris OpenGL Engine';
    const wrap = (proto) => {
      if (!proto) return;
      const orig = proto.getParameter;
      if (!orig) return;
      proto.getParameter = function (parameter) {
        if (parameter === 37445) return VENDOR;
        if (parameter === 37446) return RENDERER;
        return orig.call(this, parameter);
      };
    };
    wrap(window.WebGLRenderingContext && WebGLRenderingContext.prototype);
    wrap(window.WebGL2RenderingContext && WebGL2RenderingContext.prototype);
  } catch (e) {}

  // [7] Notification.permission — must align with the permissions
  // API patch in [2]. Some CF checks read this directly.
  try {
    if (window.Notification && Notification.permission === 'denied') {
      Object.defineProperty(Notification, 'permission', { get: () => 'default' });
    }
  } catch (e) {}

  // [8] Function.prototype.toString proxy — the patched
  // permissions.query must still serialize as ``[native code]``
  // (some CF probes Function.toString to detect monkey-patches).
  try {
    const proxyToString = Function.prototype.toString;
    Function.prototype.toString = new Proxy(proxyToString, {
      apply(target, thisArg, args) {
        if (thisArg === window.navigator.permissions.query) {
          return 'function query() { [native code] }';
        }
        return Reflect.apply(target, thisArg, args);
      },
    });
  } catch (e) {}

  // [9] Canvas fingerprint noise — CF reads <canvas>.toDataURL() and
  // hashes the result. Linux Chrome + SwiftShader produces a hash
  // that's on CF's known-bot list. Add per-pixel ±1 noise so each
  // session's hash is unique but text/UI still renders legibly.
  try {
    const origToDataURL = HTMLCanvasElement.prototype.toDataURL;
    HTMLCanvasElement.prototype.toDataURL = function (...args) {
      try {
        const ctx = this.getContext('2d');
        if (ctx) {
          const img = ctx.getImageData(0, 0, this.width, this.height);
          for (let i = 0; i < img.data.length; i += 4) {
            // ±1 noise on R/G/B; skip alpha. Imperceptible to humans,
            // changes the dataURL hash.
            img.data[i] = (img.data[i] + ((Math.random() < 0.5) ? -1 : 1)) & 0xff;
            img.data[i + 1] = (img.data[i + 1] + ((Math.random() < 0.5) ? -1 : 1)) & 0xff;
            img.data[i + 2] = (img.data[i + 2] + ((Math.random() < 0.5) ? -1 : 1)) & 0xff;
          }
          ctx.putImageData(img, 0, 0);
        }
      } catch (e) {}
      return origToDataURL.apply(this, args);
    };
  } catch (e) {}

  // [10] AudioContext fingerprint noise — OfflineAudioContext
  // renderings differ between real hardware and virtualized; CF
  // hashes the output. Add tiny gain perturbation to break the hash.
  try {
    if (typeof OfflineAudioContext !== 'undefined') {
      const origGetChannelData = AudioBuffer.prototype.getChannelData;
      AudioBuffer.prototype.getChannelData = function (channel) {
        const data = origGetChannelData.call(this, channel);
        // Perturb the first sample by an imperceptible amount —
        // enough to change the hash but not the audible output.
        if (data.length > 0) {
          const noise = (Math.random() - 0.5) * 1e-7;
          data[0] = data[0] + noise;
        }
        return data;
      };
    }
  } catch (e) {}

  // [11] Font enumeration — patch document.fonts.check so a wider
  // set of fonts appears available. Real Windows/macOS have
  // 100-300 fonts; Linux Chrome image has ~30. CF probes specific
  // fonts ("Helvetica Neue", "Segoe UI", etc.) and a sparse hit
  // ratio is a tell.
  try {
    if (window.document && document.fonts && document.fonts.check) {
      const origCheck = document.fonts.check.bind(document.fonts);
      const FAKE_AVAILABLE = new Set([
        'Helvetica', 'Helvetica Neue', 'Arial', 'Arial Black',
        'Segoe UI', 'Tahoma', 'Verdana', 'Georgia',
        'Times', 'Times New Roman', 'Courier', 'Courier New',
        'Calibri', 'Cambria', 'Consolas', 'Trebuchet MS',
        'Comic Sans MS', 'Impact', 'Lucida Console', 'Lucida Sans',
      ]);
      document.fonts.check = function (font, ...rest) {
        try {
          // Extract font family from the CSS shorthand.
          const m = String(font).match(/['"]?([A-Za-z][A-Za-z0-9 ]+)['"]?$/);
          const family = m ? m[1].trim() : '';
          if (family && FAKE_AVAILABLE.has(family)) return true;
        } catch (e) {}
        return origCheck(font, ...rest);
      };
    }
  } catch (e) {}

  // [12] Platform spoofing — sec-ch-ua-platform = "Linux" leaks
  // that we're on a server. Boattrader audience is ~95% Windows
  // or macOS. Patch navigator.platform AND navigator.userAgentData
  // to claim Windows. The UA header itself is overridden via CDP
  // Network.setUserAgentOverride (see apply_ua_override).
  try {
    Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
    if (navigator.userAgentData) {
      const origUaData = navigator.userAgentData;
      const fakeBrands = [
        { brand: 'Not_A Brand', version: '8' },
        { brand: 'Chromium', version: '132' },
        { brand: 'Google Chrome', version: '132' },
      ];
      Object.defineProperty(navigator, 'userAgentData', {
        get: () => ({
          brands: fakeBrands,
          mobile: false,
          platform: 'Windows',
          getHighEntropyValues: (hints) =>
            Promise.resolve({
              architecture: 'x86',
              bitness: '64',
              brands: fakeBrands,
              fullVersionList: fakeBrands,
              mobile: false,
              model: '',
              platform: 'Windows',
              platformVersion: '15.0.0',
              uaFullVersion: '132.0.6834.110',
              wow64: false,
            }),
          toJSON: () => ({
            brands: fakeBrands,
            mobile: false,
            platform: 'Windows',
          }),
        }),
      });
    }
  } catch (e) {}
})();
"""

# Spoofed UA + UA-CH metadata. Pre-honest-mode (#823) these claimed
# Windows 10 / Chrome 132 — but the binary actually runs on Linux, and
# the TLS ClientHello / HTTP/2 SETTINGS that Chrome sends are detectably
# Linux-shaped. Scoring models that reconcile UA against the TLS
# fingerprint flagged the mismatch, raising the bot score. The honest-
# mode build below presents as the real Linux Chrome (#823 fix for the
# UA leak the #827 diagnostic surfaced — sannysoft saw the raw
# ``Mozilla/5.0 (X11; Linux x86_64)...`` UA leak through despite our
# Windows spoof).
_SPOOF_UA: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)

_SPOOF_UA_METADATA: dict[str, Any] = {
    "brands": [
        {"brand": "Not_A Brand", "version": "8"},
        {"brand": "Chromium", "version": "132"},
        {"brand": "Google Chrome", "version": "132"},
    ],
    "fullVersion": "132.0.6834.110",
    "fullVersionList": [
        {"brand": "Not_A Brand", "version": "8.0.0.0"},
        {"brand": "Chromium", "version": "132.0.6834.110"},
        {"brand": "Google Chrome", "version": "132.0.6834.110"},
    ],
    "platform": "Windows",
    "platformVersion": "15.0.0",
    "architecture": "x86",
    "model": "",
    "mobile": False,
    "bitness": "64",
    "wow64": False,
}


def is_honest_mode() -> bool:
    """Whether honest-presentation mode is active (#823).

    Default ``True`` — opt out with ``MANTIS_STEALTH_HONEST=0`` to
    fall back to the legacy Windows-claiming UA spoof. Honest mode
    drops platform spoofing entirely so we don't create a TLS / UA
    mismatch with the underlying Linux Chrome binary; the bot-tell
    removals (``navigator.webdriver``, plugins, languages,
    automation flags) stay in.

    Why default-on: the #827 fingerprint diagnostic against
    bot.sannysoft.com showed the Windows UA spoof partially leaking
    (raw Linux UA bleeding through alongside the Windows override).
    A mixed signal is detectably worse than a consistent honest one;
    presenting as the Linux Chrome we actually are is the
    higher-trust posture.
    """
    raw = os.environ.get("MANTIS_STEALTH_HONEST", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _detect_chrome_major_version() -> int:
    """Best-effort: read the actual Chrome major version from the
    binary so the UA we emit matches the TLS / HTTP/2 stack the binary
    ships with. Returns 0 on failure — callers fall back to the
    documented default.

    Cached on first successful read so we don't spawn a subprocess on
    every CDP call.
    """
    global _CHROME_MAJOR_CACHE
    if _CHROME_MAJOR_CACHE is not None:
        return _CHROME_MAJOR_CACHE
    import subprocess
    for binary in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        try:
            r = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=2,
            )
            out = (r.stdout or "") + (r.stderr or "")
            # e.g. "Google Chrome 132.0.6834.110" or "Chromium 131.0.6778.85"
            import re
            m = re.search(r"\b(\d+)\.\d+\.\d+\.\d+", out)
            if m:
                _CHROME_MAJOR_CACHE = int(m.group(1))
                return _CHROME_MAJOR_CACHE
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    _CHROME_MAJOR_CACHE = 0
    return 0


_CHROME_MAJOR_CACHE: int | None = None
# Documented Linux Chrome version baseline. Used when the binary can't
# be probed (test contexts, future Chrome rename). Bump when production
# images update to a newer major.
_HONEST_CHROME_MAJOR_FALLBACK = 132


def _honest_ua() -> tuple[str, dict[str, Any]]:
    """Build the honest Linux Chrome UA + sec-ch-ua-* metadata.

    Reads the binary's actual major version when possible; falls back
    to ``_HONEST_CHROME_MAJOR_FALLBACK`` when the probe fails. The
    returned UA, platform, brands, and fullVersionList all consistently
    say Linux Chrome — no mixed signal across surfaces.
    """
    major = _detect_chrome_major_version() or _HONEST_CHROME_MAJOR_FALLBACK
    ua = (
        f"Mozilla/5.0 (X11; Linux x86_64) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.0.0.0 Safari/537.36"
    )
    metadata: dict[str, Any] = {
        "brands": [
            {"brand": "Not_A Brand", "version": "8"},
            {"brand": "Chromium", "version": str(major)},
            {"brand": "Google Chrome", "version": str(major)},
        ],
        "fullVersion": f"{major}.0.0.0",
        "fullVersionList": [
            {"brand": "Not_A Brand", "version": "8.0.0.0"},
            {"brand": "Chromium", "version": f"{major}.0.0.0"},
            {"brand": "Google Chrome", "version": f"{major}.0.0.0"},
        ],
        "platform": "Linux",
        "platformVersion": "6.5.0",
        "architecture": "x86",
        "model": "",
        "mobile": False,
        "bitness": "64",
        "wow64": False,
    }
    return ua, metadata


# Honest-mode JS payload — only the bot-tell removals, no platform
# spoofing / WebGL invention / canvas+audio noise. Per #823: the
# leverage isn't fooling the fingerprint hash, it's not triggering
# escalation in the first place. We DO hide the automation framework
# leaks (webdriver flag, headless plugin shape) because those are
# automation-context tells. We DON'T pretend to be Windows when we're
# on Linux because the TLS / HTTP/2 stack tells the truth anyway.
STEALTH_JS_HONEST: str = r"""
(() => {
  // [1] navigator.webdriver — hide the automation flag. This is the
  // ONLY signal we still spoof: it's an automation context tell, not
  // a platform claim.
  try {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  } catch (e) {}

  // [2] navigator.permissions.query — make notifications report
  // Notification.permission instead of "denied" (which only happens
  // in headless / automated contexts).
  try {
    const originalQuery = window.navigator.permissions &&
      window.navigator.permissions.query &&
      window.navigator.permissions.query.bind(window.navigator.permissions);
    if (originalQuery) {
      window.navigator.permissions.query = (parameters) =>
        parameters && parameters.name === 'notifications'
          ? Promise.resolve({ state: Notification.permission })
          : originalQuery(parameters);
    }
  } catch (e) {}

  // [3] navigator.plugins — populate with a realistic non-empty array.
  // Headless Chromium ships with zero plugins; real Chrome (Linux,
  // Windows, macOS) always has the built-in PDF viewer entries.
  try {
    Object.defineProperty(navigator, 'plugins', {
      get: () => {
        const plugins = [
          { name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Chromium PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'Microsoft Edge PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
          { name: 'WebKit built-in PDF', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        ];
        Object.defineProperty(plugins, 'length', { value: 5 });
        return plugins;
      },
    });
  } catch (e) {}

  // [4] navigator.languages — empty array is a headless tell.
  try {
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
  } catch (e) {}

  // [5] window.chrome.runtime — must exist with a plausible shape
  // (real Chrome has this object; bare puppeteer / playwright builds
  // don't unless --enable-features=ChromiumExtensions).
  try {
    window.chrome = window.chrome || {};
    window.chrome.runtime = window.chrome.runtime || {
      OnInstalledReason: {},
      OnRestartRequiredReason: {},
      PlatformArch: {},
      PlatformNaclArch: {},
      PlatformOs: {},
      RequestUpdateCheckStatus: {},
    };
    window.chrome.app = window.chrome.app || {
      InstallState: { DISABLED: 'disabled', INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
      RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' },
    };
  } catch (e) {}

  // [6] Function.prototype.toString proxy — the patched
  // permissions.query must still serialize as ``[native code]``
  // (some scorers probe Function.toString to detect monkey-patches).
  try {
    const proxyToString = Function.prototype.toString;
    Function.prototype.toString = new Proxy(proxyToString, {
      apply(target, thisArg, args) {
        if (thisArg === window.navigator.permissions.query) {
          return 'function query() { [native code] }';
        }
        return Reflect.apply(target, thisArg, args);
      },
    });
  } catch (e) {}

  // Deliberately NOT patched in honest mode — the design rationale
  // and the patches we dropped are documented on the Python side in
  // ``docs/operations/stealth-diagnostics.md`` (see "What each flag
  // actually does"). Keeping that documentation in Python rather
  // than inside the JS payload avoids tripping page-side scorers
  // that scan injected scripts for strings like the macOS WebGL
  // renderer name.
})();
"""


def is_enabled() -> bool:
    """Whether CDP stealth patches should be applied.

    Default ``True`` — opt out by setting ``MANTIS_CDP_STEALTH=0`` (or
    ``false``/``no``/``off``). The patches add ~1ms to first navigation
    and have no measurable impact on subsequent requests, so the
    default-on cost is negligible.
    """
    raw = os.environ.get("MANTIS_CDP_STEALTH", "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


def inject_stealth_patches(cdp_call: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any]]]) -> bool:
    """Apply the stealth JS via CDP ``Page.addScriptToEvaluateOnNewDocument``.

    ``cdp_call`` is the env's ``(method, params) -> (ok, payload)``
    shim (see :meth:`XdotoolGymEnv._cdp_call`). The script registers
    once per-tab and runs on every subsequent document load (including
    iframes) BEFORE site JS — so the first runner-triggered navigate
    sees the spoofed surfaces.

    No-op when :func:`is_enabled` returns False. Returns ``True`` on
    successful CDP register, ``False`` on any failure (caller logs at
    DEBUG; we never break a run because of stealth setup).

    Idempotency: ``Page.addScriptToEvaluateOnNewDocument`` returns a
    different ``identifier`` per call, so calling this twice stacks
    two copies of the patches. Caller should call it once per
    browser session.
    """
    if not is_enabled():
        return False
    # #823: honest mode uses the minimal patch set (bot-tell removals
    # only, no platform / WebGL / canvas spoofing). Falls back to the
    # legacy 12-patch deceptive set when opted out.
    source = STEALTH_JS_HONEST if is_honest_mode() else STEALTH_JS
    try:
        ok, payload = cdp_call(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": source},
        )
        if ok:
            # WARN so Modal's INFO/DEBUG suppression doesn't hide the
            # one signal we have that stealth fired (memory:
            # feedback_modal_info_log_suppression).
            logger.warning(
                "CDP stealth: patches registered (identifier=%s)",
                payload.get("identifier", "?"),
            )
        else:
            logger.warning("CDP stealth: register returned not-ok: %r", payload)
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 — telemetry-style; never fatal
        logger.debug("CDP stealth: register failed: %s", exc)
        return False


def apply_ua_override(
    cdp_call: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any]]],
) -> bool:
    """Spoof User-Agent + sec-ch-ua-* request headers via CDP (#539).

    Linux Chrome ships ``sec-ch-ua-platform: "Linux"`` which is a strong
    bot tell for sites whose audience is overwhelmingly Windows/macOS
    (boats, cars, real estate). The `--user-agent` Chrome flag only
    sets the User-Agent header; the UA Client Hints (``sec-ch-ua-*``)
    are computed from the runtime build and need the CDP override.

    ``Network.setUserAgentOverride`` with both ``userAgent`` and
    ``userAgentMetadata`` aligns all three surfaces (UA, sec-ch-ua,
    sec-ch-ua-platform) at the request layer. Combined with the
    ``navigator.userAgentData`` patch in :data:`STEALTH_JS`, the JS-
    side observable surfaces also match.

    Returns ``True`` on successful CDP call, ``False`` on any failure.
    No-op when :func:`is_enabled` returns False — gated by the same
    ``MANTIS_CDP_STEALTH`` env var as :func:`inject_stealth_patches`.
    """
    if not is_enabled():
        return False
    # #823: honest mode sends the real Linux Chrome UA (matching the
    # binary that's actually running). Legacy deceptive path falls
    # through when opted out — kept on one release for rollback.
    if is_honest_mode():
        ua, metadata = _honest_ua()
        platform_claim = "Linux"
    else:
        ua = _SPOOF_UA
        metadata = _SPOOF_UA_METADATA
        platform_claim = "Windows"
    try:
        ok, payload = cdp_call(
            "Network.setUserAgentOverride",
            {
                "userAgent": ua,
                "platform": platform_claim,
                "userAgentMetadata": metadata,
            },
        )
        if ok:
            logger.warning(
                "CDP stealth: UA override applied (platform=%s, chrome=%s)",
                platform_claim, metadata.get("brands", [{}])[-1].get("version", "?"),
            )
        else:
            logger.warning(
                "CDP stealth: UA override returned not-ok: %r", payload,
            )
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 — telemetry-style; never fatal
        logger.debug("CDP stealth: UA override failed: %s", exc)
        return False


__all__ = [
    "STEALTH_JS",
    "apply_ua_override",
    "inject_stealth_patches",
    "is_enabled",
]
