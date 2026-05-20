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

# Spoofed UA + UA-CH metadata to claim Windows 10 / Chrome 132.
# Applied via CDP ``Network.setUserAgentOverride`` so the sec-ch-ua-*
# request headers also align (the bare ``--user-agent`` Chrome flag
# only sets User-Agent, not the metadata headers — CF reads both).
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
    try:
        ok, payload = cdp_call(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": STEALTH_JS},
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
    try:
        ok, payload = cdp_call(
            "Network.setUserAgentOverride",
            {
                "userAgent": _SPOOF_UA,
                "platform": "Windows",
                "userAgentMetadata": _SPOOF_UA_METADATA,
            },
        )
        if ok:
            logger.warning(
                "CDP stealth: UA override applied (platform=Windows, chrome=132)",
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
