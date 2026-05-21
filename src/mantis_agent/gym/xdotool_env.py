"""XdotoolGymEnv — pure screen-level CUA environment using Xvfb + xdotool.

Strictly screenshot + input events. No CDP, no DOM, no JS injection.

Architecture:
    Xvfb (virtual display :99)
      └── Chrome (regular process, zero automation hooks)
            ↕ X11 events
    xdotool → mousemove, click, type, key
    mss/scrot → screenshot.png

    Brain sees screenshot → outputs action → xdotool executes

The model handles everything visually: cookies, popups, navigation.
The env just executes actions and returns screenshots.

Requirements:
    apt-get install xvfb xdotool scrot chromium-browser
    pip install mss pillow
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .checkpoint import BrowserState

from PIL import Image

from ..actions import Action, ActionType
from . import adaptive_settle
from .base import GymEnvironment, GymObservation, GymResult

logger = logging.getLogger(__name__)

# #320: cap on SCROLL.amount expressed in wheel notches. Each notch costs
# one xdotool subprocess fork (~100 ms), so an unbounded loop on
# ``amount=350`` (a caller meant pixels) hangs the runner for ~40 s. 40 is
# already used as the upper bound in ``browser_state._wheel_summary``.
_MAX_SCROLL_NOTCHES = 40


def scale_brain_to_display(
    x_brain: int | float,
    y_brain: int | float,
    brain_size: tuple[int, int],
    display_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale (x, y) from brain-image pixel space to display pixel space.

    See docs/reference/coordinate-spaces.md for the full contract.
    Returns the rounded display-space integer (x, y).
    """
    bw, bh = brain_size
    dw, dh = display_size
    if bw <= 0 or bh <= 0:
        raise ValueError(f"brain_size must be positive: got {brain_size}")
    if dw <= 0 or dh <= 0:
        raise ValueError(f"display_size must be positive: got {display_size}")
    return round(x_brain * dw / bw), round(y_brain * dh / bh)


class XdotoolGymEnv(GymEnvironment):
    """Pure screen-level environment — Xvfb + xdotool + mss.

    No Playwright, no CDP, no DOM access. Just pixels and input events.
    Identical to a human using a mouse and keyboard.

    Args:
        start_url: URL to open in the browser on reset.
        viewport: Screen size as (width, height).
        browser: Browser command ("chromium-browser", "firefox", "google-chrome").
        display: X11 display number (e.g., ":99"). If None, starts Xvfb.
        settle_time: Seconds to wait after actions for page to update.
        human_speed: Add realistic delays between actions.
        proxy_server: HTTP proxy URL (e.g. "http://127.0.0.1:3128").
        profile_dir: Chrome user-data-dir for cookie/session persistence.
    """

    def __init__(
        self,
        start_url: str = "about:blank",
        viewport: tuple[int, int] = (1280, 720),
        browser: str = "chromium-browser",
        display: str | None = None,
        settle_time: float = 1.5,
        human_speed: bool = False,
        proxy_server: str = "",
        profile_dir: str = "/data/chrome-profile",
        save_screenshots: str = "",
        cdp_port: int = 9222,
        reuse_session: bool = False,
    ):
        self._start_url = start_url
        self._viewport = viewport
        self._browser_cmd = browser
        self._display = display
        self._settle_time = settle_time
        self._human_speed = human_speed
        self._proxy_server = proxy_server
        self._profile_dir = profile_dir
        self._save_screenshots = save_screenshots  # Dir to save screenshots for replay
        self._step_counter = 0
        # CDP read-only access — used by current_url to query Chrome's
        # navigation state directly instead of relying on the screenshot
        # extractor reading the address bar pixels (issue #89 §1).
        self._cdp_port = cdp_port

        # #311: when True, ``close()`` is a no-op so the Xvfb + Chrome
        # processes survive past one request. The runtime's container-
        # scoped cache (``runtime._chrome_env_cache``) sets this so a
        # second request on the same warm container reuses the same
        # browser instead of paying the ~10s cold-launch tax.
        # ``XdotoolGymEnv.shutdown`` exists for the cache to force-close
        # the underlying processes when the container is recycled.
        self._reuse_session = reuse_session

        self._xvfb_proc = None
        self._browser_proc = None
        self._env = {}

    # ── Xvfb + Browser ──────────��───────────────────────────────────

    def ensure_display_ready(self, *, timeout: float = 5.0) -> str:
        """Bring up the X display and return its name (e.g. ``":99"``).

        Idempotent: if Xvfb is already alive on the configured display
        (either because a sibling process started it or because a prior
        call wired it up), returns the display name immediately without
        spawning a duplicate Xvfb. Used by callers that need the X
        display *before* ``reset()`` runs — canonical case is
        :class:`mantis_agent.recorder.ScreenRecorder`, which spawns
        ``ffmpeg -f x11grab`` against the display before the agent loop
        opens a browser. Without this hook the recorder would race
        ``reset()``'s lazy Xvfb spawn and ffmpeg would fail with
        ``Cannot open display :99`` (the recurring integrator error).

        The browser is NOT launched — that stays in ``reset()``. This
        method only guarantees the display + the ``DISPLAY`` env var.
        """
        # Determine the canonical display name (without spawning).
        # ``:99`` matches ``_start_xvfb``'s default — staying in sync
        # so the probe + spawn agree on the same target.
        display = self._display or ":99"
        if not self._xdpyinfo_alive(display):
            # Display not up yet — let ``_start_xvfb`` boot it. The
            # method already short-circuits when ``self._display`` was
            # passed in at construction time, so this is safe even if
            # the caller wired up a custom display.
            display = self._start_xvfb()
        # Propagate the display name onto self._env so subprocesses we
        # spawn (Chrome, xdotool, ffmpeg-via-recorder) all agree on the
        # same target. ``reset()`` does the same wire-up before starting
        # the browser; doing it here too is safe and idempotent.
        if not self._env:
            self._env = {**os.environ, "DISPLAY": display}
        else:
            self._env["DISPLAY"] = display
        # Bounded wait so a slow Xvfb boot doesn't leave the caller
        # racing the same problem. ``xdpyinfo`` is the cheapest probe
        # that confirms the X server is accepting connections.
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if self._xdpyinfo_alive(display):
                return display
            time.sleep(0.15)
        # Don't raise — the recorder caller treats "no display" as a
        # benign failure (record_video=False semantics on this run).
        return display

    @staticmethod
    def _xdpyinfo_alive(display: str) -> bool:
        try:
            proc = subprocess.run(
                ["xdpyinfo", "-display", display],
                capture_output=True, timeout=3,
            )
            return proc.returncode == 0
        except Exception:  # noqa: BLE001 — best-effort liveness probe
            return False

    def _start_xvfb(self) -> str:
        """Start Xvfb virtual display if not already running."""
        if self._display:
            return self._display

        display = ":99"
        cmd = [
            "Xvfb", display,
            "-screen", "0", f"{self._viewport[0]}x{self._viewport[1]}x24",
            "-ac", "-nolisten", "tcp",
        ]
        self._xvfb_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        logger.info(f"Xvfb started on {display}")
        return display

    def _start_browser(self, url: str) -> None:
        """Launch browser with persistent profile, no CDP."""
        os.makedirs(self._profile_dir, exist_ok=True)

        # Clean session recovery files to prevent "Restore pages?" dialog.
        # Preserves: Cookies, Local Storage, Login Data, Preferences.
        default_dir = os.path.join(self._profile_dir, "Default")
        if os.path.isdir(default_dir):
            for stale in ["Current Session", "Current Tabs",
                          "Last Session", "Last Tabs", "Session Storage"]:
                path = os.path.join(default_dir, stale)
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            import shutil
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            os.remove(path)
                    except OSError:
                        pass

        # Remove lock files from prior unclean shutdown
        for lock in ["SingletonLock", "SingletonSocket", "SingletonCookie"]:
            path = os.path.join(self._profile_dir, lock)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        # Disable Chrome's "Save password?" / autofill prompts via the
        # profile Preferences file. The CLI flags
        # ``--disable-save-password-bubble`` and
        # ``--disable-features=PasswordManagerOnboarding,...`` don't kill
        # the bubble in current Chromium — the canonical kill switch is
        # ``credentials_enable_service: false`` plus
        # ``profile.password_manager_enabled: false`` in
        # ``Default/Preferences``. Symptom: a "Save password?" overlay
        # rendered after the login submit on staff-crm intercepted clicks
        # on the Lead Management table (run 13).
        prefs_dir = os.path.join(self._profile_dir, "Default")
        os.makedirs(prefs_dir, exist_ok=True)
        prefs_path = os.path.join(prefs_dir, "Preferences")
        prefs: dict[str, Any] = {}
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path) as f:
                    prefs = json.load(f)
            except (OSError, json.JSONDecodeError):
                prefs = {}
        prefs["credentials_enable_service"] = False
        prefs["credentials_enable_autosignin"] = False
        profile_section = prefs.setdefault("profile", {})
        profile_section["password_manager_enabled"] = False
        try:
            with open(prefs_path, "w") as f:
                json.dump(prefs, f)
        except OSError as exc:
            logger.debug(
                "could not write Preferences to suppress password bubble: %s",
                exc,
            )

        cmd = [
            self._browser_cmd,
            "--no-sandbox",
            "--test-type",  # Suppress --no-sandbox warning bar
            "--disable-gpu",
            "--no-first-run",
            "--disable-default-apps",
            "--disable-infobars",
            "--disable-notifications",
            "--disable-popup-blocking",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--noerrdialogs",
            "--disable-features=InfiniteSessionRestore",
            "--disable-blink-features=AutomationControlled",  # Hide navigator.webdriver
            "--disable-dev-shm-usage",
            # Suppress Chrome's "Save password?" / autofill prompts that
            # block the CUA after login (run 033 hit this on staff-crm).
            "--password-store=basic",
            "--disable-save-password-bubble",
            "--disable-features=PasswordManagerOnboarding,AutofillEnableAccountWalletStorage,AutofillServerCommunication",
            # CDP for current_url + Input.insertText (run 032 found Chrome
            # rejects WS connections with 403 Forbidden without explicit
            # --remote-allow-origins). Bound to localhost only — wildcard
            # origin is safe because the port itself is unreachable from
            # outside the container.
            f"--remote-debugging-port={self._cdp_port}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-allow-origins=*",
            f"--window-size={self._viewport[0]},{self._viewport[1]}",
            "--start-maximized",
            f"--user-data-dir={self._profile_dir}",
            # #stealth-parity (parity-reference diff): force WebGL ON
            # via SwiftShader software renderer. On a virtualized
            # GPU (Modal A100 isn't exposed to Chrome) WebGL may be
            # disabled — that fingerprints as "WebGL absent" which
            # CF / Turnstile flag as a server/automation context.
            # Forcing SwiftShader makes WebGL always work + render
            # consistently, then the loaded extension below spoofs
            # the vendor/renderer strings to a realistic Intel UHD
            # value.
            "--use-gl=angle",
            "--use-angle=swiftshader-webgl",
            "--enable-unsafe-swiftshader",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
        ]
        # #stealth-parity: load the WebGL spoof Chrome extension when
        # present (deployed via the Modal image's add_local_dir).
        # Extension content scripts run at document_start in MAIN
        # world across <all_urls> + all frames — hooks WebGLRendering
        # Context at the C++ binding level. More thorough than our
        # JS-injected getParameter patch which is page-context only
        # and can be probed away.
        _stealth_ext_dir = "/opt/chrome-extensions/webgl-spoof"
        if os.path.isdir(_stealth_ext_dir):
            cmd.append(f"--load-extension={_stealth_ext_dir}")
        if self._proxy_server:
            cmd.append(f"--proxy-server={self._proxy_server}")
        cmd.append(url)

        self._browser_proc = subprocess.Popen(
            cmd, env=self._env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        logger.info(f"Browser started: {self._browser_cmd} → {url}")

        # #539: register CDP stealth patches BEFORE any runner-triggered
        # navigate. The browser opens to ``url`` (default about:blank)
        # which isn't a real page — the first navigate from the runner
        # is the first document that picks up the patches via
        # ``Page.addScriptToEvaluateOnNewDocument``. Gated by
        # ``MANTIS_CDP_STEALTH`` env var (default-on); no-op when
        # disabled or when the CDP call fails (stealth setup never
        # blocks a run).
        try:
            from .cdp_stealth import apply_ua_override, inject_stealth_patches
            inject_stealth_patches(self._cdp_call)
            # #539 follow-up: also spoof UA + sec-ch-ua-platform via CDP
            # so the request-layer headers match the JS-side patches.
            # Linux sec-ch-ua-platform is a strong bot tell on
            # consumer-facing sites; Windows + Chrome 132 blends in.
            apply_ua_override(self._cdp_call)
        except Exception as exc:  # noqa: BLE001 — never fatal
            logger.debug("CDP stealth inject/UA-override raised at startup: %s", exc)

    # ── Screenshot ──────────────────────────────────────────────────

    def screenshot(self) -> Image.Image:
        """Public: capture current screenshot as PIL Image."""
        # Epic #362: credit screenshot capture to ``perceive``.
        _t0 = time.monotonic()
        try:
            return self._screenshot()
        finally:
            try:
                from .time_meter import record_to_current
                record_to_current("perceive", time.monotonic() - _t0)
            except Exception:
                pass

    def _screenshot(self) -> Image.Image:
        """Capture screenshot via mss (fast) or scrot (fallback)."""
        try:
            import mss
            with mss.mss(display=self._env.get("DISPLAY", ":99")) as sct:
                monitor = sct.monitors[0]
                img = sct.grab(monitor)
                return Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        except Exception:
            pass

        try:
            tmp = "/tmp/mantis_screenshot.png"
            subprocess.run(
                ["scrot", "-o", tmp],
                env=self._env, capture_output=True, timeout=5,
            )
            return Image.open(tmp)
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return Image.new("RGB", self._viewport, "gray")

    # ── xdotool ───��─────────────────────────────────────────────��───

    def _xdotool(self, *args: str) -> None:
        """Run xdotool command."""
        subprocess.run(
            ["xdotool"] + list(args),
            env=self._env, capture_output=True, timeout=5,
        )

    def _cdp_call(
        self, method: str, params: dict[str, Any] | None = None,
        *, timeout: float = 3.0,
    ) -> tuple[bool, Any]:
        """One-shot CDP request against the active page.

        Resolves the active page's WebSocket debugger URL via the
        ``/json/list`` REST endpoint, opens a synchronous WebSocket
        connection, sends a single ``{id, method, params}`` request, and
        polls for the matching response.

        Returns ``(ok, payload)`` where ``payload`` is the parsed
        ``result`` dict on success and an empty dict on failure. Caller
        sites decide how to interpret a missing key — :meth:`cdp_evaluate`
        unwraps ``result.value``, :meth:`cdp_click_at_point` only cares
        about ``ok``.

        Failure paths (CDP unreachable, no eligible page, ws import
        missing, timeout) all return ``(False, {})`` so callers fall
        back to the legacy xdotool path. Errors are logged at WARNING
        level so a flaky CDP doesn't silently mute the entire SoM
        routing path.
        """
        try:
            import json as _json
            import urllib.request
            try:
                import websocket  # websocket-client package
            except ImportError:
                return False, {}
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
            ws_url: str | None = None
            for tab in tabs:
                if tab.get("type") != "page":
                    continue
                url = str(tab.get("url") or "")
                if not url or url.startswith("chrome://") or url.startswith("about:"):
                    continue
                ws_url = tab.get("webSocketDebuggerUrl")
                if ws_url:
                    break
            if not ws_url:
                return False, {}
            ws = websocket.create_connection(ws_url, timeout=timeout)
            try:
                req_id = int(time.time() * 1000) % 1_000_000
                ws.send(_json.dumps({
                    "id": req_id,
                    "method": method,
                    "params": params or {},
                }))
                ws.settimeout(timeout)
                for _ in range(16):
                    raw = ws.recv()
                    if not raw:
                        continue
                    decoded = _json.loads(raw)
                    if decoded.get("id") != req_id:
                        # Drop background CDP events while waiting for
                        # our response id (eg lifecycle, frame nav).
                        continue
                    if decoded.get("error"):
                        return False, decoded.get("error") or {}
                    return True, decoded.get("result") or {}
                return False, {}
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("CDP %s failed: %s", method, exc)
            return False, {}

    def _cdp_insert_text(self, text: str) -> bool:
        """Type text via Chrome DevTools Protocol's Input.insertText.

        Returns True on success, False on any failure (caller falls back).
        Uses the same `--remote-debugging-port` Chrome was launched with
        (`current_url` already uses the same port). Dispatches a native
        input event, which React's controlled-input onChange registers
        cleanly — same outcome as Playwright's ``el.type()``.

        This is the preferred typing path because xdotool's keypress
        events don't reliably round-trip through React's value binding
        (run 020-031, b3b4364 commit history). Click/scroll/screenshot
        stay xdotool; only the type-text execution moves to CDP.
        """
        ok, _ = self._cdp_call("Input.insertText", {"text": text})
        return ok

    def cdp_evaluate(self, expression: str) -> Any:
        """Run a JS expression via CDP ``Runtime.evaluate``.

        Returns the unwrapped ``result.value`` (any JSON-serializable
        type), or ``None`` on any failure (CDP unreachable, JS threw,
        result not serializable). Used by :class:`PageDiscovery` to
        scan the DOM and by :meth:`cdp_click_at_point` to dispatch a
        synthetic click on the topmost element at a screen point —
        both are #300 routing primitives that need to work in the
        production xdotool path where there's no Playwright ``page``.
        """
        ok, payload = self._cdp_call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": False,
            },
        )
        if not ok:
            return None
        # ``Runtime.evaluate`` returns ``{result: {type, value}}``;
        # ``value`` is missing for ``undefined`` results — treat that
        # as None too rather than KeyError-ing on the caller.
        return (payload.get("result") or {}).get("value")

    def cdp_click_at_point(self, x: int, y: int) -> bool:
        """SoM-anchored click: find the element at SCREEN (x, y) and call
        ``el.click()`` via Runtime.evaluate.

        Why this isn't just xdotool: xdotool sends X-level mouse events
        that Chrome maps to synthetic ``mousedown`` / ``mouseup`` /
        ``click`` DOM events. On SPA rows whose handler is bound via
        ``onPointerDown`` (or whose ``mousedown`` calls
        ``stopPropagation``) the actual click handler never fires — the
        well-known #88 row-click failure. Dispatching ``el.click()``
        directly invokes the element's click listeners *and* the
        framework's synthetic event chain (React onClick, etc.), so the
        click reaches the handler.

        Returns ``True`` iff an element was found at (x, y) and the JS
        click was dispatched successfully. Returns ``False`` if CDP is
        unreachable, no element exists at the point, or the JS threw.
        Caller is expected to fall back to xdotool on ``False``.

        Coordinate system note (#413): caller passes screen-space
        (x, y) — the same numbers xdotool would click. But
        ``document.elementFromPoint`` uses CSS-viewport coords (origin
        = top-left of the page area, BELOW the browser tabs + URL
        bar). Subtract ``window.outerHeight - window.innerHeight``
        so the screen-y is translated into viewport-y before the
        elementFromPoint call. Without this, the SoM ``el.click()``
        fires on whatever element sits ~85 px below the visual click
        target — the symptom that motivated the tag-guard fix in #413.

        Diagnostic mode (chromeH-introspection): when the env was
        constructed with ``MANTIS_SOM_DIAGNOSTIC=1`` (or the class
        attribute ``som_diagnostic`` is set), the JS payload returns a
        rich dict (outerHeight, innerHeight, chromeH, viewport-coord
        element tag, screen-coord element tag, click ok/fail) and the
        method emits a one-line WARNING log per call so the run's log
        capture surfaces what's actually happening. Useful when the
        translation is silently wrong in headless / Xvfb modes where
        ``outerHeight`` reports 0.
        """
        # The JS payload returns a dict (always — diagnostic info is
        # cheap), and the caller treats the ``ok`` field as the bool
        # return contract. Logging the diagnostic line surfaces it
        # under Modal's INFO-suppressed capture so it's visible without
        # an env-var toggle dance.
        js = (
            "(() => {"
            "const oh = window.outerHeight;"
            "const ih = window.innerHeight;"
            "const chromeH = Math.max(0, oh - ih);"
            f"const sx = {int(x)};"
            f"const sy = {int(y)};"
            "const vx = sx;"
            "const vy = sy - chromeH;"
            "const elv = document.elementFromPoint(vx, vy);"
            "const els = document.elementFromPoint(sx, sy);"
            "const tag = e => e ? ((e.tagName || '') + (e.id ? '#' + e.id : '') + "
            "(e.className && typeof e.className === 'string' ? '.' + e.className.split(' ').slice(0,2).join('.') : '')) : null;"
            "const text = e => e ? (e.innerText || e.textContent || '').trim().slice(0, 40) : '';"
            "let ok = false;"
            "if (elv) { try { elv.click(); ok = true; } catch (e) { ok = false; } }"
            "return {"
            "ok: ok,"
            "outerHeight: oh, innerHeight: ih, chromeH: chromeH,"
            "vx: vx, vy: vy,"
            "elv_tag: tag(elv), elv_text: text(elv),"
            "els_tag: tag(els), els_text: text(els)"
            "};"
            "})()"
        )
        try:
            result = self.cdp_evaluate(js)
        except Exception as exc:  # noqa: BLE001 — never break runs
            logger.debug("cdp_click_at_point eval raised: %s", exc)
            return False
        if not isinstance(result, dict):
            return bool(result)
        # Stash the SoM diagnostic on the env so the retry-context
        # recorder can include what was at the click point in the
        # next brain prompt. PR-H pattern (Option 1): when a click
        # fails as no_state_change, the brain's retry sees the
        # ``elv_tag``/``elv_text`` at its prior coord — a post-action
        # observation, not DOM-target derivation, so this stays
        # CUA-compliant. Cleared on next ``cdp_click_at_point``;
        # callers that want a snapshot read it BEFORE the next click.
        self._last_som_diag = {
            "x": int(x),
            "y": int(y),
            "elv_tag": str(result.get("elv_tag") or ""),
            "elv_text": str(result.get("elv_text") or ""),
            "els_tag": str(result.get("els_tag") or ""),
            "els_text": str(result.get("els_text") or ""),
            "ok": bool(result.get("ok")),
        }
        # One-line WARNING log so the SoM click's coordinate translation
        # is visible in Modal's INFO-suppressed capture without an
        # explicit env-var toggle. Once the translation is verified
        # against expectations across deploy targets, this can drop
        # to DEBUG.
        same_element = result.get("elv_tag") == result.get("els_tag")
        logger.warning(
            "  [som-click] x=%d y=%d → chromeH=%s (oh=%s ih=%s) viewport=(%s,%s) "
            "elv=%s elv_text=%r els=%s els_text=%r same=%s ok=%s",
            int(x), int(y),
            result.get("chromeH"), result.get("outerHeight"), result.get("innerHeight"),
            result.get("vx"), result.get("vy"),
            result.get("elv_tag"), result.get("elv_text"),
            result.get("els_tag"), result.get("els_text"),
            same_element, result.get("ok"),
        )
        # Also emit into the structured reasoning trace when a runner
        # back-reference is available. The env doesn't carry a direct
        # runner ref by design; callers that want to attribute SoM
        # clicks to a specific runner pass the runner via the public
        # ``record_som_click`` helper below. We skip the trace here
        # because we have no runner — see ``record_som_click`` in
        # :mod:`reasoning_trace` if/when callers wire it.
        return bool(result.get("ok"))

    def cdp_click_via_pointer(self, x: int, y: int) -> bool:
        """Dispatch a real-pointer mouse-event chain at SCREEN (x, y)
        via CDP ``Input.dispatchMouseEvent`` — audit batch follow-up
        to the ``el.click()`` ok-but-no-state-change case.

        Why this exists: ``cdp_click_at_point`` dispatches a SYNTHETIC
        click via ``Runtime.evaluate("el.click()")``. The DOM event
        emitted has ``isTrusted=false``. Some SPA frameworks (most
        notably React Router under certain navigation guards) gate
        on ``isTrusted=true`` and silently reject untrusted clicks
        without raising — exactly the ``ok=True, but page didn't
        navigate`` pattern that surfaced on staff-crm's sidebar
        anchors. ``Input.dispatchMouseEvent`` emits events at the
        protocol layer; they're ``isTrusted=true`` from the page's
        perspective, indistinguishable from a real OS mouse click.

        Sequence: mouseMoved → mousePressed → mouseReleased at the
        same viewport coords (chromeH-adjusted from screen coords,
        same translation ``cdp_click_at_point`` uses).

        Returns ``True`` when all three events dispatched without
        protocol error. Returns ``False`` on CDP unreachable or any
        dispatch failure; caller falls back to xdotool / demote.
        """
        chrome_h = self._chrome_offset_px()
        vx = int(x)
        vy = int(y) - chrome_h
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            params: dict[str, Any] = {
                "type": kind, "x": vx, "y": vy,
                "button": "left",
                "buttons": 1 if kind == "mousePressed" else 0,
            }
            if kind in ("mousePressed", "mouseReleased"):
                params["clickCount"] = 1
            ok, _ = self._cdp_call("Input.dispatchMouseEvent", params)
            if not ok:
                logger.debug(
                    "cdp_click_via_pointer: %s dispatch failed", kind,
                )
                return False
        return True

    def _chrome_offset_px(self) -> int:
        """JS-eval the chrome offset (``outerHeight - innerHeight``).

        Used by ``cdp_click_via_pointer`` so screen-y → viewport-y for
        ``Input.dispatchMouseEvent`` matches the translation
        ``elementFromPoint`` uses in ``cdp_click_at_point``. Returns 0
        when the eval can't run (no CDP, unusual headless mode).
        """
        try:
            result = self.cdp_evaluate(
                "Math.max(0, window.outerHeight - window.innerHeight)"
            )
        except Exception:  # noqa: BLE001
            return 0
        try:
            return int(result) if result is not None else 0
        except (TypeError, ValueError):
            return 0

    def _xdotool_type(self, text: str) -> None:
        """Type text via clipboard-paste (preferred) or xdotool fallback.

        Diagnosed empirically across runs 020-029 + the claude-in-chrome
        MCP comparison: xdotool's ``type`` emits X-level keypress events
        that Chrome translates to JS ``KeyboardEvent``s, but React's
        controlled inputs frequently fail to flush those into component
        state — the typed text shows visually in the field but the form's
        internal state stays empty, so submit produces ``AUTH_FAIL_001``.

        Commit b3b4364 (Apr 2026) documented this exact issue on the
        Playwright path and fixed it via ``el.type()``. When xdotool
        replaced Playwright for "pure screen-level CUA", the React
        compatibility was lost.

        Clipboard-paste sidesteps the issue: xclip writes the text to
        the X clipboard and xdotool sends ``ctrl+v``. The browser fires
        a proper ``paste`` event which React registers, populating state
        correctly. Falls back to xdotool ``type`` for environments
        without xclip or when human-speed mode is requested.
        """
        if not text:
            return

        # Preferred path: CDP Input.insertText fires a synthesized input
        # event that React's onChange picks up reliably — same mechanism
        # MCP form_input / Playwright el.type() use. Falls through to
        # clipboard-paste then xdotool type if CDP is unreachable.
        if (
            not self._human_speed
            and os.environ.get("MANTIS_DISABLE_CDP_TYPE", "") != "1"
        ):
            if self._cdp_insert_text(text):
                return
            logger.info("CDP insertText unavailable; trying clipboard paste")

        # Prefer clipboard-paste for React/Vue/framework compatibility.
        # Disabled by setting MANTIS_DISABLE_PASTE_TYPE=1 or in human-speed mode
        # (which intentionally simulates per-keystroke typing).
        use_paste = (
            not self._human_speed
            and os.environ.get("MANTIS_DISABLE_PASTE_TYPE", "") != "1"
        )
        if use_paste:
            try:
                # xclip stays alive as the X-selection owner until a paste
                # reads from it — using subprocess.run waits for that exit
                # and times out (run 030). Popen + close stdin lets xclip
                # background itself; -loops 1 makes it exit after one read.
                xclip_proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard", "-loops", "1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env,
                )
                xclip_proc.stdin.write(text.encode())
                xclip_proc.stdin.close()
                # Tiny pause so xclip is registered as selection owner
                # before xdotool requests the paste.
                time.sleep(0.05)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    env=self._env, timeout=5, check=True,
                    capture_output=True,
                )
                # xclip should self-exit after the paste consumed the
                # selection (-loops 1). Reap it; if it overstays, kill.
                try:
                    xclip_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    xclip_proc.terminate()
                    try:
                        xclip_proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        xclip_proc.kill()
                return
            except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "xclip paste failed (%s), falling back to xdotool type", exc,
                )

        # Fallback: xdotool type with configurable delay. The default 60ms
        # is what landed when paste was unavailable (run 027/029 era).
        delay_ms = int(os.environ.get("MANTIS_TYPE_DELAY_MS", "60"))
        if self._human_speed:
            for char in text:
                subprocess.run(
                    ["xdotool", "type", "--delay", str(random.randint(30, 120)), char],
                    env=self._env, capture_output=True, timeout=5,
                )
        else:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), text],
                env=self._env, capture_output=True, timeout=20,
            )

    def _navigate_running_browser(self, url: str) -> None:
        """Navigate the already-running browser to ``url``.

        Prefers CDP ``Page.navigate`` (programmatic navigation through
        Chrome's browser process). Falls back to xclip-paste then
        xdotool-type into the omnibox if CDP is unreachable.

        Why CDP first: the legacy path here was
        ``Ctrl+L`` + ``Ctrl+A`` + ``_xdotool_type(url)`` + ``Enter``,
        which silently no-op'd whenever CDP's ``Input.insertText`` was
        available. ``Input.insertText`` operates on the renderer
        (the web page's focused DOM element); the omnibox lives in
        Chrome's UI process and never receives the text. ``Ctrl+A``
        then ``Enter`` would re-navigate to whatever was already in
        the omnibox — typically the current page's URL — producing
        a navigate that "looked successful" (HTTP 200, page loaded)
        but landed on the wrong URL. Surfaced in staff-crm-long
        verification run 1eb0b0c7 (2026-05-17).

        ``Page.navigate`` is the programmatic-navigation primitive
        Chrome exposes; it preserves cookies, session storage,
        history, and back-forward state same as a user-typed URL.
        """
        logger.info(f"Navigating to {url[:80]} (via CDP Page.navigate)")
        ok, _ = self._cdp_call("Page.navigate", {"url": url})
        if ok:
            time.sleep(self._settle_time + 2)
            return

        # Fallback: omnibox typing. Bypass CDP path inside _xdotool_type
        # because we KNOW it doesn't reach Chrome's UI chrome. Use xclip
        # paste (which honors X focus, including the omnibox) or, if
        # xclip is unavailable, raw xdotool type.
        logger.warning(
            "CDP Page.navigate unavailable; falling back to omnibox typing"
        )
        self._xdotool("key", "ctrl+l")
        time.sleep(0.3)
        self._xdotool("key", "ctrl+a")
        time.sleep(0.2)
        self._type_into_omnibox(url)
        time.sleep(0.3)
        self._xdotool("key", "Return")
        time.sleep(self._settle_time + 2)

    def _type_into_omnibox(self, url: str) -> None:
        """Type ``url`` into the currently-focused omnibox via xclip
        or xdotool.

        Skips the CDP ``Input.insertText`` path that ``_xdotool_type``
        prefers: that path targets the renderer's page DOM, not Chrome's
        UI. The omnibox has X focus (set by the caller's ``Ctrl+L``),
        so a clipboard paste or keystroke chain reaches it correctly.
        """
        if not url:
            return

        # xclip + Ctrl+V — fastest, no per-character timing.
        if os.environ.get("MANTIS_DISABLE_PASTE_TYPE", "") != "1":
            try:
                xclip_proc = subprocess.Popen(
                    ["xclip", "-selection", "clipboard", "-loops", "1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env,
                )
                xclip_proc.stdin.write(url.encode())
                xclip_proc.stdin.close()
                time.sleep(0.05)
                subprocess.run(
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    env=self._env, timeout=5, check=True,
                    capture_output=True,
                )
                try:
                    xclip_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    xclip_proc.terminate()
                return
            except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "xclip omnibox paste failed (%s); using xdotool type", exc,
                )

        # Last-resort: xdotool type. Slower (per-character) but works
        # without xclip. Same delay knob as the legacy code path.
        delay_ms = int(os.environ.get("MANTIS_TYPE_DELAY_MS", "60"))
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), url],
            env=self._env, capture_output=True, timeout=20,
        )

    # ── GymEnvironment interface ─────────────────────────────────────

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        """Start Xvfb + browser, navigate to URL."""
        url = kwargs.get("start_url", "")  # Only navigate if explicitly passed

        # Browser already running
        if self._browser_proc and self._browser_proc.poll() is None:
            if url and url != "about:blank":
                self._navigate_running_browser(url)
            else:
                # No URL — just capture current page state (for sub-plan micro-steps)
                logger.info("Reusing browser (no navigation)")
            return self._capture()

        # Fresh start
        if self._browser_proc:
            self.close()

        display = self._start_xvfb()
        self._env = {**os.environ, "DISPLAY": display}

        self._start_browser(url or self._start_url)
        time.sleep(self._settle_time + 2)

        return self._capture()

    def step(self, action: Action) -> GymResult:
        """Execute action via xdotool and return screenshot."""
        if self._human_speed:
            if action.action_type == ActionType.CLICK:
                time.sleep(random.uniform(0.3, 1.0))
            elif action.action_type == ActionType.TYPE:
                time.sleep(random.uniform(0.5, 1.5))

        # Epic #362: credit the actual action dispatch to ``act`` —
        # exclusively the time inside _execute_action, not the
        # human-speed pre-sleep above (which lands in ``overhead``).
        _act_t0 = time.monotonic()
        try:
            self._execute_action(action)
        finally:
            try:
                from .time_meter import record_to_current
                record_to_current("act", time.monotonic() - _act_t0)
            except Exception:
                pass

        # Post-action settle (#294).
        #
        # Frame-stability gate replaces the fixed sleep: poll the framebuffer
        # every 100 ms, return as soon as two consecutive ``phash_64`` reads
        # agree (page has stopped repainting), cap at the legacy
        # ``settle_time`` budget. On a static page this typically lands in
        # ~200-300 ms; on a network-heavy submit it still respects the cap.
        #
        # MANTIS_ADAPTIVE_SETTLE=disabled falls back to the legacy fixed
        # sleep so an A/B comparison doesn't need a redeploy.
        if action.action_type not in (ActionType.WAIT, ActionType.DONE):
            settle = self._settle_time
            if self._human_speed:
                settle += random.uniform(0.5, 2.0)
            if adaptive_settle.is_enabled():
                adaptive_settle.wait_until_stable(
                    self._screenshot,
                    max_seconds=settle,
                    poll_interval=0.1,
                )
            else:
                time.sleep(settle)

        obs = self._capture()
        return GymResult(observation=obs, reward=0.0, done=False, info={})

    def close(self) -> None:
        """Kill browser and Xvfb.

        When ``reuse_session=True`` (set by the container-scoped env cache
        in #311) this is a no-op so the processes survive past one
        request. Call :meth:`shutdown` from the cache to force-close.
        """
        if self._reuse_session:
            return
        self.shutdown()

    def shutdown(self) -> None:
        """Force-close browser and Xvfb regardless of ``reuse_session``.

        The container-scoped cache calls this at recycle time so reused
        processes don't leak across container lifetimes.
        """
        if self._browser_proc:
            self._browser_proc.terminate()
            try:
                self._browser_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._browser_proc.kill()
            self._browser_proc = None

        if self._xvfb_proc:
            self._xvfb_proc.terminate()
            self._xvfb_proc = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._viewport

    @property
    def current_url(self) -> str:
        """Read the active tab's URL via Chrome DevTools Protocol.

        Issue #89 §1: the runner used to read the URL from the address-bar
        pixels through ClaudeExtractor.extract — fragile when the page is
        mid-render or the address bar is offscreen, surfacing as the
        well-known ``(url=)`` empty-string in click-verify logs.

        CDP's ``GET /json/list`` returns every Chrome tab; the active page
        is the first entry of type ``page`` whose ``url`` is non-empty.
        Bound to 127.0.0.1 only — never reachable from outside the container.

        Returns ``""`` if CDP is unreachable so the runner can fall back to
        screenshot extraction (preserves backward compatibility).
        """
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
        except Exception:
            return ""
        for tab in tabs:
            if tab.get("type") == "page" and tab.get("url"):
                url = str(tab["url"])
                # Filter chrome:// internal URLs — not what callers want.
                if url.startswith("chrome://") or url.startswith("about:"):
                    continue
                return url
        return ""

    def capture_browser_state(self) -> "BrowserState":
        """Snapshot URL + scroll + viewport + form input for
        pause/resume (epic #358 Phase A + B). Two CDP
        ``Runtime.evaluate`` calls: one for the URL/scroll/viewport
        primitives, one for the form-field walk.

        Returns an all-empty :class:`BrowserState` on any failure (CDP
        unreachable, page mid-navigation, JS exception). The caller
        branches on ``bool(state.url)`` to decide whether to apply a
        restore — empty url means "no browser state to restore".
        """
        from . import form_capture_js
        from .checkpoint import BrowserState, FormFieldValue
        result = self.cdp_evaluate(
            "({"
            "url: (location && location.href) || '',"
            "scroll_x: Math.round(window.scrollX || 0),"
            "scroll_y: Math.round(window.scrollY || 0),"
            "viewport_w: window.innerWidth || 0,"
            "viewport_h: window.innerHeight || 0"
            "})"
        )
        if not isinstance(result, dict):
            return BrowserState(captured_at=time.time())
        # Phase B: form-field walk. Separate eval call so a JS error
        # in form_capture_js doesn't drop the URL/scroll capture too.
        form_state: dict[str, FormFieldValue] = {}
        try:
            entries = self.cdp_evaluate(form_capture_js.capture_js())
        except Exception as exc:  # noqa: BLE001 — observability
            logger.debug("form_state capture: cdp_evaluate raised: %s", exc)
            entries = None
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                selector = entry.get("selector")
                kind = entry.get("kind")
                if not isinstance(selector, str) or not isinstance(kind, str):
                    continue
                form_state[selector] = FormFieldValue(
                    kind=kind,
                    value=str(entry.get("value", "") or ""),
                    masked=bool(entry.get("masked", False)),
                )
        return BrowserState(
            url=str(result.get("url", "") or ""),
            scroll_x=int(result.get("scroll_x", 0) or 0),
            scroll_y=int(result.get("scroll_y", 0) or 0),
            viewport_w=int(result.get("viewport_w", 0) or 0),
            viewport_h=int(result.get("viewport_h", 0) or 0),
            captured_at=time.time(),
            form_state=form_state,
        )

    def restore_browser_state(self, state: "BrowserState") -> None:
        """Replay the captured browser state — navigate to ``url``,
        wait for load, scroll to (x, y). No-op when ``state.url`` is
        empty.

        Called from ``runner.resume`` after the runner-side state is
        rehydrated. Best-effort: any failure is logged at DEBUG; the
        resumed run continues at whatever URL the env actually has.
        """
        from .checkpoint import BrowserState as _BrowserState
        if not isinstance(state, _BrowserState) or not state.url:
            return
        try:
            self.reset(task="resume", start_url=state.url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("restore_browser_state: env.reset(%r) raised: %s", state.url, exc)
            return
        if state.scroll_x or state.scroll_y:
            try:
                self.cdp_evaluate(
                    f"window.scrollTo({int(state.scroll_x)}, {int(state.scroll_y)})"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("restore_browser_state: scroll restore failed: %s", exc)
        # Phase B: replay captured form fields after URL + scroll
        # restoration. Selectors that don't resolve on the resumed
        # page (DOM shifted) are silently skipped by the JS shim
        # itself — never fail a resume because the page changed.
        if state.form_state:
            try:
                self._replay_form_state(state.form_state)
            except Exception as exc:  # noqa: BLE001
                logger.debug("restore_browser_state: form_state replay failed: %s", exc)

    def _replay_form_state(self, form_state: dict) -> None:
        """Re-apply captured form fields via a single CDP eval.

        Skips ``masked`` entries (passwords) — the caller is expected
        to re-prompt the user. Logs the JS shim's per-call outcome
        ``{applied, skipped, missing}`` for audit.
        """
        from . import form_capture_js
        entries = [
            {
                "selector": selector,
                "kind": ffv.kind,
                "value": ffv.value,
                "masked": ffv.masked,
            }
            for selector, ffv in form_state.items()
        ]
        if not entries:
            return
        serialized = json.dumps(entries)
        outcome = self.cdp_evaluate(form_capture_js.replay_js(serialized))
        if isinstance(outcome, dict):
            logger.info(
                "form_state replay: applied=%s skipped=%s missing=%s",
                outcome.get("applied"), outcome.get("skipped"),
                outcome.get("missing"),
            )

    def has_session(self, name: str) -> bool:
        return self._browser_proc is not None and self._browser_proc.poll() is None

    def save_session(self, name: str) -> None:
        pass

    def load_session(self, name: str) -> None:
        pass

    # ��─ Internal ──���──────────────────────��───────────────────────────

    def _capture(self) -> GymObservation:
        """Take screenshot and return as observation.

        If save_screenshots is set, saves each screenshot for replay testing.
        """
        # Epic #362: credit screenshot capture to ``perceive``. Uses
        # ``_screenshot`` directly so adaptive_settle's internal frame
        # polling (which also calls ``_screenshot``) stays in the
        # ``settle`` bucket — no double-counting.
        _t0 = time.monotonic()
        screenshot = self._screenshot()
        try:
            if self._save_screenshots:
                from .replay_env import save_screenshot
                save_screenshot(screenshot, self._save_screenshots, self._step_counter)
                self._step_counter += 1
            return GymObservation(screenshot=screenshot, extras={})
        finally:
            try:
                from .time_meter import record_to_current
                record_to_current("perceive", time.monotonic() - _t0)
            except Exception:
                pass

    @staticmethod
    def _to_int(val) -> int:
        """Safely extract an integer from a possibly malformed value."""
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip()
        # Extract first number from strings like "143, 417]" or "0, y>\n0..."
        m = __import__("re").match(r'-?\d+', s)
        return int(m.group(0)) if m else 0

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to viewport bounds."""
        x, y = self._to_int(x), self._to_int(y)
        return max(0, min(x, self._viewport[0] - 1)), max(0, min(y, self._viewport[1] - 1))

    def _execute_action(self, action: Action) -> None:
        """Translate Mantis Action to xdotool commands."""
        match action.action_type:
            case ActionType.CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                button = action.params.get("button", "left")
                btn_num = {"left": "1", "middle": "2", "right": "3"}.get(button, "1")
                self._xdotool("mousemove", str(x), str(y))
                if self._human_speed:
                    time.sleep(random.uniform(0.05, 0.15))
                self._xdotool("click", btn_num)

            case ActionType.DOUBLE_CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                self._xdotool("mousemove", str(x), str(y))
                self._xdotool("click", "--repeat", "2", "1")

            case ActionType.TYPE:
                text = action.params.get("text") or action.params.get("content") or ""
                if not text:
                    logger.warning(f"type_text missing text: {action.params}")
                    return
                text = str(text)
                # #405 follow-up: Fara emits ``delete_existing_text`` and
                # ``press_enter`` as flags on the same ``type`` call rather
                # than as separate verbs. Both are translated to brain-side
                # TYPE params (``clear_first``, ``press_enter``) and honoured
                # here by the env. URL-shaped text keeps its own
                # auto-navigate path (which already does its own clear+Enter).
                clear_first = bool(action.params.get("clear_first", False))
                press_enter = bool(action.params.get("press_enter", False))
                if text.startswith("http://") or text.startswith("https://"):
                    self._xdotool("key", "ctrl+l")
                    time.sleep(0.5)
                    self._xdotool("key", "ctrl+a")
                    time.sleep(0.2)
                    self._xdotool_type(text)
                    time.sleep(0.3)
                    self._xdotool("key", "Return")
                else:
                    if clear_first:
                        self._xdotool("key", "ctrl+a")
                        time.sleep(0.1)
                        self._xdotool("key", "Delete")
                        time.sleep(0.1)
                    self._xdotool_type(text)
                    if press_enter:
                        time.sleep(0.1)
                        self._xdotool("key", "Return")

            case ActionType.KEY_PRESS:
                keys = action.params.get("keys") or action.params.get("key") or ""
                if not keys:
                    logger.warning(f"key_press missing keys: {action.params}")
                    return
                keys = str(keys)
                key_map = {
                    "enter": "Return", "tab": "Tab", "escape": "Escape",
                    "backspace": "BackSpace", "delete": "Delete",
                    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
                    "home": "Home", "end": "End",
                    "pageup": "Page_Up", "pagedown": "Page_Down",
                    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
                    "f11": "F11", "f12": "F12",
                    "space": "space",
                }
                parts = keys.split("+")
                mapped = [key_map.get(p.strip().lower(), p.strip()) for p in parts]
                self._xdotool("key", "+".join(mapped))

            case ActionType.SCROLL:
                direction = action.params.get("direction", "down")
                amount = int(action.params.get("amount", 3) or 0)
                # #320: SCROLL.amount is "wheel notches", not pixels.
                # Each notch is one xdotool subprocess (~100 ms). Bound the
                # iteration count so a caller passing a pixel value (e.g. 350)
                # can't lock the runner for ~40 s.
                amount = max(0, min(amount, _MAX_SCROLL_NOTCHES))
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                self._xdotool("mousemove", str(x), str(y))
                btn = "4" if direction == "up" else "5"
                for _ in range(amount):
                    self._xdotool("click", btn)
                    if self._human_speed:
                        time.sleep(random.uniform(0.05, 0.15))

            case ActionType.DRAG:
                sx, sy = action.params.get("start_x", 0), action.params.get("start_y", 0)
                ex, ey = action.params["end_x"], action.params["end_y"]
                sx, sy = self._clamp(sx, sy)
                ex, ey = self._clamp(ex, ey)
                self._xdotool("mousemove", str(sx), str(sy))
                self._xdotool("mousedown", "1")
                time.sleep(0.1)
                self._xdotool("mousemove", str(ex), str(ey))
                self._xdotool("mouseup", "1")

            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                time.sleep(min(seconds, 10.0))

            case ActionType.DONE:
                pass

            case ActionType.LAUNCH_APP:
                self._launch_app(action.params)

    # ── App launch (issue #72) ───────────────────────────────────────
    def _launch_app(self, params: dict[str, Any]) -> None:
        """Launch a desktop binary on the env's display.

        Errors are logged + swallowed so the runner sees the action as a
        no-op rather than crashing on bad params. The caller verifies launch
        success by checking the next screenshot (same contract as a click).
        """
        name = str(params.get("name") or "").strip()
        if not name:
            logger.warning("launch_app missing 'name': %s", params)
            return
        args = params.get("args") or []
        if not isinstance(args, (list, tuple)):
            logger.warning("launch_app args must be a list: %s", params)
            args = []
        extra_env = params.get("env") or {}
        if not isinstance(extra_env, dict):
            logger.warning("launch_app env must be a dict: %s", params)
            extra_env = {}

        # Compose env: env's DISPLAY + caller's overrides.
        proc_env = dict(self._env)
        proc_env.update({str(k): str(v) for k, v in extra_env.items()})

        try:
            subprocess.Popen(
                [name, *[str(a) for a in args]],
                env=proc_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("launch_app: started %s args=%s", name, list(args)[:6])
        except FileNotFoundError:
            logger.error("launch_app: binary not found on PATH: %s", name)
        except Exception as exc:  # noqa: BLE001 — surface as no-op
            logger.error("launch_app: failed to start %s: %s", name, exc)
