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
from .vendor_trap import is_browser_vendor_url

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
        extra_http_headers: dict[str, str] | None = None,
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

        # Optional request headers applied to EVERY browser request via a
        # persistent Network-enabled CDP session (see ``_start_browser``).
        # Default ``None`` = no-op, so production CF runs are unaffected.
        # The only caller that sets this is the Learning Allocator live
        # runner, which injects ``X-Daytona-Skip-Preview-Warning: true`` to
        # get past the Daytona preview proxy's interstitial on the sim env.
        self._extra_http_headers = extra_http_headers or None
        self._header_ws = None  # persistent CDP ws holding the header
        self._header_ws_thread = None

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
        #
        # #823 honest mode: skip the WebGL spoof entirely. Inventing
        # "Intel Iris OpenGL Engine" (a macOS string) on a Linux
        # binary running SwiftShader contradicts the TLS / HTTP/2
        # signal. Real Mesa / SwiftShader strings reported by Linux
        # Chrome are perfectly legitimate — millions of Linux users
        # browse the web every day. The diagnostic (#827) against
        # bot.sannysoft.com showed the spoofed strings were leaking
        # through anyway; honesty is the higher-trust posture.
        from .cdp_stealth import is_honest_mode as _is_honest_mode
        if not _is_honest_mode():
            _stealth_ext_dir = "/opt/chrome-extensions/webgl-spoof"
            if os.path.isdir(_stealth_ext_dir):
                cmd.append(f"--load-extension={_stealth_ext_dir}")
        if self._proxy_server:
            cmd.append(f"--proxy-server={self._proxy_server}")
        # When extra request headers are configured (canonically a sim-env
        # consent Cookie like ``bt_cookie_consent``), the first document fetch
        # must wait until the persistent header session is open AND the cookie
        # is seeded into the jar. Launching Chrome with the real URL here fetches
        # it immediately — before ``_open_header_session`` and with an empty jar
        # — so the cookie-gated consent overlay renders and find_all reads it as
        # ``page_blocked`` (the SRP never reaches the vision classifier as a real
        # listings page). Launch to about:blank and defer the real navigation to
        # after the header/cookie setup below, via the cookie-seeding path.
        _defer_nav = bool(self._extra_http_headers and url and url != "about:blank")
        cmd.append("about:blank" if _defer_nav else url)

        self._browser_proc = subprocess.Popen(
            cmd, env=self._env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for Chrome to actually bind its CDP debug port before the
        # stealth-inject + header-session calls below. A fixed ``time.sleep(3)``
        # raced the cold-container cold-start: on a freshly-scheduled Modal
        # container Chrome can take >3s to bind ``--remote-debugging-port``, so
        # ``/json/list`` returned ECONNREFUSED and the persistent header seam
        # never opened — which drops the Daytona consent cookie, so the sim env
        # renders its consent overlay (read as a blocked page by the click
        # handler's find_all pre-scan → page_blocked halt → 0 leads). Poll the
        # port instead of guessing a settle time.
        self._wait_for_cdp_ready()
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

        # Persistent request-header session (no-op unless configured).
        # MUST come after the stealth block: a one-shot CDP header set
        # reverts the instant its ws detaches (CDP overrides are
        # session-scoped), so a header that has to survive runner-driven
        # navigations AND in-page link clicks needs ONE ws that stays
        # open for the browser's lifetime. See ``_open_header_session``.
        if self._extra_http_headers:
            self._open_header_session()

        # Deferred cold-start navigation (see ``_defer_nav`` above). The header
        # session is now live and ``_navigate_running_browser`` seeds the consent
        # cookie into the jar via ``Network.setCookie`` *before* ``Page.navigate``
        # — so the first real document (e.g. the by-owner SRP) loads without the
        # cookie-gated consent overlay, and the stealth patches registered just
        # above apply to it. Launching to about:blank kept that first fetch from
        # racing ahead with an empty jar.
        if _defer_nav:
            self._navigate_running_browser(url)

    def _wait_for_cdp_ready(self, deadline_s: float = 15.0) -> bool:
        """Block until Chrome's CDP HTTP endpoint answers, or ``deadline_s``.

        Returns ``True`` once ``http://127.0.0.1:{cdp_port}/json/version``
        responds 200 (the standard "DevTools is up" probe). Best-effort: on
        timeout logs a WARNING and returns ``False`` so the caller still
        proceeds (the stealth + header-session calls are each independently
        best-effort), but the poll gives a cold-start Chrome the time it needs
        to bind the debug port rather than racing a fixed sleep. A floor of one
        short sleep also lets the launch tab map under Xvfb on the warm path.
        """
        import urllib.error
        import urllib.request

        time.sleep(0.5)
        deadline = time.time() + deadline_s
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self._cdp_port}/json/version",
                    timeout=2,
                ) as resp:
                    if getattr(resp, "status", 200) == 200:
                        logger.info("CDP endpoint ready after %d attempt(s)", attempt)
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        logger.warning(
            "CDP endpoint not ready after %.0fs (port %s) — stealth + header "
            "session may fall through to the blocked interstitial",
            deadline_s, self._cdp_port,
        )
        return False

    def _open_header_session(self) -> None:
        """Hold a persistent Network-enabled CDP session that applies
        ``self._extra_http_headers`` to every page request.

        Why persistent: ``Network.setExtraHTTPHeaders`` is session-scoped —
        it reverts the moment the ws detaches. The one-shot ``_cdp_call``
        pattern therefore can't carry a header across the runner's later
        ``Page.navigate`` calls, and certainly not across in-page link
        clicks / ``location.href`` pagination (which no CDP call drives at
        all). A single ws that enables Network, sets the header, and stays
        open covers ALL of those request paths until the browser closes.

        ``Network.enable`` is mandatory first — ``setExtraHTTPHeaders``
        silently no-ops (returns ``{}`` but applies nothing) without it.

        Best-effort: any failure leaves ``self._header_ws`` ``None`` and is
        logged at WARNING (the header is the only thing past the Daytona
        preview interstitial, so a silent miss would be hard to diagnose).
        """
        try:
            import json as _json
            import threading
            import urllib.request
            try:
                import websocket  # websocket-client package
            except ImportError:
                logger.warning(
                    "extra_http_headers set but websocket-client missing — "
                    "header session not opened"
                )
                return
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
            # Unlike _cdp_call we DON'T skip about:blank — at startup the
            # only page target is the launch tab (about:blank), and that
            # is exactly the session we need to hold.
            ws_url: str | None = None
            for tab in tabs:
                if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                    ws_url = tab["webSocketDebuggerUrl"]
                    break
            if not ws_url:
                logger.warning("header session: no page target to attach to")
                return
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.settimeout(5)

            def _send(method: str, params: dict[str, Any]) -> None:
                req_id = int(time.time() * 1e6) % 1_000_000
                ws.send(_json.dumps({"id": req_id, "method": method, "params": params}))
                for _ in range(40):
                    decoded = _json.loads(ws.recv())
                    if decoded.get("id") == req_id:
                        return

            _send("Network.enable", {})
            _send("Network.setExtraHTTPHeaders", {"headers": self._extra_http_headers})
            self._header_ws = ws
            # Daemon drain: Chrome streams Network.* events on this ws once
            # enabled; if we never read them the socket buffer fills and the
            # connection wedges. The thread just discards everything and
            # exits when the ws closes (recv raises).
            ws.settimeout(None)

            def _drain() -> None:
                try:
                    while True:
                        ws.recv()
                except Exception:  # noqa: BLE001 — ws closed / shutdown
                    pass

            t = threading.Thread(target=_drain, daemon=True)
            t.start()
            self._header_ws_thread = t
            # WARNING (not INFO): this seam is load-bearing for the Daytona
            # sim-env run and Modal suppresses INFO in production logs, so a
            # visible confirmation is the only way to tell the session opened
            # vs. silently fell through to the blocked interstitial.
            logger.warning(
                "persistent header session open: %s",
                list(self._extra_http_headers.keys()),
            )
        except Exception as exc:  # noqa: BLE001 — never fatal to a run
            logger.warning("header session setup failed: %s", exc)
            self._header_ws = None

    def _close_header_session(self) -> None:
        """Close the persistent header ws (reverts the header). Idempotent."""
        ws = self._header_ws
        self._header_ws = None
        self._header_ws_thread = None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass

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

    def _xdotool_capture(self, *args: str) -> str:
        """Run xdotool and return stdout (for ``getmouselocation`` etc.)."""
        try:
            r = subprocess.run(
                ["xdotool"] + list(args),
                env=self._env, capture_output=True, timeout=2, text=True,
            )
            return r.stdout or ""
        except (subprocess.TimeoutExpired, OSError):
            return ""

    def _current_mouse_position(self) -> tuple[int, int] | None:
        """Read the cursor position via xdotool. Returns ``None`` when
        the X server / display isn't reachable (test contexts, headless
        without Xvfb)."""
        out = self._xdotool_capture("getmouselocation")
        if not out:
            return None
        try:
            x_part, y_part, *_ = out.split()
            x = int(x_part.split(":", 1)[1])
            y = int(y_part.split(":", 1)[1])
            return x, y
        except (ValueError, IndexError):
            return None

    def _mousemove_with_curve(self, x: int, y: int) -> None:
        """Move the cursor to ``(x, y)`` via a Bezier curve when
        humanlike behavioral signals are enabled (#824).

        Falls back to a direct ``mousemove`` when:
        - ``MANTIS_BEHAVIORAL_JITTER=0`` (opt-out for CI / replay)
        - Current cursor position is unreadable (no X display)
        - Start and end are the same point
        """
        from .behavioral import bezier_waypoints, is_enabled, waypoint_delay

        if not is_enabled():
            self._xdotool("mousemove", str(x), str(y))
            return
        current = self._current_mouse_position()
        if current is None:
            self._xdotool("mousemove", str(x), str(y))
            return
        for wx, wy in bezier_waypoints(current, (x, y), steps=3):
            self._xdotool("mousemove", str(wx), str(wy))
            time.sleep(waypoint_delay())
        # Always land on the exact target — the Bezier sampling
        # excludes endpoints so this final mousemove is what actually
        # parks the cursor on the click target.
        self._xdotool("mousemove", str(x), str(y))

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

    def cdp_contenteditable_insert(self, text: str) -> bool:
        """#931 P1: insert ``text`` into the focused contenteditable host and
        **verify it landed**, returning ``False`` if it did not.

        Rich-text editors (LinkedIn's message box, Reddit's composer) are
        ``contenteditable`` ``<div>``s, not ``<input>``s — the plain TYPE
        ladder focuses a pixel that may be a non-editable child and the
        text never lands.

        #931 follow-up — two fixes for the LinkedIn message box:

        1. **Fire the real input pipeline.** LinkedIn's editor (Draft/
           Lexical-style) only syncs its model — and only then enables the
           greyed-out Send button — when a native ``beforeinput`` event
           fires. ``Input.insertText`` + a hand-rolled ``input`` event does
           NOT fire ``beforeinput``, so text never reached the model. We
           now insert via ``document.execCommand('insertText', …)``, which
           drives the full ``beforeinput``→``input`` pipeline the editor
           listens to. The ``Input.insertText`` path remains a fallback.

        2. **Verify, don't assume.** The old version returned ``True`` as
           soon as the CDP call succeeded, with no read-back — so an empty
           box was reported as "filled", and the director then looped
           clicking a disabled Send until the hard-loop guard stopped the
           run. We now read the host's ``textContent`` back and return
           whether the expected text is actually present.

        Returns ``False`` (caller falls back to the verified TYPE ladder)
        when CDP is unavailable, no editable host is in focus, or the
        read-back shows the text did not land. CUA note: action-side
        dispatch of a vision-derived fill + post-action read-back verify
        (per ``feedback_cua_cdp_post_action_verify.md``), not DOM grounding.
        """
        eval_fn = getattr(self, "cdp_evaluate", None)
        if not callable(eval_fn):
            return False
        text_js = json.dumps(text)
        # Primary attempt: focus the editable host, select its contents,
        # and insert via execCommand (drives beforeinput→input natively),
        # then read the host's text back in the same call.
        try:
            result = eval_fn(
                "(() => {"
                "const a = document.activeElement;"
                "let host = (a && a.isContentEditable) ? a : "
                "(a && a.closest ? a.closest('[contenteditable=\"\"], [contenteditable=\"true\"]') : null);"
                # Fall back to whatever contenteditable / textbox currently
                # holds focus (the active pixel may be a non-editable child).
                "if (!host) host = document.querySelector("
                "'[contenteditable=\"true\"]:focus, [contenteditable=\"\"]:focus, [role=\"textbox\"]:focus');"
                "if (!host) return {ok:false};"
                "host.focus();"
                "const sel = window.getSelection();"
                "if (sel) { const r = document.createRange(); r.selectNodeContents(host);"
                " sel.removeAllRanges(); sel.addRange(r); }"
                "let inserted = false;"
                f"try {{ inserted = document.execCommand('insertText', false, {text_js}); }} catch (e) {{ inserted = false; }}"
                "return {ok:true, inserted: inserted, text: String(host.textContent == null ? '' : host.textContent)};"
                "})()"
            )
        except Exception as exc:  # noqa: BLE001 — never fatal; caller falls back
            logger.debug("cdp_contenteditable_insert execCommand raised: %s", exc)
            result = None

        if isinstance(result, dict) and not result.get("ok"):
            return False  # no editable host in focus — caller uses TYPE ladder
        if isinstance(result, dict) and self._type_matches(text, str(result.get("text") or "")):
            return True  # execCommand landed and the read-back confirms it

        # Fallback: execCommand was a no-op (some editors block it) — try the
        # native Input.insertText, fire beforeinput+input explicitly, re-read.
        if not self._cdp_insert_text(text):
            return False
        try:
            readback = eval_fn(
                "(() => {"
                "const a = document.activeElement;"
                "let host = (a && a.isContentEditable) ? a : "
                "(a && a.closest ? a.closest('[contenteditable=\"\"], [contenteditable=\"true\"]') : null);"
                "if (!host) return null;"
                "host.dispatchEvent(new InputEvent('beforeinput', "
                f"{{bubbles:true, cancelable:true, inputType:'insertText', data:{text_js}}}));"
                "host.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'insertText'}));"
                "return String(host.textContent == null ? '' : host.textContent);"
                "})()"
            )
        except Exception:  # noqa: BLE001 — best-effort event sync
            readback = None
        if readback is None:
            return False
        return self._type_matches(text, str(readback))

    def _read_active_element_text(self) -> str | None:
        """Read the focused element's current text via CDP, or ``None``.

        Returns the ``value`` of an ``<input>``/``<textarea>`` or the
        ``textContent`` of a contenteditable host. ``None`` when there is
        no focused text field (so the caller treats it as "couldn't
        verify" rather than "empty"), or on any CDP failure.

        Per ``feedback_cua_cdp_post_action_verify.md`` this is an
        action-side post-action read (confirming OUR type landed), not
        DOM-grounding — it never derives the next action's target.
        """
        try:
            return self.cdp_evaluate(
                "(() => {"
                "const el = document.activeElement;"
                "if (!el) return null;"
                "if (el.isContentEditable) return String(el.textContent == null ? '' : el.textContent);"
                "if (el.value !== undefined && el.value !== null) return String(el.value);"
                "return null;"
                "})()"
            )
        except Exception:  # noqa: BLE001 — verification must never be fatal
            return None

    @staticmethod
    def _type_matches(expected: str, actual: str) -> bool:
        """Whitespace-normalized match tolerant of field auto-formatting.

        Exact normalized equality, or expected contained in actual (covers
        phone/card inputs that inject separators, and prefixed fields).
        """
        exp = " ".join((expected or "").split())
        act = " ".join((actual or "").split())
        if not exp:
            return True
        return exp == act or exp in act

    def _active_field_probe(self) -> dict | None:
        """Probe the focused element after a TYPE, distinguishing the two
        cases the old ``_read_active_element_text`` conflated into ``None``:

        - ``None`` — CDP is unavailable or the probe threw. Genuinely
          can't tell (env without CDP). Caller stays unverified.
        - ``{"has_field": True, "text": str}`` — an editable field
          (input / textarea / contenteditable) holds focus; read its text.
        - ``{"has_field": False}`` — CDP answered but NO editable field is
          focused (``activeElement`` is ``<body>`` / null / non-editable).
          After a TYPE this is the focus-stolen signature: on login pages a
          passkey / credential popup grabs focus, so the keystrokes never
          reach the ``<input>`` and it stays empty. This MUST read as a
          failure, not a benign "unverified" — that divergence (logs say
          typed, video shows empty) is the bug this fixes.

        Per ``feedback_cua_cdp_post_action_verify.md`` this is an
        action-side post-action read (confirming OUR type landed), not
        DOM-grounding — it never derives the next action's target.
        """
        eval_fn = getattr(self, "cdp_evaluate", None)
        if not callable(eval_fn):
            return None
        try:
            state = eval_fn(
                "(() => {"
                "const el = document.activeElement;"
                "if (!el || el === document.body || el === document.documentElement) return {has_field:false};"
                "if (el.isContentEditable) return {has_field:true, text:String(el.textContent == null ? '' : el.textContent)};"
                "const tag = (el.tagName || '').toUpperCase();"
                "if ((tag === 'INPUT' || tag === 'TEXTAREA') && el.value !== undefined && el.value !== null)"
                "  return {has_field:true, text:String(el.value)};"
                "return {has_field:false};"
                "})()"
            )
        except Exception:  # noqa: BLE001 — verification must never be fatal
            return None
        if not isinstance(state, dict):
            return None
        if state.get("has_field"):
            return {"has_field": True, "text": str(state.get("text") or "")}
        return {"has_field": False}

    def _verify_typed_text(self, expected: str) -> dict | None:
        """Post-type read-back verdict for ``gym_result.info['type_verified']``.

        Returns ``{"success", "expected", "actual"}`` when CDP can observe
        the focused field, else ``None`` (truly unverifiable — CDP off).
        Gated by ``MANTIS_VERIFY_TYPE`` (default on).

        Crucially: when CDP answers but no editable field holds focus after
        the type, this returns ``success=False`` with a ``reason`` — the
        focus-stolen case (passkey popup) that previously fell through to a
        false-reassuring "(unverified)".
        """
        if os.environ.get("MANTIS_VERIFY_TYPE", "enabled").strip().lower() in ("0", "off", "disabled", "false"):
            return None
        probe = self._active_field_probe()
        if probe is None:
            return None
        if not probe.get("has_field"):
            return {
                "success": False,
                "expected": expected,
                "actual": "",
                "reason": (
                    "no input field is focused after typing — the text did not "
                    "land (a passkey / credential popup can steal focus on login "
                    "pages). Dismiss the popup, click the field, and retype."
                ),
            }
        actual = str(probe.get("text") or "")
        return {
            "success": self._type_matches(expected, actual),
            "expected": expected,
            "actual": actual,
        }

    def cdp_history_back(self, *, settle_seconds: float = 1.5) -> bool:
        """#583: navigate back via ``window.history.back()`` over CDP.

        More reliable than xdotool ``Alt+Left`` on SPA sites that use
        ``history.pushState`` — the keyboard shortcut sometimes doesn't
        pop the SPA's history state cleanly; the JS call always does
        when there's a history entry to pop.

        Returns True when the URL changed within ``settle_seconds`` of
        the back call, False otherwise (callers fall back to Alt+Left).

        Memory note: per ``feedback_cua_cdp_post_action_verify.md``,
        CDP for action dispatch + post-action verify is allowed. This
        is action-side (dispatching back), not DOM-grounding-side.
        """
        url_before = self.current_url or ""
        # Dispatch the back call. ``cdp_evaluate`` returns ``None`` when
        # the call succeeds (void return) or on failure — we can't
        # distinguish from the return value, so verify via URL change.
        try:
            self.cdp_evaluate("window.history.back()")
        except Exception as exc:  # noqa: BLE001 — fall back on any failure
            logger.debug("cdp_history_back: dispatch failed: %s", exc)
            return False
        # Poll URL up to ``settle_seconds`` for the change.
        deadline = time.time() + max(0.1, settle_seconds)
        poll_interval = 0.1
        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                url_after = self.current_url or ""
            except Exception:  # noqa: BLE001
                url_after = ""
            if url_after and url_after != url_before:
                logger.info(
                    "  [cdp-back] history.back() succeeded: %s → %s",
                    url_before[:60], url_after[:60],
                )
                return True
        return False

    def cdp_count_pages(self) -> int:
        """#582: count Chrome page-type tabs via ``/json/list``.

        Returns the number of ``type=page`` tabs whose URL isn't a
        system page (``chrome://`` / ``about:``). Returns ``0`` on any
        CDP failure (port unreachable, json decode, etc) — caller
        treats ``0`` as "couldn't check" and skips the diff.

        Used by the click handler to detect new-tab opens from
        ``window.open()`` / modifier-clicks that bypass the existing
        middle-click fallback path which is the only writer of
        ``_opened_detail_in_new_tab`` today. A snapshot before + after
        the click + comparison sets the flag regardless of which click
        primitive opened the tab.

        Memory note: per ``feedback_cua_cdp_post_action_verify.md``,
        post-action CDP reads (verifying our action's side-effect) are
        allowed. We're checking whether OUR click opened a tab —
        action-side, not DOM-grounding-side.
        """
        try:
            import json as _json
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self._cdp_port}/json/list",
                timeout=2,
            ) as resp:
                tabs = _json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 — observability, never fatal
            logger.debug("cdp_count_pages failed: %s", exc)
            return 0
        if not isinstance(tabs, list):
            return 0
        count = 0
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            if tab.get("type") != "page":
                continue
            url = str(tab.get("url") or "")
            if not url or url.startswith("chrome://") or url.startswith("about:"):
                continue
            count += 1
        return count

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

    @staticmethod
    def _url_host(url: str) -> str:
        """Bare host of ``url`` (lowercased, ``www.`` stripped), or ""."""
        try:
            from urllib.parse import urlparse
            host = (urlparse(str(url or "")).hostname or "").lower()
        except Exception:  # noqa: BLE001
            return ""
        return host[4:] if host.startswith("www.") else host

    def _await_navigation_commit(
        self, target_host: str, *, max_seconds: float | None = None,
    ) -> bool:
        """Poll the live URL until its host matches ``target_host`` (the
        navigation actually committed) or the budget expires.

        Returns True iff the live URL reached the target host. Returns True
        immediately when ``target_host`` is empty (can't verify → don't
        block) or when there is no ``current_url`` signal (legacy adapters /
        tests — preserve the old fire-and-settle behaviour). A bounded poll
        plus a short paint settle on success.
        """
        budget = (self._settle_time + 2.0) if max_seconds is None else max_seconds
        if not target_host or not hasattr(self, "current_url"):
            time.sleep(budget)
            return True
        deadline = time.time() + max(0.5, budget)
        while time.time() < deadline:
            try:
                cur = self.current_url or ""
            except Exception:  # noqa: BLE001
                cur = ""
            if cur and self._url_host(cur) == target_host:
                time.sleep(0.4)  # brief paint settle once committed
                return True
            time.sleep(0.25)
        return False

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
        # S01 guard: never navigate to a browser-vendor download/update
        # page. These are never a task destination — reaching one means a
        # stray keystroke activated Chrome's "Reinstall Chrome" nag, which
        # in cua-issues run S01 dead-ended on google.com/chrome. Refusing
        # here also covers the brain emitting such a URL directly.
        if is_browser_vendor_url(url):
            logger.warning(
                "refusing forbidden browser-vendor navigation to %s (S01 trap)",
                url[:80],
            )
            return
        self._seed_request_cookies(url)
        target_host = self._url_host(url)
        logger.info(f"Navigating to {url[:80]} (via CDP Page.navigate)")
        ok, _ = self._cdp_call("Page.navigate", {"url": url})
        if ok:
            # cua-issues 2026-06-29: Page.navigate returning ok means the
            # COMMAND was accepted, NOT that the page reached the target.
            # ~14 /v1/cua runs issued a navigate (to Reddit/jobs/login) yet
            # the cached tab kept rendering linkedin.com/in/akhil08 — the
            # screen never moved. Verify the live URL actually reaches the
            # target host; if it's still stale, re-navigate once, then force
            # a cache-bypassing reload. (Reading current_url to confirm our
            # OWN navigation landed is action-side post-verify, not target
            # grounding — feedback_cua_cdp_post_action_verify.)
            if self._await_navigation_commit(target_host):
                return
            logger.warning(
                "Page.navigate accepted but did not reach %s (stale tab) — "
                "re-navigating", target_host or url[:60],
            )
            self._cdp_call("Page.navigate", {"url": url})
            if self._await_navigation_commit(target_host):
                return
            logger.warning("still stale after retry — forcing Page.reload(ignoreCache)")
            self._cdp_call("Page.reload", {"ignoreCache": True})
            self._await_navigation_commit(target_host)
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

    def _seed_request_cookies(self, url: str) -> None:
        """Seed cookies from an extra ``Cookie`` request header into the jar.

        ``_open_header_session`` applies ``self._extra_http_headers`` via
        ``Network.setExtraHTTPHeaders``, but Chromium's network service drops a
        manually-set ``Cookie`` header from that path (it composes the request
        ``Cookie`` from the cookie store, not from extra headers) while honoring
        the custom ``x-daytona-*`` headers in the same dict. That divergence is
        why the Daytona preview-warning bypass works yet a sim-env consent
        cookie (e.g. ``bt_cookie_consent``) never reaches the server, so its
        cookie-gated banner renders and ``ClaudeExtractor.find_all_listings``
        reads the full-page overlay as ``page_blocked``. ``Network.setCookie``
        puts the cookie in the real jar for ``url``'s host, where the network
        stack *does* send it. No-op unless a ``Cookie`` header is present and
        ``url`` is http(s) — so non-sim-env runs are unaffected.
        """
        headers = getattr(self, "_extra_http_headers", None) or {}
        cookie_header = next(
            (v for k, v in headers.items() if k.lower() == "cookie"), ""
        )
        if not cookie_header or not url.startswith(("http://", "https://")):
            return
        for pair in cookie_header.split(";"):
            name, _, value = pair.strip().partition("=")
            if name:
                self._cdp_call(
                    "Network.setCookie",
                    {"url": url, "name": name, "value": value.strip()},
                )

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
        info: dict = {}
        # #931 P0: read back the focused field after a TYPE so logs and
        # verdicts reflect whether the text actually landed (kills the
        # blanket "(unverified)" log). URL-shaped text drives the omnibox
        # (no DOM field to read) — skip it.
        if action.action_type == ActionType.TYPE:
            typed = str(action.params.get("text") or action.params.get("content") or "")
            if typed and not (typed.startswith("http://") or typed.startswith("https://")):
                verdict = self._verify_typed_text(typed)
                # cua-issues 2026-06-29: the /v1/cua (Claude) loop types via
                # raw xdotool only — it never used the contenteditable insert
                # / focus-retry that form.py's fill_field has, so LinkedIn's
                # contenteditable message box (W06), Reddit's login field
                # (L06), and every search/composer left the field EMPTY while
                # reporting nothing landed. When the read-back says the text
                # didn't land, fall back to the #934 contenteditable insert
                # (execCommand→beforeinput, the editor's model syncs) and
                # re-verify. No-op for plain inputs that already succeeded, or
                # when there's no editable host (the insert self-gates). Now
                # every env.step(TYPE) caller — Claude loop included — gets it.
                if verdict is not None and not verdict.get("success"):
                    inserter = getattr(self, "cdp_contenteditable_insert", None)
                    if callable(inserter):
                        try:
                            if inserter(typed):
                                verdict = self._verify_typed_text(typed) or verdict
                        except Exception as exc:  # noqa: BLE001 — never fatal
                            logger.debug("TYPE contenteditable fallback raised: %s", exc)
                if verdict is not None:
                    info["type_verified"] = verdict
        return GymResult(observation=obs, reward=0.0, done=False, info=info)

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
        self._close_header_session()

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
                # #824 humanlike behavioral: when jitter is enabled,
                # interpolate the cursor path with a Bezier curve so
                # the click doesn't look like a teleport followed by an
                # immediate fire. Falls back to direct mousemove when
                # MANTIS_BEHAVIORAL_JITTER=0 or when the current cursor
                # position can't be read.
                self._mousemove_with_curve(x, y)
                if self._human_speed:
                    time.sleep(random.uniform(0.05, 0.15))
                self._xdotool("click", btn_num)

            case ActionType.DOUBLE_CLICK:
                x = action.params.get("x", self._viewport[0] // 2)
                y = action.params.get("y", self._viewport[1] // 2)
                x, y = self._clamp(x, y)
                self._mousemove_with_curve(x, y)
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
