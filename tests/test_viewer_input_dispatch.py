"""Source-level checks for the viewer's input-relay endpoints
(#viewer-input-dispatch).

The MJPEG stream is one-way; we add server endpoints that take
browser-coord clicks / keys / scroll and dispatch via xdotool on
the Xvfb display. Only fires when the run is paused (otherwise
the agent and user fight for the cursor).

End-to-end smoke (xdotool subprocess + real Xvfb) isn't viable in
CI; these tests pin the wire-up:

* /api/dispatch_click / dispatch_keys / dispatch_type /
  dispatch_scroll / desktop_info — endpoints exist in viewer_modal
* VIEWER_HTML has the mousedown/keydown/wheel handlers + the
  pause-gate (clicks suppressed unless currentRunStatus=paused)
* browser→desktop coord scaling code present
"""

from __future__ import annotations

import inspect


# ── Endpoint registration (viewer_modal.py) ──────────────────────────────


def test_dispatch_click_endpoint_registered():
    """``POST /api/dispatch_click`` must be defined in
    ``viewer_modal._start_background``."""
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert '"/api/dispatch_click"' in src or "'/api/dispatch_click'" in src


def test_dispatch_keys_endpoint_registered():
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert "dispatch_keys" in src


def test_dispatch_type_endpoint_registered():
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert "dispatch_type" in src


def test_dispatch_scroll_endpoint_registered():
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert "dispatch_scroll" in src


def test_desktop_info_endpoint_registered():
    """The browser needs desktop dimensions to scale coords —
    /api/desktop_info exposes them from the latest captured frame."""
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert "desktop_info" in src


def test_xdotool_helper_present():
    """Internal ``_xdotool(args)`` helper that runs the command with
    the right DISPLAY env. Without it endpoints can't dispatch."""
    from mantis_agent.viewer_modal import _start_background
    src = inspect.getsource(_start_background)
    assert "xdotool" in src
    assert "DISPLAY" in src


# ── HTML wiring ──────────────────────────────────────────────────────────


def test_html_has_mousedown_handler():
    """The <img id=feed> must have a mousedown listener that POSTs
    to /api/dispatch_click."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "mousedown" in VIEWER_HTML
    assert "/api/dispatch_click" in VIEWER_HTML


def test_html_has_keydown_handler():
    """Global keydown listener for typing + named keys (Enter,
    Tab, modifiers)."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "keydown" in VIEWER_HTML
    assert "/api/dispatch_type" in VIEWER_HTML
    assert "/api/dispatch_keys" in VIEWER_HTML


def test_html_has_wheel_scroll_handler():
    """Wheel scroll relays to xdotool button 4/5."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "wheel" in VIEWER_HTML
    assert "/api/dispatch_scroll" in VIEWER_HTML


def test_html_gates_input_on_paused_status():
    """CRITICAL: input dispatch must ONLY fire when run is paused.
    Sending clicks while the agent is running creates a mouse race —
    user and brain fighting for the same cursor."""
    from mantis_agent.viewer import VIEWER_HTML
    # The gate appears in the mousedown handler (and keydown + wheel).
    assert "currentRunStatus !== 'paused'" in VIEWER_HTML, (
        "Input dispatch handlers must guard on currentRunStatus=='paused'"
    )


def test_html_does_coord_scaling():
    """browser→desktop pixel scaling — clicks land in the right
    place even when the <img> is rendered smaller than the desktop
    capture size."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "browserToDesktop" in VIEWER_HTML
    assert "/api/desktop_info" in VIEWER_HTML
    assert "getBoundingClientRect" in VIEWER_HTML


def test_html_suppresses_context_menu_and_drag():
    """Right-click should dispatch a right-click via xdotool, NOT
    show the browser context menu. <img> drag should also be
    suppressed so click+drag dispatches properly."""
    from mantis_agent.viewer import VIEWER_HTML
    assert "contextmenu" in VIEWER_HTML
    assert "dragstart" in VIEWER_HTML
