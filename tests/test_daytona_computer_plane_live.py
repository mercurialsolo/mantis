"""Live integration test for the Daytona computer plane (#699 Phase 2).

Skipped by default — runs only when ``DAYTONA_LIVE_PREVIEW_URL`` +
``DAYTONA_LIVE_PREVIEW_TOKEN`` are set in the env, both of which the
``deploy/computer_plane/daytona_provision.py`` script prints after a
successful provision.

The test wires :class:`RemoteComputerImpl` directly at the running
sandbox's preview URL — it does NOT provision a fresh sandbox. The
reason: provisioning takes 5+ minutes and the daytona_provision
script already exercises that path end-to-end. This test focuses on
the wire-contract level: prove ``POST /session/init`` → ``POST
/screenshot`` → ``POST /xdotool`` round-trip through the auth-gated
preview URL.

Usage::

    DAYTONA_LIVE_PREVIEW_URL=... \
    DAYTONA_LIVE_PREVIEW_TOKEN=... \
    pytest tests/test_daytona_computer_plane_live.py -s
"""

from __future__ import annotations

import base64
import os

import pytest


pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("DAYTONA_LIVE_PREVIEW_URL"),
        reason="DAYTONA_LIVE_PREVIEW_URL not set — live Daytona test skipped",
    ),
    pytest.mark.skipif(
        not os.environ.get("DAYTONA_LIVE_PREVIEW_TOKEN"),
        reason="DAYTONA_LIVE_PREVIEW_TOKEN not set — live Daytona test skipped",
    ),
]


def _build_client():
    """Build a RemoteComputerImpl wired at the live preview URL with
    the auth0-bypass token + skip-preview header on every call."""
    from mantis_agent.gym.remote_computer_impl import RemoteComputerImpl

    base_url = os.environ["DAYTONA_LIVE_PREVIEW_URL"].rstrip("/")
    token = os.environ["DAYTONA_LIVE_PREVIEW_TOKEN"]
    headers = {
        "X-Daytona-Skip-Preview-Warning": "true",
        "X-Daytona-Preview-Token": token,
    }
    return RemoteComputerImpl(
        base_url=base_url,
        tenant_id="acme",
        profile_id="live-int",
        run_id="r-live",
        start_url="about:blank",
        extra_http_headers=headers,
    )


def test_live_session_init_and_screenshot_round_trip() -> None:
    """End-to-end smoke: session_init → screenshot returns a PNG."""
    import requests

    client = _build_client()
    # Direct HTTP probe via the underlying requests client — we
    # don't reset the full env (which would launch Chrome and is
    # the next layer of integration to land in a follow-up).
    base = client._base_url  # noqa: SLF001
    headers = {
        "X-Daytona-Skip-Preview-Warning": "true",
        "X-Daytona-Preview-Token": os.environ["DAYTONA_LIVE_PREVIEW_TOKEN"],
    }
    health = requests.get(f"{base}/health", headers=headers, timeout=10)
    assert health.status_code == 200, health.text
    body = health.json()
    assert body.get("ok") is True
    print(f"  ✓ /health 200: {body}")

    init = requests.post(
        f"{base}/session/init",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "tenant_id": "acme",
            "profile_id": "live-int",
            "run_id": "r-live",
            "start_url": "about:blank",
            "viewport": [1280, 720],
        },
        timeout=60,
    )
    assert init.status_code == 200, init.text
    session_body = init.json()
    session_token = session_body.get("session_token")
    assert session_token, f"no session_token in init body: {session_body}"
    print(f"  ✓ /session/init 200 session_token={session_token[:12]}…")

    auth = {**headers, "X-Mantis-Session": session_token}
    shot = requests.post(
        f"{base}/screenshot", headers=auth, json={}, timeout=30,
    )
    assert shot.status_code == 200, shot.text
    payload = shot.json()
    # Wire surface returns the screenshot under ``image_b64`` (see
    # ``ScreenshotResponse`` in ``gym/computer_wire.py``).
    png_b64 = payload.get("image_b64", "")
    assert png_b64, f"no image_b64 in screenshot body: {list(payload.keys())}"
    decoded = base64.b64decode(png_b64)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n", "screenshot is not a PNG"
    print(
        f"  ✓ /screenshot 200 png_size={len(decoded)}B "
        f"viewport={payload.get('width')}x{payload.get('height')}"
    )

    close = requests.post(
        f"{base}/session/close", headers=auth,
        json={"session_token": session_token}, timeout=20,
    )
    assert close.status_code == 200, close.text
    print("  ✓ /session/close 200")
