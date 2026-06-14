"""Drive each auth ceremony over HTTP and assert its oracle flips to pass.

This is the real proof the env works: every method, walked through the
actual routes with a cookie jar, lands a ``login_succeeded`` audit row
and reaches the console — and the matching oracle agrees.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="auth env flow test needs the server extra")
pytest.importorskip("httpx", reason="starlette TestClient needs httpx")
from fastapi.testclient import TestClient  # noqa: E402

DEMO_EMAIL = "ada@mantis.example"


@pytest.fixture
def client():
    from app.main import app  # noqa: PLC0415

    with TestClient(app) as c:
        yield c


def _grade(task_id: str) -> dict:
    from app import db  # noqa: PLC0415
    from app.oracles import grade  # noqa: PLC0415

    return grade(task_id, db.connect(), now="2026-01-15T09:00:00Z", seed_val=42)


def _emails(kind: str):
    from app import db  # noqa: PLC0415

    return db.connect().execute(
        "SELECT * FROM emails WHERE kind = ? ORDER BY id DESC", (kind,)
    ).fetchall()


# ── password ────────────────────────────────────────────────────────────


def test_password_login(client):
    r = client.post("/login", data={
        "email": DEMO_EMAIL, "password": "hunter2", "next": "/console"})
    assert r.status_code == 200
    assert "console" in r.text.lower()
    assert _grade("T01_password_login")["passed"] is True


def test_password_wrong_is_rejected(client):
    client.post("/login", data={
        "email": DEMO_EMAIL, "password": "nope", "next": "/console"})
    assert _grade("T01_password_login")["passed"] is False


# ── OAuth (every provider) ──────────────────────────────────────────────


@pytest.mark.parametrize("provider,task_id", [
    ("google", "T02_oauth_google"),
    ("github", "T03_oauth_github"),
    ("microsoft", "T04_oauth_microsoft"),
    ("okta", "T05_oauth_okta"),
])
def test_oauth_flow(client, provider, task_id):
    subject = f"{provider}-oauth2|user_00001"
    redirect_uri = f"/auth/oauth/{provider}/callback"

    picker = client.get(f"/auth/oauth/{provider}/authorize?state=login")
    assert picker.status_code == 200
    assert subject in picker.text

    consent = client.post(f"/auth/oauth/{provider}/consent", data={
        "subject": subject, "redirect_uri": redirect_uri, "state": "login"})
    assert consent.status_code == 200

    grant = client.post(f"/auth/oauth/{provider}/grant", data={
        "subject": subject, "redirect_uri": redirect_uri, "state": "login"})
    assert grant.status_code == 200
    m = re.search(r"code=([A-Za-z0-9_\-]+)", grant.text)
    assert m, "grant page should embed an authorization code"

    cb = client.get(
        f"/auth/oauth/{provider}/callback?code={m.group(1)}&state=login")
    assert cb.status_code == 200
    assert _grade(task_id)["passed"] is True


def test_oauth_code_is_single_use(client):
    subject = "google-oauth2|user_00001"
    client.post("/auth/oauth/google/grant", data={
        "subject": subject, "redirect_uri": "/auth/oauth/google/callback",
        "state": "login"})
    grant = client.post("/auth/oauth/google/grant", data={
        "subject": subject, "redirect_uri": "/auth/oauth/google/callback",
        "state": "login"})
    code = re.search(r"code=([A-Za-z0-9_\-]+)", grant.text).group(1)
    client.get(f"/auth/oauth/google/callback?code={code}&state=login")
    # Replaying the same code must not authenticate again.
    from app import db  # noqa: PLC0415
    db.connect().execute("DELETE FROM mutations")
    replay = client.get(
        f"/auth/oauth/google/callback?code={code}&state=login",
        follow_redirects=False)
    assert replay.status_code == 303
    assert "error" in replay.headers.get("location", "")


# ── email magic link ────────────────────────────────────────────────────


def test_magic_link_flow(client):
    sent = client.post("/auth/magic", data={"email": DEMO_EMAIL})
    assert sent.status_code == 200
    rows = _emails("magic_link")
    assert rows, "a magic-link email should be delivered"
    token = rows[0]["token"]

    # Read it from the inbox surface, then follow the link.
    inbox = client.get("/inbox")
    assert "magic" in inbox.text.lower()
    verified = client.get(f"/auth/magic/verify?token={token}")
    assert verified.status_code == 200
    assert _grade("T06_magic_link_email")["passed"] is True


def test_magic_link_is_single_use(client):
    client.post("/auth/magic", data={"email": DEMO_EMAIL})
    token = _emails("magic_link")[0]["token"]
    client.get(f"/auth/magic/verify?token={token}")
    replay = client.get(f"/auth/magic/verify?token={token}",
                        follow_redirects=False)
    assert replay.status_code == 303
    assert "error" in replay.headers.get("location", "")


# ── email OTP ────────────────────────────────────────────────────────────


def test_email_otp_flow(client):
    client.post("/auth/otp", data={"email": DEMO_EMAIL})
    code = _emails("otp")[0]["code"]
    assert re.fullmatch(r"\d{6}", code)

    r = client.post("/auth/otp/verify", data={"email": DEMO_EMAIL, "code": code})
    assert r.status_code == 200
    assert _grade("T07_email_otp")["passed"] is True


def test_email_otp_wrong_code_rejected(client):
    client.post("/auth/otp", data={"email": DEMO_EMAIL})
    r = client.post("/auth/otp/verify",
                    data={"email": DEMO_EMAIL, "code": "000000"})
    assert "invalid" in r.text.lower()
    assert _grade("T07_email_otp")["passed"] is False


# ── passkey ──────────────────────────────────────────────────────────────


def test_passkey_flow(client):
    page = client.get("/auth/passkey")
    challenge = re.search(r'name="challenge" value="([^"]+)"', page.text).group(1)
    assert "cred_00001" in page.text

    r = client.post("/auth/passkey/assert",
                    data={"cred_id": "cred_00001", "challenge": challenge})
    assert r.status_code == 200
    assert _grade("T08_passkey")["passed"] is True


def test_passkey_replayed_challenge_rejected(client):
    page = client.get("/auth/passkey")
    challenge = re.search(r'name="challenge" value="([^"]+)"', page.text).group(1)
    client.post("/auth/passkey/assert",
                data={"cred_id": "cred_00001", "challenge": challenge})
    replay = client.post("/auth/passkey/assert",
                         data={"cred_id": "cred_00001", "challenge": challenge},
                         follow_redirects=False)
    assert replay.status_code == 303
    assert "error" in replay.headers.get("location", "")


# ── auth gate + harness surface ──────────────────────────────────────────


def test_console_requires_auth(client):
    r = client.get("/console", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


def test_oracle_endpoint_is_gated(client):
    assert client.get("/__env__/oracle?task_id=T01_password_login").status_code == 401
    ok = client.get("/__env__/oracle?task_id=T01_password_login",
                    headers={"X-Env-Admin": "test-admin-token"})
    assert ok.status_code == 200
    assert ok.json()["passed"] is False
