"""Smoke tests for mantis-linkedin.

Boots the app in-process via Starlette's TestClient, exercises every
in-scope page (expecting 200), runs every oracle's happy-path through
HTTP, then queries the oracle endpoint to confirm passed=true.

Run with::

    cd deploy/sim_envs/mantis_linkedin
    ENV_ADMIN_TOKEN=test python scripts/smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure ``app.main`` is importable when run from the env's root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENV_ADMIN_TOKEN", "test")
os.environ.setdefault("SEED", "42")
os.environ.setdefault("FAKE_NOW", "2026-06-08T09:00:00Z")

# Defer imports until env is set.
from starlette.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


ADMIN_HEADERS = {"X-Env-Admin": os.environ["ENV_ADMIN_TOKEN"]}


def _check(label: str, predicate, *, exit_on_fail: bool = False) -> bool:
    ok = bool(predicate)
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {label}")
    if not ok and exit_on_fail:
        sys.exit(1)
    return ok


def smoke_pages(client: TestClient) -> int:
    failures = 0
    print("== Page smoke ==")
    pages = [
        ("GET /", "/", 200),
        ("GET /feed/", "/feed/", 200),
        ("GET /in/demo-mantis/", "/in/demo-mantis/", 200),
        ("GET /mynetwork/", "/mynetwork/", 200),
        ("GET /mynetwork/invitation-manager/",
         "/mynetwork/invitation-manager/", 200),
        ("GET /messaging/", "/messaging/", 200),
        ("GET /jobs/", "/jobs/", 200),
        ("GET /jobs/search/?keywords=engineer",
         "/jobs/search/?keywords=engineer", 200),
        ("GET /jobs/view/job_00003/", "/jobs/view/job_00003/", 200),
        ("GET /login", "/login", 200),
        ("GET /__env__/health", "/__env__/health", 200),
    ]
    for label, url, expected in pages:
        resp = client.get(url)
        ok = _check(
            f"{label} → {resp.status_code} (expected {expected})",
            resp.status_code == expected,
        )
        if not ok:
            failures += 1
            continue
        if url == "/__env__/health":
            ok = _check(
                "health returns ok=true",
                resp.json().get("ok") is True,
            )
            if not ok:
                failures += 1
        else:
            ok = _check(
                f"{url} returns HTML",
                "<html" in resp.text.lower(),
            )
            if not ok:
                failures += 1
    return failures


def smoke_oracle_dispatch(client: TestClient) -> int:
    failures = 0
    print("== Oracle dispatch ==")
    for task_id in ("t01", "t02", "t03", "t01_connect_with_user"):
        resp = client.get(
            f"/__env__/oracle?task_id={task_id}", headers=ADMIN_HEADERS,
        )
        if not _check(
            f"oracle?task_id={task_id} → 200",
            resp.status_code == 200,
        ):
            failures += 1
            continue
        body = resp.json()
        if not _check(
            f"oracle?task_id={task_id} returns required keys",
            {"passed", "score", "reasons"}.issubset(body.keys()),
        ):
            failures += 1
    # Unknown task should respond with passed=false rather than 500.
    resp = client.get(
        "/__env__/oracle?task_id=nonexistent", headers=ADMIN_HEADERS,
    )
    if not _check(
        "unknown task_id returns passed=false",
        resp.status_code == 200 and resp.json().get("passed") is False,
    ):
        failures += 1
    return failures


def reset(client: TestClient) -> None:
    client.post("/__env__/reset", headers=ADMIN_HEADERS)


def smoke_t01(client: TestClient) -> int:
    """Connect to user_00042 with a note."""
    failures = 0
    print("== t01_connect_with_user happy path ==")
    reset(client)
    # Find target handle.
    resp = client.get("/in/demo-mantis/")
    if not _check("baseline profile loads", resp.status_code == 200):
        failures += 1
        return failures

    # Look up handle for user_00042 via the seed (predictable: hit /in/?).
    # Cheaper: query state endpoint for the user.
    resp = client.get("/__env__/state", headers=ADMIN_HEADERS)
    _ = resp.json()  # state doesn't directly include user_00042 handle

    # Resolve via DB import — simpler than walking state.
    from app import db as _db
    conn = _db.connect()
    row = conn.execute(
        "SELECT handle FROM users WHERE id = 'user_00042'"
    ).fetchone()
    handle = row["handle"]
    print(f"  target handle: {handle}")

    resp = client.post(
        f"/in/{handle}/connect",
        data={"note": "Hi — would love to connect after PyCon."},
        follow_redirects=False,
    )
    if not _check(
        f"POST /in/{handle}/connect returns 303",
        resp.status_code == 303,
    ):
        failures += 1

    resp = client.get(
        "/__env__/oracle?task_id=t01_connect_with_user",
        headers=ADMIN_HEADERS,
    )
    body = resp.json()
    if not _check(
        f"oracle t01 passed=true (reasons={body.get('reasons')})",
        body.get("passed") is True,
    ):
        failures += 1
    return failures


def smoke_t02(client: TestClient) -> int:
    """Create a feed post with #hashtag."""
    failures = 0
    print("== t02_post_text_update happy path ==")
    reset(client)

    resp = client.post(
        "/feed/post",
        data={"body": "First day on the new team — already learning a ton. "
              "#opensource"},
        follow_redirects=False,
    )
    if not _check(
        "POST /feed/post returns 303",
        resp.status_code == 303,
    ):
        failures += 1

    resp = client.get(
        "/__env__/oracle?task_id=t02_post_text_update",
        headers=ADMIN_HEADERS,
    )
    body = resp.json()
    if not _check(
        f"oracle t02 passed=true (reasons={body.get('reasons')})",
        body.get("passed") is True,
    ):
        failures += 1
    return failures


def smoke_t03(client: TestClient) -> int:
    """Easy Apply to job_00003."""
    failures = 0
    print("== t03_easy_apply_to_job happy path ==")
    reset(client)

    resp = client.post(
        "/jobs/job_00003/apply",
        data={
            "phone": "+1 415 555 0177",
            "resume_label": "Use my profile",
            "answers": '{"years_experience": "5", "work_authorized": "Yes"}',
        },
        follow_redirects=False,
    )
    if not _check(
        "POST /jobs/job_00003/apply returns 303",
        resp.status_code == 303,
    ):
        failures += 1

    resp = client.get(
        "/__env__/oracle?task_id=t03_easy_apply_to_job",
        headers=ADMIN_HEADERS,
    )
    body = resp.json()
    if not _check(
        f"oracle t03 passed=true (reasons={body.get('reasons')})",
        body.get("passed") is True,
    ):
        failures += 1
    return failures


def main() -> int:
    client = TestClient(app)
    # Force startup so DB is seeded before the first call hits a fresh
    # TestClient request-cycle.
    with client:
        failures = 0
        failures += smoke_pages(client)
        failures += smoke_oracle_dispatch(client)
        failures += smoke_t01(client)
        failures += smoke_t02(client)
        failures += smoke_t03(client)
    print()
    if failures:
        print(f"SMOKE FAILED — {failures} check(s) failed")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
