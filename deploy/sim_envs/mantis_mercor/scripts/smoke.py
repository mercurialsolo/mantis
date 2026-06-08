"""mantis-mercor smoke test.

Boots the app in-process via TestClient and drives:

1. Every in-scope page returns 200.
2. /__env__/health returns ok=true (no auth).
3. /__env__/oracle dispatches (without admin token: 401; with: returns
   passing/failing result for each task).
4. End-to-end happy path for each oracle task:
    * t01 — apply to job_00001 as candidate_00001.
    * t02 — shortlist candidate_00007 for job_00001 (as client).
    * t03 — decline app_00001 (as client).
   After each happy-path drive, /__env__/oracle?task_id=tNN must
   return passed=true.

Run:
    cd deploy/sim_envs/mantis_mercor
    ENV_ADMIN_TOKEN=test python scripts/smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python scripts/smoke.py` from the env root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENV_ADMIN_TOKEN", "test")

from fastapi.testclient import TestClient

from app import auth as auth_mod
from app.main import app


client = TestClient(app)
ADMIN = {"X-Env-Admin": "test"}


def _mint_session(user_id: str) -> str:
    return auth_mod.mint_session(user_id)


def _login_as(user_id: str) -> None:
    """Set the session cookie to act as user_id for subsequent requests."""
    client.cookies.set("mantis_mercor_session", _mint_session(user_id))


def _logout() -> None:
    client.cookies.clear()


def _check(name: str, cond: bool, extra: str = "") -> None:
    sym = "PASS" if cond else "FAIL"
    print(f"  [{sym}] {name}{' — ' + extra if extra else ''}")
    if not cond:
        global _failed
        _failed += 1


_failed = 0


def step_health() -> None:
    print("\n# Phase 1: harness contract")
    r = client.get("/__env__/health")
    _check("GET /__env__/health", r.status_code == 200 and r.json()["ok"])

    r = client.post("/__env__/reset")  # no token
    _check("POST /__env__/reset without token → 401", r.status_code == 401)

    r = client.post("/__env__/reset", headers=ADMIN)
    _check("POST /__env__/reset with token → 200", r.status_code == 200)

    r = client.get("/__env__/state", headers=ADMIN)
    _check(
        "GET /__env__/state with counts",
        r.status_code == 200 and r.json()["counts"]["jobs"] > 0,
    )

    r = client.get("/__env__/mutations?since=0", headers=ADMIN)
    _check(
        "GET /__env__/mutations returns list",
        r.status_code == 200 and "mutations" in r.json(),
    )


def step_pages() -> None:
    print("\n# Phase 2: in-scope pages")
    pages = [
        "/", "/jobs", "/jobs/job_00001", "/login", "/signup",
        "/experts", "/dashboard", "/profile",
    ]
    for p in pages:
        r = client.get(p)
        _check(
            f"GET {p} → 200",
            r.status_code == 200,
            extra=f"(got {r.status_code})",
        )
        if p == "/":
            html = r.text
            _check(
                "  home contains 'Shape the frontier of AI'",
                "Shape the frontier of AI" in html,
            )
            _check(
                "  home contains 'Latest roles'",
                "Latest roles" in html,
            )

    # Apply root redirects to step/1.
    r = client.get("/apply/job_00001", follow_redirects=False)
    _check("GET /apply/job_00001 → 303", r.status_code == 303)


def step_t01_apply() -> None:
    print("\n# Phase 3: t01 — apply flow as candidate_00001")
    # Reset env to clean baseline.
    client.post("/__env__/reset", headers=ADMIN)
    _login_as("candidate_00001")

    # Step 1: profile
    r = client.post(
        "/apply/job_00001/step/1",
        data={
            "headline": "Internal Medicine PGY-4",
            "skills": "clinical-reasoning, literature-review",
            "hourly_rate": "150",
            "nav": "next",
        },
        follow_redirects=False,
    )
    _check("apply step 1 → 303", r.status_code == 303)

    r = client.post(
        "/apply/job_00001/step/2",
        data={"resume_text": "5 years general internal medicine.", "nav": "next"},
        follow_redirects=False,
    )
    _check("apply step 2 → 303", r.status_code == 303)

    r = client.post(
        "/apply/job_00001/step/3",
        data={
            "answer_0": "Board eligible.",
            "answer_1": "Managed a rare ARDS case last quarter.",
            "answer_2": "Two weeks notice.",
            "nav": "next",
        },
        follow_redirects=False,
    )
    _check("apply step 3 → 303", r.status_code == 303)

    r = client.post("/apply/job_00001/step/4", data={"nav": "next"},
                    follow_redirects=False)
    _check("apply step 4 → 303", r.status_code == 303)

    r = client.post("/apply/job_00001/step/5", data={"nav": "next"},
                    follow_redirects=False)
    _check("apply step 5 → 303 to /submit", r.status_code == 303)

    r = client.post("/apply/job_00001/submit")
    _check("apply submit → 200", r.status_code == 200)
    _check(
        "submit banner present",
        "Application submitted" in r.text,
    )

    r = client.get("/__env__/oracle?task_id=t01", headers=ADMIN)
    body = r.json()
    _check("t01 oracle passed", body["passed"], extra=str(body.get("reasons")))
    _logout()


def step_t02_shortlist() -> None:
    print("\n# Phase 4: t02 — shortlist candidate_00007 for job_00001")
    client.post("/__env__/reset", headers=ADMIN)
    _login_as("client_00001")

    r = client.post(
        "/dashboard/shortlist",
        data={"job_id": "job_00001", "candidate_id": "candidate_00007"},
        follow_redirects=False,
    )
    _check("POST /dashboard/shortlist → 303", r.status_code == 303)

    r = client.get("/__env__/oracle?task_id=t02", headers=ADMIN)
    body = r.json()
    _check("t02 oracle passed", body["passed"], extra=str(body.get("reasons")))
    _logout()


def step_t03_decline() -> None:
    print("\n# Phase 5: t03 — decline app_00001 with reason")
    client.post("/__env__/reset", headers=ADMIN)
    _login_as("client_00001")

    r = client.post(
        "/dashboard/decline",
        data={"application_id": "app_00001", "reason": "Insufficient experience"},
        follow_redirects=False,
    )
    _check("POST /dashboard/decline → 303", r.status_code == 303)

    r = client.get("/__env__/oracle?task_id=t03", headers=ADMIN)
    body = r.json()
    _check("t03 oracle passed", body["passed"], extra=str(body.get("reasons")))
    _logout()


def step_profile_edit() -> None:
    print("\n# Phase 6: profile edit (no oracle, audit log only)")
    client.post("/__env__/reset", headers=ADMIN)
    _login_as("candidate_00001")

    r = client.post(
        "/profile",
        data={
            "headline": "Senior Internal Med Physician",
            "skills": "clinical-reasoning, dx",
            "hourly_rate": "170",
            "availability": "Project-based",
        },
        follow_redirects=False,
    )
    _check("POST /profile → 303", r.status_code == 303)

    r = client.get("/__env__/mutations?since=0", headers=ADMIN)
    ops = [m["op"] for m in r.json()["mutations"]]
    _check(
        "profile_updated audit row present",
        "profile_updated" in ops,
        extra=f"ops={ops[-5:]}",
    )
    _logout()


def main() -> int:
    step_health()
    step_pages()
    step_t01_apply()
    step_t02_shortlist()
    step_t03_decline()
    step_profile_edit()
    print(f"\n# Result: {_failed} failure(s)")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
