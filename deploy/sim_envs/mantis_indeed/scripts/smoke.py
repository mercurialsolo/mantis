"""End-to-end smoke test for mantis_indeed.

Uses FastAPI TestClient (Starlette's WSGI-style in-process client) — no
external uvicorn needed, no port to allocate. Verifies:

1. GET / returns 200 + valid HTML.
2. Each in-scope page returns 200.
3. GET /__env__/health returns ok=true (no auth header).
4. Admin endpoints require X-Env-Admin.
5. t01_search_save_remote oracle round-trips: search audit + save → passes.
6. t02_easy_apply oracle round-trips: 4-step Easy Apply → passes.
7. t03_employer_review_applicant oracle round-trips: status change → passes.

Exit code 0 on full pass; 1 on any failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable when run directly.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("ENV_ADMIN_TOKEN", "test-smoke")
os.environ.setdefault("SEED", "42")
os.environ.setdefault("FAKE_NOW", "2026-01-15T09:00:00Z")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN_HEADERS = {"X-Env-Admin": os.environ["ENV_ADMIN_TOKEN"]}


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"OK: {msg}")


def main() -> None:
    client = TestClient(app)

    # 1. health (open)
    r = client.get("/__env__/health")
    must(r.status_code == 200 and r.json().get("ok") is True,
         "/__env__/health returns ok=true")

    # 2. health-gate on admin endpoints
    r = client.get("/__env__/state")
    must(r.status_code == 401, "/__env__/state without admin header → 401")

    r = client.get("/__env__/state", headers=ADMIN_HEADERS)
    must(r.status_code == 200 and "counts" in r.json(),
         "/__env__/state with admin header → 200 + counts")

    # 3. reset to known baseline (re-seed via /__env__/reset)
    r = client.post("/__env__/reset", headers=ADMIN_HEADERS)
    must(r.status_code == 200, "/__env__/reset → 200")

    # 4. home renders
    r = client.get("/")
    must(r.status_code == 200 and "Find your next job" in r.text,
         "GET / renders home")

    # 5. search renders
    r = client.get("/jobs", params={"q": "software engineer", "l": "Austin, TX"})
    must(r.status_code == 200 and "software engineer jobs in Austin, TX" in r.text,
         "GET /jobs renders results with correct H1")

    # 6. detail fragment swap
    # First find an existing jk via state — pull from /jobs html.
    import re
    m = re.search(r'data-jk="([0-9a-f]+)"', r.text)
    must(m is not None, "results page contains a result card with data-jk")
    a_jk = m.group(1)
    r2 = client.get("/jobs/_detail", params={"jk": a_jk})
    must(r2.status_code == 200 and "- job post" in r2.text,
         "GET /jobs/_detail returns detail fragment")

    # 7. viewjob renders
    r3 = client.get("/viewjob", params={"jk": a_jk})
    must(r3.status_code == 200 and "- job post" in r3.text,
         "GET /viewjob renders")

    # 8. myjobs / resumes / employer dashboard
    r4 = client.get("/myjobs")
    must(r4.status_code == 200 and "My jobs" in r4.text, "GET /myjobs renders")
    r5 = client.get("/resumes")
    must(r5.status_code == 200 and "Your resumes" in r5.text, "GET /resumes renders")
    r6 = client.get("/employers/dashboard")
    must(r6.status_code == 200 and "Employer dashboard" in r6.text,
         "GET /employers/dashboard renders")
    r7 = client.get("/login")
    must(r7.status_code == 200 and "Sign in" in r7.text, "GET /login renders")
    r8 = client.get("/signup")
    must(r8.status_code == 200 and "Create your account" in r8.text,
         "GET /signup renders")

    # 9. Oracle dispatch contract — unknown task returns shape, not crash.
    r9 = client.get("/__env__/oracle", params={"task_id": "no_such_task"},
                    headers=ADMIN_HEADERS)
    must(r9.status_code == 200 and "passed" in r9.json(),
         "/__env__/oracle returns shape for unknown task")

    # 10. t01 — search audit + save job_00007 (jk=0000000000000007)
    r = client.post("/__env__/reset", headers=ADMIN_HEADERS)
    must(r.status_code == 200, "reset before t01")
    # Trigger search audit via the dedicated endpoint.
    r = client.get("/jobs/_search_audit",
                   params={"q": "software engineer", "l": "Austin, TX",
                           "remote": "1"})
    must(r.status_code == 200, "search audit submitted")
    # Save job_00007 by jk=0000000000000007 (set in seed.py).
    r = client.post("/jobs/0000000000000007/save")
    must(r.status_code == 200 and r.json().get("saved") is True,
         "POST /jobs/<jk>/save saves the job")
    r = client.get("/__env__/oracle", params={"task_id": "t01_search_save_remote"},
                   headers=ADMIN_HEADERS)
    body = r.json()
    must(r.status_code == 200 and body.get("passed") is True,
         f"t01 oracle passes (reasons={body.get('reasons')})")

    # 11. t02 — Easy Apply to job_00012 (jk=000000000000000c)
    r = client.post("/__env__/reset", headers=ADMIN_HEADERS)
    must(r.status_code == 200, "reset before t02")
    jk_12 = "000000000000000c"
    # Step 1 → step 2
    r = client.post(
        f"/apply/{jk_12}/step2",
        data={"full_name": "Alex Smith", "email": "alex@example.com",
              "phone": "555-1234", "city": "Austin", "state": "TX",
              "zip": "78701"},
        follow_redirects=False,
    )
    must(r.status_code in (303, 307), "step1 → step2 redirect")
    # Step 2 → step 3 (select resume_00001)
    r = client.post(
        f"/apply/{jk_12}/questions",
        data={"resume_id": "resume_00001"},
        follow_redirects=False,
    )
    must(r.status_code in (303, 307), "step2 → step3 redirect")
    # Step 3 → step 4
    r = client.post(
        f"/apply/{jk_12}/review",
        data={"yrs_experience": "5", "auth_to_work": "Yes",
              "sponsorship": "No"},
        follow_redirects=False,
    )
    must(r.status_code in (303, 307), "step3 → review redirect")
    # Submit.
    r = client.post(f"/apply/{jk_12}/submit", follow_redirects=False)
    must(r.status_code in (303, 307), "submit application")
    r = client.get("/__env__/oracle", params={"task_id": "t02_easy_apply"},
                   headers=ADMIN_HEADERS)
    body = r.json()
    must(r.status_code == 200 and body.get("passed") is True,
         f"t02 oracle passes (reasons={body.get('reasons')})")

    # 12. t03 — employer moves the seeded new applicant on job_00003 → reviewed
    r = client.post("/__env__/reset", headers=ADMIN_HEADERS)
    must(r.status_code == 200, "reset before t03")
    r = client.post(
        "/employers/applications/application_seed_t03/status",
        data={"status": "reviewed"},
        follow_redirects=False,
    )
    must(r.status_code in (303, 307), "applicant status change → 303")
    r = client.get("/__env__/oracle",
                   params={"task_id": "t03_employer_review_applicant"},
                   headers=ADMIN_HEADERS)
    body = r.json()
    must(r.status_code == 200 and body.get("passed") is True,
         f"t03 oracle passes (reasons={body.get('reasons')})")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
