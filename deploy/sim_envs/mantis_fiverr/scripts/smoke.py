"""Smoke-test mantis-fiverr end to end.

Drives the in-process FastAPI app via httpx, exercises each in-scope
page, runs each oracle through its happy path, and asserts the
``/__env__/oracle?task_id=…`` response is passed=true.

Run from the repo root::

    ENV_ADMIN_TOKEN=test python -m deploy.sim_envs.mantis_fiverr.scripts.smoke

or from inside the env dir::

    ENV_ADMIN_TOKEN=test python scripts/smoke.py
"""

from __future__ import annotations

import os
import sys
import pathlib
import re

# Allow ``python scripts/smoke.py`` from inside the env dir.
_HERE = pathlib.Path(__file__).resolve()
_ENV_DIR = _HERE.parent.parent
sys.path.insert(0, str(_ENV_DIR.parent.parent.parent))  # repo root → can import deploy.sim_envs.mantis_fiverr.app


def _ensure_admin_token() -> None:
    os.environ.setdefault("ENV_ADMIN_TOKEN", "smoke-test-token")


def _build_client():
    from fastapi.testclient import TestClient  # type: ignore
    from deploy.sim_envs.mantis_fiverr.app.main import app  # noqa: E402
    return TestClient(app)


def _admin_headers() -> dict[str, str]:
    return {"X-Env-Admin": os.environ["ENV_ADMIN_TOKEN"]}


def _ok(client, url: str) -> None:
    r = client.get(url)
    assert r.status_code == 200, (url, r.status_code, r.text[:200])
    if "html" in r.headers.get("content-type", ""):
        body = r.text
        assert "<html" in body.lower(), (url, "no html body")
    print(f"  ✓ GET {url}  ({r.status_code})")


def main() -> int:
    _ensure_admin_token()
    client = _build_client()

    print("== sanity ==")
    h = client.get("/__env__/health")
    assert h.status_code == 200 and h.json()["ok"] is True, h.text
    print(f"  ✓ /__env__/health  ({h.json()['gigs']} gigs)")

    print("== in-scope pages ==")
    _ok(client, "/")
    _ok(client, "/search/gigs?query=logo")
    _ok(client, "/search/gigs?query=logo&level=top_rated&budget=value&delivery=3d&sort=newest")
    _ok(client, "/categories/graphics-design")
    _ok(client, "/categories/logo-design")
    _ok(client, "/orders")
    _ok(client, "/inbox")
    _ok(client, "/login")
    _ok(client, "/signup")
    _ok(client, "/assets/logo.svg")
    _ok(client, "/assets/category/graphics-design.svg")

    # Resolve canonical gig URL — derived from seed
    conn_meta = client.get(
        "/__env__/state", headers=_admin_headers(),
    ).json()["counts"]
    assert conn_meta["gigs"] >= 1
    # Find seller username + slug for gig_00001 to use canonical URL
    from deploy.sim_envs.mantis_fiverr.app import db as fdb  # type: ignore
    g1 = fdb.connect().execute(
        "SELECT g.slug, u.username FROM gigs g JOIN users u ON g.seller_id = u.id "
        "WHERE g.id = 'gig_00001'"
    ).fetchone()
    gig_url = f"/{g1['username']}/{g1['slug']}"
    _ok(client, gig_url)
    _ok(client, f"/checkout/gig_00001?tier=basic")

    print()
    print("== oracle dispatch (registered + unknown) ==")
    # Unknown task — must still return the canonical shape.
    r = client.get("/__env__/oracle?task_id=unknown_task",
                   headers=_admin_headers())
    body = r.json()
    assert r.status_code == 200
    assert body["passed"] is False and "no oracle registered" in body["reasons"][0]
    print("  ✓ unknown task returns canonical fail shape")

    # Before any agent action — t01..t03 should all fail.
    for tid in ("t01_order_basic_logo", "t02_message_seller_then_order",
                "t03_leave_5star_review"):
        r = client.get(f"/__env__/oracle?task_id={tid}",
                       headers=_admin_headers())
        assert r.status_code == 200
        assert r.json()["passed"] is False
    print("  ✓ all three oracles correctly fail before any agent action")

    print()
    print("== t01 happy path: place a Basic order on gig_00001 ==")
    # POST /checkout/gig_00001 with tier=basic, requirements, payment
    r = client.post(
        "/checkout/gig_00001",
        data={"tier": "basic", "requirements": "Quick turnaround pls",
              "payment_method": "card"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    loc = r.headers["location"]
    assert loc.startswith("/orders/order_"), loc
    print(f"  ✓ POST /checkout/gig_00001 → 303 {loc}")
    _ok(client, loc)
    r = client.get("/__env__/oracle?task_id=t01_order_basic_logo",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"], body
    print(f"  ✓ t01 passed  ({body['diff']})")

    print()
    print("== t02 happy path: send a message, then order ==")
    # Use existing conv_00001 (buyer_00001 ↔ gig_00001 seller)
    convs = fdb.connect().execute(
        "SELECT id FROM conversations WHERE buyer_id='buyer_00001' AND "
        "seller_id=(SELECT seller_id FROM gigs WHERE id='gig_00001')"
    ).fetchall()
    assert convs, "seeded conversation missing"
    conv_id = convs[0]["id"]
    # Reset and re-seed so t01 doesn't pollute t02.
    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200
    r = client.post(
        f"/inbox/{conv_id}/send",
        data={"body": "Hi! I'd love to buy your Basic package."},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    print(f"  ✓ POST /inbox/{conv_id}/send → 303")
    r = client.post(
        "/checkout/gig_00001",
        data={"tier": "basic", "requirements": "",
              "payment_method": "card"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r = client.get("/__env__/oracle?task_id=t02_message_seller_then_order",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"], body
    print(f"  ✓ t02 passed  ({body['diff']})")

    print()
    print("== t03 happy path: leave 5-star review on order_00007 ==")
    r = client.post("/__env__/reset", headers=_admin_headers())
    assert r.status_code == 200
    # order_00007 is seeded as completed.
    r = client.post(
        "/orders/order_00007/review",
        data={"stars": "5", "body": "Stellar work — fast and friendly!"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    print("  ✓ POST /orders/order_00007/review → 303")
    r = client.get("/__env__/oracle?task_id=t03_leave_5star_review",
                   headers=_admin_headers())
    body = r.json()
    assert body["passed"], body
    print(f"  ✓ t03 passed  ({body['diff']})")

    print()
    print("== all smoke checks passed ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
