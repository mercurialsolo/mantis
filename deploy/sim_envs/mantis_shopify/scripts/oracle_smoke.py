"""Local oracle smoke — drive the env via TestClient, hit happy-path for
each task, then call /__env__/oracle?task_id=tNN and assert each PASSes.

Run: ENV_ADMIN_TOKEN=test python scripts/oracle_smoke.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import create_app

ADMIN = {"X-Env-Admin": os.environ.get("ENV_ADMIN_TOKEN", "test")}


def run() -> int:
    app = create_app()
    failures: list[str] = []
    with TestClient(app) as c:
        # Reset to baseline.
        c.post("/__env__/reset", headers=ADMIN)

        # ── t01: submit Plus lead ──
        c.post("/sales/leads/new", data={
            "product": "plus",
            "merchant_name": "Acme Outerwear",
            "contact_email": "owner@acmeouterwear.example",
            "contact_name": "Acme Owner",
            "notes": "",
        }, follow_redirects=False)

        # ── t02: invite staff ──
        c.post("/team/invite", data={
            "name": "Marlowe Iverson",
            "email": "marlowe@example.com",
            "role": "staff_dev",
        }, follow_redirects=False)

        # ── t03: export payouts CSV ──
        c.get("/payouts/export")

        # ── t04: support ticket ──
        c.post("/support/contact", data={
            "subject": "Cannot access second-factor code",
            "category": "Account access",
            "description": "Lost my phone, need help recovering 2FA on the partner account.",
        }, follow_redirects=False)

        # ── t05: update business_email ──
        c.post("/settings/business", data={
            "business_name": "Mason Partners",
            "website": "https://mason.example",
            "business_email": "new-finance@mason.example",
            "support_email": "support@mason.example",
            "phone": "+1 415 555 0142",
            "address1": "1101 Mission Street",
            "address2": "Suite 400",
            "city": "San Francisco",
            "zip": "94103",
            "state": "California",
            "country": "United States",
        }, follow_redirects=False)

        # ── t06: open a payout detail page ──
        sent = c.get("/__env__/state", headers=ADMIN).json()["counts"]
        assert sent["payouts"] > 1, "seed should have multiple payouts"
        # Pick the first paid payout id.
        c.get("/payouts/payout_00001")

        # ── t07: dismiss emergency banner ──
        c.post("/team/banner/dismiss", follow_redirects=False)

        # ── t08: directory review ──
        c.post("/partner_directory/dirlst_00001/request_review",
               follow_redirects=False)

        # ── t09: search stores ──
        c.get("/stores?tab=all&q=demo&status=")

        # ── t10: submit POS lead ──
        c.post("/sales/leads/new", data={
            "product": "pos",
            "merchant_name": "Brassbird Books",
            "contact_email": "owner@brassbird.example",
            "contact_name": "Brass Owner",
            "notes": "Their in-store flow needs POS Pro.",
        }, follow_redirects=False)

        # ── t11: open a store detail page ──
        c.get("/stores/store_00001")

        # ── t12: open a catalog detail page ──
        c.get("/catalogs/catalog_00001")

        # ── t13: open a merchant order detail from store admin ──
        # Pick the first order for store_00001 from the seed
        admin_orders = c.get("/store/store_00001/admin/orders?tab=all")
        # First merchant order id in the page text
        import re as _re
        m = _re.search(r'href="(/store/store_00001/admin/orders/(mo_\d+))"',
                       admin_orders.text)
        if m:
            order_url, order_id = m.group(1), m.group(2)
            c.get(order_url)
            # ── t14: fulfill that order ──
            c.post(f"/store/store_00001/admin/orders/{order_id}/fulfill",
                   follow_redirects=False)

        # Now grade every oracle.
        for tid in ["t01", "t02", "t03", "t04", "t05",
                    "t06", "t07", "t08", "t09", "t10",
                    "t11", "t12", "t13", "t14"]:
            r = c.get(f"/__env__/oracle?task_id={tid}", headers=ADMIN)
            data = r.json()
            ok = "PASS" if data["passed"] else "FAIL"
            print(f"{ok}  {tid}  {data['reasons'][:1]}")
            if not data["passed"]:
                failures.append(tid)

        # Negative test — unknown oracle
        r = c.get("/__env__/oracle?task_id=t99", headers=ADMIN)
        assert not r.json()["passed"], "unknown oracle must fail"
        print("PASS  t99 (negative)")

    if failures:
        print(f"\n{len(failures)} oracle(s) FAILED: {failures}")
        return 1
    print("\nAll oracles PASS")
    return 0


if __name__ == "__main__":
    sys.exit(run())
