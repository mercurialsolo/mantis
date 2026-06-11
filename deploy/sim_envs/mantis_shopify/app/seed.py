"""Deterministic seed for mantis-shopify.

Same seed → identical DB. Generates:
* 1 partner record (the org)
* ~15 users (2 owners + 13 staff)
* ~12 stores across the three kinds
* 18 payouts spanning 9 months, each with line items
* 20 leads (mix of products + statuses)
* 12 referrals
* 14 directory listings
* 6 themes, 8 catalogs
* 8 changelog items
* 8 notifications
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

FAKE_NOW_DEFAULT = "2026-06-09T09:00:00Z"

CANONICAL_PARTNER_ID = "partner_00001"
CANONICAL_OWNER_ID = "owner_00001"

STORE_NAMES = [
    ("EA demostore", "ea-demostore", "client_transfer", "Plus"),
    ("demo-store-test-07", "demo-store-test-07", "client_transfer", "Grow"),
    ("store-demo-new", "store-demo-new", "client_transfer", "Basic"),
    ("mohit-demo-store-01", "mohit-demo-store-01", "client_transfer", "Grow"),
    ("Test Demostore R", "test-demostore-r", "client_transfer", "Basic"),
    ("for-demo-purposes", "for-demo-purposes", "client_transfer", "Grow"),
    ("MM-shan", "mm-shan", "client_transfer", "Plus"),
    ("Remi", "remi", "client_transfer", "Plus"),
    ("Lifelong Online", "lifelong-online", "collaborator", "Grow"),
    ("Green Heads", "green-heads", "collaborator", "Grow"),
    ("Ratton Pantry", "ratton-pantry", "collaborator", "Grow"),
    ("Salon Bosman", "salon-bosman", "collaborator", "Basic"),
    ("Remi Development Store", "remi-dev", "active_development", "Partner Test"),
    ("Tungsten and Tool", "tungsten-and-tool", "active_development", "Basic"),
    ("CylaStore1", "cylastore1", "active_development", "Custom"),
    ("caresmith-stg", "caresmith-stg", "active_development", "Partner Test"),
    ("auli-stg", "auli-stg", "active_development", "Partner Test"),
    ("mm-jop5", "mm-jop5", "client_transfer", "Plus"),
]

STAFF_NAMES = [
    ("Jophin Joseph", "jophin@example.com", "staff_business"),
    ("Mariam Khan", "mariam@example.com", "staff_dev"),
    ("Devon Park", "devon@example.com", "staff_marketing"),
    ("Sage Wong", "sage@example.com", "staff_support"),
    ("Avery Lin", "avery@example.com", "staff_finance"),
    ("Riley Tan", "riley@example.com", "staff_dev"),
    ("Casey Brooks", "casey@example.com", "staff_business"),
    ("Quinn Davies", "quinn@example.com", "staff_support"),
    ("Morgan Reyes", "morgan@example.com", "staff_marketing"),
    ("Taylor Singh", "taylor@example.com", "staff_dev"),
    ("Jordan Cole", "jordan@example.com", "staff_business"),
    ("Cameron Wu", "cameron@example.com", "staff_finance"),
    ("Hayden Choi", "hayden@example.com", "staff_dev"),
]

LEAD_MERCHANTS = [
    ("Pacific Outerwear", "ops@pacificouterwear.example"),
    ("Hearth & Hollow Furniture", "owner@hearthandhollow.example"),
    ("Tundra Coffee Co.", "buyer@tundracoffee.example"),
    ("Cedar & Pine Apothecary", "hello@cedarpine.example"),
    ("Northstar Athletics", "store@northstarath.example"),
    ("Stratos Audio", "team@stratosaudio.example"),
    ("Maple Lane Bakery", "info@maplelane.example"),
    ("Verdant Garden Supply", "garden@verdant.example"),
    ("Coastal Stoneware", "sales@coastalstoneware.example"),
    ("Brassbird Books", "shop@brassbird.example"),
    ("Hollowfield Distillery", "trade@hollowfield.example"),
    ("Lichen + Loam", "hello@lichenloam.example"),
    ("Slate Atelier", "studio@slateatelier.example"),
    ("Driftwood Surf Co.", "shop@driftwoodsurf.example"),
    ("Westwood Print Studio", "press@westwoodprint.example"),
    ("Almond Branch Co.", "owner@almondbranch.example"),
    ("Bramble Mercantile", "store@bramble.example"),
    ("Half-Moon Records", "music@halfmoon.example"),
    ("Iron Owl Bakery", "owner@ironowl.example"),
    ("Quill & Caraway", "hello@quillcaraway.example"),
]

DIRECTORY_PLANS = [
    ("MM-shan", "Plus"),
    ("Remi", "Plus"),
    ("Green Heads", "Grow"),
    ("Lifelong Online", "Grow"),
    ("Ratton Pantry", "Grow"),
    ("Salon Bosman", "Basic"),
    ("Remi Development Store", "Partner Test"),
    ("Tungsten and Tool", "Basic"),
    ("CylaStore1", "Custom"),
    ("EA Demostore", "Plus"),
    ("Northstar Athletics", "Grow"),
    ("Brassbird Books", "Basic"),
    ("Slate Atelier", "Custom"),
    ("Maple Lane Bakery", "Grow"),
]

CHANGELOG_ITEMS = [
    ("Catalogs", "Improved catalog publishing UX",
     "You can now stage and publish catalogs to multiple merchant stores from one panel."),
    ("Admin", "New permission scopes for staff roles",
     "Granular scopes let owners restrict staff access to specific surfaces."),
    ("Payments", "Local payment methods added in 4 new regions",
     "Klarna, iDEAL, and BLIK now appear in eligible checkouts."),
    ("Analytics", "Lead-to-revenue funnel chart in Sales",
     "Track which leads convert to active paying merchants over time."),
    ("POS",     "Shopify POS verified-skills assessment v3 launched",
     "Refreshed coverage of high-volume retail flows and POS Pro reporting."),
    ("Admin",   "Bulk invite for staff members",
     "Invite up to 20 staff members from a CSV upload."),
    ("Themes",  "New skeleton theme dev tooling",
     "Hot reload and section auto-population now work with the skeleton theme."),
    ("Payments","Pending payout summary moved to Home",
     "Estimated payout amount and ETA are now shown on the Home dashboard."),
]


def _hash(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def _iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _human_date(d: datetime) -> str:
    return d.strftime("%b %-d, %Y") if hasattr(d, "strftime") else str(d)


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str) -> None:
    """Re-seed the DB to a deterministic baseline."""
    rng = random.Random(seed_val)

    # Wipe tables in FK-safe order.
    for t in [
        "audit_log", "notifications", "changelog", "catalogs", "themes",
        "tickets", "directory_listings", "referrals", "leads",
        "merchant_customers", "merchant_products", "merchant_orders",
        "payout_line_items", "payouts", "stores", "users", "partners",
    ]:
        conn.execute(f"DELETE FROM {t}")

    now_dt = datetime.fromisoformat(fake_now.replace("Z", "+00:00"))

    # ── partners (single row) ───────────────────────────────────
    conn.execute(
        "INSERT INTO partners (id, partner_id, business_name, website, "
        "business_email, support_email, phone, address1, address2, city, "
        "zip, state, country, emergency_name, emergency_email, "
        "emergency_phone, payout_method, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            CANONICAL_PARTNER_ID,
            "1146365",
            "Mason Partners",
            "https://mason.example",
            "partners@mason.example",
            "support@mason.example",
            "+1 415 555 0142",
            "1101 Mission Street",
            "Suite 400",
            "San Francisco",
            "94103",
            "California",
            "United States",
            "Operations On-Call",
            "oncall@mason.example",
            "+1 415 555 0188",
            "bank account (***05)",
            fake_now,
        ),
    )

    # ── users (owners + staff) ──────────────────────────────────
    pw = _hash("password")
    owner_rows = [
        (CANONICAL_OWNER_ID, "barada@example.com", pw, "Barada Sahu",
         "owner", "active", _iso_date(now_dt - timedelta(hours=1)), "#5c6ac4"),
        ("owner_00002", "kay@example.com", pw, "Kay Mann", "owner", "active",
         _iso_date(now_dt - timedelta(days=120)), "#bf0711"),
    ]
    for r in owner_rows:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, role, status, "
            "last_login_at, avatar_color, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            r + (fake_now,),
        )

    for i, (nm, em, role) in enumerate(STAFF_NAMES, start=1):
        last_login = _iso_date(
            now_dt - timedelta(days=rng.randint(0, 540))
        )
        status = "active" if i <= 11 else "invited"
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, role, status, "
            "last_login_at, avatar_color, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                f"staff_{i:05d}", em, pw, nm, role, status,
                last_login,
                rng.choice(["#5c6ac4", "#bf0711", "#108043", "#9c6ade",
                            "#47c1bf", "#de3618", "#bf6900"]),
                fake_now,
            ),
        )

    # ── stores ──────────────────────────────────────────────────
    for i, (nm, slug, kind, plan) in enumerate(STORE_NAMES, start=1):
        days_ago = rng.randint(0, 400)
        conn.execute(
            "INSERT INTO stores (id, name, slug, kind, status, last_login_at, plan) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                f"store_{i:05d}", nm, slug, kind,
                "active",
                _iso_date(now_dt - timedelta(days=days_ago)),
                plan,
            ),
        )

    # ── payouts (last 9 months, biweekly) ───────────────────────
    payouts: list[tuple[str, str, str, str, int, str, str, str]] = []
    # Pending payout (most recent, unsent)
    pending_start = now_dt - timedelta(days=8)
    pending_end = now_dt + timedelta(days=5)
    payouts.append((
        "payout_pending",
        _iso_date(pending_start),
        _iso_date(pending_end),
        "",
        215143,
        "USD",
        "bank account (***05)",
        "pending",
    ))

    cursor = now_dt - timedelta(days=15)
    for i in range(1, 19):
        period_end = cursor
        period_start = cursor - timedelta(days=15)
        sent = period_end + timedelta(days=5)
        amt = rng.randint(85000, 260000)
        payouts.append((
            f"payout_{i:05d}",
            _iso_date(period_start),
            _iso_date(period_end),
            _iso_date(sent),
            amt,
            "USD",
            "bank account (***05)",
            "paid",
        ))
        cursor = period_start
    for p in payouts:
        conn.execute(
            "INSERT INTO payouts (id, period_start, period_end, sent_at, "
            "amount_cents, currency, method, status) VALUES (?,?,?,?,?,?,?,?)",
            p,
        )

    # Line items for each payout.
    li_counter = 1
    for payout_id, ps, pe, _sent, amt, _ccy, _method, _status in payouts:
        n_items = rng.randint(3, 7)
        remaining = amt
        for j in range(n_items):
            if j == n_items - 1:
                slice_amt = remaining
            else:
                slice_amt = rng.randint(2000, max(3000, remaining // max(1, n_items - j)))
                slice_amt = min(slice_amt, max(2000, remaining - 1000))
            remaining -= slice_amt
            occ = _iso_date(
                datetime.fromisoformat(ps.replace("Z", "+00:00"))
                + timedelta(days=rng.randint(0, 14))
            )
            kind = rng.choice([
                "Plus referral — recurring",
                "POS Pro referral — recurring",
                "Plus referral — one-time",
                "Theme commission",
                "App distribution",
            ])
            conn.execute(
                "INSERT INTO payout_line_items (id, payout_id, occurred_at, "
                "description, amount_cents) VALUES (?,?,?,?,?)",
                (f"pli_{li_counter:06d}", payout_id, occ, kind, slice_amt),
            )
            li_counter += 1

    # ── leads ───────────────────────────────────────────────────
    products = ["plus", "pos", "plus_b2b"]
    statuses = ["submitted", "qualified", "won", "lost"]
    for i, (merchant, email) in enumerate(LEAD_MERCHANTS, start=1):
        product = rng.choice(products)
        status = rng.choice(statuses)
        earnings = 0
        if status == "won":
            earnings = rng.choice([250000, 250000, 280000, 320000])
        days_ago = rng.randint(0, 270)
        conn.execute(
            "INSERT INTO leads (id, product, merchant_name, contact_email, "
            "contact_name, status, earnings_cents, submitted_at, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                f"lead_{i:05d}", product, merchant, email,
                merchant.split()[0] + " Owner",
                status, earnings,
                _iso_date(now_dt - timedelta(days=days_ago)),
                "Referred via the Partner Directory listing.",
            ),
        )

    # ── referrals ───────────────────────────────────────────────
    for i in range(1, 13):
        merchant = LEAD_MERCHANTS[i % len(LEAD_MERCHANTS)][0]
        plan = rng.choice(["Plus", "Grow", "Basic"])
        conn.execute(
            "INSERT INTO referrals (id, merchant_name, plan, status, "
            "earnings_cents, created_at) VALUES (?,?,?,?,?,?)",
            (
                f"referral_{i:05d}", merchant + f" #{i:02d}", plan,
                rng.choice(["active", "active", "active", "churned"]),
                rng.randint(15000, 92000),
                _iso_date(now_dt - timedelta(days=rng.randint(30, 540))),
            ),
        )

    # ── directory listings ──────────────────────────────────────
    for i, (biz, plan) in enumerate(DIRECTORY_PLANS, start=1):
        conn.execute(
            "INSERT INTO directory_listings (id, business_name, plan, "
            "review_status) VALUES (?,?,?,?)",
            (f"dirlst_{i:05d}", biz, plan, "available"),
        )

    # ── themes & catalogs ───────────────────────────────────────
    theme_names = [
        ("Skeleton", "approved"),
        ("Slate Studio", "in_review"),
        ("Linen", "approved"),
        ("Pinecone", "draft"),
        ("Marble", "approved"),
        ("Wave", "rejected"),
    ]
    for i, (nm, st) in enumerate(theme_names, start=1):
        conn.execute(
            "INSERT INTO themes (id, name, status, created_at) VALUES (?,?,?,?)",
            (f"theme_{i:05d}", nm, st,
             _iso_date(now_dt - timedelta(days=rng.randint(30, 400)))),
        )

    catalog_names = [
        ("Spring Wholesale", 124, "published"),
        ("Holiday 2025", 86, "published"),
        ("Outdoor Living", 52, "published"),
        ("Limited Edition Drops", 14, "draft"),
        ("Restaurant Supplies", 210, "published"),
        ("Studio Essentials", 38, "draft"),
        ("Wedding Collection", 73, "published"),
        ("Back-to-School", 19, "draft"),
    ]
    for i, (nm, pc, st) in enumerate(catalog_names, start=1):
        conn.execute(
            "INSERT INTO catalogs (id, name, products_count, status, "
            "created_at) VALUES (?,?,?,?,?)",
            (f"catalog_{i:05d}", nm, pc, st,
             _iso_date(now_dt - timedelta(days=rng.randint(15, 220)))),
        )

    # ── changelog ───────────────────────────────────────────────
    for i, (cat, title, summary) in enumerate(CHANGELOG_ITEMS, start=1):
        published_at = _iso_date(now_dt - timedelta(days=i * 4))
        conn.execute(
            "INSERT INTO changelog (id, published_at, category, title, summary) "
            "VALUES (?,?,?,?,?)",
            (f"cl_{i:05d}", published_at, cat, title, summary),
        )

    # ── per-store merchant data ────────────────────────────────
    _seed_merchant_data(conn, rng=rng, now_dt=now_dt)

    # ── notifications ──────────────────────────────────────────
    notif_titles = [
        ("Pending payout updated", "Your next payout will arrive Jun 22, 2026."),
        ("New lead status", "Pacific Outerwear marked qualified."),
        ("Staff invite accepted", "Quinn Davies joined as Support staff."),
        ("Theme submitted for review", "Slate Studio is awaiting review."),
        ("Catalog published", "Spring Wholesale was published to 6 stores."),
        ("New community forum post", "Webhook signature regen — questions."),
        ("API change notice", "Admin REST endpoints have new rate limits."),
        ("Partner Directory request received",
         "Lifelong Online requested a review."),
    ]
    for i, (t, b) in enumerate(notif_titles, start=1):
        conn.execute(
            "INSERT INTO notifications (id, occurred_at, title, body, read) "
            "VALUES (?,?,?,?,?)",
            (
                f"notif_{i:05d}",
                _iso_date(now_dt - timedelta(hours=rng.randint(1, 96))),
                t, b, 0,
            ),
        )

    conn.commit()


# ── Per-store merchant data ─────────────────────────────────────

_PRODUCT_CATALOG = [
    ("Hand-poured candle, juniper", "Bramble & Co", "Home goods", 24),
    ("Linen apron, oat", "Bramble & Co", "Apparel", 36),
    ("Brass desk lamp, matte", "Northwood Studio", "Lighting", 142),
    ("Riverstone tumbler set (4)", "Northwood Studio", "Glassware", 58),
    ("Felted wool throw, sage", "Almond Branch", "Home goods", 145),
    ("Ceramic pour-over kit", "Almond Branch", "Kitchen", 42),
    ("Cedar cutting board, large", "Slate Atelier", "Kitchen", 64),
    ("Walnut wall mirror, round", "Slate Atelier", "Home decor", 198),
    ("Hemlock incense bundle", "Cedar Pine", "Wellness", 18),
    ("Stoneware mug, slate", "Cedar Pine", "Drinkware", 22),
    ("Aged-cotton tote, indigo", "Drift Mill", "Bags", 34),
    ("Marble cheese board", "Drift Mill", "Kitchen", 76),
    ("Embossed leather wallet", "Half-Moon Goods", "Accessories", 88),
    ("Speckled clay vase, large", "Half-Moon Goods", "Home decor", 56),
]

_CUSTOMER_NAMES = [
    ("Lena Park", "lenapark", "Brooklyn, NY"),
    ("Yusuf Aksoy", "yaksoy", "Istanbul, TR"),
    ("Nora Lindgren", "nlindgren", "Malmö, SE"),
    ("Tomás Vega", "tomasv", "CDMX, MX"),
    ("Aliyah Brooks", "aliyahb", "Atlanta, GA"),
    ("Kenji Iwata", "kiwata", "Osaka, JP"),
    ("Mira Patel", "mpatel", "Bengaluru, IN"),
    ("Hugo Schmitt", "hschmitt", "Berlin, DE"),
    ("Saoirse Walsh", "swalsh", "Galway, IE"),
    ("Esther Mwangi", "emwangi", "Nairobi, KE"),
    ("Priya Lal", "plal", "Mumbai, IN"),
    ("Mateusz Nowak", "mnowak", "Warsaw, PL"),
]

_DELIVERY = ["Standard", "Express", "Local pickup", "Same-day"]
_FIN_STATUS = ["paid", "paid", "paid", "paid", "pending", "refunded"]
_FUL_STATUS = ["unfulfilled", "unfulfilled", "fulfilled", "fulfilled", "partial"]


def _seed_merchant_data(conn, *, rng, now_dt) -> None:
    """Populate merchant_orders/products/customers for every store."""
    from datetime import timedelta
    import json as _json

    stores = [dict(r) for r in conn.execute(
        "SELECT id, name, slug FROM stores ORDER BY id"
    ).fetchall()]

    order_counter = 1
    product_counter = 1
    customer_counter = 1

    for store in stores:
        sid = store["id"]

        # ── Products: 8-12 per store, deterministic per store id ──
        n_prods = 8 + (int(sid.split("_")[-1] or 0) % 5)
        seen_titles = set()
        for i in range(n_prods):
            base = _PRODUCT_CATALOG[(int(sid.split('_')[-1] or 0) * 3 + i)
                                    % len(_PRODUCT_CATALOG)]
            title = base[0]
            if title in seen_titles:
                title = f"{title} · v{i+2}"
            seen_titles.add(title)
            status = rng.choice(
                ["active", "active", "active", "draft", "archived"]
            )
            conn.execute(
                "INSERT INTO merchant_products (id, store_id, title, status, "
                "inventory, vendor, product_type, price_cents, sku, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"mp_{product_counter:06d}", sid, title, status,
                    rng.randint(0, 320), base[1], base[2],
                    base[3] * 100,
                    f"SKU-{product_counter:06d}",
                    _iso_date(now_dt - timedelta(days=rng.randint(20, 400))),
                ),
            )
            product_counter += 1

        # ── Customers: 6-9 per store ──
        n_custs = 6 + (int(sid.split("_")[-1] or 0) % 4)
        cust_ids_for_store = []
        for i in range(n_custs):
            base = _CUSTOMER_NAMES[(int(sid.split('_')[-1] or 0) * 2 + i)
                                   % len(_CUSTOMER_NAMES)]
            orders_ct = rng.randint(1, 9)
            spent = orders_ct * rng.randint(2200, 18000)
            cid = f"mc_{customer_counter:06d}"
            conn.execute(
                "INSERT INTO merchant_customers (id, store_id, name, email, "
                "orders_count, total_spent_cents, location, last_order_at, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    cid, sid, base[0],
                    f"{base[1]}@example.com",
                    orders_ct, spent, base[2],
                    _iso_date(now_dt - timedelta(days=rng.randint(1, 60))),
                    _iso_date(now_dt - timedelta(days=rng.randint(60, 700))),
                ),
            )
            cust_ids_for_store.append((cid, base[0], f"{base[1]}@example.com"))
            customer_counter += 1

        # ── Orders: 10-14 per store ──
        n_orders = 10 + (int(sid.split("_")[-1] or 0) % 5)
        for i in range(n_orders):
            cust = rng.choice(cust_ids_for_store)
            items_n = rng.randint(1, 4)
            items = []
            total = 0
            for j in range(items_n):
                p = _PRODUCT_CATALOG[(i + j + int(sid.split('_')[-1] or 0))
                                     % len(_PRODUCT_CATALOG)]
                qty = rng.randint(1, 3)
                total += p[3] * 100 * qty
                items.append({"title": p[0], "qty": qty, "price_cents": p[3] * 100})
            conn.execute(
                "INSERT INTO merchant_orders (id, store_id, order_number, "
                "customer_name, customer_email, total_cents, currency, "
                "financial_status, fulfillment_status, items_count, items_json, "
                "ordered_at, delivery_method) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"mo_{order_counter:06d}", sid,
                    f"#{1000 + order_counter}",
                    cust[1], cust[2], total, "USD",
                    rng.choice(_FIN_STATUS), rng.choice(_FUL_STATUS),
                    items_n, _json.dumps(items),
                    _iso_date(now_dt - timedelta(days=rng.randint(0, 90),
                                                  hours=rng.randint(0, 23))),
                    rng.choice(_DELIVERY),
                ),
            )
            order_counter += 1

    conn.commit()
