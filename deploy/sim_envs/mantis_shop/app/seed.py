"""Deterministic seed for mantis-shop.

Given ``SEED``, this module deterministically produces:

* 3,000 customers (+ saved addresses + mocked payment methods)
* 2,000 products (with size/color variant matrix → ~12,000 variants)
* 40 collections (rule + manual)
* 15 coupons (pct / amount / BOGO)
* 5,000 historical orders
* 1 known saved view (``saved_view_bogo_recent``) for T04
* 1 known Brooklyn address on the canonical T01 customer

Same SEED → identical row ids + columns. Tested in
``tests/sim_envs/mantis_shop/test_seed_determinism.py``.

Pinned realities (test acceptance depends on these)
---------------------------------------------------

* ``product_00001`` — "Heritage Field Jacket" in category
  ``outerwear-women``, base_price $89.00 (under $100), 6 variants
  (S/M/L × Blue/Black). The Size-M Blue variant sku is
  ``JACKET-001-M-BLUE`` and is in-stock, not region-locked for Brooklyn,
  NY 11201.
* ``customer_00001`` — "Test Customer" with one Brooklyn address
  ``addr_brooklyn_001`` (line "123 Front St", "Brooklyn", "NY", "11201")
  flagged default. T01 ships here.
* Order ``#4421`` (id ``order_04421``) — has at least 3 line items so
  "line item 2" is unambiguous; pre-existing status ``paid``.
* Coupon ``SPRING15`` — 15% off, stackable=1, no exclusions, not
  expired.
* Coupon ``BOGO`` — buy-one-get-one (kind=bogo, value=1.0). 5 historical
  orders within the last 7 days used it (deterministic ids).
* Coupon pair ``STACK_TRAP_A`` + ``STACK_TRAP_B`` — both have
  ``stackable=1`` (UI claim) but the server's stacking_exclusions list
  rejects them when applied together. The spec calls this out: "2
  coupons that look stackable in the UI but the server rejects together".
* Variant sku ``TEE-BLK-M`` exists (from ``product_00002`` = "Core
  Cotton Tee"). T05 references it directly.

Volumes are tuned to <30s cold boot.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

# ── volumes ───────────────────────────────────────────────────────────

N_CUSTOMERS = 3_000
N_PRODUCTS = 2_000
N_COLLECTIONS = 40
N_COUPONS = 15
N_ORDERS = 5_000

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"

# ── dictionaries (small, deterministic) ───────────────────────────────

_CATEGORIES = [
    "outerwear-women", "outerwear-men", "tees", "denim", "sneakers",
    "accessories", "dresses", "pants-women", "pants-men", "knitwear",
    "swim", "activewear",
]
_CATEGORY_SIZES = {
    "outerwear-women": ["S", "M", "L"],
    "outerwear-men": ["S", "M", "L", "XL"],
    "tees": ["S", "M", "L", "XL"],
    "denim": ["28", "30", "32", "34", "36"],
    "sneakers": ["8", "9", "10", "11", "12"],
    "accessories": ["one-size"],
    "dresses": ["XS", "S", "M", "L"],
    "pants-women": ["XS", "S", "M", "L"],
    "pants-men": ["S", "M", "L", "XL"],
    "knitwear": ["S", "M", "L"],
    "swim": ["S", "M", "L"],
    "activewear": ["S", "M", "L"],
}
_COLOR_PALETTE = ["BLACK", "WHITE", "BLUE", "RED", "GREEN", "GREY", "NAVY", "OLIVE"]

_TITLE_PARTS_A = [
    "Heritage", "Core", "Modern", "Field", "Studio", "Atlas", "Frontier",
    "Cascade", "Driftwood", "Harbor", "Linden", "Meridian", "Pioneer",
    "Rugged", "Solstice", "Tideline", "Voyager", "Coastal",
]
_TITLE_PARTS_B = {
    "outerwear-women": "Jacket",
    "outerwear-men": "Coat",
    "tees": "Tee",
    "denim": "Jean",
    "sneakers": "Runner",
    "accessories": "Cap",
    "dresses": "Dress",
    "pants-women": "Trouser",
    "pants-men": "Chino",
    "knitwear": "Sweater",
    "swim": "Trunks",
    "activewear": "Short",
}

_US_REGIONS = ["NY", "CA", "TX", "IL", "WA", "MA", "FL", "PA", "OR", "CO"]
_CITY_BY_REGION = {
    "NY": "New York", "CA": "Los Angeles", "TX": "Austin", "IL": "Chicago",
    "WA": "Seattle", "MA": "Boston", "FL": "Miami", "PA": "Philadelphia",
    "OR": "Portland", "CO": "Denver",
}
_PAYMENT_BRANDS = ["visa", "mc", "amex"]

# Order status mix for historical orders
_STATUS_MIX = [
    ("paid", 0.45), ("fulfilled", 0.40), ("refunded", 0.10), ("cancelled", 0.05),
]


# ── helpers ───────────────────────────────────────────────────────────


def _id(prefix: str, idx: int, *, width: int = 5) -> str:
    return f"{prefix}_{idx:0{width}d}"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    if value and value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


def _pick_status(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for s, p in _STATUS_MIX:
        cum += p
        if r < cum:
            return s
    return _STATUS_MIX[-1][0]


# ── seed entrypoint ───────────────────────────────────────────────────


def seed(conn: sqlite3.Connection, *, seed_val: int,
         fake_now: str = FAKE_NOW_DEFAULT) -> None:
    """Populate ``conn`` deterministically from ``seed_val``.

    The function clears every table first, then re-inserts. Callers can
    simply call ``seed`` again to reset.
    """
    rng = random.Random(seed_val)
    now_dt = _parse_iso(fake_now)

    cur = conn.cursor()
    for table in (
        "audit_log", "saved_view_members", "saved_views",
        "order_refunds", "order_items", "orders",
        "cart_items", "carts",
        "coupons",
        "collection_members", "collections",
        "variant_region_locks", "inventory", "variants", "products",
        "customer_payment_methods", "customer_addresses", "customers",
    ):
        cur.execute(f"DELETE FROM {table}")

    _seed_customers(cur, rng)
    _seed_products(cur, rng)
    _seed_collections(cur, rng)
    _seed_coupons(cur, rng, now_dt)
    _seed_orders(cur, rng, now_dt)
    _seed_saved_views(cur)

    conn.commit()


# ── customers + addresses + payment methods ───────────────────────────


def _seed_customers(cur: sqlite3.Cursor, rng: random.Random) -> None:
    """3,000 customers + saved addresses + mocked payment methods.

    Customer 1 is pinned as the canonical T01 buyer with one Brooklyn
    address flagged default.
    """
    customer_rows: list[tuple[Any, ...]] = []
    address_rows: list[tuple[Any, ...]] = []
    payment_rows: list[tuple[Any, ...]] = []

    # Pinned T01 customer with the Brooklyn address.
    customer_rows.append((
        "customer_00001",
        "Test Customer",
        "test.customer@mantis-shop.test",
    ))
    address_rows.append((
        "addr_brooklyn_001", "customer_00001",
        "123 Front St", "Brooklyn", "NY", "11201", 1,
    ))
    # Give them an LA fallback so the test can also exercise region-locked
    # variant rejection on the Brooklyn-default address.
    address_rows.append((
        "addr_la_001", "customer_00001",
        "500 Sunset Blvd", "Los Angeles", "CA", "90028", 0,
    ))
    payment_rows.append((
        "pay_00001", "customer_00001", "visa", "4242", "Personal Visa",
    ))

    # The other 2,999 customers — each with 1-2 addresses + 1 payment method.
    for i in range(2, N_CUSTOMERS + 1):
        name = f"Customer {i:05d}"
        email = f"customer{i:05d}@mantis-shop.test"
        customer_rows.append((_id("customer", i), name, email))

        # Primary address — region picked from the deterministic palette.
        region = rng.choice(_US_REGIONS)
        city = _CITY_BY_REGION[region]
        zip_code = f"{rng.randint(10000, 99999)}"
        address_rows.append((
            _id("addr", i * 2), _id("customer", i),
            f"{rng.randint(10, 9999)} Main St",
            city, region, zip_code, 1,
        ))
        if rng.random() < 0.3:
            alt_region = rng.choice(_US_REGIONS)
            address_rows.append((
                _id("addr", i * 2 + 1), _id("customer", i),
                f"{rng.randint(10, 9999)} Park Ave",
                _CITY_BY_REGION[alt_region], alt_region,
                f"{rng.randint(10000, 99999)}", 0,
            ))
        payment_rows.append((
            _id("pay", i), _id("customer", i),
            rng.choice(_PAYMENT_BRANDS),
            f"{rng.randint(1000, 9999)}",
            "Saved card",
        ))

    cur.executemany(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        customer_rows,
    )
    cur.executemany(
        "INSERT INTO customer_addresses "
        "(id, customer_id, line1, city, region, zip, is_default) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        address_rows,
    )
    cur.executemany(
        "INSERT INTO customer_payment_methods "
        "(id, customer_id, brand, last4, label) VALUES (?, ?, ?, ?, ?)",
        payment_rows,
    )


# ── products + variants + inventory ───────────────────────────────────


def _seed_products(cur: sqlite3.Cursor, rng: random.Random) -> None:
    """2,000 products with deterministic title + category + variants.

    Pins:

    * ``product_00001`` = "Heritage Field Jacket" in ``outerwear-women``,
      base_price $89.00 (under $100). Has variants S/M/L × BLUE/BLACK.
      The Size-M Blue variant (sku ``JACKET-001-M-BLUE``) is stocked +
      not region-locked anywhere.
    * ``product_00002`` = "Core Cotton Tee" in ``tees``. Has variants
      including ``TEE-BLK-M`` (size M, color BLACK).
    """
    product_rows: list[tuple[Any, ...]] = []
    variant_rows: list[tuple[Any, ...]] = []
    inventory_rows: list[tuple[Any, ...]] = []
    region_lock_rows: list[tuple[Any, ...]] = []

    # 10% of products have a broken image_url (NULL); ~30% on sale; ~5%
    # of variants out of stock; ~3% region-locked.
    n_broken_image = int(N_PRODUCTS * 0.005)  # 10 products per spec
    broken_image_set = set(rng.sample(range(3, N_PRODUCTS + 1), n_broken_image))

    def _add_variant(product_id: str, sku: str, size: str, color: str,
                     *, quantity: int, locked_regions: list[str]) -> str:
        variant_id = f"variant_{product_id[8:]}_{size}_{color}"
        variant_rows.append((
            variant_id, product_id, sku, size, color, None,
        ))
        inventory_rows.append((variant_id, quantity))
        for r in locked_regions:
            region_lock_rows.append((variant_id, r))
        return variant_id

    # — Pinned product_00001 — Heritage Field Jacket — outerwear-women —
    product_rows.append((
        "product_00001",
        "JACKET-001",
        "Heritage Field Jacket",
        "A timeless field jacket in waxed cotton. Two chest pockets, "
        "snap closure, and a cinch waist.",
        "outerwear-women",
        89.00,           # base_price < $100
        None,            # not on sale (keeps math simple in oracle)
        "/static/img/heritage-field-jacket.jpg",
        4.6,
    ))
    for size in ("S", "M", "L"):
        for color in ("BLUE", "BLACK"):
            sku = f"JACKET-001-{size}-{color}"
            # Make the M-BLUE always in stock, unlocked, regardless of seed
            qty = 25 if (size == "M" and color == "BLUE") else rng.randint(0, 30)
            _add_variant("product_00001", sku, size, color,
                         quantity=qty, locked_regions=[])

    # — Pinned product_00002 — Core Cotton Tee — tees —
    product_rows.append((
        "product_00002",
        "TEE-001",
        "Core Cotton Tee",
        "Mid-weight cotton tee with a relaxed fit. Pre-washed for "
        "minimal shrinkage.",
        "tees",
        24.00,
        None,
        "/static/img/core-cotton-tee.jpg",
        4.4,
    ))
    # Variants for the tee — include the canonical TEE-BLK-M for T05.
    for size in ("S", "M", "L", "XL"):
        for color in ("BLACK", "WHITE", "NAVY"):
            short_color = {"BLACK": "BLK", "WHITE": "WHT", "NAVY": "NVY"}[color]
            sku = f"TEE-{short_color}-{size}"
            # TEE-BLK-M starts at exactly 100 so the +50 oracle is unambiguous.
            qty = 100 if sku == "TEE-BLK-M" else rng.randint(5, 60)
            _add_variant("product_00002", sku, size, color,
                         quantity=qty, locked_regions=[])

    # — The other 1,998 products —
    for i in range(3, N_PRODUCTS + 1):
        product_id = _id("product", i)
        category = rng.choice(_CATEGORIES)
        a = rng.choice(_TITLE_PARTS_A)
        b = _TITLE_PARTS_B[category]
        title = f"{a} {b} #{i:04d}"
        base = round(rng.uniform(12.0, 350.0), 2)
        on_sale = rng.random() < 0.30
        sale = round(base * rng.uniform(0.5, 0.85), 2) if on_sale else None
        image = None if i in broken_image_set else f"/static/img/product-{i:04d}.jpg"
        sku_base = f"P{i:05d}"
        rating = round(rng.uniform(3.2, 5.0), 1)
        product_rows.append((
            product_id, sku_base, title,
            f"{title} — auto-generated description for seed determinism.",
            category, base, sale, image, rating,
        ))

        sizes = _CATEGORY_SIZES[category]
        colors = rng.sample(_COLOR_PALETTE, k=rng.randint(1, 3))
        for size in sizes:
            for color in colors:
                sku = f"{sku_base}-{color}-{size}"
                # 5% of variants out of stock; 3% region-locked.
                quantity = 0 if rng.random() < 0.05 else rng.randint(1, 50)
                locked = []
                if rng.random() < 0.03:
                    locked = [rng.choice(_US_REGIONS)]
                _add_variant(product_id, sku, size, color,
                             quantity=quantity, locked_regions=locked)

    cur.executemany(
        "INSERT INTO products (id, sku, title, description_md, category, "
        "base_price, sale_price, image_url, rating) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        product_rows,
    )
    cur.executemany(
        "INSERT INTO variants (id, product_id, sku, size, color, price_override) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        variant_rows,
    )
    cur.executemany(
        "INSERT INTO inventory (variant_id, quantity) VALUES (?, ?)",
        inventory_rows,
    )
    cur.executemany(
        "INSERT INTO variant_region_locks (variant_id, region) VALUES (?, ?)",
        region_lock_rows,
    )


# ── collections ───────────────────────────────────────────────────────


def _seed_collections(cur: sqlite3.Cursor, rng: random.Random) -> None:
    """40 collections — mix of rule-based ('on-sale', category-based) +
    manual member lists.

    Pin: ``outerwear-women`` collection slug must exist for T03 (a
    coupon creator targets that category — the create flow's category
    dropdown reads collection slugs)."""
    rows: list[tuple[Any, ...]] = []
    # 20 category collections + a few sale collections + 18 manual ones.
    for i, cat in enumerate(_CATEGORIES, start=1):
        rows.append((
            _id("collection", i, width=3),
            cat,
            cat.replace("-", " ").title(),
            "rule",
            json.dumps({"category": cat}, sort_keys=True),
        ))
    # "On sale" rule collection — anything with sale_price NOT NULL.
    rows.append((
        "collection_sale", "on-sale", "On Sale", "rule",
        json.dumps({"on_sale": True}, sort_keys=True),
    ))
    # "Under $50" rule
    rows.append((
        "collection_under_50", "under-50", "Under $50", "rule",
        json.dumps({"max_price": 50.0}, sort_keys=True),
    ))

    # Pad with manual collections to hit N_COLLECTIONS.
    existing = len(rows)
    for i in range(existing, N_COLLECTIONS):
        slug = f"manual-{i:02d}"
        rows.append((
            f"collection_manual_{i:02d}",
            slug, slug.replace("-", " ").title(), "manual", "{}",
        ))

    cur.executemany(
        "INSERT INTO collections (id, slug, title, kind, rule_json) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


# ── coupons ───────────────────────────────────────────────────────────


def _seed_coupons(cur: sqlite3.Cursor, rng: random.Random,
                  now_dt: datetime) -> None:
    """15 coupons including the pinned SPRING15, BOGO, and the
    looks-stackable-but-server-rejects pair STACK_TRAP_A / STACK_TRAP_B."""
    far_future = _iso(now_dt + timedelta(days=180))
    near_future = _iso(now_dt + timedelta(days=30))
    expired = _iso(now_dt - timedelta(days=30))

    rows: list[tuple[Any, ...]] = [
        # Pinned for T01: 15% off, stackable=1 (no exclusions), not expired.
        ("SPRING15", "pct", 15.0, None, 1, "[]", far_future, None, 0, None),
        # Pinned for T04: BOGO coupon. Some recent orders use it.
        ("BOGO", "bogo", 1.0, None, 0, "[]", far_future, None, 0, None),
        # The trap pair — UI says stackable=1 but the server's
        # stacking_exclusions cross-reference rejects them together.
        ("STACK_TRAP_A", "pct", 10.0, None, 1,
         json.dumps(["STACK_TRAP_B"]), far_future, None, 0, None),
        ("STACK_TRAP_B", "amount", 5.0, None, 1,
         json.dumps(["STACK_TRAP_A"]), far_future, None, 0, None),
        # A clearly-exclusive coupon (UI honest)
        ("VIPONLY", "pct", 25.0, None, 0, "[]", far_future, 1000, 0, None),
        # Five percent-off "category" coupons
        ("WINTER10", "pct", 10.0, "outerwear-women", 1, "[]", near_future, 5000, 0, None),
        ("WINTER20", "pct", 20.0, "outerwear-men", 0, "[]", near_future, 5000, 0, None),
        ("TEE5", "amount", 5.0, "tees", 1, "[]", far_future, None, 0, None),
        ("DENIM15", "pct", 15.0, "denim", 1, "[]", far_future, None, 0, None),
        # An expired one
        ("BLACKFRIDAY", "pct", 40.0, None, 0, "[]", expired, 10_000, 9_999, None),
        # A disabled one
        ("OLDSALE", "pct", 50.0, None, 0, "[]", far_future, None, 0, _iso(now_dt - timedelta(days=2))),
        # Four extra random coupons (bring total to 15).
        ("FREESHIP", "amount", 8.0, None, 1, "[]", far_future, None, 0, None),
        ("SUMMER25", "pct", 25.0, None, 0, "[]", _iso(now_dt + timedelta(days=400)), None, 0, None),
        ("REFER20", "pct", 20.0, None, 0, "[]", far_future, 500, 0, None),
        ("LOYAL10", "pct", 10.0, None, 1, "[]", far_future, None, 0, None),
    ]

    cur.executemany(
        "INSERT INTO coupons (code, kind, value, scope_category, stackable, "
        "stacking_exclusions, expires_at, max_uses, uses_count, disabled_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── orders ────────────────────────────────────────────────────────────


def _seed_orders(cur: sqlite3.Cursor, rng: random.Random,
                 now_dt: datetime) -> None:
    """5,000 historical orders. Pinned: order_04421 (#4421) has 3 line
    items, status='paid'. Five orders in the last 7 days use coupon BOGO."""
    # Pre-fetch variants so we can pick from them.
    variants = list(cur.execute(
        "SELECT v.id, v.sku, v.product_id, COALESCE(p.sale_price, p.base_price) AS price "
        "FROM variants v JOIN products p ON v.product_id = p.id"
    ).fetchall())

    # Pick the 5 "BOGO recent" orders deterministically: order ids
    # exactly bogo_set_ids placed within the last 7 days, status paid.
    bogo_recent_ids = {f"order_{i:05d}" for i in (4801, 4802, 4803, 4804, 4805)}
    order_4421 = "order_04421"

    order_rows: list[tuple[Any, ...]] = []
    item_rows: list[tuple[Any, ...]] = []
    line_id_counter = 0

    for i in range(1, N_ORDERS + 1):
        order_id = _id("order", i)
        number = f"#{i}"
        customer_idx = rng.randint(1, N_CUSTOMERS)
        customer_id = _id("customer", customer_idx)
        # Address: use the default address for this customer.
        addr_row = cur.execute(
            "SELECT id FROM customer_addresses WHERE customer_id = ? "
            "ORDER BY is_default DESC, id ASC LIMIT 1",
            (customer_id,),
        ).fetchone()
        shipping_address_id = addr_row["id"] if addr_row else None

        # Placement timestamp — spread over the last ~400 days.
        days_ago = rng.randint(0, 400)
        placed_at = _iso(now_dt - timedelta(days=days_ago,
                                            hours=rng.randint(0, 23),
                                            minutes=rng.randint(0, 59)))
        status = _pick_status(rng)
        coupon_codes: list[str] = []

        # Pin order_04421: 3 line items, paid status.
        if order_id == order_4421:
            status = "paid"
            n_items = 3
            placed_at = _iso(now_dt - timedelta(days=20))
        elif order_id in bogo_recent_ids:
            status = "paid"
            n_items = 2  # BOGO needs ≥2
            placed_at = _iso(now_dt - timedelta(days=rng.randint(0, 6),
                                                hours=rng.randint(1, 23)))
            coupon_codes = ["BOGO"]
        else:
            n_items = rng.randint(1, 4)
            if rng.random() < 0.10:
                coupon_codes = [rng.choice(
                    ["SPRING15", "WINTER10", "TEE5", "DENIM15", "FREESHIP"]
                )]

        # Line items
        subtotal = 0.0
        chosen_variants = rng.sample(variants, k=n_items) if len(variants) >= n_items else variants
        order_items: list[tuple[Any, ...]] = []
        for line_no, v in enumerate(chosen_variants, start=1):
            qty = rng.randint(1, 3)
            unit_price = float(v["price"])
            subtotal += unit_price * qty
            line_id_counter += 1
            order_items.append((
                _id("item", line_id_counter, width=7),
                order_id, line_no, v["id"], v["sku"],
                f"Item {v['sku']}", unit_price, qty,
            ))

        discount = 0.0
        if "SPRING15" in coupon_codes:
            discount += subtotal * 0.15
        elif "WINTER10" in coupon_codes:
            discount += subtotal * 0.10
        if "BOGO" in coupon_codes and chosen_variants:
            # Cheapest item free
            cheapest = min(chosen_variants, key=lambda v: float(v["price"]))
            discount += float(cheapest["price"])
        shipping = 8.00 if subtotal < 75 else 0.0
        total = round(max(0.0, subtotal - discount + shipping), 2)

        fulfilled_at = _iso(_parse_iso(placed_at) + timedelta(days=2)) if status in ("fulfilled", "refunded") else None
        cancelled_at = _iso(_parse_iso(placed_at) + timedelta(days=1)) if status == "cancelled" else None

        order_rows.append((
            order_id, number, customer_id, shipping_address_id,
            _id("pay", customer_idx),
            status,
            round(subtotal, 2), round(discount, 2), shipping, total,
            json.dumps(coupon_codes),
            0,
            placed_at, fulfilled_at, cancelled_at,
        ))
        item_rows.extend(order_items)

    cur.executemany(
        "INSERT INTO orders (id, number, customer_id, shipping_address_id, "
        "payment_method_id, status, subtotal, discount_total, shipping_total, "
        "total, coupon_codes, notify_customer, placed_at, fulfilled_at, "
        "cancelled_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        order_rows,
    )
    cur.executemany(
        "INSERT INTO order_items (id, order_id, line_no, variant_id, sku, "
        "title, unit_price, quantity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        item_rows,
    )


# ── saved views ───────────────────────────────────────────────────────


def _seed_saved_views(cur: sqlite3.Cursor) -> None:
    """One pre-seeded view for T04. Empty member list initially —
    the agent's job is to populate it."""
    cur.execute(
        "INSERT INTO saved_views (id, title, description, filter_json) "
        "VALUES (?, ?, ?, ?)",
        (
            "saved_view_bogo_recent",
            "BOGO orders — last 7 days",
            "Orders placed in the trailing 7 days that used coupon BOGO. "
            "Populated manually by an admin export.",
            json.dumps({"coupon": "BOGO", "window_days": 7}, sort_keys=True),
        ),
    )
