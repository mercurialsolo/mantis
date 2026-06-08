"""Deterministic seed for mantis-fiverr.

Same SEED → identical IDs + identical column values across runs. The
oracle pass conditions depend on this stability.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

FAKE_NOW_DEFAULT = "2026-06-01T12:00:00Z"

# Canonical categories — flat list mirroring the home tiles + a handful
# of subcategories so /categories/<slug> has variety.
CATEGORIES: list[tuple[str, str, str | None, str, int]] = [
    # (slug, title, parent_slug, icon, sort_order)
    ("graphics-design", "Graphics & Design", None, "palette", 1),
    ("programming-tech", "Programming & Tech", None, "code", 2),
    ("digital-marketing", "Digital Marketing", None, "megaphone", 3),
    ("video-animation", "Video & Animation", None, "film", 4),
    ("writing-translation", "Writing & Translation", None, "pen", 5),
    ("music-audio", "Music & Audio", None, "headphones", 6),
    ("business", "Business", None, "briefcase", 7),
    ("data", "Data", None, "database", 8),
    ("photography", "Photography", None, "camera", 9),
    ("ai-services", "AI Services", None, "sparkles", 10),
    ("lifestyle", "Lifestyle", None, "heart", 11),
    ("consulting", "Consulting", None, "chat", 12),
    # subcats of graphics-design
    ("logo-design", "Logo Design", "graphics-design", "circle", 1),
    ("brand-style-guide", "Brand Style Guide", "graphics-design", "book", 2),
    ("business-cards", "Business Cards & Stationery", "graphics-design", "card", 3),
    ("illustration", "Illustration", "graphics-design", "brush", 4),
    ("web-mobile-design", "Web & Mobile Design", "graphics-design", "monitor", 5),
    # subcats of programming-tech
    ("website-development", "Website Development", "programming-tech", "code", 1),
    ("wordpress", "WordPress", "programming-tech", "code", 2),
    ("mobile-apps", "Mobile Apps", "programming-tech", "phone", 3),
]

# (slug, title, category, basic_price, basic_delivery_d, basic_revs)
GIG_SEEDS: list[tuple[str, str, str, float, int, int]] = [
    ("modern-minimalist-logo", "I will design a modern minimalist logo for your business", "logo-design", 25.0, 2, 2),
    ("professional-wordpress-site", "I will build a professional WordPress website with elementor", "wordpress", 95.0, 5, 1),
    ("explainer-video-2d", "I will create a 2D animated explainer video for your product", "video-animation", 75.0, 5, 2),
    ("hand-drawn-illustration", "I will create a custom hand-drawn illustration for your project", "illustration", 50.0, 4, 2),
    ("seo-audit-report", "I will provide a comprehensive seo audit report with action items", "digital-marketing", 35.0, 3, 1),
    ("translate-en-to-es", "I will translate english to spanish 1000 words professionally", "writing-translation", 20.0, 2, 3),
    ("voice-over-male-american", "I will record a professional voice over in american english", "music-audio", 30.0, 2, 2),
    ("react-frontend-app", "I will develop a fast and modern react frontend application", "programming-tech", 150.0, 7, 1),
    ("brand-style-guide-pdf", "I will create a complete brand style guide pdf document", "brand-style-guide", 120.0, 7, 2),
    ("business-card-design", "I will design a professional double-sided business card", "business-cards", 15.0, 2, 3),
    ("data-cleaning-excel", "I will clean and analyze your messy excel data professionally", "data", 40.0, 3, 2),
    ("ai-chatbot-prompt-engineering", "I will engineer effective prompts for your ai chatbot", "ai-services", 60.0, 3, 2),
    ("instagram-content-strategy", "I will create a 30 day instagram content strategy", "digital-marketing", 45.0, 4, 2),
    ("ux-research-interviews", "I will conduct ux research interviews and synthesize findings", "consulting", 200.0, 10, 1),
    ("product-photography-edit", "I will edit your product photography to ecommerce standards", "photography", 25.0, 2, 3),
    ("mobile-app-design-figma", "I will design a beautiful mobile app ui in figma", "web-mobile-design", 180.0, 7, 2),
    ("react-native-mobile-app", "I will build a cross platform react native mobile app", "mobile-apps", 350.0, 14, 1),
    ("blog-post-writing-1000w", "I will write a 1000 word seo optimized blog post", "writing-translation", 30.0, 3, 2),
    ("custom-vector-illustration", "I will create custom vector illustrations for your needs", "illustration", 45.0, 4, 2),
    ("podcast-editing-1hr", "I will edit your podcast episode up to 1 hour professionally", "music-audio", 50.0, 3, 2),
    ("shopify-store-setup", "I will set up your shopify store from scratch", "programming-tech", 250.0, 10, 1),
    ("google-ads-setup", "I will set up your google ads campaign for conversions", "digital-marketing", 85.0, 5, 2),
    ("startup-pitch-deck", "I will design a stunning startup pitch deck", "business", 150.0, 5, 2),
    ("brand-name-suggestions", "I will suggest creative brand names for your startup", "business", 25.0, 2, 3),
    ("youtube-thumbnail-design", "I will design eye catching youtube thumbnails", "graphics-design", 10.0, 1, 3),
    ("python-data-script", "I will write a python script for data automation", "programming-tech", 65.0, 3, 1),
    ("character-illustration", "I will draw a custom character illustration for your brand", "illustration", 70.0, 5, 2),
    ("email-marketing-template", "I will design a responsive email marketing template", "digital-marketing", 40.0, 3, 2),
    ("article-rewriting-seo", "I will rewrite your article for better seo and readability", "writing-translation", 20.0, 2, 3),
    ("custom-music-jingle", "I will compose a custom music jingle for your brand", "music-audio", 80.0, 5, 2),
]


COUNTRIES = ["United States", "United Kingdom", "Canada", "Australia",
             "Germany", "France", "Netherlands", "India", "Brazil",
             "Spain", "Pakistan", "Argentina"]
LANGS = [["English"], ["English", "Spanish"], ["English", "French"],
         ["English", "German"], ["English", "Portuguese"],
         ["English", "Hindi", "Urdu"]]
LEVELS_DIST = ["new", "level_one", "level_one", "level_two",
               "level_two", "level_two", "top_rated"]


def _id_for(prefix: str, n: int) -> str:
    return f"{prefix}_{n:05d}"


def _hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def _wipe(conn: sqlite3.Connection) -> None:
    for tbl in (
        "audit_log", "favorites", "reviews", "messages", "conversations",
        "order_items", "orders", "gigs", "sellers", "users",
        "categories",
    ):
        conn.execute(f"DELETE FROM {tbl}")


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str) -> None:
    rng = random.Random(seed_val)
    now = datetime.fromisoformat(fake_now.replace("Z", "+00:00"))

    with conn:
        _wipe(conn)

        # Categories ------------------------------------------------
        for slug, title, parent, icon, order in CATEGORIES:
            conn.execute(
                "INSERT INTO categories (slug, title, parent_slug, icon, sort_order) "
                "VALUES (?, ?, ?, ?, ?)",
                (slug, title, parent, icon, order),
            )

        # Buyers ----------------------------------------------------
        # buyer_00001 is the canonical "you" the oracles grade against.
        buyer_ids: list[str] = []
        for i in range(1, 21):
            uid = _id_for("buyer", i)
            uname = f"buyer{i:02d}"
            email = f"{uname}@mantis.example"
            display = f"Buyer {i:02d}"
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, role, display_name, created_at) "
                "VALUES (?, ?, ?, ?, 'buyer', ?, ?)",
                (uid, uname, email, _hash_password("password"), display,
                 (now - timedelta(days=200 - i)).isoformat()),
            )
            buyer_ids.append(uid)

        # Sellers ---------------------------------------------------
        seller_ids: list[str] = []
        for i in range(1, 31):
            uid = _id_for("seller", i)
            uname = f"seller{i:02d}"
            email = f"{uname}@mantis.example"
            display = f"Seller {i:02d}"
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, role, display_name, created_at) "
                "VALUES (?, ?, ?, ?, 'seller', ?, ?)",
                (uid, uname, email, _hash_password("password"), display,
                 (now - timedelta(days=400 - i * 7)).isoformat()),
            )
            level = LEVELS_DIST[(i + seed_val) % len(LEVELS_DIST)]
            country = COUNTRIES[(i * 7 + seed_val) % len(COUNTRIES)]
            languages = LANGS[(i * 3 + seed_val) % len(LANGS)]
            conn.execute(
                "INSERT INTO sellers (user_id, level, country, languages, "
                "response_time_h, member_since, avatar_palette) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, level, country, json.dumps(languages),
                 1 + (i * 3) % 12,
                 (now - timedelta(days=400 - i * 7)).isoformat(),
                 i % 8),
            )
            seller_ids.append(uid)

        # Gigs ------------------------------------------------------
        gig_ids: list[str] = []
        for i, (slug, title, cat, basic_price, basic_delivery, basic_revs) in enumerate(GIG_SEEDS, start=1):
            gid = _id_for("gig", i)
            seller = seller_ids[(i + seed_val) % len(seller_ids)]
            # Three-tier pricing: standard = basic*2.6, premium = basic*5
            std_price = round(basic_price * 2.4, 0)
            prem_price = round(basic_price * 4.8, 0)
            # Common feature lists
            basic_feats = ["1 concept included", f"{basic_delivery}-day delivery",
                           f"{basic_revs} revision" + ("s" if basic_revs != 1 else ""),
                           "Source file"]
            std_feats = basic_feats + ["High resolution", "Vector file"]
            prem_feats = std_feats + ["3D mockup", "Commercial use", "Stationery design"]

            conn.execute(
                """INSERT INTO gigs (
                    id, seller_id, slug, title, description_md, category_slug,
                    image_palette,
                    pkg_basic_title, pkg_basic_desc, pkg_basic_price,
                    pkg_basic_delivery_d, pkg_basic_revisions, pkg_basic_features,
                    pkg_standard_title, pkg_standard_desc, pkg_standard_price,
                    pkg_standard_delivery_d, pkg_standard_revisions, pkg_standard_features,
                    pkg_premium_title, pkg_premium_desc, pkg_premium_price,
                    pkg_premium_delivery_d, pkg_premium_revisions, pkg_premium_features,
                    avg_rating, review_count, orders_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?,
                          'Basic', ?, ?, ?, ?, ?,
                          'Standard', ?, ?, ?, ?, ?,
                          'Premium', ?, ?, ?, ?, ?,
                          ?, ?, ?, ?)""",
                (
                    gid, seller, slug, title.title(),
                    _gig_description(title, slug),
                    cat,
                    i % 8,
                    f"Get started with the {title.split()[3] if len(title.split()) > 3 else 'core'} package — perfect for testing the waters.",
                    basic_price, basic_delivery, basic_revs, json.dumps(basic_feats),
                    f"Most popular — adds extra polish and faster turnaround.",
                    std_price, max(1, basic_delivery + 1), basic_revs + 1, json.dumps(std_feats),
                    f"Everything in Standard plus dedicated support and source files.",
                    prem_price, basic_delivery + 3, basic_revs + 2, json.dumps(prem_feats),
                    round(4.4 + rng.random() * 0.6, 1),
                    20 + rng.randrange(0, 200),
                    50 + rng.randrange(0, 500),
                    (now - timedelta(days=30 + (i * 11) % 300)).isoformat(),
                ),
            )
            gig_ids.append(gid)

        # Orders ----------------------------------------------------
        # Seed ~15 historical orders. The oracles look at specific ids:
        # gig_00001 Basic order placed during the run (t01),
        # order_00007 must be completed + reviewable (t03).
        for n in range(1, 16):
            oid = _id_for("order", n)
            buyer = buyer_ids[(n - 1) % len(buyer_ids)]
            gid = gig_ids[(n + 3) % len(gig_ids)]
            gig_row = conn.execute(
                "SELECT seller_id, pkg_basic_price, pkg_standard_price, pkg_premium_price "
                "FROM gigs WHERE id = ?", (gid,)
            ).fetchone()
            tier_idx = (n + seed_val) % 3
            tier = ["basic", "standard", "premium"][tier_idx]
            price_col = {
                "basic": "pkg_basic_price",
                "standard": "pkg_standard_price",
                "premium": "pkg_premium_price",
            }[tier]
            unit_price = gig_row[price_col]
            service_fee = round(unit_price * 0.055 + 2.0, 2)
            total = round(unit_price + service_fee, 2)
            placed_at = now - timedelta(days=60 - n * 3)
            due_at = placed_at + timedelta(days=5)
            status = "completed" if n <= 8 else ("delivered" if n <= 11 else "active")
            delivered = (placed_at + timedelta(days=3)).isoformat() if status != "active" else None
            completed = (placed_at + timedelta(days=4)).isoformat() if status == "completed" else None
            conn.execute(
                """INSERT INTO orders (
                    id, number, buyer_id, seller_id, gig_id, tier,
                    subtotal, service_fee, total, status, requirements,
                    placed_at, due_at, delivered_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '',
                         ?, ?, ?, ?)""",
                (oid, f"#FO{n:04d}", buyer, gig_row["seller_id"], gid, tier,
                 unit_price, service_fee, total, status,
                 placed_at.isoformat(), due_at.isoformat(),
                 delivered, completed),
            )
            conn.execute(
                "INSERT INTO order_items (id, order_id, line_no, description, unit_price, quantity) "
                "VALUES (?, ?, 1, ?, ?, 1)",
                (f"oi_{n:05d}", oid,
                 f"{tier.title()} package", unit_price),
            )

        # Conversations + messages ---------------------------------
        # Seed buyer_00001 ↔ seller for gig_00001 + a few others.
        gig_one_seller = conn.execute(
            "SELECT seller_id FROM gigs WHERE id='gig_00001'"
        ).fetchone()["seller_id"]
        for n, (buyer, seller) in enumerate([
            ("buyer_00001", gig_one_seller),
            ("buyer_00001", "seller_03"),
            ("buyer_00002", gig_one_seller),
            ("buyer_00003", "seller_05"),
        ], start=1):
            cid = _id_for("conv", n)
            last_msg = now - timedelta(days=5 - n)
            try:
                conn.execute(
                    "INSERT INTO conversations (id, buyer_id, seller_id, last_msg_at) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, buyer, seller, last_msg.isoformat()),
                )
            except sqlite3.IntegrityError:
                continue
            # 2 seed messages per thread
            conn.execute(
                "INSERT INTO messages (id, conversation_id, sender_id, body, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"msg_{cid}_1", cid, buyer,
                 "Hi! I'm interested in your service. Could you tell me more?",
                 (last_msg - timedelta(hours=2)).isoformat()),
            )
            conn.execute(
                "INSERT INTO messages (id, conversation_id, sender_id, body, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"msg_{cid}_2", cid, seller,
                 "Hi there! Thanks for reaching out. Happy to help — what's the project?",
                 (last_msg - timedelta(hours=1)).isoformat()),
            )

        # Reviews ---------------------------------------------------
        # Seed a few historical reviews on completed orders 1..5
        # so detail pages show review history. order_00007 is left
        # explicitly unreviewed for t03.
        for n in (1, 2, 3, 4, 5):
            oid = _id_for("order", n)
            order = conn.execute(
                "SELECT buyer_id, seller_id, gig_id FROM orders WHERE id=?",
                (oid,)
            ).fetchone()
            if order is None:
                continue
            rid = _id_for("review", n)
            stars = 5 if n % 3 else 4
            conn.execute(
                "INSERT INTO reviews (id, order_id, gig_id, buyer_id, seller_id, "
                "stars, body, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, oid, order["gig_id"], order["buyer_id"],
                 order["seller_id"], stars,
                 _review_body(n, stars),
                 (now - timedelta(days=20 - n)).isoformat()),
            )


def _gig_description(title: str, slug: str) -> str:
    return (
        f"Welcome! Thanks for stopping by my gig.\n\n"
        f"I'm excited to help you with **{title.lower()}**. I focus on "
        f"clean, professional results delivered on time, every time.\n\n"
        f"## What you'll get\n\n"
        f"- A clear deliverable matched to your brief\n"
        f"- Source files included\n"
        f"- Quick communication throughout\n\n"
        f"## Why work with me\n\n"
        f"I've completed hundreds of orders on Fiverr with consistent "
        f"five-star feedback. My goal is to make sure you leave happy "
        f"and come back for repeat work.\n\n"
        f"Send me a message before ordering if you have any questions!"
    )


def _review_body(n: int, stars: int) -> str:
    if stars == 5:
        bodies = [
            "Excellent service — exactly what I asked for, delivered on time.",
            "Perfect work. Will definitely order again!",
            "Super professional and great communication. 10/10.",
            "Amazing quality. Highly recommend.",
        ]
    else:
        bodies = [
            "Good work overall, a few revisions needed but happy with the result.",
            "Solid delivery, would work with again.",
        ]
    return bodies[n % len(bodies)]
