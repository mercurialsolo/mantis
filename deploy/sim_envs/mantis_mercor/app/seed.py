"""Deterministic seed for mantis-mercor.

Same seed → identical DB. Generates:

* ~50 users (40 candidates + 9 clients + 1 admin)
* ~12 companies (one per client + a few unowned shells)
* ~30 jobs spread across the 6 categories
* ~100 historical applications + some interviews / payments

ID conventions:

* ``candidate_00007``, ``client_00003``, ``job_00001``, ``app_00042``,
  ``company_00005``, ``shortlist_00012``.

Canonical fixtures the oracles assume exist:

* ``candidate_00001`` — the canonical candidate user (password: pass).
* ``client_00001`` — the canonical client (password: pass).
* ``job_00001`` — Internal Medicine Expert (Medical, $130-$180/hr).
* ``app_pending_for_client_00001`` synthesised: a pending submitted
  application from ``candidate_00007`` to ``job_00001``. (T03 decline
  oracle targets this one.)
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from typing import Any

FAKE_NOW_DEFAULT = "2026-06-08T09:00:00Z"

CATEGORIES = ["Medical", "Legal", "Finance", "Software", "Consulting", "Office"]

CATEGORY_TITLES: dict[str, list[str]] = {
    "Medical": [
        "Internal Medicine Expert",
        "Hematology/Oncology Expert",
        "Biology PhD Expert",
        "Radiology Expert",
        "Pediatrics Expert",
    ],
    "Legal": [
        "Legal Expert — Litigation",
        "Legal Expert — Contracts",
        "Legal Expert — IP",
        "Compliance Reviewer",
    ],
    "Finance": [
        "Private Equity Expert",
        "Equity Research Analyst",
        "Quantitative Strategist",
        "Tax Specialist",
    ],
    "Software": [
        "Senior Software Engineer",
        "ML Engineer",
        "Frontend Engineer",
        "Site Reliability Engineer",
    ],
    "Consulting": [
        "Management & Strategy Consultant (MBB)",
        "Operations Consultant",
        "Healthcare Consultant",
    ],
    "Office": [
        "Office-Suite Expert",
        "Executive Assistant Specialist",
        "Data Entry Expert",
    ],
}

RATE_BANDS: dict[str, tuple[int, int]] = {
    "Medical": (130, 180),
    "Legal": (110, 160),
    "Finance": (120, 200),
    "Software": (90, 150),
    "Consulting": (100, 175),
    "Office": (35, 65),
}

CATEGORY_SKILLS: dict[str, list[str]] = {
    "Medical": ["clinical-reasoning", "literature-review", "case-study"],
    "Legal": ["contract-review", "case-law", "drafting"],
    "Finance": ["financial-modeling", "valuation", "DCF"],
    "Software": ["python", "typescript", "system-design"],
    "Consulting": ["strategy", "operations", "market-analysis"],
    "Office": ["excel", "google-sheets", "data-cleanup"],
}

SCREENING_QS_DEFAULT = [
    "Briefly describe your most relevant experience.",
    "Why are you interested in this role?",
    "What is your earliest start date?",
]


def _hash(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if len(parts) >= 3:
        return (parts[0][0] + parts[1][0] + parts[2][0]).upper()
    if len(parts) == 2:
        return (parts[0][0] + parts[1][0] + parts[1][-1]).upper()
    base = (parts[0] if parts else "X").upper()
    return (base + "XX")[:3]


FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Cameron",
    "Skyler", "Quinn", "Avery", "Drew", "Hayden", "Logan", "Reese",
    "Sage", "Devon", "Emerson", "Finley", "Harper", "Justice",
    "Kennedy", "Lennon", "Marlowe", "Noor", "Onyx", "Parker",
    "Rowan", "Saylor", "Tatum", "Umber", "Vesper", "Wynn", "Xander",
    "Yael", "Zion", "Arden", "Briar", "Cedar", "Dune", "Echo",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott",
    "Torres", "Nguyen", "Hill", "Flores",
]


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str) -> None:
    """Re-seed the DB to a deterministic baseline."""
    rng = random.Random(seed_val)

    # Wipe existing rows (preserves schema).
    tables = [
        "audit_log",
        "payments",
        "reviews",
        "interviews",
        "shortlist_entries",
        "applications",
        "candidate_profiles",
        "jobs",
        "companies",
        "users",
    ]
    for t in tables:
        conn.execute(f"DELETE FROM {t}")

    # ── canonical users ──────────────────────────────────────────
    pw = _hash("pass")
    cano_users: list[dict[str, Any]] = [
        {
            "id": "candidate_00001",
            "email": "candidate@mercor.example",
            "password_hash": pw,
            "role": "candidate",
            "name": "Alex Reyes",
            "company_id": None,
            "created_at": fake_now,
        },
        {
            "id": "client_00001",
            "email": "client@mercor.example",
            "password_hash": pw,
            "role": "client",
            "name": "Jordan Chen",
            "company_id": "company_00001",
            "created_at": fake_now,
        },
        {
            "id": "admin_00001",
            "email": "admin@mercor.example",
            "password_hash": pw,
            "role": "admin",
            "name": "Admin User",
            "company_id": None,
            "created_at": fake_now,
        },
    ]
    for u in cano_users:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, company_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (u["id"], u["email"], u["password_hash"], u["role"], u["name"],
             u["company_id"], u["created_at"]),
        )

    # Extra candidates (candidate_00002 ... candidate_00040)
    for i in range(2, 41):
        nm = _name(rng)
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, company_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"candidate_{i:05d}",
                f"cand{i:03d}@mercor.example",
                pw,
                "candidate",
                nm,
                None,
                fake_now,
            ),
        )

    # Extra clients (client_00002 ... client_00009)
    for i in range(2, 10):
        nm = _name(rng)
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, company_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"client_{i:05d}",
                f"client{i:03d}@mercor.example",
                pw,
                "client",
                nm,
                f"company_{i:05d}",
                fake_now,
            ),
        )

    # ── companies ────────────────────────────────────────────────
    companies = [
        ("company_00001", "Acme AI Labs", "acmeai.example", "client_00001"),
    ]
    for i in range(2, 10):
        rng_name = f"{rng.choice(['Helio', 'Nova', 'Bright', 'North', 'Spark', 'Quanta', 'Mirror', 'Echo'])} {rng.choice(['AI', 'Labs', 'Research', 'Studio', 'Works'])}"
        companies.append(
            (f"company_{i:05d}", rng_name, f"company{i:03d}.example",
             f"client_{i:05d}")
        )
    for c in companies:
        conn.execute(
            "INSERT INTO companies (id, name, domain, owner_user_id) VALUES (?, ?, ?, ?)",
            c,
        )

    # ── candidate profiles ───────────────────────────────────────
    for i in range(1, 41):
        cid = f"candidate_{i:05d}"
        cat = CATEGORIES[i % len(CATEGORIES)]
        skills = CATEGORY_SKILLS[cat]
        rate = rng.randint(*RATE_BANDS[cat])
        avail = rng.choice(["Full-time", "Part-time", "Project-based"])
        headline = f"{cat} Specialist, {rng.randint(2, 15)} years"
        conn.execute(
            "INSERT INTO candidate_profiles (user_id, headline, skills_json, "
            "hourly_rate, availability, resume_text, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                headline,
                "[" + ", ".join(f'"{s}"' for s in skills) + "]",
                float(rate),
                avail,
                f"Seeded resume for {cid}.",
                fake_now,
            ),
        )

    # ── jobs ─────────────────────────────────────────────────────
    # canonical job_00001 — Internal Medicine Expert.
    cano_job = {
        "id": "job_00001",
        "company_id": "company_00001",
        "title": "Internal Medicine Expert",
        "category": "Medical",
        "description_md": (
            "We're hiring board-eligible Internal Medicine physicians to "
            "evaluate and improve frontier AI clinical reasoning. "
            "Sessions are async and remote.\n\n"
            "**You'll**: review AI outputs against real-world cases; flag "
            "factual errors; submit case-based critiques."
        ),
        "skills_json": '["clinical-reasoning", "literature-review", "case-study"]',
        "rate_min": 130,
        "rate_max": 180,
        "engagement": "Hourly",
        "hours_per_week": 10,
        "hires_recently": 75,
        "screening_qs": (
            '["Are you board-eligible or board-certified in Internal Medicine?",'
            '"Briefly describe a complex case you handled in the last year.",'
            '"What is your earliest start date?"]'
        ),
        "status": "open",
        "posted_at": fake_now,
    }
    conn.execute(
        "INSERT INTO jobs (id, company_id, title, category, description_md, "
        "skills_json, rate_min, rate_max, engagement, hours_per_week, "
        "hires_recently, screening_qs, status, posted_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        tuple(cano_job.values()),
    )

    # Other jobs (job_00002 ... job_00030).
    job_counter = 2
    for cat in CATEGORIES:
        titles = CATEGORY_TITLES[cat]
        for title in titles:
            if job_counter > 30:
                break
            rate_min, rate_max = RATE_BANDS[cat]
            rate_min = rate_min + rng.randint(-5, 5)
            rate_max = rate_max + rng.randint(-5, 10)
            skills = CATEGORY_SKILLS[cat]
            engagement = rng.choice(["Hourly", "Project", "Full-time"])
            company_id = f"company_{rng.randint(1, 9):05d}"
            jid = f"job_{job_counter:05d}"
            conn.execute(
                "INSERT INTO jobs (id, company_id, title, category, description_md, "
                "skills_json, rate_min, rate_max, engagement, hours_per_week, "
                "hires_recently, screening_qs, status, posted_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    company_id,
                    title,
                    cat,
                    f"## About this role\n\n{title} for {cat} domain. "
                    f"Remote, async, rate ${rate_min}-${rate_max}/hr.",
                    "[" + ", ".join(f'"{s}"' for s in skills) + "]",
                    float(rate_min),
                    float(rate_max),
                    engagement,
                    rng.choice([10, 15, 20, 30, 40]),
                    rng.randint(5, 90),
                    "[" + ", ".join(f'"{q}"' for q in SCREENING_QS_DEFAULT) + "]",
                    "open",
                    fake_now,
                ),
            )
            job_counter += 1

    # ── canonical pending application for T03 ────────────────────
    # candidate_00007 applied to job_00001 — status=submitted.
    conn.execute(
        "INSERT INTO applications (id, job_id, candidate_id, status, headline, "
        "skills_json, hourly_rate, resume_text, screening_answers, reject_reason, "
        "submitted_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "app_00001",
            "job_00001",
            "candidate_00007",
            "submitted",
            "Internal Medicine PGY-4",
            '["clinical-reasoning"]',
            145.0,
            "Seeded resume for candidate_00007.",
            '["yes", "saw a rare ARDS case in February", "two weeks"]',
            "",
            fake_now,
            fake_now,
        ),
    )

    # Historical applications (different jobs, varied status).
    app_counter = 2
    for cand_i in range(2, 41):
        if app_counter > 100:
            break
        cid = f"candidate_{cand_i:05d}"
        jobs_to_try = rng.sample(range(2, 31), k=rng.randint(1, 3))
        for jid_n in jobs_to_try:
            if app_counter > 100:
                break
            jid = f"job_{jid_n:05d}"
            status = rng.choice([
                "submitted", "under_review", "interview", "rejected", "hired",
            ])
            aid = f"app_{app_counter:05d}"
            try:
                conn.execute(
                    "INSERT INTO applications (id, job_id, candidate_id, status, "
                    "headline, skills_json, hourly_rate, resume_text, "
                    "screening_answers, reject_reason, submitted_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        aid, jid, cid, status,
                        f"{cid} headline",
                        '["domain-skill"]',
                        float(rng.randint(50, 200)),
                        f"Seeded resume for {cid}.",
                        '["seeded", "seeded", "seeded"]',
                        "low cultural fit" if status == "rejected" else "",
                        fake_now,
                        fake_now,
                    ),
                )
                app_counter += 1
            except sqlite3.IntegrityError:
                # UNIQUE(job_id, candidate_id) — skip dupes.
                continue

    # Interviews + payments — light historical sprinkle.
    int_id = 1
    for row in conn.execute(
        "SELECT id FROM applications WHERE status IN ('interview', 'hired') LIMIT 20"
    ).fetchall():
        conn.execute(
            "INSERT INTO interviews (id, application_id, scheduled_at, "
            "interviewer, outcome) VALUES (?,?,?,?,?)",
            (
                f"interview_{int_id:05d}",
                row["id"],
                fake_now,
                "client_00001",
                rng.choice(["pending", "passed", "failed"]),
            ),
        )
        int_id += 1

    pay_id = 1
    for row in conn.execute(
        "SELECT id FROM applications WHERE status='hired' LIMIT 20"
    ).fetchall():
        conn.execute(
            "INSERT INTO payments (id, application_id, amount, occurred_at) "
            "VALUES (?,?,?,?)",
            (
                f"payment_{pay_id:05d}",
                row["id"],
                float(rng.randint(500, 5000)),
                fake_now,
            ),
        )
        pay_id += 1

    conn.commit()
