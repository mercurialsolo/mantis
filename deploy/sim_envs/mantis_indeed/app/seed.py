"""Deterministic seed for mantis-indeed.

Given ``SEED``, this module deterministically produces:

* 50 companies + their employer users (1 per company)
* 50 seekers + their resumes (1 per seeker)
* 30 active jobs (varied titles, locations, salary ranges)
* ~100 historical applications spread across seekers/jobs
* 1 known T01 seeker (`user_00007`, email `seeker7@example.com`,
  password `seekerpass`) with no saved jobs at boot.
* 1 known T02 seeker (`user_00012`, email `seeker12@example.com`,
  password `seekerpass`) targeted by T02 Easy Apply.
* 1 known T03 employer (`user_emp_00003`, email
  `employer3@example.com`, password `emppass`) owning `job_00003`
  / `posting_00003` with at least one `new` applicant.

Same SEED → identical row IDs + identical column values.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"

N_COMPANIES = 50
N_SEEKERS = 50
N_JOBS = 30
N_HISTORICAL_APPS = 100

_COMPANY_NAMES = [
    "Acme Software", "Northwind Robotics", "Globex Health", "Initech Cloud",
    "Stark Industries", "Hooli", "Pied Piper", "Massive Dynamic",
    "Soylent Systems", "Cyberdyne Solutions", "Wayne Enterprises",
    "Aperture Science", "Tyrell Compute", "Weyland Logistics",
    "OsCorp Bio", "Umbrella Analytics", "Wonka Foods", "Vandelay Imports",
    "Bluth Real Estate", "Dunder Mifflin Tech", "Sterling Cooper Media",
    "Los Pollos Hermanos", "Costco North", "Mooby's Corp", "Krusty Krab",
    "Spacely Sprockets", "Cogswell Cogs", "InGen Genetics", "Yoyodyne",
    "Genco Pura", "Macrohard", "Compuglobalhypermeganet", "Bushwood",
    "Veridian Dynamics", "Globo Gym", "Trade Federation", "Cyrez Group",
    "Daystrom Institute", "Tessier-Ashpool", "Buy n Large", "Soylent Green",
    "Bigweld Industries", "Planet Express", "Slate Rock Co", "Vault-Tec",
    "Black Mesa", "Aperture Labs", "Octan Energy", "Cyberlife", "Olisarra",
]
_LOCATIONS = [
    "Austin, TX", "San Francisco, CA", "New York, NY", "Seattle, WA",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Atlanta, GA",
    "Portland, OR", "Los Angeles, CA", "Remote", "Washington, DC",
    "Raleigh, NC", "Minneapolis, MN", "Pittsburgh, PA",
]
_JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Frontend Engineer",
    "Backend Engineer", "Full Stack Engineer", "DevOps Engineer",
    "Site Reliability Engineer", "Data Engineer", "Machine Learning Engineer",
    "Engineering Manager", "Staff Software Engineer", "Principal Engineer",
    "Mobile Engineer", "iOS Engineer", "Android Engineer", "Platform Engineer",
    "Security Engineer", "QA Engineer", "Embedded Software Engineer",
    "Cloud Engineer", "Solutions Engineer", "Software Developer",
    "Junior Software Engineer", "Data Scientist", "Product Engineer",
    "Test Engineer", "Build Engineer", "Release Engineer",
    "Distributed Systems Engineer", "Database Engineer",
]
_JOB_TYPES = ["Full-time", "Part-time", "Contract", "Internship"]
_EXPERIENCE = ["Entry level", "Mid level", "Senior level", "Director"]
_INDUSTRIES = ["Software", "Healthcare", "Finance", "Retail", "Manufacturing",
               "Education", "Media", "Government"]
_FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley",
                "Avery", "Quinn", "Drew", "Reese", "Charlie", "Sky", "Robin",
                "Pat", "Jamie", "Lee", "Frankie", "Tracy", "Dakota"]
_LAST_NAMES = ["Lee", "Patel", "Chen", "Garcia", "Smith", "Brown", "Johnson",
               "Davis", "Wilson", "Anderson", "Martin", "Thompson", "White",
               "Walker", "Hall", "Green", "Adams", "Baker", "Nelson", "Hill"]
_SKILLS_POOL = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java",
                "Kotlin", "Swift", "C++", "C#", "Ruby", "PHP", "SQL",
                "PostgreSQL", "Redis", "AWS", "GCP", "Azure", "Docker",
                "Kubernetes", "Terraform", "React", "Vue", "Angular", "Django",
                "FastAPI", "Flask", "Spring", "Rails", "Node.js"]


def _sha256(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def _jk(rng: random.Random) -> str:
    return "%016x" % rng.getrandbits(64)


def _wipe(conn: sqlite3.Connection) -> None:
    for table in [
        "audit_log", "saved_jobs", "applications", "resumes",
        "jobs", "users", "companies",
    ]:
        conn.execute(f"DELETE FROM {table}")


def _parse_now(now: str) -> datetime:
    if now.endswith("Z"):
        return datetime.fromisoformat(now[:-1])
    return datetime.fromisoformat(now)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str) -> None:
    """Idempotent reseed — wipe + recreate. Mirrors mantis-shop."""
    rng = random.Random(seed_val)
    now_dt = _parse_now(fake_now)

    _wipe(conn)

    # ── companies ─────────────────────────────────────────────────────
    for i, name in enumerate(_COMPANY_NAMES[:N_COMPANIES], start=1):
        cid = f"company_{i:05d}"
        rating = round(rng.uniform(2.8, 4.7), 1)
        review_count = rng.randint(20, 4000)
        industry = rng.choice(_INDUSTRIES)
        hq = rng.choice(_LOCATIONS)
        conn.execute(
            "INSERT INTO companies (id, name, rating, review_count, industry, headquarters) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, name, rating, review_count, industry, hq),
        )

    # ── employers (one per company, but only fill first 20 with users) ──
    # T03 expects user_emp_00003.
    for i in range(1, 21):
        uid = f"user_emp_{i:05d}"
        email = f"employer{i}@example.com"
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, "
            "phone, city, state, zip, company_id, created_at) "
            "VALUES (?, ?, ?, 'employer', ?, '', '', '', '', ?, ?)",
            (
                uid, email, _sha256("emppass"),
                f"Employer #{i}",
                f"company_{i:05d}",
                _iso(now_dt - timedelta(days=180 + i)),
            ),
        )

    # ── seekers ───────────────────────────────────────────────────────
    for i in range(1, N_SEEKERS + 1):
        uid = f"user_{i:05d}"
        email = f"seeker{i}@example.com"
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        loc = rng.choice(_LOCATIONS)
        # crude location split
        if "," in loc:
            city, state = [s.strip() for s in loc.split(",")]
        else:
            city, state = loc, ""
        zip_code = f"{rng.randint(10000, 99999)}"
        phone = f"({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, name, phone, "
            "city, state, zip, company_id, created_at) "
            "VALUES (?, ?, ?, 'seeker', ?, ?, ?, ?, ?, NULL, ?)",
            (
                uid, email, _sha256("seekerpass"),
                name, phone, city, state, zip_code,
                _iso(now_dt - timedelta(days=30 + i)),
            ),
        )

    # one resume per seeker
    for i in range(1, N_SEEKERS + 1):
        rid = f"resume_{i:05d}"
        uid = f"user_{i:05d}"
        skills = rng.sample(_SKILLS_POOL, 6)
        conn.execute(
            "INSERT INTO resumes (id, user_id, title, summary, skills, experience, updated_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                rid, uid,
                "Default resume",
                f"Software professional with {rng.randint(2, 12)} years of experience.",
                json.dumps(skills),
                "Various roles across the industry.",
                _iso(now_dt - timedelta(days=rng.randint(1, 90))),
            ),
        )

    # ── jobs ──────────────────────────────────────────────────────────
    # Pin job_00001 = first software engineer in Austin TX (T01 target).
    # job_00007 = a remote SE in Austin TX (T01 save target).
    # job_00012 = T02 Easy Apply target.
    # job_00003 = T03 employer review target.
    pinned_jobs: list[tuple[str, str, str, int, int, int, str]] = [
        # (title, company_id, location, salary_low, salary_high, remote, jk_seed)
        ("Software Engineer", "company_00001", "Austin, TX", 120000, 160000, 0, "0000000000000001"),
        ("Senior Software Engineer", "company_00002", "Remote", 150000, 200000, 1, "0000000000000002"),
        ("Frontend Engineer", "company_00003", "San Francisco, CA", 130000, 170000, 0, "0000000000000003"),
        ("Backend Engineer", "company_00004", "Seattle, WA", 130000, 170000, 0, "0000000000000004"),
        ("Full Stack Engineer", "company_00005", "Austin, TX", 110000, 150000, 0, "0000000000000005"),
        ("DevOps Engineer", "company_00006", "Austin, TX", 130000, 170000, 1, "0000000000000006"),
        # job_00007 — explicit T01 save target. SE, Austin TX, remote=1.
        ("Software Engineer", "company_00007", "Austin, TX", 140000, 180000, 1, "0000000000000007"),
        ("Data Engineer", "company_00008", "New York, NY", 140000, 180000, 0, "0000000000000008"),
        ("Machine Learning Engineer", "company_00009", "Boston, MA", 150000, 200000, 0, "0000000000000009"),
        ("Mobile Engineer", "company_00010", "Chicago, IL", 120000, 160000, 0, "000000000000000a"),
        ("iOS Engineer", "company_00011", "Austin, TX", 120000, 160000, 0, "000000000000000b"),
        # job_00012 — T02 Easy Apply target. SE, Austin TX.
        ("Software Engineer", "company_00012", "Austin, TX", 130000, 175000, 0, "000000000000000c"),
        ("Android Engineer", "company_00013", "Denver, CO", 120000, 160000, 0, "000000000000000d"),
        ("Site Reliability Engineer", "company_00014", "San Francisco, CA", 150000, 200000, 1, "000000000000000e"),
        ("Platform Engineer", "company_00015", "Austin, TX", 140000, 185000, 0, "000000000000000f"),
        ("Security Engineer", "company_00016", "Washington, DC", 140000, 180000, 0, "0000000000000010"),
        ("QA Engineer", "company_00017", "Atlanta, GA", 95000, 130000, 0, "0000000000000011"),
        ("Engineering Manager", "company_00018", "Seattle, WA", 180000, 240000, 0, "0000000000000012"),
        ("Staff Software Engineer", "company_00019", "Remote", 200000, 270000, 1, "0000000000000013"),
        ("Principal Engineer", "company_00020", "New York, NY", 220000, 300000, 0, "0000000000000014"),
        ("Software Developer", "company_00021", "Pittsburgh, PA", 100000, 140000, 0, "0000000000000015"),
        ("Junior Software Engineer", "company_00022", "Minneapolis, MN", 80000, 110000, 0, "0000000000000016"),
        ("Data Scientist", "company_00023", "Boston, MA", 140000, 180000, 0, "0000000000000017"),
        ("Test Engineer", "company_00024", "Raleigh, NC", 95000, 125000, 0, "0000000000000018"),
        ("Cloud Engineer", "company_00025", "Austin, TX", 130000, 170000, 1, "0000000000000019"),
        ("Solutions Engineer", "company_00026", "Los Angeles, CA", 130000, 170000, 0, "000000000000001a"),
        ("Embedded Software Engineer", "company_00027", "Portland, OR", 120000, 160000, 0, "000000000000001b"),
        ("Build Engineer", "company_00028", "Austin, TX", 110000, 145000, 0, "000000000000001c"),
        ("Release Engineer", "company_00029", "San Francisco, CA", 130000, 170000, 1, "000000000000001d"),
        ("Database Engineer", "company_00030", "Remote", 130000, 170000, 1, "000000000000001e"),
    ]

    for idx, (title, cid, loc, sl, sh, rem, jk_seed) in enumerate(pinned_jobs, start=1):
        jid = f"job_{idx:05d}"
        snippet = (
            f"{title} role at {cid.replace('company_','co').replace('_', '')}. "
            "Work with a modern stack, ship to production, collaborate "
            "across a small autonomous team."
        )
        desc = (
            f"## About the role\n\n{snippet}\n\n"
            "## What you'll do\n\n"
            "- Own end-to-end delivery of features in production.\n"
            "- Partner with PM and design to scope and ship.\n"
            "- Mentor junior teammates and review PRs.\n\n"
            "## What you bring\n\n"
            f"- {rng.randint(2, 10)}+ years of relevant engineering experience.\n"
            "- Strong fundamentals in computer science.\n"
            "- Excellent written communication.\n\n"
            "## Benefits\n\n"
            "- Competitive salary + equity.\n"
            "- Comprehensive health insurance.\n"
            "- 401(k) match.\n"
            "- Generous PTO.\n"
        )
        # Vary posted_at: most jobs posted in last 30 days.
        days_ago = rng.randint(0, 30)
        posted_at = _iso(now_dt - timedelta(days=days_ago))
        job_type = rng.choice(_JOB_TYPES)
        experience = rng.choice(_EXPERIENCE)
        conn.execute(
            "INSERT INTO jobs (id, jk, title, company_id, location, salary_low, "
            "salary_high, salary_period, remote_flag, job_type, experience_level, "
            "posted_at, description, snippet, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'year', ?, ?, ?, ?, ?, ?, 'active')",
            (jid, jk_seed, title, cid, loc, sl, sh, rem, job_type,
             experience, posted_at, desc, snippet),
        )

    # ── historical applications ───────────────────────────────────────
    # ~100 historical applications spread across seekers/jobs. T03 needs
    # at least one `new` applicant on job_00003.
    seeker_ids = [f"user_{i:05d}" for i in range(1, N_SEEKERS + 1)]
    job_ids = [f"job_{i:05d}" for i in range(1, N_JOBS + 1)]
    placed = set()
    for n in range(N_HISTORICAL_APPS):
        uid = rng.choice(seeker_ids)
        jid = rng.choice(job_ids)
        if (uid, jid) in placed:
            continue
        placed.add((uid, jid))
        aid = f"application_{len(placed):05d}"
        rid = f"resume_{int(uid.split('_')[1]):05d}"
        # Status mix: 50% new, 30% reviewed, 10% rejected, 10% hired.
        r = rng.random()
        if r < 0.5:
            status, reviewed_at = "new", None
        elif r < 0.8:
            status, reviewed_at = "reviewed", _iso(now_dt - timedelta(days=rng.randint(0, 10)))
        elif r < 0.9:
            status, reviewed_at = "rejected", _iso(now_dt - timedelta(days=rng.randint(0, 10)))
        else:
            status, reviewed_at = "hired", _iso(now_dt - timedelta(days=rng.randint(0, 10)))
        applied_days = rng.randint(1, 60)
        conn.execute(
            "INSERT INTO applications (id, user_id, job_id, resume_id, phone, "
            "answers_json, status, applied_at, reviewed_at) "
            "VALUES (?, ?, ?, ?, '', '{}', ?, ?, ?)",
            (aid, uid, jid, rid, status,
             _iso(now_dt - timedelta(days=applied_days)), reviewed_at),
        )

    # Pinned T03: ensure `application_seed_t03` always exists on
    # job_00003 by user_00001 with status='new'. If a prior random
    # application already occupied that (user_id, job_id) slot, blow
    # it away first so the UNIQUE constraint doesn't fight us.
    conn.execute(
        "DELETE FROM applications WHERE user_id = ? AND job_id = ?",
        ("user_00001", "job_00003"),
    )
    aid = "application_seed_t03"
    rid = "resume_00001"
    conn.execute(
        "INSERT INTO applications (id, user_id, job_id, resume_id, phone, "
        "answers_json, status, applied_at, reviewed_at) "
        "VALUES (?, ?, ?, ?, '', '{}', 'new', ?, NULL)",
        (aid, "user_00001", "job_00003", rid,
         _iso(now_dt - timedelta(days=2))),
    )

    conn.commit()
