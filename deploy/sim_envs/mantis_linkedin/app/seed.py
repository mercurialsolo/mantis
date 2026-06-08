"""Deterministic seed for mantis-linkedin.

Seeded ``random.Random`` only — no ``faker``. Same SEED → identical IDs
+ identical column values.

Counts (per SEED=42):
  - 50 users (user_00001 = demo acting subject)
  - ~150 experience rows
  - ~100 education rows
  - ~250 skill rows
  - ~80 connections (mix of accepted/pending so the UI has invitations visible)
  - 30 posts
  - ~60 comments, ~120 reactions
  - 8 messaging threads + ~60 messages
  - 25 jobs
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from typing import Any

from . import auth, db

FAKE_NOW_DEFAULT = "2026-06-08T09:00:00Z"

# Compact lists for deterministic generation.
_FIRST_NAMES = [
    "Aarav", "Aiko", "Alex", "Amir", "Ana", "Ben", "Bianca", "Chen", "Daniel",
    "Diana", "Eli", "Emma", "Faisal", "Fatima", "Gabriel", "Grace", "Hassan",
    "Hina", "Ivan", "Jasmin", "Jin", "Karim", "Kira", "Lena", "Liam", "Mei",
    "Nadia", "Noah", "Olu", "Priya", "Qing", "Raj", "Rosa", "Sam", "Sara",
    "Tara", "Thabo", "Uma", "Viktor", "Wei", "Yasmin", "Zara", "Demo", "Pat",
    "Quinn", "Rio", "Sky", "Toni", "Val", "Wren",
]
_LAST_NAMES = [
    "Patel", "Smith", "Garcia", "Chen", "Kim", "Cohen", "Singh", "Brown",
    "Khan", "Davis", "Nakamura", "Rossi", "Lopez", "Müller", "Ito", "Adeyemi",
    "Ng", "Silva", "Park", "Olsen", "Hassan", "Yoon", "Lebedev", "Nguyen",
    "Oliveira", "Schmidt", "Anand", "Diaz", "Berg", "Ahmed", "Hoffmann",
    "Sato", "Tan", "Romero", "Cruz", "Fischer", "Walker", "Reyes", "Bauer",
    "Ricci", "Vogel", "Mohamed", "Marin", "Yu", "Demir", "Karlsson",
    "Whitaker", "Cohn", "Tanaka", "Singh-Roy",
]
_HEADLINES = [
    "Senior Software Engineer @ Acme",
    "Product Manager • B2B SaaS",
    "Design Lead, Platform",
    "Data Engineer specialising in real-time pipelines",
    "Director of Engineering | Building distributed systems",
    "ML Researcher → Applied Scientist",
    "Founder & CEO at NorthForge",
    "Backend Engineer @ FintechCo",
    "Engineering Manager • Hiring",
    "Staff Developer Advocate, Open Source",
    "UX Researcher • Healthcare",
    "Full-stack Engineer • TypeScript / Rust",
    "Security Engineer (Application Security)",
    "Recruiter — Engineering & Product",
    "DevRel & community at LaunchOps",
]
_LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Toronto, ON",
    "London, UK", "Berlin, DE", "Dublin, IE", "Singapore", "Bengaluru, IN",
    "Sydney, AU", "Tokyo, JP", "Mexico City, MX",
]
_COMPANIES = [
    "Acme Corp", "NorthForge", "LaunchOps", "FintechCo", "Quanta Labs",
    "Helix Robotics", "BlueOcean Analytics", "Modal Labs", "Hyperion AI",
    "Civium", "OpenForge", "Lattice Systems", "Lumen Data",
]
_SCHOOLS = [
    "Stanford University", "MIT", "Carnegie Mellon University",
    "UC Berkeley", "University of Washington", "Georgia Tech",
    "ETH Zürich", "University of Cambridge", "IIT Bombay",
    "National University of Singapore", "University of Toronto",
]
_DEGREES = [
    ("B.S.", "Computer Science"),
    ("M.S.", "Computer Science"),
    ("B.A.", "Design"),
    ("M.B.A.", "Business Administration"),
    ("Ph.D.", "Machine Learning"),
    ("B.S.", "Electrical Engineering"),
]
_SKILLS = [
    "Python", "TypeScript", "Rust", "Go", "Kubernetes", "PostgreSQL",
    "FastAPI", "React", "GraphQL", "AWS", "GCP", "Distributed Systems",
    "System Design", "Mentoring", "Leadership", "Product Strategy",
    "Figma", "Customer Research", "Data Modeling", "SQL",
]
_HASHTAGS = [
    "hiring", "remote", "buildinpublic", "ai", "softwareengineering",
    "productmanagement", "design", "leadership", "career", "opensource",
]
_POST_TEMPLATES = [
    "Excited to share that our team just shipped a major refactor of the "
    "ingestion pipeline. P99 dropped from 480ms → 90ms. #{tag}",
    "We're hiring two senior engineers on the platform team. DM me if you "
    "want to chat about distributed systems and high-throughput APIs. #{tag}",
    "Three lessons from migrating a monolith to event-driven microservices "
    "the hard way. Thread below. #{tag}",
    "Wrote up our approach to feature-flag hygiene and the playbook for "
    "removing stale flags safely. #{tag}",
    "Joined an incredible team last month. Already learning a ton from "
    "engineers who think deeply about reliability. #{tag}",
    "Hot take: most observability dashboards are noise. Three signals you "
    "should actually wake up to. #{tag}",
    "Going to ICML this year — happy to meet folks working on RL for "
    "agentic systems. #{tag}",
    "Postmortem of a 4-hour incident we just shipped to the public — the "
    "Swiss-cheese of failures was wild. #{tag}",
    "If you're early-career and asking what to learn next: get _great_ "
    "at debugging. Not glamorous. Maximally compounding. #{tag}",
    "I'm mentoring 3 engineers this quarter through ADPlist. DM me if "
    "you're navigating the senior → staff jump. #{tag}",
]
_JOB_TITLES = [
    ("Senior Software Engineer, Backend", "Acme Corp"),
    ("Staff Engineer, Platform", "NorthForge"),
    ("Product Manager, Growth", "LaunchOps"),
    ("Engineering Manager, Payments", "FintechCo"),
    ("Senior Designer, Product", "Quanta Labs"),
    ("Data Engineer (Streaming)", "BlueOcean Analytics"),
    ("Site Reliability Engineer", "Modal Labs"),
    ("ML Engineer, Recommendations", "Hyperion AI"),
    ("Director of Engineering", "Civium"),
    ("Solutions Architect", "OpenForge"),
    ("Senior Backend Engineer, Rust", "Lattice Systems"),
    ("Staff Product Designer", "Lumen Data"),
    ("Full-stack Engineer, Early Stage", "Helix Robotics"),
    ("Developer Advocate", "Modal Labs"),
    ("Security Engineer, AppSec", "Acme Corp"),
    ("Customer Engineer", "OpenForge"),
    ("Technical Recruiter, Eng", "FintechCo"),
    ("Data Scientist, Experimentation", "BlueOcean Analytics"),
    ("Frontend Engineer, Design Systems", "Quanta Labs"),
    ("Product Marketing Manager", "LaunchOps"),
    ("Engineering Manager, Infra", "NorthForge"),
    ("ML Researcher", "Hyperion AI"),
    ("Backend Engineer, Go", "Civium"),
    ("UX Researcher", "Quanta Labs"),
    ("Founding Engineer", "Helix Robotics"),
]

_JOB_DESCRIPTION_TEMPLATE = """\
## About the role

We're hiring a {title} to join {company}'s {team} team.

You'll partner with senior engineers and product leaders to ship
high-impact features used by millions of customers worldwide. Our
stack is Python 3.12, FastAPI, Postgres 16, Kubernetes, and Modal.

## What you'll do

- Design and ship async-first services with strong reliability
  guarantees.
- Mentor 2–3 engineers and raise the bar on testing + observability.
- Own one of: ingestion, query, retention, or experimentation.

## What we're looking for

- 5+ years building production back-ends in Python or Go.
- Comfortable with distributed systems trade-offs.
- Excellent written communication.

## Nice to have

- Open-source contributions.
- Experience with high-throughput event pipelines.

## Location

{location} — hybrid or remote within the U.S.
"""


def _hid(prefix: str, idx: int, width: int = 5) -> str:
    return f"{prefix}_{idx:0{width}d}"


def _handle(name: str, rng: random.Random) -> str:
    base = name.lower().replace(" ", "-").replace(".", "")
    suffix = hashlib.sha1(
        f"{name}-{rng.random()}".encode("utf-8")
    ).hexdigest()[:4]
    return f"{base}-{suffix}"


def _wipe(conn: sqlite3.Connection) -> None:
    for tbl in (
        "audit_log",
        "job_applications", "jobs",
        "messages", "threads",
        "reactions", "comments", "posts",
        "connections",
        "skills", "education", "experience",
        "users",
    ):
        conn.execute(f"DELETE FROM {tbl}")
    # Reset autoincrement counters so seed is reproducible.
    conn.execute("DELETE FROM sqlite_sequence")


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str) -> None:
    rng = random.Random(seed_val)
    db.reset_connection()
    # The previous statement closed the connection — reconnect.
    conn = db.connect()

    _wipe(conn)

    # ── users ────────────────────────────────────────────────────────
    users: list[dict[str, Any]] = []
    for i in range(1, 51):
        if i == 1:
            name = "Demo Mantis"
            handle = "demo-mantis"
            email = "demo@mantis.example"
        else:
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            name = f"{first} {last}"
            handle = _handle(name, rng)
            email = f"{first.lower()}.{last.lower().replace('-', '')}{i:03d}@example.com"
        users.append({
            "id": _hid("user", i),
            "handle": handle,
            "name": name,
            "headline": rng.choice(_HEADLINES),
            "about": (
                "I work at the intersection of distributed systems and "
                "developer productivity. Happy to chat about anything from "
                "incident reviews to writing better postmortems."
            ),
            "location": rng.choice(_LOCATIONS),
            "email": email,
            "password_hash": auth.hash_password("mantis-demo"),
            "avatar_color": rng.choice([
                "#0a66c2", "#057642", "#915907", "#7a3e9d", "#cc1016",
                "#1d4477",
            ]),
        })
    for u in users:
        conn.execute(
            "INSERT INTO users (id, handle, name, headline, about, location, "
            "email, password_hash, avatar_color, created_at) VALUES "
            "(?,?,?,?,?,?,?,?,?,?)",
            (u["id"], u["handle"], u["name"], u["headline"], u["about"],
             u["location"], u["email"], u["password_hash"],
             u["avatar_color"], fake_now),
        )

    # ── experience: 2-4 per user ─────────────────────────────────────
    exp_idx = 0
    for u in users:
        n_roles = rng.randint(2, 4)
        for k in range(n_roles):
            exp_idx += 1
            company = rng.choice(_COMPANIES)
            title = rng.choice([
                "Software Engineer", "Senior Engineer", "Staff Engineer",
                "Engineering Manager", "Product Manager", "Designer",
                "Data Engineer", "Researcher",
            ])
            start_year = 2018 + k
            end_year = start_year + rng.randint(1, 3)
            end_date = None if k == 0 else f"{end_year}-01-01"
            conn.execute(
                "INSERT INTO experience (id, user_id, title, company, location, "
                "start_date, end_date, description, sort_idx) VALUES "
                "(?,?,?,?,?,?,?,?,?)",
                (_hid("exp", exp_idx, 6), u["id"], title, company,
                 rng.choice(_LOCATIONS),
                 f"{start_year}-01-01", end_date,
                 "Led a small team owning a critical service.", k),
            )

    # ── education: 1-3 per user ──────────────────────────────────────
    edu_idx = 0
    for u in users:
        for k in range(rng.randint(1, 3)):
            edu_idx += 1
            degree, field = rng.choice(_DEGREES)
            start_year = 2010 + rng.randint(0, 6)
            conn.execute(
                "INSERT INTO education (id, user_id, school, degree, field, "
                "start_year, end_year, sort_idx) VALUES (?,?,?,?,?,?,?,?)",
                (_hid("edu", edu_idx, 6), u["id"], rng.choice(_SCHOOLS),
                 degree, field, start_year, start_year + 4, k),
            )

    # ── skills: 4-6 per user ─────────────────────────────────────────
    skill_idx = 0
    for u in users:
        chosen = rng.sample(_SKILLS, rng.randint(4, 6))
        for k, s in enumerate(chosen):
            skill_idx += 1
            conn.execute(
                "INSERT INTO skills (id, user_id, name, endorsements, sort_idx) "
                "VALUES (?,?,?,?,?)",
                (_hid("skill", skill_idx, 6), u["id"], s,
                 rng.randint(0, 24), k),
            )

    # ── connections: scatter accepted + a few pending toward demo ───
    conn_idx = 0
    pairs_seen: set[tuple[str, str]] = set()

    def _add_conn(a: str, b: str, *, status: str, note: str = "") -> None:
        nonlocal conn_idx
        key = (a, b)
        if key in pairs_seen or a == b:
            return
        pairs_seen.add(key)
        conn_idx += 1
        accepted_at = fake_now if status == "accepted" else None
        conn.execute(
            "INSERT INTO connections (id, from_user_id, to_user_id, status, "
            "note, created_at, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (_hid("conn", conn_idx, 6), a, b, status, note, fake_now,
             accepted_at),
        )

    # demo user gets ~30 accepted connections — but user_00042 is reserved
    # as the t01 oracle target and MUST stay unconnected.
    others = [u["id"] for u in users[1:] if u["id"] != "user_00042"]
    rng.shuffle(others)
    for uid in others[:30]:
        _add_conn(users[0]["id"], uid, status="accepted")
    # incoming pending invitations to demo from 4 users
    for uid in others[30:34]:
        _add_conn(uid, users[0]["id"], status="pending",
                  note="Met at PyData NYC — would love to stay in touch.")
    # outgoing pending from demo to a couple
    for uid in others[34:36]:
        _add_conn(users[0]["id"], uid, status="pending",
                  note="Enjoyed your talk on async patterns!")
    # peer connections so the feed has authors. Keep these between non-demo,
    # non-target peers so the t01 target stays clean.
    peer_pool = [uid for uid in others if uid != users[0]["id"]]
    for _ in range(40):
        a, b = rng.sample(peer_pool, 2)
        _add_conn(a, b, status="accepted")

    # ── posts: 30 in reverse-chronological time ─────────────────────
    post_idx = 0
    for i in range(30):
        post_idx += 1
        author = rng.choice(users)
        tag = rng.choice(_HASHTAGS)
        body = rng.choice(_POST_TEMPLATES).format(tag=tag)
        # Extract hashtags (no leading #) from the body so seeded posts
        # exercise the same extraction code path as user-authored posts.
        from .util import extract_hashtags
        tags = extract_hashtags(body)
        # back-dated by i hours
        created_at = f"2026-06-08T0{(8 - (i % 8)):d}:00:00Z"
        conn.execute(
            "INSERT INTO posts (id, author_id, body, hashtags, visibility, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (_hid("post", post_idx, 5), author["id"], body,
             db.pack_json(tags), "public", created_at),
        )

    # ── comments + reactions on the first 15 posts ──────────────────
    com_idx = 0
    for i in range(1, 16):
        post_id = _hid("post", i, 5)
        for _ in range(rng.randint(1, 4)):
            com_idx += 1
            author = rng.choice(users[1:])
            conn.execute(
                "INSERT INTO comments (id, post_id, author_id, body, "
                "created_at) VALUES (?,?,?,?,?)",
                (_hid("com", com_idx, 6), post_id, author["id"],
                 rng.choice([
                     "Great post — sharing with my team.",
                     "This resonates. Thanks for writing it up.",
                     "Curious about your tooling for the migration.",
                     "Following along.",
                 ]), fake_now),
            )
        for _ in range(rng.randint(2, 6)):
            reactor = rng.choice(users[1:])
            try:
                conn.execute(
                    "INSERT INTO reactions (post_id, user_id, kind, created_at) "
                    "VALUES (?,?,?,?)",
                    (post_id, reactor["id"],
                     rng.choice(["like", "celebrate", "support", "insightful"]),
                     fake_now),
                )
            except sqlite3.IntegrityError:
                pass

    # ── threads + messages: 8 threads anchored on demo ──────────────
    for t in range(1, 9):
        peer = users[t]
        tid = _hid("thread", t, 5)
        participants = db.pack_json(sorted([users[0]["id"], peer["id"]]))
        conn.execute(
            "INSERT INTO threads (id, participants, last_message_at, "
            "created_at) VALUES (?,?,?,?)",
            (tid, participants, fake_now, fake_now),
        )
        n_msgs = rng.randint(3, 8)
        for m in range(1, n_msgs + 1):
            sender = users[0] if m % 2 == 0 else peer
            mid = f"msg_{t:03d}_{m:03d}"
            conn.execute(
                "INSERT INTO messages (id, thread_id, sender_id, body, "
                "created_at, read_at) VALUES (?,?,?,?,?,?)",
                (mid, tid, sender["id"],
                 rng.choice([
                     "Hey — good to connect!",
                     "Thanks again for jumping on the call yesterday.",
                     "Sharing the doc I mentioned: see attached.",
                     "When does your team start the next planning cycle?",
                     "Quick question on the rollout plan.",
                 ]), fake_now, fake_now),
            )

    # ── jobs ────────────────────────────────────────────────────────
    for i, (title, company) in enumerate(_JOB_TITLES, start=1):
        loc = rng.choice(_LOCATIONS)
        desc = _JOB_DESCRIPTION_TEMPLATE.format(
            title=title, company=company,
            team=rng.choice([
                "platform", "growth", "infrastructure", "data",
                "experimentation", "developer-experience",
            ]),
            location=loc,
        )
        conn.execute(
            "INSERT INTO jobs (id, title, company, location, description_md, "
            "easy_apply, promoted, applicants, posted_at) VALUES "
            "(?,?,?,?,?,?,?,?,?)",
            (_hid("job", i, 5), title, company, loc, desc,
             1 if i % 5 != 0 else 0,  # most are easy_apply, ~20% external
             1 if i % 7 == 0 else 0,
             rng.randint(8, 230),
             fake_now),
        )

    conn.commit()


def find_demo_user(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, handle, name FROM users WHERE id = 'user_00001'"
    ).fetchone()
    if not row:
        raise RuntimeError("seed did not insert user_00001")
    return dict(row)
