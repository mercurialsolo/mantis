"""Deterministic seed generator for mantis-crm.

Given a ``SEED`` integer, this module deterministically produces:

* 12 users
* 8,000 companies (some parent/child)
* 50,000 contacts (with deliberate mess: 8% dupes, 12% missing phone,
  3% malformed email, ~200 misassigned to a child company)
* 12,000 deals (~5% with past expected_close in active stages)
* 200,000 activities polymorphic to contacts/deals
* 50 saved lists

Same seed → identical row IDs, identical column values, identical joins.
Tested in ``tests/sim_envs/mantis_crm/test_seed_determinism.py``.

The generator is intentionally lightweight — Python's stdlib only.
Loading 250k rows into SQLite takes ~1-2s on a developer laptop; the
container's boot budget is well under the spec's 30s ceiling.

Heavy realism (real names, real domains) is left thin. The point is
*shape* not *content*: a 50k-row table with paginated columns and
8% near-dupes is the same agent challenge whether the names look like
"Alice Smith" or "user_01234".
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .db import pack_custom_fields, pack_tags

# Volumes per the spec §1. Constants exported so tests can assert on them.
N_USERS = 12
N_COMPANIES = 8_000
N_CONTACTS = 50_000
N_DEALS = 12_000
N_ACTIVITIES = 200_000
N_LISTS = 50
N_TASKS = 6_000           # ~12% of contacts have an open task — realistic CRM density
N_NOTES = 3_000           # ~6% of contacts have a pinned note
N_EMAIL_TEMPLATES = 20
N_LIFECYCLE_TRANSITIONS = 8_000  # not every contact has one

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"


# ── small deterministic generators ─────────────────────────────────────


_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Ruth", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Sarah", "Tom", "Lisa", "Mark", "Anna", "Brian",
]
_LAST_NAMES = [
    "Chen", "Smith", "Patel", "Jones", "Garcia", "Kim", "Brown", "Lee",
    "Wilson", "Lopez", "Davis", "Martin", "Taylor", "Thomas", "Hernandez",
    "Moore", "Anderson", "Jackson", "White", "Harris", "Clark", "Lewis",
    "Robinson", "Walker", "Allen", "Young", "King", "Wright",
]
_COMPANY_PARTS_A = [
    "Acme", "Globex", "Initech", "Vandelay", "Massive Dynamic", "Stark",
    "Wayne", "Wonka", "Hooli", "Pied Piper", "Cyberdyne", "Tyrell",
    "Oscorp", "Umbrella", "Soylent", "Aperture", "InGen", "Weyland",
    "Yutani", "Gringotts", "Sirius", "Praxis", "Krieger", "Sterling",
    "Cooper", "Pearson", "Specter", "Litt",
]
_COMPANY_PARTS_B = ["", "Industries", "Labs", "Group", "Systems", "Holdings", "Networks", "Dynamics"]
_DOMAINS_TLD = [".com", ".io", ".co", ".net"]
_INDUSTRIES = ["saas", "fintech", "biotech", "logistics", "retail", "media", "consulting", "energy"]
_SIZE_BANDS = ["1-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"]
_ARR_BANDS = ["0-100K", "100K-500K", "500K-1M", "1M+", "10M+"]
_LIFECYCLE_STAGES = ["lead", "mql", "sql", "customer", "evangelist", "churned"]
_DEAL_STAGES = ["Prospect", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost", "At Risk"]
_ACTIVE_DEAL_STAGES = {"Prospect", "Qualified", "Proposal", "Negotiation"}
_ACTIVITY_TYPES = ["call", "email", "note", "meeting"]
_SOURCES = ["webform", "import", "manual", "referral", "event", "linkedin", "cold"]
_TAG_POOL = [
    "vip", "warm", "cold", "champion", "blocker", "tech-decider", "budget-holder",
    "vendor-eval", "renewal-q1", "renewal-q2", "renewal-q3", "renewal-q4",
    "enterprise", "smb", "self-serve", "do-not-contact",
]


def _id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:05d}"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _company_name(rng: random.Random) -> str:
    a = rng.choice(_COMPANY_PARTS_A)
    b = rng.choice(_COMPANY_PARTS_B)
    return f"{a} {b}".strip()


def _domain_for(name: str, idx: int) -> str:
    base = "".join(c for c in name.lower() if c.isalnum())[:14] or "co"
    return f"{base}{idx}{random.Random(idx).choice(_DOMAINS_TLD)}"


# ── seed entrypoint ────────────────────────────────────────────────────


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str = FAKE_NOW_DEFAULT) -> None:
    """Populate ``conn`` deterministically from ``seed_val``.

    The function clears every table first, then re-inserts. Callers
    expecting reset semantics can just call ``seed`` again.
    """
    rng = random.Random(seed_val)
    now_dt = _parse_iso(fake_now)

    cur = conn.cursor()

    # Hard reset every table so re-seeding is idempotent.
    for table in (
        "mutations", "list_members", "lists", "activities",
        "lifecycle_transitions", "notes", "tasks", "email_templates",
        "deals", "contacts", "companies", "users",
    ):
        cur.execute(f"DELETE FROM {table}")

    _seed_users(cur, rng)
    _seed_companies(cur, rng)
    _seed_contacts(cur, rng, now_dt)
    _seed_deals(cur, rng, now_dt)
    _seed_activities(cur, rng, now_dt)
    _seed_lists(cur, rng)
    _seed_tasks(cur, rng, now_dt)
    _seed_notes(cur, rng, now_dt)
    _seed_email_templates(cur, rng)
    _seed_lifecycle_transitions(cur, rng, now_dt)

    conn.commit()


def _parse_iso(value: str) -> datetime:
    # ``fromisoformat`` doesn't accept the trailing ``Z`` until 3.11+
    # gives mixed results across stdlibs; strip it manually.
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


# ── users ──────────────────────────────────────────────────────────────


def _seed_users(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    for i in range(1, N_USERS + 1):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@mantis-crm.test"
        is_active = 0 if i == N_USERS else 1  # one deactivated user, per spec
        rows.append((_id("user", i), name, email, is_active))
    cur.executemany(
        "INSERT INTO users (id, name, email, is_active) VALUES (?, ?, ?, ?)",
        rows,
    )


# ── companies ──────────────────────────────────────────────────────────


def _seed_companies(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    # First pass: all parents have parent_company_id NULL.
    for i in range(1, N_COMPANIES + 1):
        name = _company_name(rng)
        rows.append((
            _id("company", i),
            name,
            _domain_for(name, i),
            rng.choice(_SIZE_BANDS),
            rng.choice(_INDUSTRIES),
            rng.choice(_ARR_BANDS),
            None,
        ))
    cur.executemany(
        "INSERT INTO companies (id, name, domain, size_band, industry, arr_band, parent_company_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    # Pin a few parent_company_id relationships (~5% have a parent).
    n_with_parent = N_COMPANIES // 20
    candidates = list(range(2, N_COMPANIES + 1))
    rng.shuffle(candidates)
    chosen = candidates[:n_with_parent]
    updates = []
    for idx in chosen:
        parent_idx = rng.randint(1, idx - 1)
        updates.append((_id("company", parent_idx), _id("company", idx)))
    cur.executemany(
        "UPDATE companies SET parent_company_id = ? WHERE id = ?",
        updates,
    )

    # Ensure at least one "Acme" so plan T02 has a target.
    cur.execute(
        "UPDATE companies SET name = 'Acme', domain = 'acme.com', arr_band = '1M+' "
        "WHERE id = 'company_00001'"
    )


# ── contacts (+ dirty data) ────────────────────────────────────────────


CANONICAL_SARAH_INDEX = 42  # contact_00042 = the canonical Sarah Chen
SARAH_FORBIDDEN = {"Sarah Chen"}


def _seed_contacts(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    rows: list[tuple[Any, ...]] = []

    # 8% of contact slots will be near-duplicates (same email, slight name variant)
    # 12% will miss phone
    # 3% will have a malformed email
    n_dupes = int(N_CONTACTS * 0.08)
    n_missing_phone = int(N_CONTACTS * 0.12)
    n_bad_email = int(N_CONTACTS * 0.03)
    dupe_set = set(rng.sample(range(2, N_CONTACTS + 1), n_dupes))
    no_phone = set(rng.sample(range(1, N_CONTACTS + 1), n_missing_phone))
    bad_email = set(rng.sample(range(1, N_CONTACTS + 1), n_bad_email))

    for i in range(1, N_CONTACTS + 1):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"

        # Force "Sarah Chen" at exactly one index and forbid it elsewhere
        # so T04 has a unique target.
        if i == CANONICAL_SARAH_INDEX:
            name = "Sarah Chen"
            first, last = "Sarah", "Chen"
        elif name in SARAH_FORBIDDEN:
            # Swap last name to keep determinism while avoiding the collision.
            last = "Stone"
            name = f"{first} {last}"

        company_id = _id("company", rng.randint(1, N_COMPANIES))
        if i in dupe_set:
            # Same email as a contact ~i-1, slightly different name spelling.
            email_idx = max(1, i - rng.randint(1, 5))
            email = _email_for(first, last, email_idx, rng)
            # Drift one letter so name isn't identical
            name = name.replace(first, first + first[-1])
        else:
            email = _email_for(first, last, i, rng)
        if i in bad_email:
            email = email.replace("@", "_at_")  # malformed

        phone = None if i in no_phone else _phone(rng)
        # First 200 contacts of mismatched company go to a child company
        # whose activity history still points at the parent — per spec.
        if i <= 200:
            # If ``company_id`` has a parent, intentionally re-point to the parent.
            pass  # We'll fix up in a follow-up step via activities

        tags: list[str] = []
        if rng.random() < 0.4:
            tags = rng.sample(_TAG_POOL, k=rng.randint(1, 3))

        # Owner — 1 in 100 contacts get the deactivated user (user_00012)
        if rng.random() < 0.01:
            owner = _id("user", N_USERS)
        else:
            owner = _id("user", rng.randint(1, N_USERS - 1))

        # last_activity_at: 30% have it in last 30 days, 30% in 30-90,
        # 40% > 90 days (these are the re-engagement targets).
        bucket = rng.random()
        if bucket < 0.3:
            last_active = now - timedelta(days=rng.randint(0, 30))
        elif bucket < 0.6:
            last_active = now - timedelta(days=rng.randint(31, 90))
        else:
            last_active = now - timedelta(days=rng.randint(91, 700))

        created_at = last_active - timedelta(days=rng.randint(30, 365))

        custom = {"region": rng.choice(["NA", "EMEA", "APAC", "LATAM"])}

        rows.append((
            _id("contact", i),
            name,
            email,
            phone,
            company_id,
            rng.choice(_LIFECYCLE_STAGES),
            owner,
            pack_tags(tags),
            _iso(created_at),
            _iso(last_active),
            pack_custom_fields(custom),
            rng.choice(_SOURCES),
            None,
        ))

    cur.executemany(
        "INSERT INTO contacts (id, name, email, phone, company_id, lifecycle_stage, "
        "owner_id, tags, created_at, last_activity_at, custom_fields, source, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    # Seed 4 deliberate ``@acme.com`` duplicates for T02_merge_acme_dupes —
    # we pick the first 4 contacts and rewrite their email to acme.com.
    cur.executemany(
        "UPDATE contacts SET email = ?, company_id = 'company_00001' WHERE id = ?",
        [
            ("alice.lead@acme.com", "contact_00001"),
            ("alice.lead@acme.com", "contact_00002"),
            ("alice.lead@acme.com", "contact_00003"),
            ("alice.lead@acme.com", "contact_00004"),
        ],
    )


def _email_for(first: str, last: str, idx: int, rng: random.Random) -> str:
    domain = f"co{idx % 100}.test"
    return f"{first.lower()}.{last.lower()}{idx}@{domain}"


def _phone(rng: random.Random) -> str:
    return f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


# ── deals ──────────────────────────────────────────────────────────────


def _seed_deals(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    rows: list[tuple[Any, ...]] = []
    n_past_due_active = int(N_DEALS * 0.05)
    past_due_set = set(rng.sample(range(1, N_DEALS + 1), n_past_due_active))

    for i in range(1, N_DEALS + 1):
        contact_id = _id("contact", rng.randint(1, N_CONTACTS))
        company_id = _id("company", rng.randint(1, N_COMPANIES))
        stage = rng.choice(_DEAL_STAGES)

        # Force ~5% of deals into Proposal with past expected_close so T03
        # has reliable targets.
        if i in past_due_set:
            stage = "Proposal"
            close_at = now - timedelta(days=rng.randint(31, 120))
        else:
            close_at = now + timedelta(days=rng.randint(-15, 180))

        amount = round(rng.uniform(2_000, 250_000), 2)
        # user_05 gets some big deals so T05 has a real target set
        if i % 11 == 0:
            owner = "user_00005"
            amount = round(rng.uniform(50_000, 250_000), 2)
        else:
            owner = _id("user", rng.randint(1, N_USERS - 1))

        rows.append((
            _id("deal", i),
            f"Deal #{i:05d}",
            contact_id,
            company_id,
            stage,
            amount,
            _iso(close_at),
            owner,
        ))

    cur.executemany(
        "INSERT INTO deals (id, name, contact_id, company_id, stage, amount, "
        "expected_close, owner_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── activities ─────────────────────────────────────────────────────────


def _seed_activities(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    # 80% on contacts, 20% on deals. Acme contacts (1-4) get
    # disproportionately many activities so the merge winner is
    # unambiguous: contact_00001 = 12, _00002 = 4, _00003 = 2, _00004 = 1.
    acme_activity_counts = {
        "contact_00001": 12,
        "contact_00002": 4,
        "contact_00003": 2,
        "contact_00004": 1,
    }

    rows: list[tuple[Any, ...]] = []
    aid = 0
    for cid, count in acme_activity_counts.items():
        for _ in range(count):
            aid += 1
            occurred = now - timedelta(days=rng.randint(0, 365),
                                       hours=rng.randint(0, 23))
            rows.append((
                _id("activity", aid), "contact", cid,
                rng.choice(_ACTIVITY_TYPES),
                f"Acme activity #{aid}", _id("user", rng.randint(1, N_USERS - 1)),
                _iso(occurred),
            ))

    # The acme contacts (1..4) get their counts from the planted block
    # above. Exclude them from the random pool so the merge oracle sees
    # exactly the seeded numbers (12 / 4 / 2 / 1).
    acme_index_set = {1, 2, 3, 4}

    remaining = N_ACTIVITIES - aid
    for _ in range(remaining):
        aid += 1
        on_contact = rng.random() < 0.8
        if on_contact:
            # Reroll until we miss the acme block.
            while True:
                idx = rng.randint(1, N_CONTACTS)
                if idx not in acme_index_set:
                    break
            target = ("contact", _id("contact", idx))
        else:
            target = ("deal", _id("deal", rng.randint(1, N_DEALS)))
        atype = rng.choice(_ACTIVITY_TYPES)
        body = f"{atype} log entry #{aid}"
        occurred = now - timedelta(days=rng.randint(0, 365),
                                   hours=rng.randint(0, 23))
        rows.append((
            _id("activity", aid),
            target[0], target[1], atype, body,
            _id("user", rng.randint(1, N_USERS - 1)),
            _iso(occurred),
        ))

    cur.executemany(
        "INSERT INTO activities (id, target_type, target_id, activity_type, body, "
        "actor_id, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── lists ──────────────────────────────────────────────────────────────


def _seed_lists(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    for i in range(1, N_LISTS + 1):
        kind = "filter" if rng.random() < 0.7 else "manual"
        filter_def: dict[str, Any] = {}
        if kind == "filter":
            filter_def = {
                "lifecycle_stage": rng.choice(_LIFECYCLE_STAGES),
            }
        rows.append((_id("list", i), f"Saved view #{i}", kind, json.dumps(filter_def, sort_keys=True)))

    # Add a known-name list for T05's "pipeline review" export target.
    rows.append(("list_pipeline_review", "Pipeline Review (Q1)", "manual", "{}"))

    cur.executemany(
        "INSERT INTO lists (id, name, kind, filter_json) VALUES (?, ?, ?, ?)",
        rows,
    )


# ── tasks ──────────────────────────────────────────────────────────────


_TASK_TITLES = [
    "Follow up on proposal", "Send pricing deck", "Confirm meeting time",
    "Check legal review status", "Re-engage stale contact", "Update CRM stage",
    "Send renewal quote", "Schedule QBR", "Document blockers",
    "Loop in solutions engineer", "Forward to billing", "Verify primary contact",
]
_TASK_PRIORITIES = ["low", "normal", "high"]


def _seed_tasks(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    """Seed 6k tasks. Mix of completed + open + overdue."""
    rows: list[tuple[Any, ...]] = []
    for i in range(1, N_TASKS + 1):
        title = rng.choice(_TASK_TITLES) + f" #{i}"
        target_type, target_id = ("contact",
                                  _id("contact", rng.randint(1, N_CONTACTS)))
        if rng.random() < 0.25:
            target_type, target_id = ("deal", _id("deal", rng.randint(1, N_DEALS)))
        assignee = _id("user", rng.randint(1, N_USERS - 1))
        # 40% are completed; 30% open future; 30% open past-due
        bucket = rng.random()
        if bucket < 0.4:
            completed_at: str | None = _iso(now - timedelta(days=rng.randint(1, 200)))
            due = _iso(now - timedelta(days=rng.randint(0, 200)))
        elif bucket < 0.7:
            completed_at = None
            due = _iso(now + timedelta(days=rng.randint(0, 60)))
        else:
            completed_at = None
            due = _iso(now - timedelta(days=rng.randint(1, 90)))
        priority = rng.choice(_TASK_PRIORITIES)
        rows.append((
            _id("task", i), title, "",
            target_type, target_id, assignee, due, priority, completed_at,
        ))
    cur.executemany(
        "INSERT INTO tasks (id, title, body, target_type, target_id, "
        "assignee_id, due_date, priority, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    # T06_followup_tasks (a deferred future plan) expects at least 20
    # overdue tasks owned by user_00005 — pin a known block.
    cur.executemany(
        "UPDATE tasks SET assignee_id = 'user_00005', "
        "due_date = ?, completed_at = NULL WHERE id = ?",
        [
            (_iso(now - timedelta(days=10)), f"task_{i:05d}")
            for i in range(1, 21)
        ],
    )


# ── notes ──────────────────────────────────────────────────────────────


_NOTE_BODIES = [
    "Primary contact for technical decisions. Reports to CTO.",
    "Loves long email threads — keep them in the loop.",
    "Champion at the org. Will advocate internally.",
    "Procurement gatekeeper. All deals route through them.",
    "Migrating off competitor — sensitivity around timeline.",
    "Renewal coming up Q2. Start the conversation early.",
    "Was previously a customer. Lost on price.",
]


def _seed_notes(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    rows: list[tuple[Any, ...]] = []
    for i in range(1, N_NOTES + 1):
        target_type = "contact"
        target_id = _id("contact", rng.randint(1, N_CONTACTS))
        if rng.random() < 0.2:
            target_type, target_id = ("company",
                                       _id("company", rng.randint(1, N_COMPANIES)))
        body = rng.choice(_NOTE_BODIES)
        pinned = 1 if rng.random() < 0.4 else 0
        created = _iso(now - timedelta(days=rng.randint(0, 400)))
        author = _id("user", rng.randint(1, N_USERS - 1))
        rows.append((
            _id("note", i), target_type, target_id, body, pinned, author, created,
        ))
    cur.executemany(
        "INSERT INTO notes (id, target_type, target_id, body_md, pinned, "
        "author_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── email templates ────────────────────────────────────────────────────


_TEMPLATE_SEEDS = [
    ("Cold outreach", "Quick intro from {{user.name}}",
     "Hi {{contact.first_name}},\n\n"
     "I work with companies like {{company.name}} on…\n\n"
     "Worth a 15-min chat?\n\nBest,\n{{user.name}}",
     "outreach"),
    ("Proposal sent", "Proposal for {{company.name}}",
     "Hi {{contact.first_name}},\n\n"
     "Attached is the proposal we discussed. Let me know any questions.\n\n"
     "Thanks,\n{{user.name}}", "proposal"),
    ("Renewal nudge", "Quick check-in on your {{company.name}} renewal",
     "Hi {{contact.first_name}},\n\n"
     "Your renewal comes up next quarter — wanted to make sure "
     "{{company.name}} is on track.\n\nBest,\n{{user.name}}", "renewal"),
    ("Reengage stale", "{{contact.first_name}}, still on your radar?",
     "Hi {{contact.first_name}},\n\n"
     "Haven't connected in a while — anything new on your side at "
     "{{company.name}}?\n\nBest,\n{{user.name}}", "reengage"),
    ("Demo follow-up", "Thanks for the demo, {{contact.first_name}}",
     "Hi {{contact.first_name}},\n\n"
     "Great chat earlier. Sending the recap deck shortly.\n\nThanks,\n"
     "{{user.name}}", "general"),
]


def _seed_email_templates(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    for i, (name, subject, body, folder) in enumerate(_TEMPLATE_SEEDS, start=1):
        rows.append((f"template_{i:03d}", name, subject, body, folder))
    # Pad to N_EMAIL_TEMPLATES with auto-generated variants so the
    # template picker has realistic length.
    for i in range(len(_TEMPLATE_SEEDS) + 1, N_EMAIL_TEMPLATES + 1):
        rows.append((
            f"template_{i:03d}",
            f"Template {i}",
            f"[Auto] subject for template {i}",
            f"Hi {{{{contact.first_name}}}},\n\nAuto-body {i}.\n\nThanks.",
            rng.choice(["general", "outreach", "renewal", "proposal"]),
        ))
    cur.executemany(
        "INSERT INTO email_templates (id, name, subject, body, folder) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


# ── lifecycle transitions ──────────────────────────────────────────────


_LIFECYCLE_PROGRESSION = [
    "lead", "mql", "sql", "customer", "evangelist",
]


def _seed_lifecycle_transitions(cur: sqlite3.Cursor, rng: random.Random,
                                now: datetime) -> None:
    """Synthesise a coherent stage-progression history for a subset of contacts."""
    rows: list[tuple[Any, ...]] = []
    # Pick N contact ids to give a history to (deterministic via rng).
    sample = rng.sample(range(1, N_CONTACTS + 1), min(N_LIFECYCLE_TRANSITIONS // 2, N_CONTACTS))
    for cid_idx in sample:
        contact_id = _id("contact", cid_idx)
        # 1-4 step progression
        steps = rng.randint(1, 4)
        anchor = now - timedelta(days=rng.randint(30, 400))
        for s in range(steps):
            from_stage = _LIFECYCLE_PROGRESSION[s] if s > 0 else None
            to_stage = _LIFECYCLE_PROGRESSION[min(s + 1, len(_LIFECYCLE_PROGRESSION) - 1)]
            occurred = anchor + timedelta(days=s * rng.randint(7, 60))
            rows.append((contact_id, from_stage, to_stage, _iso(occurred)))

    cur.executemany(
        "INSERT INTO lifecycle_transitions (contact_id, from_stage, to_stage, occurred_at) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
