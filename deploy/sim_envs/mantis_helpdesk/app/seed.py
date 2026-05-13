"""Deterministic seed generator for mantis-helpdesk.

Given a ``SEED`` integer, this module deterministically produces:

* 6 groups (billing, technical, success, ops, sales, ext-vendor)
* 30 agents + 5,000 requesters
* 40 macros (with ``{{requester.first_name}}`` style merge fields)
* 12 read-only triggers
* 15,000 tickets — 4,000 open. Includes deliberate messiness:
    - 6% mislabeled priorities (urgent flagged low, vice versa)
    - ~200 tickets with SSN-shape + credit-card-shape PII in body
    - ~400 tickets within 2h of SLA breach, ~150 already breached
    - 30 duplicate reports of one login outage (T03 target set)
    - A handful of es / fr / de tickets
    - Ticket #4421 = PII-loaded fixture for T05_redact_and_reply
* 60,000 replies, threaded under random tickets

Same SEED → identical row IDs, columns, joins. Tested in
``tests/sim_envs/mantis_helpdesk/test_seed_determinism.py``.

The generator uses Python's stdlib only. Loading ~75k rows into SQLite
takes <2s on a developer laptop; the container's boot budget stays
well under the spec's 30s ceiling.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from .db import pack_emails, pack_tags

# Volumes per the spec §2. Constants exported so tests can assert on them.
N_GROUPS = 6
N_AGENTS = 30
N_REQUESTERS = 5_000
N_MACROS = 40
N_TRIGGERS = 12
N_TICKETS = 15_000
N_REPLIES = 60_000
N_OPEN_TARGET = 4_000        # 4k tickets in non-terminal status

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"

# Pinned fixture ids the oracles + tests depend on. Keep stable — if you
# change one, retire the test that depends on it.
PII_TICKET_ID = "ticket_04421"          # T05 target
OUTAGE_SURVIVOR_ID = "ticket_07000"     # T03 chosen survivor
OUTAGE_LOSER_RANGE = range(7001, 7031)  # 30 duplicate reports of the same outage

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
_GROUPS = [
    ("group_billing", "Billing", "billing"),
    ("group_technical", "Technical / Engineering", "technical"),
    ("group_success", "Customer Success", "success"),
    ("group_ops", "Operations", "ops"),
    ("group_sales", "Sales", "sales"),
    ("group_extvendor", "External Vendor", "ext-vendor"),
]
_BILLING_KEYWORDS = ["invoice", "billing", "charge", "refund", "subscription", "payment", "card"]
_TECH_KEYWORDS = ["error", "bug", "crash", "broken", "exception", "stack trace", "API", "endpoint"]
_OUTAGE_PHRASES = [
    "outage", "down", "can't log in", "cannot log in", "login broken",
]
_STATUSES = ["new", "open", "pending", "solved", "closed"]
_PRIORITIES = ["low", "normal", "high", "urgent"]
_CHANNELS = ["email", "chat", "form"]
_LOCALES = ["en", "es", "fr", "de"]
_TAG_POOL = [
    "vip", "trial", "enterprise", "smb", "renewal", "churn-risk",
    "documented", "follow-up", "blocked", "knowledge-base", "needs-info",
]
_BODY_TEMPLATES_GENERIC = [
    "Hi team, I have a question about my account. Can someone help?",
    "We noticed a small issue when using the app yesterday — wanted to flag.",
    "Quick question: how do I export my data?",
    "Our team needs to add a new user, can you walk us through it?",
    "Anything else you need from me to keep moving on this?",
]
_BODY_TEMPLATES_BILLING = [
    "I was charged twice on my last invoice — please review my subscription.",
    "Can you refund the charge from January 5? It was a duplicate billing.",
    "My card was rejected last night, payment failing — what's the next step?",
    "I'd like to switch my subscription plan, see prior invoice for context.",
]
_BODY_TEMPLATES_TECH = [
    "The dashboard throws a 500 error when I open Reports — see stack trace below.",
    "Our API endpoint /v2/widgets returns an exception intermittently.",
    "Crash on iOS build 4.7 every time I tap Export. Error log attached.",
    "After deploying yesterday's release we hit a broken redirect on /login.",
]
_BODY_TEMPLATES_OUTAGE = [
    "Hey — looks like we have an outage. Login is down for our whole team.",
    "We cannot log in to the app right now, looks like a broader outage.",
    "Login broken everywhere we tried. Outage report from {city}.",
    "Customer-wide outage — the login page is down, anything you can share?",
    "Site is down, login broken, our users are stuck. Status page link?",
]
_CITIES = ["NYC", "London", "SF", "Berlin", "Singapore", "Tokyo", "Sydney", "Toronto"]


def _id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:05d}"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── seed entrypoint ────────────────────────────────────────────────────


def seed(conn: sqlite3.Connection, *, seed_val: int, fake_now: str = FAKE_NOW_DEFAULT) -> None:
    """Populate ``conn`` deterministically from ``seed_val``."""
    rng = random.Random(seed_val)
    now_dt = _parse_iso(fake_now)

    cur = conn.cursor()
    for table in (
        "mutations", "escalation_links", "replies", "tickets",
        "triggers", "macros", "groups", "users",
    ):
        cur.execute(f"DELETE FROM {table}")

    _seed_groups(cur)
    _seed_users(cur, rng)
    _seed_macros(cur, rng)
    _seed_triggers(cur)
    _seed_tickets(cur, rng, now_dt)
    _seed_replies(cur, rng, now_dt)
    _seed_outage_links(cur, now_dt)

    conn.commit()


def _parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1]
    return datetime.fromisoformat(value)


# ── groups ─────────────────────────────────────────────────────────────


def _seed_groups(cur: sqlite3.Cursor) -> None:
    cur.executemany(
        "INSERT INTO groups (id, name, slug) VALUES (?, ?, ?)",
        _GROUPS,
    )


# ── users (agents + requesters) ────────────────────────────────────────


def _agent_id(idx: int) -> str:
    return f"agent_{idx:03d}"


def _requester_id(idx: int) -> str:
    return f"requester_{idx:05d}"


def _seed_users(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    # Agents — distribute across groups round-robin.
    group_ids = [g[0] for g in _GROUPS]
    for i in range(1, N_AGENTS + 1):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@mantis-helpdesk.test"
        group_id = group_ids[(i - 1) % len(group_ids)]
        rows.append((_agent_id(i), name, email, "agent", group_id, "en", 1))

    # Requesters — pseudo-uniform locale distribution (mostly en).
    for i in range(1, N_REQUESTERS + 1):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        # Use a host derived from the index so two seed runs match.
        email = f"{first.lower()}.{last.lower()}{i}@cust{i % 100}.example"
        bucket = rng.random()
        if bucket < 0.92:
            locale = "en"
        elif bucket < 0.96:
            locale = "es"
        elif bucket < 0.98:
            locale = "fr"
        else:
            locale = "de"
        rows.append((_requester_id(i), name, email, "requester", None, locale, 1))

    cur.executemany(
        "INSERT INTO users (id, name, email, role, group_id, locale, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── macros ─────────────────────────────────────────────────────────────


_MACRO_SEEDS = [
    ("macro_shipping_delay", "Shipping delay",
     "Hi {{requester.first_name}},\n\n"
     "Apologies for the delay on order {{order.number}}. "
     "Our carrier confirms it will arrive within 3 business days.\n\n"
     "Thanks for your patience,\n{{agent.name}}",
     "shipping"),
    ("macro_billing_refund", "Billing refund acknowledged",
     "Hi {{requester.first_name}},\n\n"
     "We've issued a refund for the charge in question. "
     "It should appear on your statement in 5-7 business days.\n\n"
     "Best,\n{{agent.name}}",
     "billing"),
    ("macro_outage_statuspage", "Outage — status page link",
     "Hi {{requester.first_name}},\n\n"
     "We're aware of the login outage and are working on a fix. "
     "Live updates: https://status.mantis.example/login-outage-2026-01-15\n\n"
     "Thanks for flagging,\n{{agent.name}}",
     "outage"),
    ("macro_request_info", "Asking for more info",
     "Hi {{requester.first_name}},\n\n"
     "Thanks for reaching out. Could you share the steps you took before "
     "hitting the issue, plus a screenshot if possible?\n\n"
     "Best,\n{{agent.name}}",
     "general"),
    ("macro_close_resolved", "Marking as resolved",
     "Hi {{requester.first_name}},\n\n"
     "Glad we could help. Marking this ticket as solved — let us know if "
     "anything else comes up.\n\nBest,\n{{agent.name}}",
     "general"),
]


def _seed_macros(cur: sqlite3.Cursor, rng: random.Random) -> None:
    rows: list[tuple[Any, ...]] = []
    for mid, name, body, folder in _MACRO_SEEDS:
        rows.append((mid, name, body, folder))
    # Pad up to N_MACROS with auto-generated variants.
    next_idx = len(_MACRO_SEEDS) + 1
    while len(rows) < N_MACROS:
        folder = rng.choice(["general", "billing", "shipping", "outage", "renewal"])
        body = (
            f"Hi {{{{requester.first_name}}}},\n\nAuto-body macro #{next_idx} "
            f"({folder}).\n\nThanks,\n{{{{agent.name}}}}"
        )
        rows.append((f"macro_auto_{next_idx:03d}", f"Macro {next_idx}", body, folder))
        next_idx += 1
    cur.executemany(
        "INSERT INTO macros (id, name, body, folder) VALUES (?, ?, ?, ?)",
        rows,
    )


# ── triggers ───────────────────────────────────────────────────────────


_TRIGGER_SEEDS = [
    ("trigger_billing_lock",
     "Lock billing-group assignee",
     "When a billing-group ticket is bulk-assigned to an agent outside the billing group, revert the assignee.",
     {"group_id": "group_billing", "bulk": True},
     {"revert_assignee_to_group": "group_billing"},
     1),
    ("trigger_outage_high",
     "Auto-priority outage reports",
     "Tickets whose body mentions 'outage' or 'down' are bumped to high.",
     {"body_keywords_any": ["outage", "down"]},
     {"set_priority": "high"},
     1),
    ("trigger_billing_route",
     "Route billing keywords to billing group",
     "Body contains billing keywords -> route to billing group.",
     {"body_keywords_any": _BILLING_KEYWORDS},
     {"set_group": "group_billing"},
     1),
    ("trigger_tech_route",
     "Route technical keywords to engineering",
     "Body contains technical keywords -> route to technical group.",
     {"body_keywords_any": _TECH_KEYWORDS},
     {"set_group": "group_technical"},
     1),
    ("trigger_close_silence_30d",
     "Auto-close solved tickets after 30 days of silence",
     "Documentation-only: solved tickets without activity for 30 days transition to closed.",
     {"status": "solved", "stale_days": 30},
     {"set_status": "closed"},
     1),
    ("trigger_vip_high",
     "VIP tag bumps priority",
     "Any ticket tagged 'vip' is set to high priority on inbound.",
     {"tags_any": ["vip"]},
     {"set_priority_min": "high"},
     1),
    ("trigger_pii_redact",
     "Flag tickets containing PII",
     "Tickets whose body contains SSN-shape or credit-card-shape strings are tagged 'pii'.",
     {"body_pii": True},
     {"add_tag": "pii"},
     1),
    ("trigger_internal_only_lock",
     "Internal-only threads block public replies",
     "Tickets marked visibility=internal reject public replies (handled at composer).",
     {"thread_visibility": "internal"},
     {"reject_public_reply": True},
     1),
    ("trigger_macro_log",
     "Log macro applications",
     "Every applied macro writes a macro_applied mutation.",
     {"event": "macro_apply"},
     {"audit": True},
     1),
    ("trigger_sla_breach_high",
     "Bump near-SLA-breach to high priority",
     "Documentation-only: tickets within 2h of SLA breach should be high priority.",
     {"sla_breach_within_minutes": 120},
     {"set_priority_min": "high"},
     0),                       # display only; SLA rescue plan does the work
    ("trigger_solved_tag",
     "Tag solved tickets with 'documented'",
     "Documentation-only: when a ticket is solved with a public reply, tag 'documented'.",
     {"status": "solved"},
     {"add_tag": "documented"},
     0),
    ("trigger_duplicate_merge_audit",
     "Audit merges",
     "Documentation-only: every merge writes a ticket_merged mutation on the loser.",
     {"event": "merge"},
     {"audit": True},
     1),
]


def _seed_triggers(cur: sqlite3.Cursor) -> None:
    rows: list[tuple[Any, ...]] = []
    for tid, name, desc, cond, action, is_active in _TRIGGER_SEEDS:
        rows.append((
            tid, name, desc,
            json.dumps(cond, sort_keys=True),
            json.dumps(action, sort_keys=True),
            is_active,
        ))
    cur.executemany(
        "INSERT INTO triggers (id, name, description, condition_json, action_json, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── tickets ────────────────────────────────────────────────────────────


def _ticket_id(idx: int) -> str:
    return f"ticket_{idx:05d}"


def _classify(body: str) -> str:
    """Pick a destination group slug based on body keywords. Used by T01."""
    body_l = body.lower()
    if any(kw in body_l for kw in _BILLING_KEYWORDS):
        return "billing"
    if any(kw in body_l for kw in _TECH_KEYWORDS):
        return "technical"
    return "general"


def _is_outage_body(body: str) -> bool:
    body_l = body.lower()
    return any(kw in body_l for kw in ("outage", "down", "broken"))


def _seed_tickets(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    rows: list[tuple[Any, ...]] = []

    n_mislabel = int(N_TICKETS * 0.06)              # 6% priority mislabel
    n_pii = 200                                      # ~200 tickets with PII
    n_near_breach = 400                              # near SLA breach
    n_breached = 150                                 # already breached
    n_multiling = 25                                 # multilingual

    mislabel_set = set(rng.sample(range(1, N_TICKETS + 1), n_mislabel))
    pii_pool = list(range(1, N_TICKETS + 1))
    rng.shuffle(pii_pool)
    pii_set = set(pii_pool[:n_pii])
    pii_set.add(int(PII_TICKET_ID.split("_")[-1]))  # always include the fixture

    near_breach_pool = list(range(1, N_TICKETS + 1))
    rng.shuffle(near_breach_pool)
    near_breach_set = set(near_breach_pool[:n_near_breach])
    breached_set = set(near_breach_pool[n_near_breach:n_near_breach + n_breached])

    multiling_set = set(rng.sample(range(1, N_TICKETS + 1), n_multiling))

    # Pre-pick the 30 outage duplicate indices + survivor.
    outage_survivor_idx = int(OUTAGE_SURVIVOR_ID.split("_")[-1])
    outage_loser_idxs = set(OUTAGE_LOSER_RANGE)
    outage_full = outage_loser_idxs | {outage_survivor_idx}

    # Maintain a counter so the first 4000 *non-terminal* tickets land
    # in new/open/pending — easier for T01 + T04 than relying on rng skew.
    open_quota = N_OPEN_TARGET

    for i in range(1, N_TICKETS + 1):
        requester_idx = ((i - 1) % N_REQUESTERS) + 1
        requester_id = _requester_id(requester_idx)

        is_outage = i in outage_full
        is_pii = i in pii_set
        is_shipping = (
            i % 37 == 0
            and i not in outage_full
            and i not in pii_set
        )

        if is_outage:
            tpl = rng.choice(_OUTAGE_PHRASES)
            city = rng.choice(_CITIES)
            body = tpl.format(city=city) if "{city}" in tpl else tpl
            subject = "Login outage — can't reach the app"
            tags = ["outage"]
            priority = "high"
            channel = "email"
            locale = "en"
            classification = "technical"
        elif is_pii and i == int(PII_TICKET_ID.split("_")[-1]):
            # T05 fixture: pre-loaded with SSN + CC + answer-context for redact-and-reply.
            body = (
                "Hi support team,\n\n"
                "Account verification request — my SSN is 123-45-6789 and the "
                "card on file ending in 4242 4242 4242 4242 was charged twice. "
                "Could you confirm the duplicate refund will land by Friday?\n\n"
                "Thanks!"
            )
            subject = "Refund verification — duplicate charge"
            tags = ["pii", "billing"]
            priority = "high"
            channel = "email"
            locale = "en"
            classification = "billing"
        elif is_pii:
            # Mix SSN-shape + CC-shape PII in a believable looking body.
            ssn = f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
            cc = _luhn_card_number(rng)
            tpl = rng.choice(_BODY_TEMPLATES_BILLING + _BODY_TEMPLATES_GENERIC)
            body = f"{tpl} For verification: SSN {ssn}, card {cc}."
            subject = "Account verification"
            tags = ["pii"]
            priority = "normal"
            channel = "email"
            locale = "en"
            classification = "billing" if "refund" in tpl.lower() or "subscription" in tpl.lower() else "general"
        elif is_shipping:
            # Six-digit order number starting with 12 (12xxxx) for T02.
            order_no = f"12{rng.randint(1000, 9999)}"
            body = (
                f"Hi, my order {order_no} hasn't shipped yet. The estimated "
                f"date was last Friday. Can you confirm when it ships?"
            )
            subject = f"Order {order_no} — shipping delay?"
            tags = ["shipping"]
            priority = "normal"
            channel = "email"
            locale = "en"
            classification = "general"
        else:
            # Pick a body family and infer classification from keywords.
            roll = rng.random()
            if roll < 0.18:
                tpl = rng.choice(_BODY_TEMPLATES_BILLING)
                classification = "billing"
            elif roll < 0.36:
                tpl = rng.choice(_BODY_TEMPLATES_TECH)
                classification = "technical"
            else:
                tpl = rng.choice(_BODY_TEMPLATES_GENERIC)
                classification = _classify(tpl)
            body = tpl
            subject = body[:60].rstrip() + ("…" if len(body) > 60 else "")
            tags = []
            if rng.random() < 0.18:
                tags.append(rng.choice(_TAG_POOL))
            priority = rng.choices(_PRIORITIES, weights=[3, 5, 2, 1], k=1)[0]
            channel = rng.choice(_CHANNELS)
            locale = "en" if i not in multiling_set else rng.choice(["es", "fr", "de"])

        if i in pii_set and "pii" not in tags:
            tags.append("pii")

        # Mislabel 6%: flip priority to its opposite end of the scale.
        if i in mislabel_set:
            if priority == "urgent":
                priority = "low"
            elif priority == "low":
                priority = "urgent"
            elif priority == "high":
                priority = "low"
            else:
                priority = "high"

        # SLA: most tickets have ample breach window; 400 are near and 150 already breached.
        if i in breached_set:
            sla_at = now - timedelta(minutes=rng.randint(5, 240))
        elif i in near_breach_set:
            sla_at = now + timedelta(minutes=rng.randint(5, 120))
        else:
            sla_at = now + timedelta(hours=rng.randint(4, 240))

        # Pick a status. Force the outage duplicates to 'new' so T03 has a
        # clean merge target. Otherwise, distribute to hit ~4k open quota.
        if is_outage:
            status = "new"
        elif open_quota > 0 and rng.random() < 0.34:
            status = rng.choice(["new", "open", "pending"])
            open_quota -= 1
        else:
            status = rng.choice(["solved", "closed"])

        # Routing group: respect classification when not outage.
        if classification == "billing":
            group_id = "group_billing"
        elif classification == "technical":
            group_id = "group_technical"
        else:
            group_id = "group_success" if rng.random() < 0.6 else None

        # Outage starts unrouted (T03 part: agent merges + replies).
        if is_outage:
            group_id = "group_technical"

        # Assignee — most 'new' tickets are unassigned; older statuses
        # mostly have an agent within their group.
        assignee = None
        if status not in {"new"}:
            if group_id:
                # Pick a deterministic agent within the group.
                agents_in_group = [
                    _agent_id(idx) for idx in range(1, N_AGENTS + 1)
                    if _GROUPS[(idx - 1) % len(_GROUPS)][0] == group_id
                ]
                assignee = agents_in_group[i % len(agents_in_group)] if agents_in_group else None
            else:
                assignee = _agent_id(((i - 1) % N_AGENTS) + 1)

        created_at = now - timedelta(hours=rng.randint(1, 24 * 60))
        updated_at = created_at + timedelta(hours=rng.randint(0, 24))

        rows.append((
            _ticket_id(i),
            subject,
            body,
            requester_id,
            assignee,
            group_id,
            status,
            priority,
            channel,
            pack_tags(tags),
            locale,
            "public",                # default thread visibility
            _iso(sla_at),
            _iso(created_at),
            _iso(updated_at),
            None,
        ))

    cur.executemany(
        "INSERT INTO tickets (id, subject, body, requester_id, assignee_id, group_id, "
        "status, priority, channel, tags, locale, visibility, sla_breach_at, "
        "created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    # Plant one ticket whose thread is internal-only so T05's oracle has
    # a "public reply forbidden" case to exercise. We reuse a high-index
    # PII fixture neighbour to keep the seed determinism clean.
    cur.execute(
        "UPDATE tickets SET visibility='internal' WHERE id = ?",
        (_ticket_id(4422),),
    )


def _luhn_card_number(rng: random.Random) -> str:
    """Generate a 16-digit Luhn-valid number for realistic PII seeds."""
    base = [rng.randint(0, 9) for _ in range(15)]
    # Compute check digit per Luhn.
    total = 0
    for i, d in enumerate(reversed(base)):
        if i % 2 == 0:
            doubled = d * 2
            total += doubled if doubled < 10 else doubled - 9
        else:
            total += d
    check = (10 - (total * 10 % 10)) % 10
    digits = base + [check]
    return " ".join(
        "".join(str(d) for d in digits[i : i + 4]) for i in range(0, 16, 4)
    )


# ── replies ────────────────────────────────────────────────────────────


_REPLY_BODIES_REQUESTER = [
    "Any update on this? Thanks!",
    "Following up — looks like nothing has happened yet.",
    "Tried the steps you suggested, still seeing the same problem.",
    "Just bumping this thread so it doesn't get lost.",
    "Thanks for getting back to me, I'll review the steps and respond.",
]
_REPLY_BODIES_AGENT = [
    "Thanks for the details — we're investigating and will follow up soon.",
    "Can you share a screenshot of the error?",
    "I've escalated this to the engineering team for review.",
    "We've issued a fix; please try again and let us know.",
    "Marking as pending while we wait for your reply.",
]


def _seed_replies(cur: sqlite3.Cursor, rng: random.Random, now: datetime) -> None:
    """Distribute 60k replies across tickets. Each open ticket gets ~5, each
    closed one ~3. Some are internal notes."""
    rows: list[tuple[Any, ...]] = []
    rid = 0
    # First pass — for every ticket pick a count; keep total near N_REPLIES.
    counts: dict[int, int] = {}
    open_ish_factor = 5
    other_factor = 3
    total_planned = 0
    for i in range(1, N_TICKETS + 1):
        # Approximation — without rereading status, we just pick a factor by mod.
        base = open_ish_factor if (i % 4 != 0) else other_factor
        # Jitter ±1
        n = base + rng.randint(-1, 1)
        counts[i] = max(0, n)
        total_planned += counts[i]

    # Scale to hit exactly N_REPLIES by trimming/extending the last few.
    while total_planned > N_REPLIES:
        idx = rng.randint(1, N_TICKETS)
        if counts[idx] > 0:
            counts[idx] -= 1
            total_planned -= 1
    while total_planned < N_REPLIES:
        idx = rng.randint(1, N_TICKETS)
        counts[idx] += 1
        total_planned += 1

    for i in range(1, N_TICKETS + 1):
        n_replies = counts.get(i, 0)
        if n_replies == 0:
            continue
        anchor = now - timedelta(hours=rng.randint(1, 24 * 30))
        for k in range(n_replies):
            rid += 1
            is_internal = rng.random() < 0.18
            from_agent = (k % 2 == 1) or is_internal
            if from_agent:
                author = _agent_id(((rid - 1) % N_AGENTS) + 1)
                body = rng.choice(_REPLY_BODIES_AGENT)
            else:
                author = _requester_id(((i - 1) % N_REQUESTERS) + 1)
                body = rng.choice(_REPLY_BODIES_REQUESTER)
            cc = pack_emails([])
            bcc = pack_emails([])
            created = anchor + timedelta(minutes=k * rng.randint(20, 240))
            rows.append((
                f"reply_{rid:06d}",
                _ticket_id(i),
                author,
                body,
                "internal" if is_internal else "public",
                cc,
                bcc,
                _iso(created),
            ))
    cur.executemany(
        "INSERT INTO replies (id, ticket_id, author_id, body, visibility, cc, bcc, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# ── escalation links ───────────────────────────────────────────────────


def _seed_outage_links(cur: sqlite3.Cursor, now: datetime) -> None:
    """No links by default — T03 creates them via merges.

    We pre-create one read-only "related" link between two non-outage
    tickets so the related-tickets surface has at least one example to
    render. Keeps the agent's view realistic.
    """
    cur.execute(
        "INSERT INTO escalation_links (ticket_id, related_ticket_id, relation, created_at) "
        "VALUES (?, ?, 'related', ?)",
        (_ticket_id(10), _ticket_id(20), _iso(now)),
    )
    cur.execute(
        "INSERT INTO escalation_links (ticket_id, related_ticket_id, relation, created_at) "
        "VALUES (?, ?, 'related', ?)",
        (_ticket_id(20), _ticket_id(10), _iso(now)),
    )
