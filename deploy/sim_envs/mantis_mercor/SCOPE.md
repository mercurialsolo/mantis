# `mantis_mercor` — Scope

Synthetic mirror of `mercor.com` — an AI talent marketplace where
candidates create profiles and apply to AI-evaluator / contractor
roles, and clients (signed-in with role=client) review applicants +
maintain shortlists.

## In-scope pages

- `/` — marketing home (hero "Shape the frontier of AI", stats strip,
  Latest roles grid, footer).
- `/jobs` — open roles list with filter rail + role cards.
- `/jobs/<id>` — role detail (description, skills, comp, Apply CTA).
- `/apply/<id>` — multi-step apply flow:
    1. Profile (headline, skills, hourly rate, availability)
    2. Resume upload (text paste fallback — no real file storage)
    3. Screening Q&A (per-role questions)
    4. Review
    5. Submit → confirmation banner
- `/login`, `/signup` — auth (email + password only; no OAuth).
- `/dashboard` — role-sensitive:
    - Candidate: My applications + status column.
    - Client: Inbox of applications for my postings, shortlist count,
      review queue.
- `/profile` — candidate profile editor (headline, skills,
  hourly rate, availability).

## In-scope interactions

1. **Apply flow** — click Apply on a role card →
   multi-step apply form → submit. Side-effects:
   `applications` row inserted, status=`submitted`, audit_log row.
2. **Shortlist** — client clicks Add to Shortlist on a candidate row
   on `/dashboard` (client view). Side-effects:
   `shortlist_entries` row, audit_log row.
3. **Decline** — client moves a pending application to `rejected`
   with a reason text. Side-effects: applications.status update,
   audit_log row.
4. **Profile edit** — candidate edits headline + skills + rate +
   availability → saves. Side-effects: candidate_profiles row update,
   audit_log row.

## Out-of-scope (explicitly NOT mirrored)

- Real brand photography → deterministic initial-block avatars only.
- Analytics / advertising / tracking scripts.
- Third-party widgets (intercom, hotjar, etc.).
- OAuth / social login (email + password only).
- Payment processing — payments table is read-only seed data.
- Real-time websockets / SSE.
- A/B variant flags — canonical variant only (the one the live site
  shows for an anonymous visitor).
- Real Inter font from Google Fonts — system stack only at runtime.

## Done bar

Section-by-section structural fidelity vs `https://www.mercor.com/`:

- Visual: structural deltas ≤ 2px on hero / nav / Latest-roles card
  grid; palette matches captured spec (indigo-600 `#4F46E5`, grays
  100/200/400/500/600/800).
- Interaction: each of the 4 interactions writes the matching
  audit_log row(s) and the matching DB state change. Oracles
  T01..T03 pass.
- Smoke: `scripts/smoke.py` boots app and asserts all routes 200
  + oracles dispatchable.

Initial pass is "close, not exact" — see `FIDELITY.md` for the
section-by-section match log and remaining gaps.
