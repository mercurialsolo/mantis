"""mantis-crm — dense tables + record detail + bulk ops on dirty data.

Per #332 / docs/envs/SPEC.md §1, the env approximates a Salesforce /
HubSpot shape with realistic mess (8% duplicate contacts, 12% missing
phones, deals with past-due close dates, etc.). The shape exists to
expose real agent failure modes, not to model a real CRM.

Stack: stdlib + FastAPI + Jinja2 + SQLite. No JS framework, no client-
side state. The agent navigates plain HTML pages, fills forms, clicks
links. Pagination is real (50/page). Bulk-edit is a multi-step modal.

Determinism rules:

* All entity IDs are derived from a single seeded ``random.Random``.
  Same ``SEED`` → identical rows, identical IDs, identical joins.
* The clock is frozen at ``FAKE_NOW`` (default ``2026-01-15T09:00:00Z``).
  Helpers in :mod:`app.db` substitute it for ``datetime.utcnow``.
* No background jobs, no auto-decay, no realtime updates.
"""
