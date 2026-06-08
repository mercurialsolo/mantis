# `/apply/<id>` — Multi-step apply

5 steps, persisted draft per step:

1. Profile — headline, skills (comma-separated), hourly rate.
2. Resume — paste raw resume text (text area). No upload widget.
3. Screening Q&A — 2-3 free-text questions per role.
4. Review — read-only summary of all fields.
5. Submit — POST that writes the application + audit_log row, then
   shows a confirmation banner.

Each step has Back + Next buttons; Back preserves filled fields.

Auth: routes require login. Anonymous user hitting `/apply/<id>` is
303'd to `/login?next=/apply/<id>` (when ENV_REQUIRE_AUTH=1) or
auto-acts-as the seeded canonical candidate (when 0).

## Mirror priorities

- Persisted draft (in-memory keyed by user_id+role_id) — Back must
  preserve fields.
- Confirmation banner copy: `Application submitted — we'll be in
  touch.`
