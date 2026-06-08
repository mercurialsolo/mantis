# `/profile` — Candidate profile editor

H1: `Your profile`

Form fields:

- `Headline` text input — e.g. "Internal Medicine Physician, 8 years"
- `Skills` (comma-separated text input)
- `Hourly rate ($/hr)` numeric input
- `Availability` select: `Full-time`, `Part-time`, `Project-based`

`Save` button (indigo-600).

## Mirror priorities

- Save writes candidate_profiles row update + audit_log
  `profile_updated`.
- Page re-renders with new state inline (303→/profile after POST).
