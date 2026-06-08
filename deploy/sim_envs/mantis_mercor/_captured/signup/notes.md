# `/signup` — Auth

Same shape as `/login` but with a role radio at the top:
"I'm a Candidate / I'm a Client" — defaults to Candidate.

## Mirror priorities

- POST creates `users` row + (for candidates) a candidate_profiles
  shell row + (for clients) a `companies` row owned by them.
- 303→/dashboard after success.
