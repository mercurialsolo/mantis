# Profiles, login reuse, and parallelism

How to keep a logged-in session across runs, and how that interacts with running
work in parallel. This is the mental model to get right before you wire up
multi-account or multi-platform automation.

## `profile_id` is the login

A `profile_id` maps 1:1 to a **Chrome user-data-dir** on the server
(`tenants/<tenant_id>/chrome-profile/<profile_id>`). Cookies, `localStorage`, and
`IndexedDB` live in that directory and persist on the data volume across runs.

So **reusing a login = reusing the same `profile_id`**. It's automatic — once a
run on `profile_id="alice-linkedin"` logs in, the next run with the same
`profile_id` is already signed in. You do **not** need `resume_state` for this:

- **`profile_id`** controls the browser identity (cookies / login). Reused by
  simply passing the same value.
- **`resume_state`** is a *separate* axis — it reconstructs the `workflow_id`
  checkpoint (where a plan left off), not the login. Leave it `false` for plain
  login reuse.

`workflow_id` is the checkpoint identity (rotate it when the plan changes).
Legacy `state_key`, when set alone, routes to **both** `profile_id` and
`workflow_id` — prefer the split fields in new code (see the
[glossary](glossary.md) and [#341](https://github.com/mercurialsolo/mantis/issues/341)).

## Recommended convention: `<user>:<platform>`

Name profiles by who is logged in and where, e.g. `alice:linkedin`,
`alice:luma`, `bob:linkedin`. One directory per (user, platform) keeps sessions
from clobbering each other.

!!! note "Identifiers are normalized to a filesystem-safe form"
    `profile_id` (and `workflow_id` / `tenant_id`) are sanitized via
    `safe_state_key`: any character outside `A-Za-z0-9_.-` becomes `_`. So
    `alice:linkedin` is stored as `alice_linkedin`. This works fine, but two
    consequences to keep in mind:

    * `alice:linkedin` and `alice_linkedin` resolve to the **same** profile —
      don't mix the two styles for different accounts.
    * `-`, `_`, and `.` survive verbatim, so `alice-linkedin` is unambiguous if
      you'd rather avoid the rewrite. Pick one separator style and stick to it.

## Parallelism: one run per `profile_id`

Chrome locks a user-data-dir to a single process, so **you cannot run two calls
on the same `profile_id` concurrently**. The server enforces this: a second run
on a busy profile gets a **`409`**:

```
409  profile_id 'alice:linkedin' is busy; held by run_id='...'.
     Use a different profile_id or wait for the existing run to finish.
```

**Different `profile_id`s parallelize fine** — distinct (user, platform) pairs
run concurrently (Modal scales out across containers; the lock is per
`(tenant_id, profile_id)`).

So the rule of thumb:

| Situation | What to do |
|---|---|
| Distinct accounts / platforms (`alice:linkedin`, `bob:linkedin`, `alice:luma`) | Distinct `profile_id`s → **run in parallel** |
| Multiple jobs that must share **one** logged-in account | Serialize them — a **per-profile queue** (one at a time) |

Both are first-class. The client-side
[`StateKeyDispatcher`](../continuous-improvement-architecture.md)
(`experiments/holdout/state_key_dispatcher.py`) encodes exactly this choice
per call: *independent* calls auto-allocate a fresh key and fan out; *session*
calls queue FIFO on a shared key — so you can saturate distinct profiles while
still funneling same-account jobs through one queue and never hitting a `409`.

## See also

- [Chrome session reuse](chrome-session-reuse.md) — the in-container warm-browser
  cache (a different, complementary optimization).
- [Glossary](glossary.md) — `profile_id`, `workflow_id`, `state_key` definitions.
- [API reference](../api.md) — the `/v1/predict` request fields.
