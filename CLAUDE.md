# Claude Code instructions for Mantis

This repo's general agent instructions live in **`AGENTS.md`** — read
that first. Everything in there applies equally to Claude Code.

## Auto-memory pointer (Claude-specific)

Your auto-memory for this project lives at:

```
~/.claude/projects/-Users-barada-Sandbox-Mason-mantis/memory/
```

The `MEMORY.md` index there is loaded into context automatically on
session start, so you'll see the one-line summaries. The individual
files contain the bodies — open them when a memory's description
matches what you're investigating. Categories:

- **`feedback_*.md`** — corrections and validated approaches the user
  has given you across prior sessions. Apply these as standing
  guidance.
- **`project_*.md`** — facts about ongoing work, deploy state,
  experiment series, PR chains. Time-bounded; verify against the
  current code before recommending from them (memories decay).

Append new memories when you learn something compounding; never
overwrite existing entries that still apply. See the auto-memory
guidance in your top-level instructions for the full save protocol.

## Verification cycle scaffolding

When debugging a CUA halt on staff-crm-long or similar, follow the
playbook in `project_cua_verification_playbook.md`. The compounding
session log that produced it (PR-G..PR-M) is captured in
`project_staff_crm_long_pr_jkl_validation.md` and
`project_staff_crm_long_session_state_2026_05_17.md`.

## Tool-use norms specific to this repo

- Use `Read`/`Edit`/`Write` rather than `cat`/`sed` for files.
- Prefer `gh` over web fetches for GitHub PR/issue/CI inspection.
- Modal log fetches must use the app NAME (`modal app logs
  mantis-cua-server`), not a cached app id — the id rotates per
  deploy (see `feedback_modal_new_app_id_per_deploy.md`).
- Baseten redeploys after touching `src/mantis_agent` must use
  `--no-cache` and a unique `--deployment-name` (see
  `feedback_baseten_deployment_name_uniqueness.md`).
- For long-running deploys (`truss push`, `modal deploy` cold start),
  run in background and let the task-notification wake you when it
  completes — don't sleep-poll.

## Don't repeat

- Re-running `modal app stop` + `modal deploy` thinking it'll fix a
  stale-code symptom — the deploy works; check whether you're
  reading logs from the OLD app id.
- Submitting a plan as `task_suite: {steps: [...]}` directly — that
  produces a silent 0/0/0 success because the executor expects
  `_micro_plan` via `build_micro_suite`. See
  `feedback_task_suite_shape.md`.
- Bumping a CI flake timeout for the third time — the third bump is
  theater. Ask if the wait is needed at all
  (`feedback_repeat_flake_root_cause.md`).
