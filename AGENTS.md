# Agent instructions for Mantis

## Read first: compounded learnings from prior sessions

Before debugging the CUA runner, plan execution, deploy pipelines, or
verification cycles, read the auto-memory index:

```
~/.claude/projects/-Users-barada-Sandbox-Mason-mantis/memory/MEMORY.md
```

That index points to ~25+ feedback / project files capturing failure
modes, triage flows, deploy-cycle caveats, and architectural decisions
that compounded across multi-day stress-test sessions on the
staff-crm-long plan and predecessors.

If you're investigating a halt on staff-crm-long or a similar plan,
start with these two:

- **`project_cua_verification_playbook.md`** — 16-step triage flow:
  pre-flight checks → submit → app-id resolve → status triage →
  per-step triage → signature grep → cost economics → stop criteria.
- **`project_staff_crm_long_pr_jkl_validation.md`** — grep signatures
  proving each merged CUA fix (PR-G..PR-M) fires as designed in prod
  logs.

Grep `MEMORY.md` for a one-line description per entry; open the
linked file for full body. Update memories when you learn something
new; never write into `MEMORY.md` directly — it's an index, not a
memory store.

## How verification cycles work

Mantis CUA fixes are validated by:
1. Local unit tests in `tests/test_form_handler.py` etc. — fast gate.
2. Modal redeploy + plan rerun against `staffai-test-crm.exe.xyz`
   (the staff-crm-long stress-test fixture) — real-world gate.
3. Grep modal logs for the PR-specific signature line. Absence of the
   signature means the deployed code doesn't have your change (often
   because of Modal warm-container or new-app-id-per-deploy caveats
   documented in memory).

Tab-walk fallbacks, native-select fast-path, retry-context, pointer-
retry, and token-subset matching all live in
`src/mantis_agent/gym/step_handlers/form.py`. Both Modal and Baseten
deployments share this code via `src/mantis_agent` (Truss
`external_package_dirs` for Baseten, `add_local_python_source` for
Modal).

## Deploy targets

- **Modal**: `modal deploy deploy/modal/modal_cua_server.py`. Each
  deploy rotates the app id — log fetch must use `modal app logs
  mantis-cua-server` (by name) or `modal container list` to resolve
  the active id.
- **Baseten**: `uvx truss push deploy/baseten/holo3 --no-cache
  --promote --wait --deployment-name baseten-holo3-workload-<sha>
  --include-git-info`. Always `--no-cache` after touching
  `src/mantis_agent` (the package is shipped via
  `external_package_dirs` and the build hash skips it). Deployment
  names must be unique per push — append a git SHA.

## Where things live

- `src/mantis_agent/gym/`: CUA execution loop, step handlers, runner.
- `src/mantis_agent/extraction/`: Claude-based field extraction.
- `src/mantis_agent/baseten_server/`: FastAPI server for Baseten.
- `deploy/modal/`: Modal app definitions.
- `deploy/baseten/`: Truss configs (one per model).
- `plans/`: micro-plan JSON files (gitignored; per-user).
- `tests/`: pytest test suite. `pytest -q` runs in parallel via xdist.
- `docs/`: mkdocs-strict source.

## Conventions

- **CUA-only at runtime**: handlers must be screenshot-grounded.
  CDP is for *dispatching* vision-derived actions, never for
  *deriving* them (see `feedback_cua_no_dom_access.md`).
- **Generic primitives, plan-specific via decomposer**: no
  vertical-specific (BoatTrader/CRM/etc) heuristics in step
  handlers. See `feedback_legacy_fallback_smell.md` and
  `feedback_no_plan_specific_framework.md`.
- **Tests**: `pytest tests/ -q` (xdist parallel). Some tests are
  pinned to xdist_groups to avoid subprocess contention.
- **CI flakes**: same test family flaking twice = root cause, not
  retry-shrug. Bump timeouts once; if still flaky, ask if the wait is
  needed at all (`feedback_repeat_flake_root_cause.md`).
