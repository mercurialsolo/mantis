# Failures

A consolidated catalog of failure modes observed running CUA plans in
production — what they look like, what causes them, and what the
current mitigation is. Maintained for operators who need to triage a
failing run without reading the source.

| Page | What's in it |
|---|---|
| [Known issues](known-issues.md) | One section per `failure_class` plus long-chain-specific modes not yet in the taxonomy. Linked to the issues that surfaced them and the PRs that fixed them. |

## How to use this catalog

1. Open a `result.json` from a failed run and read its top-level
   `failure_class` field.
2. Find that class in [known-issues.md](known-issues.md) — the
   "Observed instances" column links to the GitHub issues for past
   reports.
3. The "Mitigation" column points at the in-tree fix (`docs/reference/…`
   policy, env var, plan-shape change, etc.).

## How to extend

When a PR fixes a bug:

1. Add a row to the relevant `failure_class` section in
   [known-issues.md](known-issues.md). One line per concrete instance,
   linking to the issue and PR.
2. If the failure didn't fit any existing class, add a new subsection
   under **Modes not yet in the taxonomy** with the same shape, and
   open a follow-up issue to extend `gym/failure_class.py`.

Both happen in the **same PR** as the code fix — the catalog only stays
honest if it ships with the code that drained the bug.
