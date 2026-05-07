# `mantis` CLI

A first-class plan-authoring surface (#154). Run as a script after
`pip install -e .` or `pip install mantis-agent`.

```
mantis <command> [args...]
```

## Commands

### `mantis plan validate <path>`

Run the structural plan validator on a JSON micro-plan and report any
issues. Exits **0** on a clean plan, **2** if all issues are warnings,
**1** if at least one error is found.

```
mantis plan validate examples/extract_jobs.json
```

```
examples/extract_jobs.json: 3 steps
  ✓ clean — no issues
```

Read from stdin:

```
echo '{"steps":[]}' | mantis plan validate -
<stdin>: 0 steps
  ERROR   plan     empty_plan: Plan has no steps

result: 1 error(s), 0 warning(s)
```

Machine-readable output for CI gates and editor integrations:

```
mantis plan validate path.json --json
```

```jsonc
{
  "path": "path.json",
  "step_count": 3,
  "errors": [],
  "warnings": []
}
```

#### Checks

The validator inspects each plan for:

- **Plan-level**: empty plan, missing `navigate` step, missing gate after
  filter steps, pagination loop without an extraction loop.
- **Step-level**: filters declared in the objective but absent from the
  plan, unreachable loop targets, `extract_url` / `extract_data` steps
  missing the `claude_only` flag, inconsistent section assignments.

Issues are returned as ``PlanIssue`` records with ``severity``, ``code``,
``message``, ``step_index``, and ``auto_fix``. The validator and its
auto-fix pass are reusable as a library: ``from
mantis_agent.graph.plan_validator import PlanValidator``.

### `mantis plan dry-run <path>`

Walk the plan graph and print the step sequence the runner would attempt
— **no browser, no API calls, no model load**. Pure structural walk.
Use as the inner authoring loop before paying the 9–13 min Modal/Baseten
roundtrip.

```
mantis plan dry-run examples/extract_jobs.json
```

```
examples/extract_jobs.json: 3 steps
  sections: setup=1, —=2

  idx  type               flags          section     intent / target
  ---- ------------------ -------------- ----------- ----------------------------------------
  [00] navigate           —              setup       "Open a public Greenhouse-hosted careers page."
  [01] wait               —              —           "Wait for the listings to render."
  [02] loop               —              —           "→ step [?] (count=5)"
```

Annotations:

| Column | Meaning |
|---|---|
| **idx** | Zero-based step index — what the runner sees as `step_index`. |
| **type** | The step's `type` field (navigate, click, paginate, extract_url, ...). |
| **flags** | `!req` = required (halt on failure), `gate` = verification gate, `cl` = `claude_only`. |
| **section** | The step's `section` (setup / extraction / pagination / —). |
| **intent / target** | First 60 chars of `intent`, except for `loop` rows which show `→ step [N] (count=K)`. |

Out-of-range `loop_target` references emit a non-fatal `WARNING` line so
authors can spot misconfigured loops at dry-run rather than at first
execution.

`--json` emits the structured form (full step list + section rollup) for
editor integrations and CI gates.

## Streaming-agent run (legacy default)

`mantis "<task description>"` continues to work as before — running the
streaming CUA loop with a local model. The plan-authoring subcommands
short-circuit before any heavy import, so `mantis plan ...` invocations
don't load `transformers` / `torch` / `mss`.

```
mantis "Search for the latest Python 3.13 release notes and summarize"
```

See `mantis --help` for the full streaming-agent option set.

## Roadmap (#154)

- `mantis init <url> --task "<description>"` — scaffold a starter plan
  via PlanDecomposer; validate and dry-run before writing.
