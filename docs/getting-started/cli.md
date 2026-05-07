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

### `mantis plan init <url> --task "<description>"`

Scaffold a starter plan from a URL + one-sentence task description. Calls
`PlanDecomposer` (one Claude API call, ~$0.005), writes the resulting
plan JSON to disk, and runs `validate` + `dry-run` inline so you see
structural feedback at scaffold time.

```
export ANTHROPIC_API_KEY="<your key>"
mantis plan init https://news.ycombinator.com \
    --task "Extract the first 10 stories with title, score, and URL"
```

Sample output:

```
Decomposing via Claude (api=claude-sonnet-4-20250514)…
  wrote news_ycombinator_com_plan.json  (4 steps)
  ✓ validator clean

Dry-run preview:
  [00] navigate           !req           setup       "Navigate to https://news.ycombinator.com"
  [01] extract_data       gate cl        setup       "Verify the front page has loaded..."
  [02] extract_data       cl             extraction  "Extract the first 10 stories..."
  [03] loop               —              —           "→ step [02] (count=1)"
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `--output, -o` | `<hostname-slug>_plan.json` | Where to write the JSON. `--overwrite` to replace existing. |
| `--model` | `claude-sonnet-4-20250514` | Claude model used for decomposition. |
| `--no-validate` | off | Skip the post-decompose validator run. |
| `--no-dry-run` | off | Skip the dry-run preview. |
| `--overwrite` | off | Allow overwriting an existing output file. |

Exit codes mirror `validate`: 0 clean, 2 warnings only, 1 errors. The
file is written even when validator finds issues — the validator's
output tells you what to fix.

## Streaming-agent run (legacy default)

`mantis "<task description>"` continues to work as before — running the
streaming CUA loop with a local model. The plan-authoring subcommands
short-circuit before any heavy import, so `mantis plan ...` invocations
don't load `transformers` / `torch` / `mss`.

```
mantis "Search for the latest Python 3.13 release notes and summarize"
```

See `mantis --help` for the full streaming-agent option set.

All three plan-authoring deliverables from #154 are now shipped
(`validate` + `dry-run` + `init`).

## Trace tooling (#155)

After enabling trace export with `MANTIS_TRACE_EXPORT_DIR`, the CLI provides
two helpers for downstream SFT/DPO labelling.

### `mantis trace label <input> --output <dir>`

Batch-label trace files with the automatic heuristic ladder. Walks
`<input>` for `*.json` files (or labels a single file) and writes one
labelled JSON per input under `<output>`. The output mirrors the input
subtree so tenant-scoped directories survive the round-trip.

```
mantis trace label /data/traces --output /data/labelled
```

```
  acme/run123.json  total=8  pos=5  neg=2  neu=1
  globex/run456.json  total=3  pos=2  neg=0  neu=1

  labelled 2 traces → /data/labelled
```

Heuristic ladder (first match wins):

| Label | Reason | Trigger |
|---|---|---|
| `negative` | `escalation` | `data` matches `cloudflare` / `page_blocked` / `REJECTED_INCOMPLETE` / `antibot` / `page_exhausted` / `scan_error` |
| `negative` | `failed_step` | `success: false` (after retries) |
| `positive` | `gate_verify_pass` | `data` starts with `gate:PASS` |
| `positive` | `success_with_observed_delta` | `success: true` with non-empty `observed_outcome` |
| `neutral` | `success_no_delta` | Anything else |

### `mantis trace review <path>`

Read-only inspection of a single trace. Prints the per-step label table
to stdout for spot-checking before committing labels to a training set.

```
mantis trace review /data/traces/__shared__/run123.json
```

```
/data/traces/__shared__/run123.json: run_id=20260506_…  tenant=—  status=completed
  totals: pos=2  neg=1  neu=0

  idx  label     reason                   type         intent / data
  ---- --------- ------------------------ ------------ ----------------------------------------
  [00] positive  gate_verify_pass         extract_data Verify the front page has loaded...
  [01] negative  escalation               click        Click first listing
  [02] positive  success_with_observed_d… click        Click second listing
```

`--json` emits the labelled trace as machine-readable output for piping
into the next step of the SFT/DPO pipeline.
