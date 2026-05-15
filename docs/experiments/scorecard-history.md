# Scorecard history — schema

`SCORECARD_HISTORY.jsonl` is the append-only metric log produced by
`training/promotion_scorecard.py`. One line is one scorecard run.

The point of the file is to answer two questions cheaply:

1. **Did the last PR move a gate?** Diff the two most recent records.
2. **What's the long-term trend?** Walk the file by `timestamp`.

## Record shape

Every line is a single JSON object. Required fields:

| Field | Type | Source |
|---|---|---|
| `timestamp` | ISO-8601 UTC | when the scorecard was computed |
| `commit_sha` | string | `git rev-parse HEAD` at scorecard time |
| `tier` | `"base"` \| `"first_sft"` \| `"future"` | matches `Scorecard.tier` |
| `overall_passed` | bool | matches `Scorecard.overall_passed` |
| `gates` | object | `gate_name` → `{value, threshold, direction, passed, note}` |
| `metadata` | object | per `promotion_scorecard.Scorecard.metadata` (label counts etc.) |

Recommended (lets the file double as an experiment archive):

| Field | Type | Why |
|---|---|---|
| `linked_pr` | int \| null | PR # this scorecard run is measuring. `null` means "baseline / no PR under test". |
| `linked_issues` | int[] | issue numbers the PR addresses |
| `hypothesis` | string | one sentence: "we expected gate X to improve because Y" |
| `notes` | string | freeform — anomalies, environment caveats, why the scorecard skipped a gate |

Append, never edit. If a number was wrong, append a corrected record
and explain the correction in `notes`.

## Reading recipes

**Last two entries (`jq` only):**

```
tail -2 docs/experiments/SCORECARD_HISTORY.jsonl | jq '.gates'
```

**Trend on one gate:**

```
jq -r '[.timestamp, .gates.escalation_rate_max.value] | @tsv' \
    docs/experiments/SCORECARD_HISTORY.jsonl
```

**Did this PR help?**

```
jq -c 'select(.linked_pr == 415)' docs/experiments/SCORECARD_HISTORY.jsonl
```

## What lives in `gates`

Each gate object mirrors `GateResult` in
`training/promotion_scorecard.py`:

```jsonc
{
  "value": 0.62,           // measured value for this run
  "threshold": 0.55,       // tier threshold the gate was evaluated against
  "direction": "min",      // "min" — value ≥ threshold passes; "max" — value ≤ threshold passes
  "passed": true,
  "note": ""               // populated when an input was missing → gate skipped
}
```

The current gate set (see [operations/model-promotion.md](../operations/model-promotion.md)):

| Gate | Direction |
|---|---|
| `task_pass_rate` | min |
| `parser_validity` | min |
| `grounding_accuracy` | min |
| `forbidden_region_avoidance` | min |
| `loop_rate_max` | max |
| `gallery_recovery_rate` | min |
| `escalation_rate_max` | max |
| `done_completeness` | min |
| `cost_per_success_usd_max` | max |

Gates added later to `DEFAULT_THRESHOLDS` automatically flow into new
records — `awesome-pages` doesn't care, but `jq` queries that hard-code
old gate names will need updating.

## Example records

The two records in `SCORECARD_HISTORY.jsonl` today are templates
showing the full shape — first a `_template` baseline, then a
hypothesis-shaped entry. Replace `_template: true` and the placeholder
numbers when you ship the first real scorecard run.

## Wiring in CI

`promotion_scorecard.py` already exits non-zero when `overall_passed`
is false. The CI gate that bumps shadow-deploy share should:

1. Run the scorecard.
2. Append the result to this file in the same commit.
3. Fail the merge if any gate regressed against the *previous* line in
   the JSONL (not against the tier threshold — that's the script's
   job).

The regression check is one `jq` invocation; we'll wire it into the
release workflow once we have enough lines to make the check
meaningful.
