# Model promotion scorecard (#183)

Before a fine-tuned Holo3 checkpoint serves more than its initial
shadow share, it has to clear a named scorecard. The scorecard composes
the artefacts the rest of the continual-fine-tuning pipeline already
emits — the eval report (#155 step 4), shadow analytics (step 5), and
labelled traces (step 2) — and reports pass/fail per gate at one of
three tiers.

## Tiers

| Tier | Use |
|---|---|
| `base` | bare-minimum — no worse than Holo3 stock weights |
| `first_sft` | first fine-tune off base; the typical promotion gate |
| `future` | aspirational long-term target |

## Gates

| Gate | Direction | Reads from | What it catches |
|---|---|---|---|
| `task_pass_rate` | min | eval report | held-out task success rate |
| `parser_validity` | min | override / labeller | tool-call shape regression |
| `grounding_accuracy` | min | override / labeller | grounded clicks landing on usable elements |
| `forbidden_region_avoidance` | min | labeller (escalation = inverse signal) | clicks avoid photos / ads / social / off-site |
| `loop_rate_max` | max | labeller | repeated-action loops |
| `gallery_recovery_rate` | min | override | lightbox / gallery trap recovery |
| `escalation_rate_max` | max | shadow analytics | brain ladder Holo3 → Claude rate |
| `done_completeness` | min | override | structured `done()` summary present |
| `cost_per_success_usd_max` | max | eval report | $ per successful held-out task |

`min` gates pass when value ≥ threshold; `max` gates pass when value ≤ threshold.

Default thresholds are conservative — operators tune by editing
`DEFAULT_THRESHOLDS` in `training/promotion_scorecard.py` or by passing
their own override map into `evaluate(thresholds=...)` from Python.

## Usage

```
python -m training.promotion_scorecard \
    --eval-report reports/candidate.json \
    --shadow-summary reports/shadow_summary.json \
    --labelled-traces /data/labelled \
    --tier first_sft \
    --output reports/scorecard.json
```

Exit code is **0** when every gate passes at the chosen tier, **1**
otherwise. Drop into CI to gate the next traffic-share bump.

Override individual metrics from the command line for any signal the
artefacts don't yet expose:

```
python -m training.promotion_scorecard \
    --eval-report reports/candidate.json \
    --metric parser_validity=0.99 \
    --metric grounding_accuracy=0.78 \
    --tier first_sft
```

## Output shape

```jsonc
{
  "tier": "first_sft",
  "overall_passed": true,
  "gates": [
    {"name": "task_pass_rate", "value": 0.62, "threshold": 0.55,
     "direction": "min", "passed": true, "note": ""},
    {"name": "escalation_rate_max", "value": 0.04, "threshold": 0.05,
     "direction": "max", "passed": true, "note": ""},
    {"name": "parser_validity", "value": 0.0, "threshold": 0.98,
     "direction": "min", "passed": true,
     "note": "input missing — gate skipped"}
  ],
  "metadata": {
    "label_step_count": 312,
    "label_reason_counts": {"escalation": 12, "gate_verify_pass": 84, ...}
  }
}
```

When an artefact is missing, the corresponding gate is **skipped**
(``note`` says so) and ``overall_passed`` is unchanged. That way the
script is useful at every stage of the pipeline — operators don't need
the full set of artefacts to start gating.

## Where this fits

```
[5] eval        eval_harness.run_eval → reports/candidate.json
[5] shadow      shadow_analytics.aggregate → reports/shadow_summary.json
[5] label       mantis trace label → /data/labelled/
                                     ↓
                    promotion_scorecard.evaluate
                                     ↓
              reports/scorecard.json   exit 0 / 1
                                     ↓
[6] deploy        bump shadow share when scorecard.overall_passed
```

The scorecard is intentionally a **composer** — it doesn't run any
new evaluations itself. New gates land by extending
`DEFAULT_THRESHOLDS` + the `_GATES` list, with the metric source wired
either in `evaluate()` or via `--metric` overrides.
