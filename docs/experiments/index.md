# Experiments

An append-only record of reliability work and its measured impact.
The point is to make "did PR #X help?" a `git diff` instead of an
operator trying to remember.

| File | What's in it |
|---|---|
| [Scorecard history](scorecard-history.md) | Schema + reading guide for `SCORECARD_HISTORY.jsonl` — the chronological log of `training/promotion_scorecard.py` runs. |
| [`SCORECARD_HISTORY.jsonl`](https://github.com/mercurialsolo/mantis/blob/main/docs/experiments/SCORECARD_HISTORY.jsonl) | The raw append-only data. One JSON object per scorecard run. |

## Two reading habits

* **Did the last PR help?** Diff the most recent two entries' `gates`
  blocks. Each gate has `value` and `passed` — a regression on any
  named gate is the answer.
* **Where is the long-term trend?** Walk the file by `timestamp`,
  plot `gates.task_pass_rate.value` and
  `gates.escalation_rate_max.value`. We don't ship a plotter yet;
  `jq` + a spreadsheet is fine for the small numbers we have today.

## Related — narrative experiment journals

`docs/experiments` is the **measured-metric** archive. For the
**narrative** form (hypothesis → fix → observation, run by run), see:

* [`reports/INDEX.md`](https://github.com/mercurialsolo/mantis/blob/main/reports/INDEX.md) —
  the 33-run staff-crm benchmark journal. Format proven; we want
  more of these for other workflows.
* [`docs/learnings.md`](../learnings.md) — the canonical long-form
  journey log (~190 commits, $175 GPU spend).

Both narrative journals stay where they are. The scorecard JSONL is
what gets *queried* when someone asks "is reliability improving?"

## When to append

Whenever `python -m training.promotion_scorecard …` runs to completion
on a real candidate (not a smoke test). The script writes a single
JSON object; teeing it into this file is the convention:

```
python -m training.promotion_scorecard \
    --eval-report reports/candidate.json \
    --shadow-summary reports/shadow_summary.json \
    --labelled-traces /data/labelled \
    --tier first_sft \
    --output reports/scorecard.json

python scripts/append_scorecard.py reports/scorecard.json \
    --commit-sha "$(git rev-parse HEAD)" \
    --linked-pr 415 \
    --hypothesis "FAILURE_DATA legend in recovery_analysis should reduce escalation_rate" \
    >> docs/experiments/SCORECARD_HISTORY.jsonl
```

`scripts/append_scorecard.py` is a thin wrapper — it doesn't exist yet;
add it when the first real scorecard run lands. Until then,
hand-construct an entry that matches the schema below.
