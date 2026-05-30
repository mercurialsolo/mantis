# Mantis Cost Optimization — Autoresearch

This is an experiment to have an AI agent autonomously optimize the cost-per-valid-lead of the `boattrader_scrape` pipeline.

## Objective

Minimize `cost_usd / max(1, valid_leads)` for a single Modal submission against zip **33131 (Miami)** — the proven test bed where the pipeline reliably surfaces phone-bearing private-seller leads.

**Valid lead definition**: a row in `leads.csv` where the `Phone` column contains a complete dialable phone number (not "none", "not listed", empty, or a partial pattern) AND the `Seller_Type` column is `private` or empty (NOT `dealer`).

## Budget

**Hard cap: $10 cumulative `cost_usd` across all trials.** Before each trial, sum the `cost_usd` column of `experiments/results.tsv`. Once the sum is **≥ $10**, STOP. Print `Budget exhausted` and exit. The submit script enforces this same check and refuses to submit when the cap is breached.

This budget includes both the Claude API spend reported by the cost meter AND your own thinking time spent reading code (the latter doesn't count against the $10, but be efficient — long deliberations slow the loop).

## Setup

To set up a new experiment:

1. **Agree on a run tag** with the human: propose a tag like `may30`. The branch `autoresearch/cost-<tag>` must not already exist.
2. **Create the branch from current HEAD** (not necessarily main — the harness lives on whichever branch has the cost meter + leads CSV writer landed): `git checkout -b autoresearch/cost-<tag>`. Verify with `git log -1` that the new branch starts at the same commit you were on.
3. **Read the in-scope files** in full (they're small):
   - `experiments/program.md` — this file. Read it now.
   - `experiments/experiment.py` — the knob bundle. The PRIMARY file you edit.
   - `experiments/submit_one_trial.py` — read-only trial orchestrator. Understand what it does but don't modify.
   - `scripts/run_boattrader_scrape_with_proxy.py` — the underlying submit script. Read for context.
   - `plans/boattrader_scrape` — the plan. You MAY edit (plan changes affect cost).
   - `src/mantis_agent/plan_decomposer.py` — the decomposer prompt. You MAY edit `DECOMPOSE_PROMPT` (and bump `prompt_version` when you do).
   - `src/mantis_agent/grounding.py` — `ClaudeGrounding` default model. You MAY edit.
   - `src/mantis_agent/extraction/extractor.py` — `_VERIFY_ESCALATION_MODEL` default. You MAY edit.
4. **Verify environment**: `.env` should carry `ANTHROPIC_API_KEY`, `MANTIS_API_TOKEN`, `OXYLABS_USERNAME`, `OXYLABS_PASSWORD`. If anything is missing, ask the human and stop.
5. **Verify budget headroom**: read `experiments/results.tsv` (create with just the header row if missing). Sum `cost_usd`. If `>= $10`, STOP — print "Budget exhausted" and exit.
6. **Confirm and go**: confirm setup looks good, then enter the experiment loop.

## What you CAN edit

- `experiments/experiment.py` — the `CONFIG` dict. Knobs that flow through to the suite as runtime fields (extractor_model, fanout, max_cost, max_time, wait_after_load).
- `plans/boattrader_scrape` — plan prose. Add/remove heuristics, tighten gate criteria, change URL patterns.
- `src/mantis_agent/plan_decomposer.py` — `DECOMPOSE_PROMPT` (bump `prompt_version` to invalidate the plan-decompose cache).
- `src/mantis_agent/grounding.py` — `ClaudeGrounding.__init__` default `model=` argument.
- `src/mantis_agent/extraction/extractor.py` — `_VERIFY_ESCALATION_MODEL` default value.

## What you CANNOT edit

- `experiments/submit_one_trial.py` — read-only.
- `experiments/program.md` — these instructions are read-only.
- `experiments/results.tsv` — append-only via the script; never rewrite history.
- Any other file under `src/mantis_agent/` (other than the three listed above).
- Any file under `deploy/modal/` (unless you're consciously redeploying — see below).
- The `.env` file. The proxy / API keys are fixed.

## Deploy semantics

The grounding model + verifier escalation model are baked into the running Modal container at deploy time (the latter reads from `os.environ` at import). If you edit `grounding.py` OR `extraction/extractor.py`, you MUST also redeploy before the trial:

```bash
uv run modal deploy deploy/modal/modal_cua_server.py
```

If you ONLY edited `experiments/experiment.py` or `plans/boattrader_scrape` or `plan_decomposer.py`, no redeploy is needed — those changes are runtime-side and the submit script picks them up. The plan decomposer also runs locally in the submit script.

A redeploy adds ~5-10s wall and ~$0 to the trial. Cheap, but don't redeploy unnecessarily — it's a signal that you changed something model-side.

## Experimentation

Each trial is **one Modal submission** of the boattrader_scrape plan against zip **33131 Miami**.

**The goal**: minimize `$_per_valid_lead`. Lower is better.

**One concept per trial.** Don't change five knobs at once — you won't know which one moved the needle. Make one focused change, run, measure, decide. Karpathy's rule.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement from a complex change is not worth keeping. Conversely, removing a knob (or deleting a plan section) while getting equal-or-better results is a great outcome — that's a simplification win.

**The first run**: Establish the baseline. Run the trial with the current `CONFIG` and current plan unchanged. Record the result with description `"baseline"`.

## Output format

`submit_one_trial.py` prints a summary like this at the end:

```
---
commit:           abc1234
config_hash:      def5678
cost_usd:         0.4123
valid_leads:      2
$_per_valid:      0.2062
total_leads:      6
halt_reason:      ""
wall_seconds:     742.3
status:           candidate_keep
```

Extract by:
```bash
grep "^cost_usd:\|^valid_leads:\|^\$_per_valid:\|^halt_reason:\|^status:" trial.log
```

A "crash" is any of:
- `submit_one_trial.py` exits with non-zero (other than 2 which means "budget exhausted")
- The Modal run halts with `halt_reason` in {`required_failed:*`, `cf_challenge`, `external_pause`}
- `valid_leads == 0` AND `total_leads == 0` (didn't extract anything at all)

`valid_leads == 0` with `total_leads > 0` is NOT a crash — it just means the trial extracted listings but none were phone-bearing private sellers. That's data; treat as `discard` if it's a regression vs the current best.

## Logging results

Append one row per trial to `experiments/results.tsv` — **tab-separated, NOT comma** (commas break in descriptions). 7 columns:

```
commit	cost_usd	valid_leads	$_per_valid	total_leads	status	description
```

- `commit`: git short SHA (7 chars) — the commit of THIS trial
- `cost_usd`: 4 decimal places
- `valid_leads`: integer
- `$_per_valid`: 4 decimal places. Use `999.9999` for crashes / zero-lead runs.
- `total_leads`: integer
- `status`: `keep` | `discard` | `crash`
- `description`: 1-line text describing what this trial changed vs the prior `keep`. Be specific. Bad: "tried haiku". Good: `"extractor: sonnet → haiku; expected lower extract cost"`.

Example:
```
commit	cost_usd	valid_leads	$_per_valid	total_leads	status	description
abc1234	0.4523	2	0.2262	6	keep	baseline — sonnet grounding, sonnet verifier escalation, haiku extractor
def5678	0.3812	2	0.1906	5	keep	verifier_escalation: sonnet → haiku
ghi9012	0.4001	0	999.9999	0	crash	fanout=8 + max_cost=$0.5 — hit budget cap before extract
jkl3456	0.3920	3	0.1307	7	keep	added "skip Featured cards" rule to plan step 4
```

Do NOT commit `results.tsv` to git. It is gitignored. Leave it untracked.

## The experiment loop

LOOP UNTIL `sum(cost_usd) >= $10` OR HUMAN INTERRUPTS:

1. **Check budget**. Sum `cost_usd` across `results.tsv`. If `>= $10`, STOP — print `Budget exhausted` and exit.
2. **Look at git state**. Which branch? What commit are you on? What does the last `keep` row in `results.tsv` say about the current best?
3. **Pick a change**. Edit one of the allowed files. One concept per trial.
4. **`git commit`** with a short message describing the change.
5. **If you edited grounding.py or extractor.py**: redeploy. `uv run modal deploy deploy/modal/modal_cua_server.py 2>&1 | tail -2`.
6. **Run the trial**: `uv run python experiments/submit_one_trial.py > /tmp/trial.log 2>&1`. Redirect everything, do NOT tee.
7. **Read the result**: `grep "^cost_usd:\|^valid_leads:\|^\$_per_valid:\|^halt_reason:\|^status:" /tmp/trial.log`. If grep is empty, treat as crash (Python error before the summary printed). `tail -n 50 /tmp/trial.log` to debug.
8. **Decide keep / discard / crash**:
   - If `$_per_valid` improved AND `valid_leads > 0` → `keep`. Leave commit. Advance the branch.
   - If `$_per_valid` equal-or-worse but `valid_leads > 0` → `discard`. `git reset --hard HEAD~1`.
   - If crashed → `crash`. `git reset --hard HEAD~1`.
9. **Append to `results.tsv`** with the right status and a 1-line description.
10. Loop.

**Crashes**: use judgment. Typo / missing import? Fix and re-run (don't log the typo-fix as a separate trial). Idea fundamentally broken? Log `crash` and move on.

**Stuck**: If you've tried 5 trials in a row with no improvement, re-read `plans/boattrader_scrape` and `plan_decomposer.py` for fresh angles. Try combinations. Try radical changes — delete a section of the plan, see what happens.

**NEVER STOP** until budget hit or human interrupts. Don't ask "should I keep going?". The human is asleep or away. They wake up to a populated `results.tsv`.

## Quick reference

```bash
# Setup (once)
git checkout -b autoresearch/cost-may30

# Baseline
uv run python experiments/submit_one_trial.py > /tmp/trial.log 2>&1
grep "^cost_usd:\|^valid_leads:\|^\$_per_valid:\|^halt_reason:\|^status:" /tmp/trial.log

# Loop: edit → commit → (maybe deploy) → trial → grep → append-tsv → keep-or-reset
```
