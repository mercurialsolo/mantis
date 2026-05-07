# Continual fine-tuning pipeline (#155)

A repeatable loop for turning production traces into improved Holo3
weights. Each link in the pipeline is independently runnable; outputs
are JSON / JSONL on disk so the stages can be moved between machines
(local triage box → S3 → A100 trainer) without changing the schema.

```
[1] export    runs/* → /data/traces/<tenant>/<run_id>.json
                       — each completed run, gated on MANTIS_TRACE_EXPORT_DIR
[2] label     mantis trace label → /data/labelled/<tenant>/<run_id>.json
                       — heuristic positive / negative / neutral
[3] convert   training/convert_labelled_traces.py → distill.jsonl
                       — Holo3 chat format, label-filtered
[4] train     training/train_holo3_distill.py → weights/
                       — single-A100 SFT
[5] deploy    swap weights into the runtime; shadow-test against
              current production at 5% traffic before full rollout
```

Steps 1–3 ship today. Steps 4 (recipe wiring) and 5 (shadow-deploy
harness) are tracked as separate deliverables under #155.

## 1. Trace export

Set the env var on the runtime container:

```
MANTIS_TRACE_EXPORT_DIR=/data/traces
MANTIS_TRACE_INCLUDE_SCREENSHOTS=true     # required for SFT — image+text pairs
```

Every completed / halted / cancelled / paused run writes one JSON file
at `/data/traces/<tenant>/<run_id>.json`. With screenshots enabled,
PNGs land at `/data/traces/<tenant>/<run_id>_screens/<NNNN>.png`.

Empty tenant ids fall back to `__shared__/`. See
[env-vars.md → Trace export](../reference/env-vars.md#trace-export-155)
for the exact field semantics.

## 2. Label

```
mantis trace label /data/traces --output /data/labelled
```

Apply the heuristic ladder (see
[CLI → Trace tooling](../getting-started/cli.md#trace-tooling-155)).
Each step lands in exactly one of:

- `positive` — gate verify pass, success-with-observed-delta
- `negative` — escalation event, failed step
- `neutral` — success without an observed delta (filtered out by
  default in step 3)

To spot-check a single trace before committing the labels:

```
mantis trace review /data/labelled/acme/run123.json
```

## 3. Convert to SFT chat format

```
python training/convert_labelled_traces.py \
    --traces /data/labelled \
    --screenshots-root /data/traces \
    --output training/data/labelled_distill.jsonl
```

By default keeps `label=positive` only (conservative SFT). For
DPO-style preference pairs, pass `--keep-labels positive,negative` —
each negative row will be emitted as a "rejected" candidate that the
caller pairs with a "chosen" answer downstream.

Append to the standing distill set:

```
cat training/data/labelled_distill.jsonl \
    >> training/data/holo3_distill_train.jsonl
```

## 4. Train

The existing `training/train_holo3_distill.py` recipe consumes the
chat-format JSONL produced by step 3. Run on a single A100:

```
python training/train_holo3_distill.py \
    --train-jsonl training/data/holo3_distill_train.jsonl \
    --output-dir training/data/holo3_distill_v2 \
    --base-model /models/holo3 \
    --lora-r 32 --epochs 1
```

(See `training/modal_train_holo3.py` for the Modal-managed variant.)

## 5. Shadow-deploy

Pending — the pattern is described in #155 (5% traffic split + escalation
rate compare) but the deployment harness is a follow-up PR.

## Putting it together

End-to-end, on a single trainer box:

```bash
# 1+2. Pull this morning's traces and label them.
rsync -av prod:/data/traces /data/traces
mantis trace label /data/traces --output /data/labelled

# 3. Convert + append.
python training/convert_labelled_traces.py \
    --traces /data/labelled \
    --screenshots-root /data/traces \
    --output training/data/labelled_distill.jsonl
cat training/data/labelled_distill.jsonl \
    >> training/data/holo3_distill_train.jsonl

# 4. Train.
python training/train_holo3_distill.py \
    --train-jsonl training/data/holo3_distill_train.jsonl \
    --output-dir training/data/holo3_v$(date +%Y%m%d)
```

## Schema reference

- Trace export schema: `src/mantis_agent/gym/trace_exporter.py`
  (`SCHEMA_VERSION` is bumped on incompatible changes).
- Label fields: `src/mantis_agent/gym/trace_labeller.py`
  (`label` ∈ `positive | negative | neutral`, `label_reason` is the
  ladder rule that fired).
- SFT chat format: `training/convert_claude_trajectories.py`
  (`HOLO3_SYSTEM`, `claude_action_to_holo3`).

Anything that imports those constants stays in lockstep with the
training-side expectations.
