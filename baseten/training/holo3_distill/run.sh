#!/usr/bin/env bash
set -euxo pipefail

export PIP_CACHE_DIR="${BT_RW_CACHE_DIR:-/tmp}/pip-cache"
export HF_HOME="${BT_PROJECT_CACHE_DIR:-/tmp}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

REPO_URL="${REPO_URL:-https://github.com/mercurialsolo/cua-agent.git}"
GIT_REF="${GIT_REF:-baseten-support}"
WORKDIR="${BT_SCRATCH_DIR:-/tmp}/cua-agent"
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-/tmp/training_checkpoints}/holo3-cua-distilled"

rm -rf "${WORKDIR}"
git clone --depth 1 --branch "${GIT_REF}" "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir \
  "transformers>=5.2" "torch>=2.1" "datasets" \
  "trl>=0.14" "peft>=0.15" "bitsandbytes>=0.45" \
  "accelerate>=1.5" "huggingface-hub" "pillow" \
  "unsloth[colab-new]"

args=(
  training/train_holo3_distill.py
  --data training/data/holo3_distill_train.jsonl
  --output "${OUTPUT_DIR}"
  --epochs "${EPOCHS:-1}"
  --batch-size "${BATCH_SIZE:-1}"
  --lr "${LR:-1e-4}"
  --lora-rank "${LORA_RANK:-16}"
  --max-samples "${MAX_SAMPLES:-73}"
)

if [[ "${EXPORT_GGUF:-true}" == "true" ]]; then
  args+=(--export-gguf)
fi

python "${args[@]}"
