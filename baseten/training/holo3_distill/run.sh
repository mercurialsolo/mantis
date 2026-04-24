#!/usr/bin/env bash
set -euxo pipefail

export PIP_CACHE_DIR="${BT_RW_CACHE_DIR:-/tmp}/pip-cache"
export HF_HOME="${BT_PROJECT_CACHE_DIR:-/tmp}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

WORKDIR="$(pwd)"
OUTPUT_DIR="${BT_CHECKPOINT_DIR:-/tmp/training_checkpoints}/holo3-cua-distilled"

cd "${WORKDIR}"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
  "transformers>=5.2" "datasets" \
  "trl>=0.14" "peft>=0.15" "bitsandbytes>=0.45" \
  "accelerate>=1.5" "huggingface-hub" "pillow" \
  "xformers==0.0.30" "unsloth[colab-new]==2025.7.2"

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
