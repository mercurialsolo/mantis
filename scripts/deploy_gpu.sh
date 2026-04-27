#!/bin/bash
# Deploy and run OSWorld evaluation on a GPU server
#
# Requirements:
#   - NVIDIA GPU with >=16GB VRAM (RTX 4090, A100, etc.)
#   - Docker with nvidia-container-toolkit (for OSWorld VM)
#   - KVM support (Linux host)
#   - Python 3.11+
#
# Usage:
#   # Run full benchmark with Gemma4 E4B:
#   ./deploy_gpu.sh --model google/gemma-4-E4B-it
#
#   # Run with 26B MoE (best accuracy, needs 48GB+ VRAM):
#   ./deploy_gpu.sh --model google/gemma-4-26B-A4B-it
#
#   # Run single domain:
#   ./deploy_gpu.sh --model google/gemma-4-E4B-it --domain chrome

set -euo pipefail

MODEL="${1:---model}"
shift || true

echo "═══════════════════════════════════════════════════"
echo "  OSWorld Evaluation — Gemma4 Streaming CUA Agent"
echo "═══════════════════════════════════════════════════"

# 1. Setup environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install -q -e .
pip install -q -r OSWorld/requirements.txt
pip install -q "transformers>=4.52" accelerate

# 3. Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 4. Verify KVM
if [ -e /dev/kvm ]; then
    echo "KVM: available"
else
    echo "WARNING: KVM not available — Docker VM will be slow"
fi

# 5. Run evaluation
echo ""
echo "Starting evaluation..."
python run_osworld.py \
    --provider_name docker \
    --observation_type screenshot \
    --action_space pyautogui \
    --max_steps 15 \
    --enable_thinking \
    "$@"

echo ""
echo "Results saved to ./results/"
echo "Run 'python OSWorld/show_result.py' to see summary."
