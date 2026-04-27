"""Modal brain server — GPU inference for local browser agent.

Deploys EvoCUA-32B on Modal A100 and exposes an OpenAI-compatible
HTTP endpoint. Your local run_local.py calls this for brain inference
while running Playwright locally (residential IP).

Usage:
    # Deploy (stays running, gives you a URL)
    modal deploy deploy/modal/modal_brain_server.py

    # Then run locally
    python run_local.py \
      --task-file tasks/boattrader/full_production.json \
      --brain-url https://<your-modal-url>/v1 \
      --brain-type evocua --headed
"""

import os
import subprocess
import sys

import modal

app = modal.App("mantis-brain")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget")
    .pip_install(
        "vllm>=0.12.0",
        "openai", "requests", "huggingface-hub", "transformers", "torch",
    )
)


@app.function(
    gpu="A100-80GB:2",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
@modal.web_server(port=8000)
def brain():
    """Serve EvoCUA-32B via vLLM as an OpenAI-compatible endpoint."""
    model_key = "evocua-32b"
    repo = "meituan/EvoCUA-32B-20260105"
    tp = 2

    # Download/cache model
    model_dir = f"/data/models/{model_key.replace('-', '_')}"
    marker = os.path.join(model_dir, ".download_complete")
    if not os.path.exists(marker):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading {repo}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=model_dir, ignore_patterns=["*.md", "*.txt"])
        open(marker, "w").write("done")
        vol.commit()
    else:
        print(f"Model cached at {model_dir}")

    # Start vLLM on port 8000 (Modal exposes this as HTTPS)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--served-model-name", "model",
        "--host", "0.0.0.0", "--port", "8000",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768",
    ]
    print(f"Starting vLLM EvoCUA-32B (TP={tp})...")
    subprocess.Popen(cmd)
