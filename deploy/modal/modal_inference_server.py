"""Modal deployment: Gemma4 E4B inference server via llama.cpp.

Deploys the Gemma4 E4B GGUF model on a Modal A100-80GB GPU and exposes
it as an OpenAI-compatible HTTP endpoint.  Any client that speaks the
``/v1/chat/completions`` protocol (HUD's OpenAIChatAgent, the openai
Python SDK, curl, etc.) can point at the resulting URL.

Architecture:
    ┌──────── Modal A100 Container ────────┐
    │                                       │
    │  llama-server (Gemma4 E4B, GGUF)     │
    │     GPU-offloaded, OpenAI-compat API  │
    │     listening on 0.0.0.0:8080         │
    │                                       │
    └───────────── port 8080 ──────────────┘
            ↑ Modal web_server proxy
            ↓
    https://<app-name>--serve.modal.run/v1/chat/completions

Usage:
    # Deploy (keeps running until stopped):
    modal deploy deploy/modal/modal_inference_server.py

    # Or run ephemerally for testing:
    modal serve modal_inference_server.py

    # Then point any OpenAI-compat client at it:
    curl https://<your-app>--serve.modal.run/v1/models

    # With HUD:
    python run_hud_osworld.py --model-url https://<your-app>--serve.modal.run/v1
"""

import os
import subprocess

import modal

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("gemma4-inference-server")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake")
    .run_commands(
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80 "  # A100 only — cuts build time 5x
        "&& cmake --build build --config Release -j$(nproc)",
    )
    .pip_install("huggingface-hub[cli]", "requests")
)

# ---------------------------------------------------------------------------
# Model download (cached on volume across restarts)
# ---------------------------------------------------------------------------

MODEL_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
MODEL_FILE = "gemma-4-e4b-it-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-gemma-4-e4b-it-f16.gguf"
MODEL_DIR = "/data/models"


def download_model(vol_path: str) -> str:
    """Download Gemma4 GGUF model files if not already cached."""
    model_path = os.path.join(vol_path, MODEL_FILE)
    mmproj_path = os.path.join(vol_path, MMPROJ_FILE)

    if os.path.exists(model_path) and os.path.exists(mmproj_path):
        print(f"Model cached at {model_path}")
        return model_path

    print("Downloading Gemma4 E4B GGUF model...")
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=vol_path)
    hf_hub_download(repo_id=MODEL_REPO, filename=MMPROJ_FILE, local_dir=vol_path)
    print("Download complete.")
    return model_path


# ---------------------------------------------------------------------------
# Web endpoint — Modal proxies port 8080 to a public HTTPS URL
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,       # 24 h keep-alive
    memory=32768,        # 32 GB RAM
    cpu=4,
    scaledown_window=600,  # 10 min idle before scale-down
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=8080, startup_timeout=300)
def serve():
    """Start llama-server on A100 GPU.

    Modal routes incoming HTTPS traffic to port 8080 inside the container.
    The llama-server exposes an OpenAI-compatible API at /v1/*.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = download_model(MODEL_DIR)
    vol.commit()

    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", "8080",
        "-ngl", "99",          # offload all layers to GPU
        "-c", "4096",          # context window
        "--no-warmup",
    ]
    print(f"Starting llama-server: {' '.join(cmd)}")
    subprocess.Popen(cmd)
