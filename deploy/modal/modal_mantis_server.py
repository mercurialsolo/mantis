"""Mantis FastAPI server on Modal — mirrors the Baseten Truss shape.

Serves ``mantis_agent.baseten_server:app`` as a Modal ASGI web endpoint
backed by a llama.cpp Holo3 server in the same container. The container
exposes the same FastAPI surface as the Baseten deployment:

  - POST /predict / /v1/predict — orchestrated plan execution
  - POST /v1/chat/completions — OpenAI-compatible Holo3 inference proxy
  - GET  /v1/models / /health  — readiness + auth gate

vision_claude swaps from Baseten to Modal by changing one setting:

  VISION_CLAUDE_MANTIS_ENDPOINT=https://<workspace>--mantis-server-api.modal.run

Modal endpoints are publicly addressable but auth is still enforced by
``baseten_server.py`` via ``X-Mantis-Token`` — the platform-level
``Authorization: Api-Key …`` header that Baseten requires is NOT needed
here, so leave ``VISION_CLAUDE_MANTIS_GATEWAY_AUTHORIZATION`` empty for
this host.

## Deploy

    modal deploy deploy/modal/modal_mantis_server.py

The first cold-start downloads the Holo3 GGUF (~37 GB) onto the persistent
Modal volume. Subsequent containers reuse it — typical warm cold-start is
~90s, cold-from-zero is ~10 minutes (model download).

## Secrets

Reads from the local ``.env`` file at deploy time (same shape as
``deploy/modal/modal_cua_server.py``). The ``.env`` MUST include:

  - MANTIS_API_TOKEN   (X-Mantis-Token enforced by baseten_server.py)
  - ANTHROPIC_API_KEY  (used by ClaudeGrounding / ClaudeExtractor)
  - PROXY_URL / PROXY_USER / PROXY_PASS (optional IPRoyal residential proxy)

If you need a managed Modal Secret instead, swap
``modal.Secret.from_dotenv()`` below for ``modal.Secret.from_name(...)``.

## Configure for vision_claude / staffai

After deploy, the app URL is printed by Modal. Set on staffai side:

```bash
VISION_CLAUDE_MANTIS_ENDPOINT=<the-modal-app-url>
VISION_CLAUDE_MANTIS_API_TOKEN=<MANTIS_API_TOKEN-from-the-secret>
# Modal needs no gateway auth — leave VISION_CLAUDE_MANTIS_GATEWAY_AUTHORIZATION unset.
```
"""

from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "mantis-server"

# Persistent volume for Holo3 GGUF + run state. Same layout as Baseten.
vol = modal.Volume.from_name("mantis-server-data", create_if_missing=True)

HOLO3_REPO = "mradermacher/Holo3-35B-A3B-GGUF"
HOLO3_GGUF = "Holo3-35B-A3B.Q8_0.gguf"
HOLO3_MMPROJ = "Holo3-35B-A3B.mmproj-f16.gguf"
MODEL_DIR = "/data/models/holo3"
LLAMA_PORT = 18080
SERVER_PORT = 8000

# Image: matches deploy/baseten/holo3/config.yaml — same base, same llama.cpp SHA,
# same apt deps. Keep these in lockstep so behaviour matches across hosts.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel", add_python="3.11",
    )
    .apt_install(
        "git", "build-essential", "cmake", "curl", "wget", "gnupg",
        "xvfb", "xdotool", "scrot", "ffmpeg",
        "ca-certificates", "fonts-liberation", "fonts-noto-color-emoji",
        "libnss3", "libatk-bridge2.0-0", "libdrm2", "libxkbcommon0",
        "libgbm1", "libpango-1.0-0", "libcairo2", "libasound2", "libxshmfence1",
    )
    .run_commands(
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y google-chrome-stable",
    )
    .pip_install(
        "openai", "requests", "pillow", "mss", "huggingface-hub",
        "fastapi>=0.100", "uvicorn>=0.20", "pydantic>=2",
        "prometheus-client>=0.20",
    )
    .run_commands(
        # Pin to the same llama.cpp SHA the Baseten Truss uses (b8948).
        "git clone --depth 1 --branch b8948 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && ldconfig",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF "
        "-DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES=\"80;90\" "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON "
        "&& cmake --build build --target llama-server --config Release -j$(nproc)",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir("plans", "/packages/plans")
)

app = modal.App(APP_NAME, image=image)


def _ensure_holo3_weights() -> str:
    """Download Holo3 GGUF + mmproj on first cold-start. Returns model dir."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    gguf_path = os.path.join(MODEL_DIR, HOLO3_GGUF)
    if os.path.exists(gguf_path):
        return MODEL_DIR
    from huggingface_hub import hf_hub_download
    print("Downloading Holo3 weights...")
    hf_hub_download(repo_id=HOLO3_REPO, filename=HOLO3_GGUF, local_dir=MODEL_DIR)
    hf_hub_download(repo_id=HOLO3_REPO, filename=HOLO3_MMPROJ, local_dir=MODEL_DIR)
    vol.commit()
    print(f"Holo3 cached at {MODEL_DIR}")
    return MODEL_DIR


def _start_llama_server() -> subprocess.Popen:
    """Start llama-server in the background — same flags as Baseten Truss."""
    model_dir = _ensure_holo3_weights()
    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "--model", os.path.join(model_dir, HOLO3_GGUF),
        "--mmproj", os.path.join(model_dir, HOLO3_MMPROJ),
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "--n-gpu-layers", "99",
        "-c", "8192",
        "-ub", "2048",
        "--jinja",
        "--flash-attn", "on",
    ]
    print(f"Starting llama-server: {' '.join(cmd[:6])} ...")
    return subprocess.Popen(
        cmd,
        stdout=open("/tmp/llama.log", "w"),
        stderr=subprocess.STDOUT,
    )


@app.function(
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=86400,
    memory=65536,
    cpu=8,
    min_containers=0,           # cold-scale to zero between calls
    scaledown_window=600,       # keep warm for 10 min after last request
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    """Serve mantis_agent.baseten_server:app — same FastAPI surface as Baseten."""
    # Set env vars baseten_server.py reads at import time.
    os.environ.setdefault("MANTIS_MODEL", "holo3")
    os.environ.setdefault("MANTIS_LLAMA_PORT", str(LLAMA_PORT))
    os.environ.setdefault("MANTIS_DATA_DIR", "/data/mantis-runs")
    os.environ.setdefault("MANTIS_REPO_ROOT", "/packages")

    # Start llama-server alongside the FastAPI app. baseten_server's
    # /v1/chat/completions proxies to http://127.0.0.1:$MANTIS_LLAMA_PORT.
    _start_llama_server()

    # Late import — only after env vars are set so module-level config picks them up.
    from mantis_agent.baseten_server import app as fastapi_app
    return fastapi_app
