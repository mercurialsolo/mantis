"""Mantis FastAPI server on Modal — mirrors the Baseten Truss shape.

Serves ``mantis_agent.baseten_server:app`` as a Modal ASGI web endpoint
backed by a llama.cpp Holo3 server in the same container. The container
exposes the same FastAPI surface as the Baseten deployment:

  - POST /predict / /v1/predict — orchestrated plan execution
  - POST /v1/chat/completions — OpenAI-compatible Holo3 inference proxy
  - GET  /v1/models / /health  — readiness + auth gate

A host integration swaps from Baseten to Modal by changing one setting
on its side:

  MANTIS_ENDPOINT=https://<workspace>--mantis-server-api.modal.run

Modal endpoints are publicly addressable but auth is still enforced by
``baseten_server.py`` via ``X-Mantis-Token`` — the platform-level
``Authorization: Api-Key …`` header that Baseten requires is NOT needed
here, so any host-side gateway-auth setting can be left empty.

## Deploy

    modal deploy deploy/modal/modal_mantis_server.py

Then, to avoid the first user eating the 37 GB Holo3 download, pre-warm
the volume once:

    uv run modal run deploy/modal/modal_mantis_server.py::prewarm_weights

After that, subsequent cold-starts are ~90 s (just llama-server boot +
mmap), not ~10 min. Weights persist on the ``mantis-server-data`` volume
and are reused by every container.

## Secrets

Reads from the local ``.env`` file at deploy time. The ``.env`` MUST include:

  - ANTHROPIC_API_KEY  (used by ClaudeGrounding / ClaudeExtractor)
  - PROXY_URL / PROXY_USER / PROXY_PASS (optional IPRoyal residential proxy)

Plus a managed Modal Secret named ``mantis-tenant-keys`` containing a
single env var ``MANTIS_TENANT_KEYS_JSON`` — the multi-tenant keys file
(see ``docs/operations/tenant-keys.md`` for shape). Each tenant entry
sets its own scopes, cost cap, time cap, concurrent-run cap, rate limit,
allowed domain list, and Anthropic key — so a leaked token has bounded
blast radius (cost + rate + domains all clamped per-tenant).

If ``mantis-tenant-keys`` is missing, the server falls back to legacy
single-tenant mode using ``MANTIS_API_TOKEN`` from ``.env`` — no caps,
no allowlist, full blast radius. Avoid for any deployment you'd hand a
token out for.

To create / update the secret:

    # write a keys file locally
    cat > /tmp/tenant_keys.json <<'EOF'
    {"tenant_keys": {"<token>": {"tenant_id": "...", ...}}}
    EOF

    # push it as a Modal Secret
    modal secret create mantis-tenant-keys \\
        MANTIS_TENANT_KEYS_JSON="$(cat /tmp/tenant_keys.json)"

    # rotate / add tenants later
    modal secret create --force mantis-tenant-keys \\
        MANTIS_TENANT_KEYS_JSON="$(cat /tmp/tenant_keys.json)"

The container writes the JSON to ``/tmp/tenant_keys.json`` at boot and
points ``MANTIS_TENANT_KEYS_PATH`` there; the running FastAPI app then
hot-reloads tenant config every 5 s. Secret updates require a container
restart (Modal scales replicas at request time, so this is automatic
after a few minutes of idle).

## Configure for a host integration

After deploy, the app URL is printed by Modal. Set on the host side:

```bash
MANTIS_ENDPOINT=<the-modal-app-url>
MANTIS_API_TOKEN=<one-of-the-tokens-from-mantis-tenant-keys>
# Modal needs no gateway auth — leave any gateway-auth setting unset.
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


def _bootstrap_tenant_keys() -> None:
    """Materialize the tenant keys JSON to a file the FastAPI app can read.

    The keys live in a Modal Secret as the env var MANTIS_TENANT_KEYS_JSON;
    we write it to /tmp/tenant_keys.json and point the server at that path.
    Without this, the server falls back to single-tenant MANTIS_API_TOKEN
    mode — fine for local dev, unsafe for any token you hand out.
    """
    raw = os.environ.get("MANTIS_TENANT_KEYS_JSON", "").strip()
    if not raw:
        return
    path = "/tmp/tenant_keys.json"
    try:
        with open(path, "w") as fh:
            fh.write(raw)
        os.chmod(path, 0o600)
    except OSError as exc:
        print(f"[bootstrap] failed to write tenant keys: {exc}")
        return
    os.environ["MANTIS_TENANT_KEYS_PATH"] = path
    print(f"[bootstrap] tenant keys written to {path} (multi-tenant mode active)")


# Modal Secret containing MANTIS_TENANT_KEYS_JSON. Optional: if absent,
# the container falls back to single-tenant MANTIS_API_TOKEN.
def _tenant_keys_secret() -> list:
    try:
        return [modal.Secret.from_name("mantis-tenant-keys")]
    except Exception:
        return []


@app.function(
    gpu="H100",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv(), *_tenant_keys_secret()],
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
    # baseten_server._find_gguf defaults to /models/holo3 (Baseten Truss path).
    # Tell it where Modal actually puts the GGUF.
    os.environ.setdefault("MANTIS_HOLO3_MODEL_DIR", MODEL_DIR)
    os.environ.setdefault("MANTIS_HOLO3_GGUF", os.path.join(MODEL_DIR, HOLO3_GGUF))
    os.environ.setdefault("MANTIS_HOLO3_MMPROJ", os.path.join(MODEL_DIR, HOLO3_MMPROJ))

    # Materialize multi-tenant keys file if the secret is mounted.
    _bootstrap_tenant_keys()

    # Start llama-server alongside the FastAPI app. baseten_server's
    # /v1/chat/completions proxies to http://127.0.0.1:$MANTIS_LLAMA_PORT.
    _start_llama_server()

    # Late import — only after env vars are set so module-level config picks them up.
    from mantis_agent.baseten_server import app as fastapi_app
    return fastapi_app


@app.function(
    volumes={"/data": vol},
    timeout=3600,
    memory=8192,
    cpu=2,
)
def prewarm_weights() -> str:
    """One-shot: download Holo3 GGUF + mmproj onto the persistent volume.

    Run once after first deploy so the first real /v1/predict request
    isn't blocked by a 10-minute model download.

        uv run modal run deploy/modal/modal_mantis_server.py::prewarm_weights
    """
    model_dir = _ensure_holo3_weights()
    gguf = os.path.join(model_dir, HOLO3_GGUF)
    mmproj = os.path.join(model_dir, HOLO3_MMPROJ)
    sizes = {
        "gguf_bytes": os.path.getsize(gguf) if os.path.exists(gguf) else 0,
        "mmproj_bytes": os.path.getsize(mmproj) if os.path.exists(mmproj) else 0,
    }
    print(f"prewarm complete: {sizes}")
    return f"{model_dir}: gguf={sizes['gguf_bytes']} mmproj={sizes['mmproj_bytes']}"
