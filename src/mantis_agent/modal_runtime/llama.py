"""llama-server lifecycle inside a Modal container.

``download_model`` pulls the configured Gemma 4 GGUF into the shared
volume and caches across runs. ``start_llama_server`` spawns
llama-server on CUDA, waits for /v1/models to respond, and returns the
``Popen`` handle (caller is responsible for terminating).
"""

from __future__ import annotations

import os
import subprocess
import time

from .image import GEMMA4_MODEL, GGUF_CONFIGS, vol


def download_model(vol_path: str) -> str:
    """Download Gemma4 GGUF if not cached."""
    cfg = GGUF_CONFIGS[GEMMA4_MODEL]
    model_dir = os.path.join(vol_path, "models")
    model_path = os.path.join(model_dir, cfg["model_file"])
    if os.path.exists(model_path):
        print(f"Model cached at {model_path}")
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"Downloading Gemma4 {GEMMA4_MODEL} GGUF from {cfg['repo']}...")
    for f in [cfg["model_file"], cfg["mmproj_file"]]:
        hf_hub_download(cfg["repo"], f, local_dir=model_dir)
    vol.commit()
    print("Model downloaded.")
    return model_path


def start_llama_server(model_path: str, port: int = 8080) -> subprocess.Popen:
    """Start llama-server on CUDA GPU."""

    # Find the model and mmproj files
    model_dir = os.path.dirname(model_path)
    print(f"Model dir contents: {os.listdir(model_dir)}")
    print(f"Model path: {model_path} (exists: {os.path.exists(model_path)})")

    # Find the correct mmproj file for this model
    cfg = GGUF_CONFIGS[GEMMA4_MODEL]
    mmproj_path = os.path.join(model_dir, cfg["mmproj_file"])
    mmproj_files = [mmproj_path] if os.path.exists(mmproj_path) else []
    print(f"mmproj: {mmproj_path} (exists: {os.path.exists(mmproj_path)})")

    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0", "--port", str(port),
        "-ngl", "99",
        "-c", "32768",      # Gemma4 needs larger context for vision (native 256K)
        "-ub", "2048",       # Must be >= image token batch (~972 for 1920x1080)
        "--jinja",           # Required for proper Gemma4 chat template
        "--reasoning-budget", "0",  # Keep 0 for OSWorld CLI tasks — 4096 caused 91.7→83.3% regression
        "--flash-attn", "on", # EXP-11: ~30-50% faster attention, identical outputs
    ]

    # Add mmproj if found (needed for multimodal)
    if mmproj_files:
        cmd.extend(["--mmproj", mmproj_files[0]])

    print(f"Starting: {' '.join(cmd)}")
    log_path = "/tmp/llama.log"

    def _tail_log() -> str:
        try:
            with open(log_path) as f:
                return f.read()[-3000:]
        except OSError:
            return "(llama log unavailable)"

    log_fh = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_fh.close()
        raise

    # Check if process crashed immediately
    time.sleep(3)
    if proc.poll() is not None:
        print(f"llama-server crashed with code {proc.returncode}")
        print(_tail_log())
        log_fh.close()
        raise RuntimeError(f"llama-server crashed with code {proc.returncode}")

    import requests
    for i in range(90):  # 3 min timeout
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"llama-server ready on :{port} ({i*2}s)")
                return proc
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            print(f"llama-server died with code {proc.returncode}")
            print(_tail_log())
            log_fh.close()
            raise RuntimeError("llama-server died during startup")
        time.sleep(2)

    print("TIMEOUT - llama-server log:")
    print(_tail_log())
    log_fh.close()
    raise RuntimeError("llama-server failed to start within timeout")
