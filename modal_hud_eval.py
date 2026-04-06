"""Run OSWorld evaluation on Modal using HUD Docker images.

HUD environments are Docker images with MCP servers. We run them on Modal
alongside our Gemma4 llama-server, connecting the two via localhost.

Architecture:
    Modal Container (A100 GPU)
    ├── llama-server (Gemma4 E4B on GPU, port 8080)
    └── HUD environment (OSWorld VM, MCP on port 9090)
         ├── hud scenario setup → returns prompt + screenshots
         ├── Agent calls tools via MCP → actions execute in VM
         └── hud scenario grade → returns reward

Usage:
    # Run single task:
    modal run modal_hud_eval.py

    # Run full benchmark:
    modal run modal_hud_eval.py --tasks 367
"""

import json
import os
import subprocess
import time

import modal

app = modal.App("gemma4-osworld-hud")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# Image: CUDA + llama.cpp + HUD CLI + Python deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake", "curl")
    .run_commands(
        # Build llama.cpp with CUDA
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)",
    )
    .pip_install(
        "huggingface-hub[cli]", "requests", "pillow", "httpx",
        "hud-python>=0.5.34", "openai>=1.0",
    )
)


def download_model(vol_path: str) -> str:
    """Download Gemma4 GGUF if not cached."""
    model_path = os.path.join(vol_path, "models", "gemma-4-e4b-it-Q4_K_M.gguf")
    if os.path.exists(model_path):
        return model_path

    os.makedirs(os.path.join(vol_path, "models"), exist_ok=True)
    from huggingface_hub import hf_hub_download

    print("Downloading Gemma4 E4B GGUF...")
    for f in ["gemma-4-e4b-it-Q4_K_M.gguf", "mmproj-gemma-4-e4b-it-f16.gguf"]:
        hf_hub_download("ggml-org/gemma-4-E4B-it-GGUF", f,
                        local_dir=os.path.join(vol_path, "models"))
    vol.commit()
    return model_path


def start_llama_server(model_path: str, port: int = 8080) -> subprocess.Popen:
    """Start llama-server on CUDA GPU."""
    proc = subprocess.Popen(
        ["/opt/llama.cpp/build/bin/llama-server",
         "-m", model_path, "--host", "0.0.0.0", "--port", str(port),
         "-ngl", "99", "-c", "4096", "--no-warmup"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    import requests
    for _ in range(60):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"llama-server ready on :{port}")
                return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server failed to start")


def gemma4_agent_step(
    base_url: str,
    screenshot_b64: str,
    prompt: str,
    action_history: list[str],
) -> dict:
    """Single agent step: screenshot → Gemma4 → tool call."""
    import requests

    tools = [
        {"type": "function", "function": {"name": "click", "description": "Click at coordinates",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}}},
        {"type": "function", "function": {"name": "type", "description": "Type text",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
        {"type": "function", "function": {"name": "key", "description": "Press key combo",
            "parameters": {"type": "object", "properties": {"keys": {"type": "string"}}, "required": ["keys"]}}},
        {"type": "function", "function": {"name": "scroll", "description": "Scroll",
            "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer"}}, "required": ["direction"]}}},
        {"type": "function", "function": {"name": "screenshot", "description": "Take screenshot",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "done", "description": "Task complete",
            "parameters": {"type": "object", "properties": {"success": {"type": "boolean"}}, "required": ["success"]}}},
    ]

    content = [
        {"type": "text", "text": f"You are a computer use agent. {prompt}\nScreen: 1920x1080."},
    ]
    if screenshot_b64:
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}})

    if action_history:
        content.append({"type": "text", "text": "Recent actions:\n" + "\n".join(f"  {a}" for a in action_history[-5:])})

    resp = requests.post(f"{base_url}/chat/completions", json={
        "model": "gemma-4",
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "max_tokens": 2048,
        "temperature": 0,
    }, timeout=120)

    data = resp.json()
    msg = data["choices"][0]["message"]
    tc = msg.get("tool_calls", [])
    if tc:
        func = tc[0]["function"]
        return {"name": func["name"], "args": json.loads(func.get("arguments", "{}"))}
    return {"name": "screenshot", "args": {}}


@app.function(
    gpu="A100",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=32768,
    cpu=8,
    secrets=[modal.Secret.from_name("hud-api-key", required=False)],
)
def run_eval(max_tasks: int = 1, max_steps: int = 15):
    """Run OSWorld tasks using Gemma4 on A100 + HUD eval framework."""
    import requests

    # 1. Download model
    model_path = download_model("/data")

    # 2. Start llama-server
    print("Starting Gemma4 on A100...")
    llama_proc = start_llama_server(model_path)

    # 3. Run HUD eval via CLI
    # The hud eval command handles everything: VM provisioning, screenshots,
    # action execution, and evaluation. We just point it at our llama-server.
    hud_api_key = os.environ.get("HUD_API_KEY", "")

    print(f"\nRunning OSWorld-Verified ({max_tasks} tasks, {max_steps} steps)")
    print(f"Model: Gemma4 E4B via llama-server on A100")

    # Use hud eval with openai_compatible pointing at localhost
    cmd = [
        "hud", "eval", "OSWorld-Verified", "openai_compatible",
        "--config", "base_url=http://localhost:8080/v1",
        "--model", "gemma-4",
        "--max-steps", str(max_steps),
        "-y",
    ]
    if max_tasks > 1:
        cmd.append("--full")
        cmd.extend(["--max-concurrent", "1"])

    env = os.environ.copy()
    env["HUD_API_KEY"] = hud_api_key

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

    print(f"\n{'='*60}")
    print(f"stdout:\n{result.stdout[-2000:]}")
    if result.returncode != 0:
        print(f"stderr:\n{result.stderr[-1000:]}")
    print(f"{'='*60}")

    # Save results
    os.makedirs("/data/results", exist_ok=True)
    with open("/data/results/hud_eval_output.json", "w") as f:
        json.dump({
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }, f, indent=2)
    vol.commit()

    llama_proc.terminate()
    return {"returncode": result.returncode, "output": result.stdout[-500:]}


@app.local_entrypoint()
def main(tasks: int = 1, steps: int = 15):
    print(f"Launching OSWorld eval on Modal A100 (tasks={tasks})")
    result = run_eval.remote(max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {result}")
