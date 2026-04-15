"""Mantis CUA Server — fully remote Gemma4 + EvoCUA on Modal.

Single `modal deploy` gives you:
  - Gemma4 planner (T4 GPU, llama.cpp, persistent web server)
  - EvoCUA executors (A100 GPUs, vLLM, per-run)

Usage:
    # Deploy the planner (stays running, warm for 10 min):
    uv run modal deploy modal_cua_server.py

    # Run from a free-text plan (Gemma4 preprocesses → EvoCUA executes):
    uv run modal run modal_cua_server.py \
        --plan-file plans/boattrader/full_spec.txt \
        --model evocua-8b \
        --inputs "pop_password=SelfService38#,zip_code=33101,search_radius=35"

    # Run from a pre-built task suite (bypasses planner):
    uv run modal run modal_cua_server.py \
        --task-file tasks/boattrader/dynamic_production.json \
        --model evocua-8b
"""

import json
import os
import subprocess
import sys
import time

import modal

# ── App + shared resources ──────────────────────────────────────────

app = modal.App("mantis-cua-server")
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# ── Model configs ───────────────────────────────────────────────────

CUA_MODELS = {
    "evocua-8b": {
        "repo": "meituan/EvoCUA-8B-20260105",
        "name": "EvoCUA-8B",
        "tp": 1,
    },
    "evocua-32b": {
        "repo": "meituan/EvoCUA-32B-20260105",
        "name": "EvoCUA-32B",
        "tp": 2,
    },
    "opencua-32b": {
        "repo": "xlangai/OpenCUA-32B",
        "name": "OpenCUA-32B",
        "tp": 4,
    },
    "opencua-72b": {
        "repo": "xlangai/OpenCUA-72B",
        "name": "OpenCUA-72B",
        "tp": 8,
    },
    "gemma4-cua": {
        "repo": "local:training/gemma4-cua-gguf_gguf",
        "name": "Gemma4-31B-CUA",
        "tp": 1,
    },
}

# Fine-tuned Gemma4-31B-CUA (trained on AgentNet, native tool calling)
# Already quantized to GGUF on the Modal volume
GEMMA4_CUA_DIR = "/data/training/gemma4-cua-gguf_gguf"
GEMMA4_CUA_FILE = "gemma-4-31b-it.Q4_K_M.gguf"
GEMMA4_CUA_MMPROJ = "gemma-4-31b-it.BF16-mmproj.gguf"

# Fallback: base Gemma4-E4B (downloads from HuggingFace)
GEMMA4_E4B_REPO = "ggml-org/gemma-4-E4B-it-GGUF"
GEMMA4_E4B_FILE = "gemma-4-e4b-it-Q4_K_M.gguf"
GEMMA4_E4B_MMPROJ = "mmproj-gemma-4-e4b-it-f16.gguf"

# ── Images ──────────────────────────────────────────────────────────

# Gemma4 planner: llama.cpp + CUDA (lightweight)
planner_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "cmake")
    .run_commands(
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        # Symlink CUDA driver stubs so linker finds libcuda.so during image build
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 "
        "&& ldconfig",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON "
        "-DCMAKE_CUDA_ARCHITECTURES=80 "  # A100 — same arch as inference_server
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DLLAMA_BUILD_SERVER=ON "
        "&& cmake --build build --target llama-server --config Release -j$(nproc)",
    )
    .pip_install("huggingface-hub[cli]", "requests")
)

# EvoCUA executor: vLLM + real Chrome + Xvfb + xdotool (zero automation fingerprints)
executor_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget", "gnupg",
                 "xvfb", "xdotool", "scrot")
    .run_commands(
        # Install real Google Chrome (not Chromium)
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    )
    .pip_install(
        "vllm>=0.12.0",
        "openai", "requests", "pillow", "mss",
        "huggingface-hub", "transformers", "torch",
    )
    .add_local_python_source("mantis_agent")
)


# ═══════════════════════════════════════════════════════════════════
# A) Gemma4 Planner — persistent web server
# ═══════════════════════════════════════════════════════════════════

def _resolve_gemma4_model() -> str:
    """Resolve the best Gemma4 model available on the volume.

    Prefers fine-tuned 31B-CUA GGUF (trained on AgentNet) over base E4B.
    """
    # Prefer fine-tuned 31B-CUA (already on volume from training)
    cua_model = os.path.join(GEMMA4_CUA_DIR, GEMMA4_CUA_FILE)
    if os.path.exists(cua_model):
        print(f"Using fine-tuned Gemma4-31B-CUA: {cua_model}")
        return cua_model

    # Fallback: download base E4B from HuggingFace
    model_dir = "/data/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, GEMMA4_E4B_FILE)
    if os.path.exists(model_path):
        print(f"Using base Gemma4-E4B: {model_path}")
        return model_path

    print("Downloading Gemma4 E4B GGUF (fallback)...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=GEMMA4_E4B_REPO, filename=GEMMA4_E4B_FILE, local_dir=model_dir)
    hf_hub_download(repo_id=GEMMA4_E4B_REPO, filename=GEMMA4_E4B_MMPROJ, local_dir=model_dir)
    print("Download complete.")
    return model_path


@app.function(
    gpu="A100-80GB",
    image=planner_image,
    volumes={"/data": vol},
    timeout=86400,
    memory=16384,
    cpu=4,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8080, startup_timeout=300)
def gemma4_planner():
    """Gemma4 E4B planner via llama.cpp — text-only plan preprocessing.

    Exposes OpenAI-compatible API at /v1/chat/completions.
    Used by optimize_plan() to convert free-text plans into structured task suites.
    """
    model_path = _resolve_gemma4_model()

    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", "8080",
        "-ngl", "99",
        "-c", "8192",
        "--no-warmup",
    ]
    print(f"Starting Gemma4 planner: {' '.join(cmd[-6:])}")
    subprocess.Popen(cmd)


# ═══════════════════════════════════════════════════════════════════
# B) Real Chrome via CDP — bypasses Cloudflare
# ═══════════════════════════════════════════════════════════════════

def _start_local_proxy(upstream_proxy: dict, local_port: int = 3128) -> subprocess.Popen:
    """Start a local forward proxy that handles authentication with the upstream proxy.

    Chrome --proxy-server doesn't support user:pass in URLs.
    This starts a tiny Python proxy on localhost that forwards to the
    authenticated upstream (IPRoyal etc).
    """
    proxy_server = upstream_proxy.get("server", "")
    proxy_user = upstream_proxy.get("username", "")
    proxy_pass = upstream_proxy.get("password", "")

    # Write a minimal proxy forwarder script
    script = f"""
import http.server, urllib.request, socketserver, base64, ssl

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    upstream = "{proxy_server}"
    auth = base64.b64encode(b"{proxy_user}:{proxy_pass}").decode()

    def do_CONNECT(self):
        import socket
        from urllib.parse import urlparse
        p = urlparse(self.upstream)
        try:
            s = socket.create_connection((p.hostname, p.port), timeout=30)
            connect_req = f"CONNECT {{self.path}} HTTP/1.1\\r\\nHost: {{self.path}}\\r\\nProxy-Authorization: Basic {{self.auth}}\\r\\n\\r\\n"
            s.sendall(connect_req.encode())
            resp = s.recv(4096)
            if b"200" in resp:
                self.send_response(200)
                self.end_headers()
                import threading
                def forward(src, dst):
                    try:
                        while True:
                            data = src.recv(65536)
                            if not data: break
                            dst.sendall(data)
                    except: pass
                c = self.request
                t1 = threading.Thread(target=forward, args=(c, s), daemon=True)
                t2 = threading.Thread(target=forward, args=(s, c), daemon=True)
                t1.start(); t2.start(); t1.join(); t2.join()
            else:
                self.send_error(502)
        except Exception as e:
            self.send_error(502, str(e))

    def log_message(self, *a): pass

with socketserver.ThreadingTCPServer(("127.0.0.1", {local_port}), ProxyHandler) as srv:
    srv.serve_forever()
"""
    with open("/tmp/local_proxy.py", "w") as f:
        f.write(script)

    proc = subprocess.Popen(
        [sys.executable, "/tmp/local_proxy.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError("Local proxy forwarder crashed on startup")
    print(f"  Local proxy forwarder on :{local_port} → {proxy_server}")
    return proc


def _start_chrome_cdp(proxy: dict | None = None, port: int = 9222) -> subprocess.Popen:
    """Launch real Google Chrome with remote debugging for CDP connection.

    Real Chrome has authentic fingerprints that pass Cloudflare verification,
    unlike Playwright's bundled Chromium which is trivially detectable.

    For authenticated proxies, starts a local forwarder since Chrome
    --proxy-server doesn't support user:pass in proxy URLs.
    """
    import requests as req

    # Start Xvfb (virtual display for headless Chrome)
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x720x24", "-ac"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = ":99"
    time.sleep(1)

    # Build Chrome command
    cmd = [
        "google-chrome",
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=0.0.0.0",
        "--user-data-dir=/tmp/chrome-cdp-profile",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate",
        "--disable-infobars",
        "--disable-session-crashed-bubble",
        "--no-sandbox",
        "--disable-gpu",
        "--window-size=1280,720",
    ]

    # Proxy setup
    if proxy:
        if proxy.get("username"):
            # Authenticated proxy: use local forwarder
            _start_local_proxy(proxy, local_port=3128)
            cmd.append("--proxy-server=http://127.0.0.1:3128")
        else:
            # Unauthenticated proxy: direct
            cmd.append(f"--proxy-server={proxy['server']}")

    cmd.append("about:blank")

    env = os.environ.copy()
    env["DISPLAY"] = ":99"

    print(f"Starting Chrome CDP on :{port}")
    chrome_proc = subprocess.Popen(
        cmd, env=env,
        stdout=open("/tmp/chrome.log", "w"),
        stderr=subprocess.STDOUT,
    )

    time.sleep(3)
    if chrome_proc.poll() is not None:
        log = open("/tmp/chrome.log").read()[-2000:]
        print(f"Chrome crashed: {log}")
        raise RuntimeError("Chrome crashed on startup")

    # Wait for CDP port
    for i in range(30):
        try:
            r = req.get(f"http://localhost:{port}/json/version", timeout=2)
            if r.status_code == 200:
                browser_info = r.json()
                print(f"Chrome CDP ready ({i}s): {browser_info.get('Browser', '?')}")
                return chrome_proc
        except Exception:
            pass
        time.sleep(1)

    log = open("/tmp/chrome.log").read()[-2000:]
    print(f"Chrome CDP timeout: {log}")
    raise RuntimeError("Chrome CDP startup timeout")


def _build_proxy_config() -> dict | None:
    """Build proxy config from environment variables."""
    proxy_url = os.environ.get("PROXY_URL", "")
    if not proxy_url:
        return None

    proxy = {"server": proxy_url}
    proxy_user = os.environ.get("PROXY_USER", "")
    proxy_pass = os.environ.get("PROXY_PASS", "")
    if proxy_user:
        proxy["username"] = proxy_user
        proxy["password"] = proxy_pass
    return proxy


# ═══════════════════════════════════════════════════════════════════
# C) Shared executor logic
# ═══════════════════════════════════════════════════════════════════

def _start_vllm(model_dir: str, port: int, tp: int) -> subprocess.Popen:
    """Start vLLM and wait for readiness."""
    import requests as req

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--served-model-name", "model",
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768",
    ]
    print(f"Starting vLLM (TP={tp}): {' '.join(cmd[-8:])}")
    proc = subprocess.Popen(cmd, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)

    time.sleep(5)
    if proc.poll() is not None:
        print(f"vLLM crashed: {open('/tmp/vllm.log').read()[-3000:]}")
        raise RuntimeError("vLLM crashed on startup")

    for i in range(120):
        try:
            r = req.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready ({i * 5}s)")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"vLLM died: {open('/tmp/vllm.log').read()[-3000:]}")
            raise RuntimeError("vLLM died during startup")
        time.sleep(5)

    raise RuntimeError("vLLM startup timeout")


def _run_executor(
    task_file_contents: str,
    cua_model: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 5,
) -> dict:
    """Shared executor logic for all GPU tiers."""
    import requests as req
    from datetime import datetime, timezone

    from mantis_agent.brain_opencua import OpenCUABrain
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    from mantis_agent.gym.runner import GymRunner

    plan_inputs = plan_inputs or {}
    cua_config = CUA_MODELS.get(cua_model, CUA_MODELS["evocua-8b"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # Download + start vLLM
    slug = cua_model.replace("-", "_")
    model_dir = f"/data/models/{slug}"
    marker = os.path.join(model_dir, ".download_complete")
    if not os.path.exists(marker):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading {cua_config['repo']}...")
        from huggingface_hub import snapshot_download
        snapshot_download(cua_config["repo"], local_dir=model_dir, ignore_patterns=["*.md", "*.txt"])
        open(marker, "w").write("done")
        vol.commit()
    else:
        print(f"{cua_config['name']} cached at {model_dir}")

    tp = cua_config["tp"]
    vllm_proc = _start_vllm(model_dir, port=8000, tp=tp)

    # Parse task suite
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "cua_run")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\n{'='*60}")
    print(f"Mantis CUA Server — {cua_config['name']}")
    print(f"  Session:  {session_name}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Model:    {cua_config['name']} (TP={tp})")
    print(f"{'='*60}")

    # Create brain
    brain = OpenCUABrain(
        base_url="http://localhost:8000/v1",
        model="model",
        max_tokens=2048,
        temperature=0.0,
        screen_size=(1280, 720),
    )
    brain.load()

    # Xvfb + xdotool + real Chrome (zero automation fingerprints)
    proxy = _build_proxy_config()
    proxy_server = ""
    if proxy:
        # Start local auth forwarder for authenticated proxy
        if proxy.get("username"):
            _start_local_proxy(proxy, local_port=3128)
            proxy_server = "http://127.0.0.1:3128"
        else:
            proxy_server = proxy["server"]
        print(f"  Proxy: {proxy.get('server', '')}")

    env = XdotoolGymEnv(
        start_url=base_url,
        viewport=(1280, 720),
        browser="google-chrome",
        settle_time=2.0,
        human_speed=True,
        proxy_server=proxy_server,
    )

    # Run tasks
    scores = []
    task_details = []
    chrome_proc = None  # Not needed — XdotoolGymEnv manages its own browser
    results_path = f"/data/results/cua_results_{session_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)

    def save_progress():
        completed_at = datetime.now(timezone.utc).isoformat() if len(scores) == len(tasks) else ""
        summary = {
            "run_id": run_id,
            "session_name": session_name,
            "model": cua_config["name"],
            "tasks_run": len(tasks),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_gpu_time_s": round(time.time() - t0),
            "estimated_cost_usd": round((time.time() - t0) / 3600 * (3.25 * tp), 2),
            "scores": scores,
            "task_details": task_details,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        vol.commit()

    for i, task_config in enumerate(tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]

        print(f"\nTask {i+1}/{len(tasks)}: {task_id}")
        task_start = time.time()

        try:
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)

            # Dynamic loop task
            if task_config.get("loop"):
                from mantis_agent.gym.workflow_runner import WorkflowRunner, LoopConfig

                def on_loop_iteration(iter_num, iter_result, all_results):
                    viable = sum(1 for r in all_results if r.success)
                    total = len(all_results)
                    elapsed = time.time() - task_start
                    detail = {
                        "task_id": task_id, "success": viable > 0,
                        "steps": sum(r.steps for r in all_results),
                        "duration_s": round(elapsed),
                        "termination_reason": "loop_in_progress",
                        "iterations": total, "viable": viable,
                        "data": [r.data[:200] for r in all_results if r.data],
                    }
                    if task_details and task_details[-1].get("task_id") == task_id:
                        task_details[-1] = detail
                    else:
                        scores.append(0.0)
                        task_details.append(detail)
                    save_progress()
                    status = "VIABLE" if iter_result.success else "SKIP"
                    print(f"  [{iter_num}] {status} — {viable}/{total} viable ({elapsed:.0f}s)")

                loop_cfg = LoopConfig(
                    iteration_intent=intent,
                    pagination_intent=task_config["loop"].get("pagination_intent",
                        "Scroll to bottom, click Next page. If no next, terminate('failure')."),
                    max_iterations=task_config["loop"].get("max_iterations", 50),
                    max_pages=task_config["loop"].get("max_pages", 10),
                    max_steps_per_iteration=task_config["loop"].get("max_steps_per_iteration", max_steps),
                )
                wf_runner = WorkflowRunner(brain=brain, env=env, loop_config=loop_cfg,
                                           on_iteration=on_loop_iteration)
                results = wf_runner.run_loop()
                viable = sum(1 for r in results if r.success)
                total = len(results)
                success = viable > 0
                final = {
                    "task_id": task_id, "success": success,
                    "steps": sum(r.steps for r in results),
                    "duration_s": round(time.time() - task_start),
                    "termination_reason": "loop_complete",
                    "iterations": total, "viable": viable,
                    "data": [r.data[:200] for r in results if r.data],
                }
                if task_details and task_details[-1].get("task_id") == task_id:
                    task_details[-1] = final
                    scores[-1] = 1.0 if success else 0.0
                else:
                    scores.append(1.0 if success else 0.0)
                    task_details.append(final)
                print(f"  Loop: {viable}/{total} viable")
                save_progress()
                continue

            # Standard task with retry
            runner = GymRunner(brain=brain, env=env, max_steps=max_steps,
                               frames_per_inference=frames_per_inference)
            result = runner.run(task=intent, task_id=task_id)

            if task_config.get("save_session"):
                if result.success or ("login" not in env.current_url.lower()):
                    env.save_session(session_name)

            verify = task_config.get("verify", {})
            verified = False
            vtype, value = verify.get("type", ""), verify.get("value", "")
            try:
                if vtype == "url_contains":
                    verified = value.lower() in env.current_url.lower()
            except Exception:
                pass

            success = result.success or verified
            scores.append(1.0 if success else 0.0)
            task_details.append({
                "task_id": task_id, "success": success,
                "steps": result.total_steps,
                "duration_s": round(time.time() - task_start),
                "termination_reason": result.termination_reason,
                "final_url": env.current_url,
            })
            print(f"  {'PASS' if success else 'FAIL'} ({result.total_steps} steps)")

        except Exception as e:
            print(f"  ERROR: {e}")
            scores.append(0.0)
            task_details.append({
                "task_id": task_id, "success": False,
                "error": str(e), "duration_s": round(time.time() - task_start),
            })

        save_progress()

    env.close()
    vllm_proc.terminate()
    if chrome_proc:
        chrome_proc.terminate()

    passed = sum(1 for s in scores if s > 0)
    avg = sum(scores) / len(scores) * 100 if scores else 0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {passed}/{len(scores)} ({avg:.1f}%)")
    print(f"GPU time: {(time.time()-t0)/60:.0f} min | Cost: ${(time.time()-t0)/3600*(3.25*tp):.2f}")
    print(f"{'='*60}")

    save_progress()
    return {"passed": passed, "total": len(scores), "score": avg}


# ═══════════════════════════════════════════════════════════════════
# C) GPU-tier executor functions
# ═══════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_cua_1gpu(task_file_contents: str, cua_model: str = "evocua-8b", **kwargs) -> dict:
    """EvoCUA-8B executor (1× A100, TP=1)."""
    return _run_executor(task_file_contents, cua_model=cua_model, **kwargs)


@app.function(
    gpu="A100-80GB:2",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_cua_2gpu(task_file_contents: str, cua_model: str = "evocua-32b", **kwargs) -> dict:
    """EvoCUA-32B executor (2× A100, TP=2)."""
    return _run_executor(task_file_contents, cua_model=cua_model, **kwargs)


@app.function(
    gpu="A100-80GB:4",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_cua_4gpu(task_file_contents: str, cua_model: str = "opencua-32b", **kwargs) -> dict:
    """OpenCUA-32B executor (4× A100, TP=4)."""
    return _run_executor(task_file_contents, cua_model=cua_model, **kwargs)


@app.function(
    gpu="A100-80GB:8",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,
    memory=131072,
    cpu=32,
)
def run_cua_8gpu(task_file_contents: str, cua_model: str = "opencua-72b", **kwargs) -> dict:
    """OpenCUA-72B executor (8× A100, TP=8)."""
    return _run_executor(task_file_contents, cua_model=cua_model, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# C.2) Gemma4-31B-CUA executor (llama.cpp, native tool calling)
# ═══════════════════════════════════════════════════════════════════

def _run_gemma4_cua_executor(
    task_file_contents: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
) -> dict:
    """Execute tasks using fine-tuned Gemma4-31B-CUA via llama.cpp."""
    import requests as req
    from datetime import datetime, timezone

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    from mantis_agent.gym.runner import GymRunner

    plan_inputs = plan_inputs or {}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # Resolve model
    model_path = _resolve_gemma4_model()
    model_dir = os.path.dirname(model_path)

    # Find mmproj in same directory
    mmproj = ""
    for f in os.listdir(model_dir):
        if "mmproj" in f.lower() and f.endswith(".gguf"):
            mmproj = os.path.join(model_dir, f)
            break

    # Start llama-server
    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99", "-c", "32768", "-ub", "2048",
        "--jinja", "--reasoning-budget", "4096",
        "--flash-attn", "on",
    ]
    if mmproj:
        cmd.extend(["--mmproj", mmproj])
    print(f"Starting Gemma4-CUA: {' '.join(cmd[-8:])}")
    llama_proc = subprocess.Popen(cmd, stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)

    # Wait for readiness
    time.sleep(3)
    for i in range(90):
        try:
            r = req.get("http://localhost:8080/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"llama-server ready ({i * 2}s)")
                break
        except Exception:
            pass
        if llama_proc.poll() is not None:
            print(f"llama-server crashed: {open('/tmp/llama.log').read()[-3000:]}")
            raise RuntimeError("llama-server crashed")
        time.sleep(2)
    else:
        print(f"llama-server timeout: {open('/tmp/llama.log').read()[-2000:]}")
        raise RuntimeError("llama-server timeout")

    # Parse task suite
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "gemma4_cua")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\n{'='*60}")
    print(f"Mantis CUA Server — Gemma4-31B-CUA (llama.cpp)")
    print(f"  Session:  {session_name}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"{'='*60}")

    # Create brain (LlamaCpp, not OpenCUA)
    brain = LlamaCppBrain(
        base_url="http://localhost:8080/v1",
        model="gemma4-cua",
        max_tokens=2048,
        temperature=0.0,
        use_tool_calling=True,
    )
    brain.load()

    # Xvfb + xdotool + real Chrome (zero automation fingerprints)
    proxy = _build_proxy_config()
    proxy_server = ""
    if proxy:
        if proxy.get("username"):
            _start_local_proxy(proxy, local_port=3128)
            proxy_server = "http://127.0.0.1:3128"
        else:
            proxy_server = proxy["server"]
        print(f"  Proxy: {proxy.get('server', '')}")

    env = XdotoolGymEnv(
        start_url=base_url,
        viewport=(1280, 720),
        browser="google-chrome",
        settle_time=2.0,
        human_speed=True,
        proxy_server=proxy_server,
    )

    # Run tasks (same loop as _run_executor)
    scores = []
    task_details = []
    chrome_proc = None
    results_path = f"/data/results/gemma4cua_results_{session_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)

    def save_progress():
        completed_at = datetime.now(timezone.utc).isoformat() if len(scores) == len(tasks) else ""
        summary = {
            "run_id": run_id,
            "session_name": session_name,
            "model": "Gemma4-31B-CUA",
            "tasks_run": len(tasks),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_gpu_time_s": round(time.time() - t0),
            "estimated_cost_usd": round((time.time() - t0) / 3600 * 3.25, 2),
            "scores": scores,
            "task_details": task_details,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        vol.commit()

    for i, task_config in enumerate(tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]
        print(f"\nTask {i+1}/{len(tasks)}: {task_id}")
        task_start = time.time()

        try:
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)

            if task_config.get("loop"):
                from mantis_agent.gym.workflow_runner import WorkflowRunner, LoopConfig

                def on_loop_iteration(iter_num, iter_result, all_results):
                    viable = sum(1 for r in all_results if r.success)
                    total = len(all_results)
                    elapsed = time.time() - task_start
                    detail = {
                        "task_id": task_id, "success": viable > 0,
                        "steps": sum(r.steps for r in all_results),
                        "duration_s": round(elapsed),
                        "termination_reason": "loop_in_progress",
                        "iterations": total, "viable": viable,
                        "data": [r.data[:200] for r in all_results if r.data],
                    }
                    if task_details and task_details[-1].get("task_id") == task_id:
                        task_details[-1] = detail
                    else:
                        scores.append(0.0)
                        task_details.append(detail)
                    save_progress()
                    print(f"  [{iter_num}] {'VIABLE' if iter_result.success else 'SKIP'} — {viable}/{total} viable ({elapsed:.0f}s)")

                loop_cfg = LoopConfig(
                    iteration_intent=intent,
                    pagination_intent=task_config["loop"].get("pagination_intent",
                        "Scroll to bottom, click Next page. If no next, terminate('failure')."),
                    max_iterations=task_config["loop"].get("max_iterations", 50),
                    max_pages=task_config["loop"].get("max_pages", 10),
                    max_steps_per_iteration=task_config["loop"].get("max_steps_per_iteration", max_steps),
                )
                wf_runner = WorkflowRunner(brain=brain, env=env, loop_config=loop_cfg,
                                           on_iteration=on_loop_iteration)
                results = wf_runner.run_loop()
                viable = sum(1 for r in results if r.success)
                total = len(results)
                success = viable > 0
                final = {
                    "task_id": task_id, "success": success,
                    "steps": sum(r.steps for r in results),
                    "duration_s": round(time.time() - task_start),
                    "termination_reason": "loop_complete",
                    "iterations": total, "viable": viable,
                    "data": [r.data[:200] for r in results if r.data],
                }
                if task_details and task_details[-1].get("task_id") == task_id:
                    task_details[-1] = final
                    scores[-1] = 1.0 if success else 0.0
                else:
                    scores.append(1.0 if success else 0.0)
                    task_details.append(final)
                print(f"  Loop: {viable}/{total} viable")
                save_progress()
                continue

            runner = GymRunner(brain=brain, env=env, max_steps=max_steps, frames_per_inference=5)
            result = runner.run(task=intent, task_id=task_id)

            if task_config.get("save_session"):
                if result.success or ("login" not in env.current_url.lower()):
                    env.save_session(session_name)

            verify = task_config.get("verify", {})
            verified = False
            vtype, value = verify.get("type", ""), verify.get("value", "")
            try:
                if vtype == "url_contains":
                    verified = value.lower() in env.current_url.lower()
            except Exception:
                pass

            success = result.success or verified
            scores.append(1.0 if success else 0.0)
            task_details.append({
                "task_id": task_id, "success": success,
                "steps": result.total_steps,
                "duration_s": round(time.time() - task_start),
                "termination_reason": result.termination_reason,
            })
            print(f"  {'PASS' if success else 'FAIL'} ({result.total_steps} steps)")

        except Exception as e:
            print(f"  ERROR: {e}")
            scores.append(0.0)
            task_details.append({
                "task_id": task_id, "success": False,
                "error": str(e), "duration_s": round(time.time() - task_start),
            })

        save_progress()

    env.close()
    llama_proc.terminate()
    if chrome_proc:
        chrome_proc.terminate()

    passed = sum(1 for s in scores if s > 0)
    avg = sum(scores) / len(scores) * 100 if scores else 0
    print(f"\n{'='*60}")
    print(f"COMPLETE: {passed}/{len(scores)} ({avg:.1f}%)")
    print(f"GPU time: {(time.time()-t0)/60:.0f} min | Cost: ${(time.time()-t0)/3600*3.25:.2f}")
    print(f"{'='*60}")

    save_progress()
    return {"passed": passed, "total": len(scores), "score": avg}


@app.function(
    gpu="A100-80GB",
    image=planner_image.run_commands(
        "apt-get update && apt-get install -y gnupg curl wget",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable xvfb || true",
    ).pip_install(
        "openai", "requests", "pillow", "playwright", "playwright-stealth",
    ).run_commands("playwright install --with-deps chromium || true")
    .add_local_python_source("mantis_agent"),
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_gemma4_cua(task_file_contents: str, **kwargs) -> dict:
    """Gemma4-31B-CUA executor (1× A100, llama.cpp, native tool calling)."""
    kwargs.pop("cua_model", None)  # Not used — model is always Gemma4-31B-CUA
    return _run_gemma4_cua_executor(task_file_contents, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# D) Local entrypoint — orchestrates planner + executor
# ═══════════════════════════════════════════════════════════════════

# Map model names to executor functions
EXECUTOR_MAP = {
    "evocua-8b": run_cua_1gpu,
    "evocua-32b": run_cua_2gpu,
    "opencua-32b": run_cua_4gpu,
    "opencua-72b": run_cua_8gpu,
    "gemma4-cua": run_gemma4_cua,
}

APP_NAME = "mantis-cua-server"


@app.local_entrypoint()
def main(
    task_file: str = "",
    plan_file: str = "",
    model: str = "evocua-8b",
    max_steps: int = 30,
    max_retries: int = 2,
    inputs: str = "",
    session_name: str = "",
    max_listings: int = 50,
):
    """Mantis CUA Server — run plans or task suites on Modal.

    Two modes:
      --plan-file plans/boattrader/full_spec.txt   (Gemma4 preprocesses → EvoCUA executes)
      --task-file tasks/boattrader/dynamic.json     (direct execution, no planner)

    Models: evocua-8b, evocua-32b, opencua-32b, opencua-72b
    """
    cua_config = CUA_MODELS.get(model, CUA_MODELS["evocua-8b"])
    print(f"Mantis CUA Server — {cua_config['name']}")

    # Parse inputs
    plan_inputs = {}
    if inputs:
        for pair in inputs.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                plan_inputs[k.strip()] = v.strip()

    # Mode 1: Plan file → Gemma4 preprocessing → EvoCUA execution
    if plan_file:
        print(f"  Plan:    {plan_file}")
        print(f"  Mode:    Gemma4 → {cua_config['name']}")

        with open(plan_file) as f:
            plan_text = f.read()

        if not session_name:
            session_name = os.path.splitext(os.path.basename(plan_file))[0]

        # Resolve planner URL from deployed app
        planner_url = f"https://{APP_NAME}--gemma4-planner.modal.run"
        print(f"  Planner: {planner_url}")

        # Verify planner is reachable
        import requests
        try:
            r = requests.get(f"{planner_url}/v1/models", timeout=10)
            if r.status_code != 200:
                print(f"  WARNING: Planner not responding. Deploy first: modal deploy modal_cua_server.py")
                sys.exit(1)
            print(f"  Planner: OK")
        except Exception as e:
            print(f"  ERROR: Cannot reach planner at {planner_url}")
            print(f"  Deploy first: uv run modal deploy modal_cua_server.py")
            sys.exit(1)

        # Use Gemma4 to preprocess the plan
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from mantis_agent.brain_llamacpp import LlamaCppBrain
        from mantis_agent.gym.plan_optimizer import optimize_plan

        planner_brain = LlamaCppBrain(
            base_url=f"{planner_url}/v1",
            model="model",
            max_tokens=4096,
            temperature=0.0,
        )

        print(f"  Preprocessing plan with Gemma4...")
        task_suite = optimize_plan(
            plan_text=plan_text,
            inputs=plan_inputs,
            session_name=session_name,
            max_listings=max_listings,
            brain=planner_brain,
        )

        opt_info = task_suite.pop("_optimization", {})
        print(f"  Sections: {opt_info.get('sections_generated', '?')}")
        for t in task_suite.get("tasks", []):
            loop = " [LOOP]" if t.get("loop") else ""
            print(f"    {t['task_id']:25s}{loop}")

        task_file_contents = json.dumps(task_suite)

    # Mode 2: Pre-built task suite
    elif task_file:
        print(f"  Tasks:   {task_file}")
        print(f"  Mode:    Direct → {cua_config['name']}")

        with open(task_file) as f:
            task_file_contents = f.read()

    else:
        print("ERROR: Provide --plan-file or --task-file")
        sys.exit(1)

    # Route to correct GPU-tier executor
    executor_fn = EXECUTOR_MAP.get(model, run_cua_1gpu)
    print(f"\n  Launching {cua_config['name']} on Modal ({cua_config['tp']}× A100)...")

    result = executor_fn.remote(
        task_file_contents=task_file_contents,
        cua_model=model,
        plan_inputs=plan_inputs,
        max_steps=max_steps,
        max_retries=max_retries,
    )

    print(f"\nResult: {json.dumps(result, indent=2)}")
