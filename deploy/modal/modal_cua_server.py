"""Mantis CUA Server — fully remote Gemma4 + EvoCUA on Modal.

Single `modal deploy` gives you:
  - Gemma4 planner (T4 GPU, llama.cpp, persistent web server)
  - EvoCUA executors (A100 GPUs, vLLM, per-run)

Usage:
    # Deploy the planner (stays running, warm for 10 min):
    uv run modal deploy deploy/modal/modal_cua_server.py

    # Run from a free-text plan (Gemma4 preprocesses → EvoCUA executes):
    uv run modal run deploy/modal/modal_cua_server.py \
        --plan-file plans/boattrader/full_spec.txt \
        --model evocua-8b \
        --inputs "pop_password=SelfService38#,zip_code=33101,search_radius=35"

    # Run from a pre-built task suite (bypasses planner):
    uv run modal run deploy/modal/modal_cua_server.py \
        --task-file tasks/boattrader/dynamic_production.json \
        --model evocua-8b
"""

import json
import os
import subprocess
import sys
import time

import modal

from mantis_agent.server_utils import (
    build_micro_result,
    build_micro_suite,
    build_proxy_config,
    micro_plan_steps_to_dicts,
    plan_signature_from_steps,
    resolve_proxy_server,
    safe_state_key,
    save_result_json,
    start_local_proxy,
    wait_for_openai_server,
)

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
    "holo3": {
        "repo": "gguf:mradermacher/Holo3-35B-A3B-GGUF",  # llama.cpp — vLLM lacks qwen3_5_moe
        "name": "Holo3-35B-A3B",
        "tp": 1,  # 1x A100 — Q8_0 is 34GB + mmproj 0.8GB = ~35GB
    },
    "claude": {
        "repo": "api",
        "name": "Claude (Anthropic API)",
        "tp": 0,  # No GPU needed
    },
}


_plan_signature_from_steps = plan_signature_from_steps  # backward compat alias
_safe_state_key = safe_state_key  # backward compat alias

# Fine-tuned Gemma4-31B-CUA (trained on AgentNet, native tool calling)
# Already quantized to GGUF on the Modal volume
GEMMA4_CUA_DIR = "/data/training/gemma4-cua-gguf_gguf"
GEMMA4_CUA_FILE = "gemma-4-31b-it.Q4_K_M.gguf"
GEMMA4_CUA_MMPROJ = "gemma-4-31b-it.BF16-mmproj.gguf"

# Holo3-35B-A3B via llama.cpp GGUF (Q8_0 — 34GB fits 1x A100)
HOLO3_GGUF_REPO = "mradermacher/Holo3-35B-A3B-GGUF"
HOLO3_GGUF_FILE = "Holo3-35B-A3B.Q8_0.gguf"
HOLO3_MMPROJ_FILE = "Holo3-35B-A3B.mmproj-f16.gguf"
HOLO3_MODEL_DIR = "/data/models/holo3_gguf"

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
        "git clone --depth 1 --branch b8948 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",  # SHA 42401c72b8d239240ed0fb37694d29ac33b3bc4f
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
        "fastapi>=0.100", "uvicorn>=0.20",
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

_start_local_proxy = start_local_proxy  # backward compat alias


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


_build_proxy_config = build_proxy_config  # backward compat alias


# ═══════════════════════════════════════════════════════════════════
# C) Shared executor logic
# ═══════════════════════════════════════════════════════════════════

def _start_vllm(model_dir: str, port: int, tp: int,
                extra_args: list[str] | None = None) -> subprocess.Popen:
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
    if extra_args:
        cmd.extend(extra_args)
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
            vllm_log = open('/tmp/vllm.log').read()[-3000:]
            print(f"vLLM died: {vllm_log}")
            raise RuntimeError(f"vLLM died during startup:\n{vllm_log[-500:]}")
        time.sleep(5)

    raise RuntimeError("vLLM startup timeout")


def _run_executor(
    task_file_contents: str,
    cua_model: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 5,
    viewer: bool = False,
) -> dict:
    """Shared executor logic for all vLLM GPU tiers."""
    from datetime import datetime, timezone

    from mantis_agent.task_loop import TaskLoopConfig, setup_env, setup_viewer, run_executor_lifecycle

    plan_inputs = plan_inputs or {}
    cua_config = CUA_MODELS.get(cua_model, CUA_MODELS["evocua-8b"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ── Model download + vLLM startup (unique to vLLM executor) ──
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

    # ── Brain (model-dependent) ──
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "cua_run")

    if cua_model == "holo3":
        from mantis_agent.brain_holo3 import Holo3Brain
        brain = Holo3Brain(
            base_url="http://localhost:8000/v1", model="model",
            max_tokens=2048, temperature=0.0,
            screen_size=(1280, 720), use_tool_calling=True,
        )
    else:
        from mantis_agent.brain_opencua import OpenCUABrain
        brain = OpenCUABrain(
            base_url="http://localhost:8000/v1", model="model",
            max_tokens=2048, temperature=0.0, screen_size=(1280, 720),
        )
    brain.load()

    # ── Env + viewer ──
    env, proxy_proc = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name, settle_time=2.0,
    )
    viewer_ctx, viewer_event_bus = setup_viewer(viewer)

    # ── Delegate to shared lifecycle ──
    config = TaskLoopConfig(
        run_id=run_id, session_name=session_name,
        model_name=cua_config["name"],
        results_prefix="holo3" if cua_model == "holo3" else "cua",
        brain=brain, env=env,
        max_steps=max_steps, frames_per_inference=frames_per_inference,
        viewer_event_bus=viewer_event_bus,
        volume_commit=vol.commit,
        summary_extras={"estimated_cost_usd": round((time.time() - t0) / 3600 * (3.25 * tp), 2)},
    )
    return run_executor_lifecycle(
        task_suite, config,
        server_proc=vllm_proc, proxy_proc=proxy_proc, viewer_ctx=viewer_ctx, t0=t0,
    )


# ═══════════════════════════════════════════════════════════════════
# C) GPU-tier executor functions
# ═══════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours
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
    timeout=14400,  # 4 hours
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
    timeout=14400,  # 4 hours
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
    timeout=14400,  # 4 hours
    memory=131072,
    cpu=32,
)
def run_cua_8gpu(task_file_contents: str, cua_model: str = "opencua-72b", **kwargs) -> dict:
    """OpenCUA-72B executor (8× A100, TP=8)."""
    return _run_executor(task_file_contents, cua_model=cua_model, **kwargs)


def _run_holo3_executor(
    task_file_contents: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 1,
    viewer: bool = False,
    sub_plan: bool = True,
    **_extra,
) -> dict:
    """Execute tasks using Holo3-35B-A3B via llama.cpp (GGUF on 1x A100).

    Like Gemma4-CUA: llama-server + GGUF + mmproj for vision.
    vLLM doesn't support qwen3_5_moe yet.
    """
    from datetime import datetime, timezone

    from mantis_agent.brain_holo3 import Holo3Brain
    from mantis_agent.task_loop import setup_env, setup_viewer

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # Download GGUF if not cached
    model_path = os.path.join(HOLO3_MODEL_DIR, HOLO3_GGUF_FILE)
    mmproj_path = os.path.join(HOLO3_MODEL_DIR, HOLO3_MMPROJ_FILE)
    marker = os.path.join(HOLO3_MODEL_DIR, ".download_complete")

    if not os.path.exists(marker):
        os.makedirs(HOLO3_MODEL_DIR, exist_ok=True)
        print(f"Downloading Holo3 GGUF ({HOLO3_GGUF_FILE})...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HOLO3_GGUF_REPO, filename=HOLO3_GGUF_FILE, local_dir=HOLO3_MODEL_DIR)
        hf_hub_download(repo_id=HOLO3_GGUF_REPO, filename=HOLO3_MMPROJ_FILE, local_dir=HOLO3_MODEL_DIR)
        open(marker, "w").write("done")
        vol.commit()
        print("Download complete.")
    else:
        print(f"Holo3 GGUF cached at {HOLO3_MODEL_DIR}")

    # Start llama-server — no reasoning-budget (Holo3 overthinks with budget=512)
    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99", "-c", "8192", "-ub", "2048",
        "--jinja",
        "--flash-attn", "on",
    ]
    print(f"Starting Holo3 llama-server: {' '.join(cmd[-8:])}")
    llama_proc = subprocess.Popen(cmd, stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)
    wait_for_openai_server(8080, llama_proc, "llama-server")

    # Parse task suite
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "holo3_cua")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\n{'='*60}")
    print("Mantis CUA Server — Holo3-35B-A3B (llama.cpp)")
    print(f"  Session:  {session_name}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"{'='*60}")

    # Create brain (local llama-server, not remote API)
    brain = Holo3Brain(
        base_url="http://localhost:8080/v1",
        model="holo3",
        api_key="",
        max_tokens=2048,
        temperature=0.0,
        screen_size=(1280, 720),
        use_tool_calling=True,
    )
    brain.load()

    # Claude Sonnet grounding for click targeting
    from mantis_agent.grounding import ClaudeGrounding
    grounding = ClaudeGrounding()

    # Env + viewer via shared helpers
    env, proxy_proc = setup_env(
        base_url=base_url, run_id=run_id, session_name=session_name,
        settle_time=4.0,  # Holo3 needs longer settle — sees black screen with 2s
    )
    from mantis_agent.extraction import ClaudeExtractor, ExtractionSchema
    schema = None
    objective_data = task_suite.get("_objective")
    if objective_data:
        from mantis_agent.graph.objective import ObjectiveSpec
        objective = ObjectiveSpec.from_dict(objective_data)
        schema = ExtractionSchema.from_objective(objective)
    extractor = ClaudeExtractor(schema=schema)
    viewer_ctx, viewer_event_bus = setup_viewer(viewer)

    # ── Micro-intent mode: run MicroPlanRunner ──
    micro_plan_data = task_suite.get("_micro_plan")
    if micro_plan_data:
        from mantis_agent.plan_decomposer import MicroPlan, MicroIntent
        from mantis_agent.gym.micro_runner import MicroPlanRunner

        micro_plan = MicroPlan(domain=session_name)
        for s in micro_plan_data:
            micro_plan.steps.append(MicroIntent(**s))

        # If objective is available, run site probe to enhance the plan
        # This runs INSIDE the container where env/browser is available
        objective_data = task_suite.get("_objective")
        if objective_data and env:
            try:
                from mantis_agent.graph.objective import ObjectiveSpec as _OS
                from mantis_agent.graph.probe import SiteProber, ProbeResult
                from mantis_agent.graph.enhancer import PlanEnhancer
                from mantis_agent.graph import GraphCompiler, PlanValidator
                from mantis_agent.graph.graph import WorkflowGraph
                from mantis_agent.verification.playbook import Playbook

                obj = _OS.from_dict(objective_data)
                if obj.start_url:
                    print(f"\n  === SITE PROBE (inside container) ===")
                    prober = SiteProber(env=env)
                    probe = prober.probe(obj.start_url, obj)
                    print(f"  Probe: {probe.page_type}, {len(probe.filters_detected)} filters, {probe.estimated_listings_per_page} listings/page")

                    # Re-enhance with probe data
                    enhancer = PlanEnhancer()
                    enhancement = enhancer.enhance(obj, probe)
                    phases, edges = enhancer.build_enhanced_phases(obj, probe, enhancement)
                    print(f"  Enhanced: {len(phases)} phases, nav={enhancement.get('navigation_url', '')[:60]}")

                    # Recompile
                    graph = WorkflowGraph(
                        objective=obj, phases=phases, edges=edges,
                        playbook=Playbook(domain=obj.domains[0] if obj.domains else "", listings_per_page=probe.estimated_listings_per_page or 25),
                        domain=obj.domains[0] if obj.domains else "",
                        objective_hash=obj.objective_hash,
                    )
                    compiler = GraphCompiler()
                    micro_plan = compiler.compile(graph)

                    # Validate
                    validator = PlanValidator()
                    issues = validator.validate(micro_plan, objective=obj)
                    if issues:
                        micro_plan = validator.enhance(micro_plan, objective=obj)

                    print(f"  Probe-enhanced plan: {len(micro_plan.steps)} steps")
                    print(micro_plan.summary())
            except Exception as e:
                print(f"  Probe enhancement failed (using original plan): {e}")

        resume_state = bool(task_suite.get("_resume_state", False))
        state_key = task_suite.get("_state_key", "")
        checkpoint_path = task_suite.get("_checkpoint_path") or f"/data/checkpoints/micro_{session_name}_{run_id}.json"
        plan_signature = task_suite.get("_plan_signature", "")

        print(f"\n  === MICRO-INTENT MODE ({len(micro_plan.steps)} steps) ===")
        print(micro_plan.summary())
        if state_key:
            print(f"  State key:  {state_key}")
            print(f"  Resume:     {'on' if resume_state else 'off'}")
            print(f"  Checkpoint: {checkpoint_path}")

        micro_runner = MicroPlanRunner(
            brain=brain, env=env,
            grounding=grounding, extractor=extractor,
            on_step=viewer_event_bus.emit if viewer_event_bus else None,
            checkpoint_path=checkpoint_path,
            run_key=state_key or session_name,
            session_name=session_name,
            plan_signature=plan_signature,
            resume_state=resume_state,
            on_checkpoint=vol.commit,
            max_cost=task_suite.get("_max_cost", 10.0),
            max_time_minutes=task_suite.get("_max_time_minutes", 180),
        )
        step_results = micro_runner.run(micro_plan, resume=resume_state)

        # Build standardized result with dynamic verification
        from pathlib import Path as _Path
        result = build_micro_result(
            micro_runner,
            step_results,
            run_id=run_id,
            provider="modal",
            session_name=session_name,
            model_name="Holo3-35B-A3B (micro)",
            elapsed_seconds=time.time() - t0,
            state_key=state_key,
            checkpoint_path=checkpoint_path,
            plan_signature=plan_signature,
            resume_state=resume_state,
        )
        save_result_json(result, _Path("/data/results"), "holo3")
        vol.commit()

        viable = result["viable"]
        leads = result.get("leads", [])
        print(
            f"\n  Micro-intent complete: {viable} viable leads "
            f"({result['leads_with_phone']} with phone) from {result['steps_executed']} steps"
        )
        for i, lead in enumerate(leads, 1):
            print(f"    [{i}] {lead[:150]}")

        env.close()
        llama_proc.terminate()
        if viewer_ctx:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        return {
            "mode": "micro",
            "viable": viable,
            "steps": result["steps_executed"],
            "state_key": state_key,
            "checkpoint_path": checkpoint_path,
            "status": result.get("costs", {}).get("status", ""),
            "dynamic_verification_summary": result.get("dynamic_verification_summary"),
        }

    # ── Learning mode: run LearningRunner instead of normal task loop ──
    learn_mode = task_suite.get("_learn", False)
    learn_samples = task_suite.get("_learn_samples", 5)
    verify_mode = task_suite.get("_verify", False)

    if learn_mode:
        from mantis_agent.verification.step_verifier import StepVerifier
        from mantis_agent.verification.playbook import PlaybookStore
        from mantis_agent.gym.learning_runner import LearningRunner

        print(f"\n  === LEARNING MODE ({learn_samples} samples) ===")
        verifier = StepVerifier()
        learning_runner = LearningRunner(
            brain=brain, env=env, verifier=verifier,
            grounding=grounding, on_step=viewer_event_bus.emit if viewer_event_bus else None,
        )

        # Extract setup and extraction intents from tasks
        setup_intent = ""
        extract_intent = ""
        start_url_for_learn = base_url
        expected_filters = []
        for tc in tasks:
            if "setup" in tc.get("task_id", "") or "filter" in tc.get("task_id", ""):
                setup_intent = tc["intent"]
                start_url_for_learn = tc.get("start_url", base_url)
            elif tc.get("loop"):
                extract_intent = tc["intent"]

        # Extract filter expectations from setup intent
        import re as _re
        for kw in ["private seller", "by owner", "zip", "price", "sort"]:
            if kw in setup_intent.lower():
                expected_filters.append(kw)

        # Derive domain
        domain = ""
        m = _re.search(r"(?:https?://)?(?:www\.)?([\w\-]+\.[\w]+)", start_url_for_learn)
        if m:
            domain = m.group(1)

        playbook = learning_runner.learn(
            setup_intent=setup_intent,
            extract_intent=extract_intent,
            domain=domain,
            start_url=start_url_for_learn,
            expected_filters=expected_filters,
            n_samples=learn_samples,
        )

        # Save playbook
        PlaybookStore().save(playbook)
        vol.commit()
        print(f"\n  Playbook saved for {domain}")
        print(playbook.summary())

        env.close()
        llama_proc.terminate()
        if viewer_ctx:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        return {"mode": "learn", "domain": domain, "steps": len(playbook.setup_steps) + len(playbook.extraction_steps)}

    # ── Holo3-specific task result callback ──
    # Handles: min_steps retry, hybrid verify mode, filter validation/recovery
    def _holo3_on_task_result(task_config, task_id, result, _env, _brain, config):
        from mantis_agent.gym.runner import GymRunner
        on_step = config.viewer_event_bus.emit if config.viewer_event_bus else None

        task_max_steps = task_config.get("max_steps", config.max_steps)
        min_steps = task_config.get("min_steps", 0)
        if "setup" in task_id or "filter" in task_id:
            min_steps = max(min_steps, 5)

        # Hybrid mode: use Claude for setup tasks
        if ("setup" in task_id or "filter" in task_id) and verify_mode:
            try:
                from mantis_agent.brain_claude import ClaudeBrain as _CB
                task_brain = _CB(model="claude-sonnet-4-20250514", thinking_budget=2048, screen_size=(1280, 720))
                task_brain.load()
                print("  Using Claude Sonnet for setup (hybrid mode)")
                runner = GymRunner(brain=task_brain, env=_env, max_steps=task_max_steps,
                                   frames_per_inference=config.frames_per_inference, on_step=on_step)
                result = runner.run(task=task_config["intent"], task_id=task_id,
                                   start_url=task_config.get("start_url", ""))
            except Exception as e:
                print(f"  Claude brain failed, using Holo3: {e}")

        # Retry if model skipped interaction
        if result.total_steps <= min_steps and result.success and min_steps > 0:
            print(f"  SKIP-DETECTED: {result.total_steps} steps (min={min_steps}). Retrying...")
            retry_intent = task_config["intent"] + (
                "\n\nCRITICAL: Your previous attempt called done() without interacting with the page. "
                "You MUST click input fields, type values, and verify results BEFORE calling done(). "
                "Interact with AT LEAST 3 different UI controls (input fields, dropdowns, buttons)."
            )
            runner2 = GymRunner(brain=_brain, env=_env, max_steps=task_max_steps,
                                frames_per_inference=config.frames_per_inference,
                                grounding=config.grounding, on_step=on_step)
            result = runner2.run(task=retry_intent, task_id=f"{task_id}_retry",
                                start_url=task_config.get("start_url", ""))
            print(f"  Retry: {result.total_steps} steps, success={result.success}")

        # Post-setup filter validation
        if "setup" in task_id or "filter" in task_id:
            sc = config.site_config
            gate_prompt = (sc.gate_verify_prompt if sc else "") or "Page shows filtered results"
            print("  Validating filters...")
            validate_runner = GymRunner(brain=_brain, env=_env, max_steps=8,
                                       frames_per_inference=1, grounding=config.grounding, on_step=on_step)
            validate_result = validate_runner.run(
                task=(
                    "READ the current page. Do NOT click anything.\n\n"
                    "Check if the page shows the expected filtered results:\n"
                    f"- {gate_prompt}\n"
                    "- Read the URL and page heading for filter evidence\n"
                    "- Check the result count is reasonable (not unfiltered)\n\n"
                    "If filters appear active: done(success=true, summary='Filters verified: [evidence]')\n"
                    "If filters are NOT applied: done(success=false, summary='Filters missing: [evidence]')"
                ),
                task_id=f"{task_id}_validate",
            )
            if not validate_result.success:
                recovery_url = (sc.filtered_results_url if sc else "") or ""
                if recovery_url:
                    print(f"  FILTERS NOT APPLIED — CUA recovery navigation to {recovery_url}")
                    recovery_runner = GymRunner(brain=_brain, env=_env, max_steps=15,
                                               frames_per_inference=1, grounding=config.grounding, on_step=on_step)
                    recovery_runner.run(
                        task=(
                            f"Navigate to the filtered results page. Steps:\n"
                            f"1. Click the browser address bar at the top of the screen\n"
                            f"2. Select all text in the address bar (Ctrl+A)\n"
                            f"3. Type: {recovery_url}\n"
                            f"4. Press Enter to navigate\n"
                            f"5. Wait for the page to load\n"
                            f"6. Press Home key to scroll to the top of the page\n"
                            f"7. done(success=true, summary='Navigated to filtered page')"
                        ),
                        task_id="filter_recovery_navigate",
                    )
                    time.sleep(3)
                    try:
                        from mantis_agent.actions import Action, ActionType
                        _env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                        time.sleep(1)
                    except Exception:
                        pass
                    print(f"  CUA navigated to {recovery_url}")
                else:
                    print("  FILTERS NOT APPLIED — no recovery URL configured, continuing")
            else:
                print("  Filters VERIFIED")

        return result

    # ── Delegate task loop to shared infrastructure ──
    from mantis_agent.task_loop import TaskLoopConfig, run_executor_lifecycle

    from mantis_agent.site_config import SiteConfig
    site_cfg = SiteConfig.default_boattrader()
    site_cfg_data = task_suite.get("_site_config")
    if site_cfg_data:
        site_cfg = SiteConfig.from_dict(site_cfg_data)

    config = TaskLoopConfig(
        run_id=run_id, session_name=session_name,
        model_name="Holo3-35B-A3B", results_prefix="holo3",
        brain=brain, env=env, grounding=grounding, extractor=extractor,
        max_steps=max_steps, frames_per_inference=frames_per_inference,
        use_sub_plan=sub_plan,
        site_config=site_cfg,
        viewer_event_bus=viewer_event_bus,
        on_task_result=_holo3_on_task_result,
        on_loop_complete=vol.commit,
        volume_commit=vol.commit,
    )
    return run_executor_lifecycle(
        task_suite, config,
        server_proc=llama_proc, viewer_ctx=viewer_ctx, t0=t0,
    )



# ═══════════════════════════════════════════════════════════════════
# C.2) Gemma4-31B-CUA executor (llama.cpp, native tool calling)
# ═══════════════════════════════════════════════════════════════════

def _run_gemma4_cua_executor(
    task_file_contents: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    viewer: bool = False,
) -> dict:
    """Execute tasks using fine-tuned Gemma4-31B-CUA via llama.cpp."""
    from datetime import datetime, timezone

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.grounding import ClaudeGrounding
    from mantis_agent.task_loop import TaskLoopConfig, setup_env, setup_viewer, run_executor_lifecycle

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ── Model startup (unique to Gemma4) ──
    model_path = _resolve_gemma4_model()
    model_dir = os.path.dirname(model_path)
    mmproj = ""
    for f in os.listdir(model_dir):
        if "mmproj" in f.lower() and f.endswith(".gguf"):
            mmproj = os.path.join(model_dir, f)
            break

    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0", "--port", "8080",
        "-ngl", "99", "-c", "8192", "-ub", "2048",
        "--jinja", "--reasoning-budget", "512",
        "--flash-attn", "on",
    ]
    if mmproj:
        cmd.extend(["--mmproj", mmproj])
    print(f"Starting Gemma4-CUA: {' '.join(cmd[-8:])}")
    llama_proc = subprocess.Popen(cmd, stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)
    wait_for_openai_server(8080, llama_proc, "llama-server")

    # ── Brain ──
    brain = LlamaCppBrain(
        base_url="http://localhost:8080/v1",
        model="gemma4-cua",
        max_tokens=512,
        temperature=0.0,
        use_tool_calling=True,
    )
    brain.load()
    grounding = ClaudeGrounding()

    # ── Env + viewer ──
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "gemma4_cua")
    env, proxy_proc = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name,
        settle_time=2.0, display=":99", start_xvfb=True,
    )
    viewer_ctx, viewer_event_bus = setup_viewer(viewer)

    # ── Delegate to shared lifecycle ──
    config = TaskLoopConfig(
        run_id=run_id, session_name=session_name,
        model_name="Gemma4-31B-CUA", results_prefix="gemma4cua",
        brain=brain, env=env, grounding=grounding,
        max_steps=max_steps, frames_per_inference=2,
        viewer_event_bus=viewer_event_bus,
        volume_commit=vol.commit,
        summary_extras={"estimated_cost_usd": round((time.time() - t0) / 3600 * 3.25, 2)},
    )
    return run_executor_lifecycle(
        task_suite, config,
        server_proc=llama_proc, proxy_proc=proxy_proc, viewer_ctx=viewer_ctx, t0=t0,
    )


@app.function(
    gpu="A100-80GB",
    image=planner_image.run_commands(
        "apt-get update && apt-get install -y gnupg curl wget xvfb xdotool scrot",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    ).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20",
    ).add_local_python_source("mantis_agent"),
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours — Gemma4 via llama.cpp is slower per step
    memory=65536,
    cpu=16,
)
def run_gemma4_cua(task_file_contents: str, **kwargs) -> dict:
    """Gemma4-31B-CUA executor (1× A100, llama.cpp, native tool calling)."""
    kwargs.pop("cua_model", None)  # Not used — model is always Gemma4-31B-CUA
    return _run_gemma4_cua_executor(task_file_contents, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# C.3) Claude CUA executor (Anthropic API, no GPU needed)
# ═══════════════════════════════════════════════════════════════════

# Lightweight image: just Chrome + xdotool (no vLLM, no llama.cpp)
claude_executor_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install("curl", "wget", "gnupg", "xvfb", "xdotool", "scrot")
    .run_commands(
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    )
    .pip_install(
        "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20",
    )
    .add_local_python_source("mantis_agent")
)


def _run_claude_executor(
    task_file_contents: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 5,
    claude_model: str = "claude-sonnet-4-20250514",
    thinking_budget: int = 2048,
    viewer: bool = False,
    **_extra,
) -> dict:
    """Execute tasks using Claude CUA via Anthropic API.

    No GPU needed — inference is via API. Only needs Chrome + xdotool.
    Trajectories are saved for potential distillation training.
    """
    from datetime import datetime, timezone

    from mantis_agent.brain_claude import ClaudeBrain
    from mantis_agent.task_loop import TaskLoopConfig, setup_env, setup_viewer, run_executor_lifecycle

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ── Brain (API-based, no GPU) ──
    brain = ClaudeBrain(
        model=claude_model, max_tokens=4096,
        thinking_budget=thinking_budget, screen_size=(1280, 720),
    )
    brain.load()

    # ── Trajectory saving (unique to Claude) ──
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "claude_cua")
    trajectories_path = f"/data/results/claude_trajectories_{session_name}_{run_id}.jsonl"
    os.makedirs("/data/results", exist_ok=True)

    def save_trajectory(task_id: str, intent: str, result):
        traj_entry = {
            "task_id": task_id, "intent": intent,
            "success": result.success, "steps": result.total_steps,
            "termination_reason": result.termination_reason,
            "trajectory": [
                {
                    "step": s.step, "action": str(s.action),
                    "action_type": s.action.action_type.value,
                    "action_params": s.action.params,
                    "thinking": s.thinking, "feedback": s.feedback,
                    "inference_time": s.inference_time,
                }
                for s in result.trajectory
            ],
        }
        with open(trajectories_path, "a") as f:
            f.write(json.dumps(traj_entry) + "\n")

    # ── Env + viewer ──
    env, proxy_proc = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name,
        settle_time=2.0, display=":99", start_xvfb=True,
    )
    viewer_ctx, viewer_event_bus = setup_viewer(viewer)

    # ── Delegate to shared lifecycle ──
    config = TaskLoopConfig(
        run_id=run_id, session_name=session_name,
        model_name=f"Claude ({claude_model})", results_prefix="claude",
        brain=brain, env=env,
        max_steps=max_steps, frames_per_inference=frames_per_inference,
        viewer_event_bus=viewer_event_bus,
        on_task_complete=save_trajectory,
        volume_commit=vol.commit,
        summary_extras={"estimated_api_cost_usd": 0.0},
    )
    result = run_executor_lifecycle(
        task_suite, config,
        proxy_proc=proxy_proc, viewer_ctx=viewer_ctx, t0=t0,
    )
    print(f"Trajectories saved: {trajectories_path}")
    return result


@app.function(
    image=claude_executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours
    memory=8192,
    cpu=4,
)
def run_claude_cua(task_file_contents: str, claude_model: str = "claude-sonnet-4-20250514", **kwargs) -> dict:
    """Claude CUA executor (no GPU — API-based inference, Chrome + xdotool only)."""
    kwargs.pop("cua_model", None)
    return _run_claude_executor(task_file_contents, claude_model=claude_model, **kwargs)


@app.function(
    gpu="A100-80GB",
    image=planner_image.run_commands(
        "apt-get update && apt-get install -y gnupg curl wget xvfb xdotool scrot",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    ).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20",
    ).add_local_python_source("mantis_agent"),
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours
    memory=65536,
    cpu=16,
)
def run_holo3(task_file_contents: str, **kwargs) -> dict:
    """Holo3-35B-A3B executor (1x A100, llama.cpp GGUF + mmproj)."""
    kwargs.pop("cua_model", None)
    return _run_holo3_executor(task_file_contents, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# D) Local entrypoint — orchestrates planner + executor
# ═══════════════════════════════════════════════════════════════════

# Map model names to executor functions
EXECUTOR_MAP = {
    "evocua-8b": run_cua_1gpu,
    "evocua-32b": run_cua_2gpu,
    "opencua-32b": run_cua_4gpu,
    "opencua-72b": run_cua_8gpu,
    "holo3": run_holo3,
    "gemma4-cua": run_gemma4_cua,
    "claude": run_claude_cua,
}

APP_NAME = "mantis-cua-server"


# ═══════════════════════════════════════════════════════════════════
# E) Parallel extraction — fan-out across N workers
# ═══════════════════════════════════════════════════════════════════

def _make_page_task(original_task: dict, worker_id: int, page: int) -> dict:
    """Create an extraction task for one page of results.

    Each worker processes ALL listings on a single page (~25 per page).
    No pagination needed — one worker, one page, all listings.
    """
    base_url = original_task.get("start_url", "")
    if page > 1:
        # BoatTrader pagination: /page-N/ suffix
        base_url = base_url.rstrip("/") + f"/page-{page}/"

    loop = dict(original_task.get("loop", {}))
    loop["max_iterations"] = 30  # ~25 listings per page + buffer
    loop["max_pages"] = 1        # Stay on assigned page, no pagination

    return {
        "session_name": f"bt_worker_{worker_id}_page_{page}",
        "base_url": base_url,
        "tasks": [
            {
                "task_id": f"extract_p{page}_w{worker_id}",
                "intent": original_task["intent"],
                "loop": loop,
                "start_url": base_url,
            }
        ],
    }


@app.function(
    gpu="A100-80GB",
    image=planner_image.run_commands(
        "apt-get update && apt-get install -y gnupg curl wget xvfb xdotool scrot",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    ).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20",
    ).add_local_python_source("mantis_agent"),
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours per page worker
    memory=65536,
    cpu=16,
    retries=3,  # Auto-retry on preemption/crash
)
def run_gemma4_cua_worker(task_file_contents: str, worker_id: int = 0, **kwargs) -> dict:
    """Single Gemma4-CUA extraction worker with retry logic.

    Retries up to 3 times on failure. Logs failure mode for analysis.
    """
    kwargs.pop("cua_model", None)
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            result = _run_gemma4_cua_executor(task_file_contents, **kwargs)

            # Check if extraction actually produced results
            total = result.get("total", 0)

            if total == 0 and attempt < max_attempts:
                print(f"  Worker {worker_id}: attempt {attempt} produced 0 results, retrying...")
                continue

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"  Worker {worker_id}: attempt {attempt}/{max_attempts} failed — {error_msg[:200]}")

            if attempt == max_attempts:
                # Log failure for analysis
                failure_log = {
                    "worker_id": worker_id,
                    "attempts": attempt,
                    "error": error_msg[:500],
                    "passed": 0, "total": 0, "score": 0.0,
                }
                # Save failure to volume
                import json as _json
                fail_path = f"/data/results/worker_failure_{worker_id}_{int(time.time())}.json"
                os.makedirs("/data/results", exist_ok=True)
                with open(fail_path, "w") as f:
                    _json.dump(failure_log, f, indent=2)
                vol.commit()
                print(f"  Worker {worker_id}: all {max_attempts} attempts failed. Logged to {fail_path}")
                return {"passed": 0, "total": 0, "score": 0.0, "error": error_msg[:200]}

            # Brief pause before retry
            time.sleep(5)

    return {"passed": 0, "total": 0, "score": 0.0}


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
    claude_model: str = "claude-sonnet-4-20250514",
    thinking_budget: int = 2048,
    workers: int = 1,
    viewer: bool = False,
    sub_plan: bool = True,
    learn: bool = False,
    verify: bool = False,
    learn_samples: int = 5,
    micro: str = "",
    max_cost: float = 10.0,
    max_time_minutes: int = 180,
    resume_state: bool = False,
    state_key: str = "",
    graph_learn: bool = False,
    graph_learn_only: bool = False,
):
    """Mantis CUA Server — run plans or task suites on Modal.

    Modes:
      --task-file tasks/boattrader/dynamic.json     (direct execution)
      --plan-file plans/boattrader/full_spec.txt    (Gemma4 preprocesses → execute)
      --learn --task-file tasks/...                 (learning phase: build playbook)
      --verify --task-file tasks/...                (execution with step verification)
      --micro plans/boattrader/extract_only.txt     (micro-intent decompose + execute)
      --graph-learn --micro plan.txt                (probe site + generate dependency graph + execute)
      --graph-learn-only --micro plan.txt           (probe site + generate graph, no execution)

    Models: evocua-8b, evocua-32b, opencua-32b, opencua-72b, holo3, gemma4-cua, claude
    Parallel: --workers 5   (auto fan-out looped tasks across N GPUs)
    Claude options: --claude-model claude-sonnet-4-20250514 --thinking-budget 2048
    Viewer: --viewer   (live web viewer via modal.forward tunnel)
    Learning: --learn --learn-samples 5   (build site playbook from N samples)
    Verification: --verify   (enable step verification during execution)
    Micro: --micro plan.txt   (decompose → micro-intents → execute with checkpoint/reverse)
    Resume: --resume-state --state-key my-run   (reuse externalized micro state across sessions)
    Graph: --graph-learn   (probe + graph + compile + execute) --graph-learn-only (no execution)
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

    # Mode 1: Plan file → Gemma4 preprocessing → executor
    if plan_file:
        print(f"  Plan:    {plan_file}")
        print(f"  Mode:    Gemma4 → {cua_config['name']}")

        with open(plan_file) as f:
            plan_text = f.read()

        if not session_name:
            session_name = os.path.splitext(os.path.basename(plan_file))[0]

        planner_url = f"https://{APP_NAME}--gemma4-planner.modal.run"
        print(f"  Planner: {planner_url}")

        import requests
        try:
            r = requests.get(f"{planner_url}/v1/models", timeout=10)
            if r.status_code != 200:
                print("  WARNING: Planner not responding. Deploy first: modal deploy deploy/modal/modal_cua_server.py")
                sys.exit(1)
            print("  Planner: OK")
        except Exception:
            print(f"  ERROR: Cannot reach planner at {planner_url}")
            print("  Deploy first: uv run modal deploy deploy/modal/modal_cua_server.py")
            sys.exit(1)

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
        from mantis_agent.brain_llamacpp import LlamaCppBrain
        from mantis_agent.gym.plan_optimizer import optimize_plan

        planner_brain = LlamaCppBrain(
            base_url=f"{planner_url}/v1", model="model",
            max_tokens=4096, temperature=0.0,
        )

        print("  Preprocessing plan with Gemma4...")
        task_suite = optimize_plan(
            plan_text=plan_text, inputs=plan_inputs,
            session_name=session_name, max_listings=max_listings,
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

    # Mode 3: Micro-intent decompose + execute
    elif micro:
        print(f"  Plan:    {micro}")

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
        from mantis_agent.plan_decomposer import PlanDecomposer, MicroPlan

        # ── Graph learning mode: probe site + generate dependency graph ──
        if graph_learn or graph_learn_only:
            print(f"  Mode:    Graph Learning → {cua_config['name']}")
            from mantis_agent.graph import GraphLearner, GraphCompiler, GraphStore

            # Read plan text for objective parsing
            with open(micro) as f:
                plan_text = f.read()

            # Extract start URL from plan text
            import re as _re
            start_url = ""
            url_match = _re.search(r"https?://[^\s]+", plan_text)
            if url_match:
                start_url = url_match.group(0).rstrip("/.,;:)")

            learner = GraphLearner(store=GraphStore())
            graph = learner.learn(
                objective_text=plan_text,
                start_url=start_url,
                n_samples=0,  # No sample execution in local entrypoint (GPU not available yet)
                force_relearn=graph_learn_only,
            )
            print(f"\n  Graph: {len(graph.phases)} phases, {len(graph.edges)} edges")
            print(f"  Topo:  {graph.topological_order()}")

            if graph_learn_only:
                print(f"\n  Graph saved. Use --graph-learn (without --only) to also execute.")
                print(json.dumps({"mode": "graph_learn", "phases": len(graph.phases), "domain": graph.domain}, indent=2))
                return

            # Compile graph to MicroPlan
            compiler = GraphCompiler()
            micro_plan = compiler.compile(graph)
            print(f"\n  Compiled: {len(micro_plan.steps)} micro-intents from graph")

        elif micro.endswith(".json"):
            print(f"  Mode:    Micro-Intent → {cua_config['name']}")
            # Load pre-built micro-plan JSON directly (no decomposition)
            with open(micro) as f:
                raw_steps = json.load(f)
            micro_plan = MicroPlan(domain="direct_json")
            for s in raw_steps:
                micro_plan.steps.append(PlanDecomposer._build_intent(s))
        else:
            print(f"  Mode:    Micro-Intent → {cua_config['name']}")
            # Decompose plain text plan into micro-intents (Claude Sonnet, cached)
            decomposer = PlanDecomposer()
            micro_plan = decomposer.decompose(micro)

        print(f"  Steps:   {len(micro_plan.steps)} micro-intents")
        print(micro_plan.summary())

        # Validate and enhance the plan before execution
        from mantis_agent.graph.plan_validator import PlanValidator
        objective_for_validation = None
        if graph_learn or graph_learn_only:
            objective_for_validation = graph.objective
        validator = PlanValidator()
        issues = validator.validate(micro_plan, objective=objective_for_validation)
        if issues:
            for issue in issues:
                tag = "ERROR" if issue.severity == "error" else "WARN"
                fix = f" (auto-fix: {issue.auto_fix})" if issue.auto_fix else ""
                print(f"  [{tag}] {issue.code}: {issue.message}{fix}")
            micro_plan = validator.enhance(micro_plan, objective=objective_for_validation)
            print(f"  Enhanced plan: {len(micro_plan.steps)} steps")
            print(micro_plan.summary())

        # Embed objective when available (graph-learn path) for schema-driven extraction
        objective_dict = None
        if graph_learn or graph_learn_only:
            objective_dict = graph.objective.to_dict()

        steps_dicts = micro_plan_steps_to_dicts(micro_plan.steps)
        task_suite = build_micro_suite(
            steps_dicts,
            micro_plan.domain,
            max_cost=max_cost,
            max_time_minutes=max_time_minutes,
            resume_state=resume_state,
            state_key=state_key,
            objective=objective_dict,
        )
        resolved_state_key = task_suite["_state_key"]

        print(f"  State:   {resolved_state_key}")
        print(f"  Resume:  {'on' if resume_state else 'off'}")

        task_file_contents = json.dumps(task_suite)

    else:
        print("ERROR: Provide --plan-file, --task-file, or --micro")
        sys.exit(1)

    # ── Learning mode: build playbook with step verification ─────
    if learn:
        print(f"\n  ═══ LEARNING MODE: {learn_samples} samples ═══")
        print("  Building site playbook with step verification...")

        # Pass learn flag + samples to the executor
        task_suite_obj = json.loads(task_file_contents)
        task_suite_obj["_learn"] = True
        task_suite_obj["_learn_samples"] = learn_samples
        task_file_contents = json.dumps(task_suite_obj)

    if verify:
        print("\n  ═══ VERIFICATION MODE ═══")
        print("  Step verification enabled for critical actions...")
        task_suite_obj = json.loads(task_file_contents)
        task_suite_obj["_verify"] = True
        task_file_contents = json.dumps(task_suite_obj)

    # ── Auto-parallelize looped tasks ──────────────────────────────
    task_suite = json.loads(task_file_contents)
    tasks = task_suite.get("tasks", [])
    loop_tasks = [t for t in tasks if t.get("loop")]
    non_loop_tasks = [t for t in tasks if not t.get("loop")]

    # Auto-detect: if workers > 1 and there are looped tasks, fan out
    # Works with any model — routes to the correct executor per model type
    if workers > 1 and loop_tasks:
        print(f"\n  ═══ PARALLEL MODE: {workers} concurrent workers ═══")

        for loop_task in loop_tasks:
            max_pages = loop_task["loop"].get("max_pages", 5)

            print(f"  Task: {loop_task['task_id']}")
            print(f"    Pages: {max_pages} | Workers: {workers}")
            print("    Strategy: 1 worker = 1 page, dynamic queue")

            # Page queue — workers pull from this
            import queue
            import threading

            page_queue = queue.Queue()
            for p in range(1, max_pages + 1):
                page_queue.put(p)

            total_viable = 0
            total_scanned = 0
            results_lock = threading.Lock()
            worker_results = []

            def process_worker(worker_id: int):
                """Worker loop: grab page → process → grab next until queue empty."""
                nonlocal total_viable, total_scanned
                while True:
                    try:
                        page = page_queue.get_nowait()
                    except queue.Empty:
                        break

                    print(f"    Worker {worker_id}: starting page {page}")
                    page_task = _make_page_task(loop_task, worker_id, page)
                    page_contents = json.dumps(page_task)

                    try:
                        result = run_gemma4_cua_worker.remote(
                            task_file_contents=page_contents,
                            worker_id=worker_id,
                            max_steps=max_steps,
                        )
                        passed = result.get("passed", 0)
                        total = result.get("total", 0)
                        with results_lock:
                            total_viable += passed
                            total_scanned += total
                            worker_results.append({
                                "worker": worker_id, "page": page,
                                "passed": passed, "total": total,
                            })
                        print(f"    Worker {worker_id}: page {page} done — {passed}/{total} viable")
                    except Exception as e:
                        print(f"    Worker {worker_id}: page {page} ERROR — {e}")

            # Launch worker pool — each grabs pages from the queue
            # Use .spawn() for parallel Modal execution, then collect
            # Simpler: spawn all pages upfront, workers=min(workers, pages)
            active_workers = min(workers, max_pages)
            page_handles = []

            # Route to the right worker function based on model
            worker_fn_map = {
                "gemma4-cua": run_gemma4_cua_worker,
                "evocua-8b": run_cua_1gpu,
                "evocua-32b": run_cua_2gpu,
                "opencua-32b": run_cua_4gpu,
                "holo3": run_holo3,
            }
            worker_fn = worker_fn_map.get(model, run_gemma4_cua_worker)

            for page in range(1, max_pages + 1):
                w = (page - 1) % active_workers
                page_task = _make_page_task(loop_task, worker_id=w, page=page)
                page_contents = json.dumps(page_task)
                print(f"    → Page {page} → Worker {w}")

                spawn_kwargs = {
                    "task_file_contents": page_contents,
                    "max_steps": max_steps,
                }
                if model == "gemma4-cua":
                    spawn_kwargs["worker_id"] = w
                    handle = run_gemma4_cua_worker.spawn(**spawn_kwargs)
                else:
                    spawn_kwargs["cua_model"] = model
                    handle = worker_fn.spawn(**spawn_kwargs)
                page_handles.append((page, w, handle))

            # Collect results as workers complete
            print(f"\n  Waiting for {len(page_handles)} page workers...")
            total_viable = 0
            total_scanned = 0

            for page, w, handle in page_handles:
                try:
                    result = handle.get()
                    passed = result.get("passed", 0)
                    total = result.get("total", 0)
                    total_viable += passed
                    total_scanned += total
                    print(f"    Page {page} (W{w}): {passed}/{total} viable")
                except Exception as e:
                    print(f"    Page {page} (W{w}): ERROR — {e}")

            # Summary
            print("\n  ═══ PARALLEL RESULTS ═══")
            print(f"  Pages:     {max_pages}")
            print(f"  Workers:   {active_workers}")
            print(f"  Scanned:   {total_scanned}")
            print(f"  Viable:    {total_viable}")
            print(f"  Hit rate:  {total_viable/max(total_scanned,1)*100:.0f}%")

        # Run non-loop tasks sequentially (login, lead entry)
        if non_loop_tasks:
            print(f"\n  Running {len(non_loop_tasks)} sequential tasks...")
            seq_suite = dict(task_suite)
            seq_suite["tasks"] = non_loop_tasks
            seq_contents = json.dumps(seq_suite)

            executor_fn = EXECUTOR_MAP.get(model, run_cua_1gpu)
            kwargs = {
                "task_file_contents": seq_contents,
                "max_steps": max_steps, "max_retries": max_retries,
            }
            if model == "claude":
                kwargs["claude_model"] = claude_model
            result = executor_fn.remote(**kwargs)
            print(f"  Sequential: {json.dumps(result, indent=2)}")

        return

    # ── Single worker (default) ──────────────────────────────────
    executor_fn = EXECUTOR_MAP.get(model, run_cua_1gpu)
    gpu_desc = f"{cua_config['tp']}× A100" if cua_config['tp'] > 0 else "no GPU (API)"
    print(f"\n  Launching {cua_config['name']} on Modal ({gpu_desc})...")

    kwargs = {
        "task_file_contents": task_file_contents,
        "plan_inputs": plan_inputs,
        "max_steps": max_steps,
        "max_retries": max_retries,
        "viewer": viewer,
        "sub_plan": sub_plan,
    }
    if model == "claude":
        kwargs["claude_model"] = claude_model
        kwargs["thinking_budget"] = thinking_budget
    else:
        kwargs["cua_model"] = model

    result = executor_fn.remote(**kwargs)

    print(f"\nResult: {json.dumps(result, indent=2)}")
