"""Mantis CUA Server — fully remote Gemma4 + EvoCUA on Modal.

Single `modal deploy` gives you:
  - Gemma4 planner (T4 GPU, llama.cpp, persistent web server)
  - EvoCUA executors (A100 GPUs, vLLM, per-run)

Usage:
    # Deploy the planner (stays running, warm for 10 min):
    uv run modal deploy deploy/modal/modal_cua_server.py

    # Run from a free-text plan (Gemma4 preprocesses → EvoCUA executes):
    uv run modal run deploy/modal/modal_cua_server.py \
        --plan-file plans/example/full_spec.txt \
        --model evocua-8b \
        --inputs "admin_password=SelfService38#,zip_code=33101,search_radius=35"

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
    persist_run_artifacts,
    plan_signature_from_steps,
    safe_state_key,
    save_result_json,
    start_local_proxy,
    wait_for_openai_server,
)

# ── App + shared resources ──────────────────────────────────────────

app = modal.App("mantis-cua-server")
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)


# #346: ``add_local_python_source`` only copies ``.py`` files. Every brain
# imports ``mantis_agent.prompts``, which reads ``prompts/files/*.txt`` at
# module-init time — missing those data files crashes the container with
# ``FileNotFoundError`` before the executor runs. Bundle them explicitly
# wherever we mount ``mantis_agent``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPTS_FILES_LOCAL = os.path.join(_REPO_ROOT, "src", "mantis_agent", "prompts", "files")
_PROMPTS_FILES_REMOTE = "/root/mantis_agent/prompts/files"

# #stealth-parity: WebGL spoofing Chrome extension (ported from
# the parity-reference browser stack — see internal docs for the
# upstream source). Loaded via ``--load-extension``
# on Chrome launch in ``xdotool_env._start_browser``. The extension's
# content script runs at ``document_start`` in MAIN world across
# ``<all_urls>`` and all frames — hooks WebGLRenderingContext at the
# C++ binding level (more thorough than our ``addScriptToEvaluateOnNew
# Document`` JS patches which run page-context only and miss iframe
# probes). Returns a realistic Intel UHD Graphics renderer string
# that matches a real Linux x86_64 user.
_WEBGL_SPOOF_LOCAL = os.path.join(_REPO_ROOT, "deploy", "modal", "chrome_extensions", "webgl_spoof")
_WEBGL_SPOOF_REMOTE = "/opt/chrome-extensions/webgl-spoof"

# #stealth-parity: Fonts the parity-reference browser ships but we didn't.
# Sparse font set is a strong bot tell — CF/Turnstile fingerprints
# ``document.fonts.check('italic 9pt Arial')`` etc. for ~30 canary
# fonts; only Linux servers have a Liberation-only set.
_STEALTH_APT_FONTS_AND_LOCALE = [
    "fonts-liberation",
    "fonts-dejavu-core",
    "fonts-noto-color-emoji",
    "locales",
    # NB: tzdata is intentionally NOT here. Both nvidia/cuda:12.4.0-
    # devel-ubuntu22.04 and ubuntu:22.04 ship tzdata pre-installed
    # in the base layer, so /usr/share/zoneinfo/* + the
    # ``ln -sf .../America/New_York /etc/localtime`` step in
    # run_commands already give us proper TZ behavior. Reinstalling
    # tzdata via apt_install fires its postinst "Geographic area"
    # prompt — DEBIAN_FRONTEND=noninteractive alone doesn't suppress
    # it without also pre-seeding /etc/timezone, which would mean
    # running shell commands BEFORE apt_install (not supported in
    # the Image builder chain order). Skipping the reinstall sidesteps
    # the prompt entirely.
]

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
    "fara": {
        "repo": "microsoft/Fara-7B",  # Qwen2.5-VL base, native vLLM, MIT-licensed
        "name": "Fara-7B",
        "tp": 1,  # 7B bf16 ≈ 14 GB; comfortable on single A100-40GB / L40S
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
planner_base_image = (
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
planner_image = (
    planner_base_image
    .add_local_python_source("mantis_agent")
    .add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE)
)

# EvoCUA executor: vLLM + real Chrome + Xvfb + xdotool (zero automation fingerprints)
executor_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget", "gnupg",
                 "xvfb", "xdotool", "xclip", "scrot",
                 # #stealth-parity: fonts + locales (tzdata comes
                 # from the base image) so the browser fingerprint
                 # matches a typical US Linux desktop (the parity
                 # reference has these; we didn't).
                 *_STEALTH_APT_FONTS_AND_LOCALE)
    .run_commands(
        # Install real Google Chrome (not Chromium)
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
        # #stealth-parity: generate en_US locale + link America/New_York
        # timezone so Intl.DateTimeFormat().resolvedOptions().timeZone
        # reports 'America/New_York' (matches the US proxy IP) instead
        # of the Modal container default 'Etc/UTC'.
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime",
    )
    .env({
        # #stealth-parity: process-wide locale + TZ for child Chrome.
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
    })
    .pip_install(
        "vllm>=0.12.0",
        "openai", "requests", "pillow", "mss",
        "huggingface-hub", "transformers", "torch",
        "fastapi>=0.100", "uvicorn>=0.20", "websocket-client",
        # #509: per-run Augur DebugSession bundle + optional live streaming.
        # Pip-installed unconditionally so the wedge in
        # src/mantis_agent/observability/augur.py activates on every
        # executor container. Run-time gate via MANTIS_AUGUR_DISABLED.
        # 0.1.2+ fires an immediate session-opened heartbeat so the
        # workspace's connection badge updates the moment the SDK is
        # wired up, before the first step lands.
        # Must match pyproject.toml (``augur-sdk>=0.6.0,<0.7``). 0.2.x
        # added ``branch_context`` to ``DebugSession.__init__``; 0.4.0
        # (mercurialsolo/augur-sdk#38) added
        # ``DebugSession.open_orchestrator(...)``; 0.6.0 (#680) adds
        # ``task_spec`` / ``group_id`` / ``set_loop_detected`` /
        # ``record_subgoal_completion`` and the ``set_score`` kwargs
        # ``should_stop`` / ``uncertainty``. Stale <0.6 pins silently
        # drop the new fields — adapter swallows the TypeError and the
        # bundle ships without RL-training metadata.
        "augur-sdk>=0.6.0,<0.7",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE)
    # #stealth-parity: ship the WebGL spoof Chrome extension. The
    # loader in xdotool_env._start_browser checks os.path.isdir on
    # the remote path before appending --load-extension, so this
    # mount is what flips it on in production.
    .add_local_dir(_WEBGL_SPOOF_LOCAL, remote_path=_WEBGL_SPOOF_REMOTE)
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
    frames_per_inference: int = 2,
    viewer: bool = False,
    profile_dir: str = "",
    **_extra,
) -> dict:
    """Shared executor logic for all vLLM GPU tiers.

    #435: ``frames_per_inference`` default lowered from 5 to 2 per the
    cua_notes.md "Cost-accuracy sweet spot" guidance — keep the last
    1-3 screenshots, images dominate token cost, older frames are
    mostly redundant once you have the action log. Holo3 keeps its
    own 1-frame default (see ``_run_holo3_executor``); the Claude
    executor follows the same default. Operators can override via
    the kwarg.
    """
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
    # Fara emits OpenAI-format tool calls (computer_use). vLLM 0.11+ requires
    # both flags to honour ``tool_choice="auto"``; without them every request
    # 400s and the brain falls back to WAITs. ``hermes`` is the Qwen-family
    # parser shipped in vLLM and matches Fara's Qwen2.5-VL base.
    vllm_extra: list[str] = []
    if cua_model == "fara":
        vllm_extra = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    vllm_proc = _start_vllm(model_dir, port=8000, tp=tp, extra_args=vllm_extra)

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
    elif cua_model == "fara":
        from mantis_agent.brain_fara import FaraBrain
        brain = FaraBrain(
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
    env, proxy_proc, proxy_diag = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name, settle_time=2.0,
        proxy_city=str(task_suite.get("_proxy_city") or ""),
        proxy_state=str(task_suite.get("_proxy_state") or ""),
        proxy_provider=str(task_suite.get("_proxy_provider") or ""),
        proxy_country=str(task_suite.get("_proxy_country") or ""),
        proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
        extra_http_headers=task_suite.get("_browser_extra_headers") or None,
        profile_dir=profile_dir,
    )
    viewer_ctx, viewer_event_bus, _viewer_url = setup_viewer(viewer, proxy_diag=proxy_diag)

    # ── Delegate to shared lifecycle ──
    config = TaskLoopConfig(
        run_id=run_id, session_name=session_name,
        model_name=cua_config["name"],
        results_prefix={"holo3": "holo3", "fara": "fara"}.get(cua_model, "cua"),
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
    # #416: when set, the executor surfaces the live-viewer URL into
    # the API-side ``status.json`` so a polling client gets a hot-
    # link mid-run. Both values come from ``Function.spawn``'s
    # kwargs at the /v1/predict handler; ``viewer`` flag alone is
    # the legacy CLI path which prints the URL to stdout instead.
    api_run_id: str | None = None,
    api_tenant_id: str | None = None,
    # #341 follow-up: per-tenant, per-profile Chrome user-data-dir.
    # The /v1/predict handler computes this from ``profile_id`` so
    # cookies / localStorage / IndexedDB don't leak across runs that
    # use different profiles on the same warm container.
    profile_dir: str = "",
    **_extra,
) -> dict:
    """Execute tasks using Holo3-35B-A3B via llama.cpp (GGUF on 1x A100).

    Like Gemma4-CUA: llama-server + GGUF + mmproj for vision.
    vLLM doesn't support qwen3_5_moe yet.
    """
    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # ── #673: fanout dispatch gate (HTTP path) ──
    # Before booting the Holo3 GGUF + llama-server (~30s cold), check
    # whether this submission is in orchestrator mode:
    #   * suite carries a ``parallelizable_url_collect`` loop AND
    #   * caller opted in via ``_fanout_phase1_workers > 1`` AND
    #   * we're NOT already a Phase-1/Phase-2 sub-worker
    #     (``_fanout_phase`` set by prepare_phase1_suite /
    #     prepare_phase2_suites — guards against recursive spawning)
    # If all three hold, dispatch fanout via the shared helper in
    # ``gym/fanout_runner.run_fanout_dispatch`` and return the
    # aggregate envelope WITHOUT booting llama-server. The Phase-1/
    # Phase-2 child workers spawned by the helper re-enter this
    # function with ``_fanout_phase`` set; they fall through to the
    # single-runner path below.
    #
    # Sub-workers and unopted single-runner submissions skip this
    # block entirely and continue to the legacy boot path.
    _suite_preview = json.loads(task_file_contents)
    if not _suite_preview.get("_fanout_phase"):
        from mantis_agent.gym.fanout_runner import (
            find_url_collect_group,
            run_fanout_dispatch,
        )
        _url_collect_eligible = find_url_collect_group(_suite_preview)
        try:
            _opt_in_workers = int(
                _suite_preview.get("_fanout_phase1_workers") or 1
            )
        except (TypeError, ValueError):
            _opt_in_workers = 1
        if _url_collect_eligible is not None and _opt_in_workers > 1:
            import uuid as _uuid_for_fanout
            _fanout_parent = (
                f"fanout-{_suite_preview.get('_plan_signature', 'unknown')[:12]}-"
                f"{_uuid_for_fanout.uuid4().hex[:8]}"
            )
            _suite_preview["_fanout_parent_run_id"] = _fanout_parent
            print(
                f"\n  ═══ #673: HTTP-path fanout orchestrator "
                f"(parent_run_id={_fanout_parent}, "
                f"phase1_workers={_opt_in_workers}) ═══"
            )
            _result = run_fanout_dispatch(
                _suite_preview,
                executor_fn=run_holo3,
                model="holo3",
                claude_model="",
                max_steps=max_steps,
                workers=max(_opt_in_workers, 1),
                fanout_parent_run_id=_fanout_parent,
                shared_seen_printer=_print_shared_seen_metrics,
            )
            if _result is not None:
                return _result
            print(
                "  [#673] Fanout fell through (no URLs harvested); "
                "continuing to single-runner sequential path."
            )

    from mantis_agent.brain_holo3 import Holo3Brain
    from mantis_agent.task_loop import setup_env, setup_viewer

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
    claude_fallback_brain = None
    claude_fallback_disabled = bool(task_suite.get("_claude_fallback_disabled", False))
    if not claude_fallback_disabled:
        try:
            from mantis_agent.brain_claude import ClaudeBrain
            claude_fallback_brain = ClaudeBrain(
                model=str(task_suite.get("_claude_fallback_model") or "claude-sonnet-4-6"),
                thinking_budget=2048,
                screen_size=(1280, 720),
            )
            claude_fallback_brain.load()
            print("Claude fallback enabled for failed Holo3 task sections")
        except Exception as exc:
            print(f"Claude fallback unavailable for failed Holo3 sections: {exc}")

    # Claude Sonnet 4.6 grounding for click targeting (was Opus —
    # 5× cheaper for the same different-model-from-Holo3 property).
    from mantis_agent.grounding import ClaudeGrounding
    grounding = ClaudeGrounding()

    # Env + viewer via shared helpers
    env, proxy_proc, proxy_diag = setup_env(
        base_url=base_url, run_id=run_id, session_name=session_name,
        settle_time=4.0,  # Holo3 needs longer settle — sees black screen with 2s
        proxy_city=str(task_suite.get("_proxy_city") or ""),
        proxy_state=str(task_suite.get("_proxy_state") or ""),
        proxy_provider=str(task_suite.get("_proxy_provider") or ""),
        proxy_country=str(task_suite.get("_proxy_country") or ""),
        proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
        extra_http_headers=task_suite.get("_browser_extra_headers") or None,
        profile_dir=profile_dir,
    )
    from mantis_agent.extraction import ClaudeExtractor, ExtractionSchema
    schema = None
    objective_data = task_suite.get("_objective")
    if objective_data:
        from mantis_agent.graph.objective import ObjectiveSpec
        objective = ObjectiveSpec.from_dict(objective_data)
        schema = ExtractionSchema.from_objective(objective)
    # Per-plan extractor model override — A/B-able from the submit
    # payload via `build_micro_suite(extractor_model=...)`. Empty
    # falls through to ClaudeExtractor's default. Print so operators
    # see which side of the A/B each container is on.
    extractor_model = str(task_suite.get("_extractor_model", "") or "")
    extractor_kwargs: dict = {"schema": schema}
    if extractor_model:
        extractor_kwargs["model"] = extractor_model
        print(f"  Extractor: model={extractor_model} (A/B override)")
    extractor = ClaudeExtractor(**extractor_kwargs)
    # #416 follow-up: the live-viewer's screen-capture thread reads
    # ``os.environ["DISPLAY"]`` (via mss) but ``setup_env`` doesn't
    # propagate that into the executor process's environ — the env
    # only stashes it on its own ``self._env`` for subprocess use.
    # Bring up Xvfb explicitly and set the process-wide DISPLAY so
    # the capture thread can attach. Skip when viewer isn't requested
    # — Chrome's launch path (inside ``env.reset()``) handles its own
    # subprocess env.
    if viewer:
        try:
            display = env.ensure_display_ready()
            if display:
                os.environ["DISPLAY"] = display
        except Exception as exc:  # noqa: BLE001 — best-effort
            print(f"  WARNING: ensure_display_ready before viewer failed: {exc}")
    viewer_ctx, viewer_event_bus, viewer_url = setup_viewer(
        viewer,
        proxy_diag=proxy_diag,
        api_run_id=api_run_id or "",
        api_tenant_id=api_tenant_id or "",
    )
    # #416: persist the tunnel URL so a caller polling ``action=status``
    # gets a hot-link to the live screen. We write to a side-channel
    # ``viewer.json`` rather than merging into the API-owned
    # ``status.json``: the executor races the API's initial
    # ``_write_status`` (spawn returns before the queued-status commit
    # lands) and ``vol.reload()`` from inside the executor doesn't
    # reliably bridge that gap. ``_do_action`` reads both files and
    # merges the URL onto the response. Only fires when the /v1/predict
    # handler forwarded the API's run_id + tenant_id alongside
    # ``viewer=True``; the legacy CLI path (no api_run_id) keeps the
    # old behaviour — URL is printed to stdout by ``modal_viewer``.
    if viewer_url and api_run_id and api_tenant_id:
        try:
            _write_viewer_url(api_tenant_id, api_run_id, viewer_url)
            print(f"  live-viewer URL written to viewer.json: {viewer_url}")
        except Exception as exc:  # noqa: BLE001 — viewer is best-effort
            print(f"  WARNING: failed to surface live-viewer URL into viewer.json: {exc}")

    # #541: wire the external-pause sentinel path so the API
    # container's ``action=pause`` can signal this executor, and
    # the runner's auto-pause-on-captcha can write its own
    # sentinel. ``wait_while_paused`` keeps the executor alive
    # (Chrome + viewer up) until ``action=resume`` clears the
    # sentinel.
    if api_run_id and api_tenant_id:
        try:
            from mantis_agent.gym import external_pause
            external_pause.init_paths(
                str(_run_dir(api_tenant_id, api_run_id) / "pause_request.json"),
                # vol.reload invalidates the executor's volume cache
                # so we see API-container sentinel deletes (action=resume).
                # Without this the executor's stat keeps returning
                # exists=True even after the API cleared it; runner
                # loops in wait_while_paused for 30 min until timeout
                # while the viewer button stays stuck on "Resume".
                reload_cb=vol.reload,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: external_pause init failed: {exc}")

    # ── Micro-intent mode: run MicroPlanRunner ──
    micro_plan_data = task_suite.get("_micro_plan")
    if micro_plan_data:
        from mantis_agent.gym.checkpoint import PauseRequested, PauseState
        from mantis_agent.gym.micro_runner import MicroPlanRunner
        from mantis_agent.plan_decomposer import MicroIntent, MicroPlan

        micro_plan = MicroPlan(domain=session_name)
        for s in micro_plan_data:
            micro_plan.steps.append(MicroIntent(**s))

        # If objective is available, run site probe to enhance the plan
        # This runs INSIDE the container where env/browser is available
        objective_data = task_suite.get("_objective")
        if objective_data and env:
            try:
                from mantis_agent.graph.objective import ObjectiveSpec as _OS
                from mantis_agent.graph.probe import SiteProber
                from mantis_agent.graph.enhancer import PlanEnhancer
                from mantis_agent.graph import GraphCompiler, PlanValidator
                from mantis_agent.graph.graph import WorkflowGraph
                from mantis_agent.verification.playbook import Playbook

                obj = _OS.from_dict(objective_data)
                if obj.start_url:
                    print("\n  === SITE PROBE (inside container) ===")
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
        profile_id = task_suite.get("_profile_id", state_key)
        workflow_id = task_suite.get("_workflow_id", state_key)
        checkpoint_path = task_suite.get("_checkpoint_path") or f"/data/checkpoints/micro_{session_name}_{run_id}.json"
        plan_signature = task_suite.get("_plan_signature", "")
        # #638 axis 2 follow-up: stable human-readable plan identifier
        # (e.g. ``boattrader_scrape_urlnav``). Set by the orchestrator
        # at suite-build time from the source file stem. The Augur
        # adapter reads ``runner.plan_name`` and emits it as a tag so
        # the Runs list can group different runs of the same plan.
        plan_name = task_suite.get("_plan_name", "")

        print(f"\n  === MICRO-INTENT MODE ({len(micro_plan.steps)} steps) ===")
        print(micro_plan.summary())
        if profile_id or workflow_id:
            print(f"  Profile:    {profile_id}")
            print(f"  Workflow:   {workflow_id}")
            print(f"  Resume:     {'on' if resume_state else 'off'}")
            print(f"  Checkpoint: {checkpoint_path}")

        # #649: mint a per-session Augur run_id that's distinct from
        # the (stable) ``workflow_id``. Two callers may invoke the same
        # workflow_id (intentional, for resume / profile-lock reuse);
        # each invocation gets its own Augur session id so the Runs
        # list doesn't pile overlapping rows under the same identifier.
        # The workflow_id is preserved as a prefix for searchability
        # and as a separate Augur tag for cross-run grouping.
        import uuid as _uuid_for_augur
        augur_run_id = (
            f"{workflow_id}-{_uuid_for_augur.uuid4().hex[:8]}"
            if workflow_id else _uuid_for_augur.uuid4().hex[:12]
        )

        # Stamp the API-run-id → augur-run-id mapping into a side-channel
        # file so the lifecycle endpoint can surface it to operators.
        # Best-effort — telemetry must never block the run.
        try:
            if api_run_id and api_tenant_id:
                _write_augur_metadata(api_tenant_id, api_run_id, augur_run_id)
        except Exception as _augur_exc:  # noqa: BLE001
            print(f"  WARNING: augur metadata write failed: {_augur_exc}")

        # #657 PR 2: read the per-domain ``SiteConfig`` from the suite
        # (written by ``build_micro_suite`` after resolving
        # ``MicroPlan.domain`` → DomainProfile). Passing it explicitly
        # here means the HTTP path stops silently inheriting the
        # runner's ``default_boattrader()`` fallback for any plan that
        # doesn't have a profile entry. ``None`` → runner picks its
        # default (today still boattrader; flipped to generic in
        # #657 PR 3).
        site_config_for_runner = None
        _sc_dict = task_suite.get("_site_config")
        if _sc_dict:
            try:
                from mantis_agent.site_config import SiteConfig as _SiteConfig
                site_config_for_runner = _SiteConfig.from_dict(_sc_dict)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: _site_config deserialization failed: {exc}")

        micro_runner = MicroPlanRunner(
            brain=brain, env=env,
            grounding=grounding, extractor=extractor,
            on_step=viewer_event_bus.emit if viewer_event_bus else None,
            checkpoint_path=checkpoint_path,
            run_key=workflow_id or session_name,
            session_name=session_name,
            plan_signature=plan_signature,
            augur_run_id=augur_run_id,
            resume_state=resume_state,
            on_checkpoint=vol.commit,
            max_cost=task_suite.get("_max_cost", 10.0),
            max_time_minutes=task_suite.get("_max_time_minutes", 180),
            # Plan-evolution Phase 2 (#706) attrs are stamped AFTER
            # construction (the runner doesn't accept them as kwargs).
            # See the block below.
            # #657 PR 2: per-domain SiteConfig resolved from suite.
            # None means "use runner default" (preserves legacy
            # behaviour for callers that pre-build a suite without
            # going through ``build_micro_suite``).
            site_config=site_config_for_runner,
            # #560: ``None`` (key absent) → runner falls back to
            # ``Holo3StepHandler.DEFAULT_BRAIN_BUDGET_CAPS``.
            brain_budgets=task_suite.get("_brain_budgets"),
            # #570: ``None`` → runner falls back to
            # ``MANTIS_PAUSE_ON_CAPTCHA`` env (default on).
            pause_on_captcha=task_suite.get("_pause_on_captcha"),
            # #561: ``None`` → no ceiling, each settle uses its own max.
            settle_ceiling_seconds=task_suite.get("_settle_ceiling_seconds"),
            # #567: ``None`` → fall back to DEFAULT_MAX_RECOVERIES_PER_*.
            max_recoveries_per_run=task_suite.get("_max_recoveries_per_run"),
            max_recoveries_per_step=task_suite.get("_max_recoveries_per_step"),
        )

        # Claude cost meter (#675 A/B follow-up). Bind a fresh meter
        # per run + stamp api_run_id / api_tenant_id on the runner so
        # finalize_to_disk can write /data/runs/<tenant>/<run_id>/
        # claude_cost_by_path.json at terminal. The api_run_id is the
        # one the API container assigned (the canonical run identifier
        # operators see in status payloads); falls back to workflow_id
        # for callers that don't propagate it.
        try:
            from mantis_agent.observability.claude_cost_meter import (
                ClaudeCostMeter, set_current_meter,
            )
            _cost_meter = ClaudeCostMeter()
            set_current_meter(_cost_meter)
            micro_runner._cost_meter = _cost_meter
            micro_runner._api_run_id = str(
                api_run_id
                or task_suite.get("_state_key", "")
                or task_suite.get("_workflow_id", "")
                or ""
            )
            micro_runner._api_tenant_id = str(
                api_tenant_id
                or task_suite.get("_profile_id", "")
                or "default"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: cost meter init failed: {exc}")

        # Cross-run dedup: seed the scanner's seen_urls with any
        # ``_skip_urls`` the suite carries. Callers (e.g. the 6-zip
        # parallel script) pre-fetch prior runs' leads.csv files,
        # union the URL column, and pass the result here so a
        # re-run of the same zip doesn't re-extract listings it
        # already processed (and doesn't pay the ~$0.20 / detail-page
        # claude_extract cost on a confirmed duplicate).
        # No-op when the field is absent.
        skip_urls = task_suite.get("_skip_urls") or []
        if isinstance(skip_urls, list) and skip_urls:
            try:
                for u in skip_urls:
                    if isinstance(u, str) and u:
                        micro_runner.scanner.seen_urls.add(u)
                print(
                    f"  [skip-urls] seeded scanner with {len(skip_urls)} "
                    f"URLs from prior runs",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: skip_urls seed failed: {exc}")

        # Plan-evolution Phase 2 (#706): stamp the plan_hash + scope_id
        # the recovery layer needs to record candidates and the
        # micro_runner needs at terminal to finalize promotion gates.
        # Empty strings disable persistence (no-op for legacy callers
        # that don't pass plan_hash through build_micro_suite).
        micro_runner._plan_hash = str(task_suite.get("_plan_hash", "") or "")
        micro_runner._workflow_id = str(
            task_suite.get("_plan_evolution_scope_id", "")
            or task_suite.get("_profile_id", "")
            or task_suite.get("_workflow_id", "")
            or ""
        )
        micro_runner._applied_plan_rewrites = []
        if micro_runner._plan_hash and micro_runner._workflow_id:
            print(
                f"  Plan-evolution: scope={micro_runner._workflow_id} "
                f"plan_hash={micro_runner._plan_hash}"
            )

        # Trajectory hint memory (#643 / #670). Tenant-scoped disk store
        # at /data/hints/<tenant_id>/<plan_signature>.json. Producer
        # side: ``run_executor._record_hint_memory`` reads ``_hint_store``
        # off the runner after every step success. Consumer side: the
        # holo3 step handler reads ``preferred_target_description`` from
        # ``step.hints`` (stamped by ``apply_hint_overlay`` pre-flight
        # below).
        #
        # Tenant id is the profile_id (which already carries the
        # ``<tenant>__<workflow>`` shape from the API container's
        # acquire_profile_lock) so customer A's anchors can never leak
        # into customer B's runs.
        #
        # The backend is suite-driven (``build_hint_store``): the
        # Learning Allocator steers it per trial via ``_hint_store_disabled``
        # (frozen → NullHintStore, no overlay/record) and
        # ``_hint_store_dict_name`` (S0 retrieval → a shared modal.Dict so
        # anchors accumulate across workers + runs). Absent both flags it
        # stays the production DiskHintStore.
        try:
            from mantis_agent.gym.hint_memory import (
                apply_hint_overlay, build_hint_store,
            )
            hint_tenant = str(task_suite.get("_profile_id", "") or "") or "default"
            micro_runner._hint_store = build_hint_store(
                task_suite, tenant_id=hint_tenant,
            )
            applied = apply_hint_overlay(
                micro_plan,
                store=micro_runner._hint_store,
                plan_signature=plan_signature,
                start_url=str(task_suite.get("base_url", "") or ""),
            )
            store_kind = type(micro_runner._hint_store).__name__
            if applied:
                print(
                    f"  Hint-memory[{store_kind}]: tenant={hint_tenant} "
                    f"plan_sig={plan_signature[:12]} applied={applied} hints"
                )
            elif store_kind != "DiskHintStore":
                # Surface the allocator-steered backend (frozen / S0) even
                # when nothing overlaid — e.g. first run of a fresh shared
                # store — so the selection is greppable in Modal logs.
                print(
                    f"  Hint-memory[{store_kind}]: tenant={hint_tenant} "
                    f"plan_sig={plan_signature[:12]} (no overlay yet)"
                )
        except Exception as exc:  # noqa: BLE001 — never block executor
            print(f"  WARNING: hint memory init failed (recording disabled): {exc}")
            from mantis_agent.gym.hint_memory import NullHintStore
            micro_runner._hint_store = NullHintStore()

        # S1 exemplar replay (Learning Allocator). When the S1 rung pre-seeds
        # worked procedures into the suite (``_exemplars``), stamp them onto
        # the plan's matching steps so the holo3 brain surfaces a "Worked
        # example" (consumer: ``_build_scoped_task``). Parallel to the hint
        # overlay above but a *different* signal — what WORKED (action→
        # outcome), never a coordinate — which is what lets the allocator
        # tell S1 (policy cluster) apart from S0 (knowledge cluster). Absent
        # the flag this is a no-op, so frozen / S0 runs are unchanged.
        try:
            exemplars = task_suite.get("_exemplars") or []
            if exemplars:
                from mantis_agent.gym.exemplar_memory import apply_exemplar_overlay
                stamped = apply_exemplar_overlay(
                    micro_plan, exemplars, plan_signature=plan_signature,
                )
                # print (not logger.info): the S1-vs-frozen separation hinges
                # on whether this fired, and print survives ``modal app logs``.
                print(
                    f"  Exemplar-replay: plan_sig={plan_signature[:12]} "
                    f"exemplars={len(exemplars)} stamped={stamped}"
                )
        except Exception as exc:  # noqa: BLE001 — never block the executor
            print(f"  WARNING: exemplar overlay failed (S1 disabled): {exc}")

        # #627: bind a cross-worker shared seen-URL set from the suite
        # metadata. Modal orchestrator sets ``_fanout_seen_dict_name``
        # before spawning; workers re-attach to the same modal.Dict by
        # that name so a listing already extracted by one worker won't
        # be re-extracted by a sibling. Defaults to NullSharedSeenSet
        # (no-op) when the field is absent — preserves single-worker
        # behaviour.
        try:
            from mantis_agent.gym.fanout_runner import build_shared_seen_set
            micro_runner._shared_seen_set = build_shared_seen_set(task_suite)
        except Exception as exc:  # noqa: BLE001 — never block a run
            print(f"  WARNING: shared seen-set init failed: {exc}")

        # #631: bind a fan-out branch_context from the suite metadata so
        # the AugurAdapter labels this worker's DebugSession under a
        # shared parent_run_id. ``None`` (no fan-out / single worker) →
        # the adapter opens without a branch label, preserving today's
        # session shape.
        try:
            from mantis_agent.gym.fanout_runner import build_fanout_branch_context
            micro_runner._fanout_branch_context = build_fanout_branch_context(
                task_suite,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: fanout branch_context init failed: {exc}")

        # #680: forward the orchestrator-set ``_fanout_group_id`` to
        # the runner so AugurAdapter opens the child session with
        # ``group_id=`` (GRPO sibling-rollout correlation). Same value
        # as ``branch_context.parent_run_id`` today — kept on a
        # separate attribute so a future loop with rollouts that don't
        # share a parent can set group_id alone.
        micro_runner._fanout_group_id = str(
            task_suite.get("_fanout_group_id", "") or ""
        ) or None

        # #683: forward the orchestrator-composed ``task_spec`` so the
        # child Phase-1 / Phase-2 worker session opens with the same
        # canonical task definition. The trajectory buffer's
        # task_spec_ids filter matches against child bundles too —
        # without this hop the parent row is the only one with a
        # task_spec_id.
        _ts = task_suite.get("_fanout_task_spec")
        micro_runner._fanout_task_spec = (
            dict(_ts) if isinstance(_ts, dict) and _ts else None
        )

        # #638 axis 2 follow-up: derive a short worker tag from the
        # fan-out branch_id so per-step log lines can be greppable per-
        # worker in the interleaved orchestrator stdout. The branch_id
        # shape is ``{parent_run_id}:{phase_tag}`` (e.g.
        # ``fanout-XXX:phase2_w3``) — take only the suffix after ``:``.
        # Empty when no branch_id is set (single-container runs) so
        # ``log_progress`` falls back to the legacy format unchanged.
        branch_id = str(task_suite.get("_fanout_branch_id", "") or "")
        micro_runner._worker_tag = branch_id.rsplit(":", 1)[-1] if ":" in branch_id else ""
        # #638 axis 2 follow-up: expose the plan_name as a runner
        # attribute so ``RunExecutor`` can emit it as an Augur tag.
        # Empty string is fine — Augur stores it as a tag regardless,
        # and ad-hoc runs with no plan-file source legitimately have no
        # canonical name.
        micro_runner.plan_name = plan_name

        # Reasoning-trace stream → ``<run_dir>/reasoning.jsonl``. The
        # API container's ``action=reasoning_trace`` reads this file
        # so a viewer overlay can render a structured timeline of
        # critic decisions / Claude recovery results / etc.
        # alongside the live MJPEG feed. Only configured when the
        # API forwarded the run identity (CLI path uses neither).
        if api_run_id and api_tenant_id:
            try:
                from mantis_agent.gym import reasoning_trace as _rt
                _rt.configure_disk_stream(
                    micro_runner,
                    _run_dir(api_tenant_id, api_run_id) / "reasoning.jsonl",
                )
                print(f"  reasoning trace → {api_run_id}/reasoning.jsonl")
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: reasoning-trace stream init failed: {exc}")

        # #347: default ``request_user_input`` host tool. Brains that emit
        # ``Action(TOOL_CALL, name="request_user_input")`` get a paused-run
        # snapshot on the first call, and the staged ``user_input`` on the
        # second (after action=resume rehydrates).
        def _request_user_input(args):
            staged = micro_runner.consume_pause_input(default=None)
            if staged is None:
                raise PauseRequested(
                    reason="user_input",
                    prompt=str(args.get("prompt", "")),
                )
            return staged
        micro_runner.register_tool(
            "request_user_input",
            {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "additionalProperties": False,
            },
            _request_user_input,
        )

        # #347: resume continuation. The Modal API container layers
        # ``_resume_pause_state`` + ``_resume_user_input`` onto the
        # task_suite on action=resume; we feed them to runner.resume().
        resume_blob = task_suite.get("_resume_pause_state")
        if resume_blob is not None:
            pause_state_obj = (
                PauseState.from_dict(resume_blob)
                if isinstance(resume_blob, dict) else resume_blob
            )
            runner_result = micro_runner.resume(
                pause_state_obj,
                user_input=task_suite.get("_resume_user_input"),
                plan=micro_plan,
            )
        else:
            runner_result = micro_runner.run_with_status(
                micro_plan, resume=resume_state,
            )
        step_results = runner_result.steps

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
            profile_id=profile_id,
            workflow_id=workflow_id,
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
        envelope = {
            "mode": "micro",
            "viable": viable,
            "steps": result["steps_executed"],
            "state_key": state_key,
            "profile_id": profile_id,
            "workflow_id": workflow_id,
            "checkpoint_path": checkpoint_path,
            "status": result.get("costs", {}).get("status", ""),
            "dynamic_verification_summary": result.get("dynamic_verification_summary"),
            "run_id": run_id,
            # #audit item 4 follow-up: the Modal entry was dropping
            # ``terminal_status`` + ``halt_reason`` from the envelope
            # so the API container couldn't read them off the result.
            # Re-export both so ``_do_action`` can map the wire
            # ``status`` honestly instead of stamping "succeeded" on
            # every non-exception result.
            "terminal_status": result.get("terminal_status", ""),
            "halt_reason": result.get("halt_reason", ""),
            # #508: surface the legacy leads list AND the new artifacts
            # array on the envelope so the API-container ``_write_result``
            # call has data to feed ``persist_run_artifacts`` (which
            # otherwise sees a stripped dict and writes nothing).
            "leads": leads,
            "artifacts": result.get("artifacts", []),
            # #628 / #631 follow-up: forward the orchestrator-side
            # contract fields. read_partition_result on the orchestrator
            # reads these to drive Phase-2 (collected_urls), aggregate
            # phone counts (leads_with_phone), and report shared-seen
            # savings (shared_seen_hits). Without this re-export, the
            # envelope at line 1016 strips them — Phase-1 silently
            # collected URLs and silently dropped them on return,
            # making Phase-1/Phase-2 fail every time.
            "collected_urls": result.get("collected_urls", []),
            "leads_with_phone": result.get("leads_with_phone", 0),
            "shared_seen_hits": result.get("shared_seen_hits", 0),
        }
        # #347: surface paused state to the Modal API container. The poll
        # path detects ``_paused`` on the FunctionCall result and writes
        # pause_state.json + flips status.json to paused for the next
        # action=resume round-trip.
        if runner_result.paused and runner_result.pause_state is not None:
            envelope["_paused"] = True
            envelope["pause_state"] = runner_result.pause_state.to_dict()
            envelope["prompt"] = runner_result.pause_state.prompt
            envelope["reason"] = runner_result.pause_state.pending_reason
        return envelope

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
        fallback_used = getattr(result, "fallback_used", "")
        if "setup" in task_id or "filter" in task_id:
            min_steps = max(min_steps, 5)

        # Hybrid mode: use Claude for setup tasks
        if (
            ("setup" in task_id or "filter" in task_id)
            and verify_mode
            and fallback_used != "claude"
        ):
            try:
                from mantis_agent.brain_claude import ClaudeBrain as _CB
                task_brain = _CB(model="claude-sonnet-4-6", thinking_budget=2048, screen_size=(1280, 720))
                task_brain.load()
                print("  Using Claude Sonnet for setup (hybrid mode)")
                runner = GymRunner(brain=task_brain, env=_env, max_steps=task_max_steps,
                                   frames_per_inference=config.frames_per_inference, on_step=on_step)
                result = runner.run(task=task_config["intent"], task_id=task_id,
                                   start_url=task_config.get("start_url", ""))
            except Exception as e:
                print(f"  Claude brain failed, using Holo3: {e}")

        # Retry if model skipped interaction
        if (
            result.total_steps <= min_steps
            and result.success
            and min_steps > 0
            and fallback_used != "claude"
        ):
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
        fallback_brain=claude_fallback_brain,
        fallback_label="claude",
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
    profile_dir: str = "",
    **_extra,
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
    env, proxy_proc, proxy_diag = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name,
        settle_time=2.0, display=":99", start_xvfb=True,
        proxy_city=str(task_suite.get("_proxy_city") or ""),
        proxy_state=str(task_suite.get("_proxy_state") or ""),
        proxy_provider=str(task_suite.get("_proxy_provider") or ""),
        proxy_country=str(task_suite.get("_proxy_country") or ""),
        proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
        extra_http_headers=task_suite.get("_browser_extra_headers") or None,
        profile_dir=profile_dir,
    )
    viewer_ctx, viewer_event_bus, _viewer_url = setup_viewer(viewer, proxy_diag=proxy_diag)

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
    image=planner_base_image.run_commands(
        "apt-get update && apt-get install -y gnupg curl wget xvfb xdotool xclip scrot",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
    ).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20", "websocket-client",
        # #509: parity with run_holo3 — Gemma4 also runs the augur wedge.
        # Must match pyproject.toml (``augur-sdk>=0.6.0,<0.7``). 0.2.x
        # added ``branch_context`` to ``DebugSession.__init__``; 0.4.0
        # (mercurialsolo/augur-sdk#38) added
        # ``DebugSession.open_orchestrator(...)``; 0.6.0 (#680) adds
        # ``task_spec`` / ``group_id`` / ``set_loop_detected`` /
        # ``record_subgoal_completion`` and the ``set_score`` kwargs
        # ``should_stop`` / ``uncertainty``. Stale <0.6 pins silently
        # drop the new fields — adapter swallows the TypeError and the
        # bundle ships without RL-training metadata.
        "augur-sdk>=0.6.0,<0.7",
    ).add_local_python_source("mantis_agent").add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE),
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
    .apt_install("curl", "wget", "gnupg", "xvfb", "xdotool", "xclip", "scrot",
                 # #stealth-parity: same fonts+locale+tzdata as the
                 # vLLM executor_image — the Claude executor launches
                 # Chrome via the same xdotool_env path and would
                 # otherwise present a Liberation-only sparse font set
                 # + Etc/UTC timezone, both strong bot signals.
                 *_STEALTH_APT_FONTS_AND_LOCALE)
    .run_commands(
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y google-chrome-stable || true",
        # #stealth-parity: generate en_US locale + America/New_York TZ.
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime",
    )
    .env({
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
    })
    .pip_install(
        "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20", "websocket-client",
        # #509: parity with run_holo3 / run_gemma4_cua — Claude executor
        # also runs the augur wedge so bundles + streaming work for the
        # Anthropic-API tier.
        # Must match pyproject.toml (``augur-sdk>=0.6.0,<0.7``). 0.2.x
        # added ``branch_context`` to ``DebugSession.__init__``; 0.4.0
        # (mercurialsolo/augur-sdk#38) added
        # ``DebugSession.open_orchestrator(...)``; 0.6.0 (#680) adds
        # ``task_spec`` / ``group_id`` / ``set_loop_detected`` /
        # ``record_subgoal_completion`` and the ``set_score`` kwargs
        # ``should_stop`` / ``uncertainty``. Stale <0.6 pins silently
        # drop the new fields — adapter swallows the TypeError and the
        # bundle ships without RL-training metadata.
        "augur-sdk>=0.6.0,<0.7",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE)
    .add_local_dir(_WEBGL_SPOOF_LOCAL, remote_path=_WEBGL_SPOOF_REMOTE)
)


def _resolve_claude_computer_plane_config():
    """Build the `ComputerPlaneConfig` for `run_claude_cua`.

    Reads `MANTIS_COMPUTER_PLANE_BACKEND` (default `local`). When set
    to `modal`, resolves the `computer_plane` Modal function's web URL
    if `MANTIS_COMPUTER_PLANE_URL` isn't explicitly set in the secret
    — keeps the rollback story to a single env-var edit.

    Returns `None` on `local` so the call site can rely on `setup_env`'s
    env-var-driven default path (and tests that monkeypatch `setup_env`
    don't need to know about this helper).
    """
    backend = (os.environ.get("MANTIS_COMPUTER_PLANE_BACKEND") or "local").strip().lower()
    if backend == "local":
        return None
    from mantis_agent.gym.computer_client import ComputerPlaneConfig

    base_url = (os.environ.get("MANTIS_COMPUTER_PLANE_URL") or "").strip()
    if not base_url and backend == "modal":
        try:
            base_url = modal.Function.from_name(APP_NAME, "computer_plane").get_web_url()
        except Exception as exc:  # noqa: BLE001 — print at WARNING via Modal
            # Modal suppresses INFO/DEBUG (see
            # `feedback_warning_level_for_modal_observability.md`); use
            # print so the diagnostic survives `modal app logs`.
            print(
                f"  WARNING: Failed to resolve computer_plane web URL "
                f"via Modal SDK ({exc}); set MANTIS_COMPUTER_PLANE_URL "
                f"explicitly in the secret."
            )
            base_url = ""
    enable_cdp = (os.environ.get("MANTIS_COMPUTER_PLANE_ENABLE_CDP") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    return ComputerPlaneConfig(
        backend=backend,  # type: ignore[arg-type]
        remote_base_url=base_url or None,
        remote_auth_token=(os.environ.get("MANTIS_COMPUTER_PLANE_TOKEN") or "").strip() or None,
        enable_cdp=enable_cdp,
    )


def _run_claude_executor(
    task_file_contents: str,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 2,
    claude_model: str = "claude-sonnet-4-6",
    thinking_budget: int = 2048,
    viewer: bool = False,
    profile_dir: str = "",
    **_extra,
) -> dict:
    """Execute tasks using Claude CUA via Anthropic API.

    No GPU needed — inference is via API. Only needs Chrome + xdotool.
    Trajectories are saved for potential distillation training.

    Phase-1 migration target (#698): set
    `MANTIS_COMPUTER_PLANE_BACKEND=modal` in the Modal secret to flip
    this executor onto the remote computer plane. Default `local`
    keeps the in-process behavior; rollback is a single secret edit.
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
    computer_plane_config = _resolve_claude_computer_plane_config()
    env, proxy_proc, proxy_diag = setup_env(
        base_url=task_suite.get("base_url", ""),
        run_id=run_id, session_name=session_name,
        settle_time=2.0, display=":99", start_xvfb=True,
        proxy_city=str(task_suite.get("_proxy_city") or ""),
        proxy_state=str(task_suite.get("_proxy_state") or ""),
        proxy_provider=str(task_suite.get("_proxy_provider") or ""),
        proxy_country=str(task_suite.get("_proxy_country") or ""),
        proxy_disabled=bool(task_suite.get("_proxy_disabled", False)),
        extra_http_headers=task_suite.get("_browser_extra_headers") or None,
        profile_dir=profile_dir,
        computer_plane_config=computer_plane_config,
        executor_name="run_claude_cua",
    )
    viewer_ctx, viewer_event_bus, _viewer_url = setup_viewer(viewer, proxy_diag=proxy_diag)

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
def run_claude_cua(task_file_contents: str, claude_model: str = "claude-sonnet-4-6", **kwargs) -> dict:
    """Claude CUA executor (no GPU — API-based inference, Chrome + xdotool only)."""
    kwargs.pop("cua_model", None)
    return _run_claude_executor(task_file_contents, claude_model=claude_model, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# D) HTTP API (#342) — concurrent multi-plan submission via ASGI
# ═══════════════════════════════════════════════════════════════════

# Lightweight image — no Chrome, no CUDA, no vLLM. Just FastAPI +
# pydantic + the mantis_agent package. The ASGI container only validates,
# locks, and dispatches; all heavy work happens in the executor functions
# above via .spawn().
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.0",
        "requests",
        "anthropic",
        # #509: api container imports ``mantis_agent.gym.run_executor``
        # transitively (via the persist + dispatch paths). The adapter
        # is lazy-imported but if augur_sdk is missing it falls back
        # silently — keeping the dep here makes the import succeed and
        # lets future API-side bundle reads work without a redeploy.
        # Must match pyproject.toml (``augur-sdk>=0.6.0,<0.7``). 0.2.x
        # added ``branch_context`` to ``DebugSession.__init__``; 0.4.0
        # (mercurialsolo/augur-sdk#38) added
        # ``DebugSession.open_orchestrator(...)``; 0.6.0 (#680) adds
        # ``task_spec`` / ``group_id`` / ``set_loop_detected`` /
        # ``record_subgoal_completion`` and the ``set_score`` kwargs
        # ``should_stop`` / ``uncertainty``. Stale <0.6 pins silently
        # drop the new fields — adapter swallows the TypeError and the
        # bundle ships without RL-training metadata.
        "augur-sdk>=0.6.0,<0.7",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE)
)


def _executor_for_model(model: str):
    """Map ``cua_model`` → the appropriate ``@app.function`` executor."""
    table = {
        "evocua-8b": run_cua_1gpu,
        "evocua-32b": run_cua_2gpu,
        "opencua-32b": run_cua_4gpu,
        "opencua-72b": run_cua_8gpu,
        "holo3": run_holo3,
        "fara": run_fara,
        "gemma4-cua": run_gemma4_cua,
        "claude": run_claude_cua,
    }
    return table.get(model, run_holo3)


def _build_suite_from_payload(payload: dict) -> str:
    """Build the ``task_file_contents`` string the executors expect.

    Supports three pre-decomposed shapes for Phase 1 of #342:

    * ``task_suite`` dict — used verbatim.
    * ``task_file_contents`` string — used verbatim.
    * ``micro`` ending in ``.json`` — loaded, packed via
      :func:`build_micro_suite` with the resolved identity (#341).

    ``plan_text`` and ``.txt`` ``micro`` paths require Claude
    decomposition and are explicitly out of scope for Phase 1 — see
    issue #342 Phase 2.
    """
    if payload.get("task_suite"):
        return json.dumps(payload["task_suite"])
    if payload.get("task_file_contents"):
        return payload["task_file_contents"]

    micro = payload.get("micro") or payload.get("micro_path") or ""
    if not micro or not micro.endswith(".json"):
        raise ValueError(
            "Modal HTTP endpoint v1 requires task_suite, task_file_contents, "
            "or a .json micro path. Free-text decomposition is Phase 2 (#342)."
        )

    from pathlib import Path as _Path
    from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer

    path = _Path(micro)
    if not path.is_absolute():
        repo_root = _Path(os.environ.get("MANTIS_REPO_ROOT", "/workspace/cua-agent"))
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"micro plan not found: {path}")

    raw_steps = json.loads(path.read_text())
    micro_plan = MicroPlan(domain=path.stem)
    for step in raw_steps:
        micro_plan.steps.append(PlanDecomposer._build_intent(step))

    steps_dicts = micro_plan_steps_to_dicts(micro_plan.steps)
    suite = build_micro_suite(
        steps_dicts,
        micro_plan.domain,
        max_cost=float(payload.get("max_cost", 10.0)),
        max_time_minutes=int(payload.get("max_time_minutes", 180)),
        resume_state=bool(payload.get("resume_state", False)),
        state_key=str(payload.get("state_key") or ""),
        profile_id=str(payload.get("profile_id") or ""),
        workflow_id=str(payload.get("workflow_id") or ""),
        proxy_city=str(payload.get("proxy_city") or ""),
        proxy_state=str(payload.get("proxy_state") or ""),
        # #stealth-parity bug fix: previously this path dropped
        # ``proxy_provider`` while threading proxy_city + proxy_state.
        # The downstream ``build_proxy_config`` then defaulted to
        # ``iproyal`` (stale creds) instead of the explicit provider
        # the caller asked for. Plans submitted via the .json micro
        # path (vs the pre-built task_suite path) silently fell
        # back to no-proxy egress through Modal IPs.
        proxy_provider=str(payload.get("proxy_provider") or ""),
        proxy_country=str(payload.get("proxy_country") or ""),
        proxy_disabled=bool(payload.get("proxy_disabled", False)),
        # #560: forward only when the caller supplied one — ``None``
        # lets the runner pick its DEFAULT_BRAIN_BUDGET_CAPS.
        brain_budgets=payload.get("brain_budgets"),
        # #570: per-run cf_challenge auto-pause override. ``None`` →
        # runner falls back to MANTIS_PAUSE_ON_CAPTCHA env.
        pause_on_captcha=payload.get("pause_on_captcha"),
        # #561: per-run ceiling on adaptive_settle waits.
        settle_ceiling_seconds=payload.get("settle_ceiling_seconds"),
        # #567: per-run agentic-recovery budget overrides.
        max_recoveries_per_run=payload.get("max_recoveries_per_run"),
        max_recoveries_per_step=payload.get("max_recoveries_per_step"),
    )
    # #638 axis 2 follow-up: stamp a human-readable plan_name on the
    # suite so the Augur Runs list can group runs of the same plan.
    # ``payload['plan_name']`` wins when the caller sets it; otherwise
    # fall back to the source file stem (e.g. ``boattrader_scrape_urlnav``).
    suite["_plan_name"] = str(payload.get("plan_name") or path.stem or "")
    return json.dumps(suite)


def _run_dir(tenant_id: str, run_id: str):
    from pathlib import Path as _Path
    from mantis_agent.server_utils import safe_state_key as _safe
    root = _Path(os.environ.get("MANTIS_DATA_DIR", "/data"))
    return root / "tenants" / _safe(tenant_id) / "runs" / _safe(run_id)


def _chrome_profile_dir(tenant_id: str, profile_id: str) -> str:
    """Per-tenant, per-profile Chrome user-data-dir (#341).

    Without this, every Modal run reused Chrome's default
    ``/data/chrome-profile`` directory — so cookies, localStorage and
    IndexedDB from one run leaked into the next regardless of the
    API-side ``profile_id``. Returns a stringified path the executor
    forwards to ``setup_env(profile_dir=...)``.
    """
    from pathlib import Path as _Path
    from mantis_agent.server_utils import safe_state_key as _safe
    root = _Path(os.environ.get("MANTIS_DATA_DIR", "/data"))
    path = root / "tenants" / _safe(tenant_id) / "chrome-profile" / _safe(profile_id)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _commit_volume() -> None:
    """Best-effort ``vol.commit()`` — no-op when the Modal volume isn't mounted (tests)."""
    try:
        vol.commit()
    except Exception:
        pass


# In-memory recent-runs cache. Modal Volume commit + reload has a small
# eventual-consistency window: an immediate poll right after submit
# could land on a container whose volume mount hasn't yet seen the
# status.json we just wrote, returning a misleading 404 / "unknown
# run_id". This cache backstops that window — when the same container
# that wrote a status reads it back, the cache is authoritative.
#
# Bounded LRU-style: at 1024 entries we drop the oldest. The cache is
# also cleared lazily when a status flips to a terminal phase (no point
# caching a finished run that the client will fetch via /result).
_RECENT_RUNS: dict[str, dict] = {}
_RECENT_RUNS_MAX = 1024
_TERMINAL_STATUSES = frozenset({
    "succeeded", "failed", "cancelled", "completed_with_failures",
    "timeout", "halted",
})


def _cache_status(tenant_id: str, run_id: str, status: dict) -> None:
    """Stash ``status`` in the in-memory cache keyed by (tenant, run)."""
    key = f"{tenant_id}::{run_id}"
    _RECENT_RUNS[key] = dict(status)
    if len(_RECENT_RUNS) > _RECENT_RUNS_MAX:
        # Drop the oldest ~25% so we don't trim on every write.
        drop = max(1, _RECENT_RUNS_MAX // 4)
        for k in list(_RECENT_RUNS.keys())[:drop]:
            _RECENT_RUNS.pop(k, None)


def _write_status(tenant_id: str, run_id: str, status: dict) -> None:
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    _cache_status(tenant_id, run_id, status)
    _commit_volume()


def _read_status(tenant_id: str, run_id: str) -> dict | None:
    """Return the run's status dict.

    Reads the in-memory cache first (covers the volume-commit
    eventual-consistency window), then falls back to the file-backed
    status.json. When the file read returns a newer status than the
    cache (executor wrote it from a different container), the cache
    is refreshed so subsequent calls see the latest.
    """
    key = f"{tenant_id}::{run_id}"
    cached = _RECENT_RUNS.get(key)
    path = _run_dir(tenant_id, run_id) / "status.json"
    on_disk: dict | None = None
    if path.exists():
        try:
            on_disk = json.loads(path.read_text())
        except Exception:
            on_disk = None
    # Prefer the on-disk view when it exists AND its updated_at is
    # newer than the cached view — that's the executor having
    # progressed the run from a different container.
    if on_disk is not None:
        if cached is None or str(on_disk.get("updated_at", "")) >= str(cached.get("updated_at", "")):
            _cache_status(tenant_id, run_id, on_disk)
            # Evict from cache once terminal — no need to keep
            # finished runs around.
            if str(on_disk.get("status", "")).lower() in _TERMINAL_STATUSES:
                _RECENT_RUNS.pop(key, None)
            return on_disk
    return cached


def _write_viewer_url(tenant_id: str, run_id: str, viewer_url: str) -> None:
    """Persist the live-viewer URL in a side-channel file (#416).

    Kept separate from ``status.json`` so the executor never has to
    read-merge-write a file the API container also owns.
    """
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "viewer.json").write_text(json.dumps({"viewer_url": viewer_url}))
    _commit_volume()


def _read_viewer_url(tenant_id: str, run_id: str) -> str | None:
    path = _run_dir(tenant_id, run_id) / "viewer.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("viewer_url")
    except Exception:
        return None


def _augur_bundle_dir(augur_run_id: str):
    """Resolve the on-disk Augur bundle directory for ``augur_run_id``.

    Mirrors :func:`mantis_agent.observability.augur.default_out_dir` but
    runs in the API container's filesystem (no extra import of the
    augur_sdk package needed). Honors ``MANTIS_AUGUR_DIR`` for the same
    override the runner reads.
    """
    from pathlib import Path as _Path
    override = os.environ.get("MANTIS_AUGUR_DIR", "").strip()
    if override:
        return _Path(override) / augur_run_id
    root = os.environ.get("MANTIS_DATA_DIR", "/data").strip() or "/data"
    return _Path(root) / "augur" / augur_run_id


def _write_augur_metadata(
    tenant_id: str, run_id: str, augur_run_id: str,
) -> None:
    """Persist the API-run-id → augur-run-id mapping (gap #1 of the
    observability ergonomics audit).

    Stored in a side-channel file so the executor can stamp it from a
    different container than the API. The API's lifecycle endpoint
    reads this on every poll and surfaces ``augur_run_id`` plus a
    derived bundle URL.
    """
    if not augur_run_id:
        return
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "augur.json").write_text(json.dumps({
        "augur_run_id": augur_run_id,
        "bundle_dir": str(_augur_bundle_dir(augur_run_id)),
        "dsn_workspace": os.environ.get("AUGUR_DSN_WORKSPACE_URL", "") or "",
    }))
    _commit_volume()


def _read_augur_metadata(tenant_id: str, run_id: str) -> dict | None:
    path = _run_dir(tenant_id, run_id) / "augur.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_result(tenant_id: str, run_id: str, result: dict) -> None:
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    # #508: materialize leads.csv / extracted_rows.csv / extracted_rows.json
    # alongside result.json so the artifact endpoint can serve them by name.
    # Best-effort — if any one file write fails we still persist result.json.
    try:
        file_artifacts = persist_run_artifacts(result, run_dir, run_id=run_id)
        if file_artifacts:
            result["artifacts"] = list(result.get("artifacts") or []) + file_artifacts
    except Exception as exc:  # noqa: BLE001 — artifact write must never block result.json
        # Modal suppresses INFO; use warning so this stays visible in `modal app logs`.
        print(f"WARNING: persist_run_artifacts failed for {run_id}: {exc}")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    _commit_volume()


def _read_result(tenant_id: str, run_id: str) -> dict | None:
    path = _run_dir(tenant_id, run_id) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_pause_state(tenant_id: str, run_id: str, pause_state: dict) -> None:
    """Persist the paused-run snapshot for the next action=resume (#347)."""
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pause_state.json").write_text(json.dumps(pause_state, indent=2))
    _commit_volume()


def _read_pause_state(tenant_id: str, run_id: str) -> dict | None:
    path = _run_dir(tenant_id, run_id) / "pause_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_task_suite(tenant_id: str, run_id: str, task_file_contents: str) -> None:
    """Snapshot the raw task_suite string so resume can re-spawn the executor (#347)."""
    run_dir = _run_dir(tenant_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_suite.json").write_text(task_file_contents)
    _commit_volume()


def _read_task_suite(tenant_id: str, run_id: str) -> str | None:
    path = _run_dir(tenant_id, run_id) / "task_suite.json"
    if not path.exists():
        return None
    try:
        return path.read_text()
    except Exception:
        return None


def build_api_app(executor_resolver=None, function_call_lookup=None):
    """Construct the FastAPI app exposed by the Modal ASGI endpoint (#342).

    Factored into a plain function (not wrapped by ``@modal.asgi_app()``)
    so unit tests can drive it through ``fastapi.testclient.TestClient``
    without going through Modal's runtime.

    * ``executor_resolver(model) -> executor_fn`` — overrides the
      built-in ``_executor_for_model`` map. Tests pass a stub that
      returns a fake ``.spawn(...)`` handle.
    * ``function_call_lookup(call_id) -> FunctionCall`` — overrides
      ``modal.FunctionCall.from_id``. Tests pass a stub that returns
      a fake ``.get(timeout=...) / .cancel()`` interface so polling
      doesn't need a live Modal runtime.
    """
    from datetime import datetime, timezone

    from fastapi import Depends, FastAPI, HTTPException, Request

    from mantis_agent.baseten_server.middleware import require_run_scope
    from mantis_agent.server.run_dispatch import (
        DispatchError,
        acquire_profile_lock,
        prepare_predict_payload,
        read_profile_lock,
        release_profile_lock,
    )
    from mantis_agent.server_utils import new_run_id as _mint_run_id
    from mantis_agent.tenant_auth import TenantConfig

    resolve_executor = executor_resolver or _executor_for_model
    lookup_function_call = function_call_lookup or modal.FunctionCall.from_id

    fastapi_app = FastAPI(
        title="Mantis CUA (Modal)",
        version="0.1",
        description="Concurrent multi-plan submission via @modal.asgi_app — #342",
    )

    @fastapi_app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", "service": "mantis-cua-modal-api"}

    @fastapi_app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "mantis"}
                for name in (
                    "evocua-8b", "evocua-32b", "opencua-32b", "opencua-72b",
                    "holo3", "fara", "gemma4-cua", "claude",
                )
            ],
        }

    @fastapi_app.post("/v1/predict")
    async def predict(
        request: Request,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        try:
            raw = await request.json()
        except Exception:
            raise HTTPException(400, "request body must be JSON")
        try:
            payload = prepare_predict_payload(raw, tenant)
        except DispatchError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

        # Action mode: poll/cancel an existing run.
        if payload.get("action"):
            return _do_action(payload, tenant)

        # Run mode: build suite, lock the profile, spawn the executor.
        try:
            task_file_contents = _build_suite_from_payload(payload)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc

        # DX-3 (#785 follow-up): dry-run preview. Returns the resolved
        # task_suite + summary + cost estimate without acquiring the
        # Chrome lock or spawning the executor. Lets devs iterate on
        # plan_text → MicroPlan without burning GPU credit.
        if bool(payload.get("dry_run")):
            from mantis_agent.server.dry_run import build_dry_run_response
            return build_dry_run_response(
                task_file_contents, payload, tenant.tenant_id
            )

        profile_id = payload["profile_id"]
        workflow_id = payload["workflow_id"]
        run_id = _mint_run_id()

        # 409 if another run is using this profile (Chrome lock).
        if not acquire_profile_lock(tenant, profile_id, run_id):
            existing = read_profile_lock(tenant, profile_id)
            raise HTTPException(
                409,
                f"profile_id {profile_id!r} is busy; held by run_id={existing!r}. "
                "Use a different profile_id or wait for the existing run to finish.",
            )

        model = payload.get("cua_model") or payload.get("model") or "holo3"
        executor_fn = resolve_executor(model)

        # Forward only what the executors accept. `task_file_contents`
        # is positional; everything else is **kwargs.
        spawn_kwargs: dict = {
            "max_steps": int(payload.get("max_steps", 30)),
            # Per-tenant, per-profile Chrome user-data-dir (#341).
            # Until this was added, Modal executors fell back to
            # XdotoolGymEnv's default ``/data/chrome-profile`` and
            # every run on the same warm container shared one Chrome
            # profile — so ``profile_id`` was only honoured by the
            # API-side lock and not by Chrome itself.
            "profile_dir": _chrome_profile_dir(tenant.tenant_id, profile_id),
        }
        if model != "claude":
            spawn_kwargs["cua_model"] = model
        # #416: forward the live-viewer flag into the executor so it
        # can call setup_viewer + write the tunnel URL into the API-
        # side status.json. The executor needs the API's run_id +
        # tenant_id to know which status file to update (the
        # executor's internal run_id differs).
        if payload.get("live_viewer"):
            spawn_kwargs["viewer"] = True
            spawn_kwargs["api_run_id"] = run_id
            spawn_kwargs["api_tenant_id"] = tenant.tenant_id
        try:
            call_handle = executor_fn.spawn(
                task_file_contents=task_file_contents,
                **spawn_kwargs,
            )
        except Exception as exc:
            release_profile_lock(tenant, profile_id)
            raise HTTPException(500, f"executor spawn failed: {exc}") from exc

        now = datetime.now(timezone.utc).isoformat()
        status = {
            "run_id": run_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "modal_call_id": getattr(call_handle, "object_id", "") or "",
            "tenant_id": tenant.tenant_id,
            "profile_id": profile_id,
            "workflow_id": workflow_id,
            "state_key": payload["state_key"],
            "model": model,
            "max_steps": int(payload.get("max_steps", 30)),
        }
        _write_status(tenant.tenant_id, run_id, status)
        # #347: snapshot the suite so a later action=resume can re-spawn
        # the executor with the same plan + identity + caps.
        _write_task_suite(tenant.tenant_id, run_id, task_file_contents)

        run_dir = _run_dir(tenant.tenant_id, run_id)
        return {
            "status": "queued",
            "mode": "detached",
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "model": model,
            "payload": {
                "profile_id": profile_id,
                "workflow_id": workflow_id,
            },
            "status_path": str(run_dir / "status.json"),
            "result_path": str(run_dir / "result.json"),
            "csv_path": str(run_dir / "leads.csv"),
            "events_path": str(run_dir / "events.log"),
        }

    def _do_action(payload: dict, tenant: TenantConfig) -> dict:
        """Handle ``action=status|result|cancel|resume`` against an existing run."""
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            raise HTTPException(400, "action requires run_id")
        # #416 follow-up: Modal Volume reads are cached per container.
        # When an executor commits a status update from a different
        # container (e.g. the live-viewer URL written from inside
        # ``_run_holo3_executor``), the API container's mount keeps
        # serving the old file content until ``vol.reload()`` invalidates
        # the cache. Without this reload, ``status.json`` from the
        # caller's perspective stays frozen at the initial ``queued``
        # state for the whole run — and the executor-side
        # ``viewer_url`` never surfaces. Best-effort: a reload failure
        # is non-fatal; the caller just sees the stale view, same as
        # before this fix.
        try:
            vol.reload()
        except Exception as exc:  # noqa: BLE001 — volume reload is best-effort
            print(f"  WARNING: vol.reload() failed in _do_action: {exc}")
        status = _read_status(tenant.tenant_id, run_id)
        if status is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")

        viewer_url = _read_viewer_url(tenant.tenant_id, run_id)
        if viewer_url:
            status["viewer_url"] = viewer_url

        # #541: synthesize ``status=paused`` when the external-pause
        # sentinel exists. The Modal function call is still "running"
        # from Modal's perspective (executor is sleeping in
        # wait_while_paused), but for the caller we want the
        # paused-state signal so the dashboard / poll loop knows to
        # surface the viewer URL + the resume hint.
        pause_path = _run_dir(tenant.tenant_id, run_id) / "pause_request.json"
        if pause_path.exists() and status.get("status") in {"queued", "running"}:
            try:
                pause_blob = json.loads(pause_path.read_text())
                status["status"] = "paused"
                status["pause_reason"] = pause_blob.get("reason", "")
                status["paused_at"] = pause_blob.get("requested_at", "")
            except (OSError, json.JSONDecodeError):
                pass

        action = payload["action"]

        # action=reasoning_trace — return the structured event stream
        # the runner writes to ``<run_dir>/reasoning.jsonl`` during
        # execution. Used by viewer overlays to render the runner's
        # decision timeline (critic gates, Claude recovery results,
        # ReplaceStep / InsertStep directives) beside the MJPEG feed.
        # Cheap, file-backed read — no compute.
        #
        # Inlined rather than importing ``mantis_agent.gym.reasoning_trace``
        # because the API container's tiny image (fastapi + pydantic +
        # requests) doesn't have PIL — and importing ``mantis_agent.gym``
        # transitively pulls in PlaywrightGymEnv → PIL. The JSONL read
        # is trivial; duplicating those ~12 lines here keeps the import
        # chain clean.
        if action == "reasoning_trace":
            since_ts = str(payload.get("since") or "") or None
            jsonl_path = _run_dir(tenant.tenant_id, run_id) / "reasoning.jsonl"
            events: list[dict] = []
            if jsonl_path.exists():
                try:
                    with jsonl_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            if not isinstance(ev, dict):
                                continue
                            if since_ts and ev.get("ts", "") <= since_ts:
                                continue
                            events.append(ev)
                except OSError:
                    pass
            return {
                **status,
                "events": events,
                "count": len(events),
            }

        if action == "cancel":
            if status.get("status") in {"queued", "running", "paused"}:
                call_id = status.get("modal_call_id", "")
                if call_id and status.get("status") != "paused":
                    try:
                        lookup_function_call(call_id).cancel()
                    except Exception:
                        pass
                status["status"] = "cancelled"
                status["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_status(tenant.tenant_id, run_id, status)
                release_profile_lock(tenant, status.get("profile_id", ""))
            # #541: tidy the pause sentinel on cancel too.
            try:
                pause_path.unlink(missing_ok=True)
            except OSError:
                pass
            return status

        if action == "pause":
            # #541: external pause — write the sentinel file the
            # executor polls between steps. Chrome + noVNC viewer
            # stay alive while the runner sleeps in
            # ``wait_while_paused``. User takes over via the
            # live-viewer URL, then ``action=resume`` clears the
            # sentinel and the runner picks up from the next step.
            current = status.get("status", "")
            if current not in {"queued", "running"}:
                raise HTTPException(
                    400,
                    f"action='pause' requires a running run; "
                    f"this run is {current!r}",
                )
            reason = str(payload.get("reason") or "external") or "external"
            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                pause_path.parent.mkdir(parents=True, exist_ok=True)
                pause_path.write_text(json.dumps({
                    "reason": reason,
                    "requested_at": now_iso,
                }))
            except OSError as exc:
                raise HTTPException(500, f"failed to write pause sentinel: {exc}")
            status["status"] = "paused"
            status["pause_reason"] = reason
            status["paused_at"] = now_iso
            status["updated_at"] = now_iso
            _write_status(tenant.tenant_id, run_id, status)
            return status

        if action == "resume":
            # #541: external-pause resume — when ``pause_request.json``
            # exists, the executor is still running (sleeping in
            # ``wait_while_paused``). Just delete the sentinel; the
            # runner picks up from the next step automatically. No
            # task_suite rehydration / spawn / user_input needed —
            # this path is for human-takeover-via-viewer, where the
            # user has already cleared the page state manually.
            if pause_path.exists():
                try:
                    pause_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise HTTPException(
                        500, f"failed to clear pause sentinel: {exc}",
                    )
                status["status"] = "running"
                status.pop("pause_reason", None)
                status.pop("paused_at", None)
                status["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_status(tenant.tenant_id, run_id, status)
                return status

            # #347: rehydrate a snapshot-paused Modal run. Validate,
            # layer the caller's user_input + stored pause_state onto
            # the saved task_suite, re-spawn the executor, update
            # modal_call_id.
            if status.get("status") != "paused":
                raise HTTPException(
                    400,
                    f"action='resume' requires a paused run; "
                    f"run_id={run_id!r} is in status={status.get('status')!r}",
                )
            user_input = payload.get("user_input")
            if user_input is None:
                raise HTTPException(400, "action='resume' requires user_input")
            pause_state_blob = _read_pause_state(tenant.tenant_id, run_id)
            if pause_state_blob is None:
                raise HTTPException(
                    404, f"pause_state.json missing for run {run_id!r}",
                )
            saved_suite = _read_task_suite(tenant.tenant_id, run_id)
            if saved_suite is None:
                raise HTTPException(
                    404, f"task_suite.json missing for run {run_id!r}",
                )
            # Re-acquire the profile lock under the same run_id; the
            # paused state released it, so this is the gate against a
            # concurrent submission stealing the profile while the human
            # was fetching the OTP.
            profile_id = status.get("profile_id", "")
            if not acquire_profile_lock(tenant, profile_id, run_id):
                held = read_profile_lock(tenant, profile_id)
                raise HTTPException(
                    409,
                    f"profile_id {profile_id!r} is now busy with run_id={held!r}; "
                    "cannot resume while another run holds the lock.",
                )
            # Layer the resume hints onto the task_suite and re-spawn.
            try:
                suite_dict = json.loads(saved_suite)
            except Exception as exc:
                release_profile_lock(tenant, profile_id)
                raise HTTPException(500, f"corrupt task_suite.json: {exc}") from exc
            suite_dict["_resume_pause_state"] = pause_state_blob
            suite_dict["_resume_user_input"] = user_input
            model = status.get("model", "holo3")
            executor_fn = resolve_executor(model)
            spawn_kwargs: dict = {
                "max_steps": int(status.get("max_steps", 30)),
                "profile_dir": _chrome_profile_dir(tenant.tenant_id, profile_id),
            }
            if model != "claude":
                spawn_kwargs["cua_model"] = model
            try:
                call_handle = executor_fn.spawn(
                    task_file_contents=json.dumps(suite_dict),
                    **spawn_kwargs,
                )
            except Exception as exc:
                release_profile_lock(tenant, profile_id)
                raise HTTPException(500, f"executor spawn failed: {exc}") from exc
            now = datetime.now(timezone.utc).isoformat()
            status["status"] = "running"
            status["resumed_at"] = now
            status["updated_at"] = now
            status["modal_call_id"] = getattr(call_handle, "object_id", "") or ""
            # Clear stale pause-surface fields on the resumed status.
            status["prompt"] = ""
            status["reason"] = ""
            _write_status(tenant.tenant_id, run_id, status)
            return {
                "run_id": run_id,
                "status": "running",
                "resumed_at": now,
            }

        # status / result: check the FunctionCall handle if still running.
        if status.get("status") in {"queued", "running"}:
            call_id = status.get("modal_call_id", "")
            if call_id:
                try:
                    result = lookup_function_call(call_id).get(timeout=0.1)
                except modal.exception.OutputExpiredError:
                    status["status"] = "expired"
                except Exception as exc:
                    # TimeoutError from .get(timeout=0.1) means still running.
                    msg = str(type(exc).__name__).lower()
                    if "timeout" not in msg:
                        # Real error — mark failed, release the lock,
                        # attach human-actionable failure_help (#841).
                        from mantis_agent.run_failure_help import failure_help_for
                        status["status"] = "failed"
                        status["error"] = f"{type(exc).__name__}: {exc}"
                        # Best-effort halt-class derivation from the raw
                        # exception type. The runner's per-step
                        # ``halt_class`` is richer when set; this is the
                        # fallback for executor-spawn-side failures.
                        _exc_name = type(exc).__name__.lower()
                        if "connectionreset" in _exc_name or "connectionerror" in _exc_name:
                            status.setdefault("halt_class", "anthropic_unreachable")
                        status["failure_help"] = failure_help_for(
                            status.get("halt_class", "unknown"),
                            run_id=run_id,
                        )
                        status["updated_at"] = datetime.now(timezone.utc).isoformat()
                        _write_status(tenant.tenant_id, run_id, status)
                        release_profile_lock(tenant, status.get("profile_id", ""))
                    else:
                        status["status"] = "running"
                else:
                    # #347: the executor surfaces ``_paused`` when a host
                    # tool raised PauseRequested. Persist the snapshot,
                    # flip status to paused, release the lock so the
                    # next action=resume can re-acquire under the same
                    # run_id.
                    if isinstance(result, dict) and result.get("_paused"):
                        pause_blob = result.get("pause_state") or {}
                        _write_pause_state(tenant.tenant_id, run_id, pause_blob)
                        status["status"] = "paused"
                        status["paused_at"] = datetime.now(timezone.utc).isoformat()
                        status["updated_at"] = status["paused_at"]
                        status["prompt"] = str(result.get("prompt", ""))
                        status["reason"] = str(result.get("reason", "user_input"))
                        _write_status(tenant.tenant_id, run_id, status)
                        release_profile_lock(tenant, status.get("profile_id", ""))
                    else:
                        # #audit item 4 follow-up: read the honest
                        # terminal_status off the envelope (the Modal
                        # entry surfaces it explicitly now). Map the
                        # internal value to the wire status with the
                        # same rules as ``baseten_server/runtime.py``
                        # (which already had this fix):
                        #   completed              → succeeded (back-compat)
                        #   completed_with_failures → completed_with_failures
                        #   halted / budget_exceeded / time_exceeded → halted
                        #   anything else / missing  → succeeded (defensive)
                        rt_status = ""
                        halt_reason = ""
                        if isinstance(result, dict):
                            rt_status = str(result.get("terminal_status") or "")
                            halt_reason = str(result.get("halt_reason") or "")
                        if rt_status == "completed":
                            wire_status = "succeeded"
                        elif rt_status == "completed_with_failures":
                            wire_status = "completed_with_failures"
                        elif rt_status in (
                            "halted", "budget_exceeded", "time_exceeded",
                        ):
                            wire_status = "halted"
                        else:
                            wire_status = "succeeded"
                        status["status"] = wire_status
                        status["updated_at"] = datetime.now(timezone.utc).isoformat()
                        # Surface the finer detail under explicit keys
                        # alongside the back-compat wire status so
                        # callers that want to distinguish budget vs
                        # time vs step-halt can branch on them.
                        if rt_status:
                            status["terminal_status"] = rt_status
                        if halt_reason:
                            status["halt_reason"] = halt_reason
                        _write_status(tenant.tenant_id, run_id, status)
                        _write_result(
                            tenant.tenant_id,
                            run_id,
                            result if isinstance(result, dict) else {"result": result},
                        )
                        release_profile_lock(tenant, status.get("profile_id", ""))

        # action=status on a paused run: inline pause_state for the caller.
        if action == "status" and status.get("status") == "paused":
            pause_blob = _read_pause_state(tenant.tenant_id, run_id)
            if pause_blob is not None:
                status["pause_state"] = pause_blob

        if action == "result":
            if status.get("status") != "succeeded":
                return {**status, "result": None}
            result = _read_result(tenant.tenant_id, run_id)
            return {**status, "result": result}
        return status

    @fastapi_app.get("/v1/runs/{run_id}/status")
    def get_status(
        run_id: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        return _do_action({"action": "status", "run_id": run_id}, tenant)

    @fastapi_app.get("/v1/runs/{run_id}/result")
    def get_result(
        run_id: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        return _do_action({"action": "result", "run_id": run_id}, tenant)

    @fastapi_app.post("/v1/runs/{run_id}/cancel")
    def cancel_run(
        run_id: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        return _do_action({"action": "cancel", "run_id": run_id}, tenant)

    # ── Lifecycle routes (#806 — PR #792 data layer wiring) ─────────
    #
    # These three routes give clients a cheap "phase + backoff hint"
    # poll surface in addition to the existing detail-heavy
    # ``GET /v1/runs/{id}/status``. Phase is derived from the
    # file-backed ``status.json`` the executor already writes — no
    # in-memory store needed (Modal runs the API + executor in
    # different containers, where a singleton store wouldn't be
    # visible anyway).

    @fastapi_app.get("/v1/runs/{run_id}")
    def get_run_phase(
        run_id: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Cheap phase poll + adaptive backoff hint (#806).

        Returns ``RunPhaseResponse`` derived from status.json. Clients
        SHOULD honor ``polling_backoff_ms_hint`` to stop hammering
        terminal or idle runs. For full detail (per-step results,
        artifacts), use ``GET /v1/runs/{id}/status`` or
        ``GET /v1/runs/{id}/result`` after this returns a terminal phase.
        """
        from mantis_agent.run_lifecycle import build_phase_response_from_status

        try:
            vol.reload()
        except Exception:
            pass
        status = _read_status(tenant.tenant_id, run_id)
        if status is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")
        body = build_phase_response_from_status(status).model_dump()
        # Surface the Augur run id when available so consumers can
        # cross-link to the Augur workspace / bundle (gap #1 of the
        # observability ergonomics audit). Best-effort — missing
        # metadata never breaks the lifecycle response.
        augur_meta = _read_augur_metadata(tenant.tenant_id, run_id)
        if augur_meta and augur_meta.get("augur_run_id"):
            body["augur_run_id"] = augur_meta["augur_run_id"]
            body["augur_bundle_url"] = f"/v1/runs/{run_id}/augur"
        # Surface failure_help on terminal halted / cancelled phases
        # (#841). Prefer the help dict already attached to status.json
        # by the failure path; otherwise synthesize from halt_class so
        # older executor crashes still get an actionable response.
        terminal_failure = body.get("phase") in {"halted", "cancelled"} or (
            body.get("phase") == "complete" and status.get("error")
        )
        if terminal_failure:
            help_dict = status.get("failure_help")
            if not help_dict and status.get("halt_class"):
                from mantis_agent.run_failure_help import failure_help_for
                help_dict = failure_help_for(
                    status.get("halt_class", ""), run_id=run_id,
                )
            if help_dict:
                body["failure_help"] = help_dict
        return body

    @fastapi_app.get("/v1/queue")
    def get_queue(
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Per-tenant queue snapshot (#806).

        Scans the tenant's run directory and counts active phases.
        Terminal runs are excluded — operators wanting historical
        totals can grep ``status.json`` files directly.
        """
        from mantis_agent.run_lifecycle import (
            QueueStatusResponse,
            RunPhase,
            phase_from_status_string,
        )

        try:
            vol.reload()
        except Exception:
            pass
        from pathlib import Path as _Path

        from mantis_agent.server_utils import safe_state_key as _safe

        queued = running = recovering = 0
        root = _Path(os.environ.get("MANTIS_DATA_DIR", "/data"))
        tenant_dir = root / "tenants" / _safe(tenant.tenant_id) / "runs"
        if tenant_dir.exists() and tenant_dir.is_dir():
            for run_subdir in tenant_dir.iterdir():
                if not run_subdir.is_dir():
                    continue
                status_path = run_subdir / "status.json"
                if not status_path.exists():
                    continue
                try:
                    status_blob = json.loads(status_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                phase = phase_from_status_string(str(status_blob.get("status", "") or ""))
                if phase is RunPhase.QUEUED:
                    queued += 1
                elif phase is RunPhase.RUNNING:
                    running += 1
                elif phase is RunPhase.RECOVERING:
                    recovering += 1
        return QueueStatusResponse(
            tenant_id=tenant.tenant_id,
            queued=queued,
            running=running,
            recovering=recovering,
            eta_ms=None,
        ).model_dump()

    # #508 artifact download. Allowlisted filenames so we never stream
    # arbitrary files from the run dir; path-traversal guard via
    # Path.resolve() to block ``..`` even if a future allowlist entry
    # carries a slash. Mirrors the Baseten-side handler in
    # ``baseten_server/routes.py``.
    _ARTIFACT_ALLOWLIST = {
        "leads.csv": "text/csv",
        "extracted_rows.csv": "text/csv",
        "extracted_rows.json": "application/json",
        "result.json": "application/json",
    }

    @fastapi_app.get("/v1/runs/{run_id}/artifacts/{name}")
    def get_run_artifact(
        run_id: str,
        name: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ):
        from fastapi.responses import FileResponse

        media_type = _ARTIFACT_ALLOWLIST.get(name)
        if media_type is None:
            raise HTTPException(status_code=404, detail=f"unknown artifact: {name}")
        run_dir = _run_dir(tenant.tenant_id, run_id).resolve()
        candidate = (run_dir / name).resolve()
        try:
            candidate.relative_to(run_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid artifact name")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(
                status_code=404, detail=f"artifact not available: {name}",
            )
        return FileResponse(candidate, media_type=media_type, filename=name)

    # ── Augur bundle access (observability ergonomics gap #2) ──────
    #
    # The runner writes the per-run Augur DebugSession bundle under
    # ``/data/augur/<augur_run_id>/`` — separate from the API run dir.
    # These routes let operators fetch the bundle by Mantis run_id
    # without needing the Augur SDK workspace.
    #
    # Path-traversal guard via ``Path.resolve() + relative_to``. The
    # allowlist of fetchable files is intentionally narrow — we
    # don't stream the per-step PNG screenshots (heavyweight, the
    # live viewer covers that lane).

    _AUGUR_BUNDLE_FILE_TYPES: dict[str, str] = {
        ".json": "application/json",
        ".jsonl": "application/jsonl",
        ".png": "image/png",
    }

    @fastapi_app.get("/v1/runs/{run_id}/augur")
    def get_augur_envelope(
        run_id: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Return the Augur metadata envelope for this run.

        Includes the ``augur_run_id``, the on-disk bundle directory,
        a list of fetchable files, and (when configured) the workspace
        URL the live stream targets.
        """
        from pathlib import Path as _Path

        try:
            vol.reload()
        except Exception:
            pass
        meta = _read_augur_metadata(tenant.tenant_id, run_id)
        if meta is None:
            raise HTTPException(
                404,
                f"no Augur metadata for run_id={run_id} — either the run "
                "hasn't started or Augur is disabled",
            )
        bundle = _Path(meta.get("bundle_dir", ""))
        files: list[dict] = []
        if bundle.exists() and bundle.is_dir():
            for path in sorted(bundle.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(bundle)
                files.append({
                    "name": str(rel),
                    "size_bytes": path.stat().st_size,
                    "fetch_url": f"/v1/runs/{run_id}/augur/files/{rel}",
                })
        return {
            "run_id": run_id,
            "augur_run_id": meta.get("augur_run_id", ""),
            "bundle_dir": meta.get("bundle_dir", ""),
            "dsn_workspace": meta.get("dsn_workspace", ""),
            "bundle_present": bool(files),
            "files": files,
        }

    @fastapi_app.get("/v1/runs/{run_id}/augur/files/{path:path}")
    def get_augur_file(
        run_id: str,
        path: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ):
        """Stream a specific file from the Augur bundle.

        Path-traversal guard: the resolved path must live under the
        bundle directory. File extension must be in the allowlist
        (``.json``, ``.jsonl``, ``.png``). Returns 404 for missing
        files; 400 for traversal attempts or disallowed extensions.
        """
        from pathlib import Path as _Path

        from fastapi.responses import FileResponse

        try:
            vol.reload()
        except Exception:
            pass
        meta = _read_augur_metadata(tenant.tenant_id, run_id)
        if meta is None:
            raise HTTPException(404, f"no Augur metadata for run_id={run_id}")
        bundle = _Path(meta.get("bundle_dir", "")).resolve()
        if not bundle.exists():
            raise HTTPException(404, "augur bundle not present on volume")
        candidate = (bundle / path).resolve()
        try:
            candidate.relative_to(bundle)
        except ValueError:
            raise HTTPException(400, "invalid bundle path")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, f"file not in bundle: {path}")
        ext = candidate.suffix.lower()
        media_type = _AUGUR_BUNDLE_FILE_TYPES.get(ext)
        if media_type is None:
            raise HTTPException(400, f"file type not exposed via HTTP: {ext}")
        return FileResponse(
            candidate, media_type=media_type, filename=candidate.name,
        )

    # ── SSE event stream (#808) ────────────────────────────────────
    #
    # Server-Sent Events wrapper over the file-backed reasoning.jsonl
    # the executor already writes. Reuses the same per-line JSON shape
    # as ``action=reasoning_trace`` so consumers can re-use their
    # existing event parsers. Adds:
    #
    # - ``phase`` events on every transition derived from status.json
    # - ``terminal`` event when the run reaches a terminal phase
    # - Heartbeat ``: ping\n\n`` every ~25s so reverse proxies don't
    #   drop the connection (most close idle conns at 30-60s)
    # - ``Last-Event-ID`` honored if the client supplies one; otherwise
    #   the ``since`` query param works
    #
    # Reads are tail-only (no exclusive lock) so concurrent consumers
    # don't fight each other or the writer.

    @fastapi_app.get("/v1/runs/{run_id}/events")
    def stream_run_events(
        run_id: str,
        request: Request,
        sse: bool = False,
        since: str = "",
        tenant: TenantConfig = Depends(require_run_scope),
    ):
        """Stream run events as SSE (#808).

        With ``?sse=true`` returns ``text/event-stream``. Without it,
        falls back to the same JSON payload as
        ``POST /v1/predict {action: reasoning_trace}`` for parity.
        """
        from fastapi.responses import StreamingResponse

        try:
            vol.reload()
        except Exception:
            pass
        status = _read_status(tenant.tenant_id, run_id)
        if status is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")

        jsonl_path = _run_dir(tenant.tenant_id, run_id) / "reasoning.jsonl"

        if not sse:
            # Non-SSE fallback: same shape as action=reasoning_trace.
            since_ts = since or None
            events: list[dict] = []
            if jsonl_path.exists():
                try:
                    with jsonl_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            try:
                                ev = json.loads(stripped)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            if not isinstance(ev, dict):
                                continue
                            if since_ts and ev.get("ts", "") <= since_ts:
                                continue
                            events.append(ev)
                except OSError:
                    pass
            return {**status, "events": events, "count": len(events)}

        # SSE path. Honor Last-Event-ID per the SSE spec; fall through
        # to ``?since`` if the header is absent.
        last_event_id = request.headers.get("last-event-id", "") or since
        last_phase_seen = ""

        def _sse_format(event_name: str, payload: dict, event_id: str = "") -> str:
            parts = []
            if event_id:
                parts.append(f"id: {event_id}")
            parts.append(f"event: {event_name}")
            parts.append(f"data: {json.dumps(payload, default=str)}")
            return "\n".join(parts) + "\n\n"

        # Local phase mapping — kept inline so this PR doesn't depend on
        # the lifecycle-routes branch landing the shared helper. When
        # both PRs merge, this collapses to a one-line call.
        _STATUS_TO_PHASE = {
            "queued": "queued",
            "running": "running",
            "paused": "running",
            "recovering": "recovering",
            "cancelled": "cancelled",
            "succeeded": "complete",
            "completed_with_failures": "complete",
            "failed": "halted",
            "halted": "halted",
            "timeout": "halted",
        }
        _TERMINAL_PHASES = {"complete", "halted", "cancelled"}

        def _current_phase(status_blob: dict) -> str:
            s = str(status_blob.get("status", "") or "").lower()
            return _STATUS_TO_PHASE.get(s, "running")

        def _terminal_phase(status_blob: dict) -> str | None:
            p = _current_phase(status_blob)
            return p if p in _TERMINAL_PHASES else None

        async def event_stream():
            import asyncio

            nonlocal last_event_id, last_phase_seen

            # Stream guardrails. Bounded so a stuck client doesn't keep
            # a container input slot forever; clients reconnect with
            # ``Last-Event-ID`` to resume.
            max_stream_seconds = 600
            poll_interval = 1.0
            heartbeat_interval = 25.0
            t_start = time.monotonic()
            t_last_heartbeat = t_start

            # Emit initial phase event so clients have the current
            # ground state immediately.
            cur_phase = _current_phase(status)
            yield _sse_format("phase", {"phase": cur_phase, "run_id": run_id})
            last_phase_seen = cur_phase

            while True:
                if await request.is_disconnected():
                    return
                # Refresh volume so writes from the executor container
                # are visible.
                try:
                    vol.reload()
                except Exception:
                    pass

                # Drain new reasoning events.
                if jsonl_path.exists():
                    try:
                        with jsonl_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                stripped = line.strip()
                                if not stripped:
                                    continue
                                try:
                                    ev = json.loads(stripped)
                                except (json.JSONDecodeError, ValueError):
                                    continue
                                if not isinstance(ev, dict):
                                    continue
                                ts = str(ev.get("ts", "") or "")
                                if last_event_id and ts <= last_event_id:
                                    continue
                                # Use the event's own ``kind`` field as
                                # the SSE event name when present;
                                # ``message`` is the safe default the
                                # SSE spec mandates for unspecified events.
                                event_name = str(ev.get("kind") or ev.get("type") or "message")
                                yield _sse_format(event_name, ev, event_id=ts)
                                if ts:
                                    last_event_id = ts
                    except OSError:
                        pass

                # Re-read status for phase transitions + terminal check.
                cur_status = _read_status(tenant.tenant_id, run_id) or status
                phase_now = _current_phase(cur_status)
                if phase_now != last_phase_seen:
                    yield _sse_format(
                        "phase",
                        {"phase": phase_now, "run_id": run_id},
                    )
                    last_phase_seen = phase_now

                terminal = _terminal_phase(cur_status)
                if terminal is not None:
                    yield _sse_format(
                        "terminal",
                        {
                            "phase": terminal,
                            "run_id": run_id,
                            "halt_class": cur_status.get("halt_class")
                            or cur_status.get("halt_reason"),
                        },
                    )
                    return

                # Heartbeat to keep proxies happy.
                t_now = time.monotonic()
                if (t_now - t_last_heartbeat) >= heartbeat_interval:
                    yield ": ping\n\n"
                    t_last_heartbeat = t_now

                if (t_now - t_start) >= max_stream_seconds:
                    # Clean close without a terminal event — client
                    # reconnects with Last-Event-ID and we pick up.
                    return

                await asyncio.sleep(poll_interval)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",  # nginx: don't buffer
                "Connection": "keep-alive",
            },
        )

    # ── Runtime recipe registration (#809) ────────────────────────
    #
    # Tenant-scoped CRUD over /data/tenants/<tenant>/recipes/. Lets
    # integrators register an ExtractionSchema by name over HTTP
    # instead of forking + redeploying for every domain. The recipe
    # loader (mantis_agent.recipes.load_schema) consults the tenant's
    # runtime dir first when ``tenant_id`` is supplied, then falls
    # back to the code-shipped recipes.

    @fastapi_app.post("/v1/recipes")
    def register_recipe(
        body: dict,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Register or overwrite a tenant runtime recipe (#809)."""
        from mantis_agent.recipes import runtime_store

        try:
            persisted = runtime_store.register(
                tenant.tenant_id,
                str(body.get("name") or ""),
                body.get("schema") or {},
            )
        except runtime_store.RuntimeRecipeError as exc:
            raise HTTPException(400, str(exc)) from exc
        _commit_volume()
        return persisted

    @fastapi_app.get("/v1/recipes")
    def list_recipes(
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """List runtime recipes registered under the caller's tenant."""
        from mantis_agent.recipes import runtime_store

        try:
            vol.reload()
        except Exception:
            pass
        return {"recipes": runtime_store.list_recipes(tenant.tenant_id)}

    @fastapi_app.get("/v1/recipes/{name}")
    def get_recipe(
        name: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Fetch a registered runtime recipe by name."""
        from mantis_agent.recipes import runtime_store

        try:
            vol.reload()
        except Exception:
            pass
        try:
            body = runtime_store.get(tenant.tenant_id, name)
        except runtime_store.RuntimeRecipeError as exc:
            raise HTTPException(400, str(exc)) from exc
        if body is None:
            raise HTTPException(404, f"unknown recipe: {name}")
        return body

    @fastapi_app.delete("/v1/recipes/{name}")
    def delete_recipe(
        name: str,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Delete a tenant runtime recipe. Idempotent."""
        from mantis_agent.recipes import runtime_store

        try:
            deleted = runtime_store.delete(tenant.tenant_id, name)
        except runtime_store.RuntimeRecipeError as exc:
            raise HTTPException(400, str(exc)) from exc
        if deleted:
            _commit_volume()
        return {"name": name, "deleted": deleted}

    # ── Fingerprint diagnostic (#827) ──────────────────────────────
    #
    # Synthesizes + submits a fingerprint-test plan against a public
    # bot-detection diagnostic page (``bot.sannysoft.com`` by default;
    # configurable via the request body). The plan navigates to the
    # test page, waits for the JS to populate the results table, then
    # ``extract_data`` reads each row's pass/fail signal. Returns
    # ``{run_id, target_url, poll_via}`` immediately — the operator
    # polls the run via the lifecycle endpoints to read the scorecard.
    #
    # Use this to verify the stealth posture before/after a config
    # change: flip ``MANTIS_STEALTH_HONEST=1`` or
    # ``MANTIS_BEHAVIORAL_JITTER=1``, redeploy, run /diagnose/
    # fingerprint, compare the row counts in extracted_rows.json.

    @fastapi_app.post("/v1/diagnose/fingerprint")
    def diagnose_fingerprint(
        body: dict | None = None,
        tenant: TenantConfig = Depends(require_run_scope),
    ) -> dict:
        """Submit a bot-detection diagnostic run (#827).

        Optional body:
            {"target_url": "https://bot.sannysoft.com/",
             "cua_model":  "holo3"}

        Returns the standard detached-run envelope; poll via
        ``GET /v1/runs/{run_id}`` and read the rows via
        ``GET /v1/runs/{run_id}/artifacts/extracted_rows.json``.
        """
        from mantis_agent.server.run_dispatch import (
            DispatchError,
            acquire_profile_lock,
            release_profile_lock,
        )

        body = body or {}
        target_url = str(body.get("target_url") or "https://bot.sannysoft.com/").strip()
        if not target_url.startswith(("http://", "https://")):
            raise HTTPException(400, "target_url must be http(s)://")
        model = str(body.get("cua_model") or "holo3").strip() or "holo3"
        if model not in {"holo3", "claude"}:
            raise HTTPException(400, "cua_model must be holo3 or claude")

        plan = {
            "_micro_plan": [
                {
                    "intent": f"Navigate to {target_url} and wait for the fingerprint test to complete",
                    "type": "navigate",
                    "params": {"url": target_url, "wait_after_load_seconds": 8},
                    "section": "setup", "required": True, "budget": 4,
                },
                {
                    "intent": (
                        "Extract each fingerprint test row visible on the page. Each "
                        "row has a test name (e.g. 'navigator.webdriver', 'WebGL "
                        "Vendor') and a status (Pass / Fail / value). Return one row "
                        "per visible test."
                    ),
                    "type": "extract_data",
                    "params": {"claude_only": True},
                    "section": "extraction", "required": False, "budget": 0,
                    "claude_only": True, "hints": {"layout": "listings"},
                    "extract": {
                        "schema_name": "fingerprint_diagnostic",
                        "entity_name": "fingerprint_test",
                        "fields": [
                            {"name": "test_name", "type": "str", "required": True},
                            {"name": "result", "type": "str", "required": True},
                        ],
                        "max_items": 60,
                    },
                },
            ],
        }

        try:
            task_file_contents = _build_suite_from_payload({
                "task_suite": plan,
                "cua_model": model,
                "profile_id": f"fp-diag-{int(time.time())}",
                "workflow_id": f"fp-diag-{int(time.time())}",
                "max_cost": 0.30,
                "max_time_minutes": 3,
            })
        except (ValueError, FileNotFoundError, DispatchError) as exc:
            raise HTTPException(400, str(exc)) from exc

        profile_id = f"fp-diag-{int(time.time())}"
        run_id = _mint_run_id()
        if not acquire_profile_lock(tenant, profile_id, run_id):
            raise HTTPException(409, "profile lock busy — retry in a moment")

        executor_fn = resolve_executor(model)
        spawn_kwargs: dict = {
            "max_steps": 8,
            "profile_dir": _chrome_profile_dir(tenant.tenant_id, profile_id),
        }
        if model != "claude":
            spawn_kwargs["cua_model"] = model
        try:
            call_handle = executor_fn.spawn(
                task_file_contents=task_file_contents,
                **spawn_kwargs,
            )
        except Exception as exc:
            release_profile_lock(tenant, profile_id)
            raise HTTPException(500, f"executor spawn failed: {exc}") from exc

        now = datetime.now(timezone.utc).isoformat()
        status = {
            "run_id": run_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "modal_call_id": getattr(call_handle, "object_id", "") or "",
            "tenant_id": tenant.tenant_id,
            "profile_id": profile_id,
            "workflow_id": profile_id,
            "state_key": profile_id,
            "model": model,
            "max_steps": 8,
            "diagnostic_kind": "fingerprint",
            "diagnostic_target": target_url,
        }
        _write_status(tenant.tenant_id, run_id, status)
        _write_task_suite(tenant.tenant_id, run_id, task_file_contents)
        return {
            "run_id": run_id,
            "target_url": target_url,
            "poll_via": f"/v1/runs/{run_id}",
            "rows_via": f"/v1/runs/{run_id}/artifacts/extracted_rows.json",
            "status": "queued",
            "stealth_snapshot": {
                "honest_mode": os.environ.get("MANTIS_STEALTH_HONEST", "1") not in {"0", "false"},
                "behavioral_jitter": os.environ.get("MANTIS_BEHAVIORAL_JITTER", "1") not in {"0", "false"},
                "geo_consistency": os.environ.get("MANTIS_GEO_CONSISTENCY", "1") not in {"0", "false"},
                "cdp_stealth": os.environ.get("MANTIS_CDP_STEALTH", "1") not in {"0", "false"},
                "proxy_provider": os.environ.get("MANTIS_PROXY_PROVIDER", "privateproxy"),
            },
        }

    return fastapi_app


@app.function(
    image=api_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=300,
    memory=2048,
    cpu=2,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def api():
    """Mantis CUA HTTP API on Modal (#342).

    Mirrors the Baseten ``/v1/predict`` shape: caller submits a plan with
    ``detached: true`` (default), gets back ``{run_id, status, ...}``,
    polls via ``action=status|result|cancel`` with the same ``run_id``.

    Concurrency: ``@modal.concurrent(max_inputs=64)`` lets one ASGI
    container service many concurrent submissions; each spawned
    executor opens its own dedicated container (1 input per container,
    matching the pre-existing executor decorators).

    Per-profile lock: two concurrent runs against the same
    ``profile_id`` get a 409 with the conflicting ``run_id`` — Chrome
    cannot share a user-data-dir between processes (#341 motivation,
    #342 enforcement).
    """
    return build_api_app()


# ═══════════════════════════════════════════════════════════════════
# D2) Computer Plane (#698, Phase 1) — Xvfb + Chrome + xdotool behind
#     a thin FastAPI service. Optional, off by default; brain executors
#     still construct ``LocalXdotoolImpl`` in-process until a per-
#     executor override flips them to ``ComputerPlaneConfig.backend=
#     "modal"`` pointing at this function's web URL.
# ═══════════════════════════════════════════════════════════════════

# Reuses the run_holo3 image's apt layer (Xvfb + xdotool + Chrome + the
# stealth fonts/locale/TZ). The brain image deliberately keeps these
# layers too in Phase 1 — Phase 2 of the migration slims the brain
# image once the computer plane is proven in production.
computer_plane_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get install -y gnupg curl wget xvfb xdotool xclip scrot "
        "fonts-liberation fonts-dejavu-core fonts-noto-color-emoji locales",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y google-chrome-stable || true",
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime",
    )
    .env({
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .pip_install(
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2",
        "pillow",
        "mss",
        "requests",
        "websocket-client",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE)
    .add_local_dir(_WEBGL_SPOOF_LOCAL, remote_path=_WEBGL_SPOOF_REMOTE)
)


@app.function(
    image=computer_plane_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours — match executor timeouts
    memory=8192,
    cpu=4.0,
    scaledown_window=600,
    # The ASGI app holds the active session in process-local state
    # (`ComputerAgentState._state.session`). With more than one
    # replica, the brain's second request fans out to a different
    # container which has no session and 401s — the failure mode the
    # prior holo3 smoke surfaced. Phase 2 (#699) lifts this via
    # per-session `.spawn()`; for Phase 1 we hard-pin to a single
    # container and turn up per-container concurrency so the brain's
    # parallel screenshot + xdotool calls still don't queue serially.
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def computer_plane():
    """Computer Plane RPC server — Xvfb + Chrome + xdotool over HTTPS.

    Exposes the wire contract defined in
    ``mantis_agent.gym.computer_wire``:

      * POST /session/init      — bind tenant/profile/run, launch Xvfb + Chrome
      * POST /session/close     — SIGTERM Chrome, stop Xvfb
      * POST /screenshot        — base64 PNG + viewport metadata
      * POST /xdotool           — argv with step_id (LRU dedup, TTL=30s)
      * POST /cdp               — opt-in, off by default
      * POST /cdp_click_at_point — SoM-anchored click via CDP (opt-in)
      * GET  /current_url       — active tab URL
      * GET  /cdp_count_pages   — open Chrome tabs (page-type)
      * GET  /health            — liveness + last-action timestamp

    Phase 1 rollout (#698): brain executors call this via
    ``RemoteComputerImpl`` once ``ComputerPlaneConfig.backend='modal'``
    is set on the executor (with ``remote_base_url`` pointing at this
    function's web URL — resolve via
    ``modal.Function.from_name('mantis-cua-server', 'computer_plane').get_web_url()``).
    The Claude executor flip is the first migration target; flip via
    `modal secret create --from-dotenv` after setting
    ``MANTIS_COMPUTER_PLANE_BACKEND=modal`` in the dotenv. Rollback by
    re-pushing with ``local``.
    """
    from mantis_agent.server.computer_agent import build_app
    return build_app()


@app.function(
    gpu="A100-80GB",
    image=planner_base_image.run_commands(
        # #stealth-parity: install fonts/locale alongside the existing
        # Chrome + Xvfb deps. ``run_holo3`` is the most-used executor
        # tier in production (`cua_model=holo3` is the default for the
        # `task_suite` HTTP path) — without these fonts present here
        # the in-prod Chrome still rendered with the sparse Linux-server
        # font set even after the executor_image fix.
        #
        # ``tzdata`` prompts for Geographic area interactively under
        # ``run_commands`` (Modal's ``apt_install`` injects
        # DEBIAN_FRONTEND=noninteractive but raw run_commands does
        # not). Pre-seed the answer via TZ env + DEBIAN_FRONTEND
        # prefix so the install completes unattended.
        "DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get install -y gnupg curl wget xvfb xdotool xclip scrot "
        "fonts-liberation fonts-dejavu-core fonts-noto-color-emoji locales",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y google-chrome-stable || true",
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime",
    ).env({
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
        # Keep DEBIAN_FRONTEND set for any downstream pip / apt that
        # may run during further image steps.
        "DEBIAN_FRONTEND": "noninteractive",
    }).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20", "websocket-client",
        # #509: run_holo3 uses its own inline image (NOT executor_image),
        # so augur-sdk has to be added here separately. Without this the
        # AugurAdapter init logs sdk_available=False and is a no-op even
        # though the package is in executor_image for the other tiers.
        # Must match pyproject.toml (``augur-sdk>=0.6.0,<0.7``). 0.2.x
        # added ``branch_context`` to ``DebugSession.__init__``; 0.4.0
        # (mercurialsolo/augur-sdk#38) added
        # ``DebugSession.open_orchestrator(...)``; 0.6.0 (#680) adds
        # ``task_spec`` / ``group_id`` / ``set_loop_detected`` /
        # ``record_subgoal_completion`` and the ``set_score`` kwargs
        # ``should_stop`` / ``uncertainty``. Stale <0.6 pins silently
        # drop the new fields — adapter swallows the TypeError and the
        # bundle ships without RL-training metadata.
        "augur-sdk>=0.6.0,<0.7",
    ).add_local_python_source("mantis_agent").add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE).add_local_dir(_WEBGL_SPOOF_LOCAL, remote_path=_WEBGL_SPOOF_REMOTE),
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


@app.function(
    gpu="A100-40GB",
    image=executor_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=14400,  # 4 hours
    memory=65536,
    cpu=16,
)
def run_fara(task_file_contents: str, **kwargs) -> dict:
    """Fara-7B executor (1x A100-40GB, native vLLM, OpenAI-compatible).

    Microsoft's Qwen2.5-VL-based 7B CUA model. Routes through the shared
    vLLM ``_run_executor`` path; the brain dispatch picks ``FaraBrain``
    when ``cua_model="fara"``.
    """
    kwargs.pop("cua_model", None)
    return _run_executor(task_file_contents, cua_model="fara", **kwargs)


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
    "fara": run_fara,
    "gemma4-cua": run_gemma4_cua,
    "claude": run_claude_cua,
}

APP_NAME = "mantis-cua-server"


def _gemma4_planner_url() -> str:
    """Resolve the deployed planner URL for the current Modal workspace."""
    override = os.environ.get("MANTIS_GEMMA4_PLANNER_URL", "").strip().rstrip("/")
    if override:
        return override
    try:
        return modal.Function.from_name(APP_NAME, "gemma4_planner").get_web_url()
    except Exception as exc:
        print(f"  WARNING: Failed to resolve planner URL via Modal SDK: {exc}")
        return f"https://{APP_NAME}--gemma4-planner.modal.run"


# ═══════════════════════════════════════════════════════════════════
# E) Parallel extraction — fan-out across N workers (#617)
# ═══════════════════════════════════════════════════════════════════
#
# Per-page partition tasks are built by
# ``mantis_agent.gym.fanout_runner.prepare_modal_partitions`` from the
# plan's ``MicroPlan.loop_groups`` (#614). The legacy
# ``_make_page_task`` helper — which hardcoded the BoatTrader
# ``{base}/page-{n}/`` URL pattern and the ``task.loop`` schema — was
# removed in #618. Pagination URL synthesis now flows through
# ``fanout_runner.partition_urls_for_pagination`` with the per-plan
# ``paginate.params['url_template']`` override.


@app.function(
    gpu="A100-80GB",
    image=planner_base_image.run_commands(
        # #stealth-parity: same fonts+locale+TZ as run_holo3's image.
        # See run_holo3 image for the DEBIAN_FRONTEND/TZ rationale —
        # tzdata prompts interactively in run_commands otherwise.
        "DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get update && DEBIAN_FRONTEND=noninteractive TZ=America/New_York "
        "apt-get install -y gnupg curl wget xvfb xdotool xclip scrot "
        "fonts-liberation fonts-dejavu-core fonts-noto-color-emoji locales",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y google-chrome-stable || true",
        "sed -i 's/# en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && locale-gen",
        "ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime",
    ).env({
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TZ": "America/New_York",
        "DEBIAN_FRONTEND": "noninteractive",
    }).pip_install(
        "openai", "requests", "pillow", "mss",
        "fastapi>=0.100", "uvicorn>=0.20", "websocket-client",
    ).add_local_python_source("mantis_agent").add_local_dir(_PROMPTS_FILES_LOCAL, remote_path=_PROMPTS_FILES_REMOTE).add_local_dir(_WEBGL_SPOOF_LOCAL, remote_path=_WEBGL_SPOOF_REMOTE),
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


def _print_shared_seen_metrics(
    task_suite: dict, cumulative_hits: int,
) -> None:
    """Print the cross-worker shared-seen aggregate (#631 follow-up).

    Two signals operators need after a fan-out run:

      * ``cumulative_hits`` — number of times a worker short-circuited
        an extract because a sibling worker had already extracted the
        URL. Each hit avoided the ~$0.20 Claude extract cost.
      * ``final dict size`` — total unique URLs the fan-out has seen
        across all workers. Queried from the Modal Dict by name; this
        is the ground-truth unique-URL count without the per-worker
        log archaeology Modal makes hard on stopped containers.

    Print-only — no exception escapes, never breaks the fan-out
    summary even when Modal Dict isn't reachable.
    """
    dict_name = task_suite.get("_fanout_seen_dict_name", "")
    final_size = 0
    if dict_name:
        try:
            import modal as _modal
            _shared = _modal.Dict.from_name(dict_name)
            # modal.Dict doesn't implement __len__; use .len() method
            # (returns int) and fall back to counting keys() on older
            # SDKs that may not expose it.
            if hasattr(_shared, "len"):
                final_size = int(_shared.len())
            else:
                final_size = sum(1 for _ in _shared.keys())
        except Exception as exc:
            print(
                f"  [shared-seen] could not query final dict size "
                f"({dict_name}): {exc}"
            )
    estimated_savings = cumulative_hits * 0.20  # ~$0.20 per skipped Claude extract
    print(
        f"  [shared-seen] cumulative cross-worker hits: {cumulative_hits} "
        f"(~${estimated_savings:.2f} avoided extract cost)"
    )
    if dict_name:
        print(f"  [shared-seen] final dict size: {final_size} unique URLs")


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
    claude_model: str = "claude-sonnet-4-6",
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
    profile_id: str = "",
    workflow_id: str = "",
    graph_learn: bool = False,
    graph_learn_only: bool = False,
    proxy_provider: str = "oxylabs",
    proxy_city: str = "miami",
    proxy_state: str = "florida",
    disable_proxy: bool = False,
):
    """Mantis CUA Server — run plans or task suites on Modal.

    Modes:
      --task-file tasks/boattrader/dynamic.json     (direct execution)
      --plan-file plans/example/full_spec.txt    (Gemma4 preprocesses → execute)
      --learn --task-file tasks/...                 (learning phase: build playbook)
      --verify --task-file tasks/...                (execution with step verification)
      --micro plans/example/extract_only.txt     (micro-intent decompose + execute)
      --graph-learn --micro plan.txt                (probe site + generate dependency graph + execute)
      --graph-learn-only --micro plan.txt           (probe site + generate graph, no execution)

    Models: evocua-8b, evocua-32b, opencua-32b, opencua-72b, holo3, fara, gemma4-cua, claude
    Parallel: --workers 5   (auto fan-out looped tasks across N GPUs)
    Claude options: --claude-model claude-sonnet-4-6 --thinking-budget 2048
    Viewer: --viewer   (live web viewer via modal.forward tunnel)
    Learning: --learn --learn-samples 5   (build site playbook from N samples)
    Verification: --verify   (enable step verification during execution)
    Micro: --micro plan.txt   (decompose → micro-intents → execute with checkpoint/reverse)
    Resume: --resume-state --workflow-id my-run   (reuse externalized micro state across sessions)
            --profile-id alice-prod  (Chrome user-data-dir identity, sticky across plan revisions, #341)
    Graph: --graph-learn   (probe + graph + compile + execute) --graph-learn-only (no execution)
    Proxy: --proxy-provider privateproxy|oxylabs|iproyal --proxy-city miami --proxy-state florida
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

        planner_url = _gemma4_planner_url()
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
        # optimize_plan is scoped to the marketplace_listings recipe
        # (#462). The `--plan-file` Modal path is BoatTrader-shaped today;
        # generic plans should go through `--micro` instead.
        task_suite = optimize_plan(
            plan_text=plan_text, inputs=plan_inputs,
            session_name=session_name, max_listings=max_listings,
            brain=planner_brain,
            recipe_name="marketplace_listings",
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
                print("\n  Graph saved. Use --graph-learn (without --only) to also execute.")
                print(json.dumps({"mode": "graph_learn", "phases": len(graph.phases), "domain": graph.domain}, indent=2))
                return

            # Compile graph to MicroPlan
            compiler = GraphCompiler()
            micro_plan = compiler.compile(graph)
            print(f"\n  Compiled: {len(micro_plan.steps)} micro-intents from graph")

        elif micro.endswith(".json"):
            print(f"  Mode:    Micro-Intent → {cua_config['name']}")
            # Load pre-built micro-plan JSON directly (no decomposition).
            # Accept either a raw steps list OR a dict with a ``steps`` key
            # — the latter is what plan files (e.g. ``plans/boattrader_*``)
            # carry alongside ``shapes`` metadata.
            with open(micro) as f:
                raw = json.load(f)
            raw_steps = raw["steps"] if isinstance(raw, dict) and "steps" in raw else raw
            # #638 axis 2 follow-up: domain mirrors the file stem so
            # Augur tags expose the same human-readable identifier as
            # the suite-level _plan_name (set on the suite below). The
            # legacy hardcode ``"direct_json"`` made every JSON-micro
            # run look identical in the Runs list regardless of plan.
            from pathlib import Path as _Path
            micro_plan = MicroPlan(domain=_Path(micro).stem)
            for s in raw_steps:
                micro_plan.steps.append(PlanDecomposer._build_intent(s))
            # Hand-authored / cached step lists frequently omit
            # ``loop_target`` (issue #605) — the orchestrator's fan-out
            # classifier then sees self-pointing loops and falls back to
            # ``sequential``, defeating the parallelizable_url_collect /
            # parallelizable_pagination path entirely. Run the same
            # normalization + classification that ``decompose()`` runs so
            # ``--micro <file.json> --workers N`` actually fans out.
            PlanDecomposer._fix_loop_targets(micro_plan)
            PlanDecomposer._classify_loop_groups(micro_plan)
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
        # #617: serialize plan-level loop classifications so the Modal
        # fan-out orchestrator can route parallelizable loops without
        # re-running the classifier on the deserialised plan in-container.
        loop_groups_dicts = [
            {
                "loop_step_idx": g.loop_step_idx,
                "body_range": list(g.body_range),
                "shape": g.shape,
            }
            for g in micro_plan.loop_groups
        ]
        task_suite = build_micro_suite(
            steps_dicts,
            micro_plan.domain,
            max_cost=max_cost,
            max_time_minutes=max_time_minutes,
            resume_state=resume_state,
            state_key=state_key,
            profile_id=profile_id,
            workflow_id=workflow_id,
            objective=objective_dict,
            loop_groups=loop_groups_dicts,
            # #629: thread the plan-level pagination URL template
            # through to the orchestrator. Empty when the decomposer
            # didn't infer one (which is most plans today).
            pagination_url_template=getattr(
                micro_plan, "pagination_url_template", "",
            ) or "",
        )
        # #638 axis 2 follow-up: stable human-readable plan identifier
        # for Augur grouping. ``micro_plan.domain`` is already set from
        # the source file stem (JSON-micro / freetext path) or
        # ``decomposer.decompose(micro)`` returns it. Stamping on the
        # suite makes it explicit for the executor + survives the
        # fan-out partition rewrite (each partition inherits via
        # ``dict(suite_dict)`` in prepare_phase1/2_suite).
        task_suite["_plan_name"] = str(micro_plan.domain or "")

        print(f"  Profile:  {task_suite['_profile_id']}")
        print(f"  Workflow: {task_suite['_workflow_id']}")
        print(f"  Resume:   {'on' if resume_state else 'off'}")

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

    if disable_proxy:
        print("\n  Proxy disabled for this run")
        task_suite_obj = json.loads(task_file_contents)
        task_suite_obj["_proxy_disabled"] = True
        task_file_contents = json.dumps(task_suite_obj)
    elif proxy_provider:
        print(f"\n  Proxy provider: {proxy_provider}")
        task_suite_obj = json.loads(task_file_contents)
        task_suite_obj["_proxy_provider"] = proxy_provider
        task_suite_obj["_proxy_city"] = proxy_city
        task_suite_obj["_proxy_state"] = proxy_state
        task_file_contents = json.dumps(task_suite_obj)

    # ── #617: generic fan-out via plan-level loop_groups ──────────────
    # Preferred path when the submitted micro_plan carries
    # parallelizable_pagination loop_groups (set by PlanDecomposer's
    # #614 classifier). Drops the BoatTrader-specific `_make_page_task`
    # hardcode in favor of MicroPlan.loop_groups + the partition
    # synthesizer in mantis_agent.gym.fanout_runner.
    task_suite = json.loads(task_file_contents)
    if workers > 1 and task_suite.get("_micro_plan"):
        # #673: Phase-1/Phase-2 spawn primitives are now consumed
        # internally by ``run_fanout_dispatch`` (imported at the call
        # site below). CLI still uses ``find_url_collect_group`` to
        # gate and ``prepare_modal_partitions`` for the #617
        # pagination-partition fallback.
        from mantis_agent.gym.fanout_runner import (
            find_url_collect_group, prepare_modal_partitions,
        )

        # #627: create a per-run shared seen-URL set keyed by a unique
        # dict name. The name lands on every spawned sub-suite via
        # ``_fanout_seen_dict_name`` so workers attach to the same
        # modal.Dict on their side. Modal Dicts created with
        # ``create_if_missing=True`` are GC'd by Modal after a TTL —
        # we don't explicitly delete here.
        import uuid as _uuid
        shared_dict_name = (
            f"mantis-fanout-seen-{_uuid.uuid4().hex[:12]}"
        )
        # Pre-create the dict so spawn-time attach is guaranteed to
        # find it (the worker's ``modal.Dict.from_name(name,
        # create_if_missing=True)`` would also work, but pre-creating
        # surfaces wiring errors here instead of in every worker.
        try:
            import modal as _modal
            _modal.Dict.from_name(shared_dict_name, create_if_missing=True)
            task_suite["_fanout_seen_dict_name"] = shared_dict_name
            print(f"\n  [shared-seen] created Modal Dict: {shared_dict_name}")
        except Exception as exc:
            print(f"  WARNING: shared-seen Dict init failed ({exc}) — running without cross-worker dedup")

        # #631: generate a parent run_id for Augur branch_context grouping.
        # Every spawned worker labels its DebugSession with this parent
        # so the Augur UI groups N partition rows under one logical
        # fan-out parent. Per augur-sdk 0.2.1, mantis fan-out is
        # ``mutated_axis="action"`` (different URL per worker = action
        # mutation); the SDK's auto-mode resolves to ``sandbox``
        # (execute fresh, no replay prefix).
        fanout_parent_run_id = (
            f"fanout-{task_suite.get('_plan_signature', 'unknown')[:12]}-"
            f"{_uuid.uuid4().hex[:8]}"
        )
        task_suite["_fanout_parent_run_id"] = fanout_parent_run_id
        print(f"  [fanout/augur] parent run_id: {fanout_parent_run_id}")

        # ── #628: Phase-1/Phase-2 path for parallelizable_url_collect ─
        # Prefer this over per-page partitioning when the plan exposes
        # a url-collect-shaped loop. Phase 1 runs the setup chain +
        # collect_urls (one container) to harvest unique listing URLs;
        # Phase 2 partitions those URLs across N workers, each running
        # a one-shot navigate+scroll+extract sub-plan. No cross-partition
        # duplicates by construction.
        # #673: the orchestration body that previously lived inline
        # here moved into ``gym/fanout_runner.run_fanout_dispatch`` so
        # the HTTP path can call the same code. We keep the CLI's
        # gating + the post-fanout fall-through unchanged.
        url_collect_group = find_url_collect_group(task_suite)
        if url_collect_group is not None:
            executor_fn = EXECUTOR_MAP.get(model, run_holo3)
            from mantis_agent.gym.fanout_runner import run_fanout_dispatch
            _phase1_workers_req = task_suite.get("_fanout_phase1_workers", 1)
            try:
                _phase1_workers_cli = max(1, int(_phase1_workers_req or 1))
            except (TypeError, ValueError):
                _phase1_workers_cli = 1
            _fanout_result = run_fanout_dispatch(
                task_suite,
                executor_fn=executor_fn,
                model=model,
                claude_model=claude_model,
                max_steps=max_steps,
                workers=max(_phase1_workers_cli, workers),
                fanout_parent_run_id=fanout_parent_run_id,
                shared_seen_printer=_print_shared_seen_metrics,
            )
            if _fanout_result is not None:
                return
            print(
                "    [phase1] no URLs harvested — falling through to "
                "pagination-partition path (#617)"
            )


        # ── #617: per-page partitioning for parallelizable_pagination ─
        partitions = prepare_modal_partitions(task_suite, workers)
        if partitions:
            print(f"\n  ═══ FANOUT (loop_groups, #617): {len(partitions)} partition(s) × {workers} workers ═══")
            executor_fn = EXECUTOR_MAP.get(model, run_holo3)
            partition_handles = []
            for i, sub_suite in enumerate(partitions):
                # #631: per-partition branch_id for Augur grouping;
                # also tag the phase as ``pagination_partition`` so
                # the branch_context.mutation distinguishes pagination
                # workers from Phase-2 url-collect workers.
                sub_suite["_fanout_branch_id"] = (
                    f"{fanout_parent_run_id}:page{i + 1}"
                )
                sub_suite["_fanout_phase"] = "pagination_partition"
                sub_contents = json.dumps(sub_suite)
                spawn_kwargs = {
                    "task_file_contents": sub_contents,
                    "max_steps": max_steps,
                }
                if model == "claude":
                    spawn_kwargs["claude_model"] = claude_model
                handle = executor_fn.spawn(**spawn_kwargs)
                partition_handles.append((i, handle))
                print(f"    [fanout] partition {i + 1}/{len(partitions)} spawned")

            print(f"\n  Waiting for {len(partition_handles)} partition workers...")
            from mantis_agent.gym.fanout_runner import (
                dedup_leads_by_url, read_partition_result,
            )
            merged_phone = 0
            merged_shared_seen_hits = 0
            per_partition_leads: list[list[dict]] = []
            for i, handle in partition_handles:
                try:
                    summary = read_partition_result(handle.get())
                    merged_phone += summary["with_phone"]
                    merged_shared_seen_hits += summary["shared_seen_hits"]
                    per_partition_leads.append(summary["leads"])
                    print(
                        f"    [fanout] partition {i + 1}: "
                        f"viable={summary['viable']} "
                        f"phone={summary['with_phone']} "
                        f"shared_seen_hits={summary['shared_seen_hits']}"
                    )
                except Exception as e:
                    print(f"    [fanout] partition {i + 1}: ERROR — {e}")

            # #621: cross-partition dedup by listing_url. Featured /
            # sponsored listings repeat across pages, and pagination
            # drift mid-run can shift listings onto adjacent pages —
            # the orchestrator deduplicates so ``Total leads`` is the
            # true unique-count, not a naive sum.
            _deduped, raw_total, dedup_total = dedup_leads_by_url(
                per_partition_leads,
            )
            print("\n  ═══ FANOUT RESULTS ═══")
            print(f"  Partitions:    {len(partitions)}")
            print(f"  Total leads (raw):     {raw_total}")
            print(f"  Total leads (deduped): {dedup_total}")
            if raw_total > dedup_total:
                print(
                    f"  Duplicates collapsed: {raw_total - dedup_total} "
                    f"(cross-partition URL overlap)"
                )
            print(f"  With phone:    {merged_phone}")
            # #631 follow-up: shared-seen aggregate metric line.
            _print_shared_seen_metrics(task_suite, merged_shared_seen_hits)
            return

    # ── Single worker (default) ──────────────────────────────────
    # The legacy ``task.loop`` + ``_make_page_task`` parallel path
    # (#598) was removed in #618 in favor of the loop_groups-driven
    # fan-out above. Plans that used the hand-authored ``task.loop``
    # schema with ``--task-file`` now run on a single worker; convert
    # to the micro-plan + ``--micro`` path to opt back into parallel
    # extraction.
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
