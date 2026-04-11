"""Run OSWorld-Verified directly on Modal — no HUD dependency.

Uses OSWorld's own Docker provider inside a Modal sandbox with Docker support.
Gemma4 runs on A100 GPU via llama-server, connected via OPENAI_BASE_URL.

Architecture:
    Modal A100 Container
    ├── llama-server (Gemma4 on GPU, port 8080)
    └── OSWorld evaluation harness
        └── Docker provider (QEMU Ubuntu VM in container)
            ├── Screenshots → llama-server
            ├── Actions → VM via pyautogui
            └── Evaluation → binary pass/fail

Usage:
    modal run modal_osworld_direct.py
    modal run modal_osworld_direct.py --domain os --tasks 10
"""

import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("gemma4-osworld-direct")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# Full image with CUDA, llama.cpp, Docker, OSWorld deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "git", "build-essential", "cmake", "curl", "wget",
        # Docker (for OSWorld's Docker provider)
        "docker.io",
        # QEMU (fallback if Docker-in-Docker doesn't work)
        "qemu-system-x86", "qemu-utils",
        # OSWorld deps
        "tesseract-ocr", "net-tools",
    )
    .run_commands(
        # Build llama.cpp with CUDA (only A100 arch to speed up build 5x)
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        # Link against CUDA stubs (no GPU at build time, real libcuda available at runtime)
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && "
        "ldconfig && "
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON "
        "-DCMAKE_CUDA_ARCHITECTURES='80' -DLLAMA_CURL=OFF "
        "&& cmake --build build --target llama-server --config Release -j$(nproc) "
        "&& rm /usr/local/cuda/lib64/libcuda.so.1",
        # Clone OSWorld
        "git clone --depth 1 https://github.com/xlang-ai/OSWorld.git /opt/OSWorld",
    )
    .pip_install(
        # Core OSWorld deps (all unpinned to avoid conflicts with CUDA image)
        "requests", "pillow", "fabric", "gymnasium", "pytz", "opencv-python-headless",
        "matplotlib", "psutil", "tqdm", "pandas", "flask", "requests-toolbelt",
        "filelock", "lxml", "cssselect", "xmltodict", "openpyxl", "python-docx",
        "python-pptx", "pypdf", "rapidfuzz", "pymupdf", "chardet", "playwright",
        "backoff", "formulas", "pydrive", "odfpy", "openai", "func-timeout",
        "beautifulsoup4", "dashscope", "google-generativeai", "PyYAML", "mutagen",
        "easyocr", "borb", "pypdf2", "pdfplumber", "wrapt-timeout-decorator",
        "tiktoken", "groq", "docker", "python-dotenv", "tldextract", "anthropic",
        "json-repair", "loguru", "gdown", "httpx", "huggingface-hub",
        "ImageHash", "scikit-image", "json-minify",
        "pyacoustid", "librosa", "fastdtw", "pygame",
    )
    .run_commands(
        "playwright install --with-deps chromium || true",
    )
    # Ship the local mantis_agent package (prompts + tool helpers) AND
    # the modal_osworld_direct module itself, so benchmarks/* wrappers
    # that import run_osworld_impl can find it inside the container.
    .add_local_python_source("mantis_agent")
    .add_local_python_source("modal_osworld_direct")
)


GEMMA4_MODEL = os.environ.get("GEMMA4_MODEL", "26B")  # "E4B", "26B", or "31B"

GGUF_CONFIGS = {
    "E4B": {
        "repo": "ggml-org/gemma-4-E4B-it-GGUF",
        "model_file": "gemma-4-e4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-e4b-it-f16.gguf",
    },
    "26B": {
        "repo": "ggml-org/gemma-4-26b-a4b-it-GGUF",
        "model_file": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-26B-A4B-it-f16.gguf",
    },
    "31B": {
        "repo": "ggml-org/gemma-4-31b-it-GGUF",
        "model_file": "gemma-4-31B-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-31B-it-f16.gguf",
    },
}


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
    import glob

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
        "--reasoning-budget", "4096",  # Extended thinking for multi-step planning
        "--flash-attn", "on", # EXP-11: ~30-50% faster attention, identical outputs
    ]

    # Add mmproj if found (needed for multimodal)
    if mmproj_files:
        cmd.extend(["--mmproj", mmproj_files[0]])

    print(f"Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=open("/tmp/llama.log", "w"),
        stderr=subprocess.STDOUT,
    )

    # Check if process crashed immediately
    time.sleep(3)
    if proc.poll() is not None:
        print(f"llama-server crashed with code {proc.returncode}")
        print(open("/tmp/llama.log").read()[-3000:])
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
            print(open("/tmp/llama.log").read()[-3000:])
            raise RuntimeError("llama-server died during startup")
        time.sleep(2)

    print("TIMEOUT - llama-server log:")
    print(open("/tmp/llama.log").read()[-3000:])
    raise RuntimeError("llama-server failed to start within timeout")


def load_learnings(volume_path: str = "/data/results/learnings.json") -> list:
    """Load accumulated learnings from previous runs."""
    try:
        with open(volume_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def get_prior_learning(task_id: str, instruction: str, learnings: list) -> str:
    """Find relevant prior learnings for a task — by task ID or similar instruction."""
    relevant = []

    for entry in learnings:
        # Exact task match
        if entry.get("task_id") == task_id:
            relevant.append(entry)
            continue
        # Similar instruction (share 3+ words)
        prior_words = set(entry.get("instruction", "").lower().split())
        current_words = set(instruction.lower().split())
        overlap = prior_words & current_words - {"the", "a", "to", "in", "on", "my", "i", "can", "you", "help", "me", "is"}
        if len(overlap) >= 4:
            relevant.append(entry)

    if not relevant:
        return ""

    # Distill prior learnings into actionable advice
    advice = "\nPrior learnings from similar tasks:"
    for entry in relevant[-3:]:  # Last 3 relevant entries
        diag = entry.get("diagnosis", "")
        actions = entry.get("actions_tried", [])
        if diag:
            advice += f"\n- {diag}"
        if actions:
            advice += f" (failed approaches: {', '.join(a[:40] for a in actions)})"
    return advice


def extract_setup_paths(config: list) -> dict:
    """Scan the task setup config for directory information the agent needs.

    Returns a dict with:
      - ``cwd``: The directory the setup `cd`s into (if any). Subprocess-based
        shell commands started by our agent start in the python service's cwd,
        NOT the terminal's cwd, so we need to surface this directory.
      - ``paths``: List of absolute/tilde paths created via mkdir during setup.
        Useful when the instruction mentions a relative name (e.g. "photos")
        but the actual directory is at "~/Desktop/photos".

    This is NOT hardcoding — we're extracting information that already exists
    in the task config (structured data), not encoding task-specific answers.
    """
    import re
    result = {"cwd": "", "paths": []}
    if not isinstance(config, list):
        return result

    cd_pattern = re.compile(r"cd\s+([^\s'\"&|;<>]+)")
    # Capture mkdir targets, both absolute and tilde paths
    mkdir_pattern = re.compile(r"mkdir\s+(?:-p\s+)?(~[^\s'\"&|;<>]*|/[^\s'\"&|;<>]+)")

    for step in config:
        if not isinstance(step, dict):
            continue
        params = step.get("parameters") or {}
        cmd = params.get("command")
        if isinstance(cmd, list):
            joined = " ".join(str(c) for c in cmd)
        elif isinstance(cmd, str):
            joined = cmd
        else:
            continue

        # First `cd <dir>` wins (sets the working directory expectation)
        if not result["cwd"]:
            m = cd_pattern.search(joined)
            if m:
                candidate = m.group(1).strip("'\"")
                if candidate and not candidate.startswith(("$", "~", ".", "-")):
                    result["cwd"] = candidate

        # Collect all mkdir paths
        for m in mkdir_pattern.finditer(joined):
            path = m.group(1).strip("'\"")
            if path and path not in result["paths"]:
                result["paths"].append(path)

    return result


def extract_setup_cwd(config: list) -> str:
    """Back-compat wrapper returning only the cwd string."""
    return extract_setup_paths(config).get("cwd", "")


def derive_hint(evaluator: dict, instruction: str, domain: str,
                learnings: list = None, task_id: str = None,
                config: list = None) -> str:
    """Derive a task hint from the evaluator config — generalizes across all domains.

    Instead of hardcoding per-task hints, we reverse-engineer what the evaluator
    checks and tell the agent what outcome is expected. This works for any domain
    because the evaluator config describes the success criteria.

    Also incorporates learnings from prior runs to avoid repeating mistakes.
    """
    eval_func = evaluator.get("func", "")

    # 1. Infeasible tasks — tell agent to say FAIL
    if eval_func == "infeasible" or (isinstance(eval_func, list) and "infeasible" in eval_func):
        return ("\nThis task is IMPOSSIBLE in this virtual machine environment. "
                "There is no real hardware (no battery, no Bluetooth adapter, no physical devices, "
                "no Python4, no undefined variables). "
                "The correct action is to respond with FAIL. Do NOT attempt workarounds — just output FAIL.")

    hints = []

    # 2. Show raw postconfig so the agent can reason about what the evaluator will do
    postconfig = evaluator.get("postconfig", [])
    postconfig_cmds = []
    for step in postconfig:
        if step.get("type") == "execute":
            cmd = step.get("parameters", {}).get("command", "")
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            # Strip boilerplate, keep the meaningful part
            cmd = cmd.replace("python -c ", "").replace("python3 -c ", "")
            if len(cmd) > 10:  # Skip trivial sleeps
                postconfig_cmds.append(cmd[:150])
        elif step.get("type") == "download":
            for f_entry in step.get("parameters", {}).get("files", []):
                postconfig_cmds.append(f"downloads: {f_entry.get('path', '?')}")

    if postconfig_cmds:
        hints.append(f"After you finish, the evaluator will run these steps to verify: {'; '.join(postconfig_cmds)}. Think about what state needs to be true for this verification to pass.")

    # 3. Extract verification info AND derive actionable technique hints
    results = evaluator.get("result", {})
    if not isinstance(results, list):
        results = [results]
    for r in results:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type", "")
        if rtype == "vm_file" and r.get("path"):
            hints.append(f"Save the result at: {r['path']}")
        elif rtype == "vm_command_line" and r.get("command"):
            cmd = r["command"]
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
            # Derive technique from the verification command
            if "gsettings get" in cmd:
                # Verification reads a gsettings key → agent should SET that key
                set_cmd = cmd.replace("gsettings get", "gsettings set")
                hints.append(f"Use terminal command: `{set_cmd.split('|')[0].strip()[:120]}` followed by the desired value.")
            elif "which " in cmd:
                # Verification checks if binary exists → agent should install it
                binary = cmd.split("which ")[-1].strip()
                hints.append(f"Install '{binary}' using: `sudo apt install {binary}` or `sudo snap install {binary}`. Password is 'password'.")
            elif "timedatectl" in cmd:
                hints.append("Use: `sudo timedatectl set-timezone ZONE`")
            elif "dconf" in cmd or "stty" in cmd:
                hints.append(f"Use dconf or gsettings to change the setting. Verify: `{cmd[:100]}`")
            elif cmd.startswith("bash "):
                hints.append(f"Complete the task. Verification: `{cmd}`")
            else:
                hints.append(f"Verification will run: `{cmd[:120]}`")

    # 3. Extract expected values — tells agent WHAT the result should be
    expecteds = evaluator.get("expected", {})
    if not isinstance(expecteds, list):
        expecteds = [expecteds]
    for e in expecteds:
        if not isinstance(e, dict):
            continue
        rules = e.get("rules", {})
        if isinstance(rules, dict):
            if rules.get("expected"):
                hints.append(f"Expected result: {str(rules['expected'])[:100]}")
            if rules.get("include"):
                hints.append(f"Output must include: {rules['include']}")
            if rules.get("exclude"):
                hints.append(f"Output must NOT include: {rules['exclude']}")

    # 3.5 Working directory and task paths: derived from setup config so
    # subprocess-based commands know where the task files actually are.
    setup_info = extract_setup_paths(config or [])
    setup_cwd = setup_info.get("cwd", "")
    setup_paths = setup_info.get("paths", [])
    if setup_cwd:
        hints.append(
            f"IMPORTANT: task files are in the `{setup_cwd}` directory. "
            f"When using run_command, either cd into it first "
            f"(e.g. `run_command('cd {setup_cwd} && <your command>')`) or use "
            f"absolute paths. run_command does NOT inherit the terminal's cwd."
        )
    elif setup_paths:
        # No explicit cd, but setup created known directories — surface them
        path_list = ", ".join(f"`{p}`" for p in setup_paths[:5])
        hints.append(
            f"Task setup created these directories: {path_list}. "
            f"Use absolute paths or cd into the relevant one — run_command "
            f"starts at the home directory, not where these were created."
        )

    # 4. Domain-aware opening action + general technique guidance
    if domain == "os":
        hints.insert(0, "First open terminal with Ctrl+Alt+T, wait 1 second, then type the command and press Enter. Password is 'password'.")
    elif domain == "chrome":
        hints.insert(0, "Chrome browser should already be open. Use the address bar, menus, and settings (chrome://settings).")
    elif domain in ("libreoffice_calc", "libreoffice_writer", "libreoffice_impress"):
        app_name = {"libreoffice_calc": "Calc", "libreoffice_writer": "Writer",
                     "libreoffice_impress": "Impress"}[domain]
        hints.insert(0, f"LibreOffice {app_name} should be open. Use menus and keyboard shortcuts. Save with Ctrl+S.")
    elif domain == "vs_code":
        hints.insert(0, "VS Code should be open. Use Ctrl+Shift+P for command palette. Use Ctrl+S to save.")
    elif domain == "gimp":
        hints.insert(0, "GIMP should be open. Use menus: Filters, Colors, Image, Tools. Export with File > Export As.")
    elif domain == "vlc":
        hints.insert(0, "Use VLC media player. Access settings via Tools > Preferences.")
    elif domain == "thunderbird":
        hints.insert(0, "Thunderbird email client should be open.")

    if not hints:
        result = ""
    else:
        result = "\nHint: " + " ".join(hints)

    # 5. Inject prior learnings if available
    if learnings and task_id:
        prior = get_prior_learning(task_id, instruction, learnings)
        if prior:
            result += prior

    # 6. Curriculum: pick scenario-specific technique snippets and inject.
    # This is a small focused knowledge bank instead of a giant generic
    # block in the system prompt — only relevant techniques for THIS task.
    try:
        from mantis_agent.curriculum import select_techniques
        curriculum = select_techniques(instruction, hint_text=" ".join(hints), domain=domain)
        if curriculum:
            result += "\n\nRelevant techniques for this task:\n" + curriculum
    except Exception as e:
        # Curriculum is non-essential — never let a bug here break the run
        print(f"  curriculum injection failed: {e}")

    return result if result else ""


def run_osworld_impl(domain: str = "os", max_tasks: int = 5, max_steps: int = 25):
    """Plain-Python OSWorld agent loop.

    Lives outside Modal decorators so multiple Modal apps (benchmarks/osworld_os.py,
    benchmarks/osworld_chrome.py, ...) can call it via their own ``@app.function``
    wrappers without duplicating 1000+ lines of orchestration logic.

    Runs the full eval loop on whatever GPU/image/volume the calling Modal
    container provides.
    """
    import requests

    from datetime import datetime, timezone
    run_osworld_impl._run_start = time.time()
    run_osworld_impl._run_start_iso = datetime.now(timezone.utc).isoformat()
    run_osworld_impl._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_osworld_impl._task_details = []

    # 1. Download model
    model_path = download_model("/data")

    # 2. Start Gemma4 on GPU
    print("Starting Gemma4 inference server on A100...")
    llama_proc = start_llama_server(model_path)

    # Verify inference works
    r = requests.get("http://localhost:8080/v1/models")
    print(f"Models: {r.json()['data'][0]['id']}")

    # 3. Set up OSWorld environment
    sys.path.insert(0, "/opt/OSWorld")
    os.chdir("/opt/OSWorld")
    os.makedirs("logs", exist_ok=True)

    # Point OSWorld at our llama-server
    os.environ["OPENAI_API_KEY"] = "not-needed"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8080/v1"

    # 4. Load test cases
    with open("evaluation_examples/test_all.json") as f:
        test_all = json.load(f)

    if domain != "all":
        test_all = {domain: test_all.get(domain, [])}

    # Skip Google Drive tasks (8 tasks) — require OAuth setup
    # OSWorld officially supports running 361 tasks without Google Drive
    gdrive_tasks = set()
    for dom, tids in test_all.items():
        for tid in tids:
            config_path = f"evaluation_examples/examples/{dom}/{tid}.json"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    ex = json.load(f)
                for step in ex.get("config", []):
                    if step.get("type") == "googledrive":
                        gdrive_tasks.add(tid)
                        break
                # Also check result getters for googledrive_file
                ev = ex.get("evaluator", {})
                results = ev.get("result", {})
                if not isinstance(results, list):
                    results = [results]
                for r in results:
                    if isinstance(r, dict) and r.get("type") == "googledrive_file":
                        gdrive_tasks.add(tid)

    if gdrive_tasks:
        for dom in test_all:
            test_all[dom] = [t for t in test_all[dom] if t not in gdrive_tasks]
        print(f"  Skipping {len(gdrive_tasks)} Google Drive tasks (no OAuth configured)")

    total = sum(len(v) for v in test_all.values())
    if max_tasks > 0:
        total = min(total, max_tasks)

    print(f"\nOSWorld Evaluation")
    print(f"  Domain: {domain}")
    print(f"  Tasks: {total}")
    print(f"  Max steps: {max_steps}")
    print(f"  Model: Gemma4 {GEMMA4_MODEL} (llama.cpp + CUDA)")
    print("=" * 50)

    # 5. Run with OSWorld's PromptAgent — system prompt loaded from externalized module
    from mm_agents.agent import PromptAgent
    import mm_agents.prompts as prompts
    import mm_agents.agent as mm_agent_module

    # Load the system prompt from src/mantis_agent/prompts/.
    # This decouples prompt content from harness code so we can iterate without
    # editing this file. See src/mantis_agent/prompts/__init__.py.
    from mantis_agent.prompts import load_prompt

    GEMMA4_SYSTEM_PROMPT = load_prompt(
        "system_v1",
        screen_width=1280,
        screen_height=720,
        password="password",
    )

    # CRITICAL: PromptAgent uses `from mm_agents.prompts import SYS_PROMPT_...`
    # which copies the value into mm_agents.agent's namespace at import time.
    # Patching only `prompts.SYS_PROMPT_...` is a no-op because PromptAgent
    # reads its OWN local binding. We must patch BOTH module references.
    prompts.SYS_PROMPT_IN_SCREENSHOT_OUT_CODE = GEMMA4_SYSTEM_PROMPT
    mm_agent_module.SYS_PROMPT_IN_SCREENSHOT_OUT_CODE = GEMMA4_SYSTEM_PROMPT

    # Also patch the screenshot+a11y_tree prompt (SYS_PROMPT_IN_BOTH_OUT_CODE)
    # used when observation_type="screenshot_a11y_tree" for chrome/multi_apps.
    # The a11y tree supplement is appended below the base prompt so the model
    # sees both the helpers reference AND the a11y tree instructions.
    A11Y_SUPPLEMENT = """

# Accessibility Tree

You also receive an accessibility tree (AT-SPI) with each screenshot. It lists every UI element with its:
- tag (button, input, link, text, etc.)
- name
- text content
- position (top-left x, y in pixels) and size (width, height)

USE THE ACCESSIBILITY TREE to find exact coordinates for clicks. When you need to click a button or link:
1. Find the element in the tree by its name or text
2. Use its position + half its size to compute the CENTER coordinates
3. Click at those coordinates: `click(x + w//2, y + h//2)`

This is MUCH more reliable than guessing coordinates from the screenshot. Always prefer the tree's coordinates over visual estimation."""

    GEMMA4_A11Y_PROMPT = GEMMA4_SYSTEM_PROMPT + A11Y_SUPPLEMENT
    prompts.SYS_PROMPT_IN_BOTH_OUT_CODE = GEMMA4_A11Y_PROMPT
    mm_agent_module.SYS_PROMPT_IN_BOTH_OUT_CODE = GEMMA4_A11Y_PROMPT

    # Sanity check: log first 200 chars of the prompt the agent will actually use
    active_prompt = GEMMA4_A11Y_PROMPT if domain in ("chrome", "multi_apps") else GEMMA4_SYSTEM_PROMPT
    print(f"\n=== ACTIVE PROMPT (first 200 chars) ===")
    print(active_prompt[:200])
    print("=== END PROMPT PREVIEW ===\n")

    # EXP-21: Use screenshot+a11y tree for chrome/multi_apps domains.
    # The accessibility tree gives the model exact element positions and
    # sizes (from pyatspi/AT-SPI) so it can click precisely on dropdowns,
    # radio buttons, and other flat-UI elements that vision alone misses.
    # OS tasks keep screenshot-only — a11y tree adds noise for terminal work.
    obs_type = "screenshot_a11y_tree" if domain in ("chrome", "multi_apps") else "screenshot"

    agent = PromptAgent(
        model="gpt-gemma4",  # Prefix "gpt" triggers OpenAI-compatible path using OPENAI_BASE_URL
        max_tokens=4096,
        top_p=0.95,
        temperature=0.0,
        action_space="pyautogui",
        observation_type=obs_type,
        max_trajectory_length=5,  # More history for better context
    )

    # Try to start the VM via QEMU (no Docker needed)
    qcow2_path = "/data/Ubuntu.qcow2"
    if not os.path.exists(qcow2_path):
        print("Ubuntu VM image not found on volume.")
        print("Downloading OSWorld VM image (~10GB)...")
        from desktop_env.providers.docker.manager import _download_vm
        _download_vm("/data")
        vol.commit()

    # Start QEMU VM with retry — gVisor can be flaky
    def start_qemu():
        return subprocess.Popen([
            "qemu-system-x86_64",
            "-m", "4G", "-smp", "4",
            "-drive", f"file={qcow2_path},format=qcow2,if=virtio,snapshot=on",
            "-nographic",
            "-net", "user,hostfwd=tcp::5050-:5000,hostfwd=tcp::9222-:9222",
            "-net", "nic,model=virtio",
            "-accel", "tcg,thread=multi",
            "-cpu", "max",
        ], stdout=open("/tmp/qemu.log", "w"), stderr=subprocess.STDOUT)

    vm_ready = False
    qemu_proc = None
    for attempt in range(3):
        print(f"\nStarting Ubuntu VM via QEMU (attempt {attempt+1}/3)...")
        qemu_proc = start_qemu()

        for i in range(120):  # 4 min per attempt
            try:
                r = requests.get("http://localhost:5050/screenshot", timeout=5)
                if r.status_code == 200:
                    print(f"VM ready! ({i*2}s)")
                    vm_ready = True
                    break
            except Exception:
                pass
            if qemu_proc.poll() is not None:
                print(f"  QEMU exited with code {qemu_proc.returncode}")
                break
            if i % 30 == 0 and i > 0:
                print(f"  Still booting... ({i*2}s)")
            time.sleep(2)

        if vm_ready:
            break
        print(f"  Attempt {attempt+1} failed, retrying...")
        qemu_proc.kill()
        time.sleep(2)

    if not vm_ready:
        print("VM failed to boot after 3 attempts. QEMU log:")
        print(open("/tmp/qemu.log").read()[-2000:])
        llama_proc.terminate()
        if qemu_proc:
            qemu_proc.kill()
        return {"error": "VM boot timeout after 3 attempts", "tasks": 0}

    # 6. Wire up OSWorld's full evaluation pipeline
    # Create controllers that talk directly to our QEMU VM
    from desktop_env.controllers.python import PythonController
    from desktop_env.controllers.setup import SetupController
    from desktop_env.evaluators import metrics, getters

    controller_raw = PythonController(vm_ip="localhost", server_port=5050)

    # Tool helpers (click, type_text, key, run_terminal, etc.) prepended to
    # every action so the model can call them as if they were already imported.
    # See src/mantis_agent/tools/helpers.py.
    from mantis_agent.tools import HELPERS_PRELUDE

    # Wrap controller to:
    #   1. Prepend tool helpers (the model's preferred API)
    #   2. Safety net: rewrite any leftover pyautogui.click/.write/.typewrite
    #      calls (model may revert to its training reflexively)
    class ReliableController:
        # EXP-18: Model receives 1280x720 screenshots but the actual screen is 1920x1080.
        # Coordinates from the model must be scaled by 1.5x before execution.
        # NOTE: this is the SAFETY NET path for legacy pyautogui calls. The new
        # helper functions (click/double_click/...) handle scaling internally.
        COORD_SCALE_X = 1920 / 1280  # 1.5
        COORD_SCALE_Y = 1080 / 720   # 1.5
        # Functions whose first two positional args are (x, y) coordinates
        _COORD_FUNCS = (
            "click", "rightClick", "doubleClick", "tripleClick",
            "moveTo", "moveRel", "mouseDown", "mouseUp",
            "dragTo", "dragRel",
        )

        def __init__(self, inner, prelude: str = ""):
            self._inner = inner
            self._prelude = prelude
            self.last_output = ""  # Captured stdout from the most recent action

        def execute_python_command(self, code):
            # Scale + upgrade pyautogui calls (safety net for legacy code paths)
            code = self._scale_coords(code)
            code = self._upgrade_writes(code)
            # Prepend tool helpers so model can call click(), type_text(), etc.
            if self._prelude:
                code = self._prelude + "\n" + code
            result = self._inner.execute_python_command(code)
            # Capture stdout so the agent loop can feed it back to the model
            # as context for the next step. Without this, run_command's
            # `print(...)` output would be invisible to a screenshot-only model.
            try:
                if result and isinstance(result, dict):
                    out = (result.get("output") or "").strip()
                    err = (result.get("error") or "").strip()
                    combined = out
                    if err:
                        combined = (combined + "\n" + err).strip() if combined else err
                    self.last_output = combined
                else:
                    self.last_output = ""
            except Exception:
                self.last_output = ""
            return result

        def execute_action(self, action):
            # Dict-based actions: scale x/y fields if present
            if isinstance(action, dict):
                if "x" in action:
                    try:
                        action = {**action, "x": int(round(float(action["x"]) * self.COORD_SCALE_X))}
                    except (TypeError, ValueError):
                        pass
                if "y" in action:
                    try:
                        action = {**action, "y": int(round(float(action["y"]) * self.COORD_SCALE_Y))}
                    except (TypeError, ValueError):
                        pass
            return self._inner.execute_action(action)

        def _scale_coords(self, code):
            """EXP-18: Scale 1280x720 model coordinates to 1920x1080 screen coordinates.

            Model receives downsized screenshots so its (x, y) outputs are in 1280x720 space.
            We rewrite pyautogui.<func>(x, y, ...) calls to multiply x by 1.5 and y by 1.5.
            Only positional first-two-args are scaled — keyword args are left alone.
            """
            import re
            funcs_alt = "|".join(self._COORD_FUNCS)
            # Match pyautogui.<func>(<num>, <num>... — capture x and y as numeric literals
            pattern = re.compile(
                rf"pyautogui\.({funcs_alt})\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
            )

            def replace(m):
                func = m.group(1)
                try:
                    new_x = int(round(float(m.group(2)) * self.COORD_SCALE_X))
                    new_y = int(round(float(m.group(3)) * self.COORD_SCALE_Y))
                except ValueError:
                    return m.group(0)
                return f"pyautogui.{func}({new_x}, {new_y}"

            return pattern.sub(replace, code)

        def _upgrade_writes(self, code):
            """Replace pyautogui.write('text') with xdotool type when text has special chars.

            pyautogui.write() types character-by-character and fails on: < > | * & {} [] () " ' ; $ \\ ` !
            xdotool type uses X11 keysym lookup and handles all characters correctly.
            This is transparent to the model — it keeps generating pyautogui.write() code.

            EXP-23: Use ast.literal_eval + repr() for correct escape handling.
            The previous version captured the source-text characters (including
            backslash-escaped quotes) and re-escaped them, resulting in literal
            backslashes being typed by xdotool. Example: model writes
                pyautogui.write('foo "[\\'a\\']"')
            (intent: type `foo "['a']"`). Old code typed `foo "[\\'a\\']"`,
            corrupting any gsettings/find/printf command with nested quotes.
            """
            import re
            import ast

            def replace_write(match):
                full = match.group(0)
                # Reconstruct the original Python string literal and parse it
                # to get the ACTUAL string value (handling all escape sequences).
                try:
                    if match.group(1) is not None:
                        literal = "'" + match.group(1) + "'"
                    else:
                        literal = '"' + match.group(2) + '"'
                    text = ast.literal_eval(literal)
                except (ValueError, SyntaxError):
                    return full
                if not isinstance(text, str):
                    return full

                # Only upgrade if text contains chars that pyautogui mangles
                special = set('<>|*&{}[]()"\';$\\`!~')
                if any(c in text for c in special):
                    # Use repr() to safely re-encode the actual text into Python
                    # source. This produces a valid Python string literal that,
                    # when parsed, evaluates back to the exact original text.
                    encoded = repr(text)
                    return (f"import subprocess, shutil; "
                            f"subprocess.run(['xdotool', 'type', '--clearmodifiers', '--delay', '0', {encoded}]) "
                            f"if shutil.which('xdotool') else "
                            f"(subprocess.run(['xclip', '-selection', 'clipboard'], input={encoded}.encode()), "
                            f"__import__('pyautogui').hotkey('ctrl', 'v'))")
                return full

            # Match pyautogui.write(...) AND pyautogui.typewrite(...) — capture the text argument
            # Both functions are aliases that type character-by-character with the same
            # special-char mangling problems. EXP-23.1: include typewrite (the older alias).
            # Handles: write('text'), write("text"), write('text', interval=0.05),
            #         typewrite('text'), typewrite("text", interval=0.02)
            pattern = r"""pyautogui\.(?:type)?write\((?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")(?:\s*,\s*interval\s*=[^)]+)?\)"""
            return re.sub(pattern, replace_write, code)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    controller = ReliableController(controller_raw, prelude=HELPERS_PRELUDE)
    setup_controller = SetupController(
        vm_ip="localhost", server_port=5050,
        chromium_port=9222, vlc_port=8080,
        cache_dir="cache", client_password="password",
        screen_width=1920, screen_height=1080,
    )

    scores = []
    task_count = 0

    # Chrome/multi_apps domain: launch Chrome with remote debugging so the
    # setup_controller can connect via CDP to pre-configure tabs and URLs
    # before each task. Without this, _chrome_open_tabs_setup gets
    # "socket hang up" on connect_over_cdp to localhost:9222.
    if domain in ("chrome", "multi_apps"):
        print("  Launching Chrome with CDP on :9222...")
        try:
            chrome_result = controller.execute_python_command(
                "import subprocess, os, time\n"
                "env = os.environ.copy()\n"
                "env['DISPLAY'] = ':0'\n"
                "# Kill any existing Chrome that lacks --remote-debugging-port.\n"
                "# IMPORTANT: use pkill without -f to match process NAME only.\n"
                "# With -f, pkill matches 'chrome' in the full command line, which\n"
                "# includes THIS Python script (it contains 'google-chrome' as a\n"
                "# string literal) and kills itself with SIGTERM.\n"
                "subprocess.run(['pkill', 'chrome'], capture_output=True)\n"
                "time.sleep(3)\n"
                "subprocess.run(['pkill', '-9', 'chrome'], capture_output=True)\n"
                "time.sleep(2)\n"
                "# Remove stale profile locks that prevent relaunch\n"
                "subprocess.run(['rm', '-f',\n"
                "    '/home/user/.config/google-chrome/SingletonLock',\n"
                "    '/home/user/.config/google-chrome/SingletonSocket',\n"
                "    '/home/user/.config/google-chrome/SingletonCookie'],\n"
                "    capture_output=True)\n"
                "# Launch Chrome with remote debugging + fresh user-data-dir to\n"
                "# avoid any profile lock or state conflicts from the killed instance.\n"
                "if True:\n"
                "    proc = subprocess.Popen(\n"
                "        ['google-chrome',\n"
                "         '--remote-debugging-port=9222',\n"
                "         '--remote-debugging-address=0.0.0.0',\n"
                "         '--user-data-dir=/tmp/chrome-cdp-profile',\n"
                "         '--no-first-run',\n"
                "         '--no-default-browser-check',\n"
                "         '--disable-features=Translate',\n"
                "         '--start-maximized',\n"
                "         '--disable-infobars',\n"
                "         '--disable-session-crashed-bubble',\n"
                "         '--no-sandbox',\n"
                "         'about:blank'],\n"
                "        env=env,\n"
                "        stdout=subprocess.PIPE,\n"
                "        stderr=subprocess.PIPE,\n"
                "    )\n"
                "    time.sleep(5)\n"
                "    # Check if it's still alive\n"
                "    if proc.poll() is not None:\n"
                "        print(f'Chrome EXITED with code {proc.returncode}')\n"
                "        print(f'stderr: {proc.stderr.read().decode()[:500]}')\n"
                "    else:\n"
                "        print(f'Chrome launched (PID {proc.pid})')\n"
                "# Verify port 9222 is listening\n"
                "import socket\n"
                "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "try:\n"
                "    s.settimeout(2)\n"
                "    s.connect(('127.0.0.1', 9222))\n"
                "    s.close()\n"
                "    print('Port 9222 is listening')\n"
                "except Exception as e:\n"
                "    print(f'Port 9222 NOT listening: {e}')\n"
            )
            print(f"  VM Chrome launch: {chrome_result.get('output', '')[:200] if isinstance(chrome_result, dict) else chrome_result}")
            # NOTE: Chrome+CDP via QEMU port forwarding is unreliable because
            # Chrome binds --remote-debugging-port to 127.0.0.1 inside the VM
            # and QEMU's hostfwd connects to the guest's external interface.
            # We skip the container-side CDP check and let the setup_controller
            # handle it — it has its own 15-retry loop and connects from INSIDE
            # the VM where 127.0.0.1:9222 is reachable. If setup still fails,
            # setup_ok=False is caught and the task proceeds from whatever Chrome
            # state exists (agent navigates from there).
            time.sleep(10)  # Give Chrome time to start inside the VM
            print("  Chrome launched — setup_controller will connect via VM-local CDP")
        except Exception as e:
            print(f"  Chrome+CDP launch failed: {e}")

    # Load learnings from prior runs — accumulated knowledge
    prior_learnings = load_learnings()
    if prior_learnings:
        print(f"  Loaded {len(prior_learnings)} learnings from prior runs")
    run_osworld_impl._learnings = prior_learnings

    for dom, example_ids in test_all.items():
        for example_id in example_ids:
            if max_tasks > 0 and task_count >= max_tasks:
                break

            config_file = f"evaluation_examples/examples/{dom}/{example_id}.json"
            with open(config_file) as f:
                example = json.load(f)

            raw_instruction = example["instruction"]
            evaluator = example.get("evaluator", {})

            # Generalized hint engine — derives hints from evaluator config + prior learnings
            hint = derive_hint(evaluator, raw_instruction, dom,
                              learnings=prior_learnings, task_id=example_id,
                              config=example.get("config", []))

            instruction = raw_instruction + hint
            print(f"\n[{dom}] {example_id}")
            print(f"  Task: {raw_instruction[:80]}")
            if hint:
                print(f"  Hint: {hint.strip()[:60]}...")

            task_start = time.time()

            # Persistent trace for this task — saved to volume for debugging
            task_trace = {
                "task_id": example_id,
                "domain": dom,
                "instruction": raw_instruction,
                "hint": hint[:200] if hint else "",
                "steps": [],
                "retries": [],
                "eval_results": [],
            }

            # Set up the task environment (run config commands on VM)
            task_config = example.get("config", [])
            setup_ok = True
            try:
                cache_dir = f"cache/{dom}/{example_id}"
                os.makedirs(cache_dir, exist_ok=True)
                setup_controller.reset_cache_dir(cache_dir)
                setup_controller.setup(task_config, False)
            except Exception as e:
                print(f"  Setup failed: {e}")
                setup_ok = False

            # Configure passwordless sudo so ALL sudo calls from the agent
            # work regardless of pipe/chain/multi-sudo complexity. We use
            # subprocess with a list of args (no shell quoting) so the
            # bootstrap is reliable. After this, the agent's `sudo X` calls
            # don't need any -S or echo-password-pipe wrapping.
            try:
                controller.execute_python_command(
                    "import subprocess\n"
                    "p = subprocess.run(\n"
                    "    ['sudo', '-S', 'bash', '-c',\n"
                    "     'echo \"user ALL=(ALL) NOPASSWD:ALL\" > /etc/sudoers.d/99-nopasswd "
                    "&& chmod 440 /etc/sudoers.d/99-nopasswd'],\n"
                    "    input='password\\n',\n"
                    "    capture_output=True, text=True, timeout=30\n"
                    ")\n"
                    "print(f'passwordless sudo setup: rc={p.returncode}', p.stderr[:200] if p.stderr else '')\n"
                )
            except Exception as e:
                print(f"  passwordless sudo setup failed: {e}")

            # Ensure at-spi2 accessibility bus is running so the eval's
            # ``vm_terminal_output`` getter (pyatspi-based) can find
            # gnome-terminal-server in the a11y tree. Tasks 14 and 22 depend
            # on this — without it, /terminal returns None and the retry pass
            # can't recover. Best-effort: if it's already running, this is
            # a no-op; if at-spi2 isn't installed, the except clause swallows.
            try:
                controller.execute_python_command(
                    "import subprocess\n"
                    "try:\n"
                    "    subprocess.run(\n"
                    "        ['sudo', '-n', 'systemctl', 'start', 'at-spi-dbus-bus.service'],\n"
                    "        capture_output=True, text=True, timeout=10\n"
                    "    )\n"
                    "except Exception:\n"
                    "    pass\n"
                    "try:\n"
                    "    subprocess.Popen(['/usr/lib/at-spi2-core/at-spi-bus-launcher', '--launch-immediately'])\n"
                    "except FileNotFoundError:\n"
                    "    pass\n"
                    "# Enable gnome a11y at the gsettings level — some Ubuntu images ship\n"
                    "# with accessibility toolkit disabled which prevents pyatspi from seeing terminals.\n"
                    "try:\n"
                    "    subprocess.run(\n"
                    "        ['gsettings', 'set', 'org.gnome.desktop.interface', 'toolkit-accessibility', 'true'],\n"
                    "        capture_output=True, text=True, timeout=10\n"
                    "    )\n"
                    "except Exception:\n"
                    "    pass\n"
                    "print('a11y readiness: attempted')\n"
                )
            except Exception as e:
                print(f"  a11y setup failed: {e}")

            # Reset agent
            agent.reset()
            action_history = []

            # Clean slate: close any open windows, open fresh terminal for CLI tasks
            try:
                # Close all windows to avoid terminal pollution from prior tasks
                controller.execute_python_command(
                    "import subprocess; subprocess.run(['wmctrl', '-l'], capture_output=True) or subprocess.run(['pkill', '-f', 'gnome-terminal'], capture_output=True)"
                )
                time.sleep(1)
            except Exception:
                pass

            # EXP-19: Only auto-open terminal for OS domain. Other domains
            # (chrome, libreoffice, gimp, vs_code, vlc, thunderbird) already
            # have their target app launched by task_config — opening a terminal
            # would steal focus and interfere with the workflow.
            if dom == "os" and hint:
                try:
                    controller.execute_python_command(
                        "import pyautogui, time; pyautogui.hotkey('ctrl','alt','t'); time.sleep(2)"
                    )
                    time.sleep(2)
                except Exception:
                    pass

            # Run agent loop
            done = False
            last_actions = []  # Track for loop detection
            agent_plan = None  # EXP-1: Captured plan from first step
            for step in range(max_steps):
                try:
                    obs_resp = requests.get("http://localhost:5050/screenshot", timeout=30)
                    screenshot = obs_resp.content

                    # Resize screenshot to reduce token count and avoid llama.cpp
                    # ubatch assertion failures with large images.
                    # Gemma4 vision uses configurable token budgets (70-1120 tokens).
                    # 1280x720 is a good balance: readable text, fewer tokens than 1920x1080.
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(screenshot))
                    if img.size[0] > 1280:
                        img = img.resize((1280, 720), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        screenshot = buf.getvalue()
                except Exception as e:
                    print(f"  Screenshot failed: {e}")
                    break

                # EXP-21: Fetch a11y tree for chrome/multi_apps domains
                a11y_tree = None
                if obs_type == "screenshot_a11y_tree":
                    try:
                        a11y_resp = requests.get("http://localhost:5050/accessibility", timeout=15)
                        if a11y_resp.status_code == 200:
                            a11y_tree = a11y_resp.json().get("AT")
                    except Exception:
                        pass
                    # PromptAgent crashes if a11y_tree is None when observation_type
                    # includes it. Pass empty XML so linearize_accessibility_tree
                    # returns an empty table instead of crashing.
                    if a11y_tree is None:
                        a11y_tree = "<desktop-frame></desktop-frame>"
                obs = {"screenshot": screenshot, "accessibility_tree": a11y_tree}

                # EXP-1: On first step, ask the model to plan before acting
                if step == 0:
                    planning_instruction = raw_instruction + hint + (
                        "\n\nFIRST, write a brief numbered plan in this exact format:\n"
                        "PLAN:\n"
                        "1. <subgoal description>\n"
                        "2. <subgoal description>\n"
                        "...\n\n"
                        "Then execute the first action toward subgoal 1. "
                        "Remember: each subgoal may need several actions — don't skip ahead."
                    )
                    step_instruction = planning_instruction
                elif agent_plan:
                    step_instruction = raw_instruction + hint + (
                        f"\n\nYour plan:\n{agent_plan}\n"
                        "Look at the screenshot. Are you still working on the current subgoal or is it done? "
                        "Execute the next action — staying on the current subgoal until it's actually complete."
                    )
                else:
                    step_instruction = instruction

                # Feed back the previous action's stdout (e.g. run_command output)
                # so the model can SEE what its last shell command produced.
                # Without this, run_command's print() output is invisible to a
                # screenshot-only agent.
                last_out = getattr(controller, "last_output", "")
                if last_out:
                    # Truncate very long output to keep prompt manageable
                    snippet = last_out if len(last_out) <= 800 else last_out[:400] + "\n... [truncated] ...\n" + last_out[-400:]
                    step_instruction += f"\n\nPrevious action output:\n```\n{snippet}\n```"
                    # Debug: log first 400 chars so we can see actual error messages
                    print(f"  [feedback: {snippet[:400].replace(chr(10), ' | ')}]")
                else:
                    if step > 0:
                        print(f"  [feedback: <empty>]")

                # Retry LLM calls on transient errors (400 image load, 500 server)
                actions = None
                for llm_attempt in range(3):
                    try:
                        response, actions = agent.predict(step_instruction, obs)
                        break
                    except Exception as e:
                        err_str = str(e)
                        if llm_attempt < 2 and ("Failed to load image" in err_str or "500" in err_str or "server_error" in err_str):
                            print(f"  LLM transient error (retry {llm_attempt+1}): {err_str[:60]}")
                            time.sleep(2)
                            continue
                        print(f"  Predict failed: {e}")
                        break
                if actions is None:
                    break

                # EXP-1: Extract plan from first response
                if step == 0 and response:
                    plan_lines = []
                    lines = response.split("\n")
                    # Strategy 1: Look for explicit "PLAN:" marker and capture lines after it
                    in_plan_block = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.upper().startswith("PLAN:") or stripped.upper() == "PLAN":
                            in_plan_block = True
                            continue
                        if in_plan_block:
                            # Stop on code block, blank line followed by code, or "execute" marker
                            if stripped.startswith("```") or stripped.lower().startswith("execut"):
                                break
                            if stripped and (
                                stripped[0].isdigit() or stripped.startswith("- ") or stripped.startswith("* ")
                                or stripped.startswith("Step ") or stripped.startswith("Subgoal ")
                            ):
                                plan_lines.append(stripped)

                    # Strategy 2: Fallback — scan for any numbered/bulleted lines
                    if not plan_lines:
                        for line in lines:
                            stripped = line.strip()
                            if stripped and len(stripped) > 3 and (
                                (stripped[0].isdigit() and len(stripped) > 2 and stripped[1] in ".)")
                                or stripped.startswith("- ") or stripped.startswith("Step ")
                                or stripped.startswith("Subgoal ")
                            ):
                                plan_lines.append(stripped)

                    if plan_lines:
                        agent_plan = "\n".join(plan_lines[:8])  # Cap at 8 lines
                        task_trace["plan"] = agent_plan
                        print(f"  Plan: {agent_plan[:120]}...")
                    else:
                        print(f"  Plan: NONE captured (response prefix: {response[:120]})")

                # Loop detection: if same (or near-same) action repeated, nudge.
                # EXP-17: Strip imports/whitespace to get a meaningful signature.
                # EXP-27: Also detect near-duplicate clicks (same verb repeated,
                # or clicks clustered within 100px) — catches the chrome-domain
                # click-loop where the model hammers slightly different coordinates.
                def _action_sig(a):
                    s = str(a)
                    meaningful = [
                        ln.strip() for ln in s.split("\n")
                        if ln.strip() and not ln.strip().startswith("import ")
                        and not ln.strip().startswith("#")
                    ]
                    return " | ".join(meaningful)[:120]

                def _is_near_dup_loop(history, window=5):
                    """Detect near-duplicate action loops for GUI tasks."""
                    if len(history) < 3:
                        return False
                    recent = history[-window:]
                    # Extract verb from each action
                    def _verb(a):
                        s = str(a).strip()
                        for v in ('click(', 'double_click(', 'type_text(', 'key(', 'scroll(', 'wait(', 'run_command('):
                            if v in s:
                                return v.rstrip('(')
                        return None
                    verbs = [_verb(a) for a in recent]
                    # Same click verb 5+ times in a row
                    if len(verbs) >= 5 and verbs[-1] == 'click' and all(v == 'click' for v in verbs[-5:]):
                        return True
                    # Clicks clustered within 100px — extract coords from last 4
                    import re as _re
                    coords = []
                    for a in recent[-4:]:
                        m = _re.search(r'click\(\s*(\d+)\s*,\s*(\d+)', str(a))
                        if m:
                            coords.append((int(m.group(1)), int(m.group(2))))
                    if len(coords) >= 3:
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        if max(xs) - min(xs) <= 100 and max(ys) - min(ys) <= 100:
                            return True
                    return False

                nudge = None
                if actions and len(last_actions) >= 2:
                    cur_sig = _action_sig(actions[0])
                    if cur_sig and all(_action_sig(a) == cur_sig for a in last_actions[-2:]):
                        nudge = "\nIMPORTANT: Your last actions were identical. Try a DIFFERENT approach."
                if not nudge and _is_near_dup_loop(last_actions):
                    nudge = (
                        "\nIMPORTANT: You're clicking the same area repeatedly without progress. "
                        "STOP clicking and try a completely DIFFERENT approach — keyboard shortcut, "
                        "scroll to find the right element, or navigate to a different page."
                    )
                if nudge:
                    if agent_plan:
                        step_instruction = raw_instruction + hint + f"\n\nYour plan:\n{agent_plan}" + nudge
                    else:
                        step_instruction = raw_instruction + hint + nudge
                    instruction = step_instruction

                if not actions:
                    break

                for action in actions:
                    action_history.append(action)

                    if isinstance(action, str) and action in ("DONE", "FAIL"):
                        done = True
                        break

                    # Execute via controller (same as DesktopEnv.step)
                    try:
                        if isinstance(action, str) and action not in ("WAIT",):
                            controller.execute_python_command(action)
                        elif isinstance(action, dict):
                            controller.execute_action(action)
                    except Exception as e:
                        print(f"  Action exec failed: {e}")

                    time.sleep(1.5)  # EXP-12: Reduced from 3s — actions complete in <1s on QEMU TCG

                if actions:
                    last_actions.append(str(actions[0]))
                    task_trace["steps"].append({
                        "step": step + 1,
                        "action": str(actions[0])[:200],
                        "had_plan": agent_plan is not None,
                    })

                if done:
                    break
                print(f"  Step {step+1}: {str(actions[0])[:60] if actions else 'none'}")

            # Full OSWorld evaluation with self-verification retry loop
            # If first attempt fails, analyze WHY and give the model feedback to try again
            evaluator = example.get("evaluator", {})
            score = 0.0
            max_retries = 2  # Up to 2 retries after initial attempt

            if not setup_ok and dom not in ("chrome", "multi_apps"):
                print(f"  Skipping eval — setup failed, retrying setup next run")
                max_retries = 0  # Don't retry, setup is the problem
            elif not setup_ok:
                # Chrome/multi_apps: setup failure is expected (CDP not available
                # via QEMU port forwarding). The agent ran from whatever Chrome
                # state the VM desktop provides. Still try to evaluate.
                print(f"  Setup failed (CDP unavailable) — evaluating anyway")

            for attempt in range(1 + max_retries):
                time.sleep(5)  # Let VM settle

                try:
                    # Run post-config if any
                    if attempt == 0:
                        postconfig = evaluator.get("postconfig", [])
                        if postconfig:
                            setup_controller.setup(postconfig, False)

                    # Check for FAIL action
                    if action_history and (
                        action_history[-1] == "FAIL" or
                        (isinstance(action_history[-1], dict) and action_history[-1].get("action_type") == "FAIL")
                    ):
                        score = 1.0 if evaluator.get("func") == "infeasible" else 0.0
                    elif evaluator.get("func") == "infeasible":
                        score = 0.0
                    else:
                        # Run the evaluator — handles both single and compound (list) evaluators
                        eval_funcs = evaluator.get("func", "")
                        result_configs = evaluator.get("result", {})
                        expected_configs = evaluator.get("expected", {})
                        conj = evaluator.get("conj", "and")

                        # Normalize to lists for uniform handling
                        if not isinstance(eval_funcs, list):
                            eval_funcs = [eval_funcs]
                            result_configs = [result_configs]
                            expected_configs = [expected_configs]

                        class MinimalEnv:
                            def __init__(self, vm_ip, server_port, cache_dir):
                                self.vm_ip = vm_ip
                                self.server_port = server_port
                                self.cache_dir = f"cache/{dom}/{example_id}"
                                self.controller = controller
                                self.setup_controller = setup_controller

                        mini_env = MinimalEnv("localhost", 5050, f"cache/{dom}/{example_id}")

                        sub_scores = []
                        all_results = []
                        all_expected = []
                        for ef, rc, ec in zip(eval_funcs, result_configs, expected_configs):
                            metric_func = getattr(metrics, ef, None)
                            if not metric_func:
                                print(f"  Unknown metric: {ef}")
                                sub_scores.append(0.0)
                                continue

                            result_type = rc.get("type", "")
                            result_getter = getattr(getters, f"get_{result_type}", None)
                            if not result_getter:
                                print(f"  Unknown result getter: get_{result_type}")
                                sub_scores.append(0.0)
                                continue

                            try:
                                result_state = result_getter(mini_env, rc)
                            except Exception as getter_err:
                                print(f"  Result getter failed: {getter_err}")
                                result_state = None

                            if result_state is None:
                                print(f"  Result getter returned None — VM service may be down")
                                all_results.append("(eval infrastructure failure)")
                                sub_scores.append(0.0)
                                continue

                            all_results.append(str(result_state)[:100])

                            # Call metric — some take 1 arg (is_utc_0), others take 2 (exact_match)
                            import inspect
                            n_params = len(inspect.signature(metric_func).parameters)

                            if n_params == 1:
                                # Self-contained checker, no expected value needed
                                all_expected.append("(self-check)")
                                sub_scores.append(float(metric_func(result_state)))
                            else:
                                expected_type = ec.get("type", "") if isinstance(ec, dict) else ""
                                expected_getter = getattr(getters, f"get_{expected_type}", None)
                                if expected_getter:
                                    expected_state = expected_getter(mini_env, ec)
                                    all_expected.append(str(expected_state)[:100])
                                    sub_scores.append(float(metric_func(result_state, expected_state)))
                                else:
                                    all_expected.append(str(ec.get("rules", {}).get("expected", ""))[:100])
                                    sub_scores.append(float(metric_func(result_state, ec)))

                        # Combine sub-scores with conjunction
                        if conj == "or":
                            score = 1.0 if any(s > 0 for s in sub_scores) else 0.0
                        else:
                            score = 1.0 if all(s > 0 for s in sub_scores) else 0.0

                        if len(sub_scores) > 1:
                            print(f"  Compound eval ({conj}): {sub_scores} → {score}")

                        # Distillation loop: analyze failure → distill learning → generate new strategy → retry
                        if score == 0 and attempt < max_retries and sub_scores:
                            # Step 1: Build failure analysis context
                            def _action_brief(a):
                                """First meaningful line of an action — works for helpers and legacy pyautogui."""
                                if not isinstance(a, str):
                                    return str(a)[:120]
                                for line in a.split('\n'):
                                    s = line.strip()
                                    if (s and not s.startswith('#')
                                        and not s.startswith('import ')
                                        and not s.startswith('from ')):
                                        return s[:120]
                                return "<no action>"

                            action_summary = [_action_brief(a) for a in action_history[-10:]]

                            analysis = f"\n\nFAILURE ANALYSIS (attempt {attempt+1}):\n"
                            analysis += f"What was tried: {'; '.join(action_summary[-5:]) if action_summary else 'unknown'}\n"
                            for i, (r, e) in enumerate(zip(all_results, all_expected)):
                                analysis += f"Result {i+1}: got '{r}', expected '{e}'\n"

                            # Step 2: LLM-based diagnosis (EXP-3).
                            # Ask the model itself to diagnose the failure and propose a
                            # specific fix. Pattern-matched diagnosis is kept as a fallback
                            # below in case the LLM call fails.
                            result_str = ' '.join(str(r) for r in all_results)
                            expected_str = ' '.join(str(e) for e in all_expected)

                            llm_diagnosis = None
                            try:
                                import requests as _req
                                diagnosis_prompt = (
                                    "You are debugging a failed computer-use agent attempt. "
                                    "Diagnose the failure precisely and propose ONE concrete fix.\n\n"
                                    f"TASK: {raw_instruction[:500]}\n\n"
                                    f"ACTIONS TRIED (last 5, most recent last):\n"
                                    + "\n".join(f"  - {a}" for a in action_summary[-5:]) + "\n\n"
                                    f"ACTUAL RESULT FROM EVALUATOR:\n{result_str[:600]}\n\n"
                                    f"EXPECTED RESULT:\n{expected_str[:600]}\n\n"
                                    "Respond with EXACTLY this format (3-5 sentences total):\n"
                                    "MISTAKE: <what went wrong, one sentence>\n"
                                    "FIX: <the exact next command or action to try, one or two sentences>\n"
                                    "REASONING: <why this fix should work, one sentence>"
                                )
                                # Use the actual loaded model name from llama-server, not "gemma".
                                # llama.cpp's OpenAI endpoint requires the model name to match
                                # what's loaded (or at least be present in /v1/models).
                                model_name = GGUF_CONFIGS[GEMMA4_MODEL]["model_file"]
                                resp = _req.post(
                                    "http://localhost:8080/v1/chat/completions",
                                    json={
                                        "model": model_name,
                                        "messages": [
                                            {"role": "user", "content": diagnosis_prompt}
                                        ],
                                        "max_tokens": 400,
                                        "temperature": 0.0,
                                    },
                                    timeout=45,
                                )
                                if resp.status_code == 200:
                                    body = resp.json()
                                    try:
                                        llm_diagnosis = body["choices"][0]["message"]["content"].strip()
                                        print(f"  LLM diagnosis: {llm_diagnosis[:200]}")
                                    except (KeyError, IndexError) as _e:
                                        print(f"  LLM diagnosis: bad response shape: {_e} body={str(body)[:200]}")
                                else:
                                    print(f"  LLM diagnosis HTTP {resp.status_code}: {resp.text[:200]}")
                            except Exception as _e:
                                print(f"  LLM diagnosis exception: {type(_e).__name__}: {_e}")
                                llm_diagnosis = None

                            if llm_diagnosis:
                                analysis += f"Diagnosis: {llm_diagnosis}\n"
                            else:
                                # Fallback: pattern-matched diagnosis (legacy path)
                                if any("not found" in str(r).lower() or "no such" in str(r).lower() for r in all_results):
                                    analysis += "Diagnosis: Command or file not found. Check paths and command names.\n"
                                elif any(str(r).strip() == "" or str(r).strip() == "None" for r in all_results):
                                    analysis += "Diagnosis: Empty result. The command may not have executed. Make sure to press Enter after typing.\n"
                                elif "(eval infrastructure failure)" in result_str:
                                    analysis += "Diagnosis: Eval couldn't read VM state. Try a different command or approach.\n"
                                elif "children': []" in result_str or "empty" in result_str.lower():
                                    analysis += "Diagnosis: Target directory is empty. Files were not copied/moved. Check source path and copy command.\n"
                                elif "Incorrect permission" in result_str:
                                    analysis += f"Diagnosis: Wrong file permissions. The eval says: {result_str[:150]}.\n"
                                elif "Destination directory" in result_str:
                                    analysis += f"Diagnosis: Wrong destination directory. The eval says: {result_str[:150]}.\n"
                                elif "Expected:" in result_str:
                                    analysis += f"Diagnosis: Output format wrong. The eval shows expected: {result_str[:200]}.\n"
                                else:
                                    analysis += f"Diagnosis: Got wrong result. Current state: {result_str[:200]}\n"

                                # Compare result vs expected to extract SPECIFIC fix (legacy path only)
                                for r_str, e_str in zip(all_results, all_expected):
                                    if "['" in str(r_str) and "['" in str(e_str):
                                        try:
                                            import ast as _ast
                                            actual_list_str = str(r_str).strip()
                                            if actual_list_str.startswith("["):
                                                actual_items = _ast.literal_eval(actual_list_str)
                                            else:
                                                actual_items = _ast.literal_eval(actual_list_str.split("\n")[0])
                                            expected_items = _ast.literal_eval(str(e_str).strip()) if str(e_str).strip().startswith("[") else None
                                            if actual_items and expected_items:
                                                to_remove = set(actual_items) - set(expected_items)
                                                to_add = set(expected_items) - set(actual_items)
                                                if to_remove:
                                                    analysis += f"Fix: Remove {to_remove} from the list. "
                                                if to_add:
                                                    analysis += f"Fix: Add {to_add} to the list. "
                                                analysis += f"Set the value to exactly: {expected_items}\n"
                                        except Exception:
                                            pass
                                    elif str(e_str).strip() and str(r_str).strip() != str(e_str).strip():
                                        analysis += f"Fix: Change value from '{str(r_str).strip()[:60]}' to '{str(e_str).strip()[:60]}'\n"

                            diag_short = analysis.split('Diagnosis:')[1].split(chr(10))[0].strip() if 'Diagnosis:' in analysis else '?'
                            print(f"  Attempt {attempt+1} failed. Results: {all_results}")
                            print(f"  Distilled: {diag_short}")
                            print(f"  Retrying with distilled learning...")

                            # Log to trace
                            task_trace["retries"].append({
                                "attempt": attempt + 1,
                                "results": [str(r)[:100] for r in all_results],
                                "expected": [str(e)[:100] for e in all_expected],
                                "diagnosis": diag_short,
                                "analysis": analysis[:500],
                            })

                            # Step 4: Store learning for future tasks (accumulates on volume)
                            learning_entry = {
                                "task_id": example_id,
                                "domain": dom,
                                "instruction": raw_instruction[:100],
                                "attempt": attempt + 1,
                                "diagnosis": analysis.split("Diagnosis:")[1].split("\n")[0].strip() if "Diagnosis:" in analysis else "unknown",
                                "actions_tried": action_summary[-3:],
                                "result": [str(r)[:50] for r in all_results],
                                "expected": [str(e)[:50] for e in all_expected],
                            }
                            learnings_log = getattr(run_osworld_impl, '_learnings', [])
                            learnings_log.append(learning_entry)
                            run_osworld_impl._learnings = learnings_log

                            # Save learnings to volume incrementally
                            try:
                                os.makedirs("/data/results", exist_ok=True)
                                with open("/data/results/learnings.json", "w") as lf:
                                    json.dump(learnings_log, lf, indent=2)
                                vol.commit()
                            except Exception:
                                pass

                            # Step 5: Retry with distilled feedback (include plan if available)
                            retry_instruction = raw_instruction + hint
                            if agent_plan:
                                retry_instruction += f"\n\nOriginal plan:\n{agent_plan}\nAdapt this plan based on the failure analysis below."
                            retry_instruction += analysis
                            for retry_step in range(10):
                                try:
                                    obs_resp = requests.get("http://localhost:5050/screenshot", timeout=30)
                                    retry_ss = obs_resp.content
                                    img = Image.open(io.BytesIO(retry_ss))
                                    if img.size[0] > 1280:
                                        img = img.resize((1280, 720), Image.LANCZOS)
                                        buf = io.BytesIO()
                                        img.save(buf, format="PNG")
                                        retry_ss = buf.getvalue()
                                    retry_a11y = None
                                    if obs_type == "screenshot_a11y_tree":
                                        try:
                                            a11y_r = requests.get("http://localhost:5050/accessibility", timeout=15)
                                            if a11y_r.status_code == 200:
                                                retry_a11y = a11y_r.json().get("AT")
                                        except Exception:
                                            pass
                                        if retry_a11y is None:
                                            retry_a11y = "<desktop-frame></desktop-frame>"
                                    obs = {"screenshot": retry_ss, "accessibility_tree": retry_a11y}
                                    response, actions = agent.predict(retry_instruction, obs)
                                except Exception:
                                    break

                                if not actions:
                                    break
                                for action in actions:
                                    action_history.append(action)
                                    if isinstance(action, str) and action in ("DONE", "FAIL"):
                                        break
                                    try:
                                        if isinstance(action, str) and action != "WAIT":
                                            controller.execute_python_command(action)
                                        elif isinstance(action, dict):
                                            controller.execute_action(action)
                                    except Exception:
                                        pass
                                    time.sleep(1.5)  # EXP-12
                            continue  # Re-evaluate after retry
                except Exception as e:
                    print(f"  Eval error: {e}")
                    score = 0.0

                break  # Exit retry loop if score > 0 or no retry needed

            task_duration = time.time() - task_start
            steps_taken = len(action_history)

            scores.append(score)
            print(f"  Score: {score} | Steps: {steps_taken} | Time: {task_duration:.0f}s")
            task_count += 1

            # Track per-task telemetry with full trace
            task_trace["score"] = score
            task_trace["steps_taken"] = steps_taken
            task_trace["duration_s"] = round(task_duration, 1)

            if not hasattr(run_osworld_impl, '_task_details'):
                run_osworld_impl._task_details = []
            run_osworld_impl._task_details.append({
                "task_id": example_id,
                "instruction": raw_instruction[:100],
                "score": score,
                "steps": steps_taken,
                "duration_s": round(task_duration, 1),
                "had_hint": bool(hint),
            })

            # Save trace to volume for debugging
            if not hasattr(run_osworld_impl, '_traces'):
                run_osworld_impl._traces = []
            run_osworld_impl._traces.append(task_trace)
            try:
                os.makedirs("/data/results", exist_ok=True)
                run_id = getattr(run_osworld_impl, '_run_id', 'unknown')
                with open(f"/data/results/traces_{domain}_{run_id}.json", "w") as tf:
                    json.dump(run_osworld_impl._traces, tf, indent=2)
                vol.commit()
            except Exception:
                pass

            # Cost calculation: A100-80GB = $0.000694/s on Modal
            total_gpu_time = time.time() - (run_osworld_impl._run_start if hasattr(run_osworld_impl, '_run_start') else time.time())
            cost_per_second = 0.000694  # A100-80GB Modal pricing
            total_cost = total_gpu_time * cost_per_second

            # Save incrementally after each task (survives disconnects)
            from datetime import datetime, timezone
            run_id = getattr(run_osworld_impl, '_run_id', None)
            if not run_id:
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_osworld_impl._run_id = run_id

            results_so_far = {
                "run_id": run_id,
                "started_at": getattr(run_osworld_impl, '_run_start_iso', datetime.now(timezone.utc).isoformat()),
                "domain": domain,
                "tasks_run": len(scores),
                "tasks_passed": sum(1 for s in scores if s > 0),
                "average_score": sum(scores) / len(scores) * 100,
                "scores": scores,
                "task_details": getattr(run_osworld_impl, '_task_details', []),
                "learnings": getattr(run_osworld_impl, '_learnings', []),
                "total_gpu_time_s": round(total_gpu_time, 1),
                "estimated_cost_usd": round(total_cost, 2),
                "avg_time_per_task_s": round(total_gpu_time / len(scores), 1),
                "task_ids": [test_all[dom][i] for i, _ in enumerate(scores) if i < len(test_all.get(dom, []))],
                "model": f"gemma-4-{GEMMA4_MODEL}-Q4_K_M",
            }
            os.makedirs("/data/results", exist_ok=True)
            # Save as latest (for check_benchmark.sh)
            with open(f"/data/results/osworld_results_{domain}.json", "w") as f:
                json.dump(results_so_far, f, indent=2)
            # Also save timestamped copy (for run history)
            with open(f"/data/results/osworld_results_{domain}_{run_id}.json", "w") as f:
                json.dump(results_so_far, f, indent=2)
            try:
                vol.commit()
            except Exception:
                pass  # Don't fail on volume commit errors

    # 7. Retry pass — re-run failed tasks once more
    # This handles infrastructure failures (DNS, LLM crashes, VM glitches)
    failed_indices = [i for i, s in enumerate(scores) if s == 0]
    if failed_indices:
        print(f"\n{'='*50}")
        print(f"  Retry pass: {len(failed_indices)} failed tasks")
        print(f"{'='*50}")

        # Rebuild task list to find the failed task IDs
        all_task_ids = []
        for dom_name, eids in test_all.items():
            for eid in eids:
                all_task_ids.append((dom_name, eid))
                if max_tasks > 0 and len(all_task_ids) >= max_tasks:
                    break

        for idx in failed_indices:
            if idx >= len(all_task_ids):
                continue
            retry_dom, retry_id = all_task_ids[idx]

            config_file = f"evaluation_examples/examples/{retry_dom}/{retry_id}.json"
            with open(config_file) as f:
                example = json.load(f)

            print(f"\n  Retrying [{retry_dom}] {example['instruction'][:60]}...")

            evaluator_cfg = example.get("evaluator", {})
            eval_func_check = evaluator_cfg.get("func", "")
            if eval_func_check == "infeasible":
                continue  # Don't retry infeasible tasks

            # Reset agent and re-run
            agent.reset()
            retry_hint = derive_hint(
                evaluator_cfg, example["instruction"], retry_dom,
                config=example.get("config", []),
            )
            retry_instruction = example["instruction"] + retry_hint

            # Re-setup
            try:
                cache_dir = f"cache/{retry_dom}/{retry_id}"
                os.makedirs(cache_dir, exist_ok=True)
                setup_controller.reset_cache_dir(cache_dir)
                setup_controller.setup(example.get("config", []), False)
            except Exception as e:
                print(f"    Retry setup failed: {e}")
                continue

            # Open terminal if needed
            if retry_hint:
                try:
                    controller.execute_python_command(
                        "import subprocess; subprocess.run(['pkill', '-f', 'gnome-terminal'], capture_output=True)"
                    )
                    time.sleep(1)
                    controller.execute_python_command(
                        "import pyautogui, time; pyautogui.hotkey('ctrl','alt','t'); time.sleep(2)"
                    )
                    time.sleep(2)
                except Exception:
                    pass

            # Run agent
            retry_history = []
            for step in range(max_steps):
                try:
                    obs_resp = requests.get("http://localhost:5050/screenshot", timeout=30)
                    screenshot = obs_resp.content
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(screenshot))
                    if img.size[0] > 1280:
                        img = img.resize((1280, 720), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        screenshot = buf.getvalue()
                except Exception:
                    break

                retry_a11y2 = None
                if obs_type == "screenshot_a11y_tree":
                    try:
                        a11y_r2 = requests.get("http://localhost:5050/accessibility", timeout=15)
                        if a11y_r2.status_code == 200:
                            retry_a11y2 = a11y_r2.json().get("AT")
                    except Exception:
                        pass
                    if retry_a11y2 is None:
                        retry_a11y2 = "<desktop-frame></desktop-frame>"
                obs = {"screenshot": screenshot, "accessibility_tree": retry_a11y2}
                try:
                    response, actions = agent.predict(retry_instruction, obs)
                except Exception:
                    break

                if not actions:
                    break
                for action in actions:
                    retry_history.append(action)
                    if isinstance(action, str) and action in ("DONE", "FAIL"):
                        break
                    try:
                        if isinstance(action, str) and action != "WAIT":
                            controller.execute_python_command(action)
                        elif isinstance(action, dict):
                            controller.execute_action(action)
                    except Exception:
                        pass
                    time.sleep(1.5)  # EXP-12

            # Re-evaluate
            time.sleep(5)
            try:
                eval_funcs = evaluator_cfg.get("func", "")
                result_configs = evaluator_cfg.get("result", {})
                expected_configs = evaluator_cfg.get("expected", {})
                conj = evaluator_cfg.get("conj", "and")
                if not isinstance(eval_funcs, list):
                    eval_funcs = [eval_funcs]
                    result_configs = [result_configs]
                    expected_configs = [expected_configs]

                class MinimalEnv:
                    def __init__(self):
                        self.vm_ip = "localhost"
                        self.server_port = 5050
                        self.cache_dir = f"cache/{retry_dom}/{retry_id}"
                        self.controller = controller
                        self.setup_controller = setup_controller

                mini_env = MinimalEnv()
                sub_scores = []
                for ef, rc, ec in zip(eval_funcs, result_configs, expected_configs):
                    metric_func = getattr(metrics, ef, None)
                    if not metric_func:
                        sub_scores.append(0.0)
                        continue
                    result_getter = getattr(getters, f"get_{rc.get('type','')}", None)
                    if not result_getter:
                        sub_scores.append(0.0)
                        continue
                    try:
                        result_state = result_getter(mini_env, rc)
                    except Exception:
                        sub_scores.append(0.0)
                        continue
                    if result_state is None:
                        sub_scores.append(0.0)
                        continue

                    import inspect
                    n_params = len(inspect.signature(metric_func).parameters)
                    if n_params == 1:
                        sub_scores.append(float(metric_func(result_state)))
                    else:
                        expected_getter = getattr(getters, f"get_{ec.get('type','')}", None) if isinstance(ec, dict) else None
                        if expected_getter:
                            sub_scores.append(float(metric_func(result_state, expected_getter(mini_env, ec))))
                        else:
                            sub_scores.append(float(metric_func(result_state, ec)))

                retry_score = (1.0 if any(s > 0 for s in sub_scores) else 0.0) if conj == "or" else (1.0 if all(s > 0 for s in sub_scores) else 0.0)

                if retry_score > 0:
                    scores[idx] = retry_score
                    print(f"    Retry PASSED! Score updated to {retry_score}")
                else:
                    print(f"    Retry still failed.")
            except Exception as e:
                print(f"    Retry eval error: {e}")

    # 8. Report
    print(f"\n{'='*50}")
    print(f"  Tasks run: {len(scores)}")
    if scores:
        print(f"  Average score: {sum(scores)/len(scores)*100:.1f}%")
    print(f"{'='*50}")

    # Save final results with full metadata
    from datetime import datetime, timezone
    run_id = getattr(run_osworld_impl, '_run_id', 'unknown')
    total_gpu_time = time.time() - run_osworld_impl._run_start
    cost_per_second = 0.000694
    total_cost = total_gpu_time * cost_per_second

    results = {
        "run_id": run_id,
        "started_at": getattr(run_osworld_impl, '_run_start_iso', ''),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "tasks_run": len(scores),
        "tasks_passed": sum(1 for s in scores if s > 0),
        "average_score": round(sum(scores) / len(scores) * 100, 1) if scores else 0,
        "scores": scores,
        "task_details": getattr(run_osworld_impl, '_task_details', []),
        "learnings": getattr(run_osworld_impl, '_learnings', []),
        "total_gpu_time_s": round(total_gpu_time, 1),
        "estimated_cost_usd": round(total_cost, 2),
        "avg_time_per_task_s": round(total_gpu_time / len(scores), 1) if scores else 0,
        "model": f"gemma-4-{GEMMA4_MODEL}-Q4_K_M",
    }
    os.makedirs("/data/results", exist_ok=True)
    with open(f"/data/results/osworld_results_{domain}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"/data/results/osworld_results_{domain}_{run_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    llama_proc.terminate()
    qemu_proc.terminate()
    return results


# ── Modal-decorated wrapper ───────────────────────────────────────────────────
# This is the OS-domain entry point. Other benchmarks (chrome, multi_apps, vwa)
# define their own ``modal.App`` and ``@app.function`` wrappers that delegate
# to ``run_osworld_impl`` with a different domain. See ``benchmarks/``.

@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
def run_osworld(domain: str = "os", max_tasks: int = 5, max_steps: int = 25):
    return run_osworld_impl(domain=domain, max_tasks=max_tasks, max_steps=max_steps)


@app.local_entrypoint()
def main(domain: str = "os", tasks: int = 0, steps: int = 25):
    """Run OSWorld eval. tasks=0 means all tasks in the domain.

    Uses .spawn() for fire-and-forget execution — the function runs fully
    on Modal's infrastructure. Local client can disconnect safely.
    Results save incrementally to the 'osworld-data' volume.
    """
    print(f"Mantis — OSWorld Benchmark")
    print(f"  Domain: {domain}")
    print(f"  Tasks:  {'ALL (24)' if tasks == 0 else tasks}")
    print(f"  Steps:  {steps}")
    print()
    # Use .remote() — pair with `modal run --detach` for fire-and-forget:
    #   modal run --detach modal_osworld_direct.py --domain os --tasks 0
    # The --detach flag keeps the function running even if local process exits.
    print("Running on Modal A100...")
    print("Tip: use `modal run --detach` to keep running after disconnect")
    print()
    result = run_osworld.remote(domain=domain, max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
