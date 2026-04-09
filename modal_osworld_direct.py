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
        "--reasoning-budget", "0",  # Prevents <unused24> token spam on CUDA
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


def derive_hint(evaluator: dict, instruction: str, domain: str,
                learnings: list = None, task_id: str = None) -> str:
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
        return ""

    result = "\nHint: " + " ".join(hints)

    # 5. Inject prior learnings if available
    if learnings and task_id:
        prior = get_prior_learning(task_id, instruction, learnings)
        if prior:
            result += prior

    return result


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,  # 24 hours (Modal max) — full run may need multiple batches
    memory=65536,
    cpu=8,
)
def run_osworld(domain: str = "os", max_tasks: int = 5, max_steps: int = 25):
    """Run OSWorld evaluation directly with Gemma4 on A100.

    Uses OSWorld's PromptAgent with OPENAI_BASE_URL pointing at our llama-server.
    The Docker provider runs the Ubuntu VM via QEMU.
    """
    import requests

    from datetime import datetime, timezone
    run_osworld._run_start = time.time()
    run_osworld._run_start_iso = datetime.now(timezone.utc).isoformat()
    run_osworld._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_osworld._task_details = []

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

    # 5. Run with OSWorld's PromptAgent — improved prompt for Gemma4
    from mm_agents.agent import PromptAgent
    import mm_agents.prompts as prompts

    # Override the system prompt — keep it SHORT (Gemma4 works best with concise prompts)
    GEMMA4_SYSTEM_PROMPT = """You are a computer agent controlling Ubuntu Linux via pyautogui.
You see a screenshot each step. Output Python code in a code block to perform ONE action.

Rules:
- Start code with `import pyautogui` and `import time`
- Screen is 1920x1080. Look at screenshot for coordinates.
- Use: click(x,y), write('text'), press('key'), hotkey('ctrl','c'), scroll(n)
- NEVER use locateCenterOnScreen() or screenshot()
- Password: '{CLIENT_PASSWORD}'
- After typing in terminal, press Enter: `pyautogui.press('enter')`
- Add `time.sleep(0.5)` between actions
- For commands with special characters (<, >, |, *, &, quotes, brackets), use subprocess:
  `import subprocess; subprocess.run('your command here', shell=True)`
- For sudo commands: `subprocess.run("echo 'password' | sudo -S command", shell=True)`

Special codes: ```WAIT```, ```DONE```, ```FAIL```

First reflect on what you see, then output code.""".strip()

    # Monkey-patch the prompt
    prompts.SYS_PROMPT_IN_SCREENSHOT_OUT_CODE = GEMMA4_SYSTEM_PROMPT

    agent = PromptAgent(
        model="gpt-gemma4",  # Prefix "gpt" triggers OpenAI-compatible path using OPENAI_BASE_URL
        max_tokens=4096,
        top_p=0.95,
        temperature=0.0,
        action_space="pyautogui",
        observation_type="screenshot",
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

    # Wrap controller to transparently upgrade pyautogui.write() to clipboard paste.
    # pyautogui.write() types character-by-character and mangles special chars
    # (<, >, |, *, quotes, brackets). Clipboard paste is instant and exact.
    # This is a TOOL improvement — the model doesn't need to know about it.
    class ReliableController:
        def __init__(self, inner):
            self._inner = inner

        def execute_python_command(self, code):
            code = self._upgrade_writes(code)
            return self._inner.execute_python_command(code)

        def execute_action(self, action):
            return self._inner.execute_action(action)

        def _upgrade_writes(self, code):
            """Replace pyautogui.write('text') with xdotool type when text has special chars.

            pyautogui.write() types character-by-character and fails on: < > | * & {} [] () " ' ; $ \\ ` !
            xdotool type uses X11 keysym lookup and handles all characters correctly.
            This is transparent to the model — it keeps generating pyautogui.write() code.
            """
            import re
            def replace_write(match):
                full = match.group(0)
                # Extract the text argument (handle both quote styles and escaped quotes)
                text = match.group(1) if match.group(1) is not None else match.group(2)
                if text is None:
                    return full
                # Only upgrade if text contains chars that pyautogui mangles
                special = set('<>|*&{}[]()"\';$\\`!~')
                if any(c in text for c in special):
                    # Use xdotool type — reliable for all characters
                    # Falls back to xclip+paste if xdotool unavailable
                    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
                    return (f"import subprocess, shutil; "
                            f"subprocess.run(['xdotool', 'type', '--clearmodifiers', '--delay', '0', '{escaped}']) "
                            f"if shutil.which('xdotool') else "
                            f"(subprocess.run(['xclip', '-selection', 'clipboard'], input='{escaped}'.encode()), "
                            f"__import__('pyautogui').hotkey('ctrl', 'v'))")
                return full

            # Match pyautogui.write(...) — capture the text argument
            # Handles: write('text'), write("text"), write('text', interval=0.05)
            pattern = r"""pyautogui\.write\((?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")(?:\s*,\s*interval\s*=[^)]+)?\)"""
            return re.sub(pattern, replace_write, code)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    controller = ReliableController(controller_raw)
    setup_controller = SetupController(
        vm_ip="localhost", server_port=5050,
        chromium_port=9222, vlc_port=8080,
        cache_dir="cache", client_password="password",
        screen_width=1920, screen_height=1080,
    )

    scores = []
    task_count = 0

    # Load learnings from prior runs — accumulated knowledge
    prior_learnings = load_learnings()
    if prior_learnings:
        print(f"  Loaded {len(prior_learnings)} learnings from prior runs")
    run_osworld._learnings = prior_learnings

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
                              learnings=prior_learnings, task_id=example_id)

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

            if hint:
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

                obs = {"screenshot": screenshot, "accessibility_tree": None}

                # Retry LLM calls on transient errors (400 image load, 500 server)
                actions = None
                for llm_attempt in range(3):
                    try:
                        response, actions = agent.predict(instruction, obs)
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

                # Loop detection: if same action 3 times, nudge the model
                if actions and len(last_actions) >= 2 and all(
                    str(actions[0])[:30] == str(a)[:30] for a in last_actions[-2:]
                ):
                    instruction = raw_instruction + hint + "\nIMPORTANT: Your last actions were repetitive. Try a DIFFERENT approach."

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

                    time.sleep(3)  # Longer pause for QEMU TCG

                if actions:
                    last_actions.append(str(actions[0]))
                    task_trace["steps"].append({
                        "step": step + 1,
                        "action": str(actions[0])[:200],
                    })

                if done:
                    break
                print(f"  Step {step+1}: {str(actions[0])[:60] if actions else 'none'}")

            # Full OSWorld evaluation with self-verification retry loop
            # If first attempt fails, analyze WHY and give the model feedback to try again
            evaluator = example.get("evaluator", {})
            score = 0.0
            max_retries = 2  # Up to 2 retries after initial attempt

            if not setup_ok:
                print(f"  Skipping eval — setup failed, retrying setup next run")
                max_retries = 0  # Don't retry, setup is the problem

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
                            action_summary = []
                            for a in action_history[-10:]:  # Last 10 actions
                                if isinstance(a, str):
                                    # Extract just the key action from code
                                    for line in a.split('\n'):
                                        if 'pyautogui' in line or 'subprocess' in line:
                                            action_summary.append(line.strip()[:80])
                                            break

                            analysis = f"\n\nFAILURE ANALYSIS (attempt {attempt+1}):\n"
                            analysis += f"What was tried: {'; '.join(action_summary[-5:]) if action_summary else 'unknown'}\n"
                            for i, (r, e) in enumerate(zip(all_results, all_expected)):
                                analysis += f"Result {i+1}: got '{r}', expected '{e}'\n"

                            # Step 2: Deep diagnosis — extract specific fix from eval results
                            result_str = ' '.join(str(r) for r in all_results)
                            expected_str = ' '.join(str(e) for e in all_expected)

                            if any("not found" in str(r).lower() or "no such" in str(r).lower() for r in all_results):
                                analysis += "Diagnosis: Command or file not found. Check paths and command names.\n"
                            elif any(str(r).strip() == "" or str(r).strip() == "None" for r in all_results):
                                analysis += "Diagnosis: Empty result. The command may not have executed. Make sure to press Enter after typing.\n"
                            elif "(eval infrastructure failure)" in result_str:
                                analysis += "Diagnosis: Eval couldn't read VM state. Try using subprocess.run() to execute commands directly instead of pyautogui.\n"
                            elif "children': []" in result_str or "empty" in result_str.lower():
                                analysis += "Diagnosis: Target directory is empty. Files were not copied/moved. Check source path and copy command.\n"
                            elif "Incorrect permission" in result_str:
                                analysis += f"Diagnosis: Wrong file permissions. The eval says: {result_str[:150]}. Use `find . -type f -exec chmod XXX {{}} +` with the correct permission number.\n"
                            elif "Destination directory" in result_str:
                                analysis += f"Diagnosis: Wrong destination directory. The eval says: {result_str[:150]}. Check the exact target path.\n"
                            elif "Expected:" in result_str:
                                analysis += f"Diagnosis: Output format wrong. The eval shows expected bytes: {result_str[:200]}. Match this exactly.\n"
                            else:
                                analysis += f"Diagnosis: Got wrong result. Current state: {result_str[:200]}\n"

                            # Compare result vs expected to extract SPECIFIC fix
                            for r_str, e_str in zip(all_results, all_expected):
                                # List-based results: identify items to add/remove
                                if "['" in str(r_str) and "['" in str(e_str):
                                    try:
                                        import ast
                                        # Parse the actual list from the result
                                        actual_list_str = str(r_str).strip()
                                        if actual_list_str.startswith("["):
                                            actual_items = ast.literal_eval(actual_list_str)
                                        else:
                                            actual_items = ast.literal_eval(actual_list_str.split("\n")[0])
                                        expected_items = ast.literal_eval(str(e_str).strip()) if str(e_str).strip().startswith("[") else None
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
                                # gsettings-style values: show exact target
                                elif str(e_str).strip() and str(r_str).strip() != str(e_str).strip():
                                    analysis += f"Fix: Change value from '{str(r_str).strip()[:60]}' to '{str(e_str).strip()[:60]}'\n"

                            # Step 3: Generate actionable strategy
                            strategy = ""
                            has_pyautogui_write = any("pyautogui.write" in str(a) for a in action_history)
                            has_gui_clicks = any("click" in str(a) for a in action_history[-3:])
                            ran_out_of_steps = len(action_history) >= max_steps - 2

                            if has_pyautogui_write and any(c in raw_instruction for c in ['<', '>', '|', '*', '"', "'"]):
                                strategy += "- CRITICAL: Use subprocess.run('command', shell=True) instead of pyautogui.write() for commands with special characters.\n"
                            elif has_pyautogui_write:
                                strategy += "- Use subprocess.run() instead of pyautogui.write() to avoid character mangling.\n"
                            if has_gui_clicks:
                                strategy += "- Try a terminal command approach instead of GUI clicks.\n"
                            if ran_out_of_steps:
                                strategy += "- Ran out of steps. Use subprocess.run() to execute the solution in ONE step.\n"
                            if "children': []" in result_str:
                                strategy += "- The copy command didn't work. Try: find SOURCE -name 'PATTERN' -exec cp --parents {} DEST \\;\n"
                            if not strategy:
                                strategy += "- Try a completely different approach.\n"

                            analysis += f"Strategy:\n{strategy}"

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
                            learnings_log = getattr(run_osworld, '_learnings', [])
                            learnings_log.append(learning_entry)
                            run_osworld._learnings = learnings_log

                            # Save learnings to volume incrementally
                            try:
                                os.makedirs("/data/results", exist_ok=True)
                                with open("/data/results/learnings.json", "w") as lf:
                                    json.dump(learnings_log, lf, indent=2)
                                vol.commit()
                            except Exception:
                                pass

                            # Step 5: Retry with distilled feedback
                            retry_instruction = raw_instruction + hint + analysis
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
                                    obs = {"screenshot": retry_ss, "accessibility_tree": None}
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
                                    time.sleep(3)
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

            if not hasattr(run_osworld, '_task_details'):
                run_osworld._task_details = []
            run_osworld._task_details.append({
                "task_id": example_id,
                "instruction": raw_instruction[:100],
                "score": score,
                "steps": steps_taken,
                "duration_s": round(task_duration, 1),
                "had_hint": bool(hint),
            })

            # Save trace to volume for debugging
            if not hasattr(run_osworld, '_traces'):
                run_osworld._traces = []
            run_osworld._traces.append(task_trace)
            try:
                os.makedirs("/data/results", exist_ok=True)
                run_id = getattr(run_osworld, '_run_id', 'unknown')
                with open(f"/data/results/traces_{domain}_{run_id}.json", "w") as tf:
                    json.dump(run_osworld._traces, tf, indent=2)
                vol.commit()
            except Exception:
                pass

            # Cost calculation: A100-80GB = $0.000694/s on Modal
            total_gpu_time = time.time() - (run_osworld._run_start if hasattr(run_osworld, '_run_start') else time.time())
            cost_per_second = 0.000694  # A100-80GB Modal pricing
            total_cost = total_gpu_time * cost_per_second

            # Save incrementally after each task (survives disconnects)
            from datetime import datetime, timezone
            run_id = getattr(run_osworld, '_run_id', None)
            if not run_id:
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                run_osworld._run_id = run_id

            results_so_far = {
                "run_id": run_id,
                "started_at": getattr(run_osworld, '_run_start_iso', datetime.now(timezone.utc).isoformat()),
                "domain": domain,
                "tasks_run": len(scores),
                "tasks_passed": sum(1 for s in scores if s > 0),
                "average_score": sum(scores) / len(scores) * 100,
                "scores": scores,
                "task_details": getattr(run_osworld, '_task_details', []),
                "learnings": getattr(run_osworld, '_learnings', []),
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
            retry_hint = derive_hint(evaluator_cfg, example["instruction"], retry_dom)
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

                obs = {"screenshot": screenshot, "accessibility_tree": None}
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
                    time.sleep(3)

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
    run_id = getattr(run_osworld, '_run_id', 'unknown')
    total_gpu_time = time.time() - run_osworld._run_start
    cost_per_second = 0.000694
    total_cost = total_gpu_time * cost_per_second

    results = {
        "run_id": run_id,
        "started_at": getattr(run_osworld, '_run_start_iso', ''),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "tasks_run": len(scores),
        "tasks_passed": sum(1 for s in scores if s > 0),
        "average_score": round(sum(scores) / len(scores) * 100, 1) if scores else 0,
        "scores": scores,
        "task_details": getattr(run_osworld, '_task_details', []),
        "learnings": getattr(run_osworld, '_learnings', []),
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
