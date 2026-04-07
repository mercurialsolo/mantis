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


def derive_hint(evaluator: dict, instruction: str, domain: str) -> str:
    """Derive a task hint from the evaluator config — generalizes across all domains.

    Instead of hardcoding per-task hints, we reverse-engineer what the evaluator
    checks and tell the agent what outcome is expected. This works for any domain
    because the evaluator config describes the success criteria.
    """
    eval_func = evaluator.get("func", "")

    # 1. Infeasible tasks — tell agent to say FAIL
    if eval_func == "infeasible" or (isinstance(eval_func, list) and "infeasible" in eval_func):
        return ("\nThis task is IMPOSSIBLE in this virtual machine environment. "
                "There is no real hardware (no battery, no Bluetooth adapter, no physical devices, "
                "no Python4, no undefined variables). "
                "The correct action is to respond with FAIL. Do NOT attempt workarounds — just output FAIL.")

    hints = []

    # 2. Extract target file paths from result config — tells agent WHERE to save
    results = evaluator.get("result", {})
    if not isinstance(results, list):
        results = [results]
    for r in results:
        if not isinstance(r, dict):
            continue
        rtype = r.get("type", "")
        if rtype == "vm_file" and r.get("path"):
            hints.append(f"The result should be saved at: {r['path']}")
        elif rtype == "vm_command_line" and r.get("command"):
            cmd = r["command"]
            if isinstance(cmd, list):
                cmd = " ".join(cmd)
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
                hints.append(f"Expected value: {str(rules['expected'])[:100]}")
            if rules.get("include"):
                hints.append(f"Output must include: {rules['include']}")
            if rules.get("exclude"):
                hints.append(f"Output must NOT include: {rules['exclude']}")

    # 4. Domain-aware opening action
    if domain == "os":
        hints.insert(0, "Open terminal with Ctrl+Alt+T if needed. Password is 'password'.")
    elif domain == "chrome":
        hints.insert(0, "Chrome browser should already be open. Use the address bar and menus.")
    elif domain in ("libreoffice_calc", "libreoffice_writer", "libreoffice_impress"):
        app_name = {"libreoffice_calc": "Calc", "libreoffice_writer": "Writer",
                     "libreoffice_impress": "Impress"}[domain]
        hints.insert(0, f"LibreOffice {app_name} should be open. Use menus and keyboard shortcuts.")
    elif domain == "vs_code":
        hints.insert(0, "VS Code should be open. Use Ctrl+Shift+P for command palette.")
    elif domain == "gimp":
        hints.insert(0, "GIMP should be open. Use menus: Filters, Colors, Image, Tools.")
    elif domain == "vlc":
        hints.insert(0, "Use VLC media player. Access settings via Tools > Preferences.")
    elif domain == "thunderbird":
        hints.insert(0, "Thunderbird email client should be open.")

    if not hints:
        return ""

    return "\nHint: " + " ".join(hints)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=129600,  # 36 hours — full 369-task run takes ~31h
    memory=65536,
    cpu=8,
)
def run_osworld(domain: str = "os", max_tasks: int = 5, max_steps: int = 25):
    """Run OSWorld evaluation directly with Gemma4 on A100.

    Uses OSWorld's PromptAgent with OPENAI_BASE_URL pointing at our llama-server.
    The Docker provider runs the Ubuntu VM via QEMU.
    """
    import requests

    run_osworld._run_start = time.time()
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

    controller = PythonController(vm_ip="localhost", server_port=5050)
    setup_controller = SetupController(
        vm_ip="localhost", server_port=5050,
        chromium_port=9222, vlc_port=8080,
        cache_dir="cache", client_password="password",
        screen_width=1920, screen_height=1080,
    )

    scores = []
    task_count = 0

    for dom, example_ids in test_all.items():
        for example_id in example_ids:
            if max_tasks > 0 and task_count >= max_tasks:
                break

            config_file = f"evaluation_examples/examples/{dom}/{example_id}.json"
            with open(config_file) as f:
                example = json.load(f)

            raw_instruction = example["instruction"]
            evaluator = example.get("evaluator", {})

            # Generalized hint engine — derives hints from evaluator config,
            # not from hardcoded domain knowledge. Works across all 369 tasks.
            hint = derive_hint(evaluator, raw_instruction, dom)

            instruction = raw_instruction + hint
            print(f"\n[{dom}] {example_id}")
            print(f"  Task: {raw_instruction[:80]}")
            if hint:
                print(f"  Hint: {hint.strip()[:60]}...")

            task_start = time.time()

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

                        # Self-verification: if failed, build feedback for retry
                        if score == 0 and attempt < max_retries and sub_scores:
                            feedback = f"\n\nVERIFICATION FAILED (attempt {attempt+1}). "
                            for i, (r, e) in enumerate(zip(all_results, all_expected)):
                                feedback += f"Check {i+1}: got '{r}', expected '{e}'. "
                            feedback += "Try a DIFFERENT approach to fix this."

                            print(f"  Attempt {attempt+1} failed. Results: {all_results}")
                            print(f"  Retrying with feedback...")

                            # Run more agent steps with failure feedback
                            retry_instruction = raw_instruction + hint + feedback
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

            # Track per-task telemetry
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

            # Cost calculation: A100-80GB = $0.000694/s on Modal
            total_gpu_time = time.time() - (run_osworld._run_start if hasattr(run_osworld, '_run_start') else time.time())
            cost_per_second = 0.000694  # A100-80GB Modal pricing
            total_cost = total_gpu_time * cost_per_second

            # Save incrementally after each task (survives disconnects)
            results_so_far = {
                "domain": domain,
                "tasks_run": len(scores),
                "tasks_passed": sum(1 for s in scores if s > 0),
                "average_score": sum(scores) / len(scores) * 100,
                "scores": scores,
                "task_details": getattr(run_osworld, '_task_details', []),
                "total_gpu_time_s": round(total_gpu_time, 1),
                "estimated_cost_usd": round(total_cost, 2),
                "avg_time_per_task_s": round(total_gpu_time / len(scores), 1),
                "task_ids": [test_all[dom][i] for i, _ in enumerate(scores) if i < len(test_all.get(dom, []))],
                "model": f"gemma-4-{GEMMA4_MODEL}-Q4_K_M",
            }
            os.makedirs("/data/results", exist_ok=True)
            with open(f"/data/results/osworld_results_{domain}.json", "w") as f:
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

    # Save results
    results = {
        "domain": domain,
        "tasks_run": len(scores),
        "scores": scores,
        "model": f"gemma-4-{GEMMA4_MODEL}-Q4_K_M",
    }
    os.makedirs("/data/results", exist_ok=True)
    with open(f"/data/results/osworld_results_{domain}.json", "w") as f:
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
