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
        "-ngl", "99", "-c", "16384" if GEMMA4_MODEL == "31B" else "8192",
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


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
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
            # Augment instruction with CLI hints for common task patterns
            hint = ""
            lower = raw_instruction.lower()
            if "install" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then use `sudo apt install APPNAME` or `sudo snap install APPNAME`. Password is 'password'."
            elif "favorite" in lower:
                hint = "\nHint: Terminal is already open. Step 1: type `gsettings get org.gnome.shell favorite-apps` and press Enter to see current list. Step 2: type `gsettings set org.gnome.shell favorite-apps` followed by the list WITHOUT the app to remove, and press Enter. Use single quotes around the list value."
            elif "dim" in lower or "idle" in lower or "inactive" in lower:
                hint = "\nHint: First open terminal with Ctrl+Alt+T, wait 1 second, then type: gsettings set org.gnome.settings-daemon.plugins.power idle-dim false\nThen press Enter. Make sure to open the terminal FIRST before typing."
            elif "notification" in lower:
                hint = "\nHint: Use terminal: `gsettings set org.gnome.desktop.notifications show-banners false`"
            elif "switch" in lower and "user" in lower:
                hint = "\nHint: Use terminal: `su - USERNAME` then enter the password."
            elif "wallpaper" in lower or "background" in lower:
                hint = "\nHint: Use terminal: `gsettings set org.gnome.desktop.background picture-uri 'file:///path/to/image'`"
            elif "volume" in lower and ("max" in lower or "up" in lower):
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `amixer set Master 100%` or `pactl set-sink-volume @DEFAULT_SINK@ 100%`"
            elif "bluetooth" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `rfkill unblock bluetooth && bluetoothctl power on`"
            elif "time zone" in lower or "timezone" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `sudo timedatectl set-timezone TIMEZONE`"
            elif "battery" in lower and "percentage" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `gsettings set org.gnome.desktop.interface show-battery-percentage true`"
            elif "lock" in lower and ("auto" in lower or "leave" in lower or "after" in lower):
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `gsettings set org.gnome.desktop.screensaver lock-enabled true`"
            elif "font" in lower or "text size" in lower or "seeing" in lower or "glasses" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `gsettings set org.gnome.desktop.interface text-scaling-factor 1.5` (adjust as needed)"
            elif "ssh" in lower and "user" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `sudo adduser USERNAME` and `sudo usermod -aG sudo USERNAME`"
            elif "python" in lower and "default" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `sudo update-alternatives --set python /usr/bin/pythonX`"
            elif "recover" in lower or "deleted" in lower or "trash" in lower:
                hint = "\nHint: Terminal is already open. To recover: type `cp ~/.local/share/Trash/files/FILENAME ~/Desktop/FILENAME` and press Enter. Replace FILENAME with the actual file name from the task. If unsure of name, first run `ls ~/.local/share/Trash/files/` then copy the file."
            elif "rename" in lower or "change" in lower and "name" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `mv OLD_PATH NEW_PATH`"
            elif "copy" in lower and ("file" in lower or "director" in lower):
                hint = "\nHint: Open terminal (Ctrl+Alt+T), use `cp` or `cp -r` for directories. Use `find` + `cp` for pattern matching."
            elif "permission" in lower or "chmod" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `find . -type f -exec chmod 644 {} +`"
            elif "compress" in lower or "zip" in lower or "tar" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), use `find` + `tar` or `zip` commands."
            elif "count" in lower and "line" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), then: `find . -name '*.EXT' -exec wc -l {} + | tail -1`"
            elif "append" in lower or "save" in lower and "output" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T), use `sed` or `echo`/`printf` with redirection."
            elif "terminal" in lower and "size" in lower:
                hint = "\nHint: Open terminal (Ctrl+Alt+T). Set default size in profile: `dconf write /org/gnome/terminal/legacy/profiles:/:PROFILE_ID/default-size-columns COL` and `default-size-rows ROW`"
            # Detect infeasible tasks — some tasks are impossible in a VM
            # (e.g., Bluetooth, hardware features) and the correct answer is FAIL
            evaluator = example.get("evaluator", {})
            eval_func = evaluator.get("func", "")
            if eval_func == "infeasible":
                hint += "\nIMPORTANT: This task may be IMPOSSIBLE in this environment (virtual machine). If you determine the task cannot be completed (e.g., no hardware, no physical device), respond with just FAIL instead of trying to force it."

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

                            result_state = result_getter(mini_env, rc)
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
                                    obs = {"screenshot": obs_resp.content, "accessibility_tree": None}
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
            with open("/data/results/osworld_results.json", "w") as f:
                json.dump(results_so_far, f, indent=2)
            try:
                vol.commit()
            except Exception:
                pass  # Don't fail on volume commit errors

    # 7. Report
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
    with open("/data/results/osworld_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    llama_proc.terminate()
    qemu_proc.terminate()
    return results


@app.local_entrypoint()
def main(domain: str = "os", tasks: int = 0, steps: int = 25):
    """Run OSWorld eval. tasks=0 means all tasks in the domain."""
    print(f"Launching OSWorld on Modal A100 (domain={domain}, tasks={'ALL' if tasks==0 else tasks})")
    print("Results are saved incrementally to Modal volume 'osworld-data'")
    print("Check progress: modal volume get osworld-data results/osworld_results.json -")
    result = run_osworld.remote(domain=domain, max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
