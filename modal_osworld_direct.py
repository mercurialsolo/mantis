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


GEMMA4_MODEL = os.environ.get("GEMMA4_MODEL", "26B")  # "E4B" or "26B"

GGUF_CONFIGS = {
    "E4B": {
        "repo": "ggml-org/gemma-4-E4B-it-GGUF",
        "model_file": "gemma-4-e4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-e4b-it-f16.gguf",
    },
    "26B": {
        "repo": "ggml-org/gemma-4-26b-a4b-it-GGUF",
        "model_file": "gemma-4-26b-a4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-26b-a4b-it-f16.gguf",
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

    # Find mmproj file
    mmproj_files = glob.glob(os.path.join(model_dir, "mmproj*"))
    print(f"mmproj files: {mmproj_files}")

    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0", "--port", str(port),
        "-ngl", "99", "-c", "8192",
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
def run_osworld(domain: str = "os", max_tasks: int = 5, max_steps: int = 15):
    """Run OSWorld evaluation directly with Gemma4 on A100.

    Uses OSWorld's PromptAgent with OPENAI_BASE_URL pointing at our llama-server.
    The Docker provider runs the Ubuntu VM via QEMU.
    """
    import requests

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

    # Override the system prompt with one optimized for Gemma4's strengths
    GEMMA4_SYSTEM_PROMPT = """You are an expert computer use agent controlling an Ubuntu Linux desktop.
You receive a screenshot of the current screen state each step.
You must output pyautogui Python code to perform ONE action at a time.

CRITICAL RULES:
1. LOOK CAREFULLY at the screenshot before acting. Identify exact pixel coordinates of UI elements.
2. Output ONLY valid Python code in a code block. No explanations outside the code block.
3. Use pyautogui functions: click(x,y), doubleClick(x,y), write('text'), press('key'), hotkey('ctrl','c'), scroll(clicks), moveTo(x,y)
4. NEVER use pyautogui.locateCenterOnScreen() or pyautogui.screenshot()
5. Add time.sleep(0.5) between multiple actions in one step
6. The screen resolution is 1920x1080. Coordinates must be within this range.
7. The computer password is '{CLIENT_PASSWORD}'. Use it for sudo or login prompts.

SPECIAL CODES (return these INSTEAD of Python code when appropriate):
- ```WAIT``` - when waiting for something to load or complete
- ```DONE``` - when the task is successfully completed
- ```FAIL``` - ONLY when the task is truly impossible (try hard first!)

STRATEGY:
- For terminal tasks: Open terminal (Ctrl+Alt+T), type commands, wait for output
- For GUI tasks: Click on relevant UI elements, navigate menus step by step
- For settings: Open Settings app from Activities or use gsettings command
- If you see a login screen, type the password and press Enter
- After typing a command in terminal, ALWAYS press Enter to execute it
- Be precise with coordinates - aim for the CENTER of buttons/elements
- If an action didn't work, try a different approach

First briefly describe what you see on screen and what you'll do, then output the code.""".strip()

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

            instruction = example["instruction"]
            print(f"\n[{dom}] {example_id}")
            print(f"  Task: {instruction[:80]}")

            # Set up the task environment (run config commands on VM)
            task_config = example.get("config", [])
            try:
                setup_controller.reset_cache_dir(f"cache/{dom}/{example_id}")
                setup_controller.setup(task_config, False)
            except Exception as e:
                print(f"  Setup failed: {e}")

            # Reset agent
            agent.reset()
            action_history = []

            # Run agent loop
            done = False
            for step in range(max_steps):
                try:
                    obs_resp = requests.get("http://localhost:5050/screenshot", timeout=30)
                    screenshot = obs_resp.content
                except Exception as e:
                    print(f"  Screenshot failed: {e}")
                    break

                obs = {"screenshot": screenshot, "accessibility_tree": None}

                try:
                    response, actions = agent.predict(instruction, obs)
                except Exception as e:
                    print(f"  Predict failed: {e}")
                    break

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

                if done:
                    break
                print(f"  Step {step+1}: {str(actions[0])[:60] if actions else 'none'}")

            # Full OSWorld evaluation using task's evaluator config
            time.sleep(5)  # Let VM settle
            evaluator = example.get("evaluator", {})
            score = 0.0

            try:
                # Run post-config if any
                postconfig = evaluator.get("postconfig", [])
                if postconfig:
                    setup_controller.setup(postconfig, False)

                # Check for FAIL action
                if action_history and (
                    action_history[-1] == "FAIL" or
                    (isinstance(action_history[-1], dict) and action_history[-1].get("action_type") == "FAIL")
                ):
                    if evaluator.get("func") == "infeasible":
                        score = 1.0
                    else:
                        score = 0.0
                elif evaluator.get("func") == "infeasible":
                    score = 0.0
                else:
                    # Get the metric function and result getter
                    eval_func = evaluator.get("func", "")
                    metric_func = getattr(metrics, eval_func, None)

                    if metric_func:
                        result_config = evaluator.get("result", {})
                        expected_config = evaluator.get("expected", {})

                        # Get result from VM
                        result_type = result_config.get("type", "")
                        result_getter = getattr(getters, f"get_{result_type}", None)

                        if result_getter:
                            # Create a minimal env-like object for the getter
                            class MinimalEnv:
                                def __init__(self, vm_ip, server_port, cache_dir):
                                    self.vm_ip = vm_ip
                                    self.server_port = server_port
                                    self.cache_dir = f"cache/{dom}/{example_id}"
                                    self.controller = controller
                                    self.setup_controller = setup_controller

                            mini_env = MinimalEnv("localhost", 5050, f"cache/{dom}/{example_id}")
                            result_state = result_getter(mini_env, result_config)

                            # Get expected value
                            expected_type = expected_config.get("type", "")
                            expected_getter = getattr(getters, f"get_{expected_type}", None)

                            if expected_getter:
                                expected_state = expected_getter(mini_env, expected_config)
                                score = float(metric_func(result_state, expected_state))
                            else:
                                score = float(metric_func(result_state, expected_config))
                        else:
                            print(f"  Unknown result getter: get_{result_type}")
                    else:
                        print(f"  Unknown metric: {eval_func}")
            except Exception as e:
                print(f"  Eval error: {e}")
                score = 0.0

            scores.append(score)
            print(f"  Score: {score}")
            task_count += 1

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
def main(domain: str = "os", tasks: int = 5, steps: int = 15):
    print(f"Launching OSWorld on Modal A100 (domain={domain}, tasks={tasks})")
    result = run_osworld.remote(domain=domain, max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
