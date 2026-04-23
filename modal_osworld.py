"""Run OSWorld evaluation on Modal with A100 GPU.

Architecture:
    ┌─────────────────── Modal A100 Container ───────────────────┐
    │                                                             │
    │  llama-server (Gemma4 E4B)  ←→  OSWorld Agent  ←→  QEMU VM │
    │        GPU (A100)                  Python           (TCG)   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

- llama-server runs on A100 GPU → ~2-5s per inference step
- QEMU runs the Ubuntu VM in TCG mode (software emulation, no KVM)
- Agent connects to both via localhost

Usage:
    # Upload VM image to Modal volume (one-time):
    modal volume create osworld-data
    modal volume put osworld-data ./OSWorld/docker_vm_data/Ubuntu.qcow2 /Ubuntu.qcow2

    # Run single domain:
    modal run modal_osworld.py --domain os

    # Run full benchmark:
    modal run modal_osworld.py --domain all
"""

import os
import subprocess
import sys
import time

import modal

# ── Modal config ──────────────────────────────────────────────────────────────

app = modal.App("osworld-gemma4-cua")

# Volume for the 24GB Ubuntu QCOW2 image + model weights
vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# Build the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # QEMU for running the Ubuntu VM (TCG mode, no KVM needed)
        "qemu-system-x86", "qemu-utils", "qemu-system-gui",
        # System deps for OSWorld
        "git", "curl", "wget", "tesseract-ocr",
        # Build deps for llama.cpp (we'll use pre-built)
        "build-essential", "cmake",
        # Networking
        "net-tools", "iproute2",
    )
    .pip_install(
        # Our agent
        "transformers>=4.52", "torch", "pillow", "mss", "pyautogui",
        # OSWorld deps (subset needed for eval)
        "requests", "tqdm", "psutil", "filelock", "wrapt-timeout-decorator",
        "opencv-python-headless", "python-docx", "python-pptx", "openpyxl",
        "pypdf", "beautifulsoup4", "lxml", "easyocr", "xmltodict",
        # Docker client (for OSWorld's provider interface)
        "docker",
    )
    .run_commands(
        # Install llama.cpp from source with CUDA support
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)",
    )
    .pip_install(
        # HuggingFace CLI for model download
        "huggingface-hub[cli]",
    )
)


def start_llama_server(model_path: str, port: int = 8080) -> subprocess.Popen:
    """Start llama-server on GPU."""
    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "-ngl", "99",  # Offload all layers to GPU
        "-c", "4096",  # Context window
        "--no-warmup",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to be ready
    import requests
    for _ in range(120):
        try:
            resp = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if resp.status_code == 200:
                print(f"llama-server ready on port {port}")
                return proc
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("llama-server failed to start")


def start_qemu_vm(qcow2_path: str, server_port: int = 5050) -> subprocess.Popen:
    """Start QEMU VM with Ubuntu for OSWorld evaluation.

    Runs in TCG mode (software emulation) — no KVM needed.
    The VM runs OSWorld's server on the specified port.
    """
    cmd = [
        "qemu-system-x86_64",
        "-m", "4G",
        "-smp", "4",
        "-drive", f"file={qcow2_path},format=qcow2,if=virtio",
        "-nographic",  # No GUI needed
        "-net", f"user,hostfwd=tcp::{server_port}-:5000,"
                f"hostfwd=tcp::9222-:9222,"
                f"hostfwd=tcp::8006-:8006,"
                f"hostfwd=tcp::8080-:8080",
        "-net", "nic,model=virtio",
        # TCG (software emulation) — works without KVM
        "-accel", "tcg,thread=multi",
        "-cpu", "max",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for VM to boot and screenshot endpoint to respond
    import requests
    print("Waiting for VM to boot (TCG mode, may take 5-10 min)...")
    for i in range(300):  # Up to 10 minutes
        try:
            resp = requests.get(f"http://localhost:{server_port}/screenshot", timeout=5)
            if resp.status_code == 200:
                print(f"VM ready on port {server_port}")
                return proc
        except Exception:
            pass
        if i % 30 == 0 and i > 0:
            print(f"  Still booting... ({i}s)")
        time.sleep(2)

    raise RuntimeError("VM failed to boot within timeout")


def download_model(vol_path: str) -> str:
    """Download Gemma4 GGUF model if not cached."""
    model_path = os.path.join(vol_path, "gemma-4-e4b-it-Q4_K_M.gguf")
    mmproj_path = os.path.join(vol_path, "mmproj-gemma-4-e4b-it-f16.gguf")

    if os.path.exists(model_path) and os.path.exists(mmproj_path):
        print(f"Model already cached at {model_path}")
        return model_path

    print("Downloading Gemma4 E4B GGUF model...")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="ggml-org/gemma-4-E4B-it-GGUF",
        filename="gemma-4-e4b-it-Q4_K_M.gguf",
        local_dir=vol_path,
    )
    hf_hub_download(
        repo_id="ggml-org/gemma-4-E4B-it-GGUF",
        filename="mmproj-gemma-4-e4b-it-f16.gguf",
        local_dir=vol_path,
    )
    print("Model downloaded.")
    return model_path


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,  # 24 hours max
    memory=32768,   # 32GB RAM
    cpu=8,
)
def run_osworld_eval(domain: str = "os", max_steps: int = 15, max_tasks: int = 5):
    """Run OSWorld evaluation on Modal.

    Args:
        domain: OSWorld domain to evaluate (os, chrome, gimp, etc. or "all")
        max_steps: Max steps per task
        max_tasks: Max tasks to run (for quick testing, set to 0 for all)
    """
    import json

    # 1. Download model if needed
    model_path = download_model("/data")
    vol.commit()

    # 2. Check for QCOW2 image
    qcow2_path = "/data/Ubuntu.qcow2"
    if not os.path.exists(qcow2_path):
        raise FileNotFoundError(
            "Ubuntu.qcow2 not found. Upload it first:\n"
            "  modal volume put osworld-data ./OSWorld/docker_vm_data/Ubuntu.qcow2 /Ubuntu.qcow2"
        )

    # 3. Clone OSWorld
    if not os.path.exists("/opt/OSWorld"):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/xlang-ai/OSWorld.git", "/opt/OSWorld"],
            check=True,
        )

    sys.path.insert(0, "/opt/OSWorld")

    # 4. Start llama-server on GPU
    print("Starting Gemma4 inference server on A100...")
    llama_proc = start_llama_server(model_path, port=8080)

    # 5. Start QEMU VM
    print("Starting Ubuntu VM (QEMU TCG)...")
    qemu_proc = start_qemu_vm(qcow2_path, server_port=5050)

    # 6. Set up agent
    # We need to add our agent code to the path
    # For now, use a minimal inline agent that calls llama-server
    from run_osworld_modal_agent import ModalGemma4Agent

    agent = ModalGemma4Agent(
        llamacpp_url="http://localhost:8080/v1",
        max_tokens=2048,
    )

    # 7. Load test cases
    with open("/opt/OSWorld/evaluation_examples/test_all.json") as f:
        test_all = json.load(f)

    if domain != "all":
        test_all = {domain: test_all.get(domain, [])}

    # 8. Run evaluation
    scores = []
    total = sum(len(v) for v in test_all.values())
    if max_tasks > 0:
        total = min(total, max_tasks)

    print(f"\nRunning {total} tasks across domains: {list(test_all.keys())}")
    print("=" * 60)

    task_count = 0
    for dom, example_ids in test_all.items():
        for example_id in example_ids:
            if max_tasks > 0 and task_count >= max_tasks:
                break

            config_file = f"/opt/OSWorld/evaluation_examples/examples/{dom}/{example_id}.json"
            with open(config_file) as f:
                example = json.load(f)

            instruction = example["instruction"]
            print(f"\n[{dom}] {example_id}")
            print(f"  Task: {instruction}")

            # Run the agent on this task
            try:
                score = run_single_task(
                    agent, example, instruction, max_steps,
                    server_port=5050, result_dir=f"/data/results/{dom}/{example_id}",
                )
                scores.append(score)
                print(f"  Score: {score}")
            except Exception as e:
                print(f"  Error: {e}")
                scores.append(0.0)

            task_count += 1

    # 9. Report results
    if scores:
        avg = sum(scores) / len(scores) * 100
        passed = sum(1 for s in scores if s > 0)
        print("\n" + "=" * 60)
        print(f"  Success rate: {avg:.1f}%")
        print(f"  Passed: {passed}/{len(scores)}")
        print("=" * 60)

    # Save results to volume
    results = {
        "domain": domain,
        "scores": scores,
        "success_rate": sum(scores) / len(scores) * 100 if scores else 0,
        "total_tasks": len(scores),
        "passed": sum(1 for s in scores if s > 0),
    }
    os.makedirs("/data/results", exist_ok=True)
    with open("/data/results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    # Cleanup
    llama_proc.terminate()
    qemu_proc.terminate()

    return results


def run_single_task(agent, example, instruction, max_steps, server_port, result_dir):
    """Run a single OSWorld task using direct QEMU VM connection."""
    import requests

    os.makedirs(result_dir, exist_ok=True)

    # Reset agent
    agent.reset()

    # Get initial screenshot
    screenshot_url = f"http://localhost:{server_port}/screenshot"
    execute_url = f"http://localhost:{server_port}/execute"

    for step in range(max_steps):
        # Get screenshot
        try:
            resp = requests.get(screenshot_url, timeout=30)
            screenshot_bytes = resp.content
        except Exception as e:
            print(f"    Step {step+1}: Screenshot failed: {e}")
            break

        # Save screenshot
        with open(os.path.join(result_dir, f"step_{step+1}.png"), "wb") as f:
            f.write(screenshot_bytes)

        # Agent predict
        obs = {"screenshot": screenshot_bytes, "accessibility_tree": None}
        response, actions = agent.predict(instruction, obs)

        print(f"    Step {step+1}: {actions[0] if actions else 'no action'}")

        # Check for terminal actions
        if not actions:
            break

        for action in actions:
            if isinstance(action, str):
                if action == "DONE":
                    # Task complete — evaluate
                    return evaluate_task(example, server_port)
                elif action == "FAIL":
                    return 0.0
                elif action == "WAIT":
                    time.sleep(2)
                    continue

                # Execute pyautogui code on the VM
                try:
                    payload = {"action": action}
                    requests.post(execute_url, json=payload, timeout=30)
                except Exception as e:
                    print(f"    Action failed: {e}")

            time.sleep(1)

    # Max steps reached — evaluate anyway
    return evaluate_task(example, server_port)


def evaluate_task(example, server_port):
    """Run OSWorld's evaluation for the current task."""
    # OSWorld uses custom evaluation functions per task
    # For now, return 0.0 (proper eval requires the full OSWorld env)
    # TODO: Integrate with OSWorld's evaluation module
    return 0.0


@app.local_entrypoint()
def main(domain: str = "os", max_steps: int = 15, max_tasks: int = 5):
    """CLI entrypoint for Modal deployment."""
    print(f"Running OSWorld eval on Modal (domain={domain}, max_tasks={max_tasks})")
    result = run_osworld_eval.remote(
        domain=domain, max_steps=max_steps, max_tasks=max_tasks
    )
    print(f"\nResult: {result}")
