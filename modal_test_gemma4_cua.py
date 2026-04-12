"""Test fine-tuned Gemma4-CUA model on web tasks.

Serves Gemma4-31B + LoRA adapter via vLLM, runs CRM test suite,
compares against base model.

Usage:
    # Test fine-tuned model
    modal run modal_test_gemma4_cua.py --task-file tasks/crm/original_test.json

    # Compare fine-tuned vs base
    modal run modal_test_gemma4_cua.py --task-file tasks/crm/original_test.json --baseline
"""

import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("gemma4-cua-test")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

BASE_MODEL = "google/gemma-4-31b-it"
ADAPTER_DIR = "/data/training/gemma4-31b-cua"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget")
    .pip_install(
        "vllm>=0.12.0",
        "openai", "requests", "pillow", "playwright",
        "huggingface-hub", "transformers", "torch", "peft",
    )
    .run_commands("playwright install --with-deps chromium || true")
    .add_local_python_source("mantis_agent")
)


def start_vllm_gemma4(adapter_dir: str | None, port: int = 8000) -> subprocess.Popen:
    """Start vLLM with optional LoRA adapter."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--trust-remote-code",
        "--tensor-parallel-size", "2",
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "16384",
    ]

    model_name = "gemma4-base"
    if adapter_dir and os.path.exists(adapter_dir):
        cmd.extend([
            "--enable-lora",
            "--lora-modules", f"gemma4-cua={adapter_dir}",
        ])
        model_name = "gemma4-cua"
        print(f"Starting vLLM with LoRA adapter: {adapter_dir}")
    else:
        print(f"Starting vLLM base model (no adapter)")

    cmd.extend(["--served-model-name", model_name])

    print(f"vLLM command: {' '.join(cmd[-8:])}")
    proc = subprocess.Popen(cmd, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)

    import requests
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready on :{port} ({i*5}s) — model: {model_name}")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"vLLM crashed: {open('/tmp/vllm.log').read()[-2000:]}")
            raise RuntimeError("vLLM crashed")
        time.sleep(5)
    raise RuntimeError("vLLM timeout")


@app.function(
    gpu="A100-80GB:2",
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    memory=65536,
    cpu=8,
)
def test_model(
    task_file_contents: str,
    use_adapter: bool = True,
    max_steps: int = 30,
    max_retries: int = 2,
):
    """Test Gemma4 (base or fine-tuned) on web tasks."""
    import requests as req
    from datetime import datetime, timezone

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    from mantis_agent.gym.runner import GymRunner

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # Start vLLM
    adapter = ADAPTER_DIR if use_adapter else None
    vllm_proc = start_vllm_gemma4(adapter)
    model_name = "gemma4-cua" if use_adapter and os.path.exists(ADAPTER_DIR) else "gemma4-base"

    r = req.get("http://localhost:8000/v1/models")
    print(f"Model: {r.json()['data'][0]['id']}")

    # Create brain (Gemma4 uses tool-calling, not pyautogui)
    brain = LlamaCppBrain(
        base_url="http://localhost:8000/v1",
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
    )
    brain.load()

    # Parse tasks
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "test")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\nTesting {model_name} on {len(tasks)} tasks")

    # Create env
    env = PlaywrightGymEnv(
        start_url=base_url, viewport=(1280, 720),
        headless=True, browser_type="chromium",
        session_dir="/data/sessions", settle_time=1.5,
    )

    # Run tasks
    scores = []
    task_details = []

    for i, task_config in enumerate(tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]
        print(f"\nTask {i+1}/{len(tasks)}: {task_id}")

        task_start = time.time()
        try:
            runner = GymRunner(brain=brain, env=env, max_steps=max_steps)
            result = runner.run(task=intent, task_id=task_id)

            # Verify
            verified = False
            vc = task_config.get("verify", {})
            vtype, value = vc.get("type", ""), vc.get("value", "")
            try:
                if vtype == "url_contains":
                    verified = value.lower() in env.current_url.lower()
                elif vtype == "url_not_contains":
                    verified = value.lower() not in env.current_url.lower()
            except Exception:
                pass

            success = result.success or verified
            scores.append(1.0 if success else 0.0)
            task_details.append({
                "task_id": task_id, "success": success,
                "steps": result.total_steps,
                "termination_reason": result.termination_reason,
            })
            print(f"  {'PASS' if success else 'FAIL'} ({result.total_steps} steps)")

        except Exception as e:
            scores.append(0.0)
            task_details.append({"task_id": task_id, "success": False, "error": str(e)})
            print(f"  ERROR: {e}")

    env.close()
    vllm_proc.terminate()

    passed = sum(1 for s in scores if s > 0)
    total_time = time.time() - t0
    pct = passed / len(scores) * 100 if scores else 0

    # Save results
    results_path = f"/data/results/gemma4cua_test_{model_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "model": model_name, "use_adapter": use_adapter,
            "passed": passed, "total": len(scores), "score": pct,
            "total_gpu_time_s": round(total_time),
            "task_details": task_details,
        }, f, indent=2)
    vol.commit()

    print(f"\n{'='*50}")
    print(f"{model_name}: {passed}/{len(scores)} = {pct:.0f}%")
    print(f"GPU time: {total_time/60:.0f}min | Results: {results_path}")
    print(f"{'='*50}")

    return {"model": model_name, "passed": passed, "total": len(scores), "score": pct}


@app.local_entrypoint()
def main(
    task_file: str = "tasks/crm/original_test.json",
    baseline: bool = False,
    max_steps: int = 30,
):
    """Test Gemma4-CUA on web tasks."""
    print(f"Gemma4-CUA Test")
    print(f"  Tasks: {task_file}")
    print(f"  Baseline: {baseline}")

    with open(task_file) as f:
        contents = f.read()

    # Test with adapter
    result_cua = test_model.remote(contents, use_adapter=True, max_steps=max_steps)
    print(f"\nFine-tuned: {json.dumps(result_cua, indent=2)}")

    if baseline:
        # Test without adapter
        result_base = test_model.remote(contents, use_adapter=False, max_steps=max_steps)
        print(f"\nBase model: {json.dumps(result_base, indent=2)}")

        print(f"\n{'='*50}")
        print(f"COMPARISON:")
        print(f"  Fine-tuned (CUA): {result_cua['passed']}/{result_cua['total']} = {result_cua['score']:.0f}%")
        print(f"  Base (Gemma4):    {result_base['passed']}/{result_base['total']} = {result_base['score']:.0f}%")
        print(f"{'='*50}")
