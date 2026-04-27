"""Test fine-tuned Gemma4-CUA model on web tasks.

Uses llama.cpp (not vLLM — vLLM doesn't support gemma4 model type yet).
Two modes:
  1. Fine-tuned: merge LoRA adapter into base, quantize to GGUF, serve via llama-server
  2. Baseline: use existing base Gemma4 GGUF from OSWorld runs

Usage:
    modal run modal_test_gemma4_cua.py --task-file tasks/crm/original_test.json
    modal run modal_test_gemma4_cua.py --task-file tasks/crm/original_test.json --baseline
"""

import json
import os
import subprocess
import sys
import time

import modal

from mantis_agent.modal_runtime import (
    download_model as download_base_model,
    start_llama_server as start_base_llama_server,
    image as base_image,
    vol,
)

app = modal.App("gemma4-cua-test")

ADAPTER_DIR = "/data/training/gemma4-31b-cua"

image = (
    base_image
    .add_local_python_source("mantis_agent")
)


def merge_and_quantize_adapter(adapter_dir: str, output_dir: str) -> str:
    """Merge LoRA adapter into base model and quantize to GGUF."""
    # Check for previously generated GGUF in output_dir or _gguf suffix dir
    for search_dir in [output_dir, output_dir + "_gguf"]:
        if os.path.isdir(search_dir):
            for f in os.listdir(search_dir):
                if f.endswith(".gguf") and "Q4_K_M" in f:
                    path = os.path.join(search_dir, f)
                    print(f"Merged GGUF cached at {path}")
                    return path

    print("Merging LoRA adapter + quantizing to GGUF...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            adapter_dir,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        vol.commit()
        # Unsloth saves to output_dir + "_gguf" subdirectory
        for search_dir in [output_dir, output_dir + "_gguf"]:
            if os.path.isdir(search_dir):
                for f in os.listdir(search_dir):
                    if f.endswith(".gguf") and "Q4_K_M" in f:
                        return os.path.join(search_dir, f)
    except Exception as e:
        print(f"Merge/quantize failed: {e}")
    return ""


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    memory=81920,
    cpu=8,
)
def test_model(
    task_file_contents: str,
    use_adapter: bool = True,
    max_steps: int = 30,
    max_retries: int = 2,
):
    """Test Gemma4 (base or fine-tuned) on web tasks via llama.cpp."""
    import requests as req
    from datetime import datetime, timezone

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    from mantis_agent.gym.runner import GymRunner

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    # Install deps at runtime
    subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "libssl-dev", "libcurl4-openssl-dev"], check=False, capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth[colab-new]", "peft", "--quiet"], check=False)

    if use_adapter and os.path.exists(ADAPTER_DIR):
        # Merge adapter + quantize to GGUF
        gguf_dir = "/data/training/gemma4-cua-gguf"
        gguf_path = merge_and_quantize_adapter(ADAPTER_DIR, gguf_dir)
        if not gguf_path:
            print("GGUF merge failed — falling back to base model")
            use_adapter = False

    if not use_adapter or not os.path.exists(ADAPTER_DIR):
        # Use base Gemma4 GGUF (same as OSWorld)
        gguf_path = download_base_model("/data")
        print(f"Using base model: {gguf_path}")

    model_name = "gemma4-cua" if use_adapter else "gemma4-base"

    # Start llama-server with correct mmproj
    if use_adapter and gguf_path:
        # Find mmproj in the same directory as the GGUF
        gguf_dir = os.path.dirname(gguf_path)
        mmproj = ""
        for f in os.listdir(gguf_dir):
            if "mmproj" in f.lower() and f.endswith(".gguf"):
                mmproj = os.path.join(gguf_dir, f)
                break
        # Custom llama-server start for CUA model
        cmd = [
            "/opt/llama.cpp/build/bin/llama-server",
            "-m", gguf_path,
            "--host", "0.0.0.0", "--port", "8080",
            "-ngl", "99", "-c", "32768", "-ub", "2048",
            "--jinja", "--reasoning-budget", "4096",
            "--flash-attn", "on",
        ]
        if mmproj:
            cmd.extend(["--mmproj", mmproj])
            print(f"Using mmproj: {mmproj}")
        print(f"Starting llama-server: {' '.join(cmd[-6:])}")
        llama_proc = subprocess.Popen(cmd, stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(3)
        import requests as _req
        for i in range(90):
            try:
                r = _req.get("http://localhost:8080/v1/models", timeout=2)
                if r.status_code == 200:
                    print(f"llama-server ready ({i*2}s)")
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print(f"llama-server timeout: {open('/tmp/llama.log').read()[-2000:]}")
            raise RuntimeError("llama-server timeout")
    else:
        llama_proc = start_base_llama_server(gguf_path)

    r = req.get("http://localhost:8080/v1/models")
    print(f"Model: {r.json()['data'][0]['id']}")

    brain = LlamaCppBrain(
        base_url="http://localhost:8080/v1",
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
        # CUA model outputs text actions, not structured tool calls
        use_tool_calling=not use_adapter,
    )
    brain.load()

    # Parse tasks
    task_suite = json.loads(task_file_contents)
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
    llama_proc.terminate()

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
    print("Gemma4-CUA Test")
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
        print("COMPARISON:")
        print(f"  Fine-tuned (CUA): {result_cua['passed']}/{result_cua['total']} = {result_cua['score']:.0f}%")
        print(f"  Base (Gemma4):    {result_base['passed']}/{result_base['total']} = {result_base['score']:.0f}%")
        print(f"{'='*50}")
