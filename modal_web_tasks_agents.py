"""Run Agent-S3 CUA against web tasks on Modal.

Agent-S3 uses a two-model architecture:
  Worker: reasoning agent (OpenCUA/EvoCUA via vLLM, or OpenAI/Claude)
  Grounding: element locator (UI-TARS-7B via vLLM)

Without a grounding model, falls back to API-only mode (worker model
does both reasoning and coordinate prediction from screenshots).

Usage:
    # With EvoCUA-32B as worker (API-only, no grounding model)
    modal run modal_web_tasks_agents.py \
      --task-file tasks/crm/original_test.json \
      --worker-model evocua-32b

    # With full Agent-S3 (worker + grounding)
    modal run modal_web_tasks_agents.py \
      --task-file tasks/crm/original_test.json \
      --worker-model evocua-32b \
      --grounding-model ui-tars-7b
"""

import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("agents3-web-tasks")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

# Model configs
WORKER_MODELS = {
    "evocua-8b": {"repo": "meituan/EvoCUA-8B-20260105", "tp": 1},
    "evocua-32b": {"repo": "meituan/EvoCUA-32B-20260105", "tp": 2},
    "opencua-32b": {"repo": "xlangai/OpenCUA-32B", "tp": 4},
}

GROUNDING_MODELS = {
    "ui-tars-7b": {"repo": "bytedance-research/UI-TARS-1.5-7B", "tp": 1},
}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget", "tesseract-ocr")
    .pip_install(
        "vllm>=0.12.0",
        "gui-agents",
        "openai", "requests", "pillow", "playwright",
        "huggingface-hub", "transformers", "torch",
        "pytesseract",
    )
    .run_commands("playwright install --with-deps chromium || true")
    .add_local_python_source("mantis_agent")
)


def download_model(vol_path: str, model_key: str, models_dict: dict) -> str:
    """Download a model if not cached."""
    cfg = models_dict[model_key]
    slug = model_key.replace("-", "_")
    model_dir = os.path.join(vol_path, "models", slug)
    marker = os.path.join(model_dir, ".download_complete")
    if os.path.exists(marker):
        print(f"{model_key} cached at {model_dir}")
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading {cfg['repo']}...")
    from huggingface_hub import snapshot_download
    snapshot_download(cfg["repo"], local_dir=model_dir, ignore_patterns=["*.md", "*.txt"])
    open(marker, "w").write("done")
    vol.commit()
    return model_dir


def start_vllm(model_dir: str, port: int, tp: int) -> subprocess.Popen:
    """Start vLLM server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--served-model-name", "model",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "16384",
    ]
    print(f"Starting vLLM on :{port} (TP={tp})")
    proc = subprocess.Popen(cmd, stdout=open(f"/tmp/vllm_{port}.log", "w"), stderr=subprocess.STDOUT)

    import requests
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready on :{port} ({i*5}s)")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"vLLM crashed: {open(f'/tmp/vllm_{port}.log').read()[-2000:]}")
            raise RuntimeError(f"vLLM crashed on port {port}")
        time.sleep(5)
    raise RuntimeError(f"vLLM timeout on port {port}")


@app.function(
    gpu="A100-80GB:4",
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_agents_tasks(
    task_file_contents: str,
    worker_model: str = "evocua-32b",
    grounding_model: str = "",
    max_steps: int = 50,
    max_retries: int = 5,
):
    """Run Agent-S3 brain against web tasks."""
    from datetime import datetime, timezone

    from mantis_agent.brain_agents import AgentSBrain
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    from mantis_agent.gym.runner import GymRunner

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # 1. Download and start worker model
    worker_cfg = WORKER_MODELS.get(worker_model)
    if not worker_cfg:
        raise ValueError(f"Unknown worker model: {worker_model}. Options: {list(WORKER_MODELS.keys())}")

    worker_dir = download_model("/data", worker_model, WORKER_MODELS)
    worker_proc = start_vllm(worker_dir, port=8000, tp=worker_cfg["tp"])

    # 2. Optionally start grounding model
    grounding_proc = None
    grounding_url = ""
    if grounding_model and grounding_model in GROUNDING_MODELS:
        grounding_dir = download_model("/data", grounding_model, GROUNDING_MODELS)
        grounding_proc = start_vllm(grounding_dir, port=8001, tp=GROUNDING_MODELS[grounding_model]["tp"])
        grounding_url = "http://localhost:8001/v1"

    # 3. Create Agent-S brain
    brain = AgentSBrain(
        worker_model="model",
        worker_provider="vllm",
        worker_base_url="http://localhost:8000/v1",
        grounding_model=grounding_model if grounding_url else "",
        grounding_provider="vllm" if grounding_url else "",
        grounding_url=grounding_url,
        screen_size=(1280, 720),
    )
    brain.load()

    # 4. Parse tasks
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "web_task")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\n{'='*60}")
    print(f"Mantis — Agent-S3 ({worker_model}) Web Tasks")
    print(f"  Worker:    {worker_model} (vLLM TP={worker_cfg['tp']})")
    print(f"  Grounding: {grounding_model or 'none (API-only)'}")
    print(f"  Tasks:     {len(tasks)}")
    print(f"  Steps:     {max_steps}, Retries: {max_retries}")
    print(f"{'='*60}")

    # 5. Create env
    session_dir = "/data/sessions"
    os.makedirs(session_dir, exist_ok=True)
    env = PlaywrightGymEnv(
        start_url=base_url,
        viewport=(1280, 720),
        headless=True,
        browser_type="chromium",
        session_dir=session_dir,
        settle_time=1.5,
    )

    # 6. Order tasks
    login_tasks = [t for t in tasks if t.get("save_session")]
    other_tasks = [t for t in tasks if not t.get("save_session")]
    ordered_tasks = (login_tasks + other_tasks) if login_tasks and not env.has_session(session_name) else tasks

    # 7. Run
    scores = []
    task_details = []
    results_path = f"/data/results/agents3_results_{session_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)

    def save_progress():
        completed_at = datetime.now(timezone.utc).isoformat() if len(scores) == len(ordered_tasks) else ""
        with open(results_path, "w") as f:
            json.dump({
                "run_id": run_id,
                "benchmark": "agents3_web_tasks",
                "session_name": session_name,
                "model": f"Agent-S3({worker_model})",
                "grounding": grounding_model or "none",
                "tasks_run": len(ordered_tasks),
                "started_at": started_at,
                "completed_at": completed_at,
                "total_gpu_time_s": round(time.time() - t0),
                "estimated_cost_usd": round((time.time() - t0) / 3600 * 10.0, 2),
                "scores": scores,
                "task_details": task_details,
            }, f, indent=2)
        vol.commit()

    for i, task_config in enumerate(ordered_tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]

        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(ordered_tasks)}: {task_id}")
        print(f"{'='*60}")

        task_start = time.time()

        try:
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)

            # Retry loop with learning
            prior_learnings = ""
            success = False
            result = None

            for attempt in range(1, max_retries + 1):
                attempt_intent = intent + prior_learnings if prior_learnings else intent

                runner = GymRunner(
                    brain=brain,
                    env=env,
                    max_steps=max_steps,
                    frames_per_inference=3,
                )

                result = runner.run(task=attempt_intent, task_id=task_id)

                # Session save
                if task_config.get("save_session"):
                    if result.success or ("login" not in env.current_url.lower()):
                        env.save_session(session_name)

                # Verify
                verified = False
                vc = task_config.get("verify", {})
                vtype, value = vc.get("type", ""), vc.get("value", "")
                try:
                    if vtype == "url_contains":
                        verified = value.lower() in env.current_url.lower()
                    elif vtype == "url_not_contains":
                        verified = value.lower() not in env.current_url.lower()
                    elif vtype == "page_contains_text" and env.page:
                        verified = value.lower() in env.page.inner_text("body").lower()
                except Exception:
                    pass

                success = result.success or verified
                if success:
                    print(f"  Attempt {attempt}: PASS")
                    break

                # Distill failure
                if attempt < max_retries:
                    actions = [str(s.action)[:60] for s in result.trajectory]
                    clicks = sum(1 for a in actions if "click" in a)
                    types = sum(1 for a in actions if "type" in a)
                    prior_learnings = f"\n\nPRIOR ATTEMPT {attempt} FAILED ({result.termination_reason}):"
                    if clicks > len(actions) * 0.6:
                        prior_learnings += f"\n- Too many clicks ({clicks}/{len(actions)}). Use typewrite() after clicking."
                    if types == 0:
                        prior_learnings += "\n- Never typed! Use typewrite() after clicking input fields."
                    prior_learnings += f"\n- Last actions: {' → '.join(actions[-5:])}"
                    prior_learnings += (
                        "\n- FOR LOGIN: click username field → typewrite('username') → "
                        "press('tab') → typewrite('password') → press('enter')"
                    )
                    print(f"  Attempt {attempt}: FAIL — retrying")

            task_duration = time.time() - task_start
            scores.append(1.0 if success else 0.0)
            task_details.append({
                "task_id": task_id,
                "success": success,
                "steps": result.total_steps if result else 0,
                "duration_s": round(task_duration),
                "termination_reason": result.termination_reason if result else "error",
                "final_url": env.current_url,
                "attempts": attempt,
            })

            print(f"  Result: {'PASS' if success else 'FAIL'} ({result.total_steps if result else 0} steps, attempt {attempt})")

        except Exception as e:
            print(f"  ERROR: {e}")
            scores.append(0.0)
            task_details.append({"task_id": task_id, "success": False, "error": str(e)})

        save_progress()

    env.close()
    worker_proc.terminate()
    if grounding_proc:
        grounding_proc.terminate()

    passed = sum(1 for s in scores if s > 0)
    total_time = time.time() - t0
    avg = sum(scores) / len(scores) * 100 if scores else 0

    print(f"\n{'='*60}")
    print(f"COMPLETE: {passed}/{len(scores)} ({avg:.1f}%)")
    print(f"GPU time: {total_time/60:.0f} min | Cost: ${total_time/3600*10:.2f}")
    print(f"{'='*60}")

    save_progress()
    return {"passed": passed, "total": len(scores), "score": avg}


@app.local_entrypoint()
def main(
    task_file: str = "tasks/crm/original_test.json",
    worker_model: str = "evocua-32b",
    grounding_model: str = "",
    max_steps: int = 50,
    max_retries: int = 5,
):
    print("Mantis — Agent-S3 Web Tasks (Modal)")
    print(f"  Worker:    {worker_model}")
    print(f"  Grounding: {grounding_model or 'none'}")
    print(f"  Steps:     {max_steps}, Retries: {max_retries}")
    print()

    with open(task_file) as f:
        task_file_contents = f.read()

    result = run_agents_tasks.remote(
        task_file_contents=task_file_contents,
        worker_model=worker_model,
        grounding_model=grounding_model,
        max_steps=max_steps,
        max_retries=max_retries,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
