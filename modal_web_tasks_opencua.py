"""Run Mantis CUA with OpenCUA/EvoCUA brain on Modal — vLLM + Playwright.

Supported models:
  EvoCUA-8B:   46.1% OSWorld, 1× A100 (BEST value)
  EvoCUA-32B:  56.7% OSWorld, 2× A100 (SOTA open-source)
  OpenCUA-32B: 34.8% OSWorld, 4× A100
  OpenCUA-72B: 45.0% OSWorld, 8× A100

Usage:
    # EvoCUA-8B (default — best single-GPU performance)
    modal run modal_web_tasks_opencua.py --task-file tasks/crm/original_test.json

    # EvoCUA-32B (SOTA)
    CUA_MODEL=evocua-32b modal run modal_web_tasks_opencua.py --task-file tasks/crm/original_test.json

    # OpenCUA-32B
    CUA_MODEL=opencua-32b modal run modal_web_tasks_opencua.py --task-file tasks/crm/original_test.json
"""

import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("opencua-web-tasks")

vol = modal.Volume.from_name("osworld-data", create_if_missing=True)

CUA_MODELS = {
    "evocua-8b": {
        "repo": "meituan/EvoCUA-8B-20260105",
        "name": "EvoCUA-8B",
        "tp": 1,
        "gpus": "A100-80GB",
    },
    "evocua-32b": {
        "repo": "meituan/EvoCUA-32B-20260105",
        "name": "EvoCUA-32B",
        "tp": 2,
        "gpus": "A100-80GB:2",
    },
    "opencua-32b": {
        "repo": "xlangai/OpenCUA-32B",
        "name": "OpenCUA-32B",
        "tp": 4,
        "gpus": "A100-80GB:4",
    },
    "opencua-72b": {
        "repo": "xlangai/OpenCUA-72B",
        "name": "OpenCUA-72B",
        "tp": 8,
        "gpus": "A100-80GB:8",
    },
}

CUA_MODEL_KEY = os.environ.get("CUA_MODEL", "evocua-8b")
CUA_CONFIG = CUA_MODELS.get(CUA_MODEL_KEY, CUA_MODELS["evocua-8b"])

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "curl", "wget")
    .pip_install(
        "vllm>=0.12.0",
        "openai", "requests", "pillow", "playwright",
        "huggingface-hub", "transformers", "torch",
    )
    .run_commands(
        "playwright install --with-deps chromium || true",
    )
    .add_local_python_source("mantis_agent")
)


def download_model(vol_path: str) -> str:
    """Download CUA model if not cached on volume."""
    slug = CUA_MODEL_KEY.replace("-", "_")
    model_dir = os.path.join(vol_path, "models", slug)
    marker = os.path.join(model_dir, ".download_complete")
    if os.path.exists(marker):
        print(f"{CUA_CONFIG['name']} cached at {model_dir}")
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading {CUA_CONFIG['repo']}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        CUA_CONFIG["repo"],
        local_dir=model_dir,
        ignore_patterns=["*.md", "*.txt"],
    )
    open(marker, "w").write("done")
    vol.commit()
    print("Download complete.")
    return model_dir


def start_vllm_server(model_dir: str, port: int = 8000, tp: int = 4) -> subprocess.Popen:
    """Start vLLM with OpenCUA model."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp),
        "--served-model-name", "opencua",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "32768",
    ]
    print(f"Starting vLLM: {' '.join(cmd[-10:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=open("/tmp/vllm.log", "w"),
        stderr=subprocess.STDOUT,
    )

    time.sleep(5)
    if proc.poll() is not None:
        print(f"vLLM crashed: {open('/tmp/vllm.log').read()[-3000:]}")
        raise RuntimeError("vLLM crashed")

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
            print(f"vLLM died: {open('/tmp/vllm.log').read()[-3000:]}")
            raise RuntimeError("vLLM died during startup")
        time.sleep(5)

    raise RuntimeError("vLLM startup timeout")


@app.function(
    gpu=CUA_CONFIG["gpus"],
    image=image,
    volumes={"/data": vol},
    timeout=7200,
    memory=65536,
    cpu=16,
)
def run_opencua_tasks(
    task_file_contents: str,
    plan_files: dict[str, str] | None = None,
    plan_inputs: dict[str, str] | None = None,
    max_steps: int = 30,
    max_retries: int = 2,
    frames_per_inference: int = 5,
):
    """Run OpenCUA-32B against web tasks."""
    import requests as req
    from datetime import datetime, timezone

    from mantis_agent.brain_opencua import OpenCUABrain
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    from mantis_agent.gym.runner import GymRunner

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # 1. Download + start vLLM
    model_dir = download_model("/data")
    tp = CUA_CONFIG["tp"]
    vllm_proc = start_vllm_server(model_dir, port=8000, tp=tp)

    r = req.get("http://localhost:8000/v1/models")
    print(f"Model: {r.json()['data'][0]['id']}")

    # 2. Parse task suite
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "web_task")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])
    plan_files = plan_files or {}
    plan_inputs = plan_inputs or {}

    print(f"\n{'='*60}")
    print(f"Mantis — {CUA_CONFIG['name']} Web Tasks")
    print(f"  Session:  {session_name}")
    print(f"  Base URL: {base_url}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Model:    {CUA_CONFIG['name']} (vLLM, TP={tp})")
    print(f"  Steps:    {max_steps}")
    print(f"{'='*60}")

    # 3. Create OpenCUA brain
    brain = OpenCUABrain(
        base_url="http://localhost:8000/v1",
        model="opencua",
        max_tokens=2048,
        temperature=0.0,
        screen_size=(1280, 720),
    )
    brain.load()

    # 4. Create Playwright env
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

    # 5. Parse plans
    parsed_plans = {}
    for task_id, plan_content in plan_files.items():
        try:
            import tempfile
            suffix = ".txt"
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write(plan_content)
                tmp_path = f.name
            from mantis_agent.gym.plans import load_plan
            parsed_plans[task_id] = load_plan(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"  Warning: plan parse failed for {task_id}: {e}")

    # 6. Order tasks
    login_tasks = [t for t in tasks if t.get("save_session")]
    other_tasks = [t for t in tasks if not t.get("save_session")]
    if not env.has_session(session_name) and login_tasks:
        ordered_tasks = login_tasks + other_tasks
    else:
        ordered_tasks = tasks

    # 7. Run tasks
    scores = []
    task_details = []
    results_path = f"/data/results/opencua_results_{session_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)

    def save_progress():
        completed_at = datetime.now(timezone.utc).isoformat() if len(scores) == len(ordered_tasks) else ""
        summary = {
            "run_id": run_id,
            "benchmark": "opencua_web_tasks",
            "session_name": session_name,
            "base_url": base_url,
            "domain": session_name,
            "model": CUA_CONFIG["name"],
            "tasks_run": len(ordered_tasks),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_gpu_time_s": round(time.time() - t0),
            "estimated_cost_usd": round((time.time() - t0) / 3600 * 10.0, 2),  # 4× A100
            "scores": scores,
            "task_details": task_details,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        vol.commit()

    for i, task_config in enumerate(ordered_tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]

        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(ordered_tasks)}: {task_id}")
        print(f"Intent: {intent[:120]}")
        print(f"{'='*60}")

        task_start = time.time()

        try:
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)

            # Set up executor if plans available
            executor = None
            discoverer = None
            active_plan = None
            if task_id in parsed_plans:
                active_plan = parsed_plans[task_id]
                resolved_intent, missing = active_plan.resolve_inputs(plan_inputs)
                if not missing:
                    intent = resolved_intent
                    from mantis_agent.gym.plan_executor import PlanExecutor
                    from mantis_agent.gym.page_discovery import PageDiscovery
                    executor = PlanExecutor(env=env, settle_time=1.5)
                    discoverer = PageDiscovery(env=env, max_elements=50)

            # Retry loop
            prior_learnings = ""
            success = False
            result = None

            for attempt in range(1, max_retries + 1):
                attempt_intent = intent + prior_learnings if prior_learnings else intent

                runner = GymRunner(
                    brain=brain,
                    env=env,
                    max_steps=max_steps,
                    frames_per_inference=frames_per_inference,
                    plan_executor=executor,
                    page_discovery=discoverer if active_plan else None,
                )

                result = runner.run(
                    task=attempt_intent,
                    task_id=task_id,
                    plan=active_plan,
                    plan_inputs=plan_inputs,
                )

                if task_config.get("save_session"):
                    current_url = env.current_url
                    if result.success or (current_url and "login" not in current_url.lower()):
                        env.save_session(session_name)

                verify_config = task_config.get("verify", {})
                # Inline verify
                verified = False
                vtype = verify_config.get("type", "")
                value = verify_config.get("value", "")
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

                if attempt < max_retries:
                    # Distill failure for next attempt
                    actions = [str(s.action)[:60] for s in result.trajectory]
                    clicks = sum(1 for a in actions if "click" in a)
                    types = sum(1 for a in actions if "type" in a)
                    prior_learnings = f"\n\nPRIOR ATTEMPT FAILED ({result.termination_reason}):"
                    if clicks > len(actions) * 0.6:
                        prior_learnings += f"\n- Too many clicks ({clicks}/{len(actions)}). Use typewrite() after clicking fields."
                    if types == 0:
                        prior_learnings += "\n- Never typed! Use typewrite() after clicking input fields."
                    prior_learnings += f"\n- Last actions: {' → '.join(actions[-5:])}"
                    print(f"  Attempt {attempt}: FAIL — retrying with learnings")

            task_duration = time.time() - task_start
            score = 1.0 if success else 0.0
            scores.append(score)

            task_details.append({
                "task_id": task_id,
                "instruction": intent[:200],
                "success": success,
                "steps": result.total_steps if result else 0,
                "duration_s": round(task_duration),
                "termination_reason": result.termination_reason if result else "error",
                "final_url": env.current_url,
            })

            status = "PASS" if success else "FAIL"
            print(f"  Result: {status} ({result.total_steps if result else 0} steps, {task_duration:.0f}s)")

        except Exception as e:
            task_duration = time.time() - task_start
            print(f"  ERROR: {e}")
            scores.append(0.0)
            task_details.append({
                "task_id": task_id, "success": False,
                "error": str(e), "duration_s": round(task_duration),
            })

        save_progress()

    env.close()
    vllm_proc.terminate()

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
    plan_dir: str = "",
    max_steps: int = 30,
    max_retries: int = 2,
    inputs: str = "",
):
    """Run OpenCUA-32B against web tasks on Modal."""
    import glob

    print(f"Mantis — OpenCUA-32B Web Tasks (Modal)")
    print(f"  Task file: {task_file}")
    print(f"  Model:     {CUA_CONFIG['name']} ({CUA_CONFIG['gpus']})")
    print(f"  Max steps: {max_steps}")

    with open(task_file) as f:
        task_file_contents = f.read()

    plan_files = {}
    if plan_dir and os.path.isdir(plan_dir):
        for ext in ("*.txt", "*.yaml", "*.yml"):
            for plan_path in glob.glob(os.path.join(plan_dir, ext)):
                tid = os.path.splitext(os.path.basename(plan_path))[0]
                with open(plan_path) as f:
                    plan_files[tid] = f.read()
        print(f"  Plans:     {list(plan_files.keys())}")

    plan_inputs = {}
    if inputs:
        for pair in inputs.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                plan_inputs[k.strip()] = v.strip()
    print()

    result = run_opencua_tasks.remote(
        task_file_contents=task_file_contents,
        plan_files=plan_files,
        plan_inputs=plan_inputs,
        max_steps=max_steps,
        max_retries=max_retries,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
