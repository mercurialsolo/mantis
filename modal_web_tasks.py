"""Run Mantis CUA against live web tasks on Modal — Playwright + Gemma4.

Same architecture as OSWorld Modal runs but instead of QEMU VM, we use
Playwright+Chromium to drive a real browser against live web apps (CRMs,
SaaS tools, etc.). Gemma4 runs on A100 GPU via llama-server.

Architecture:
    Modal A100 Container
    ├── llama-server (Gemma4 on GPU, port 8080)
    ├── Playwright + Chromium (headless, 1280x720)
    ├── PlaywrightGymEnv (translates Mantis actions ↔ Playwright)
    ├── GymRunner (frame history, loop detection, trajectory)
    └── Session persistence (.sessions/ for login state)

Usage:
    # Run CRM tasks
    modal run modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json

    # Detached (background)
    modal run --detach modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json

    # Override model
    GEMMA4_MODEL=26B modal run modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json
"""

import json
import os
import sys
import time

import modal

from modal_osworld_direct import (
    GEMMA4_MODEL,
    GGUF_CONFIGS,
    download_model,
    start_llama_server,
    image as base_image,
    vol,
)

app = modal.App("gemma4-web-tasks")

# Extend the base OSWorld image with our gym module
image = (
    base_image
    .add_local_python_source("mantis_agent")
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=3600,
    memory=32768,
    cpu=8,
)
def run_web_tasks(
    task_file_contents: str,
    max_steps: int = 30,
    frames_per_inference: int = 5,
):
    """Run Mantis agent against live web tasks.

    Args:
        task_file_contents: JSON string of the task suite config.
        max_steps: Maximum steps per task.
        frames_per_inference: Number of recent frames to feed the brain.
    """
    import requests
    from datetime import datetime, timezone
    from PIL import Image
    from io import BytesIO

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    from mantis_agent.gym.runner import GymRunner
    from mantis_agent.actions import ActionType

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # 1. Download model + start llama-server
    model_path = download_model("/data")
    cfg = GGUF_CONFIGS[GEMMA4_MODEL]
    print(f"Starting Gemma4 {GEMMA4_MODEL} on A100...")
    llama_proc = start_llama_server(model_path)

    r = requests.get("http://localhost:8080/v1/models")
    print(f"Model: {r.json()['data'][0]['id']}")

    # 2. Parse task suite
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "web_task")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    print(f"\n{'='*60}")
    print(f"Mantis — Web Task Benchmark")
    print(f"  Session:  {session_name}")
    print(f"  Base URL: {base_url}")
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Model:    Gemma4 {GEMMA4_MODEL}")
    print(f"  Steps:    {max_steps}")
    print(f"{'='*60}")

    # 3. Create brain
    brain = LlamaCppBrain(
        base_url="http://localhost:8080/v1",
        model=cfg["model_file"],
        max_tokens=2048,
        temperature=0.0,
    )
    brain.load()

    # 4. Create Playwright environment
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

    # 5. Order tasks: login first if no session exists
    login_tasks = [t for t in tasks if t.get("save_session")]
    other_tasks = [t for t in tasks if not t.get("save_session")]

    if not env.has_session(session_name) and login_tasks:
        ordered_tasks = login_tasks + other_tasks
    else:
        ordered_tasks = tasks

    # 6. Run each task
    scores = []
    task_details = []
    results_path = f"/data/results/web_results_{session_name}_{run_id}.json"
    os.makedirs("/data/results", exist_ok=True)

    def save_progress():
        """Save intermediate results to volume for live monitoring."""
        completed_at = datetime.now(timezone.utc).isoformat() if len(scores) == len(ordered_tasks) else ""
        summary = {
            "run_id": run_id,
            "benchmark": "web_tasks",
            "session_name": session_name,
            "base_url": base_url,
            "domain": session_name,
            "model": f"gemma-4-{GEMMA4_MODEL}-Q4_K_M",
            "tasks_run": len(ordered_tasks),
            "started_at": started_at,
            "completed_at": completed_at,
            "total_gpu_time_s": round(time.time() - t0),
            "estimated_cost_usd": round((time.time() - t0) / 3600 * 2.50, 2),
            "scores": scores,
            "task_details": task_details,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        vol.commit()

    for i, task_config in enumerate(ordered_tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]
        start_url = task_config.get("start_url", base_url)

        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(ordered_tasks)}: {task_id}")
        print(f"Intent: {intent[:120]}")
        print(f"{'='*60}")

        task_start = time.time()

        try:
            # Restore session for authenticated tasks
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)
                print(f"  Session '{session_name}' restored")

            runner = GymRunner(
                brain=brain,
                env=env,
                max_steps=max_steps,
                frames_per_inference=frames_per_inference,
            )

            result = runner.run(
                task=intent,
                task_id=task_id,
            )

            task_duration = time.time() - task_start

            # Save session after login
            if task_config.get("save_session"):
                current_url = env.current_url
                if result.success or (current_url and "login" not in current_url.lower()):
                    saved_path = env.save_session(session_name)
                    print(f"  Session saved: {saved_path}")

            # Verify task completion
            verify_config = task_config.get("verify", {})
            verified = _verify_task(env, verify_config)

            success = result.success or verified
            score = 1.0 if success else 0.0
            scores.append(score)

            detail = {
                "task_id": task_id,
                "instruction": intent,
                "success": success,
                "agent_done": result.success,
                "verified": verified,
                "steps": result.total_steps,
                "duration_s": round(task_duration),
                "termination_reason": result.termination_reason,
                "final_url": env.current_url,
                "trajectory": [
                    {
                        "step": s.step,
                        "action": str(s.action)[:200],
                        "thinking": s.thinking[:300] if s.thinking else "",
                        "inference_time": round(s.inference_time, 2),
                    }
                    for s in result.trajectory
                ],
            }
            task_details.append(detail)

            status = "PASS" if success else "FAIL"
            print(f"  Result: {status} ({result.total_steps} steps, {task_duration:.0f}s)")
            print(f"  Verified: {verified} | Agent done: {result.success}")
            print(f"  URL: {env.current_url}")

        except Exception as e:
            task_duration = time.time() - task_start
            print(f"  ERROR: {type(e).__name__}: {e}")
            scores.append(0.0)
            task_details.append({
                "task_id": task_id,
                "instruction": intent,
                "success": False,
                "error": str(e),
                "steps": 0,
                "duration_s": round(task_duration),
            })

        # Save progress after each task
        save_progress()

    # 7. Final summary
    env.close()
    llama_proc.terminate()

    passed = sum(1 for s in scores if s > 0)
    total_time = time.time() - t0
    avg = sum(scores) / len(scores) * 100 if scores else 0

    print(f"\n{'='*60}")
    print(f"COMPLETE: {passed}/{len(scores)} passed ({avg:.1f}%)")
    print(f"GPU time: {total_time/60:.0f} min | Cost: ${total_time/3600*2.50:.2f}")
    print(f"Results: {results_path}")
    print(f"{'='*60}")

    save_progress()
    return {"passed": passed, "total": len(scores), "score": avg, "results_path": results_path}


def _verify_task(env: "PlaywrightGymEnv", verify_config: dict) -> bool:
    """Run verification checks against the current browser state."""
    if not verify_config:
        return False

    vtype = verify_config.get("type", "")
    value = verify_config.get("value", "")

    try:
        if vtype == "url_contains":
            return value.lower() in env.current_url.lower()

        elif vtype == "url_exact":
            return env.current_url == value

        elif vtype == "page_contains_text":
            if env.page:
                page_text = env.page.inner_text("body")
                return value.lower() in page_text.lower()

        elif vtype == "element_exists":
            if env.page:
                return env.page.query_selector(value) is not None

        elif vtype == "element_text":
            selector = verify_config.get("selector", "")
            if env.page and selector:
                el = env.page.query_selector(selector)
                if el:
                    return value.lower() in el.inner_text().lower()

    except Exception as e:
        print(f"  Verify error: {e}")

    return False


@app.local_entrypoint()
def main(
    task_file: str = "tasks/crm/staffai_tasks.json",
    max_steps: int = 30,
):
    """Run Mantis against live web tasks on Modal A100."""
    print(f"Mantis — Web Task Benchmark (Modal)")
    print(f"  Task file: {task_file}")
    print(f"  Model:     Gemma4 {GEMMA4_MODEL}")
    print(f"  Max steps: {max_steps}")
    print()

    # Read task file locally and send contents to Modal
    with open(task_file) as f:
        task_file_contents = f.read()

    result = run_web_tasks.remote(
        task_file_contents=task_file_contents,
        max_steps=max_steps,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
