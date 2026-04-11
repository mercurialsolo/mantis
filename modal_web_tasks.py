"""Run Mantis CUA against live web tasks on Modal — Playwright or QEMU VM.

Two modes:
  - playwright: Headless Chromium, fast, good for most web tasks
  - qemu: Full Ubuntu desktop VM with Firefox, same as OSWorld — use when
    the task needs desktop-level interaction or a full browser profile

Architecture (playwright mode):
    Modal A100 Container
    ├── llama-server (Gemma4 on GPU, port 8080)
    ├── Playwright + Chromium (headless, 1280x720)
    └── GymRunner (frame history, loop detection)

Architecture (qemu mode):
    Modal A100 Container
    ├── llama-server (Gemma4 on GPU, port 8080)
    └── QEMU Ubuntu VM
        ├── Firefox/Chrome (full desktop)
        ├── pyautogui execution via Python controller
        └── Screenshots via /screenshot endpoint

Usage:
    # Playwright mode (default, fast)
    modal run modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json

    # QEMU VM mode (full desktop)
    modal run modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json --mode qemu

    # Detached
    modal run --detach modal_web_tasks.py --task-file tasks/crm/staffai_tasks.json
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
    plan_files: dict[str, str] | None = None,
    plan_inputs: dict[str, str] | None = None,
    mode: str = "playwright",
    max_steps: int = 30,
    frames_per_inference: int = 5,
):
    """Run Mantis agent against live web tasks.

    Supports two modes of task definition:
    1. JSON task suite (task_file_contents) — freeform intents
    2. Plan files (plan_files) — structured step-by-step plans

    When plan files are provided, each task gets an explicit numbered plan
    injected into the prompt so the model follows it instead of generating
    its own plan from scratch.

    Args:
        task_file_contents: JSON string of the task suite config.
        plan_files: Dict of task_id → plan file contents (YAML/JSON/text).
        plan_inputs: Dict of input_name → value for resolving plan placeholders.
        mode: "playwright" (headless Chromium) or "qemu" (full desktop VM).
        max_steps: Maximum steps per task.
        frames_per_inference: Number of recent frames to feed the brain.
    """
    import subprocess
    import requests
    from datetime import datetime, timezone
    from PIL import Image
    from io import BytesIO

    from mantis_agent.brain_llamacpp import LlamaCppBrain
    from mantis_agent.gym.runner import GymRunner
    from mantis_agent.gym.plans import Plan, load_text_plan, PlanInput, PlanStep
    from mantis_agent.actions import ActionType

    plan_files = plan_files or {}
    plan_inputs = plan_inputs or {}

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

    # 2. Parse task suite and load plans
    task_suite = json.loads(task_file_contents)
    session_name = task_suite.get("session_name", "web_task")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    # Parse plan files into Plan objects
    parsed_plans: dict[str, Plan] = {}
    for task_id, plan_content in plan_files.items():
        try:
            # Write to temp file for parsing (load_text_plan expects a path)
            import tempfile
            suffix = ".txt" if not plan_content.strip().startswith("{") and not plan_content.strip().startswith("name:") else ".yaml"
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write(plan_content)
                tmp_path = f.name
            from mantis_agent.gym.plans import load_plan
            parsed_plans[task_id] = load_plan(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            print(f"  Warning: Failed to parse plan for {task_id}: {e}")

    print(f"\n{'='*60}")
    print(f"Mantis — Web Task Benchmark")
    print(f"  Mode:     {mode}")
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

    # 4. Create environment based on mode
    session_dir = "/data/sessions"
    os.makedirs(session_dir, exist_ok=True)
    qemu_proc = None

    if mode == "qemu":
        env, qemu_proc = _create_qemu_env(base_url, session_dir, requests)
    else:
        from mantis_agent.gym.playwright_env import PlaywrightGymEnv
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

            # Set up plan executor for direct DOM execution
            executor = None
            active_plan = None
            if task_id in parsed_plans:
                active_plan = parsed_plans[task_id]
                resolved_intent, missing = active_plan.resolve_inputs(plan_inputs)
                if missing:
                    print(f"  WARNING: Plan missing required inputs: {missing}")
                    active_plan = None
                else:
                    intent = resolved_intent
                    print(f"  Plan loaded: {len(active_plan.steps)} steps")
                    # Create executor if env has a Playwright page
                    if hasattr(env, 'page') and env.page is not None:
                        from mantis_agent.gym.plan_executor import PlanExecutor
                        executor = PlanExecutor(page=env.page, settle_time=1.5)
                        print(f"  Direct execution: enabled (Playwright)")

            runner = GymRunner(
                brain=brain,
                env=env,
                max_steps=max_steps,
                frames_per_inference=frames_per_inference,
                plan_executor=executor,
            )

            result = runner.run(
                task=intent,
                task_id=task_id,
                plan=active_plan,
                plan_inputs=plan_inputs,
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
    if qemu_proc:
        qemu_proc.kill()

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


def _create_qemu_env(base_url: str, session_dir: str, requests_mod):
    """Boot a QEMU Ubuntu VM and return a QemuGymEnv + the QEMU process."""
    import subprocess

    qcow2_path = "/data/Ubuntu.qcow2"
    if not os.path.exists(qcow2_path):
        print("Downloading Ubuntu VM image...")
        from desktop_env.providers.docker.manager import _download_vm
        _download_vm("/data")
        vol.commit()

    qemu_proc = subprocess.Popen([
        "qemu-system-x86_64",
        "-m", "4G", "-smp", "4",
        "-drive", f"file={qcow2_path},format=qcow2,if=virtio,snapshot=on",
        "-nographic",
        "-net", "user,hostfwd=tcp::5050-:5000,hostfwd=tcp::9222-:9222",
        "-net", "nic,model=virtio",
        "-accel", "tcg,thread=multi",
        "-cpu", "max",
    ], stdout=open("/tmp/qemu.log", "w"), stderr=subprocess.STDOUT)

    # Wait for VM to boot
    vm_ready = False
    for i in range(120):
        try:
            r = requests_mod.get("http://localhost:5050/screenshot", timeout=5)
            if r.status_code == 200:
                print(f"VM ready! ({i*2}s)")
                vm_ready = True
                break
        except Exception:
            pass
        if qemu_proc.poll() is not None:
            raise RuntimeError(f"QEMU exited with code {qemu_proc.returncode}")
        if i % 30 == 0 and i > 0:
            print(f"  Still booting... ({i*2}s)")
        time.sleep(2)

    if not vm_ready:
        qemu_proc.kill()
        raise RuntimeError("VM boot timeout")

    # Open the target URL in Firefox inside the VM
    from desktop_env.controllers.python import PythonController
    controller = PythonController(vm_ip="localhost", server_port=5050)

    # Launch Firefox with the target URL
    controller.execute_python_command(
        f"import subprocess, os; "
        f"env = os.environ.copy(); env['DISPLAY'] = ':0'; "
        f"subprocess.Popen(['firefox', '{base_url}'], env=env); "
        f"import time; time.sleep(5)"
    )

    env = QemuGymEnv(
        controller=controller,
        vm_url="http://localhost:5050",
        base_url=base_url,
        session_dir=session_dir,
    )
    return env, qemu_proc


class QemuGymEnv:
    """GymEnvironment backed by a QEMU Ubuntu VM.

    Screenshots come from the VM's /screenshot endpoint.
    Actions are executed via pyautogui inside the VM through PythonController.
    """

    def __init__(self, controller, vm_url: str, base_url: str, session_dir: str):
        self._controller = controller
        self._vm_url = vm_url
        self._base_url = base_url
        self._session_dir = session_dir
        self._viewport = (1280, 720)

    def reset(self, task, **kwargs):
        """Navigate to start URL and return initial screenshot."""
        from mantis_agent.gym.base import GymObservation

        start_url = kwargs.get("start_url", self._base_url)
        # Open URL in Firefox inside the VM
        self._controller.execute_python_command(
            f"import subprocess, os; "
            f"env = os.environ.copy(); env['DISPLAY'] = ':0'; "
            f"subprocess.Popen(['firefox', '{start_url}'], env=env); "
            f"import time; time.sleep(3)"
        )
        time.sleep(2)
        return self._capture()

    def step(self, action):
        """Execute action via pyautogui inside the VM."""
        import requests
        from mantis_agent.gym.base import GymObservation, GymResult
        from mantis_agent.actions import ActionType

        code = self._action_to_pyautogui(action)
        if code:
            self._controller.execute_python_command(code)
            time.sleep(1.5)

        obs = self._capture()
        return GymResult(observation=obs, reward=0.0, done=False, info={})

    def close(self):
        pass

    @property
    def screen_size(self):
        return self._viewport

    @property
    def current_url(self):
        return self._base_url

    @property
    def page(self):
        return None

    def has_session(self, name):
        return os.path.exists(os.path.join(self._session_dir, f"{name}_state.json"))

    def load_session(self, name):
        pass  # QEMU sessions use Firefox profile, not Playwright state

    def save_session(self, name):
        # Save a marker file so has_session() returns True
        path = os.path.join(self._session_dir, f"{name}_state.json")
        import json
        with open(path, "w") as f:
            json.dump({"mode": "qemu", "saved_at": time.time()}, f)
        return path

    def _capture(self):
        """Get screenshot from VM."""
        import requests
        from mantis_agent.gym.base import GymObservation
        from PIL import Image
        from io import BytesIO

        resp = requests.get(f"{self._vm_url}/screenshot", timeout=30)
        img = Image.open(BytesIO(resp.content))
        # Resize to match viewport for consistent model input
        if img.size[0] > self._viewport[0]:
            img = img.resize(self._viewport, Image.LANCZOS)
        return GymObservation(screenshot=img)

    @staticmethod
    def _action_to_pyautogui(action):
        """Convert Mantis Action to pyautogui code for VM execution."""
        from mantis_agent.actions import ActionType

        match action.action_type:
            case ActionType.CLICK:
                x, y = action.params["x"], action.params["y"]
                # Scale 1280x720 → 1920x1080
                sx, sy = int(x * 1.5), int(y * 1.5)
                button = action.params.get("button", "left")
                return f"import pyautogui; pyautogui.click({sx}, {sy}, button='{button}')"
            case ActionType.DOUBLE_CLICK:
                x, y = action.params["x"], action.params["y"]
                sx, sy = int(x * 1.5), int(y * 1.5)
                return f"import pyautogui; pyautogui.doubleClick({sx}, {sy})"
            case ActionType.TYPE:
                text = action.params["text"].replace("'", "\\'")
                return f"import pyautogui; pyautogui.typewrite('{text}', interval=0.02)"
            case ActionType.KEY_PRESS:
                keys = action.params["keys"]
                parts = [k.strip() for k in keys.split("+")]
                if len(parts) == 1:
                    return f"import pyautogui; pyautogui.press('{parts[0]}')"
                keys_repr = ", ".join(f"'{k}'" for k in parts)
                return f"import pyautogui; pyautogui.hotkey({keys_repr})"
            case ActionType.SCROLL:
                direction = action.params["direction"]
                amount = action.params.get("amount", 3)
                clicks = amount if direction == "up" else -amount
                return f"import pyautogui; pyautogui.scroll({clicks})"
            case ActionType.DRAG:
                sx, sy = int(action.params["start_x"] * 1.5), int(action.params["start_y"] * 1.5)
                ex, ey = int(action.params["end_x"] * 1.5), int(action.params["end_y"] * 1.5)
                return f"import pyautogui; pyautogui.moveTo({sx}, {sy}); pyautogui.drag({ex-sx}, {ey-sy}, duration=0.5)"
            case ActionType.WAIT:
                seconds = action.params.get("seconds", 1.0)
                return f"import time; time.sleep({min(seconds, 5.0)})"
            case ActionType.DONE:
                return None
        return None


def _verify_task(env, verify_config: dict) -> bool:
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
    plan_dir: str = "plans/crm",
    mode: str = "playwright",
    max_steps: int = 30,
    inputs: str = "",
):
    """Run Mantis against live web tasks on Modal A100.

    Args:
        task_file: Path to task suite JSON.
        plan_dir: Directory containing .txt/.yaml plan files.
            Plans are matched to tasks by filename = task_id.
        mode: "playwright" (headless, fast) or "qemu" (full desktop VM).
        max_steps: Max steps per task.
        inputs: Comma-separated key=value pairs for plan inputs.
            Example: "password=skynet99,industry_value=Space Exploration"
    """
    import glob

    print(f"Mantis — Web Task Benchmark (Modal)")
    print(f"  Task file: {task_file}")
    print(f"  Plan dir:  {plan_dir}")
    print(f"  Mode:      {mode}")
    print(f"  Model:     Gemma4 {GEMMA4_MODEL}")
    print(f"  Max steps: {max_steps}")

    with open(task_file) as f:
        task_file_contents = f.read()

    # Load plan files from plan_dir, keyed by filename stem (= task_id)
    plan_files: dict[str, str] = {}
    if os.path.isdir(plan_dir):
        for ext in ("*.txt", "*.yaml", "*.yml", "*.json"):
            for plan_path in glob.glob(os.path.join(plan_dir, ext)):
                task_id = os.path.splitext(os.path.basename(plan_path))[0]
                with open(plan_path) as f:
                    plan_files[task_id] = f.read()
        print(f"  Plans:     {list(plan_files.keys())}")

    # Parse inputs
    plan_inputs: dict[str, str] = {}
    if inputs:
        for pair in inputs.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                plan_inputs[k.strip()] = v.strip()
    if plan_inputs:
        print(f"  Inputs:    {list(plan_inputs.keys())}")
    print()

    result = run_web_tasks.remote(
        task_file_contents=task_file_contents,
        plan_files=plan_files,
        plan_inputs=plan_inputs,
        mode=mode,
        max_steps=max_steps,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
