#!/usr/bin/env python3
"""Run CUA agent locally — uses your real Chrome/IP to bypass Cloudflare.

No Modal, no datacenter IPs. Runs on your machine with your residential IP.
Requires a running llama-server or vLLM endpoint for the brain.

Usage:
    # With local llama-server (Gemma4)
    python run_local.py --task-file tasks/boattrader/full_production.json \
        --brain-url http://localhost:8080/v1 --brain-type gemma4

    # With remote vLLM (EvoCUA on Modal)
    python run_local.py --task-file tasks/boattrader/full_production.json \
        --brain-url http://your-modal-endpoint/v1 --brain-type evocua

    # With headed browser (see what the agent does)
    python run_local.py --task-file tasks/boattrader/full_production.json \
        --brain-url http://localhost:8080/v1 --headed
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_local")


def main():
    parser = argparse.ArgumentParser(description="Run CUA agent locally")

    parser.add_argument("--task-file", required=True, help="Task suite JSON")
    parser.add_argument("--brain-url", default="http://localhost:8080/v1", help="Brain API endpoint")
    parser.add_argument("--brain-type", choices=["gemma4", "gemma4-cua", "evocua"], default="gemma4")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output", default="results/local_run.json")

    args = parser.parse_args()

    # Load tasks
    with open(args.task_file) as f:
        task_suite = json.load(f)

    session_name = task_suite.get("session_name", "local")
    base_url = task_suite.get("base_url", "")
    tasks = task_suite.get("tasks", [])

    # Create brain
    if args.brain_type in ("evocua",):
        from mantis_agent.brain_opencua import OpenCUABrain
        brain = OpenCUABrain(
            base_url=args.brain_url,
            model="model",
            max_tokens=2048,
            temperature=0.0,
        )
    else:
        from mantis_agent.brain_llamacpp import LlamaCppBrain
        brain = LlamaCppBrain(
            base_url=args.brain_url,
            model="model",
            max_tokens=2048,
            temperature=0.0,
            use_tool_calling=(args.brain_type == "gemma4"),
        )

    brain.load()
    logger.info(f"Brain: {args.brain_type} at {args.brain_url}")

    # Create local Playwright env — your real IP, not datacenter
    from mantis_agent.gym.playwright_env import PlaywrightGymEnv
    env = PlaywrightGymEnv(
        start_url=base_url,
        viewport=(1280, 720),
        headless=not args.headed,
        browser_type="chromium",
        session_dir=".sessions",
        settle_time=1.5,
    )

    from mantis_agent.gym.runner import GymRunner

    # Run tasks
    scores = []
    task_details = []
    t0 = time.time()

    for i, task_config in enumerate(tasks):
        task_id = task_config["task_id"]
        intent = task_config["intent"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(tasks)}: {task_id}")
        logger.info(f"{'='*60}")

        task_start = time.time()

        try:
            if task_config.get("require_session") and env.has_session(session_name):
                env.load_session(session_name)

            # Retry loop with learning
            prior_learnings = ""
            success = False
            result = None
            extracted_data = ""

            for attempt in range(1, args.max_retries + 1):
                attempt_intent = intent + prior_learnings if prior_learnings else intent

                runner = GymRunner(
                    brain=brain, env=env,
                    max_steps=args.max_steps,
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
                except Exception:
                    pass

                success = result.success or verified

                # Extract data from model's output
                if result and result.trajectory:
                    for step in reversed(result.trajectory):
                        if step.action.action_type.value == "done":
                            extracted_data = step.action.params.get("summary", "")
                            break
                        if step.thinking and len(step.thinking) > 20:
                            extracted_data = step.thinking
                            break

                if success:
                    logger.info(f"  Attempt {attempt}: PASS ({result.total_steps} steps)")
                    break

                if attempt < args.max_retries:
                    actions = [str(s.action)[:60] for s in result.trajectory]
                    prior_learnings = f"\n\nPRIOR ATTEMPT FAILED ({result.termination_reason})."
                    logger.info(f"  Attempt {attempt}: FAIL — retrying")

            task_duration = time.time() - task_start
            scores.append(1.0 if success else 0.0)
            task_details.append({
                "task_id": task_id,
                "success": success,
                "steps": result.total_steps if result else 0,
                "duration_s": round(task_duration),
                "termination_reason": result.termination_reason if result else "error",
                "final_url": env.current_url,
                "extracted_data": extracted_data[:1000],
            })

            status = "PASS" if success else "FAIL"
            logger.info(f"  Result: {status} ({result.total_steps if result else 0} steps, {task_duration:.0f}s)")
            if extracted_data:
                logger.info(f"  Data: {extracted_data[:200]}")

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            scores.append(0.0)
            task_details.append({"task_id": task_id, "success": False, "error": str(e)})

    env.close()

    # Save results
    passed = sum(1 for s in scores if s > 0)
    total_time = time.time() - t0
    pct = passed / len(scores) * 100 if scores else 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "brain": args.brain_type,
            "brain_url": args.brain_url,
            "local": True,
            "passed": passed,
            "total": len(scores),
            "score": pct,
            "total_time_s": round(total_time),
            "task_details": task_details,
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {passed}/{len(scores)} ({pct:.0f}%)")
    logger.info(f"Time: {total_time/60:.0f}min | Results: {args.output}")

    # Print extracted contacts
    contacts = []
    for td in task_details:
        data = td.get("extracted_data", "")
        if data and td.get("success"):
            contacts.append({"task": td["task_id"], "data": data[:300]})

    if contacts:
        logger.info(f"\nExtracted contacts ({len(contacts)}):")
        for c in contacts:
            logger.info(f"  {c['task']}: {c['data'][:150]}")
    else:
        logger.info("\nNo contacts extracted")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
