"""Run OSWorld evaluation with our Gemma4 streaming CUA agent.

Usage:
    # Full evaluation via Docker:
    python run_osworld.py --provider_name docker --model google/gemma-4-E4B-it

    # Single domain test:
    python run_osworld.py --provider_name docker --domain chrome --max_steps 15

    # With 4-bit quantization (lower memory):
    python run_osworld.py --provider_name docker --quantize

    # With specific model:
    python run_osworld.py --provider_name docker --model google/gemma-4-27B-A4B-it
"""

import argparse
import datetime
import json
import logging
import os
import sys

# Add OSWorld to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "OSWorld"))

from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from cua_agent.osworld import Gemma4Agent

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", f"gemma4-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.INFO)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
)
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld evaluation with Gemma4 streaming CUA agent"
    )

    # Environment
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="docker")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument(
        "--observation_type", default="screenshot",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # Agent / model
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_trajectory_length", type=int, default=5)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--quantize", action="store_true",
                        help="Enable 4-bit quantization")

    # Evaluation
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_config_base_dir", type=str,
                        default="OSWorld/evaluation_examples")
    parser.add_argument("--test_all_meta_path", type=str,
                        default="OSWorld/evaluation_examples/test_all.json")
    parser.add_argument("--result_dir", type=str, default="./results")

    # Misc
    parser.add_argument("--stop_token", type=str, default=None)

    args = parser.parse_args()
    if args.no_thinking:
        args.enable_thinking = False
    return args


def get_unfinished(action_space, model, observation_type, result_dir, total_file_json):
    """Find tasks that haven't been evaluated yet (supports resume)."""
    target_dir = os.path.join(result_dir, action_space, observation_type, model)
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        finished[domain].append(example_id)

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]
    return total_file_json


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    # ── Build agent ───────────────────────────────────────────────────────────
    logger.info(f"Model: {args.model}")
    logger.info(f"Action space: {args.action_space}")
    logger.info(f"Provider: {args.provider_name}")

    agent = Gemma4Agent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        action_space=args.action_space,
        observation_type=args.observation_type,
        max_trajectory_length=args.max_trajectory_length,
        enable_thinking=args.enable_thinking,
        quantize_4bit=args.quantize,
    )

    logger.info("Loading Gemma4 model...")
    agent.load()
    logger.info("Model loaded.")

    # ── Build environment ─────────────────────────────────────────────────────
    env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in [
            "a11y_tree", "screenshot_a11y_tree", "som"
        ],
    )

    # ── Load test cases ───────────────────────────────────────────────────────
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space, args.model, args.observation_type,
        args.result_dir, test_all_meta,
    )

    # Count remaining tasks
    total_remaining = sum(len(v) for v in test_file_list.values())
    logger.info(f"Tasks remaining: {total_remaining}")
    for domain, examples in test_file_list.items():
        if examples:
            logger.info(f"  {domain}: {len(examples)}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    scores = []
    for domain in tqdm(test_file_list, desc="Domain"):
        for example_id in tqdm(test_file_list[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            instruction = example["instruction"]
            logger.info(f"[{domain}] {example_id}: {instruction}")

            example_result_dir = os.path.join(
                args.result_dir, args.action_space, args.observation_type,
                args.model, domain, example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                lib_run_single.run_single_example(
                    agent, env, example, args.max_steps,
                    instruction, args, example_result_dir, scores,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"Error": str(e)}) + "\n")

    env.close()

    # ── Report results ────────────────────────────────────────────────────────
    if scores:
        avg = sum(scores) / len(scores) * 100
        logger.info(f"{'═' * 50}")
        logger.info(f"  Average success rate: {avg:.1f}%")
        logger.info(f"  Tasks completed: {len(scores)}")
        logger.info(f"  Tasks passed: {sum(1 for s in scores if s > 0)}")
        logger.info(f"{'═' * 50}")
    else:
        logger.info("No scores collected.")


if __name__ == "__main__":
    main()
