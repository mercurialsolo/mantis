"""Run OSWorld evaluation via HUD using our custom Gemma4 MCPAgent.

Uses our Gemma4MCPAgent (which wraps LlamaCppBrain) as a custom HUD agent,
giving us control over the system prompt, multi-frame screenshot reasoning,
action history tracking, and loop detection.

Prerequisites:
    1. Start llama-server or deploy on Modal
    2. Set HUD API key: export HUD_API_KEY=<your-key>

Usage:
    # Quick smoke test (1 task):
    python run_hud_osworld.py --model-url http://localhost:8080/v1 --max-tasks 1

    # Full OSWorld-Verified benchmark:
    python run_hud_osworld.py --model-url https://<app>--serve.modal.run/v1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OSWorld evaluation on HUD with our custom Gemma4 agent.",
    )
    p.add_argument("--model-url", required=True,
                    help="Base URL of the OpenAI-compatible inference server.")
    p.add_argument("--model", default="gemma-4")
    p.add_argument("--dataset", default="hud-evals/OSWorld-Verified",
                    help="HuggingFace dataset slug.")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--max-tasks", type=int, default=0,
                    help="Limit tasks (0 = all).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if not os.environ.get("HUD_API_KEY"):
        logger.warning("HUD_API_KEY is not set.")

    from hud.datasets import run_dataset
    from cua_agent.hud_mcp_agent import Gemma4MCPAgent

    logger.info(
        "Starting evaluation: model_url=%s  dataset=%s  max_steps=%d",
        args.model_url, args.dataset, args.max_steps,
    )

    split = "train"
    if args.max_tasks > 0:
        split = f"train[:{args.max_tasks}]"

    results = await run_dataset(
        name=f"gemma4-osworld-{args.model}",
        dataset=args.dataset,
        agent_class=Gemma4MCPAgent,
        agent_config={
            "base_url": args.model_url,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        max_steps=args.max_steps,
        max_concurrent=args.max_concurrent,
        split=split,
    )

    # Summarize
    valid = [r for r in results if r is not None and not isinstance(r, Exception)]
    rewards = [r.reward for r in valid if hasattr(r, "reward") and r.reward is not None]

    print()
    print("=" * 60)
    print("  OSWorld Evaluation Results (Gemma4 via HUD)")
    print("=" * 60)
    print(f"  Dataset     : {args.dataset}")
    print(f"  Model URL   : {args.model_url}")
    print(f"  Tasks run   : {len(valid)}")

    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        passed = sum(1 for r in rewards if r > 0)
        print(f"  Avg reward  : {avg_reward:.4f}")
        print(f"  Success rate: {passed}/{len(rewards)} ({passed / len(rewards) * 100:.1f}%)")
    else:
        print("  Avg reward  : N/A (no rewards returned)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
