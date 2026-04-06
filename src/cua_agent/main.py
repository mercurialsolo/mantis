"""CLI entry point for the streaming CUA agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .agent import StreamingCUA
from .brain import DEFAULT_MODEL, Gemma4Brain
from .executor import ActionExecutor
from .streamer import ScreenStreamer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cua",
        description="Streaming CUA agent — Gemma4 watches your screen and acts.",
    )
    p.add_argument("task", help="Task to accomplish (natural language)")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemma4 model to use (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Screen capture FPS (default: 3.0)",
    )
    p.add_argument(
        "--buffer-size",
        type=int,
        default=15,
        help="Frame buffer size (default: 15)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Frames per inference cycle (default: 5)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps before stopping (default: 50)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Screen capture scale factor (default: 1.0, use 0.5 for half-res)",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Seconds to wait after each action (default: 0.5)",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking mode (faster but less accurate)",
    )
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Enable 4-bit quantization (reduces VRAM usage)",
    )
    p.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor to capture (0=all, 1=primary, default: 1)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy library logs
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger("cua")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")

    # ── Build components ──────────────────────────────────────────────────────
    brain = Gemma4Brain(
        model_name=args.model,
        enable_thinking=not args.no_thinking,
        quantize_4bit=args.quantize,
    )

    streamer = ScreenStreamer(
        fps=args.fps,
        buffer_size=args.buffer_size,
        monitor=args.monitor,
        scale=args.scale,
    )

    executor = ActionExecutor(safe_mode=True)

    agent = StreamingCUA(
        brain=brain,
        streamer=streamer,
        executor=executor,
        max_steps=args.max_steps,
        frames_per_inference=args.frames,
        settle_time=args.settle,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model (this may take a moment)...")
    brain.load()

    # ── Run ───────────────────────────────────────────────────────────────────
    result = asyncio.run(agent.run(args.task))

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    status = "SUCCESS" if result.success else "FAILED"
    print(f"  {status}: {result.summary}")
    print(f"  Steps: {result.total_steps}")
    print(f"  Time:  {result.total_time:.1f}s")
    print("═" * 60)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
