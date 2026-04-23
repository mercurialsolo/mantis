"""OSWorld Multi-App benchmark — own Modal app for parallel execution.

Multi-app tasks span Chrome + LibreOffice / Thunderbird / VS Code etc.,
testing the agent's ability to switch between apps. Has its own Modal
app namespace so it can run alongside the OS and Chrome benchmarks.

Run with:
    uv run modal run --detach benchmarks/osworld_multiapp.py --tasks 0 --steps 40
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from modal_osworld_direct import (
    image,
    vol,
    run_osworld_impl,
)

multiapp_app = modal.App("gemma4-osworld-multiapp")


@multiapp_app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
def run_multiapp(max_tasks: int = 5, max_steps: int = 40):
    return run_osworld_impl(domain="multi_apps", max_tasks=max_tasks, max_steps=max_steps)


@multiapp_app.local_entrypoint()
def main(tasks: int = 0, steps: int = 40):
    """Run OSWorld multi-app eval. Multi-app tasks need more steps (~40)."""
    print("Mantis — OSWorld Multi-App Benchmark")
    print("  Domain: multi_apps")
    print(f"  Tasks:  {'ALL' if tasks == 0 else tasks}")
    print(f"  Steps:  {steps}")
    print()
    print("Running on Modal A100...")
    print()
    result = run_multiapp.remote(max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
