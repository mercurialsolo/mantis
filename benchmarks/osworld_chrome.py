"""OSWorld Chrome benchmark — own Modal app for parallel execution.

Identical infrastructure to ``modal_osworld_direct.py`` but runs in a
distinct Modal app namespace so chrome runs do not collide with the
ongoing OS run. Both benchmarks share the underlying ``run_osworld_impl``
agent loop, the same image, the same llama-server setup, and the same
volume — only the app name and the default domain differ.

Run with:
    uv run modal run --detach benchmarks/osworld_chrome.py --tasks 0 --steps 30
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the parent directory importable so this file works when launched
# locally with `modal run`. Inside the Modal container, modal_osworld_direct
# is shipped via image.add_local_python_source (see modal_osworld_direct.py)
# so this sys.path mutation is a no-op there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from modal_osworld_direct import (
    image,
    vol,
    run_osworld_impl,
)

# Distinct app namespace from the OS benchmark.
chrome_app = modal.App("gemma4-osworld-chrome")


@chrome_app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
def run_chrome(max_tasks: int = 5, max_steps: int = 30):
    """Modal-decorated entry — delegates to the shared agent loop."""
    return run_osworld_impl(domain="chrome", max_tasks=max_tasks, max_steps=max_steps)


@chrome_app.local_entrypoint()
def main(tasks: int = 0, steps: int = 30):
    """Run OSWorld Chrome eval. tasks=0 means all 46 chrome tasks."""
    print("Mantis — OSWorld Chrome Benchmark")
    print("  Domain: chrome")
    print(f"  Tasks:  {'ALL (46)' if tasks == 0 else tasks}")
    print(f"  Steps:  {steps}")
    print()
    print("Running on Modal A100...")
    print("Tip: use `modal run --detach` to keep running after disconnect")
    print()
    result = run_chrome.remote(max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
