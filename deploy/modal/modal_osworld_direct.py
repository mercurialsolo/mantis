"""Modal entrypoint: run OSWorld-Verified directly on Modal.

Uses OSWorld's own Docker provider inside a Modal sandbox with Docker support.
Gemma 4 runs on A100 via llama-server, connected via OPENAI_BASE_URL.

The agent loop, image, volume, and helpers live in
``mantis_agent.modal_runtime`` so siblings (``modal_web_tasks.py``,
``benchmarks/osworld_chrome.py``, etc.) can import them without duplicating
1900+ lines of orchestration.

Usage:
    modal run deploy/modal/modal_osworld_direct.py
    modal run deploy/modal/modal_osworld_direct.py --domain os --tasks 10
"""

import json

import modal

from mantis_agent.modal_runtime import image, run_osworld_impl, vol

app = modal.App("gemma4-osworld-direct")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/data": vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
def run_osworld(domain: str = "os", max_tasks: int = 5, max_steps: int = 25):
    return run_osworld_impl(domain=domain, max_tasks=max_tasks, max_steps=max_steps)


@app.local_entrypoint()
def main(domain: str = "os", tasks: int = 0, steps: int = 25):
    """Run OSWorld eval. tasks=0 means all tasks in the domain.

    Uses .remote() — pair with `modal run --detach` for fire-and-forget:
        modal run --detach modal_osworld_direct.py --domain os --tasks 0
    The --detach flag keeps the function running even if the local process exits.
    Results save incrementally to the 'osworld-data' volume.
    """
    print("Mantis — OSWorld Benchmark")
    print(f"  Domain: {domain}")
    print(f"  Tasks:  {'ALL (24)' if tasks == 0 else tasks}")
    print(f"  Steps:  {steps}")
    print()
    print("Running on Modal A100...")
    print("Tip: use `modal run --detach` to keep running after disconnect")
    print()
    result = run_osworld.remote(domain=domain, max_tasks=tasks, max_steps=steps)
    print(f"\nResult: {json.dumps(result, indent=2)}")
