"""Modal-side ``MicroPlanRunner`` host — full mantis stack inside a container.

The local ``mantis plan run`` CLI runs Playwright on the dev laptop and
talks to a remote brain. That works for unit-of-work testing but it
doesn't reproduce the host integration's production architecture, where
the BROWSER also runs inside a Modal container under Xvfb so the
chromium-detection signals (no DISPLAY, missing GPU, ``navigator.
webdriver``, …) that Cloudflare keys on are absent.

This module deploys ``mantis-plan-runner`` — a Modal app exposing a
single function that constructs ``XdotoolGymEnv`` (Xvfb-backed real
Chromium) + ``Holo3Brain`` (HTTP to the brain endpoint) +
``ClaudeGrounding`` + ``ClaudeExtractor``, runs ``MicroPlanRunner``
end-to-end, and returns the same result dict the local CLI writes to
``result.json``. The local ``mantis plan run-modal`` subcommand
submits the plan and renders the same per-step rollup the local CLI
prints.

## Deploy

    uv run modal deploy deploy/modal/modal_plan_runner.py

That gives you an app named ``mantis-plan-runner`` whose ``run_plan``
function is callable via ``modal.Function.lookup`` from any local
process. The image bakes in xvfb, xdotool, scrot, chromium, and the
``mantis_agent`` package; first cold-start is ~60 s after deploy.

## Required Modal Secret

Mount a Modal Secret named ``mantis-plan-runner-secrets`` containing:

    ANTHROPIC_API_KEY=sk-ant-...
    MANTIS_API_TOKEN=...                # optional — used as the
                                         # X-Mantis-Token header on
                                         # the brain endpoint when no
                                         # caller-supplied headers
    OXYLABS_USERNAME=...                # optional — only consulted
    OXYLABS_PASSWORD=...                # when ``use_proxy=True``
    OXYLABS_ENTRYPOINT=pr.oxylabs.io:10000
    OXYLABS_COUNTRY=US
    OXYLABS_STATE=florida               # optional
    OXYLABS_CITY=miami                  # optional

Create / update::

    modal secret create mantis-plan-runner-secrets \\
        ANTHROPIC_API_KEY=sk-ant-... \\
        MANTIS_API_TOKEN=...

## Calling from the CLI

    uv run mantis plan run-modal plans/staff-crm.txt \\
        --endpoint https://workspace--mantis-server-api.modal.run/v1 \\
        --header X-Mantis-Token=... \\
        --output-dir outputs/staff-crm-modal-validation

The CLI is a thin remote driver — same args as ``mantis plan run``,
plus ``--app-name`` (default ``mantis-plan-runner``) to override the
deployed function name.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import modal


APP_NAME = "mantis-plan-runner"

# Lightweight image: just what XdotoolGymEnv + ClaudeExtractor +
# ClaudeGrounding + Holo3Brain need. No CUDA, no llama.cpp, no
# OSWorld — those live in the brain-side image (modal_mantis_server).
runner_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        # Browser + automation tools (XdotoolGymEnv requires these).
        "xvfb",
        "xdotool",
        "scrot",
        "chromium",
        "chromium-driver",
        "x11-utils",
        # CDP read for current-URL — used by _runner_helpers.read_current_url.
        "curl",
        # Fonts for realistic rendering (Cloudflare flags missing fonts).
        "fonts-liberation",
        "fonts-noto-color-emoji",
        "libnss3",
        "libxss1",
        "libasound2",
    )
    .pip_install(
        # Pinned to the orchestrator extras + a few extras XdotoolGymEnv pulls.
        "pillow>=10.0",
        "requests>=2.28",
        "pydantic>=2",
        "anthropic>=0.34",
        "mss>=9.0",
        "python-dotenv",
    )
    .add_local_python_source("mantis_agent")
    .add_local_dir(
        "src/mantis_agent/prompts/files",
        remote_path="/root/mantis_agent/prompts/files",
    )
)

app = modal.App(APP_NAME)


# ── Helper: bring up Xvfb on :99 once per container ──────────────────


_XVFB_DISPLAY = ":99"
_XVFB_VIEWPORT = (1280, 720)


def _ensure_xvfb_running() -> None:
    """Start Xvfb on the canonical display if it isn't already.

    Modal containers don't ship with X. ``XdotoolGymEnv`` can spawn
    its own Xvfb (when ``display=None``) but inside a function we
    benefit from spinning it up once and reusing across calls in
    the same warm container. The display number is fixed at ``:99``
    matching staffai's convention.
    """
    # Fast path: another invocation in this container already booted Xvfb.
    if os.environ.get("DISPLAY") == _XVFB_DISPLAY and _xvfb_alive():
        return

    width, height = _XVFB_VIEWPORT
    subprocess.Popen(
        [
            "Xvfb",
            _XVFB_DISPLAY,
            "-screen", "0", f"{width}x{height}x24",
            "-ac",
            "+extension", "RANDR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = _XVFB_DISPLAY
    # Wait briefly for Xvfb to come up — xdpyinfo confirms.
    deadline = time.time() + 10
    while time.time() < deadline:
        if _xvfb_alive():
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"Xvfb did not come up on {_XVFB_DISPLAY} within 10s — "
        "check that xvfb is installed in the image."
    )


def _xvfb_alive() -> bool:
    try:
        proc = subprocess.run(
            ["xdpyinfo", "-display", _XVFB_DISPLAY],
            capture_output=True,
            timeout=3,
        )
        return proc.returncode == 0
    except Exception:
        return False


# ── Helper: build the Oxylabs proxy dict from Modal-Secret env ──────


def _build_oxylabs_proxy_url(session: str = "mantis") -> str | None:
    """Build the canonical ``http://customer-…:pass@host:port`` URL.

    Reads ``OXYLABS_*`` env vars from the mounted Modal Secret. Returns
    None when the proxy is not requested or credentials are missing —
    callers should treat that as "direct connection" rather than
    erroring.
    """
    entrypoint = os.environ.get("OXYLABS_ENTRYPOINT", "").strip()
    raw_user = os.environ.get("OXYLABS_USERNAME", "").strip()
    password = os.environ.get("OXYLABS_PASSWORD", "").strip()
    if not entrypoint or not raw_user or not password:
        return None

    parts = [f"customer-{raw_user}"]
    for key, label in (
        ("OXYLABS_COUNTRY", "cc"),
        ("OXYLABS_STATE", "st"),
        ("OXYLABS_CITY", "city"),
    ):
        v = os.environ.get(key, "").strip()
        if v:
            parts += [label, v]
    if session:
        parts += ["sessid", session]
    username = "-".join(parts)

    # Same scheme normalisation as the local CLI.
    if "://" not in entrypoint:
        entrypoint = f"http://{entrypoint}"
    # Splice in user:pass.
    scheme, _, rest = entrypoint.partition("://")
    return f"{scheme}://{username}:{password}@{rest}"


# ── The function ─────────────────────────────────────────────────────


@app.function(
    image=runner_image,
    secrets=[modal.Secret.from_name("mantis-plan-runner-secrets")],
    cpu=2.0,
    memory=4096,
    timeout=1800,  # hard 30-min cap per plan
)
def run_plan(
    *,
    plan_text: str | None = None,
    plan_json: dict | None = None,
    brain_endpoint: str,
    brain_extra_headers: dict[str, str] | None = None,
    brain_model: str = "Hcompany/Holo3-35B-A3B",
    start_url: str,
    detail_page_pattern: str = "",
    max_cost_usd: float = 10.0,
    max_time_minutes: int = 30,
    use_proxy: bool = False,
    proxy_session: str = "mantis",
    session_name: str = "mantis_run",
) -> dict:
    """Run :class:`MicroPlanRunner` end-to-end inside this container.

    Either ``plan_text`` (decomposed via Claude) or ``plan_json``
    (pre-decomposed micro-plan dict) must be provided. ``plan_json``
    short-circuits the decompose call — useful when the local CLI
    has already paid that cost or when calling from a cached plan.

    Returns the same result-payload dict the local CLI writes to
    ``result.json``: ``plan_signature``, ``session``, ``step_count``,
    ``successes``, ``failures``, ``elapsed_seconds``, ``final_url``,
    ``costs``, ``steps``.
    """
    # Imports are inside the function so the CLI can ``modal.Function
    # .lookup`` without paying the heavy import cost client-side.
    from mantis_agent.brain_holo3 import Holo3Brain
    from mantis_agent.extraction import ClaudeExtractor
    from mantis_agent.grounding import ClaudeGrounding
    from mantis_agent.gym.micro_runner import MicroPlanRunner
    from mantis_agent.gym.xdotool_env import XdotoolGymEnv
    from mantis_agent.plan_decomposer import MicroPlan, PlanDecomposer
    from mantis_agent.site_config import SiteConfig

    if not plan_text and not plan_json:
        raise ValueError("run_plan: pass either plan_text or plan_json")

    # 1. Resolve / decompose plan.
    if plan_json is not None:
        plan = MicroPlan.from_dict(plan_json)
    else:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing from Modal Secret — "
                "decomposing plan_text requires Claude access"
            )
        plan = PlanDecomposer().decompose_text(plan_text)

    if not plan.steps:
        raise RuntimeError("decomposed plan has no steps")

    # 2. Bring up Xvfb + build env. XdotoolGymEnv spawns Chromium itself
    # on the display we just provisioned.
    _ensure_xvfb_running()
    proxy_url = _build_oxylabs_proxy_url(proxy_session) if use_proxy else None
    env = XdotoolGymEnv(
        start_url=start_url,
        viewport=_XVFB_VIEWPORT,
        browser="chromium",
        display=_XVFB_DISPLAY,
        proxy_server=proxy_url or "",
    )

    # 3. Brain + Claude extractor / grounding.
    brain = Holo3Brain(
        base_url=brain_endpoint,
        model=brain_model,
        extra_headers=brain_extra_headers or None,
    )
    grounding = ClaudeGrounding()
    extractor = ClaudeExtractor()

    # 4. Site config — neutral by default; per-plan tightening via
    # ``--detail-page-pattern`` flows through here.
    site_config = SiteConfig(detail_page_pattern=detail_page_pattern or "")

    # 5. Pre-warm env at start_url before runner.run, mirroring the
    # local CLI's prewarm contract from PR #218.
    env.reset(task="modal_plan_run", start_url=start_url)

    # 6. Construct + run runner.
    runner = MicroPlanRunner(
        brain=brain,
        env=env,
        grounding=grounding,
        extractor=extractor,
        site_config=site_config,
        run_key=session_name,
        session_name=session_name,
        max_cost=max_cost_usd,
        max_time_minutes=max_time_minutes,
    )
    t0 = time.time()
    step_results = runner.run(plan, resume=False)
    elapsed = time.time() - t0

    # 7. Build the same result-payload shape the local CLI writes.
    successes = sum(1 for r in step_results if r.success)
    failures = len(step_results) - successes
    final_url = getattr(runner, "_last_known_url", "")

    return {
        "plan_signature": runner.plan_signature,
        "session": session_name,
        "step_count": len(step_results),
        "successes": successes,
        "failures": failures,
        "elapsed_seconds": round(elapsed, 2),
        "final_url": final_url,
        "costs": dict(getattr(runner, "costs", {})),
        "steps": [
            {
                "index": r.step_index,
                "intent": r.intent,
                "success": r.success,
                "data": getattr(r, "data", ""),
                "duration": getattr(r, "duration", 0.0),
                "steps_used": getattr(r, "steps_used", 0),
            }
            for r in step_results
        ],
    }


# ── Entrypoint for ``modal run`` (ad-hoc invocation, no deploy needed) ─


@app.local_entrypoint()
def cli(
    plan_path: str,
    endpoint: str,
    header: str = "",
    start_url: str = "",
    detail_page_pattern: str = "",
    use_proxy: bool = False,
    proxy_session: str = "mantis",
    output_dir: str = "outputs/modal-run",
):
    """Ad-hoc: ``uv run modal run deploy/modal/modal_plan_runner.py --plan-path …``.

    For repeated runs, prefer ``mantis plan run-modal`` which uses the
    deployed function directly (no per-call image build).
    """
    plan_text: str | None = None
    plan_json: dict | None = None
    if plan_path.endswith(".json"):
        with open(plan_path) as fh:
            plan_json = json.load(fh)
    else:
        with open(plan_path) as fh:
            plan_text = fh.read()

    headers: dict[str, str] = {}
    if header:
        for entry in header.split(","):
            k, _, v = entry.partition("=")
            if k and v:
                headers[k.strip()] = v.strip()

    result = run_plan.remote(
        plan_text=plan_text,
        plan_json=plan_json,
        brain_endpoint=endpoint,
        brain_extra_headers=headers or None,
        start_url=start_url,
        detail_page_pattern=detail_page_pattern,
        use_proxy=use_proxy,
        proxy_session=proxy_session,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
