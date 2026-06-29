"""Modal deploy for mantis-linkedin — the server-rendered LinkedIn mirror.

Builds a Modal image from ``deploy/sim_envs/mantis_linkedin/`` and exposes
the FastAPI app under ``mantis-sim-env-mantis-linkedin`` so the harness
modal backend resolves it. This gives the `/v1/cua` canary a network-
reachable LinkedIn-shaped target (the Modal CUA container cannot reach a
localhost sim env) without hammering real linkedin.com from Modal IPs —
which is what paused the real-LinkedIn canary (anti-bot form_target_not_found).

## Deploy

    modal secret create mantis-sim-env-mantis-linkedin-secrets \\
        ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')

    uv run modal deploy deploy/sim_envs/modal_mantis_linkedin.py

The deployed URL exposes the public ``/__env__/health`` probe and the
``/feed/``, ``/in/<handle>/``, ``/jobs/``, ``/login`` surfaces described in
``mantis_linkedin/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "mantis-sim-env-mantis-linkedin"

SRC_DIR = Path(__file__).parent / "mantis_linkedin"

image = (
    modal.Image.from_dockerfile(str(SRC_DIR / "Dockerfile"), context_dir=str(SRC_DIR))
    # Modal injects its own ``modal_requirements.txt`` which pins
    # ``typing-extensions==4.12.2``, downgrading the 4.15.x that fastapi's
    # ``pydantic_core`` pulls in. pydantic_core imports ``Sentinel`` from
    # typing_extensions (added in 4.13), so the downgrade crash-loops the
    # container at boot with ``ImportError: cannot import name 'Sentinel'``.
    # This trailing layer runs AFTER the modal downgrade and restores a
    # compatible version. (The older shop/crm sim-env images predate this
    # resolution and were cached before the conflict surfaced.)
    .pip_install("typing-extensions>=4.13.0")
)

secret = modal.Secret.from_name(
    "mantis-sim-env-mantis-linkedin-secrets",
    required_keys=["ENV_ADMIN_TOKEN"],
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[secret],
    min_containers=0,
    max_containers=8,
    timeout=600,
)
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app  # type: ignore[import-not-found]
    return fastapi_app
