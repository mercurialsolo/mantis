"""Modal deploy for mantis-shop (#334).

Builds a Modal image from ``deploy/sim_envs/mantis_shop/`` and exposes
the FastAPI app under ``mantis-sim-env-mantis-shop`` so the harness
modal backend resolves it.

## Deploy

    modal secret create mantis-sim-env-mantis-shop-secrets \\
        ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')

    uv run modal deploy deploy/sim_envs/modal_mantis_shop.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "mantis-sim-env-mantis-shop"

SRC_DIR = Path(__file__).parent / "mantis_shop"

image = (
    modal.Image.from_dockerfile(str(SRC_DIR / "Dockerfile"), context_dir=str(SRC_DIR))
)

secret = modal.Secret.from_name(
    "mantis-sim-env-mantis-shop-secrets",
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
