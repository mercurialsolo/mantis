"""Modal deploy for mantis-crm (#332).

Builds a Modal image from ``deploy/sim_envs/mantis_crm/`` and exposes
the FastAPI app under ``mantis-sim-env-mantis-crm`` so the harness
modal backend resolves it. Per-run app suffixing is deferred until the
batch volume justifies it (see #336 §"Modal per-run app vs per-run
scope").

## Deploy

    modal secret create mantis-sim-env-mantis-crm-secrets \\
        ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')

    uv run modal deploy deploy/sim_envs/modal_mantis_crm.py

That gives you ``mantis-sim-env-mantis-crm`` whose ``web`` function is
reachable by the harness ModalBackend:

    uv run mantis plan run examples/sim_envs/mantis_crm/T01_tag_reengage.json \\
        --env mantis-crm --runtime modal --endpoint <BRAIN_URL>
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "mantis-sim-env-mantis-crm"

# Build the image from the same Dockerfile as the local Docker path so
# the deploy is bit-identical to ``docker run``.
SRC_DIR = Path(__file__).parent / "mantis_crm"

image = (
    modal.Image.from_dockerfile(str(SRC_DIR / "Dockerfile"), context_dir=str(SRC_DIR))
    # The ASGI mount Modal exposes serves on whatever PORT is set in the
    # image's CMD line; we just need the FastAPI app symbol.
)

secret = modal.Secret.from_name(
    "mantis-sim-env-mantis-crm-secrets",
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
    # Import inside the function so the local CLI doesn't need
    # FastAPI on its path to introspect this deploy file.
    from app.main import app as fastapi_app  # type: ignore[import-not-found]

    return fastapi_app
