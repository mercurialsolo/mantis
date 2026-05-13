"""Modal deploy for mantis-helpdesk (#333).

Builds a Modal image from ``deploy/sim_envs/mantis_helpdesk/`` and
exposes the FastAPI app under ``mantis-sim-env-mantis-helpdesk`` so
the harness modal backend resolves it. Mirrors ``modal_mantis_crm.py``
(#332) — per-run app suffixing is deferred until batch volume
justifies it (see #336 §"Modal per-run app vs per-run scope").

## Deploy

    modal secret create mantis-sim-env-mantis-helpdesk-secrets \\
        ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))')

    uv run modal deploy deploy/sim_envs/modal_mantis_helpdesk.py

That gives you ``mantis-sim-env-mantis-helpdesk`` whose ``web`` function
is reachable by the harness ModalBackend:

    uv run mantis plan run examples/sim_envs/mantis_helpdesk/T01_triage_inbox.json \\
        --env mantis-helpdesk --runtime modal --endpoint <BRAIN_URL>
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "mantis-sim-env-mantis-helpdesk"

SRC_DIR = Path(__file__).parent / "mantis_helpdesk"

image = modal.Image.from_dockerfile(
    str(SRC_DIR / "Dockerfile"), context_dir=str(SRC_DIR),
)

secret = modal.Secret.from_name(
    "mantis-sim-env-mantis-helpdesk-secrets",
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
