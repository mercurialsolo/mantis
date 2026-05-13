# mantis-crm — simulated CRM env (#332)
#
# Image goals:
#   * <500MB final size (slim base + ~3 pip deps).
#   * <30s cold boot — measured: ~5s on warm Docker cache, ~12s cold.
#   * No outbound network needed at runtime (--network none friendly).
#
# Build:
#     docker build -t mantis/sim-env-mantis-crm:latest deploy/sim_envs/mantis_crm
#
# Run (locally):
#     docker run --rm -p 8001:8080 \\
#         -e SEED=42 \\
#         -e FAKE_NOW=2026-01-15T09:00:00Z \\
#         -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \\
#         mantis/sim-env-mantis-crm:latest
#
# The mantis harness (#336) wraps this with the same env var contract;
# see ``deploy/sim_envs/modal_mantis_crm.py`` for the Modal path.

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080 \
    SEED=42 \
    FAKE_NOW=2026-01-15T09:00:00Z \
    PYTHONUNBUFFERED=1

EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=2s --start-period=10s --retries=6 \
    CMD curl -fs http://127.0.0.1:8080/__env__/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
