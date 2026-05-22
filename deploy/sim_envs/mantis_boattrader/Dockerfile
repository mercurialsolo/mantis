# mantis-boattrader — simulated boat-marketplace env (boattrader.com shape).
#
# Image goals:
#   * <500MB final size.
#   * <30s cold boot — generates ~600 boats + 30 dealers + ad inventory.
#   * No outbound network required at runtime (`--network none` friendly).
#
# Build:
#     docker build -t mantis/sim-env-mantis-boattrader:latest \
#         deploy/sim_envs/mantis_boattrader
#
# Run (locally):
#     docker run --rm -p 8033:8080 \
#         -e SEED=42 \
#         -e FAKE_NOW=2026-01-15T09:00:00Z \
#         -e LATENCY_MS_MIN=120 \
#         -e LATENCY_MS_MAX=480 \
#         -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
#         mantis/sim-env-mantis-boattrader:latest

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
    LATENCY_MS_MIN=0 \
    LATENCY_MS_MAX=0 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=2s --start-period=10s --retries=6 \
    CMD curl -fs http://127.0.0.1:8080/__env__/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
