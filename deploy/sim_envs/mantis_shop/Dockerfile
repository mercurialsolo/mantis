# mantis-shop — simulated Shopify-shape storefront + admin env (#334)
#
# Image goals:
#   * <500MB final size (slim base + ~3 pip deps).
#   * <30s cold boot — measured: ~6s on warm Docker cache, ~14s cold
#     (2,000 products with variant matrix + 5,000 historical orders).
#   * No outbound network needed at runtime (--network none friendly).
#
# Build:
#     docker build -t mantis/sim-env-mantis-shop:latest deploy/sim_envs/mantis_shop
#
# Run (locally):
#     docker run --rm -p 8031:8080 \
#         -e SEED=42 \
#         -e FAKE_NOW=2026-01-15T09:00:00Z \
#         -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
#         mantis/sim-env-mantis-shop:latest
#
# The mantis harness (#336) wraps this with the same env var contract;
# see ``deploy/sim_envs/modal_mantis_shop.py`` for the Modal path.

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
