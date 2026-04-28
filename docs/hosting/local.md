# Local (Docker)

For development, debugging, or single-machine deployments. You bring the GPU; the container handles the rest.

## Prerequisites

- A machine with an NVIDIA GPU and ≥48 GB VRAM (L40S, A100 40 GB+, H100, etc.)
- Docker 24+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~50 GB free disk for the image + Holo3 weights

## 1. Build the image

```bash
docker build --platform linux/amd64 \
  -f docker/server.Dockerfile \
  -t mantis-holo3-server:local .
```

First build is ~30 min (llama.cpp + CUDA compile). Subsequent builds reuse the layer cache and complete in seconds.

## 2. Pre-warm Holo3 weights

The image ships without the GGUF weights (~34 GB). Download once into a host directory you'll mount:

```bash
mkdir -p /srv/mantis-models/holo3
docker run --rm -v /srv/mantis-models/holo3:/models python:3.11-slim bash -c "
  pip install --no-cache-dir 'huggingface-hub[cli]' &&
  huggingface-cli download mradermacher/Holo3-35B-A3B-GGUF \
    Holo3-35B-A3B.Q8_0.gguf \
    Holo3-35B-A3B.mmproj-f16.gguf \
    --local-dir /models
"
```

## 3. Run it

```bash
docker run --rm -d --name mantis-holo3 \
  --gpus all \
  -p 8000:8000 \
  -e MANTIS_API_TOKEN="$(openssl rand -hex 32)" \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e PROXY_URL="geo.iproyal.com:12321" \
  -e PROXY_USER="..." \
  -e PROXY_PASS="..." \
  -v /srv/mantis-data:/workspace/mantis-data \
  -v /srv/mantis-models/holo3:/models/holo3 \
  mantis-holo3-server:local

# Save the token you just generated — it's the X-Mantis-Token for callers.
docker logs mantis-holo3 | grep -i token
```

The first start takes ~3 min for the in-pod llama.cpp server to load the GGUF onto the GPU. Subsequent restarts are quick (weights stay on the mounted volume).

## 4. Smoke-test

```bash
TOK="<the token you set above>"

curl -fsS http://localhost:8000/health
# {"ok": true, "model": "holo3"}

curl -fsS -X POST http://localhost:8000/v1/predict \
  -H "X-Mantis-Token: $TOK" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/boattrader/extract_url_filtered_3listings.json",
    "state_key": "local-smoke",
    "max_cost": 2,
    "max_time_minutes": 20
  }'
```

## Variations

### Multi-tenant locally

Mount a tenant keys file:

```bash
cat > /srv/mantis-secrets/tenant_keys.json <<EOF
{
  "tenant_keys": {
    "$(openssl rand -hex 32)": {
      "tenant_id": "alice",
      "scopes": ["run", "status", "result"],
      "max_concurrent_runs": 2,
      "rate_limit_per_minute": 30
    },
    "$(openssl rand -hex 32)": {
      "tenant_id": "bob",
      "scopes": ["run", "status", "result"],
      "allowed_domains": ["*.boattrader.com"]
    }
  }
}
EOF

docker run ... \
  -e MANTIS_TENANT_KEYS_PATH=/secrets/tenant_keys.json \
  -v /srv/mantis-secrets:/secrets:ro \
  ...
```

### Without a GPU (orchestrator-only)

If you just want to run the orchestrator and have a remote Holo3 endpoint somewhere else:

```bash
pip install -e '.[orchestrator]'
# Use mantis_agent.gym.micro_runner.MicroPlanRunner directly with a
# remote BrainHolo3 client. See the integration docs.
```

This is exactly the path vision_claude takes — see [Integrations / vision_claude](../integration-vision_claude.md).

### Cost guardrails

Set these env vars on the container if you want lower caps than the defaults:

```bash
-e MANTIS_MAX_STEPS_PER_PLAN=100 \
-e MANTIS_MAX_LOOP_ITERATIONS=20 \
-e MANTIS_MAX_RUNTIME_MINUTES=30 \
-e MANTIS_MAX_COST_USD=10 \
```

## Cleanup

```bash
docker stop mantis-holo3
docker rm mantis-holo3
# Mounts persist (/srv/mantis-data + /srv/mantis-models). Delete them if you're done.
```

## See also

- [`docker/server.Dockerfile`](https://github.com/mercurialsolo/mantis/blob/main/docker/server.Dockerfile) — the image we just built
- [Hosting overview](index.md)
- [Tenant keys](../operations/tenant-keys.md) — multi-tenant setup
