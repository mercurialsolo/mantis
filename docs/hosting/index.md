# Hosting

Pick the path that matches your infra.

| Platform | Best for | Cost shape | Provisioned by |
|---|---|---|---|
| [**Baseten**](baseten.md) | Fastest path to prod; managed autoscale; nothing to operate | $/hour active GPU + per-call ($0.08–$0.15 per request) | `truss push` |
| [**Modal**](modal.md) | Detached batch runs, scale-to-zero, GPU on demand | $/second of GPU active | `modal run --detach` |
| [**AWS (EKS)**](aws.md) | Existing AWS estate, control over networking + node pools | EC2 g6e.2xlarge ~$1.86/hr + EFS + ECR | Terraform + `kubectl apply` |
| [**GKE**](gke.md) | Existing GCP estate; A100 spot pricing | a2-highgpu-1g ~$3.67/hr + Filestore + Artifact Registry | Terraform + `kubectl apply` |
| [**Local (Docker)**](local.md) | Dev / single-machine; you bring the GPU | your own metal | `docker run` |

## What you provision regardless of platform

Every deployment needs the same five secrets, named the way the container expects (the platform-specific pages walk through how to set them):

| Secret name | Used by | How to get one |
|---|---|---|
| `mantis_api_token` (single-tenant) **or** `mantis_tenant_keys` (multi-tenant) | Container auth | `openssl rand -hex 32` for single; JSON keys file for multi — see [Tenant keys](../operations/tenant-keys.md) |
| `anthropic_api_key` | Claude grounding / extraction / gates | console.anthropic.com → API keys |
| `proxy_url` | IPRoyal proxy host:port | iproyal.com → residential plan |
| `proxy_user` | IPRoyal session ID | same |
| `proxy_pass` | IPRoyal password | same |

For multi-tenant deployments add as many `anthropic_api_key_<tenant>` secrets as you have tenants — the keys file routes each tenant to its own.

## Smoke test (any platform)

Once your deploy is up, this curl validates the whole chain end-to-end:

```bash
ENDPOINT="https://your-mantis-host.example.com"
TOKEN="<your tenant token>"

# Health check (no auth)
curl -fsS "$ENDPOINT/health"

# Auth + plan submission
RESP=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "X-Mantis-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/example/extract_listings.json",
    "state_key": "deploy-smoke",
    "max_cost": 2,
    "max_time_minutes": 20
  }')
echo "$RESP" | jq .
```

Expected: a `queued` response with a `run_id` in under a second. If you get a 401 → token wrong. 503 → server auth not configured. 429 → tenant rate-limited (or above its concurrency cap).

## Common operational tasks

After the deploy works:

- **Provision tenants** → [Tenant keys](../operations/tenant-keys.md)
- **Wire up monitoring** → [Metrics](../operations/metrics.md)
- **Cap blast radius** → [Rate limits](../operations/rate-limits.md), [URL allowlist](../operations/allowlist.md)
- **Get notified on run completion** → [Webhooks](../operations/webhooks.md)
- **Make retries safe** → [Idempotency](../operations/idempotency.md)
