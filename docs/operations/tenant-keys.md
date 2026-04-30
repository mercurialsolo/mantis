# Tenant keys (provisioning + registration)

The operator-side runbook for everything tenant-shaped: issuing tokens, attaching scopes/caps/Anthropic keys, rotating, revoking.

## How it works

The server reads a JSON keys file at `MANTIS_TENANT_KEYS_PATH` (typically a Baseten / k8s secret mounted as a file). Every `POST /v1/predict` looks the presented `X-Mantis-Token` up in this file (constant-time compare, 5 s read cache for hot reload) and resolves a `TenantConfig` for the request.

```
JSON keys file → in-memory hash → constant-time lookup per request
                        ↑
                        │ 5s cache, hot-reloads on edit
                        │
                operator edits secret store → mount → file changes
```

If `MANTIS_TENANT_KEYS_PATH` is not set, the server falls back to single-tenant mode using `MANTIS_API_TOKEN`. The same token works for all callers in that mode.

## Keys file shape

```jsonc
{
  "tenant_keys": {
    "<x-mantis-token-value-1>": {
      "tenant_id": "tenant_a",
      "scopes": ["run", "status", "result", "logs"],
      "max_concurrent_runs": 3,
      "max_cost_per_run": 5.0,
      "max_time_minutes_per_run": 30,
      "rate_limit_per_minute": 60,
      "anthropic_secret_name": "anthropic_api_key_tenant_a",
      "allowed_domains": ["*.boattrader.com", "crm.example.com"],
      "webhook_url": "https://callbacks.example.com/mantis",
      "webhook_secret_name": "webhook_secret_tenant_a"
    },
    "<x-mantis-token-value-2>": {
      "tenant_id": "readonly_dashboard",
      "scopes": ["status", "result"]
    }
  }
}
```

Every field except `tenant_id` is optional and falls back to the `DEFAULT_TENANT` defaults if missing. Use the minimal shape for read-only / observability keys.

## Field reference

| Field | Default | Purpose |
|---|---|---|
| `tenant_id` | (required) | Identifier used as the prefix for `state_key`, browser profile dir, run dir, log lines, and metric labels. Pick something stable and human-readable. |
| `scopes` | `["run", "status", "result", "logs"]` | Which actions this token can perform. Use `["status", "result"]` for read-only consumers. |
| `max_concurrent_runs` | 5 | Per-tenant concurrency gauge. 6th in-flight run returns 429. |
| `max_cost_per_run` | 25.0 | Per-tenant cost cap (clamps each request's `max_cost`). |
| `max_time_minutes_per_run` | 60 | Per-tenant wall-time cap. |
| `rate_limit_per_minute` | 30 | Token-bucket rate limit per tenant. 0 disables. |
| `anthropic_secret_name` | `anthropic_api_key` | Secret file name to read this tenant's Anthropic key from. Lets each tenant have its own Anthropic billing. |
| `allowed_domains` | `[]` (no restriction) | Wildcards matched against `navigate` URLs. Empty = no allowlist (legacy). |
| `webhook_url` | `""` | Per-tenant default webhook for run-completion notifications. Caller can override with per-request `callback_url`. |
| `webhook_secret_name` | `""` | Secret file name for the HMAC-SHA256 signing secret. |

## Issuing a new tenant token

1. **Pick a tenant_id.** Stable, human-readable. Used in metrics labels, log lines, and the data-volume directory layout. `tenant_a`, `customer_acme`, `internal_ops_team`. Avoid spaces, slashes, or anything URL-unsafe.

2. **Generate a token.** Hex-encoded 256 bits is plenty:
   ```bash
   TOKEN=$(openssl rand -hex 32)
   echo "Send this token securely to the tenant: $TOKEN"
   ```

3. **Decide scopes + caps.** Start tight:
   ```jsonc
   {
     "tenant_id": "<id>",
     "scopes": ["run", "status", "result"],
     "max_concurrent_runs": 2,
     "max_cost_per_run": 5.0,
     "max_time_minutes_per_run": 20,
     "rate_limit_per_minute": 30
   }
   ```
   Raise caps later as the tenant proves out.

4. **Add it to the keys file.** Pull the current file, append the entry, push it back:
   ```bash
   # Pull the current keys
   aws secretsmanager get-secret-value \
     --secret-id mantis-prod/mantis_tenant_keys \
     --query SecretString --output text > /tmp/keys.json

   # Edit /tmp/keys.json, add the new entry under tenant_keys
   jq --arg tok "$TOKEN" \
      --argjson cfg '{ "tenant_id": "new_tenant", "scopes": ["run","status","result"], "max_cost_per_run": 5 }' \
      '.tenant_keys[$tok] = $cfg' \
      /tmp/keys.json > /tmp/keys-new.json

   # Push back
   aws secretsmanager put-secret-value \
     --secret-id mantis-prod/mantis_tenant_keys \
     --secret-string file:///tmp/keys-new.json
   ```
   Within 5 s the new token is live (hot reload).

5. **Provision the tenant's Anthropic key** (if the tenant needs its own billing):
   ```bash
   aws secretsmanager create-secret \
     --name mantis-prod/anthropic_api_key_new_tenant \
     --secret-string "sk-ant-..."
   ```
   Reference it from the tenant config: `"anthropic_secret_name": "anthropic_api_key_new_tenant"`.

6. **Hand off the token** out-of-band — 1Password share / Vault entry / signed email. Never paste in Slack.

## Rotation

When you need to rotate:

1. Generate a new token.
2. Add a **second** entry for the same `tenant_id` with the new token.
3. Tenant updates their secret store with the new token.
4. After 24 h (or whatever overlap window you're comfortable with), remove the old token entry.

```jsonc
{
  "tenant_keys": {
    "OLD_TOKEN_HEX": { "tenant_id": "acme", ... },
    "NEW_TOKEN_HEX": { "tenant_id": "acme", ... }   // ← add during rotation
  }
}
```

Both tokens resolve to the same `tenant_id`. Once the tenant has flipped, drop the old entry.

## Revocation

Remove the token entry from the keys file. Within 5 s the server rejects it with 401. No pod restart needed.

```bash
# Compact one-liner
jq 'del(.tenant_keys["BAD_TOKEN_HEX"])' /tmp/keys.json > /tmp/keys-new.json
aws secretsmanager put-secret-value --secret-id mantis-prod/mantis_tenant_keys --secret-string file:///tmp/keys-new.json
```

## Migrating from single-tenant to multi-tenant

Existing single-tenant deployments use `MANTIS_API_TOKEN` env. To migrate:

1. **Create the keys file** with one entry that uses your existing token:
   ```jsonc
   {
     "tenant_keys": {
       "<existing-MANTIS_API_TOKEN-value>": {
         "tenant_id": "default",
         "scopes": ["run", "status", "result", "logs"]
       }
     }
   }
   ```

2. **Mount it as a secret** at `MANTIS_TENANT_KEYS_PATH`. For Baseten:
   ```yaml
   # deploy/baseten/holo3/config.yaml
   secrets:
     mantis_tenant_keys: null
   environment_variables:
     MANTIS_TENANT_KEYS_PATH: /secrets/mantis_tenant_keys
   ```
   Push the deployment.

3. **Add real tenants** to the keys file as you onboard them.

4. **Eventually drop `MANTIS_API_TOKEN`** once nothing relies on the legacy single-tenant path.

The default tenant entry can stay forever — it's harmless and gives existing callers continuity.

## Per-platform secret-store snippets

=== "Baseten"

    ```bash
    BASETEN_API_KEY="..."
    cat /tmp/keys.json | curl -sS -X POST \
      -H "Authorization: Api-Key $BASETEN_API_KEY" \
      -H "Content-Type: application/json" \
      --data-binary "$(jq -n --arg v "$(cat /tmp/keys.json)" '{name:"mantis_tenant_keys",value:$v}')" \
      https://api.baseten.co/v1/secrets
    ```

=== "AWS Secrets Manager"

    ```bash
    aws secretsmanager put-secret-value \
      --secret-id mantis-prod/mantis_tenant_keys \
      --secret-string file:///tmp/keys.json
    ```

=== "GCP Secret Manager"

    ```bash
    gcloud secrets versions add mantis-prod-mantis_tenant_keys \
      --data-file=/tmp/keys.json
    ```

=== "Plain k8s Secret"

    ```bash
    kubectl create secret generic mantis-tenant-keys \
      --from-file=mantis_tenant_keys=/tmp/keys.json \
      --dry-run=client -o yaml | kubectl apply -f -
    ```

## Auditing the live keys

The runtime never logs tokens, only `tenant_id`. Every request emits:

```jsonc
{
  "ts": "2026-04-28T02:14:32Z",
  "level": "INFO",
  "logger": "mantis_agent.baseten_server",
  "msg": "predict tenant=tenant_a scope=run state_key=… detached=true action=run",
  "tenant_id": "tenant_a"
}
```

To see who's active right now:

```bash
kubectl logs -l app=mantis-holo3-server --tail=1000 \
  | jq -r 'select(.tenant_id) | .tenant_id' \
  | sort | uniq -c | sort -rn
```

For longer windows, use the `mantis_predict_requests_total` counter from `/metrics` (see [Metrics](metrics.md)).

## What's NOT in this PR

- **Self-service token issuance** (admin API for tenants to rotate their own keys without operator action) — Tier 3 follow-up.
- **Tenant-scoped Anthropic budget tracking** — costs are reported per-run today; aggregating to a tenant-level monthly budget is left to your billing system using the metrics counters.
- **Token expiration / TTL** — currently tokens are immortal until removed. Add a cron that prunes by hand or wait for the Tier 3 admin API.

## See also

- [Authentication (caller side)](../client/auth.md) — what tenants do with their tokens
- [Rate limits](rate-limits.md) — caps the keys file controls
- [URL allowlist](allowlist.md) — `allowed_domains` enforcement detail
- [Webhooks](webhooks.md) — `webhook_url` + signature verification
