# Webhooks

Tenants can receive HMAC-signed POSTs when their runs reach a terminal state, instead of polling.

## How it triggers

Two ways to configure a callback URL:

1. **Per-tenant default** — `webhook_url` in the tenant keys file. Every run that tenant submits notifies this URL.
2. **Per-request override** — `callback_url` field in `POST /v1/predict`. Wins over the tenant default.

When the run reaches `succeeded` / `failed` / `cancelled`, the server fires the callback in a detached thread.

## Body

```jsonc
{
  "run_id": "20260428_021432_076255ef",
  "tenant_id": "vision_claude_prod",
  "status": "succeeded",
  "summary": { ... same shape as the /v1/predict status response ... },
  "delivered_at": "2026-04-28T02:24:01Z"
}
```

## Signing

Header: `X-Mantis-Signature: sha256=<hex>`

The secret is read from `$/secrets/<webhook_secret_name>` (named by the tenant config) or falls back to the `MANTIS_WEBHOOK_SECRET_DEFAULT` env var. If neither is set, the signature header is omitted entirely — your verifier can treat that as auth failure.

Verifier:

```python
import hmac, hashlib

def verify(body: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

## Retries

3 attempts with exponential backoff: 1 s, 5 s, 30 s. After that, the server gives up — keep polling as a fallback.

Retry triggers:

| Receiver behavior | Retry? |
|---|---|
| 2xx | No (success) |
| Connection error / timeout | Yes |
| 4xx (any) | Yes |
| 5xx | Yes |

This is intentionally aggressive — webhooks are best-effort and idempotency is the receiver's responsibility. If your endpoint has expensive side effects, dedupe by `run_id`.

## Per-tenant secret provisioning

Generate a secret once per tenant:

```bash
WEBHOOK_SECRET=$(openssl rand -hex 32)

# Push to wherever your platform stores them
aws secretsmanager create-secret \
  --name mantis-prod/webhook_secret_acme \
  --secret-string "$WEBHOOK_SECRET"

# Reference from the tenant config:
# "webhook_secret_name": "webhook_secret_acme"
```

Share the secret with the tenant out-of-band (1Password / Vault). Their HMAC verifier needs it.

## What's NOT in this PR

- Delivery state persistence (today, the server logs the delivery outcome but doesn't keep an audit trail).
- Per-event filtering (e.g., "only fire on `failed`"). Today every terminal status fires.
- Replay / DLQ for failed deliveries beyond the 3 retries.

## See also

- [Client / Runs and polling](../client/runs-and-polling.md#webhooks-instead-of-polling) — caller-side usage
- [Tenant keys](tenant-keys.md) — where to set `webhook_url` / `webhook_secret_name`
