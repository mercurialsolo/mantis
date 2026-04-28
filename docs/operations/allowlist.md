# URL allowlist

Constrain a tenant to a fixed set of target hosts. The server scans every plan submission for `navigate` URLs / `task_suite.base_url` / `task.start_url` and rejects 403 if any host is off the list.

## Configuration

Set per tenant in the keys file:

```jsonc
{
  "tenant_keys": {
    "<token>": {
      "tenant_id": "vision_claude_prod",
      "allowed_domains": [
        "*.boattrader.com",
        "staffai-test-crm.exe.xyz"
      ]
    }
  }
}
```

Empty `allowed_domains` (or absent field) = no restriction. Useful for trusted internal tenants; tighten for external customers.

## Matching rules

| Pattern | Matches |
|---|---|
| `boattrader.com` | exactly `boattrader.com` |
| `*.boattrader.com` | any subdomain like `www.boattrader.com`, `api.boattrader.com` |
| `*.boattrader.com` | does **not** match `boattrader.com.evil.com` (suffix check is on `.boattrader.com`) |

Hosts are extracted from URLs by regex; case is folded; ports / paths / query are ignored.

## What gets scanned

| Plan shape | What's scanned |
|---|---|
| Micro-plan list | each step's `intent` for `https?://...` URLs |
| Task suite | `base_url` + each task's `start_url` + the task's `intent` text |
| Plain text | not scanned — the decomposer runs server-side and gets the same allowlist applied to its output |

## Error response

```jsonc
{ "detail": "plan references host(s) not in tenant allowlist: evil.com" }
```

Status code: `403 Forbidden`. The plan never starts running. The tenant's `mantis_predict_requests_total{outcome="denied_allowlist"}` counter increments.

## Why this matters

Without an allowlist, a tenant with a `run` scope could submit a plan that navigates to:

- An internal admin panel (lateral movement)
- Another tenant's target site (probing / stealing data)
- A malicious site that drops a payload onto the Mantis runtime

The allowlist is a defense-in-depth measure on top of the proxy + the runner's other safety nets.

## What this doesn't catch

- A user-visible URL on an allowed page that the agent might click. Once the page is loaded, the agent's clicks can navigate anywhere on that domain. If you need stricter per-page constraints, you'll need application-level enforcement (CSP, network policy, etc.).
- Off-host iframes / image URLs / API endpoints called by the page itself.
- Redirects from an allowed host to a non-allowed host. The browser follows the redirect; the runner only checks the explicit `navigate` step's URL.

The allowlist is a **pre-flight check** on what the agent is told to navigate to, not a runtime network policy.

## See also

- [Tenant keys](tenant-keys.md)
- [Client / Errors](../client/errors.md) — caller-side 403 handling
