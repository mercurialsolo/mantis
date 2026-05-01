# Modal

Modal is the easiest path for ad-hoc / batch runs that benefit from scale-to-zero. The full set of Modal entry-points lives at `deploy/modal/`. The most common one is `modal_cua_server.py` which exposes the same `/v1/predict` semantics via Modal's `local_entrypoint`.

## Prerequisites

```bash
pip install modal
modal token new   # authenticates this machine to your Modal workspace
```

You'll also need an `.env` file at the repo root with the same five secrets Baseten uses. Modal's `Secret.from_dotenv()` picks them up at deploy time.

## Submit a plan (one-off)

```bash
uv run modal run --detach deploy/modal/modal_cua_server.py \
  --micro plans/example/extract_listings.json \
  --model holo3 \
  --max-cost 2 \
  --max-time-minutes 20 \
  --session-name marketplace-smoke
```

`--detach` returns immediately; the run continues in Modal. Each run gets its own GPU container, scales to zero when done.

## Inspecting a running app

```bash
modal app list
modal app logs <app-id>
```

Results land on the `osworld-data` Modal volume:

```bash
modal volume ls osworld-data results
modal volume get osworld-data results/holo3_results_*.json local_results/
```

## When to choose Modal over Baseten

| Pick Modal if… | Pick Baseten if… |
|---|---|
| You want true scale-to-zero (no idle GPU) | You want a stable HTTPS endpoint other services can call |
| Bursty / batch workloads | Steady traffic |
| You're comfortable invoking via the Modal CLI / SDK | You need a tenant-keyed multi-caller surface |
| You want per-run cost attribution at the run level | You want per-run cost + Prometheus metrics + webhooks |

The `/v1/predict` Tier-1/Tier-2 features (rate limits, idempotency, webhooks, allowlist, metrics) are only exposed through the FastAPI server — i.e., on Baseten / EKS / GKE / local. Modal entry-points are direct, single-tenant.

## See also

- `deploy/modal/` — every Modal entry-point in the repo
- [Plan formats](../getting-started/plan-formats.md)
- [Hosting overview](index.md)
