# Independent grounding for high-risk clicks (#181)

The executor brain (Holo3) sometimes clicks visually salient but wrong
targets — most often a listing **photo** instead of the adjacent
**title text**. Same-model grounding inherits the same visual bias and
the [grounding cache](../reference/env-vars.md) introduced for #117 can
pin a stale wrong coordinate across pages.

This page documents the policy that decides when a click step bypasses
the cache and forces a fresh independent grounding call.

## When to enable

Turn on for layouts where the executor's photo-vs-text bias has cost
extractions: listing card grids, contact buttons next to avatars, links
adjacent to large hero images, destructive controls, pagination.

Don't turn it on globally — every high-risk click trades the cache hit
(~$0.005 + ~5–10s latency saved) for a fresh Claude call. Use it where
the failure mode actually shows up.

## Two opt-in surfaces

### 1. Per-step hint (one step)

```jsonc
{
  "intent": "Click the title under the first listing photo",
  "type": "click",
  "section": "extraction",
  "hints": {
    "layout": "listings",
    "independent_grounding": true     // ← forces a fresh grounding call
  }
}
```

Use when one step in an otherwise routine plan is high-risk (e.g. a
single contact reveal).

### 2. SiteConfig flag (whole layouts)

```python
SiteConfig(
    domain="hn.example.com",
    require_independent_grounding=("listings", "extraction", "click"),
)
```

Each entry is matched against the click step's `hints["layout"]`,
`section`, or `type`. Any match → force grounding. Use this when an
entire site / recipe class needs the protection.

Both surfaces are OR'd; either is sufficient. Default behaviour is
unchanged — without either flag, clicks use the cache as before.

## How it works

1. Click handler reads `_should_force_independent_grounding(step, site_config)`.
2. When True, calls `grounding.ground(..., force_compute=True)`.
3. `ClaudeGrounding` / `LLMGrounding` skip the cache lookup and run a
   fresh remote call.
4. Brain coordinates, grounded coordinates, confidence, and the
   accept/reject verdict are logged on the runner.

## Metrics

Two new Prometheus handles ship with this policy. Combine with the
cache hit/miss counters from #117 to dashboard the full grounding flow:

```promql
# Grounding call rate, split by force vs not
sum by (force_compute) (rate(mantis_grounding_call_total[5m]))

# Acceptance rate among forced calls (high-risk clicks)
sum(rate(mantis_grounding_call_total{outcome="accepted",force_compute="true"}[15m]))
  / sum(rate(mantis_grounding_call_total{force_compute="true"}[15m]))

# p95 correction distance, force vs cached
histogram_quantile(0.95,
  sum by (force_compute, le) (
    rate(mantis_grounding_correction_distance_pixels_bucket[10m])
  )
)
```

A growing `correction_distance` p95 is the signal that brain coordinates
are drifting from grounded coordinates — usually a sign of a regressing
checkpoint or a UI redesign on the target site.

## Relation to #117

#117 shipped the `GroundingCache` (process-level shared cache, per-tenant
metrics). #181 layers a *bypass switch* on top: cache stays hot for
routine clicks; high-risk paths force a fresh call. The two policies
work together — the cache amortises cost for the safe case, the bypass
catches the visually-biased case.

## Acceptance criteria status

| Criterion | Status |
|---|---|
| Site or recipe config can require independent grounding for selected click steps | ✅ ``SiteConfig.require_independent_grounding`` + ``hints["independent_grounding"]`` |
| ``MicroPlanRunner`` / ``GymRunner`` records grounding decision metadata on each click | ✅ ``[grounding] refined to (x,y) delta=(dx,dy) force=...`` log line |
| A benchmark case demonstrates a photo-adjacent title click is corrected away from the image region | ⏳ runs against operator's recipe corpus — see `mantis_grounding_correction_distance_pixels` |
| Metrics expose grounding call rate, correction distance, cache hit rate, and post-click success | ✅ this page + #117 cache metrics |
| Documentation explains when to enable this policy and how it relates to #117 | ✅ this page |
