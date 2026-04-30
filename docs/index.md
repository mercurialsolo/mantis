# Mantis CUA

A unified perception-reasoning-action agent for computer use. Given a structured plan, Mantis drives a real browser (or any Xvfb-rendered application), takes actions, extracts structured data, and produces both a JSON result and an optional polished video walkthrough.

```
       ┌──────────────────────┐         ┌─────────────────────────┐
3p ──► │ Mantis CUA service   │ ──────► │ Target app (Chrome,     │
caller │ Holo3 + Claude       │         │ file manager, terminal, │
       │ /v1/predict          │         │ LibreOffice, …)         │
       └──────────┬───────────┘         └─────────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ Result + lead CSV +  │
       │ polished screencast  │
       └──────────────────────┘
```

## What you get

- **Reliable multi-step plans.** A structured `MicroPlanRunner` enforces section / gate / loop semantics so even small models behave on long workflows.
- **Cheap inference** at click-and-scroll latency. Holo3 (35B GGUF on a single GPU) for tactical actions; Claude API only for surgical reasoning steps (extract / verify / ground a click).
- **Real browser, real desktop.** Xvfb + Chrome + xdotool. No Playwright fingerprints. Works against sites with bot detection.
- **Cloud-portable.** Same image runs on Baseten, Modal, EKS, GKE, or your own Docker host.
- **Multi-tenant out of the box.** Per-key auth, per-tenant rate limits, idempotency, webhooks, URL allowlists, Prometheus metrics.
- **Screencast included.** Every run can produce a title-card → captioned-run-with-action-overlays → outro video that's ready to share.

## Pick a path

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **I just want to try it**

    ---

    Hit the live Baseten endpoint with a curl. No deploy needed.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-server:{ .lg .middle } **I want to host it**

    ---

    Deploy on Baseten / Modal / EKS / GKE / your own Docker host.

    [:octicons-arrow-right-24: Hosting](hosting/index.md)

-   :material-key:{ .lg .middle } **I want to integrate from my app**

    ---

    Auth, sending plans, polling for results, downloading recordings.

    [:octicons-arrow-right-24: Client](client/index.md)

-   :material-account-multiple:{ .lg .middle } **I run a multi-tenant fleet**

    ---

    Provision tenant keys, enforce rate limits, wire webhooks + metrics.

    [:octicons-arrow-right-24: Operations](operations/index.md)

</div>

## Verified end-to-end

| Path | Run | Result |
|---|---|---|
| Modal | 3-listing extraction | 2 / 3 leads with phone, ~$0.42, 13 min |
| Baseten | 3-listing extraction | 3 / 3 leads with phone, ~$0.42, 9.5 min |

Both deployments produce structured JSON rows (year / make / model / price / phone / url) for every successfully extracted listing.

## At a glance

| | Notes |
|---|---|
| Languages | Python 3.11+ |
| GPU footprint | 1× H100 / A100 80 GB / L40S 48 GB (for Holo3 inference). Orchestrator can run on CPU. |
| Cost per task (3-listing reference) | GPU ~$0.12 + Claude ~$0.12 + proxy ~$0.18 = **~$0.42** |
| Auth | `X-Mantis-Token` (custom header) + Baseten gateway `Authorization: Api-Key` |
| API style | OpenAI-compatible `/v1/chat/completions` for inference; Mantis-shape `/v1/predict` for orchestrated runs |
| Cloud paths | Baseten Truss · Modal · EKS (Terraform + k8s) · GKE (Terraform + k8s) · raw Docker |
| Multi-tenancy | File-backed JSON keys, per-tenant scopes / caps / Anthropic key / allowed domains / webhooks |
| Recording | Optional `record_video: true` produces a polished walkthrough with overlays for clicks / keys / scrolls / typing / drags |

## License

MIT. Source on [GitHub](https://github.com/mercurialsolo/mantis).
