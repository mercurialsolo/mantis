# Glossary

Quick definitions for project-specific terms. For deeper explanations, follow the links into the relevant section.

| Term | Definition |
|---|---|
| **Action** | A unit of agent behavior the env can execute. One of `CLICK`, `DOUBLE_CLICK`, `TYPE`, `KEY_PRESS`, `SCROLL`, `DRAG`, `WAIT`, `DONE`. |
| **Brain** | An inference client. Today: `BrainHolo3` (llama.cpp / OpenAI-compat), `BrainClaude` (Anthropic API). |
| **CUA** | Computer Use Agent. The category Mantis sits in. |
| **Detached run** | A run started with `detached: true`. The server returns a `run_id` immediately and continues work in the background. Caller polls / waits for a webhook. |
| **Gate** | A micro-plan step with `gate: true`. Claude verifies a free-text condition; if it fails, the run halts. |
| **GymEnvironment** | The abstract env interface: `reset()`, `step(action)`, `screenshot()`, `close()`. `XdotoolGymEnv` is the concrete production impl. |
| **Holo3** | Holo3-35B-A3B, the small specialist multimodal model used for click / scroll / type / drag. Runs on a single GPU as GGUF. |
| **Idempotency-Key** | Header-based dedup key. Same key → same `run_id` returned. 24 h cache. |
| **MicroPlan** | A list of micro-step objects: `[{intent, type, section, ...}, ...]`. Reliable for high-volume workflows. |
| **MicroPlanRunner** | The orchestrator. Walks the micro-plan, runs each step, handles retries / gates / loops / checkpoint resume. |
| **Plan text** | Plain-English instruction submitted via `plan_text`; the server decomposes via Claude (cached). |
| **Polished video** | The composed walkthrough (title card → captioned run with action overlays → outro card) produced when `record_video: true`. |
| **Raw recording** | The unprocessed Xvfb screencast. Available via `?raw=1` on the video endpoint. |
| **`run_id`** | Server-generated identifier for a detached run. Format: `<YYYYMMDD>_<HHMMSS>_<random_hex>`. |
| **`state_key`** | Caller-chosen identifier for a workflow. Server prefixes it with `tenant_id` to isolate. Controls browser-profile reuse and checkpoint resume. |
| **Step** | One element of a micro-plan. Has `intent`, `type`, optional `section`, `gate`, `verify`, `loop_target`, etc. |
| **Task suite** | Multi-task plan format (`{tasks: [...]}`). Each task has its own `verify` predicate. |
| **Tenant** | A unit of isolation. Each tenant has its own token, scopes, caps, browser profile dir, run dir, and optionally its own Anthropic key. |
| **TenantConfig** | The resolved configuration for one request: `(tenant_id, scopes, max_*, anthropic_secret_name, allowed_domains, webhook_*)`. |
| **Tier 1 / Tier 2** | The hardening tiers. Tier 1: multi-tenant safe (auth, caps, isolation). Tier 2: production-quality (rate limits, idempotency, webhooks, allowlist, metrics). |
| **xdotool** | The X11 input-injection tool used to send clicks / keys to Chrome inside Xvfb. Fingerprint-free (no DevTools / CDP signal). |
| **Xvfb** | X virtual framebuffer. The headless display the agent's browser paints to and ffmpeg captures. |
| **`X-Mantis-Token`** | The container-level auth header. Custom header (not `Authorization: Bearer`) so it doesn't collide with platform-gateway auth. |
