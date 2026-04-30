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
| **`register_tool`** | Host-provided tool registration on `MicroPlanRunner`. Lets a host (e.g. vision_claude) expose its own tools — `auth_credentials`, `request_user_input`, `bash`, etc. — to the brain mid-plan. JSON-schema input definition. See [issue #71](https://github.com/mercurialsolo/mantis/issues/71). |
| **`PauseRequested`** | Exception a registered tool raises when it needs out-of-band input (OTP, 2FA, confirmation). The runner catches it, snapshots state, and returns a serializable `PauseState`. Resume by calling `runner.resume(state, user_input=..., plan=plan)`. See [issue #73](https://github.com/mercurialsolo/mantis/issues/73). |
| **`PauseState`** | JSON-safe snapshot of a paused run: step index, pending tool + arguments, replayed step results, loop counters. Designed to round-trip through Postgres JSONB. |
| **`RunnerResult`** | Rich return type from `MicroPlanRunner.run_with_status(plan)` and `runner.resume(...)`. Carries `steps`, `status` (`completed` / `halted` / `cancelled` / `paused`), `cancelled` bool, and optional `pause_state`. |
| **`cancel_event`** | External cancellation hook on `MicroPlanRunner`. Pass any object with `.is_set()` (e.g. `threading.Event`) or a no-arg callable. Checked at every step boundary. ECS deploy SIGTERM is the canonical use case. See [issue #76](https://github.com/mercurialsolo/mantis/issues/76). |
| **`step_callback`** | Per-step observability hook: `Callable[[idx, intent, action_or_none, ok], None]`. Errors are logged, never raised — observability bugs cannot break a run. See [issue #74](https://github.com/mercurialsolo/mantis/issues/74). |
| **`LAUNCH_APP`** | An `ActionType` that lets a plan start a desktop binary on demand (`chromium`, custom wrapper script, ...). Closes the symmetry gap with the Claude backend's `bash` tool. See [issue #72](https://github.com/mercurialsolo/mantis/issues/72). |
| **`scale_brain_to_display`** | Helper that maps brain-space pixel coordinates to display-space pixels when a host resizes screenshots before inference. The contract that prevents the 1.5× click-offset bug class. See [`reference/coordinate-spaces.md`](coordinate-spaces.md) and [issue #75](https://github.com/mercurialsolo/mantis/issues/75). |
| **`fill_field`** | Step type for a labelled text input. `params={"label": "User ID", "value": "sarah.connor"}`. The runner clicks the field, clears any pre-filled value, and types `value`. Uses `find_form_target` (single labelled element), not `find_all_listings`. See [issue #80](https://github.com/mercurialsolo/mantis/issues/80). |
| **`submit`** | Step type for a single labelled button — Login / Save / Update / Submit / Continue. `params={"label": "Login", "aliases"?: ["Sign In", "Continue"]}`. One click, then waits 2.5 s. The runner scrolls Page_Down up to 4 times when the button isn't in viewport (issue #89 §2); `aliases` covers copy variation across products (e.g. "Update Lead" / "Save" / "Save Changes"). |
| **`select_option`** | Step type for a dropdown / `<select>` choice. `params={"dropdown_label": "Industry Vertical", "option_label": "Space Exploration"}`. Two-phase: click dropdown → screenshot the open menu → click option. |
| **`find_form_target`** | `ClaudeExtractor` method that locates one labelled element on a non-listings page. Counterpart to `find_all_listings` for forms. Returns `{x, y, action, value, label}`. |
