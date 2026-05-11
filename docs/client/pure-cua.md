# Pure CUA mode (`/v1/cua`)

Use this when you want Mantis to be a **thin pass-through for Holo3** —
no plan decomposition, no Claude grounding, no Claude extraction. The
instruction is handed verbatim to the brain, which drives the headed
browser inside the deployment via xdotool.

If you've used `/v1/predict`, the mental model is: same auth, same
tenant caps, same allowlist, same rate limit / concurrency / recording
plumbing — but the inside of the container is a single
`brain ↔ XdotoolGymEnv` loop instead of the orchestrated
`MicroPlanRunner`.

## When to use it

| Pick `/v1/cua` when… | Pick `/v1/predict` when… |
|---|---|
| You want to benchmark Holo3's intrinsic plan-following | You want reliable multi-step extraction with verification |
| You're integrating Mantis as a "Holo3-on-Modal" backend for an existing CUA harness | You're building an end-to-end extraction pipeline |
| You don't have an Anthropic key on the deployment | You want Claude-quality grounding on small click targets |
| You want zero Claude spend per run | You're OK with ~$0.005/click for grounding accuracy |

The tradeoff: without `ClaudeGrounding`, click coordinates come straight
from Holo3's smart-resize model space (converted to screen pixels in
`brain_holo3._model_coords_to_screen`). On small targets accuracy drops
versus the grounded path. That's the whole point of "pure CUA" — you're
measuring what the brain can do unassisted.

## Action surface

What the brain can emit, all executed by xdotool against the headed
Chrome inside Xvfb:

| Verb | Args | Effect |
|---|---|---|
| `click` | `x`, `y` | Single click at screen pixel `(x, y)` |
| `double_click` | `x`, `y` | Double-click |
| `type_text` | `text` | Type into the focused element |
| `key_press` | `key` | Single key or chord (e.g. `"ctrl+a"`, `"enter"`) |
| `scroll` | direction, amount | Scroll the viewport |
| `drag` | `start_x`, `start_y`, `end_x`, `end_y` | Drag from start to end |
| `wait` | `seconds` | Sleep — useful for slow page transitions |
| `done` | `success`, `summary` | Terminate the loop |

Parsing supports five fallback strategies — OpenAI tool_calls, Holo3
native `Action: name({...})` text, JSON action blob, pyautogui-style
calls, and bare keywords (`DONE` / `FAIL`). You don't configure this;
the brain handles it.

## Request

```bash
curl -X POST "$ENDPOINT/v1/cua" \
  -H "X-Mantis-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Find tomorrow’s music events on lu.ma and click the first one",
    "start_url": "https://lu.ma/discover",
    "max_steps": 25,
    "settle_time": 4.0
  }'
```

### Body fields

| Field | Default | Effect |
|---|---|---|
| `instruction` | — *(required)* | Free-text instruction handed verbatim to the brain |
| `start_url` | `""` | Initial URL the browser navigates to before the first inference |
| `max_steps` | `30` | Cap on brain↔env iterations; server-clamped to `MANTIS_MAX_STEPS_PER_PLAN` |
| `frames_per_inference` | `1` | Screenshots fed to the brain per step. Holo3 wants 1 |
| `settle_time` | `2.0` | Seconds to wait after each action before re-screenshotting |
| `detached` | `false` | Queue as background; return `run_id` to poll via `/v1/predict` actions |
| `state_key` | unset | Reuses the Chrome profile + tenant dir tied to this key |
| `max_cost` | `25.0` | USD cap (mostly informational — pure CUA spends nothing on Claude) |
| `max_time_minutes` | `60` | Wall-clock cap; clamped against tenant cap |
| `proxy_city` / `proxy_state` | unset | Geo override (allowlist-gated) |
| `proxy_disabled` | `false` | Skip the residential proxy entirely |
| `record_video` | `false` | Capture screencast; fetch via `GET /v1/runs/{run_id}/video` |
| `video_format` | `"mp4"` | One of `mp4`, `webm`, `gif` |
| `video_fps` | `5` | Capture rate `[1, 30]` |

### Sync response

```jsonc
{
  "run_id": "20260511_143215",
  "mode": "pure_cua",
  "provider": "baseten",
  "session_name": "pure_cua",
  "model": "holo3",
  "instruction": "Find tomorrow's music events on lu.ma…",
  "start_url": "https://lu.ma/discover",
  "success": true,
  "termination_reason": "done",         // or "max_steps" / "loop" / "env_done"
  "steps": 18,
  "duration_s": 142,
  "elapsed_seconds": 142.36,
  "trajectory_len": 18
}
```

### Detached response

When `"detached": true`, the response matches `/v1/predict`'s detached
shape — a `queued` envelope with `status_path` / `result_path` /
`events_path` pointers. Poll via the standard polling actions on
`/v1/predict`:

```bash
curl -X POST "$ENDPOINT/v1/predict" \
  -H "X-Mantis-Token: $TOKEN" \
  -d '{"action": "status", "run_id": "<run_id>"}'
```

The result JSON written to disk has the same shape as the sync
response above.

## CLI

The `mantis cua run` subcommand is a thin HTTP shim — no local browser,
no Anthropic key, no Playwright. It just POSTs to `/v1/cua`:

```bash
export MANTIS_ENDPOINT="https://workspace--app-fn.modal.run"
export MANTIS_TOKEN="<tenant_token>"

mantis cua run "Find tomorrow's music events on lu.ma and click the first one" \
  --start-url https://lu.ma/discover \
  --max-steps 25 \
  --settle-time 4.0
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--endpoint` | Mantis deployment base URL (overrides `MANTIS_ENDPOINT`) |
| `--token` | Per-tenant token (overrides `MANTIS_TOKEN`) |
| `--header KEY=VALUE` | Add HTTP header (e.g. `Authorization=Api-Key …` on Baseten) |
| `--start-url` | Initial URL for the browser |
| `--max-steps` | Cap on the CUA loop |
| `--settle-time` | Seconds to wait after each action |
| `--state-key` | Namespace Chrome profile + checkpoint dir |
| `--detached` | Queue as background; print `run_id` and exit |
| `--proxy-disabled` | Skip the residential proxy |
| `--record-video` | Capture screencast |
| `--json` | Print the raw response JSON instead of a summary |

Exit code is `0` on `success=true`, `1` otherwise. Same as `mantis plan run`.

## How this differs from `/v1/predict` under the hood

```
/v1/predict   →   _task_suite_from_payload   →   MicroPlanRunner
                                                    ├── ClaudeExtractor    (extract / verify)
                                                    ├── ClaudeGrounding    (refine click coords)
                                                    └── Holo3Brain         (tactical actions)
                                                          └── GymRunner ↔ XdotoolGymEnv

/v1/cua       →   _run_pure_cua              →   GymRunner
                                                    └── Holo3Brain         (all actions)
                                                          └── XdotoolGymEnv
```

Concretely: `_run_pure_cua` instantiates `GymRunner(brain=self.brain,
grounding=None)` and calls `runner.run(task=<instruction>)`. There is
no extractor, no decomposer, no per-step verification. Whatever the
brain emits, xdotool executes.

## What you give up

- **No structured extraction.** The response is `success` /
  `steps` / `duration_s` / `termination_reason` — there's no `leads`
  array, no extracted data. The brain ends by emitting `done` with a
  free-text summary; if you need structured data, parse the trajectory
  events or use `/v1/predict` with an `extraction_schema`.
- **No grounding correction.** Holo3 occasionally picks coordinates a
  few pixels off the target. `/v1/predict` recovers via Claude; `/v1/cua`
  doesn't.
- **No section / gate / loop semantics.** It's one open-ended loop until
  `done` or `max_steps`.
- **No skip envelope.** The `skip` / `skip_reason` fields documented in
  [Sending plans](plans.md) are produced by `MicroPlanRunner`. They
  don't appear on `/v1/cua` responses.

If any of those are dealbreakers, you want `/v1/predict`. If they're
fine, `/v1/cua` is the cheapest, simplest way to drive Holo3 on Modal /
Baseten.
