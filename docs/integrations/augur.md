# Augur — per-run debug bundles + live streaming

[`augur-sdk`](https://github.com/mercurialsolo/augur-sdk) instruments any
screenshot-grounded CUA with a single `DebugSession` context manager.
Mantis ships the wedge in `src/mantis_agent/observability/augur.py` so
every gym run produces a path-stable, schema-validated bundle on disk —
and, when `AUGUR_DSN` is exported, streams the same records live to the
Augur workspace (Sentry-style DSN, connection-status badge).

> The upstream integration guide is the canonical reference for the
> bundle layout, schema, and replay contract:
> <https://mercurialsolo.github.io/augur-sdk/integrations/mantis/>.
> This page documents how Mantis itself wires in.

## Install

The adapter only activates when the `augur-sdk` package is importable.
Pick whichever install matches your deployment:

```bash
# Direct
pip install augur-sdk

# Or via the Mantis observability extra (adds nothing else)
pip install 'mantis-agent[observability]'

# Or pull everything Mantis ships
pip install 'mantis-agent[full]'
```

When the package isn't installed, the wedge is a hard no-op — no error,
no overhead, no bundle.

## What gets written

Every gym run that reaches `RunExecutor.execute()` opens a
`DebugSession`. The default on-disk location is:

```
${MANTIS_DATA_DIR:-./data}/augur/<run_id>/
```

Override with `MANTIS_AUGUR_DIR=/custom/root` (the run id is still
appended). Inside each bundle the SDK writes:

```
augur/<run_id>/
├── manifest.json          # bundle metadata (schema_version, paths, ...)
├── trace.json             # session + step traces
├── screenshots/0001_post.png ...
├── events/0001.jsonl      # one DecisionEvent per line, grouped by step
├── steps/0001.json
└── schema/                # vendored JSON Schemas for offline validation
```

Validate any bundle with:

```python
from augur_sdk.validation import validate_bundle
issues = validate_bundle("data/augur/<run_id>/")
assert not issues
```

## What gets streamed

If `AUGUR_DSN` is set in the runtime environment, the SDK opens a
streaming sink at session start and forwards every `record_step`,
`record_event`, and `attach_observation` call to the workspace
identified by the DSN. The on-disk bundle stays the source of truth —
streaming is observe-only.

Network failures are non-fatal. A misconfigured DSN, a firewall block,
or a transient blip lands as a debug log line; the run continues.

## Mantis → Augur vocabulary mapping

The adapter normalizes Mantis-internal layer names to the Augur
[`DecisionLayer`](https://mercurialsolo.github.io/augur-sdk/spec/#decision-event)
enum:

| Mantis layer            | Augur layer       |
|---|---|
| `critic-frontier`       | `verifier`        |
| `critic`                | `verifier`        |
| `gate-decision`         | `verifier`        |
| `preview-gate`          | `verifier`        |
| `agentic-recovery`      | `step_recovery`   |
| `recovery`              | `step_recovery`   |
| `som-click`             | `grounding`       |
| `planner`               | `planner`         |
| anything else           | `runner`          |

Mantis healing-event `kind`s collapse onto Augur's five-state
`DecisionKind`:

| Mantis kind     | Augur kind        |
|---|---|
| `fire` / `skip` / `decision` | `decision`  |
| `result` / `observation`     | `observation` |
| `info`          | `info`            |
| `error`         | `error`           |
| `metric`        | `metric`          |

Mantis runner status maps onto `RunStatus`:

| Mantis status                                                           | Augur status |
|---|---|
| `completed`, `completed_with_failures`, `succeeded`                     | `succeeded`  |
| `halted`, `budget_exceeded`, `time_exceeded`                            | `halted`     |
| `failed`                                                                | `failed`     |
| `cancelled`                                                             | `cancelled`  |
| `running`, `paused`                                                     | `running`    |

## Environment knobs

| Variable | Effect |
|---|---|
| `AUGUR_DSN` | Sentry-style DSN. When set, the SDK opens a streaming sink to the workspace; when unset, only the on-disk bundle is produced. |
| `AUGUR_CAPTURE_MODE` | One of `off`, `metadata`, `trace`, `screenshots` (default), `video`, `model_io`, `dispatch`, `replay`, `full`. Controls what the SDK captures. |
| `MANTIS_AUGUR_DIR` | Override the root directory where bundles are written (run id is still appended). |
| `MANTIS_AUGUR_DISABLED` | `1` / `true` / `yes` → adapter is a no-op even with the SDK installed. Useful for tests / CI. |
| `MANTIS_DATA_DIR` | Default bundle root when `MANTIS_AUGUR_DIR` is unset. Defaults to `./data`. |
| `MANTIS_VERSION` | Surfaced as `client.version` on the manifest — useful when bisecting which build produced a bundle. |
| `MANTIS_GIT_SHA` | Surfaced as `client.git_sha` on the manifest. |
| `MANTIS_AUGUR_VERBOSE` | `1` → adapter diagnostics log at `WARN` instead of `DEBUG`. Modal suppresses `INFO`/`DEBUG`, so flip this on when verifying a deploy. |

## Structured session costs (augur-sdk 0.1.8+)

The Augur workspace's Runs-list **COST** column reads from the
canonical `session.costs` record (added in SDK 0.1.6, populated via
the `set_costs` helper in 0.1.8). Mantis stamps it from the run's
`CostMeter` totals at finalization:

```python
augur.set_costs(
    total_usd=..., model_usd=..., gpu_usd=..., proxy_usd=...,
    tokens_in=..., tokens_out=..., cache_hit_tokens=...,
)
```

The legacy `add_tag("cost_usd", ...)` / `add_tag("cost_claude_usd",
...)` chain was retired alongside the bump — keep one source of
truth. `elapsed_seconds` stays a tag because the costs schema has no
slot for run wallclock.

Per-step cost deltas land on the `StepTrace.costs` field (filled
inline by `_build_step_trace` from a `CostMeter.totals_from(snapshot)`
diff). The adapter additionally exposes `set_step_costs(step_index,
...)` for callers that compute costs after the fact, but the inline
path is preferred when it's available.

## Per-LLM-call modelio capture (augur-sdk 0.1.6+, opt-in)

`modelio/<step:04d>-<layer>-<seq>.json` is the canonical per-call
training-data record (one file per LLM invocation under the bundle).
Mantis ships the plumbing in
`src/mantis_agent/observability/modelio.py` plus a hook inside the
shared `AnthropicToolUseClient`; per-layer call sites opt in by
publishing a context.

**Live streaming (augur-sdk 0.1.10+):** with the pin bumped to
`augur-sdk>=0.1.10`, every `record_modelio` call ALSO POSTs the
redacted record live to `/api/v1/runs/<run_id>/modelio/<relpath>`
on a background thread (in addition to staging the file for the
on-close bundle). The local bundle behavior is unchanged — streaming
is purely additive. If the server returns `403` on the first POST
(tenant not opted in to modelio streaming), the SDK latches the
sink off for the rest of the session and bundle-only consumers see
no regression. `AugurAdapter.record_modelio` is a verbatim forward
to `DebugSession.record_modelio` so all five wired layers
(planner / grounding / model / verifier / step_recovery) get the
streaming behavior automatically with no per-site code changes.

```python
from mantis_agent.observability.modelio import publish_modelio_context

# Around a planner / grounding / verifier / step_recovery / judge LLM call:
with publish_modelio_context(runner._augur, layer="planner", step_index=0):
    plan = decomposer.decompose_text(plan_text)  # auto-captures
    # any call via AnthropicToolUseClient inside this block is also captured
```

Captured layers must be one of the SDK enum literals: `planner`,
`grounding`, `model`, `verifier`, `step_recovery`, `judge`. Unknown
layer names log a warning and fall through as a no-op (typo guard).

**Vendor field-name mapping (gotcha to remember):** the canonical
`modelio.schema.json` `response.usage` block uses **OpenAI** field
names — `prompt_tokens` / `completion_tokens` — with
`additionalProperties: false`. Anthropic responses ship
`input_tokens` / `output_tokens` plus
`cache_creation_input_tokens` / `cache_read_input_tokens`. The
`record_anthropic_modelio` mapper translates at the boundary; never
construct a modelio record by hand from an Anthropic response or
the SDK's Draft 2020-12 validation will reject it.

**Status (#523 multi-PR campaign — closed):** all four LLM layers
Mantis actually has are wired. The fifth layer the SDK enum lists
(`judge`) is N/A for Mantis — no dedicated judge LLM call site
exists in the codebase.

| Layer | Where it's wired | PR |
|---|---|---|
| `planner` | `plan_decomposer.decompose_text` — inline `record_anthropic_modelio` call after the `requests.post` | #527 |
| `grounding` | `gym/step_handlers/form.py:ClaudeGuidedFormHandler.execute` — wraps the dispatch via `_dispatch(...)` helper; covers all ~10 `target_provider.find_*` calls | #531 |
| `verifier` | `gym/_runner_helpers.py:ensure_results_filters` (visual filter gate) + `gym/_runner_helpers.py:execute_step` (plan-author `step.gate=true` branch) | #532 |
| `step_recovery` | `agentic_recovery._call_recovery_tool` (inline wire) + `gym/step_recovery.StepRecoveryPolicy._try_agentic_recovery` (caller wrap) | #533 |
| `judge` | N/A — no judge layer in Mantis | — |

Each wire is a no-op when augur is None or inactive — the default
control path is unchanged from before any of this landed. When
augur is active, an entire run with all four layers exercising
Anthropic will produce something like:

```
modelio/
├── 0000-planner-001.json       # plan_decomposer (run-scoped)
├── 0003-grounding-001.json     # ClaudeGuidedFormHandler on step 2
├── 0003-grounding-002.json     # second find_form_target on same step
├── 0004-verifier-001.json      # step.gate=true verify
├── 0005-step_recovery-001.json # agentic recovery analysis
└── ...
```

Open #523 acceptance items not addressed by this campaign (filed
as separate follow-ups when scheduled): `references` field on
each StepTrace pointing at its modelio files, redaction policy
tightening, `capture_mode`-gated emission (skip on
`metadata`-only runs), 50 MB modelio bundle-size budget warning.

## Continuous verdict score (augur-sdk 0.1.7+)

When a step's typed `Verdict.confidence` is non-zero, the adapter
patches the step's verdict with a continuous reward score via
`augur.set_score(step_index, confidence, comparator="verifier")`.

This replaces Augur's default binary status→score map
(`passed=1.0` / `failed=0.0`) with the verifier's actual signal,
which RLHF / DPO pipelines need for ranked preferences.

- `comparator` must be one of the SDK's canonical values:
  `verifier`, `model-judge`, `exact-match`, `human`. Mantis stamps
  `verifier`.
- Score is clamped to `[0.0, 1.0]` SDK-side; out-of-band values
  trigger a `ValueError` that the wrapper swallows (telemetry
  never breaks runs).
- Steps with `Verdict.confidence == 0` (the default — most
  deterministic handlers never set it) are **skipped**, so we
  don't pollute the bundle with bogus scores.

## Capture-mode upgrade on first failure (augur-sdk 0.1.3+)

The runner upgrades the active capture mode from `metadata` (cheap)
to `screenshots` (full evidence) the first time a step fails:

```python
if not step_result.success and not runner._augur_capture_upgraded:
    augur.set_capture_mode("screenshots")
    runner._augur_capture_upgraded = True
```

Healthy runs stay on the original (cheap) mode; failing runs auto-
collect screenshot evidence from the first failure onward. The
sentinel is idempotent — only the first failure flips the switch.

The override is per-step (stamped on the next `record_step` call,
not on the manifest baseline). To upgrade the baseline globally,
set `AUGUR_CAPTURE_MODE=screenshots` at the runtime instead.

## Failure-class hygiene diagnostic (augur-sdk 0.1.4+)

0.1.4 added a server-side rule, **`cua.uncategorized_failure`**, that
flags any step landing with `status: "failed"` but an empty or literal-
`"unknown"` `failure_class`. The rule surfaces in the per-run
**Diagnostics** panel with severity `low` and recommends tightening
the producer's classifier.

No code change on the Mantis side — purely a server-side surface that
audits the bundles we already ship. Practical impact: a few Mantis
recovery paths emit `failure_class="unknown"` (most visibly on
halted click steps); after the 0.1.4 bump those will start showing
up as `low`-severity diagnostics. Addressing them is hygiene work
for a follow-up, not a regression in the wedge.

## Log streaming (augur-sdk 0.1.3+)

`AugurAdapter.append_log(text, *, step_index=None, name="run")` is a
thin wrapper over `DebugSession.append_log` (added upstream in 0.1.3).
When streaming is enabled, it POSTs the chunk to
`/api/v1/runs/<run_id>/logs` — the server appends it to
`logs/<name>.log` (or `logs/step-<idx>.log` when `step_index` is set),
bounded at 1 MB per file. Workspace surfaces these in the per-step
**logs** panel.

The wrapper is a clean no-op when:
- The runtime ships an older `augur-sdk` install without `append_log`
- No `AUGUR_DSN` is set (no server to stream to)

Mantis does not auto-pipe `logger.*` output into this surface today —
callers explicitly call `runner._augur.append_log(...)` when they have
text worth showing in the viewer. Per-step structured data already
goes through `record_step` / `record_event`; `append_log` is for
free-text additions (e.g. raw model output dumps, recovery rationale
prose, long verifier explanations).

## Coexistence with `TraceExporter`

`mantis_agent.gym.trace_exporter.TraceExporter` keeps writing one JSON
per run to `MANTIS_TRACE_EXPORT_DIR` when configured. The Augur adapter
sits alongside it — the two writers don't share state and neither
replaces the other. Augur's bundle is finer-grained (per-step events,
typed layers, on-disk screenshots) and supports live streaming; the
trace export is the legacy coarse-grained record.

## Verification

To confirm streaming works against a workspace:

1. Export `AUGUR_DSN=https://<key>@<host>/<project>` (and any required
   API key) in the Mantis runtime environment.
2. Submit any gym run (`/v1/predict`, the CLI, or an embedded
   `MicroPlanRunner`).
3. Check the workspace badge — the SDK heartbeats every ~15 s, so the
   "connected" indicator should turn green within one heartbeat of
   the first step landing.
4. After the run finishes, verify the bundle on disk:

   ```bash
   ls "$MANTIS_DATA_DIR/augur/<run_id>/"
   python -c "from augur_sdk.validation import validate_bundle; print(validate_bundle('$MANTIS_DATA_DIR/augur/<run_id>/'))"
   ```

## References

- Augur SDK source: <https://github.com/mercurialsolo/augur-sdk>
- Integration guide: <https://mercurialsolo.github.io/augur-sdk/integrations/mantis/>
- Normative SDK spec: <https://mercurialsolo.github.io/augur-sdk/spec/>
- Adapter authoring: <https://mercurialsolo.github.io/augur-sdk/reference/adapter-authoring/>
