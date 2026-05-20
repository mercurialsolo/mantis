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
