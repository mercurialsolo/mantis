# Streaming run events (SSE)

For interactive UIs and live dashboards, polling
`POST /v1/predict {action: status}` on a fixed cadence is laggy — a
step change in the runner takes up to one cycle to appear. The SSE
event stream pushes events as soon as the runner writes them.

## Endpoint

```
GET /v1/runs/{run_id}/events?sse=true
```

Headers:

| Header | Required | Notes |
|---|---|---|
| `X-Mantis-Token` | yes | Same per-tenant token as every other `/v1/*` route |
| `Last-Event-ID` | no | Resume cursor — server skips rows with `ts ≤ this` |
| `Accept` | no | Defaults to `text/event-stream` when `sse=true` |

Query params:

| Param | Default | Notes |
|---|---|---|
| `sse` | `false` | Set to `true` to opt into the streaming response |
| `since` | `""` | Alternative to `Last-Event-ID` — same skip semantics |

## Event shape

The server emits one event per row in the run's `reasoning.jsonl`,
plus a `phase` event on every phase transition and a single
`terminal` event when the run finishes.

```
event: phase
id: 
data: {"phase": "running", "run_id": "<run_id>"}

event: step
id: 2026-06-08T00:00:01
data: {"ts": "2026-06-08T00:00:01", "kind": "step", "step_index": 4, "type": "click", ...}

event: extract
id: 2026-06-08T00:00:05
data: {"ts": "2026-06-08T00:00:05", "kind": "extract", "step_index": 7, "row": {...}}

event: terminal
id: 
data: {"phase": "complete", "run_id": "<run_id>", "halt_class": null}
```

Event names match the row's `kind` (or `type`) field, falling back to
`message` for rows that don't carry one. Phases are the same six-phase
taxonomy the lifecycle routes use: `queued / running / recovering /
complete / halted / cancelled`.

## Resumability

The server bounds each stream at ~10 minutes — long-running plans need
the client to reconnect. The standard SSE pattern (most browser and
Python EventSource libraries do this automatically): the client tracks
the last `id:` it saw and sends it back as the `Last-Event-ID` header
on reconnect. The server skips all rows with timestamp ≤ that cursor.

```bash
# Curl example — replace LAST_ID with the id of the last event you saw
curl -N \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Last-Event-ID: 2026-06-08T00:00:05" \
  "$ENDPOINT/v1/runs/$RUN_ID/events?sse=true"
```

Heartbeat comments (`:ping`) arrive every ~25 seconds so reverse
proxies don't drop the idle connection — the standard SSE parsers
ignore them automatically.

## Python SDK

`MantisClient.stream_events` wraps the SSE parsing so callers iterate
over dicts:

```python
from mantis_agent.client import MantisClient

client = MantisClient.from_env()
handle = client.predict({"task_suite": {"_micro_plan": [...]}})

for event in client.stream_events(handle.run_id):
    if event["event"] == "phase":
        print(f"phase → {event['data']['phase']}")
    elif event["event"] == "extract":
        print(f"row: {event['data']['row']}")
    elif event["event"] == "terminal":
        print(f"done: {event['data']}")
        break
```

The iterator exits on the `terminal` event, on a hung connection, or
when the server reaches its 10-minute soft limit. For long runs, wrap
in a reconnect loop that passes the last seen `id` back as `since=`:

```python
last_id = ""
while True:
    for event in client.stream_events(handle.run_id, since=last_id):
        last_id = event["id"] or last_id
        if event["event"] == "terminal":
            break
    else:
        # Soft timeout — reconnect.
        continue
    break
```

## When NOT to use SSE

| Need | Use |
|---|---|
| One-shot result fetch | `POST /v1/predict {action: result, run_id}` |
| Polling status from a cron job | `GET /v1/runs/{id}` (cheap-poll + backoff hint) |
| Backend-to-backend notification | Webhooks (see [Runs and polling](runs-and-polling.md#webhooks-instead-of-polling)) |

SSE shines for interactive UX where every step transition should
appear immediately. For batched / async consumers, the polling-with-
backoff-hint path is cheaper.

## See also

- [Runs and polling](runs-and-polling.md) — the action-based status surface and webhooks
- [Errors](errors.md) — error reference
- [Reference / HTTP API](../api.md) — full endpoint detail
