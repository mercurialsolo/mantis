# Client / integration

For applications calling a hosted Mantis instance.

| Topic | What it covers |
|---|---|
| [Authentication](auth.md) | Getting a token, request headers, scope, error responses |
| [Sending plans](plans.md) | Request body shape for each plan format, run options, caps |
| [Pure CUA mode](pure-cua.md) | Holo3 pass-through via `/v1/cua` — no decomposition, no Claude |
| [Runs and polling](runs-and-polling.md) | Detached runs, status/result/logs/cancel actions, idempotency, webhooks |
| [Recordings](recordings.md) | Requesting screencasts, downloading polished or raw, action overlays |
| [Errors](errors.md) | Error codes, retry guidance, debugging |

Looking for the full HTTP API surface? See [Reference / HTTP API](../api.md).

## Typed Python client

```bash
pip install "mantis-agent[client]"
```

```python
from mantis_agent.client import MantisClient, PredictRequest

# Reads MANTIS_ENDPOINT / BASETEN_API_KEY / MANTIS_API_TOKEN from env.
client = MantisClient.from_env()

result = client.run_to_completion(
    PredictRequest(
        micro="plans/example/extract_listings.json",
        profile_id="marketplace-prod",
        workflow_id="marketplace-listings-v1",
        max_cost=2,
        max_time_minutes=20,
        record_video=True,
    ),
)
print(result["result"]["leads"])
```

That's the whole integration. The client wraps the five end-user
endpoints with typed requests / responses, structured errors, and a
polling helper. Sync `requests` + `pydantic` only — no FastAPI, no GPU
deps, no browser.

### Fire-and-poll with status callbacks

For longer runs where you want to surface progress to a UI:

```python
handle = client.predict(PredictRequest(
    micro="plans/example/extract_listings.json",
    profile_id="marketplace-prod",
    workflow_id="marketplace-listings-v1",
    record_video=True,
))

final = client.wait_for_completion(
    handle.run_id,
    poll_interval_s=30,
    on_status=lambda s: print(f"{s.status} — {s.summary or '—'}"),
)
print(final.summary)

result = client.result(handle.run_id)
client.fetch_video(handle.run_id, dest_path="recording.mp4")
```

### Pure CUA pass-through

For single-instruction Holo3 runs (no Claude decomposition):

```python
response = client.cua_run(
    "Open https://example.com and click the docs link",
    start_url="https://example.com",
    max_steps=20,
    detached=False,
)
```

### Error handling

Every server failure surfaces as a typed exception so callers can branch
on the cause without parsing string messages:

```python
from mantis_agent.client import (
    MantisAuthError, MantisRateLimitError, MantisAPIError,
    MantisRunFailed, MantisTimeoutError,
)

try:
    result = client.run_to_completion(req, timeout_s=1800)
except MantisAuthError:
    # 401 / 403 — bad or missing X-Mantis-Token
    raise
except MantisRateLimitError as exc:
    time.sleep(exc.retry_after_seconds or 60)
except MantisTimeoutError as exc:
    # Run is still in flight; cancel or keep polling.
    client.cancel(exc.run_id)
except MantisRunFailed as exc:
    # Terminal failure — exc.status carries the snapshot.
    print(exc.status.error)
```

### Raw HTTP

Prefer to drive the API directly? Build a request body following the
shape on [Sending plans](plans.md) and POST it to `/v1/predict` with
`Authorization: Api-Key …` + `X-Mantis-Token: …` headers. The typed
client is a thin wrapper around that — nothing else.

The rest of this section is the deep-dive on each piece.
