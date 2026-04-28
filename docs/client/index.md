# Client / integration

For applications calling a hosted Mantis instance.

| Topic | What it covers |
|---|---|
| [Authentication](auth.md) | Getting a token, request headers, scope, error responses |
| [Sending plans](plans.md) | Request body shape for each plan format, run options, caps |
| [Runs and polling](runs-and-polling.md) | Detached runs, status/result/logs/cancel actions, idempotency, webhooks |
| [Recordings](recordings.md) | Requesting screencasts, downloading polished or raw, action overlays |
| [Errors](errors.md) | Error codes, retry guidance, debugging |

Looking for the full HTTP API surface? See [Reference / HTTP API](../api.md).

## Minimal client (Python)

```python
import os, time, requests

ENDPOINT = os.environ["MANTIS_ENDPOINT"]      # e.g. https://model-qvvgkneq.api.baseten.co/production
PLATFORM = os.environ["BASETEN_API_KEY"]
TENANT   = os.environ["MANTIS_API_TOKEN"]

def headers():
    h = {"X-Mantis-Token": TENANT, "Content-Type": "application/json"}
    if PLATFORM:
        h["Authorization"] = f"Api-Key {PLATFORM}"
    return h

def submit(plan_path: str, state_key: str, **opts) -> str:
    r = requests.post(f"{ENDPOINT}/v1/predict", headers=headers(), json={
        "detached": True,
        "micro": plan_path,
        "state_key": state_key,
        **opts,
    }, timeout=60)
    r.raise_for_status()
    return r.json()["run_id"]

def poll(run_id: str, every: int = 30) -> dict:
    while True:
        r = requests.post(f"{ENDPOINT}/v1/predict", headers=headers(), json={
            "action": "status", "run_id": run_id,
        }, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(every)

def fetch_result(run_id: str) -> dict:
    r = requests.post(f"{ENDPOINT}/v1/predict", headers=headers(), json={
        "action": "result", "run_id": run_id,
    }, timeout=60)
    r.raise_for_status()
    return r.json()

# Usage
run_id = submit(
    "plans/boattrader/extract_url_filtered_3listings.json",
    "boattrader-prod",
    max_cost=2, max_time_minutes=20, record_video=True,
)
final = poll(run_id)
print(final["summary"])
result = fetch_result(run_id)
print(result["result"]["leads"])
```

That's the whole integration. The rest of this section is the deep-dive on each piece.
