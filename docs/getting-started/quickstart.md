# Quickstart — five minutes to a real extraction

If someone has already deployed Mantis (or you're using the public Baseten reference deployment), all you need is a curl, your tenant token, and a plan. This page does an end-to-end run that produces three real BoatTrader leads.

## Prerequisites

- `curl`, `jq`
- A **Baseten API key** (`BASETEN_API_KEY`) for the gateway
- A **Mantis tenant token** (`MANTIS_API_TOKEN`) issued by your operator — see [Tenant keys](../operations/tenant-keys.md) if you're the operator

```bash
export BASETEN_API_KEY="<your baseten api key>"
export MANTIS_API_TOKEN="<your mantis tenant token>"
# Baseten forwards /sync/<any path> to the container's FastAPI app. The
# legacy /predict route still works (gateway treats it as /sync/predict).
export ENDPOINT="https://model-qvvgkneq.api.baseten.co/production/sync"
```

## 1. Submit the plan

```bash
RESP=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "detached": true,
    "micro": "plans/boattrader/extract_url_filtered_3listings.json",
    "state_key": "first-quickstart",
    "max_cost": 2,
    "max_time_minutes": 20,
    "record_video": true
  }')

RUN_ID=$(echo "$RESP" | jq -r .run_id)
echo "run_id: $RUN_ID"
```

Expected response:

```jsonc
{
  "status": "queued",
  "model": "holo3",
  "mode": "detached",
  "run_id": "20260428_021432_076255ef",
  ...
}
```

## 2. Poll for completion

```bash
while true; do
  STATUS=$(curl -fsS -X POST "$ENDPOINT/v1/predict" \
    -H "Authorization: Api-Key $BASETEN_API_KEY" \
    -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"action\":\"status\",\"run_id\":\"$RUN_ID\"}" \
    | jq -r .status)
  echo "$(date '+%H:%M:%S')  $STATUS"
  case "$STATUS" in succeeded|failed|cancelled) break ;; esac
  sleep 30
done
```

A successful run completes in ~10 minutes (cold-start image build is ~50 min if the instance scaled to zero — typically already warm).

## 3. Fetch the leads

```bash
curl -fsS -X POST "$ENDPOINT/v1/predict" \
  -H "Authorization: Api-Key $BASETEN_API_KEY" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"action\":\"result\",\"run_id\":\"$RUN_ID\"}" \
  | jq .result.leads
```

You should see something like:

```jsonc
[
  "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | Price: <Price> | Phone: <Phone or 'none'>",
  "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | Price: <Price> | Phone: none",
  "VIABLE | Year: <YYYY> | Make: <Make> | Model: <Model> | Price: <Price> | Phone: none"
]
```

The exact field set depends on the `extraction_schema` you submit
([see ExtractionSchema in the recipes](../integrations/recipes.md)).

## 4. Download the screencast

```bash
curl -fsS -o demo.mp4 \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  "$ENDPOINT/v1/runs/$RUN_ID/video"
open demo.mp4
```

You'll see the three-segment walkthrough:

1. Title card (Mantis CUA · plan name · tenant · run id)
2. The actual BoatTrader navigation, with per-step captions and overlays for every action (click ripples, scroll arrows, key chord badges, type captions)
3. Outro card with the result summary (3 viable leads, 1 with phone, $0.42, 9.5 min)

Pass `?raw=1` to fetch the un-overlaid screen capture instead.

## What just happened

```
your curl                         Baseten gateway          Mantis container         Holo3 GPU
──────────                        ────────────────         ────────────────         ─────────
POST /v1/predict ──────────────►  auth: Api-Key  ─────────► auth: X-Mantis-Token
                                                            │
                                                            ▼
                                                     MicroPlanRunner
                                                     loads the 3-listing plan
                                                            │
                                                            ▼
                                                     for each step:
                                                       ↓
                                                     Holo3 ◄──────────────────────► /v1/chat/completions
                                                       ↓                            (action proposal)
                                                     ClaudeGrounding ◄───────────►  Anthropic API
                                                       ↓                            (refined click)
                                                     xdotool click on Xvfb
                                                       ↓
                                                     ClaudeExtractor ◄───────────►  Anthropic API
                                                       ↓                            (lead row)
                                                     checkpoint to volume
                                                            ▼
                                                     ffmpeg captures screencast in parallel
                                                            ▼
GET /v1/runs/<id>/video ──────────►                  polished video composed
```

You ran one orchestrated extraction against a live website, used both Holo3 (cheap GPU clicks) and Claude (extraction reasoning), with a typed video walkthrough at the end.

## Next steps

- Read the [Concepts](concepts.md) page so you understand `state_key`, `max_cost`, gates, and sections before designing your own plan.
- Browse the [Plan formats](plan-formats.md) to pick the right shape for your workflow.
- Want to host your own instance? Go to [Hosting](../hosting/index.md).
- Building an integration? Start with [Client](../client/index.md) → [Authentication](../client/auth.md).
