# mantis-boattrader

High-fidelity replica of [boattrader.com](https://www.boattrader.com/) backed
by a closed, deterministic catalog. Used as a sim env for marketplace-shaped
agent tasks: faceted search, lead capture, ad rotation, cookie consent,
configurable latency.

## Pages

| URL | Description |
|----|----|
| `/`                          | Home — hero search panel, hero ad carousel, "Boats Near You", featured brands, popular boat types, popular boats, mid + footer ad strips, recent articles |
| `/boats/`                    | Search results page (SRP) with full filter sidebar (location, condition, length, year, price, boat type, make), pre-qualify banner, sort dropdown, 3-up listing grid, pagination, native ad card mid-grid |
| `/boat/<slug>/`              | Boat detail page — gallery, badges, price + monthly, view/save counters, dealer card with contact form, owner highlights, spec grid, propulsion, similar boats |
| `POST /boat/<slug>/contact`  | Submit a lead — re-renders detail with confirmation banner |
| `POST /__site/consent`       | Accept / reject cookies (sets `bt_cookie_consent` cookie) |

## Harness endpoints (require `X-Env-Admin: <token>`)

| URL | Description |
|----|----|
| `GET /__env__/health`     | Liveness + boat/dealer counts + latency settings (open — no admin required) |
| `POST /__env__/reset`     | Reseed catalog + clear leads / mutations |
| `GET /__env__/state`      | Catalog facets + lead count + mutation count + last 50 mutations |
| `GET /__env__/leads`      | All lead-form submissions |
| `GET /__env__/mutations?since=<id>` | Full mutations audit log (or tail since the given id) |
| `GET /__env__/oracle?task_id=<id>` | Grade an agent run against the named plan task. Returns `{passed, score, reasons, diff, task_id}`. |
| `POST /__env__/config?latency_ms_min=…&latency_ms_max=…&failure_rate=…` | Mutate latency / failure injection live |

### Mutations audit log

Every state-changing public route appends a row to the in-memory
mutations log so oracles can grade what the agent actually did:

| Operation | Fired by | Target |
|----|----|----|
| `lead_submitted` | `POST /boat/<slug>/contact` | `boat` (boat id) |
| `consent_set`    | `POST /__site/consent`      | `session` (empty) |
| `phone_revealed` | `POST /boat/<slug>/show-phone` | `boat` (boat id) |
| `env_reset`      | `POST /__env__/reset`       | `env` (empty) |

The log lives on the in-memory `Store` (see `app/db.py`); it's cleared
on every `/__env__/reset`.

### Plan tasks + oracles

Each plan task registers a server-side grader under `app/oracles/`.
The agent (or harness) calls `GET /__env__/oracle?task_id=<id>` to get
a deterministic verdict: same store snapshot → same `{passed, score}`.

| Task ID | Goal | Grader |
|----|----|----|
| `BT01_lead_capture_filtered_search` | Apply `condition=used` + `make=Sea Ray` + `price_max=200000` filters, click a matching listing, submit the dealer contact form. | `app/oracles/bt01_lead_capture_filtered_search.py` — F1 over (hit leads on qualifying boats, miss leads on non-qualifying / malformed). Pass = ≥1 hit, 0 misses. |

New plan tasks land as a new `app/oracles/<task_id>.py` + an entry in
`app/oracles/__init__.py:GRADERS`. Each grader must be deterministic;
the contract is asserted in `tests/sim_envs/mantis_boattrader/test_oracle_determinism.py`.

## Configurable latency

Every public request sleeps a uniform random time between
`LATENCY_MS_MIN` and `LATENCY_MS_MAX`. Set both to `0` for no latency
(default). `LATENCY_FAILURE_RATE` (0..1) makes that fraction of requests
return `503` with a `retry_after_ms` payload.

```
docker run -p 8033:8080 \
    -e LATENCY_MS_MIN=120 \
    -e LATENCY_MS_MAX=480 \
    -e LATENCY_FAILURE_RATE=0.05 \
    mantis/sim-env-mantis-boattrader:latest
```

## Local

```
python -m uvicorn app.main:app --port 8033
open http://127.0.0.1:8033/
```

## Daytona deploy

```
uv pip install daytona
uv run python deploy/sim_envs/daytona_mantis_boattrader.py \
    --latency-min 120 --latency-max 480
```

Reads `DAYTONA_API_KEY` from repo `.env`. Prints public URL + preview token + admin token.

## Catalog

- 600 boats across 10 boat types and 25 makes
- 20 dealers across 30 US coastal/lake markets
- Procedural SVG hero/listing imagery (no outbound network)
- 6 sponsor ad creatives in rotation across leaderboard/footer/rail slots

Tune via env vars: `SEED`, `BOAT_COUNT`, `FAKE_NOW`.
