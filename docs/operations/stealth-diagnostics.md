# Stealth diagnostics + tuning

Mantis's stealth posture is layered across the JS-injected CDP patches
(`src/mantis_agent/gym/cdp_stealth.py`), behavioral signals at the
xdotool layer, timezone/locale consistency with the proxy exit, and
the Browser-Use Plane driver choice. This page documents the env
flags you can flip, the diagnostic endpoint you use to measure the
result, and the recommended tuning workflow.

> Background and design rationale: [#822 stealth posture v2 epic](https://github.com/mercurialsolo/mantis/issues/822)
> and its children.

## Env flags (all default-on)

| Variable | Default | Effect |
|---|---|---|
| `MANTIS_CDP_STEALTH` | `1` | CDP-injected fingerprint patches (`navigator.webdriver` undefined, plugins, WebGL, etc.). |
| `MANTIS_BEHAVIORAL_JITTER` | `1` | Pre-click Bezier mouse path + jittered settle times ([#824](https://github.com/mercurialsolo/mantis/issues/824)). |
| `MANTIS_GEO_CONSISTENCY` | `1` | Set Chrome's `TZ` and `LANG` to match the proxy exit geo ([#825](https://github.com/mercurialsolo/mantis/issues/825)). |
| `MANTIS_BROWSER_USE_DRIVER` | unset → `patchright` | Browser-Use Plane driver. Set to `playwright` to force the vanilla import ([#826](https://github.com/mercurialsolo/mantis/issues/826)). |
| `MANTIS_PROXY_PROVIDER` | `oxylabs` | Residential proxy provider. PrivateProxy and IPRoyal also wired up. |

Every flag accepts the falsy set `0` / `false` / `no` / `off` (case-insensitive)
to disable. Unknown values are treated as truthy — opt-out is explicit.

## Diagnostic endpoint — `POST /v1/diagnose/fingerprint`

Submits a fingerprint-test plan against a public bot-detection
diagnostic page. The run extracts every visible test row (test name +
result) into the standard artifact pipeline so you can grep / diff
across runs.

```bash
curl -fsS -X POST "$ENDPOINT/v1/diagnose/fingerprint" \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"target_url": "https://bot.sannysoft.com/"}'
```

Response (detached run handle + active-stealth snapshot):

```jsonc
{
  "run_id": "20260609_181500_a1b2c3d4",
  "target_url": "https://bot.sannysoft.com/",
  "poll_via": "/v1/runs/20260609_181500_a1b2c3d4",
  "rows_via": "/v1/runs/20260609_181500_a1b2c3d4/artifacts/extracted_rows.json",
  "status": "queued",
  "stealth_snapshot": {
    "honest_mode": true,
    "behavioral_jitter": true,
    "geo_consistency": true,
    "cdp_stealth": true,
    "proxy_provider": "oxylabs"
  }
}
```

Poll lifecycle, fetch rows:

```bash
RID=20260609_181500_a1b2c3d4
# Poll
curl "$ENDPOINT/v1/runs/$RID" -H "X-Mantis-Token: $TOKEN"
# Fetch the per-test rows once terminal
curl "$ENDPOINT/v1/runs/$RID/artifacts/extracted_rows.json" -H "X-Mantis-Token: $TOKEN"
```

### Body fields

| Field | Default | Notes |
|---|---|---|
| `target_url` | `https://bot.sannysoft.com/` | Must start with `http://` or `https://`. Other candidates: `https://abrahamjuliot.github.io/creepjs/`, `https://browserleaks.com/` |
| `cua_model` | `holo3` | One of `holo3` or `claude`. |

The endpoint is intentionally cheap: max 8 steps, 3-minute time cap,
$0.30 cost cap. It uses the multi-row `extract_data` branch ([#820](https://github.com/mercurialsolo/mantis/pull/820))
so up to 60 fingerprint test rows come back from one Claude call.

## Recommended tuning workflow

1. **Baseline.** Run the diagnostic with default flags. Save the
   resulting `extracted_rows.json` as `baseline.json`.

   ```bash
   curl -X POST "$ENDPOINT/v1/diagnose/fingerprint" -d '{}' ...
   # wait for terminal, fetch rows
   curl "$ENDPOINT/v1/runs/$RID/artifacts/extracted_rows.json" > baseline.json
   ```

2. **Flip one flag at a time.** E.g. disable behavioral jitter:

   - Set `MANTIS_BEHAVIORAL_JITTER=0` in `.env`.
   - Redeploy Modal: `modal app stop mantis-cua-server && modal deploy ...`.
   - Re-run the diagnostic. Save as `no-behavioral.json`.

3. **Diff the scorecards:**

   ```bash
   diff <(jq -S '.' baseline.json) <(jq -S '.' no-behavioral.json)
   ```

4. **Verify in production** with a canonical CF-protected target
   (boattrader.com / luma.com / your domain). Compare the auto-pause-
   on-`cf_challenge` halt rate week-over-week in the
   `mantis_loop_termination_total{reason}` metric.

## What each flag actually does

### `MANTIS_CDP_STEALTH=1` (default)

`src/mantis_agent/gym/cdp_stealth.py` injects 12 JS patches at
`Page.addScriptToEvaluateOnNewDocument` time, plus a CDP
`Network.setUserAgentOverride` that aligns `sec-ch-ua-*` headers with
the UA we claim. Patches cover `navigator.webdriver`, `plugins`,
`languages`, WebGL vendor/renderer, canvas + audio fingerprint noise,
font enumeration, and platform spoofing.

### `MANTIS_BEHAVIORAL_JITTER=1` (default)

`src/mantis_agent/gym/behavioral.py` exposes:

- `bezier_waypoints(start, end, steps)` — sampled Bezier curve
  between the current cursor position and the click target. Used in
  the xdotool click handler so the cursor moves along a curved path
  before the click instead of teleporting and firing.
- `jittered_settle(base)` / `jittered_wait(base)` — randomized delay
  helpers (`uniform(-0.3, +0.6)` around the base, floor at `base / 2`).

### `MANTIS_GEO_CONSISTENCY=1` (default)

`src/mantis_agent/gym/geo_consistency.py` resolves the proxy's
`diagnose_proxy_egress` payload to an IANA timezone + BCP-47 language
tag. `setup_env` writes the resolved `TZ` and `LANG` into the process
env *before* Chrome starts, so Chrome inherits a wall-clock that
matches the proxy exit IP. State-level resolution for US proxies
(Phoenix AZ → `America/Phoenix`); falls back to the country default
for multi-tz states. Unknown countries land on `America/New_York` +
`en-US` — the helper never invents a country it can't verify.

### `MANTIS_BROWSER_USE_DRIVER=patchright` (default)

Per `feedback_headless_vs_xvfb.md`, Cloudflare detects headless
Playwright with high reliability. The Browser-Use Plane now defaults
to importing [`patchright`](https://github.com/Kaliiiiiiiiii-Vinyxhi/patchright-python),
a drop-in patched fork that strips Playwright's automation tells at
the binary level. Set `MANTIS_BROWSER_USE_DRIVER=playwright` to fall
back to vanilla for the rare targets that detect patchright
specifically.

## Out of scope

- Solving CF Turnstile challenges interactively (the pause-on-challenge
  flow from [#541](https://github.com/mercurialsolo/mantis/pull/541)
  already handles residual cases).
- TLS / JA3 spoofing at the network layer. Real Chrome's TLS is the
  honest answer — see the epic for the rationale.
- `MANTIS_STEALTH_HONEST` (drop Windows spoof, present as real Linux
  Chrome) — tracked separately as [#823](https://github.com/mercurialsolo/mantis/issues/823).

## See also

- Epic: [#822 stealth posture v2 — present honestly](https://github.com/mercurialsolo/mantis/issues/822)
- Metrics: [Prometheus per-action dashboard](metrics.md)
- Augur per-run debug bundles: [Augur integration](../integrations/augur.md)
