# Ablation harness

Quality-related changes need paired ON/OFF evidence — running ON-arm
only and inspecting the response surface misses silent regressions.
The `scripts/ablate_v1_cua.py` harness does single-deploy A/B against
a warm Modal container using per-request overrides on `/v1/cua`.

## Usage

```bash
export MANTIS_ENDPOINT=https://getmason--mantis-server-api.modal.run
export MANTIS_API_TOKEN=mantis_…

python scripts/ablate_v1_cua.py \
    --toggle perceptual_verify \
    --instruction "Find a sign-in button and click it. Then call done." \
    --start-url https://lu.ma/discover \
    --pairs 2
```

What it does:

1. **Warm-up** — one request whose results are discarded so subsequent
   requests reuse the Chrome session cache (#311) and don't include
   ~10 s cold-start noise. Skip with `--skip-warmup` when the
   container is already warm from a previous run.
2. **Paired requests** — for each pair: one OFF arm (toggle=false),
   one ON arm (toggle=true). Identical instruction, same warm container.
3. **Diff report** — prints per-pair side-by-side fields, then an
   aggregate (success rate Δ, wall-time Δ, per-reason count Δ,
   request errors). Marks regression flags explicitly.

## Available toggles

Every major runner toggle has a per-request override on `/v1/cua`:

| Toggle name (payload field) | Issue | Default env var |
|---|---|---|
| `perceptual_verify` | #293 | `MANTIS_PERCEPTUAL_VERIFY` |
| `loop_recovery` | #302 | `MANTIS_LOOP_RECOVERY` |
| `done_gate` | #303 | `MANTIS_DONE_GATE` |
| `predicate_verify` | #291 | `MANTIS_PREDICATE_VERIFY` |
| `adaptive_settle` | #294 | `MANTIS_ADAPTIVE_SETTLE` |
| `form_controller` | #301 | `MANTIS_FORM_CONTROLLER` |
| `reuse_session` | #311 | `MANTIS_CHROME_REUSE` |
| `speculation` | #118 | `MANTIS_SPECULATIVE_INFERENCE` |

When a per-request override is present, it overrides the container's
env var for that single request only. Other concurrent requests on the
container would not be affected (only one `/v1/cua` request runs at a
time on a single container today, so this is safe).

## Reading the report

Each pair prints a field table marked `Δ` on changed fields:

```
=== Pair 1 ===
  field                      OFF                               ON
Δ done_rejections_by_reason  {}                                {"empty_summary":2}
Δ steps                      1                                 3
  success                    True                              True
```

The aggregate at the bottom sums signals across pairs and flags
regressions:

```
=== Aggregate (done_gate, n=2 pairs) ===
  success rate           OFF=2/2  ON=2/2
  wall mean              OFF=15.1s  ON=13.5s
  steps mean             OFF=2.0   ON=3.0
  done_rejections_by_reason
    OFF: {}
    ON:  {"empty_summary":4}
  ✅ no regression flags
```

Flags emitted automatically:

- `success regression: ON dropped N successes vs OFF` — fires when
  the ON arm's success count is strictly lower.
- `ON arm had request errors — investigate logs` — fires on any
  HTTP error or JSON decode failure.
- `wall-time regression: ON > OFF * 1.5` — fires on a substantial
  per-arm wall increase. Verify against the Modal logs before
  blaming the toggle — pre-existing hangs (e.g. [#320](https://github.com/mercurialsolo/mantis/issues/320))
  can surface as an outlier on one arm.

## Discipline

**Required** for any PR that touches:
- `gym/runner.py` paths that affect the brain↔env loop
- `gym/done_gate.py`, `gym/predicates.py`, `gym/perceptual_diff.py`,
  `gym/loop_recovery.py`, `gym/form_controller.py`, `gym/adaptive_settle.py`
- `brain_*` parsers / classifiers
- `verification/`

The PR body must include an **Ablation report** section with the
harness output, at least 2 pairs, and a `✅ no regression flags` line
or an explicit explanation of any flagged anomaly.

**Optional** for:
- Pure refactors with no observable behaviour change
- Pure plumbing PRs that don't change runner decisions
- Pure docs / typing / lint changes

## Retroactive ablations (2026-05-12)

To verify the existing quality PRs didn't introduce silent regressions:

| PR | Toggle | Verdict | Notes |
|---|---|---|---|
| #316 (#293 perceptual diff) | `perceptual_verify` | ✅ no regression | One outlier traced to [#320](https://github.com/mercurialsolo/mantis/issues/320), pre-existing |
| #317 (#302 loop recovery) | `loop_recovery` | ✅ no regression | Recovery fires on submit-shaped loops as designed |
| #306 (#303 done gate) | `done_gate` | ✅ no regression | Gate fires on empty-summary dones; intended ~1 extra step |

All three quality PRs are clean. The harness flagged one infrastructure
issue (#320) during the #293 run — exactly the kind of hidden regression
the discipline is meant to catch.

## Limitations

- **Single-container assumption**: the harness reuses the same Chrome
  session across pairs (via #311 cache) so toggle effects aren't
  conflated with cold-start variance. If Modal scales up and routes
  pairs to different containers, the wall-time comparison gets noisy.
- **Holo3-Q8 stochasticity**: temperature is 0 but Holo3 still emits
  slightly different actions on identical inputs sometimes (vLLM
  numerical noise). For high-confidence regression detection, `--pairs 4`
  or higher; the default 2 is enough for catching obvious failures.
- **No quality metric beyond success/steps/aggregate counters**: the
  harness can't tell you whether a successful run actually achieved
  the user's goal (vs the brain emitting a fake `done(success=true)`
  with a fabricated summary). Pair with #303 done-gate to mitigate.

## See also

- [#261](https://github.com/mercurialsolo/mantis/issues/261) Harness
  ablation discipline — the original framing this work delivers on.
- [Done-acceptance gate](done-gate.md), [Perceptual diff verifier](perceptual-diff.md),
  [Loop recovery policy](loop-recovery.md) — the three quality features
  this harness retroactively ablated.
