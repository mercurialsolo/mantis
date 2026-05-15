# Form-target providers

`fill_field` / `submit` / `select_option` steps locate their targets
through a **form-target provider** — an implementation of
`mantis_agent.form_targeting.FormTargetProvider`. The provider returns
pixel coordinates given a screenshot plus a description of the
element to find.

Two providers ship today:

| Provider | Backend | When to use |
|----------|---------|-------------|
| `ClaudeFormTargetProvider` (default) | Anthropic Claude vision | Stable. Best for forms with non-English labels or icon-only controls — Claude reads prose well. |
| `Holo3FormTargetProvider` | Existing Holo3 brain endpoint | Independent quota pool — survives Claude API overload (529). Cheaper per call. Less reliable on tightly-packed forms; reads prose poorly. |

## Selection

The runner picks a provider at startup based on `MANTIS_FORM_TARGET_PROVIDER`:

| Value | Effect |
|-------|--------|
| unset / `""` / `claude` | `ClaudeFormTargetProvider` |
| `holo3` | `Holo3FormTargetProvider` with a Claude fallback for `verify_dropdown_value` (Holo3 isn't tuned for reading dropdown text) |
| anything else | Warning logged, falls back to `claude` |

Unset is the safe default — change only after running the smoke gate
below.

## Why two providers

Issue #403 surfaced the failure mode: a single Anthropic 529 spike
during a `fill_field` step's budget halted the lu.ma host-question
plan. The #404 retry-with-backoff survives 1–2 transient blips but
can't outlast a multi-minute overload window. Holo3 already runs in
the same Modal container with its own quota — routing form-target
calls there is independent recovery, not a model swap.

## Smoke gate before flipping the default

Holo3 has higher coordinate variance than Claude on tightly-packed
forms. Before changing the default away from `claude`:

1. Run each plan in `plans/` that contains a form (lu.ma register,
   staff-crm lead update, login flows) five times under each provider.
2. Record success rate, mean cost / step, mean wall-clock / step.
3. Update the table below.
4. Flip the default only when Holo3 is within 5% success-rate parity
   and >30% faster or cheaper.

| Plan | Provider model | Runs | Success rate | Mean grounding calls / run | Mean $ / run |
|------|----------------|-----:|-------------:|---------------------------:|-------------:|
| `plans/staff-crm` | Opus (default) | 4 | 2/4 (1 complete on best, 1 complete after recovery, 2 halts on form-validation) | 36 (good) / 75-90 (bad) | $0.21 (good) / $0.54-0.65 (bad) |
| `plans/staff-crm` | Haiku (override) | 1 | 0/1 | 93 | $0.65 |

The Haiku-on-grounding A/B (#434) ran a side-by-side comparison of
the same plan with the same code. Haiku's per-call price is ~25%
of Opus, but its lower accuracy increased the grounding-call count
~24% (75 → 93 calls on the canonical staff-crm halt). The retry
amplification more than ate the per-call savings — total cost was
*higher* on Haiku. The default reverted to "match the extractor's
main model" with `form_target_model` kept as an opt-in kwarg for
future experiments.

Recovery interventions that landed on top of this (#435 region
cropping + tab-blur + cascade cap) brought the staff-crm happy
path down to $0.19 / 14 grounding calls — these matter much more
for cost than the per-call model price.

## Wiring summary

- `mantis_agent/_anthropic/client.py` — shared `AnthropicToolUseClient`
  with retry/backoff (#403). Both providers (when applicable) and the
  extraction code share one instance per runner.
- `mantis_agent/form_targeting/base.py` — protocol + result shape.
- `mantis_agent/form_targeting/claude.py` — Claude implementation.
- `mantis_agent/form_targeting/holo3.py` — Holo3 implementation.
- `mantis_agent/form_targeting/factory.py` — env-var selection.
- `mantis_agent/gym/step_context.py` — `form_target_provider` field;
  the form handler reads it via `ctx.form_target_provider`.

`MicroPlanRunner.__init__` calls `build_form_target_provider` and
stores the result on `self.form_target_provider`. The runner passes
that down through `StepContext` for every step. Callers that build a
`StepContext` directly (tests, ad-hoc scripts) can leave
`form_target_provider=None` — the form handler falls back to the
extractor's back-compat shims.
