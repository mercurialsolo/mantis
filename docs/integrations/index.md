# Integrations

Mantis ships as a library and a service. The right integration shape depends
on what you're building.

## Decision flow

```
Need to call Mantis from your existing app?
  ├─ Just need extraction from a website        → HTTP only          (5 min onboarding)
  ├─ Have your own Python desktop / Xvfb stack  → Library embedding  (~150 LoC adapter)
  └─ Custom CUA backend in someone else's product → Host integration  (parity layer)
```

## The five integration patterns

| Pattern | When | Doc |
|---|---|---|
| **Quickstart** — curl, plan_text, get leads back | First time, want to see it work in 5 minutes | [Quickstart](../getting-started/quickstart.md) |
| **Generic CUA over HTTP** — your own domain, your own ExtractionSchema, all via `/v1/predict` | Building an extraction pipeline for any site (jobs, products, listings, profiles, etc.) without writing Python or hosting Mantis | [Generic CUA usage](generic-cua.md) |
| **Recipes** — copy-paste micro-plans for common patterns (jobs, e-commerce, news, real-estate) | You want a working starting point for a typical pattern, not theory | [Recipes](recipes.md) |
| **Library embedding** — `MicroPlanRunner` in your own Python process, your own `GymEnvironment` | You already own a desktop / Xvfb / Chrome stack and want Mantis as the brain | [Embedding MicroPlanRunner](embedding-microplanrunner.md) |
| **Any-agent integration checklist** — runtime contract, diagnostic log lines, the integration mistakes that produce silent halts | You're plugging a third-party agent (OpenAI CUA, Anthropic CUA, Voyager, custom) into Mantis and want to avoid rediscovering known failure modes | [Integrating any agent](any-agent.md) |

## What's host-agnostic vs. host-specific

The library surface (`mantis_agent.MicroPlanRunner`, `PauseRequested`, `register_tool`, `step_callback`, `cancel_event`) makes no assumptions about what host you're running in. Every doc under `docs/integrations/` is host- and client-neutral by design — it works the same regardless of what desktop wrapper, agent loop, or downstream product you're plugging into. Client-specific narratives (file paths, internal SHAs, named tenants) live in `internal-docs/` and are not part of the public site.

If you're integrating a fresh codebase that has no existing browser stack, the [Client](../client/index.md) section + the recipes here are the right starting point. If you're plugging Mantis into an established codebase that already runs Xvfb, Chrome, or its own agent loop, [Embedding MicroPlanRunner](embedding-microplanrunner.md) is the doc you want.
