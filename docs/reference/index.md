# Reference

| | What it covers |
|---|---|
| [HTTP API](../api.md) | Every endpoint, request body, response shape, error code |
| [Architecture](../architecture.md) | Internal architecture: brains, runner, env, gym, verification |
| [Environment variables](env-vars.md) | Server-side env knobs (caps, paths, log format, model routing) |
| [Glossary](glossary.md) | Quick definitions of project-specific terms |
| [Predicate grammar](predicates.md) | World-model verification predicates emitted by brains and evaluated by the runner |
| [Done-acceptance gate](done-gate.md) | Deterministic predicates the runner applies before accepting `done(success=True)` |
| [Form controller](form-controller.md) | Single object owning runtime form-filling state — pending values, used regions, submit latch, director hooks |
| [Adaptive settle](adaptive-settle.md) | Replaces fixed `time.sleep(settle_time)` with frame-stability / network-idle gates |
