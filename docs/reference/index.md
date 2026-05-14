# Reference

| | What it covers |
|---|---|
| [HTTP API](../api.md) | Every endpoint, request body, response shape, error code |
| [Architecture](../architecture.md) | Internal architecture: brains, runner, env, gym, verification |
| [CUA models](cua-models.md) | Available `cua_model` backends (Holo3, Fara, EvoCUA, OpenCUA, Gemma4-CUA, Claude) with action-space caveats |
| [Environment variables](env-vars.md) | Server-side env knobs (caps, paths, log format, model routing) |
| [Glossary](glossary.md) | Quick definitions of project-specific terms |
| [Predicate grammar](predicates.md) | World-model verification predicates emitted by brains and evaluated by the runner |
| [Done-acceptance gate](done-gate.md) | Deterministic predicates the runner applies before accepting `done(success=True)` |
| [Form controller](form-controller.md) | Single object owning runtime form-filling state — pending values, used regions, submit latch, director hooks |
| [Adaptive settle](adaptive-settle.md) | Replaces fixed `time.sleep(settle_time)` with frame-stability / network-idle gates |
| [Chrome session reuse](chrome-session-reuse.md) | Container-scoped cache that reuses live Xvfb + Chrome across `/v1/cua` requests |
| [Speculative inference](speculative-inference.md) | Overlaps `brain.think()` with the post-action settle to remove the serial inference cost |
| [Perceptual diff verifier](perceptual-diff.md) | Detects silent failures on high-risk actions by comparing pre/post frame hashes |
| [Loop recovery policy](loop-recovery.md) | Forces action-class transitions (Tab / Return / Type) when the brain loops on a no-effect class |
| [Adaptive loop windows](adaptive-loop-windows.md) | Per-call adjustment of the soft/hard loop-detection windows by recent action diversity + state progress |
| [Adaptive click tolerance](adaptive-click-tol.md) | Drift-loop tolerance scaled by screen DPI and per-action element class (button / link / dropdown) |
| [Ablation harness](ablation-harness.md) | Single-deploy paired ON/OFF A/B against `/v1/cua`; required for quality-touching PRs |
