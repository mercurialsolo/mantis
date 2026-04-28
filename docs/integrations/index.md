# Integrations

Specific runbooks for plugging Mantis into other systems.

| | What it covers |
|---|---|
| [vision_claude](../integration-vision_claude.md) | Replacing Anthropic Computer Use with Mantis as the perception+reasoning backend in StaffAI's vision_claude agent. Uses the orchestrator-only path: vision_claude keeps Xvfb + Chrome + the mounted profile; Mantis exposes Holo3 inference over `/v1/chat/completions`. |

If you're integrating from a fresh codebase, the [Client](../client/index.md) section is the right starting point. Use the integration runbooks in this section for established codebases that already have their own desktop / browser / agent loop.
