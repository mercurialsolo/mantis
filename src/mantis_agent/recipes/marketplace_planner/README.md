# marketplace_planner (legacy)

A CLI tool that takes a plain-text BoatTrader-style listings plan and
produces a Claude-Opus-shaped micro-plan + browse summaries. The
system prompts and planning rules bake in marketplace assumptions
(listing cards, dealer / sponsor rejection, phone extraction, Private
Seller filtering, contact-area / seller-notes scrolling).

Per issue #462 the module is recipe-scoped rather than retired —
it has no importers in the core but the CLI is preserved as a
documented reference for the BoatTrader planning patterns. New
work should use the generic planning surfaces instead:

- ``--micro`` (`mantis_agent.plan_decomposer.PlanDecomposer`) for
  objective → micro-plan decomposition (works on any vertical).
- ``--graph-learn`` (`mantis_agent.graph.learner.GraphLearner`) for
  probe-driven plan enhancement with recipe overlays.

## Usage

```bash
python -m mantis_agent.recipes.marketplace_planner.planner <plan.txt>
python -m mantis_agent.recipes.marketplace_planner.planner <plan.txt> --browse <screenshots_dir>
```

## Status

| | |
|---|---|
| On default path | No (CLI-only, no Python importers) |
| Vertical | marketplace_listings (BoatTrader / vehicle / RV / boat) |
| Recommended replacement | `--micro` + `marketplace_listings` recipe overlay |
| Sites tested | BoatTrader |
