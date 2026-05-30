"""Mantis cost-optimization knob bundle — autoresearch trial config.

THIS IS THE ONE FILE THE AGENT EDITS PER TRIAL (along with optional edits
to plans/boattrader_scrape, src/mantis_agent/plan_decomposer.py,
src/mantis_agent/grounding.py, and src/mantis_agent/extraction/extractor.py).

Each top-level Modal submission of submit_one_trial.py reads CONFIG and
threads its values through to the task suite + runtime parameters.

Knobs that DO flow through this file (runtime-side, no redeploy needed):

- ``extractor_model``: Anthropic model id for ClaudeExtractor (`_call`,
  `_call_many`, `_call_with_tool_schema*`). Surfaces in cost meter as
  ``claude_extract`` / ``extract_single`` / ``extract_multi``. Haiku 4.5
  is currently the cheapest production-quality option.
- ``fanout_phase1_workers``: int 1-8. Spawns N parallel Modal containers
  for the URL-collect phase. Higher = lower wall-time, similar Claude
  spend.
- ``max_cost``: float dollars — circuit-breaker budget per run (covers
  Claude API + estimated proxy cost). Modal halts the run when this is
  exceeded.
- ``max_time_minutes``: int — wall-time cap.
- ``wait_after_load_seconds``: int — settle time after navigate steps.
  Affects whether the gate verifier sees a fully-loaded page (too short
  → verifier sees "Loading...", fails; too long → wasted seconds × N
  pages).
- ``plan_path``: relative path to the plan source. Default
  ``plans/boattrader_scrape``. Change to test a forked plan variant.

Knobs that need code edits + redeploy (NOT in this dict):

- ``grounding_model`` — edit ``src/mantis_agent/grounding.py``
  ``ClaudeGrounding.__init__`` default ``model=`` argument. Redeploy.
- ``verifier_escalation`` — edit
  ``src/mantis_agent/extraction/extractor.py``
  ``_VERIFY_ESCALATION_MODEL`` default. Redeploy.

Edit CONFIG below. Then ``git commit`` and run the trial.
"""

CONFIG: dict = {
    # Extractor model — the cheapest piece of the budget today
    # (~$0.04 / run total at Haiku). Sonnet would 5× this — only switch
    # if you have evidence Haiku is dropping fields (mostly it doesn't).
    "extractor_model":          "claude-haiku-4-5-20251001",

    # Phase-1 URL-collect parallel workers. 4 was the production
    # default. 1 = sequential (slow but cheap). 8 = aggressive parallel
    # (eats budget faster, more proxy cost).
    "fanout_phase1_workers":    4,

    # Per-run circuit-breaker. Claude + proxy combined. The autoresearch
    # budget is $10 across ALL trials — this is the per-trial ceiling.
    "max_cost":                 1.0,

    # Wall-time cap. Mantis runs at ~25 min for a successful boattrader
    # extraction; bump to 45 if you're pushing deeper pagination.
    "max_time_minutes":         30,

    # Settle time after navigate. 4 = default. Bump to 8 if the gate
    # verifier flakes on "Loading..." state.
    "wait_after_load_seconds":  4,

    # Path to the plan file. Change to point at a variant if you want
    # to A/B two plan shapes without overwriting the canonical one.
    "plan_path":                "plans/boattrader_scrape",
}
