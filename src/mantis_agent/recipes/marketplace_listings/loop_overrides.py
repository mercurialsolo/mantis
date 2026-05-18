"""Loop-runner overrides for the marketplace_listings recipe.

Per issue #463 these constants were previously hardcoded inside
``gym/workflow_runner.py`` and ``gym/listings_scanner.py``, biasing the
generic loop runner toward BoatTrader-shaped flows. They now live with
the recipe so the generic runner stays neutral and only sees marketplace
heuristics when a plan explicitly opts in via the ``marketplace_listings``
recipe.

Each constant maps 1:1 to a field on ``gym.workflow_runner.LoopConfig``.
``recipes.load_loop_overrides("marketplace_listings")`` returns the
mapping below; callers (e.g. ``deploy/modal/modal_web_tasks_opencua.py``)
spread it into their ``LoopConfig`` constructor when the active recipe
is marketplace_listings.
"""

from __future__ import annotations

from ._data import DEALER_TEXT_INDICATORS

# Vertical entity name. Drives prompt copy that refers to "the X" / "the
# next X title". ``listings_scanner.scroll_directive_for`` falls back to
# "listing" if no override is provided, so this preserves the pre-#463
# wording byte-identically.
ENTITY_NAME = "listing"

# Failure-classification spam tokens. Superset of the pre-#463
# ``DEFAULT_DEALER_SIGNALS`` list that lived in workflow_runner — the
# recipe already maintained a richer ``DEALER_TEXT_INDICATORS`` set, so
# the loop runner now reuses that single source of truth.
SPAM_SIGNALS: tuple[str, ...] = DEALER_TEXT_INDICATORS

# External brand sites the agent should never follow off the marketplace.
# These are extracted from the pre-#463 ``external_sites`` list in
# ``WorkflowRunner._distill_learning``. Adding a new boat-brand domain
# here automatically gets picked up the next time loop overrides are
# loaded; no core change is needed.
OFFSITE_BRAND_DOMAINS: tuple[str, ...] = (
    "hanover yachts",
    "tige boats",
    "cobalt boats",
    "bayliner.com",
    "sea ray.com",
    "boston whaler.com",
    "grady-white.com",
)

# Viability-check fallback when the extractor schema doesn't expose
# entity keywords. Identical to the pre-#463 hardcoded list in
# ``WorkflowRunner._validate_viable``.
ENTITY_KEYWORDS: tuple[str, ...] = (
    "make", "model", "hull", "engine", "console", "cabin",
    "grady", "boston whaler", "sea hunt", "tracker", "yamaha",
    "mercury", "suzuki", "honda", "evinrude", "intrepid",
    "azimut", "sea ray", "sundeck", "walkaround", "sportfish",
    "bayliner", "chaparral", "everglades", "cigarette", "century",
    "cobia", "nor-tech", "may-craft", "key west", "robalo",
)

# Gallery-trap recovery copy. ``{entity_name}`` is interpolated by
# ``WorkflowRunner._distill_learning``. Marketplace listings always have
# a photo + name + price + 'View Details' surface; this copy is the
# pre-#463 verbatim wording, parameterised on the entity-name only.
GALLERY_TRAP_HINT = (
    "You clicked the PHOTO and got trapped in an image gallery. "
    "Press Escape then Alt+Left to go back. "
    "NEXT TIME: do NOT click the {entity_name} photo. Instead click one of these:\n"
    "  - The {entity_name} NAME TEXT (e.g. '2022 Grady-White Freedom 235')\n"
    "  - The PRICE text (e.g. '$145,000')\n"
    "  - A 'View Details', 'See Details', or 'Contact Seller' link\n"
    "These open the detail page WITHOUT the gallery trap."
)


# Loader contract — keys mirror LoopConfig field names.
LOOP_OVERRIDES: dict[str, object] = {
    "entity_name": ENTITY_NAME,
    "spam_signals": SPAM_SIGNALS,
    "offsite_brand_domains": OFFSITE_BRAND_DOMAINS,
    "entity_keywords": ENTITY_KEYWORDS,
    "gallery_trap_hint": GALLERY_TRAP_HINT,
}
