"""marketplace_listings recipe — vehicle / boat / RV listing extraction.

Owns its own dealer-vs-private indicators, field schema, and reward.
Imported lazily by the legacy ``ExtractionSchema.default_boattrader()``
shim and by callers that resolve the recipe by name.
"""
