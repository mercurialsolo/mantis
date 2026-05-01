# Recipe: marketplace_listings

Walks a vehicle-marketplace search-results page, opens each listing, and
extracts a structured row per listing (year / make / model / price / phone /
seller / url) plus a spam classification (private vs dealer).

## Status

**Reference recipe.** Today this recipe re-exports the legacy
`ExtractionSchema.default_boattrader()` from the core so the existing run
paths keep working. The migration path is:

1. Now: this recipe is a thin re-export. Core still owns the schema constants.
2. Next: invert the dependency — move `DEALER_TEXT_INDICATORS`,
   `DEALER_SELLER_INDICATORS`, and `_boattrader_fields()` here, and have
   `extraction.ExtractionSchema.default_boattrader()` load them from this
   recipe.
3. Eventually: remove `default_boattrader()` from the core entirely; callers
   must declare a recipe explicitly.

## Tested against

- BoatTrader.com search-results pages (private-seller filter applied)
- Generic enough that it has handled comparable RV / motorcycle marketplaces
  with no schema change

## Known limits

- Phone numbers behind multi-step "Show phone" interactions occasionally fail
  on the second loop iteration; the verifier flags these but does not retry
- Spam classification is rule-based (`DEALER_TEXT_INDICATORS`) — no
  ML-trained classifier
- Required fields are `year` and `make`; listings that omit either are
  dropped as malformed rather than partial-rescued

## Usage

```python
from recipes.marketplace_listings.schema import SCHEMA
from mantis_agent.extraction import ClaudeExtractor

extractor = ClaudeExtractor(schema=SCHEMA)
```

Or pass `recipe="marketplace_listings"` in your plan once the dispatcher is
wired (TODO — see issue tracker).
