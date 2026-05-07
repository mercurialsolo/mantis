"""ShadowRouter — deterministic A/B traffic split for shadow-deploys (#155 step 5).

The last leg of the continual-fine-tuning pipeline. After a candidate
model passes the eval-harness gate (step 4), it goes live to a small
percentage of production traffic alongside the baseline. The router
decides which variant a given request gets, in a way that's:

* **deterministic** — the same ``key`` (typically ``run_key`` or
  ``tenant_id``) always lands on the same variant. Lets a re-run of a
  flaky task still hit the same weights, and lets per-tenant testing
  pin one tenant to one variant for the whole evaluation window.
* **stable across processes** — uses ``hashlib.sha1`` for the bucket
  computation (Python's built-in ``hash()`` is salted between
  interpreters). A worker pool can route the same key the same way.
* **bounded** — ``candidate_pct`` is clamped to ``[0, 100]`` so a
  config typo can't accidentally promote 100% to the candidate.

Wiring
------

Operators construct one ``ShadowRouter`` per process (typically in the
runtime / FastAPI startup) and call ``route(key)`` per request to
decide which weights to serve. The chosen variant is stamped onto the
trace via ``runner.shadow_variant`` so the downstream labeller +
escalation-rate analytics can attribute outcomes per variant.

Disabled by default — when ``candidate_pct=0`` (the safe default),
``route`` always returns ``baseline``. Setting it to a small positive
value (5, 10, 25) flips the router into split-mode without the caller
needing a separate enable flag.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


# Variant labels used both as return values from ``route`` and as the
# ``variant`` tag stamped on trace files. Kept as constants so callers
# don't have to remember the spelling.
VARIANT_BASELINE: str = "baseline"
VARIANT_CANDIDATE: str = "candidate"


@dataclass
class ShadowRouter:
    """Deterministic A/B router.

    Args:
        candidate_pct: Percentage of requests that get the candidate
            variant. Clamped to ``[0, 100]``. ``0`` (default) disables
            the split entirely — every request gets ``baseline``.
        salt: Optional string mixed into the bucketing hash so two
            independent shadow rollouts on the same key space don't
            collide. Useful when running multiple candidate variants
            simultaneously: salt each router with the candidate's
            weights id and the buckets stay independent.
    """

    candidate_pct: float = 0.0
    salt: str = ""

    def __post_init__(self) -> None:
        # Clamp once at construction so each ``route`` call is cheap.
        if self.candidate_pct < 0.0:
            self.candidate_pct = 0.0
        elif self.candidate_pct > 100.0:
            self.candidate_pct = 100.0

    def route(self, key: str) -> str:
        """Return ``baseline`` or ``candidate`` for ``key``.

        Empty / falsy keys deterministically land on ``baseline`` —
        we'd rather every anonymous request stay on the safer variant
        than scatter across both buckets when the caller forgot to
        thread a key through.
        """
        if not key or self.candidate_pct <= 0.0:
            return VARIANT_BASELINE
        if self.candidate_pct >= 100.0:
            return VARIANT_CANDIDATE
        bucket = self._bucket(key)
        return VARIANT_CANDIDATE if bucket < self.candidate_pct else VARIANT_BASELINE

    def _bucket(self, key: str) -> float:
        """Hash ``key`` into ``[0, 100)``. Stable across processes."""
        digest = hashlib.sha1(f"{self.salt}:{key}".encode("utf-8")).digest()
        # First 4 bytes give us 32 bits of entropy — plenty for
        # percentage bucketing. Modulo by 10000 then divide by 100 to
        # keep the float resolution at 0.01 percentage points.
        n = int.from_bytes(digest[:4], "big") % 10000
        return n / 100.0


__all__ = [
    "ShadowRouter",
    "VARIANT_BASELINE",
    "VARIANT_CANDIDATE",
]
