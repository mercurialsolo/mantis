"""Marketplace-listing extraction reward.

Terminal grader for tasks that ask the agent to extract a single
marketplace-listing record. The reward is a gate predicate:
  +1.0  when every required field parses out of the done() summary AND
        the recorded URL is on the configured allowlist AND any
        ground-truth constraints (min_price, zip, etc.) hold.
  -0.5  per off-site visit (carried from PlanAdherenceReward step rewards).
   0.0  when done(success=false) or DONE never fired.

Per-step shaping is reused from PlanAdherenceReward — same format / loop /
off-site signals.

Required fields parsed from ``done(summary=...)``:
  year   — 4-digit year (1900-2099)
  make   — non-empty token
  model  — non-empty token
  price  — looks like "$12,345" / "$12345" / "12500"
  url    — appears on one of the allowed domains

Ground truth (optional, passed into ``episode()``):
  {"min_price": 35000, "zip": "33101", "year_min": 2010}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...rewards.base import EpisodeState, RewardSignal
from ...rewards.components import oracle_step_reward
from ...rewards.plan_adherence import PlanAdherenceReward

if TYPE_CHECKING:
    from ...actions import Action
    from ...gym.base import GymResult
    from ...gym.runner import RunResult


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DOLLAR_PRICE_RE = re.compile(r"\$\s*([\d,]{3,})")
_LABELED_PRICE_RE = re.compile(r"price\s*[:=]\s*\$?\s*([\d,]{3,})", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
_LABELED_FIELD_RE = re.compile(
    r"\b(year|make|model|phone|type|url)\s*[:=]\s*([^,;\n]+)",
    re.IGNORECASE,
)


def _parse_summary(summary: str) -> dict[str, Any]:
    """Extract structured fields from a free-form done() summary.

    The agent is instructed to format extractions as
    "Year, Make, Model, Price, Phone, Type, URL" but in practice
    dumps a free-form sentence. Be lenient: regex out what we can and let
    presence/absence drive the gate.

    Two passes:
      1. Labeled fields ("Year: 2015, Make: Bayliner, ...") win.
      2. Fallback: positional "<year> <make> <model>" at start of text.

    Prices are sourced from $-prefixed numbers or "Price: ..." labels only —
    avoids picking up numeric IDs from URL slugs.
    """
    out: dict[str, Any] = {}

    # Strip URLs from a working copy before number extraction so URL slugs
    # like "...240-sundeck-9876543" don't masquerade as prices.
    url_match = _URL_RE.search(summary)
    if url_match:
        out["url"] = url_match.group(0).rstrip(".,;)")
    no_url = _URL_RE.sub(" ", summary)

    if m := _YEAR_RE.search(no_url):
        out["year"] = int(m.group(1))

    # Pass 1: labeled fields override everything else.
    for label, raw in _LABELED_FIELD_RE.findall(summary):
        key = label.lower()
        val = raw.strip().rstrip(".,;")
        if not val:
            continue
        if key == "year":
            if m := _YEAR_RE.search(val):
                out["year"] = int(m.group(1))
        elif key == "url":
            if m := _URL_RE.search(val):
                out["url"] = m.group(0).rstrip(".,;)")
        else:
            out[key] = val

    # Prices: dollar-prefixed first, then "Price: ..." label, then largest.
    prices: list[int] = []
    for m in _DOLLAR_PRICE_RE.finditer(no_url):
        try:
            prices.append(int(m.group(1).replace(",", "")))
        except ValueError:
            continue
    for m in _LABELED_PRICE_RE.finditer(summary):
        try:
            prices.append(int(m.group(1).replace(",", "")))
        except ValueError:
            continue
    if prices:
        out["price"] = max(prices)

    # Pass 2: positional "<year> <make> <model>" if make/model still missing.
    if "year" in out and ("make" not in out or "model" not in out):
        head = no_url.strip().split(".")[0]
        tokens = head.split()
        try:
            yi = tokens.index(str(out["year"]))
            if "make" not in out and yi + 1 < len(tokens):
                out["make"] = tokens[yi + 1].rstrip(",")
            if "model" not in out and yi + 2 < len(tokens):
                out["model"] = tokens[yi + 2].rstrip(",")
        except ValueError:
            pass

    return out


@dataclass
class MarketplaceListingReward(PlanAdherenceReward):
    """Plan adherence + marketplace-listing terminal gate.

    Inherits per-step shaping from PlanAdherenceReward. Overrides
    episode() to grade the done() summary against required fields and
    ground-truth constraints. Caller specifies ``allowed_domains`` for
    the target site.
    """

    allowed_domains: tuple[str, ...] = ()
    success_weight: float = 1.0
    required_fields: tuple[str, ...] = ("year", "make", "model", "price", "url")
    field_partial_credit: float = 0.0  # set >0 to reward partial extractions
    plan_progress_weight: float = 0.0  # turn off generic plan term; gate dominates

    def episode(
        self,
        *,
        run_result: "RunResult",
        state: EpisodeState,
        ground_truth: dict[str, Any] | None = None,
    ) -> RewardSignal:
        components: dict[str, float] = {}

        # Find the terminal done() — anywhere in the trajectory, not just last.
        done_step = None
        for tstep in reversed(run_result.trajectory):
            if tstep.action.action_type.value == "done":
                done_step = tstep
                break

        if done_step is None or not done_step.action.params.get("success", False):
            components["task_success"] = 0.0
            return RewardSignal(value=0.0, components=components)

        summary = str(done_step.action.params.get("summary", ""))
        record = _parse_summary(summary)

        # Required-field gate
        present = [f for f in self.required_fields if f in record]
        missing = [f for f in self.required_fields if f not in record]
        all_present = len(missing) == 0

        # URL must be on boattrader.com
        url_ok = "url" in record and any(
            d in record["url"].lower() for d in self.allowed_domains
        )

        # Ground-truth constraints
        constraints_ok = True
        if ground_truth:
            min_price = ground_truth.get("min_price")
            if min_price is not None and (record.get("price") or 0) < min_price:
                constraints_ok = False
            year_min = ground_truth.get("year_min")
            if year_min is not None and (record.get("year") or 0) < year_min:
                constraints_ok = False

        if all_present and url_ok and constraints_ok:
            components["gate_passed"] = self.success_weight
        elif self.field_partial_credit > 0:
            frac = len(present) / len(self.required_fields)
            components["gate_partial"] = self.field_partial_credit * frac
            if not url_ok:
                components["url_offsite"] = -self.field_partial_credit
        else:
            components["gate_failed"] = 0.0

        # Stash the parsed record for downstream logging.
        state.extras["listing_record"] = record
        state.extras["listing_missing_fields"] = missing
        state.extras["listing_url_ok"] = url_ok
        state.extras["listing_constraints_ok"] = constraints_ok

        return RewardSignal(value=sum(components.values()), components=components)


# ── Synthetic-env variant — uses sim-env oracle as ground truth ────────


# Default mapping for mantis-boattrader. Recipes targeting a different
# sim env override ``expected_ops_by_step_kind`` at construction time.
DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND: dict[str, frozenset[str]] = {
    # Filter application — the boattrader env stamps no dedicated
    # ``filter_applied`` mutation today (filters are read off the URL,
    # not stored). Empty set means oracle_step_reward returns 0 — fall
    # back to format/loop/off-site shaping for these.
    "filter": frozenset(),
    # Detail-page navigation — boattrader doesn't stamp a mutation for
    # plain GETs either. Same fallback as filters.
    "navigate": frozenset(),
    # Form submissions are the high-value mutations: the lead row
    # carries the boat_id + payload the terminal oracle grades on.
    "submit_lead": frozenset({"lead_submitted"}),
    # Pre-action gates the agent has to clear before extracting.
    "phone_reveal": frozenset({"phone_revealed"}),
    "consent": frozenset({"consent_set"}),
}


@dataclass
class SyntheticEnvReward(MarketplaceListingReward):
    """Marketplace-listing reward extended with sim-env oracle signal.

    Drop-in replacement for ``MarketplaceListingReward`` when training
    against a Mantis sim env (``mantis-boattrader``, etc.) that exposes
    ``/__env__/oracle`` and ``/__env__/mutations``. The reward picks up
    two extra signals the caller wires into the gym loop via
    ``gym_result.info``:

    * ``info["oracle_mutations_delta"]`` (``list[dict]``) — mutations
      recorded since the previous step. Used by :func:`oracle_step_reward`
      to award per-step credit for plan-aligned state changes (lead
      submitted, consent accepted, phone revealed). The caller fetches
      this via
      :func:`mantis_agent.sim_envs.oracle_client.fetch_mutations` after
      each step and stuffs the result on the info dict.

    * ``info["oracle_step_kind"]`` (``str``) — optional. When set,
      indexes :attr:`expected_ops_by_step_kind` to pick which operations
      count as progress for THIS step. Without it the reward only fires
      on the union of all expected ops (back-compat: if every step
      could plausibly land any mutation, every matching mutation
      counts).

    Terminal reward uses ``info["oracle_terminal"]`` (the
    ``GradingResult.to_dict()`` from
    :func:`mantis_agent.gym.grading.grade_run`) when present; F1 score
    plus a +1 bonus on pass. When the caller didn't run a terminal
    oracle call, falls back to the parent class's done-summary gate.

    This class does NOT call out to the env itself — the caller's loop
    populates the info dict. That keeps the reward fn deterministic
    given inputs (testable as a pure function) and lets the same
    fetcher feed multiple reward implementations.
    """

    oracle_step_weight: float = 0.1
    oracle_terminal_weight: float = 1.0
    expected_ops_by_step_kind: dict[str, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_MARKETPLACE_OPS_BY_STEP_KIND),
    )

    def step(
        self,
        *,
        action: "Action",
        gym_result: "GymResult",
        state: EpisodeState,
    ) -> RewardSignal:
        # Inherit format / loop / off-site shaping.
        base = super().step(action=action, gym_result=gym_result, state=state)

        info = gym_result.info or {}
        delta = info.get("oracle_mutations_delta") or []
        if not isinstance(delta, list) or not delta:
            return base

        # Pick the expected_ops scope. When the caller tagged the step
        # kind we use the matching set; otherwise the union of all kinds
        # acts as "any mutation we know about" — still rewards real
        # progress, just doesn't penalise the caller for not tagging.
        step_kind = str(info.get("oracle_step_kind") or "")
        if step_kind and step_kind in self.expected_ops_by_step_kind:
            expected = self.expected_ops_by_step_kind[step_kind]
        else:
            expected = frozenset().union(*self.expected_ops_by_step_kind.values())

        if not expected:
            return base

        bonus = oracle_step_reward(delta, expected, value=self.oracle_step_weight)
        if bonus == 0.0:
            return base

        merged = dict(base.components)
        merged["oracle_step"] = bonus
        # Track the cumulative oracle bonus on the state for log/debug.
        state.extras["oracle_step_total"] = (
            state.extras.get("oracle_step_total", 0.0) + bonus
        )
        return RewardSignal(value=base.value + bonus, components=merged)

    def episode(
        self,
        *,
        run_result: "RunResult",
        state: EpisodeState,
        ground_truth: dict[str, Any] | None = None,
    ) -> RewardSignal:
        # Start with the parent's done-summary gate as a fallback signal.
        base = super().episode(
            run_result=run_result, state=state, ground_truth=ground_truth,
        )

        oracle = state.extras.get("oracle_terminal")
        if not isinstance(oracle, dict):
            # No terminal oracle call attached — keep the parent reward.
            return base

        # The oracle is the canonical truth source for sim-env runs.
        # Replace the parent's gate term with the oracle's F1 score plus
        # a pass bonus, so the trainer optimises against ground truth
        # instead of the done() summary.
        try:
            score = float(oracle.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        passed = bool(oracle.get("passed", False))

        components = {k: v for k, v in base.components.items()
                      if k not in ("gate_passed", "gate_partial", "gate_failed")}
        components["oracle_score"] = self.oracle_terminal_weight * score
        if passed:
            components["oracle_passed_bonus"] = self.oracle_terminal_weight

        state.extras["oracle_passed"] = passed
        state.extras["oracle_score"] = score
        state.extras["oracle_reasons"] = oracle.get("reasons") or []

        return RewardSignal(value=sum(components.values()), components=components)
