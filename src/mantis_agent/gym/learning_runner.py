"""Learning Runner — builds verified playbooks from CUA exploration.

Runs the Learning Phase: executes a plan with full step-by-step
verification to build a site-specific Playbook. The Playbook is then
used by the Execution Phase (WorkflowRunner) for reliable, at-scale
extraction.

Architecture:
    Plan → LearningRunner (slow, thorough, verified)
      → Verified Playbook (JSON)
        → WorkflowRunner (fast, confident, at-scale)

Usage:
    verifier = StepVerifier()
    runner = LearningRunner(brain=brain, env=env, verifier=verifier)
    playbook = runner.learn(setup_intent, extract_intent, n_samples=5)
    PlaybookStore().save(playbook)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..verification.playbook import (
    ExtractionPattern,
    Playbook,
    PlaybookStep,
)
from ..verification.step_verifier import StepVerifier, VerificationResult
from .runner import GymRunner

logger = logging.getLogger(__name__)


@dataclass
class LearningResult:
    """Result of a learning attempt for one step."""
    step_name: str
    attempt: int
    verified: bool
    verification: VerificationResult
    action_taken: str
    duration: float


class LearningRunner:
    """Runs the Learning Phase with full step verification.

    For each step in the plan:
    1. Capture screenshot BEFORE
    2. Execute the step (bounded GymRunner)
    3. Capture screenshot AFTER
    4. StepVerifier checks: did it work?
    5. Record result in Playbook
    """

    def __init__(
        self,
        brain: Any,
        env: Any,
        verifier: StepVerifier,
        grounding: Any = None,
        on_step: Any = None,
        max_attempts_per_step: int = 3,
    ):
        self.brain = brain
        self.env = env
        self.verifier = verifier
        self.grounding = grounding
        self.on_step = on_step
        self.max_attempts = max_attempts_per_step

    def learn(
        self,
        setup_intent: str,
        extract_intent: str,
        domain: str = "",
        start_url: str = "",
        expected_filters: list[str] | None = None,
        n_samples: int = 5,
    ) -> Playbook:
        """Run full learning phase: setup + extraction samples.

        Args:
            setup_intent: The setup/filter task intent.
            extract_intent: The extraction task intent.
            domain: Target domain (e.g. "boattrader.com").
            start_url: Starting URL for the site.
            expected_filters: Filters to verify after setup.
            n_samples: Number of listings to extract for learning.

        Returns:
            Verified Playbook with setup steps, extraction pattern, known traps.
        """
        playbook = Playbook(domain=domain)
        expected_filters = expected_filters or []

        logger.info(f"=== LEARNING PHASE: {domain} ===")
        logger.info(f"  Setup intent: {setup_intent[:80]}...")
        logger.info(f"  Extract intent: {extract_intent[:80]}...")
        logger.info(f"  Samples: {n_samples}")

        # Phase 1: Learn setup (filter application)
        if setup_intent:
            logger.info("\n--- Phase 1: Learning setup/filters ---")
            self._learn_setup(playbook, setup_intent, start_url, expected_filters)

        # Phase 2: Learn extraction (listing processing)
        logger.info(f"\n--- Phase 2: Learning extraction ({n_samples} samples) ---")
        self._learn_extraction(playbook, extract_intent, n_samples)

        logger.info(f"\n=== LEARNING COMPLETE ===")
        logger.info(playbook.summary())

        return playbook

    def _learn_setup(
        self,
        playbook: Playbook,
        setup_intent: str,
        start_url: str,
        expected_filters: list[str],
    ):
        """Learn how to apply filters by executing setup with verification."""
        # Navigate to start
        if start_url:
            self.env.reset(start_url=start_url)
            time.sleep(3)

        # Take before screenshot
        before = self.env.screenshot()

        # Run setup task
        runner = GymRunner(
            brain=self.brain, env=self.env,
            max_steps=60, frames_per_inference=1,
            grounding=self.grounding,
            on_step=self.on_step,
        )
        t0 = time.time()
        result = runner.run(task=setup_intent, task_id="learn_setup", start_url=start_url)
        duration = time.time() - t0

        # Take after screenshot
        after = self.env.screenshot()

        # Verify: did setup work?
        step_verify = self.verifier.verify_step(
            before, after,
            intent="Apply search filters",
            action=f"Setup task ({result.total_steps} steps)",
        )

        # Also verify specific filters
        filter_verify = self.verifier.verify_filter(
            after, expected_filters, max_results=50000,
        )

        setup_step = PlaybookStep(
            name="setup_filters",
            intent=setup_intent[:200],
            expected_outcome=f"Filters applied: {', '.join(expected_filters)}",
            recovery_action=f"Navigate to {start_url}" if start_url else "Refresh page",
        )
        setup_step.update_confidence(filter_verify.verified)

        playbook.setup_steps.append(setup_step)

        if filter_verify.verified:
            logger.info(f"  Setup VERIFIED in {result.total_steps} steps ({duration:.0f}s)")
            logger.info(f"  Filters confirmed: {filter_verify.details[:100]}")
        else:
            logger.warning(f"  Setup FAILED verification: {filter_verify.issue}")
            logger.warning(f"  Suggestion: {filter_verify.suggestion}")
            playbook.known_traps.append(f"Setup filter failed: {filter_verify.issue}")

            # Recovery: try navigating to start_url
            if start_url:
                logger.info(f"  Recovery: navigating to {start_url}")
                self.env.reset(start_url=start_url)
                time.sleep(3)

    def _learn_extraction(
        self,
        playbook: Playbook,
        extract_intent: str,
        n_samples: int,
    ):
        """Learn extraction pattern by processing N listings with verification."""
        total_scrolls_to_desc = []
        phones_found = 0
        dealer_listings = 0
        traps_found = []

        for i in range(1, n_samples + 1):
            logger.info(f"\n  Sample {i}/{n_samples}")

            # Before: on search results page
            before = self.env.screenshot()

            # Verify we're on a results page
            page_check = self.verifier.verify_on_correct_page(
                before,
                expected_description="Search results page with boat listings",
                expected_signals=["listing", "boat", "price", "photo"],
            )
            if not page_check.verified:
                logger.warning(f"  Not on results page: {page_check.issue}")
                playbook.known_traps.append(f"Lost results page at iteration {i}: {page_check.issue}")
                if page_check.suggestion:
                    logger.info(f"  Recovery: {page_check.suggestion}")
                break

            # Run one extraction iteration
            ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(i, f"#{i}")
            iter_intent = extract_intent.replace("{ORDINAL}", ordinal).replace("{N}", str(i))

            runner = GymRunner(
                brain=self.brain, env=self.env,
                max_steps=40, frames_per_inference=2,
                grounding=self.grounding,
                on_step=self.on_step,
            )
            t0 = time.time()
            result = runner.run(task=iter_intent, task_id=f"learn_extract_{i}")
            duration = time.time() - t0

            # After: check what happened
            after = self.env.screenshot()

            # Verify the iteration
            verify = self.verifier.verify_step(
                before, after,
                intent=f"Extract data from listing #{i}",
                action=f"Extraction ({result.total_steps} steps)",
            )

            # Analyze the trajectory for patterns
            scroll_count = sum(
                1 for s in result.trajectory
                if s.action.action_type.value == "scroll"
            )
            total_scrolls_to_desc.append(scroll_count)

            # Check for phone in thinking/output
            all_text = " ".join(str(s.thinking or "") for s in result.trajectory)
            import re
            phone_found = bool(re.search(r'\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}', all_text))
            if phone_found:
                phones_found += 1

            # Check for dealer signals
            if any(sig in all_text.lower() for sig in ["more from this dealer", "view dealer website"]):
                dealer_listings += 1
                playbook.known_traps.append(f"Sample {i} was a dealer listing")

            # Check for traps
            if "gallery" in all_text.lower() or "1 of" in all_text.lower():
                traps_found.append("gallery_trap")
            if "popup" in all_text.lower() or "contact seller" in all_text.lower():
                traps_found.append("popup")

            # Record extraction step
            step = PlaybookStep(
                name=f"extract_listing_{i}",
                intent=f"Extract data from listing #{i}",
                expected_outcome="Year, Make, Model, Price, URL extracted",
            )
            step.update_confidence(verify.verified and result.success)
            playbook.extraction_steps.append(step)

            viable = result.success and verify.verified
            logger.info(f"  {'VIABLE' if viable else 'SKIP'} — {result.total_steps} steps, "
                        f"{scroll_count} scrolls, phone={'yes' if phone_found else 'no'}, "
                        f"verified={verify.verified} ({duration:.0f}s)")

            time.sleep(0.5)

        # Build extraction pattern from samples
        avg_scrolls = sum(total_scrolls_to_desc) / max(len(total_scrolls_to_desc), 1)
        playbook.extraction_pattern = ExtractionPattern(
            scrolls_to_description=round(avg_scrolls),
            scrolls_to_phone=round(avg_scrolls) + 1,
            has_visible_phone=phones_found > 0,
            phone_location="In Description section" if phones_found > 0 else "Not visible (hidden behind Contact form)",
            dealer_signals=["More From This Dealer", "View Dealer Website"],
        )
        playbook.listings_per_page = n_samples  # Will be refined with more data

        # Deduplicate traps
        playbook.known_traps = list(set(playbook.known_traps + traps_found))

        logger.info(f"\n  Extraction summary:")
        logger.info(f"    Samples: {n_samples}")
        logger.info(f"    Avg scrolls: {avg_scrolls:.0f}")
        logger.info(f"    Phones found: {phones_found}/{n_samples}")
        logger.info(f"    Dealer listings: {dealer_listings}/{n_samples}")
        logger.info(f"    Traps: {traps_found}")
