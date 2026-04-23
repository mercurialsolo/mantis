"""WorkflowRunner — dynamic loop execution with pagination.

Instead of unrolling loops into N fixed sections, WorkflowRunner
iterates dynamically: process items until none remain, then paginate.

Each iteration is a bounded GymRunner.run() call with independent
retry support. The model signals "no more items" or "no more pages"
via its terminate() message to control loop flow.

Usage:
    from mantis_agent.gym.workflow_runner import WorkflowRunner, LoopConfig

    config = LoopConfig(
        iteration_intent="Process the {ORDINAL} listing: ...",
        pagination_intent="Click the Next page button",
    )
    runner = WorkflowRunner(brain=brain, env=env, loop_config=config)
    results = runner.run_loop()
"""

from __future__ import annotations

import logging
import random
import re as _re_module
import time
from dataclasses import dataclass
from typing import Any

from ..actions import ActionType
from .runner import GymRunner

logger = logging.getLogger(__name__)


def _extract_url_latest(trajectory: list, year: str = "") -> str | None:
    """Extract the most recent listing URL from a trajectory.

    Searches steps in REVERSE order (latest first) to get the URL
    for the current listing, not a previous one. If `year` is provided,
    validates the URL slug contains the same year to avoid cross-listing
    contamination (e.g., thinking about "2008 Intrepid" while on a
    "2026 Tracker" detail page).
    """
    best_url = None
    for step in reversed(trajectory):
        for text in [str(step.thinking or ""),
                     str(step.action.params.get("summary", "")) if step.action.action_type.value == "done" else ""]:
            url = _extract_url_from_text(text)
            if not url:
                continue
            # If we have the boat's year, check the URL slug starts with it
            if year and len(year) == 4:
                slug = url.split("/boat/")[-1] if "/boat/" in url else ""
                if slug.startswith(year):
                    return url  # Year-matched URL — high confidence
                # URL year mismatch — keep looking but save as fallback
                if best_url is None:
                    best_url = url
                continue
            return url  # No year to validate against — take first found
    return best_url


def _extract_url_from_text(text: str) -> str | None:
    """Extract a BoatTrader listing URL from text.

    Matches patterns like:
    - boattrader.com/boat/2018-everglades-355-cc-1234567/
    - boattrader.com/boats/2018-everglades-355/
    - www.boattrader.com/boat/...
    - https://www.boattrader.com/boat/...

    Returns the URL without protocol prefix, or None if not found.
    """
    # Try full URL first (with or without protocol)
    match = _re_module.search(
        r"(?:https?://)?(?:www\.)?boattrader\.com/boats?/[\w\-]+(?:/[\w\-]*)*/?",
        text,
    )
    if match:
        url = match.group()
        # Strip protocol and www prefix for consistency
        url = _re_module.sub(r"^https?://(?:www\.)?", "", url)
        # Don't return the base listings URL (that's not a specific listing)
        if url.rstrip("/") in ("boattrader.com/boats", "boattrader.com/boat"):
            return None
        # Must have at least a slug after /boat(s)/
        slug = _re_module.search(r"/boats?/([\w\-]+)", url)
        if slug and len(slug.group(1)) > 5:
            return url
    return None

ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
}

NO_MORE_SIGNALS = [
    "no more", "no listing", "no items", "no results",
    "empty page", "end of", "last page", "no next",
    "all processed", "none found", "none left",
]

# Signals that the model has scrolled past all listings (page exhaustion)
PAGE_EXHAUSTED_SIGNALS = [
    "footer", "bottom of the page", "scrolled past all",
    "no more listings", "can't find", "cannot find",
    "powered by", "boat loan calculator", "become a member",
    "haven't found", "still haven't found",
]


@dataclass
class LoopConfig:
    """Configuration for a dynamic loop."""
    iteration_intent: str       # Template with {ORDINAL}, {N} placeholders
    pagination_intent: str = "Scroll to the bottom and click the Next page button or '>' arrow. If there is no next page, call terminate('failure') with 'no more pages'."
    max_iterations: int = 50    # Safety cap
    max_pages: int = 10
    max_steps_per_iteration: int = 60
    max_retries_per_iteration: int = 2
    max_steps_pagination: int = 20


class FailureCategory:
    """Structured failure categories for analysis and learning."""
    VIABLE = "viable"
    POPUP_TRAP = "popup_trap"           # Modal/popup blocked interaction
    OFF_SITE = "off_site"               # Navigated to external domain
    DEAD_LINK = "dead_link_404"         # Listing removed / 404
    GALLERY_TRAP = "gallery_trap"       # Clicked photo → fullscreen gallery
    PAGE_EXHAUSTED = "page_exhausted"   # No more listings on page
    CONTEXT_ROT = "context_rot"         # Model lost track (blank page, wrong page)
    CLOUDFLARE = "cloudflare"           # Bot detection blocked
    DEALER_LISTING = "dealer_listing"   # Clicked dealer listing, not private seller
    PARSE_FAILURE = "parse_failure"     # Model output unparseable
    UNKNOWN = "unknown"


def _classify_failure(data: str, result=None) -> tuple[str, str]:
    """Classify a failed iteration into category + reason.

    Returns (category, reason) tuple for structured logging.
    """
    text = (data or "").lower()

    # Dealer listing detection (highest priority — wrong listing type entirely)
    dealer_signals = ["more from this dealer", "view dealer website", "dealer listing",
                      "visit seller website", "more boats from this dealer"]
    if any(sig in text for sig in dealer_signals):
        return FailureCategory.DEALER_LISTING, "Clicked dealer listing instead of private seller"

    if "popup" in text or "modal" in text or "contact seller" in text or "request info" in text:
        return FailureCategory.POPUP_TRAP, "Modal/popup blocked interaction"

    if "facebook" in text or "instagram" in text or "off-site" in text or "dealer website" in text:
        domain = ""
        for d in ["facebook.com", "instagram.com", "elitemarine", "dealer"]:
            if d in text:
                domain = d
                break
        return FailureCategory.OFF_SITE, f"Navigated to external site ({domain})"

    if "404" in text or "page not found" in text or "page error" in text or "listing was removed" in text:
        return FailureCategory.DEAD_LINK, "Listing returned 404 or was removed"

    if "gallery" in text or "1 of" in text or "fullscreen" in text or "lightbox" in text:
        return FailureCategory.GALLERY_TRAP, "Clicked photo, entered image gallery"

    for sig in PAGE_EXHAUSTED_SIGNALS:
        if sig in text:
            return FailureCategory.PAGE_EXHAUSTED, "Scrolled to footer, no more listings"

    if "cloudflare" in text or "verify you are human" in text:
        return FailureCategory.CLOUDFLARE, "Bot detection blocked access"

    if "blank" in text or "about:blank" in text or "navigation menu" in text or "homepage" in text:
        return FailureCategory.CONTEXT_ROT, "Model lost track of page state"

    if result and hasattr(result, 'total_steps'):
        pf = getattr(result, 'parse_failures', 0) or 0
        if pf > result.total_steps * 0.5:
            return FailureCategory.PARSE_FAILURE, f"{pf}/{result.total_steps} steps unparseable"

    return FailureCategory.UNKNOWN, "Unclassified failure"


class LearningStore:
    """Persist and load site-specific learnings across runs.

    Saves learnings to /data/learnings/<domain>.json on Modal volume.
    Each learning has: rule, category, count (how many times observed).
    """

    def __init__(self, domain: str = "", base_path: str = "/data/learnings"):
        self.domain = domain
        self.path = f"{base_path}/{domain.replace('.', '_')}.json" if domain else ""
        self.learnings: list[dict] = []
        self._load()

    def _load(self):
        """Load existing learnings from disk."""
        if not self.path:
            return
        try:
            import json
            with open(self.path) as f:
                self.learnings = json.load(f)
            logger.info(f"Loaded {len(self.learnings)} learnings for {self.domain}")
        except (FileNotFoundError, Exception):
            self.learnings = []

    def save(self):
        """Persist learnings to disk."""
        if not self.path:
            return
        try:
            import json
            import os
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.learnings, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save learnings: {e}")

    def add(self, rule: str, category: str):
        """Add or reinforce a learning. Deduplicates by rule text."""
        for existing in self.learnings:
            if existing["rule"] == rule:
                existing["count"] += 1
                return
        self.learnings.append({"rule": rule, "category": category, "count": 1})

    def get_top(self, n: int = 5) -> list[str]:
        """Get top N learnings by frequency, for prompt injection."""
        sorted_l = sorted(self.learnings, key=lambda x: x["count"], reverse=True)
        return [learning["rule"] for learning in sorted_l[:n]]


@dataclass
class IterationResult:
    """Result of a single loop iteration."""
    iteration: int
    page: int
    success: bool
    data: str               # Model's terminate summary (extracted data)
    no_more_items: bool     # Model signaled end of items on page
    steps: int
    duration: float
    parse_failures: int = 0  # Steps where action couldn't be parsed (WAIT fallback)
    failure_category: str = ""   # Structured failure category
    failure_reason: str = ""     # Human-readable failure reason


class WorkflowRunner:
    """Drives dynamic loops and pagination over GymRunner.

    Args:
        brain: Brain backend (any with think() method).
        env: GymEnvironment (PlaywrightGymEnv etc).
        loop_config: Loop parameters.
        session_name: For session persistence.
    """

    def __init__(self, brain: Any, env: Any, loop_config: LoopConfig,
                 session_name: str = "workflow",
                 on_iteration: Any = None,
                 on_trajectory: Any = None,
                 start_url: str | None = None,
                 grounding: Any = None,
                 on_step: Any = None,
                 use_sub_plan: bool = False,
                 extractor: Any = None):
        self.brain = brain
        self.grounding = grounding
        self.on_step = on_step
        self.use_sub_plan = use_sub_plan
        self.extractor = extractor  # ClaudeExtractor for screenshot-based extraction
        # Dedup: track processed phone numbers and listing URLs
        self._seen_phones: set[str] = set()
        self._seen_urls: set[str] = set()
        self.env = env
        self.config = loop_config
        self.session_name = session_name
        self.on_iteration = on_iteration
        self.on_trajectory = on_trajectory
        self.start_url = start_url

        # Cross-run learning store (persists to /data/learnings/)
        domain = ""
        # Try start_url first, fall back to iteration_intent for domain hints
        for url_source in [start_url, loop_config.iteration_intent]:
            if url_source:
                import re as _r
                m = _r.search(r"(?:https?://)?(?:www\.)?([\w\-]+\.[\w]+)", url_source)
                if m:
                    domain = m.group(1)
                    break
        self._learning_store = LearningStore(domain=domain)

        # Sub-plan runner for micro-step decomposition
        self._sub_plan_runner = None
        if use_sub_plan:
            from .sub_plan import SubPlanRunner
            self._sub_plan_runner = SubPlanRunner(
                brain=brain, env=env, grounding=grounding,
                extractor=extractor, on_step=on_step,
            )

    def run_loop(self) -> list[IterationResult]:
        """Execute the full loop: iterate items on each page, paginate."""
        results: list[IterationResult] = []
        page = 1
        global_iteration = 0
        page_iteration = 0
        learnings: list[str] = []  # Agentic: accumulate learnings from prior iterations

        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5  # Force-paginate after this many failures in a row

        logger.info(f"Starting dynamic loop (max {self.config.max_iterations} items, {self.config.max_pages} pages)")

        while page <= self.config.max_pages and global_iteration < self.config.max_iterations:
            # Ordinal is tentative — only advance after confirmed progress
            tentative_ordinal = page_iteration + 1

            # Build iteration intent with ordinal
            ordinal = ORDINALS.get(tentative_ordinal, f"#{tentative_ordinal}")
            global_iteration += 1

            intent = self.config.iteration_intent
            intent = intent.replace("{ORDINAL}", ordinal)
            intent = intent.replace("{N}", str(global_iteration))
            intent = intent.replace("{PAGE_N}", str(tentative_ordinal))
            intent = intent.replace("{PAGE}", str(page))

            # Add context about position + already-seen URLs for dedup
            if page_iteration > 0:
                intent += (
                    f"\n\nYou are on page {page}. You have already processed {page_iteration} listings on this page. "
                    f"Find the NEXT unprocessed listing below the ones you already handled. "
                    f"If you reach the page footer (copyright, 'Become a Member', etc.) without finding another listing, "
                    f"call done(success=false, summary='no more listings on this page') so we can go to the next page."
                )
            if self._seen_urls:
                # Tell the model which listings to skip (last 5 URLs for context)
                recent = list(self._seen_urls)[-5:]
                slugs = [u.split("/boat/")[-1][:30] if "/boat/" in u else u[:30] for u in recent]
                intent += f"\n\nALREADY EXTRACTED (skip these): {', '.join(slugs)}"

            # Dynamic hints based on consecutive failures
            if consecutive_failures >= 3:
                intent += (
                    f"\n\nURGENT: You have FAILED {consecutive_failures} times in a row on this page. "
                    f"You are clicking the WRONG element — likely an ad, dealer link, or broken external link. "
                    f"CHANGE YOUR APPROACH: scroll DOWN past the elements you've been clicking "
                    f"and find a DIFFERENT listing card further down the page. "
                    f"If you cannot find any more listings, scroll to the bottom and click the 'Next' page button."
                )
            elif consecutive_failures >= 2:
                intent += (
                    f"\n\nWARNING: Your last {consecutive_failures} attempts failed (404 or off-site). "
                    f"You may be clicking the same broken link repeatedly. "
                    f"Try a DIFFERENT listing — scroll down to find one you haven't tried yet."
                )

            # Agentic learning: inject what worked/failed in prior iterations
            all_tips = []
            # Persistent cross-run learnings (site-specific)
            all_tips.extend(self._learning_store.get_top(3))
            # In-session learnings (this run)
            all_tips.extend(learnings[-5:])
            if all_tips:
                intent += "\n\nLEARNINGS (apply these):"
                for tip in all_tips:
                    intent += f"\n- {tip}"

            logger.info(f"Iteration {global_iteration} (page {page}, item {tentative_ordinal})")
            t0 = time.time()

            # ── Pre-iteration recovery protocol ──
            if global_iteration > 1:
                time.sleep(0.5)
                try:
                    from ..actions import Action
                    # Dismiss any popup/modal (Escape is harmless if none)
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                    time.sleep(0.3)
                    # Scroll to top — prevents "stuck on footer" after page navigation
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                    time.sleep(0.5)
                except Exception:
                    pass

            # Escalating recovery after consecutive failures
            if consecutive_failures >= 3:
                # Try closing any rogue tabs first (Ctrl+W closes current tab)
                try:
                    from ..actions import Action
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "ctrl+w"}))
                    time.sleep(1.0)
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Escape"}))
                    time.sleep(0.5)
                except Exception:
                    pass

                # Navigate back — use checkpointed page URL if available
                base_url = self.start_url or getattr(self.env, '_start_url', '')
                # Build page-specific recovery URL (append page param if past page 1)
                recovery_url = base_url
                if base_url and page > 1:
                    sep = "&" if "?" in base_url else "?"
                    recovery_url = f"{base_url}{sep}page={page}"

                if recovery_url:
                    logger.warning(f"  Recovery: {consecutive_failures} failures — navigating to {recovery_url} (page {page})")
                    try:
                        self.env.reset(task="recovery", start_url=recovery_url)
                        time.sleep(3.0)
                        # Keep page_iteration — we're resuming on the same page
                        # The dedup will skip already-extracted URLs instantly
                        logger.info(f"  Resumed on page {page}, {len(self._seen_urls)} URLs already seen (will dedup)")
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                        time.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"  Recovery navigation failed: {e}")
                else:
                    # No URL to recover to — try Alt+Left repeatedly
                    logger.warning(f"  Recovery: {consecutive_failures} failures — pressing Alt+Left 3x")
                    try:
                        from ..actions import Action
                        for _ in range(3):
                            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "alt+Left"}))
                            time.sleep(1.0)
                        self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                        time.sleep(0.5)
                    except Exception:
                        pass

            # Run bounded iteration — sub-plan or monolithic
            if self._sub_plan_runner:
                # Micro-step path: FIND→CLICK→READ_URL→SCROLL→EXTRACT→RETURN
                sub_result = self._sub_plan_runner.run_listing(
                    iteration=global_iteration,
                    page=page,
                    page_iteration=tentative_ordinal,
                )
                extracted = sub_result.data
                viable = sub_result.success
                parse_failures = 0

                # Handle dedup — URL already seen
                if sub_result.dedup:
                    logger.info(f"  [dedup] Skipped: {sub_result.url[:60]}")
                    # Count as progress (listing exists, just already extracted)
                    page_iteration = tentative_ordinal
                    consecutive_failures = 0
                    iter_result = IterationResult(
                        iteration=global_iteration, page=page,
                        success=False, data=f"DUPLICATE | URL: {sub_result.url}",
                        no_more_items=False, steps=sub_result.steps,
                        duration=time.time() - t0,
                    )
                    results.append(iter_result)
                    if self.on_iteration:
                        try:
                            self.on_iteration(global_iteration, iter_result, results)
                        except Exception:
                            pass
                    continue

                # Handle page exhaustion
                if sub_result.page_exhausted:
                    logger.info("  Page exhausted — paginating")
                    paginated = self._sub_plan_runner.run_pagination()
                    if paginated:
                        page += 1
                        page_iteration = 0
                        consecutive_failures = 0
                        logger.info(f"  Paginated to page {page}")
                    else:
                        logger.info("  No more pages")
                        break
                    continue

                # Build minimal RunResult for compatibility
                from .runner import RunResult
                result = RunResult(
                    task=intent, task_id=f"iter_{global_iteration}",
                    success=sub_result.success,
                    total_reward=0.0,
                    total_steps=sub_result.steps,
                    total_time=sub_result.duration,
                    trajectory=[],
                    termination_reason="done" if sub_result.success else "sub_plan_skip",
                )

                logger.info(f"  Sub-plan: {sub_result.steps} steps, viable={viable}, url={sub_result.url[:40] if sub_result.url else '?'}")
            else:
                # Legacy monolithic path
                result = self._run_iteration(intent, f"iter_{global_iteration}")

                # Claude extraction: use screenshot instead of thinking text
                if self.extractor and hasattr(self.env, 'screenshot'):
                    try:
                        top_shot = self.env.screenshot()
                        ext_result = self.extractor.extract(top_shot)

                        if ext_result.is_viable():
                            # Early dedup: check URL before full extraction
                            ext_url = ext_result.url
                            if ext_url and ext_url in self._seen_urls:
                                logger.info(f"  [dedup] URL already seen: {ext_url[:60]} — skipping")
                                extracted = f"DUPLICATE | URL: {ext_url}"
                                parse_failures = self._count_parse_failures(result)
                                # Skip to next iteration
                                iter_result = IterationResult(
                                    iteration=global_iteration, page=page,
                                    success=False, data=extracted,
                                    no_more_items=False, steps=result.total_steps,
                                    duration=time.time() - t0, parse_failures=0,
                                )
                                results.append(iter_result)
                                # Count as progress (we did find a listing, just a dup)
                                page_iteration = tentative_ordinal
                                consecutive_failures = 0
                                if self.on_iteration:
                                    try:
                                        self.on_iteration(global_iteration, iter_result, results)
                                    except Exception:
                                        pass
                                continue

                            extracted = ext_result.to_summary()
                            logger.info(f"  [claude-extract] {extracted[:100]}")

                            # Track URL for future dedup
                            if ext_url:
                                self._seen_urls.add(ext_url)
                        else:
                            extracted = self._extract_data(result)
                    except Exception as e:
                        logger.warning(f"  Claude extraction failed: {e}")
                        extracted = self._extract_data(result)
                else:
                    extracted = self._extract_data(result)

                parse_failures = self._count_parse_failures(result)

            # Extract listing URL for dedup before validation (monolithic path)
            if not self._sub_plan_runner:
                _listing_url = _extract_url_from_text(extracted)
                viable = self._validate_viable(extracted, listing_url=_listing_url)

            # Classify failure for structured logging
            fail_cat, fail_reason = "", ""
            if not viable:
                fail_cat, fail_reason = _classify_failure(extracted, result)

            iter_result = IterationResult(
                iteration=global_iteration,
                page=page,
                success=viable,
                data=extracted,
                no_more_items=self._check_no_more(result),
                steps=result.total_steps,
                duration=time.time() - t0,
                parse_failures=parse_failures,
                failure_category=fail_cat,
                failure_reason=fail_reason,
            )
            results.append(iter_result)

            # Save trajectory for distillation (if callback provided)
            if self.on_trajectory:
                try:
                    self.on_trajectory(global_iteration, result)
                except Exception as e:
                    logger.warning(f"  Trajectory save failed: {e}")

            # Gallery retry: if this iteration hit a gallery trap, retry the SAME listing
            # with explicit instruction to click title text, not photo
            extracted_lower = extracted.lower() if extracted else ""
            if not viable and ("gallery" in extracted_lower or "1 of" in extracted_lower):
                logger.info("  Gallery trap detected — retrying same listing")
                gallery_intent = intent + (
                    "\n\nCRITICAL: Your previous attempt opened a photo gallery by clicking the boat PHOTO. "
                    "This time, click the TITLE TEXT (Year Make Model text) which is BELOW the photo. "
                    "The title text is smaller and appears under the large photo image. "
                    "Do NOT click anywhere on the photo area."
                )
                retry_result = self._run_iteration(gallery_intent, f"iter_{global_iteration}_retry")
                retry_extracted = self._extract_data(retry_result)
                retry_viable = self._validate_viable(retry_extracted)
                if retry_viable:
                    logger.info("  Gallery retry SUCCEEDED")
                    iter_result = IterationResult(
                        iteration=global_iteration, page=page,
                        success=True, data=retry_extracted,
                        no_more_items=False, steps=retry_result.total_steps,
                        duration=time.time() - t0, parse_failures=0,
                    )
                    results[-1] = iter_result  # Replace the failed iteration
                    viable = True
                    extracted = retry_extracted
                else:
                    logger.info("  Gallery retry also failed")

            if result.success and not viable:
                logger.warning("  Model claimed success but data failed validation (Cloudflare/empty)")
                logger.warning(f"  Extracted data was: {extracted[:300]}")

            # Mostly-failed iterations = model couldn't act, don't count as real progress
            if parse_failures > 0 and result.total_steps > 0:
                fail_ratio = parse_failures / result.total_steps
                if fail_ratio > 0.5:
                    logger.warning(f"  {parse_failures}/{result.total_steps} steps had parse failures — model is stuck")

            # Agentic learning: distill what happened for future iterations
            learning = self._distill_learning(result, iter_result, viable)
            if learning:
                learnings.append(learning)
                logger.info(f"  Learning: {learning}")
                # Persist to cross-run store for non-trivial failures
                if fail_cat and fail_cat != FailureCategory.UNKNOWN:
                    self._learning_store.add(learning, fail_cat)

            # Advance ordinal only when the iteration produced real progress:
            # viable data extracted, or explicit skip with actual listing data
            iteration_had_progress = viable or (iter_result.data and len(iter_result.data) > 30 and not iter_result.no_more_items)
            if iteration_had_progress:
                page_iteration = tentative_ordinal
            else:
                logger.info(f"  Ordinal stays at {page_iteration} (no progress this iteration)")

            status = "VIABLE" if viable else ("END_OF_PAGE" if iter_result.no_more_items else "SKIP")
            fail_info = f" [{fail_cat}]" if fail_cat else ""
            logger.info(f"  → {status}{fail_info} ({result.total_steps} steps, {parse_failures} parse failures, {iter_result.duration:.0f}s)")
            if fail_reason:
                logger.info(f"  Reason: {fail_reason}")
            if iter_result.data and viable:
                logger.info(f"  Data: {iter_result.data[:100]}")

            # Progress callback — allows caller to write intermediate results
            if self.on_iteration:
                try:
                    self.on_iteration(global_iteration, iter_result, results)
                except Exception as e:
                    logger.warning(f"on_iteration callback error: {e}")

            # Check for end of page
            if iter_result.no_more_items:
                logger.info(f"No more items on page {page}. Attempting pagination...")
                time.sleep(random.uniform(3.0, 6.0))  # Pause before pagination
                paginated = self._run_pagination()
                if paginated:
                    page += 1
                    page_iteration = 0
                    logger.info(f"Advanced to page {page}")
                else:
                    logger.info("No more pages. Loop complete.")
                    break

            # Track consecutive failures for force-pagination
            if viable:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Detect page exhaustion: model says "footer", "bottom of page", etc.
            if not viable and extracted:
                extracted_lower = extracted.lower()
                page_exhausted = sum(1 for sig in PAGE_EXHAUSTED_SIGNALS if sig in extracted_lower)
                if page_exhausted >= 1:
                    logger.info(f"  Page exhaustion detected ({page_exhausted} signals) — paginating")
                    time.sleep(random.uniform(2.0, 4.0))
                    paginated = self._run_pagination()
                    if paginated:
                        page += 1
                        page_iteration = 0
                        consecutive_failures = 0
                        logger.info(f"  Advanced to page {page}")
                    else:
                        logger.info("  No more pages. Loop complete.")
                        break
                    continue  # Skip remaining checks, start fresh on new page

            # Check for hard failure (loop/max_steps without useful output)
            if result.termination_reason == "loop" and not result.success and not iter_result.data:
                logger.warning("Iteration stuck (loop). Moving to pagination.")
                paginated = self._run_pagination()
                if paginated:
                    page += 1
                    page_iteration = 0
                    consecutive_failures = 0
                else:
                    break

            # Force-paginate after too many consecutive failures
            # (model is stuck clicking the same broken link or off-site element)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"  {consecutive_failures} consecutive failures — force-paginating to next page")
                time.sleep(random.uniform(2.0, 4.0))
                paginated = self._run_pagination()
                if paginated:
                    page += 1
                    page_iteration = 0
                    consecutive_failures = 0
                    logger.info(f"  Force-paginated to page {page}")
                else:
                    logger.info("  No more pages after force-pagination. Loop complete.")
                    break

        # ── End of loop: save learnings and log failure summary ──
        self._learning_store.save()

        # Failure breakdown
        from collections import Counter
        fail_counts = Counter(r.failure_category for r in results if r.failure_category)
        viable_count = sum(1 for r in results if r.success)
        logger.info(f"Loop complete: {len(results)} iterations, {page} pages, {viable_count} viable")
        if fail_counts:
            logger.info("  Failure breakdown:")
            for cat, count in fail_counts.most_common():
                logger.info(f"    {cat}: {count}")
        if self._learning_store.learnings:
            logger.info(f"  Persisted {len(self._learning_store.learnings)} learnings for {self._learning_store.domain}")

        return results

    def _run_iteration(self, intent: str, task_id: str):
        """Run a single iteration as a bounded GymRunner call."""
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=self.config.max_steps_per_iteration,
            frames_per_inference=2,
            grounding=self.grounding,
            on_step=self.on_step,
        )

        # Retry loop
        best_result = None
        for attempt in range(1, self.config.max_retries_per_iteration + 1):
            attempt_intent = intent
            if attempt > 1 and best_result:
                attempt_intent += (
                    f"\n\nPrevious attempt failed ({best_result.termination_reason}). "
                    f"Try a different approach."
                )

            result = runner.run(task=attempt_intent, task_id=task_id, start_url=self.start_url)
            best_result = result

            if result.success or self._check_no_more(result):
                break

        return best_result

    def _run_pagination(self) -> bool:
        """Run a pagination step. Returns True if next page loaded."""
        # First scroll to top so the model can see the full page
        try:
            from ..actions import Action
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1)
        except Exception:
            pass

        # Enhanced pagination intent — specific and positive
        pagination_task = (
            self.config.pagination_intent + "\n\n"
            "Scroll to the BOTTOM of the page. Look for page numbers (1, 2, 3...) "
            "or a 'Next' link. Click the next page number or 'Next'. "
            "If the page changes and shows new listings, call done(success=true). "
            "If there is no Next button or no more pages, call done(success=false, summary='no more pages')."
        )

        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=self.config.max_steps_pagination,
            frames_per_inference=2,
            grounding=self.grounding,
            on_step=self.on_step,
        )

        result = runner.run(
            task=pagination_task,
            task_id="pagination",
        )

        # If model signaled success, pagination worked
        if result.success:
            # Scroll back to top of new page so listings are visible
            try:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(1.5)
            except Exception:
                pass
            return True

        # Check if model signaled "no more pages"
        if self._check_no_more(result, check_pages=True):
            return False

        # Ambiguous — try ONE more time with explicit scroll-to-bottom
        logger.info("  Pagination ambiguous — retrying with scroll to bottom")
        retry_runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=15,
            frames_per_inference=2,
            grounding=self.grounding,
            on_step=self.on_step,
        )
        retry_result = retry_runner.run(
            task=(
                "Scroll to the very bottom of this page. "
                "Find and click the 'Next' button or the next page number. "
                "done(success=true) if you clicked it."
            ),
            task_id="pagination_retry",
        )
        if retry_result.success:
            try:
                self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
                time.sleep(1.5)
            except Exception:
                pass
            return True

        return False

    def _check_no_more(self, result, check_pages: bool = False) -> bool:
        """Check if the model signaled no more items/pages.

        Only inspects the terminate() summary — not thinking text, which
        may echo the task prompt and contain false-positive signal words.
        Requires success == False to distinguish genuine end-of-page from
        a successful extraction that happens to mention 'no more'.
        """
        for step in result.trajectory:
            if step.action.action_type.value == "done":
                if step.action.params.get("success", True):
                    continue  # Successful extraction, not an end signal
                summary = str(step.action.params.get("summary", "")).lower()

                for signal in NO_MORE_SIGNALS:
                    if signal in summary:
                        return True

        return False

    @staticmethod
    def _count_parse_failures(result) -> int:
        """Count steps where the brain failed to produce a valid action.

        These are WAIT actions with 'Could not parse' reasoning — the model
        outputted text but the parser couldn't extract an action. They represent
        wasted steps that shouldn't count as real navigation progress.
        """
        count = 0
        for step in result.trajectory:
            if step.action.action_type == ActionType.WAIT:
                reasoning = getattr(step.action, "reasoning", "") or ""
                # WAIT with parse failure reasoning
                if "could not parse" in reasoning.lower():
                    count += 1
                    continue
                # WAIT from vLLM/llama-server errors (empty response)
                if not step.thinking and not reasoning:
                    count += 1
        return count

    @staticmethod
    def _distill_learning(result, iter_result, viable: bool) -> str:
        """Analyze the trajectory and distill actionable learnings.

        Examines the full trajectory to detect failure patterns:
        - Off-site navigation (Facebook, Instagram, external dealer sites)
        - Backtracking loops (repeated Alt+Left)
        - Image gallery traps
        - Wrong clicks (ads, social icons, menus)
        - Cookie/popup blocking
        - Successful patterns to reinforce

        Returns a specific, actionable learning for future iterations.
        """
        data = str(iter_result.data).lower() if iter_result.data else ""
        all_thinking = " ".join(str(s.thinking or "") for s in result.trajectory).lower()
        actions = [str(s.action)[:80] for s in result.trajectory] if result.trajectory else []
        steps = result.total_steps

        # ── Off-site navigation (highest priority — wastes entire iteration) ──
        offsite_domains = [
            "facebook.com", "instagram.com", "twitter.com", "x.com",
            "youtube.com", "tiktok.com", "pinterest.com", "linkedin.com",
        ]
        for domain in offsite_domains:
            if domain in all_thinking or domain in data:
                return (
                    f"CRITICAL: You navigated to {domain} by clicking a social media link. "
                    f"Social icons are SMALL colored squares (20-40px): blue=Facebook, gradient=Instagram. "
                    f"They appear in the FOOTER (bottom of page, y>650) or SIDEBAR (right edge, x>1100). "
                    f"Listing cards are LARGE rectangles (~250px tall) in the CENTER showing: "
                    f"[boat photo] [Year Make Model] [$Price]. ONLY click elements with a boat photo AND a price."
                )

        # External dealer/brand sites
        external_sites = [
            "hanover yachts", "tige boats", "cobalt boats", "bayliner.com",
            "sea ray.com", "boston whaler.com", "grady-white.com",
        ]
        for site in external_sites:
            if site in all_thinking or site in data:
                return (
                    f"You navigated to an external dealer/brand website ({site}). "
                    f"Only click listings within BoatTrader search results. "
                    f"If you end up on a different site, press Alt+Left immediately to go back."
                )

        # ── Backtracking loops (model clicking wrong things repeatedly) ──
        back_actions = sum(1 for a in actions if "alt" in a.lower() and "left" in a.lower())
        if back_actions >= 3:
            return (
                "You pressed Alt+Left (back) too many times — you keep clicking wrong elements. "
                "STOP and look carefully at the search results. Listing cards show a boat image, "
                "Year/Make/Model text, and a price. Click the TITLE TEXT or the BOAT IMAGE, "
                "not menu items, ads, social icons, or the 'Research' dropdown."
            )

        # ── Image gallery trap ──
        if "image viewer" in all_thinking or "lightbox" in all_thinking or "1 of" in all_thinking or "gallery" in all_thinking:
            return (
                "You clicked the PHOTO and got trapped in an image gallery. "
                "Press Escape then Alt+Left to go back. "
                "NEXT TIME: do NOT click the boat photo. Instead click one of these:\n"
                "  - The boat NAME TEXT (e.g. '2022 Grady-White Freedom 235')\n"
                "  - The PRICE text (e.g. '$145,000')\n"
                "  - A 'View Details', 'See Details', or 'Contact Seller' link\n"
                "These open the detail page WITHOUT the gallery trap."
            )

        # ── 404 / Page Not Found / external site ──
        external_domain = ""
        for domain in ["elitemarine", "eliteboat", "survey.com", "boats-group"]:
            if domain in all_thinking or domain in data:
                external_domain = domain
                break
        if "page not found" in all_thinking or "404" in all_thinking:
            msg = (
                "You clicked a BROKEN LINK that returned 404. "
                "Press Alt+Left to go back to search results. "
                "Then SCROLL DOWN past this broken listing to find the NEXT one. "
                "Do NOT click the same element again."
            )
            if external_domain:
                msg += (
                    f" The broken link was to an EXTERNAL site ({external_domain}) — "
                    f"not a real boat listing. It might be an ad, dealer badge, or sponsored link. "
                    f"Look for listing cards with a boat PHOTO + Year Make Model TEXT + PRICE."
                )
            return msg

        # ── Cloudflare / verification ──
        if "cloudflare" in data or "verify you are human" in data:
            return "Cloudflare blocked this listing. Press Alt+Left and skip to the next one."

        # ── Cookie/popup blocking ──
        if "cookie" in all_thinking and steps > 10 and not viable:
            return "Cookie popup appeared. It should be auto-dismissed. If you see it, click Accept once and move on."

        # ── Too much scrolling, no clicking ──
        clicks = sum(1 for a in actions if "click" in a.lower())
        scrolls = sum(1 for a in actions if "scroll" in a.lower())
        if scrolls > clicks * 2 and not viable:
            return (
                "Too much scrolling without clicking into a listing. "
                "CLICK the boat listing card first (title or image), THEN scroll on the detail page."
            )

        # ── Model ran out of steps ──
        if result.termination_reason == "max_steps" and not viable:
            return (
                "Ran out of steps. Be faster: click listing → scroll down 5x aggressively → "
                "read Description/Seller Notes for phone → Alt+Left back. Don't re-scroll."
            )

        # ── Successful extraction — reinforce what worked ──
        if viable and iter_result.data:
            return "Successfully extracted data. Keep using: click listing title → scroll past photos → read description → Alt+Left back."

        # ── Model couldn't navigate back ──
        if "timeout" in data or "go_back" in data:
            return "Navigation back failed. Use key_press('alt+Left') — do NOT click browser back button."

        # ── Parse failures (model producing unparseable output) ──
        pf = getattr(iter_result, 'parse_failures', 0) or 0
        if pf > steps * 0.4:
            return (
                f"{pf}/{steps} of your actions couldn't be parsed. "
                f"Output actions in the correct format. Don't just describe what you see — ACT."
            )

        # ── Generic: no phone found (legitimate) ──
        if not viable and iter_result.data:
            return "No phone found on this listing. That's OK — skip quickly and try the next one."

        return ""

    @staticmethod
    def _extract_phone(text: str) -> str | None:
        """Extract first real phone number from text.

        Real = 10+ digits, not 555 exchange, not embedded in URLs.
        Per spec: (555)555-5555, 555-555-5555, 555.555.5555, 10+ digits.
        NOT: prices, years, zip codes, partial numbers, URL fragments.
        """
        import re
        # Match phone patterns
        phone_pattern = re.compile(r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}')
        for match in phone_pattern.finditer(text):
            phone = match.group()
            digits = re.sub(r'\D', '', phone)
            if len(digits) < 10 or len(digits) > 11:
                continue
            # Filter 555 exchange (fictitious)
            if digits[3:6] == "555":
                continue
            # Filter if embedded in URL
            start = max(0, match.start() - 20)
            context = text[start:match.start()].lower()
            if any(c in context for c in ["http", "://", "url=", "&", "?"]):
                continue
            return phone
        return None

    @staticmethod
    def _extract_data(result) -> str:
        """Extract data from the model's output.

        Holo3 often outputs "VIABLE | Year" in done() but puts real data
        in its thinking text. So we check both:
        1. done() summary — if it has real values (Year: 2024), use it
        2. Thinking text — parse boat names, prices, phones from reasoning
        3. Merge: if done() has real data but is missing URL, backfill from thinking

        IMPORTANT: URLs are searched in REVERSE step order (latest first) to avoid
        grabbing URLs from previous listings that bleed into the thinking context.
        """
        import re as _re

        # Priority 1: done() summary IF it has real data (not just "VIABLE | Year")
        for step in reversed(result.trajectory):
            if step.action.action_type.value == "done":
                summary = str(step.action.params.get("summary", ""))
                # Check if summary has actual data (year number, price, etc.)
                if summary and _re.search(r"Year: (?:19|20)\d{2}", summary):
                    # Backfill missing URL from thinking — search LATEST steps first
                    if "URL:" not in summary or "URL: none" in summary.lower():
                        year_m = _re.search(r"Year: ((?:19|20)\d{2})", summary)
                        extracted_year = year_m.group(1) if year_m else ""
                        url = _extract_url_latest(result.trajectory, year=extracted_year)
                        if url:
                            if "URL:" in summary:
                                summary = _re.sub(r"URL:\s*\S*", f"URL: {url}", summary)
                            else:
                                summary += f" | URL: {url}"
                    return summary

        # Priority 2: Build structured data from thinking text
        # Use only the LAST HALF of steps to avoid cross-listing contamination
        trajectory = result.trajectory
        half = max(len(trajectory) // 2, 1)
        recent_thinking = " ".join(str(s.thinking or "") for s in trajectory[half:])
        all_thinking = " ".join(str(s.thinking or "") for s in trajectory)
        # Use recent_thinking for boat data (avoids previous listing bleed)
        search_text = recent_thinking if len(recent_thinking) > 50 else all_thinking

        if len(search_text) > 50:
            # Extract boat data from thinking
            parts = []

            # Year + Make + Model pattern
            boat_match = _re.search(r"(\d{4})\s+([\w\-]+)\s+([\w\-]+(?:\s+[\w\-]+)?(?:\s+[\w\-]+)?)", search_text)
            if boat_match:
                year, make, model = boat_match.group(1), boat_match.group(2), boat_match.group(3)
                parts.append(f"Year: {year}")
                parts.append(f"Make: {make}")
                parts.append(f"Model: {model}")

            # Price
            price_match = _re.search(r"\$[\d,]+", search_text)
            if price_match:
                parts.append(f"Price: {price_match.group()}")

            # Phone
            phone_match = _re.search(r"\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}", search_text)
            if phone_match:
                digits = _re.sub(r"\D", "", phone_match.group())
                if digits[3:6] != "555":
                    parts.append(f"Phone: {phone_match.group()}")

            # URL — search latest steps first, validate year matches
            extracted_year = boat_match.group(1) if boat_match else ""
            url = _extract_url_latest(trajectory, year=extracted_year)
            if url:
                parts.append(f"URL: {url}")

            if parts:
                return "VIABLE | " + " | ".join(parts)

        # Priority 3: Last done() summary even if it's just "VIABLE | Year"
        for step in reversed(result.trajectory):
            if step.action.action_type.value == "done":
                summary = str(step.action.params.get("summary", ""))
                if summary and len(summary) > 10:
                    return summary

        # Priority 4: Last thinking
        for step in reversed(result.trajectory):
            thinking = str(step.thinking or "")
            if len(thinking) > 30:
                return thinking[:500]

        return ""

    def _validate_viable(self, data: str, listing_url: str | None = None) -> bool:
        """A listing is viable if we extracted ANY useful boat data.

        Per spec: fill PopYachts form with whatever we have —
        Year, Make, Model, Type, Price, URL are always present.
        Phone and Seller Name are optional bonus fields.

        Viable = reached the listing detail page and got boat data.
        Not viable = 404, Cloudflare, off-site, blank page, or prompt echo.
        """
        if not data:
            return False

        import re
        data = str(data)
        text = data.lower()

        # Reject dealer listings (not private sellers — no phone numbers)
        dealer_signals = [
            "more from this dealer", "view dealer website",
            "more boats from this dealer", "visit seller website",
        ]
        if any(sig in text for sig in dealer_signals):
            return False

        # Reject error/blocked pages
        reject_signals = [
            "cloudflare", "verify you are human",
            "about:blank", "blank page", "captcha",
            "page not found", "404", "can't be reached",
            "err_tunnel", "connection failed",
        ]
        if any(sig in text for sig in reject_signals):
            return False

        # Reject prompt echoes
        prompt_markers = ["formats like", "do not confuse", "output format",
                         "terminate('success') with:", "terminate('failure') with:"]
        if sum(1 for m in prompt_markers if m in text) >= 2:
            return False

        # Reject off-site pages
        offsite = ["facebook.com", "instagram.com", "twitter.com", "youtube.com"]
        if any(site in text for site in offsite):
            return False

        # Dedup by listing URL
        if listing_url:
            if listing_url in self._seen_urls:
                logger.info(f"  Dedup: URL {listing_url} already seen, skipping")
                return False
            self._seen_urls.add(listing_url)

        # Check for boat data — need at least Year OR (Make/Model + Price)
        has_year = bool(re.search(r'(?:19|20)\d{2}', data))
        has_price = bool(re.search(r'\$[\d,]+', data))
        has_boat_info = False
        boat_keywords = [
            "make", "model", "hull", "engine", "console", "cabin",
            "grady", "boston whaler", "sea hunt", "tracker", "yamaha",
            "mercury", "suzuki", "honda", "evinrude", "intrepid",
            "azimut", "sea ray", "sundeck", "walkaround", "sportfish",
            "bayliner", "chaparral", "everglades", "cigarette", "century",
            "cobia", "nor-tech", "may-craft", "key west", "robalo",
        ]
        for kw in boat_keywords:
            if kw in text:
                has_boat_info = True
                break

        # Viable if we have meaningful boat data
        if has_year and (has_price or has_boat_info):
            # Track phone for dedup if present
            phone = self._extract_phone(data)
            if phone:
                digits = re.sub(r'\D', '', phone)
                if digits not in self._seen_phones:
                    self._seen_phones.add(digits)
            return True

        # Also viable if model explicitly said VIABLE
        if "viable" in text[:50].lower():
            return True

        return False
