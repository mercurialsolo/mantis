"""Sub-plan decomposition with state checkpointing for Mantis CUA agent.

Breaks each listing extraction into 4 deterministic micro-steps:
  FIND  -> locate the Nth listing card on the results page
  CLICK -> click the listing title text (not photo) to open detail page
  EXTRACT -> scroll past photos, extract Year/Make/Model/Price/Phone/Seller
  RETURN -> navigate back to the search results page

Each micro-step is a bounded GymRunner.run() call with its own max_steps,
grounding config, and failure handling. Checkpoints are saved after every
micro-step so runs can resume from the last successful state.

This replaces the monolithic single-intent approach (WorkflowRunner) with
fine-grained control — each step gets a focused prompt, reducing model
confusion and enabling targeted retry on failure.

Usage:
    from mantis_agent.gym.sub_plan import SubPlanRunner

    runner = SubPlanRunner(brain=brain, env=env, grounding=grounding)
    result = runner.run_listing(iteration=1, page=1, page_iteration=1)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..actions import Action, ActionType
from .runner import GymRunner, RunResult

logger = logging.getLogger(__name__)

# ── Ordinals for natural-language listing references ────────────────────────

ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth",
}


# ── Micro-step definitions ──────────────────────────────────────────────────

@dataclass
class SubPlanStep:
    """A single micro-step within the sub-plan pipeline."""

    name: str
    intent_template: str
    max_steps: int
    success_signal: str
    failure_action: str
    grounding_enabled: bool


# The 4 fixed micro-steps, executed in order for every listing.
MICRO_STEPS: list[SubPlanStep] = [
    SubPlanStep(
        name="FIND",
        intent_template=(
            "You are on a search results page with boat listing cards. "
            "Look at the screen. Find the {ORDINAL} listing card — it shows a boat photo with title text below it.\n"
            "Do NOT click anything. Just LOOK at the listing cards and read the title text.\n"
            "If you can see the {ORDINAL} listing card, report its title.\n"
            "If you need to scroll down to see more listings, scroll(direction='down', amount=3).\n"
            "done(success=true, summary='FOUND: <Year Make Model from title text>')\n"
            "If no more listings visible after scrolling: done(success=false, summary='no more listings')"
        ),
        max_steps=8,
        success_signal="FOUND",
        failure_action="abort",
        grounding_enabled=False,
    ),
    SubPlanStep(
        name="CLICK",
        intent_template=(
            "Click the listing title text '{TITLE}'. The title is the text BELOW the large boat photo.\n"
            "The title shows Year Make Model (e.g. '2024 Grady-White Freedom 235').\n"
            "Click on this TEXT, NOT on the photo image above it.\n"
            "After clicking, a detail page should load with more boat info.\n"
            "done(success=true, summary='detail page loaded')\n"
            "If you see a fullscreen photo ('1 of N'): done(success=false, summary='gallery trap')"
        ),
        max_steps=5,
        success_signal="detail page loaded",
        failure_action="retry",
        grounding_enabled=True,
    ),
    SubPlanStep(
        name="EXTRACT",
        intent_template=(
            "You are on a boat listing detail page.\n\n"
            "STEP 1: Look at the title bar — it shows the boat's Year, Make, Model and Price. READ these now.\n"
            "STEP 2: scroll(direction='down', amount=5) — this gets past the photo gallery.\n"
            "STEP 3: scroll(direction='down', amount=3) — now you should see Description or Seller Info.\n"
            "STEP 4: Read the URL from the browser address bar — it contains the listing URL.\n"
            "STEP 5: Read any phone number visible (format: 305-555-1234). Read seller name if shown.\n"
            "STEP 6: Call done() with the ACTUAL data you read from the page:\n\n"
            "Report what you found — replace <...> with ACTUAL values from the page:\n"
            "done(success=true, summary='VIABLE | Year: <year> | Make: <make> "
            "| Model: <model> | Price: <price> | Phone: <phone or none> | URL: <from address bar>')\n\n"
            "Example: if you see a 2018 Everglades 355 at $239,000 with URL boattrader.com/boat/123:\n"
            "done(success=true, summary='VIABLE | Year: 2018 | Make: Everglades "
            "| Model: 355 CC | Price: $239000 | Phone: none | URL: boattrader.com/boat/123')\n\n"
            "CRITICAL: Fill in REAL data from the page. Do NOT write 'Year: Year'."
        ),
        max_steps=15,
        success_signal="VIABLE",
        failure_action="skip",
        grounding_enabled=False,
    ),
    SubPlanStep(
        name="RETURN",
        intent_template=(
            "Go back to the search results page.\n"
            "key_press(keys='alt+left') to navigate back.\n"
            "done(success=true, summary='back on results') when you see listing cards."
        ),
        max_steps=5,
        success_signal="back on results",
        failure_action="navigate",
        grounding_enabled=False,
    ),
]

# Map step names to their index for ordering comparisons.
_STEP_ORDER = {step.name: i for i, step in enumerate(MICRO_STEPS)}


# ── Checkpoint persistence ──────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """Snapshot of a listing extraction run after each micro-step."""

    iteration: int
    page: int
    step_name: str  # last completed step
    found: bool = False
    listing_title: str = ""
    detail_page_loaded: bool = False
    gallery_detected: bool = False
    extracted_data: dict = field(default_factory=dict)
    back_on_results: bool = False
    attempts: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "page": self.page,
            "step_name": self.step_name,
            "found": self.found,
            "listing_title": self.listing_title,
            "detail_page_loaded": self.detail_page_loaded,
            "gallery_detected": self.gallery_detected,
            "extracted_data": self.extracted_data,
            "back_on_results": self.back_on_results,
            "attempts": self.attempts,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(
            iteration=d["iteration"],
            page=d["page"],
            step_name=d.get("step_name", ""),
            found=d.get("found", False),
            listing_title=d.get("listing_title", ""),
            detail_page_loaded=d.get("detail_page_loaded", False),
            gallery_detected=d.get("gallery_detected", False),
            extracted_data=d.get("extracted_data", {}),
            back_on_results=d.get("back_on_results", False),
            attempts=d.get("attempts", 0),
            timestamp=d.get("timestamp", 0.0),
        )


class CheckpointStore:
    """Persists checkpoints to disk as JSON files.

    Directory layout:
        {base_dir}/{session_name}/iter_{N}.json
    """

    def __init__(self, base_dir: str = "/data/checkpoints"):
        self.base_dir = base_dir

    def _session_dir(self, session_name: str) -> str:
        return os.path.join(self.base_dir, session_name)

    def save(self, session_name: str, checkpoint: Checkpoint) -> None:
        """Write checkpoint to disk. Creates directories as needed."""
        d = self._session_dir(session_name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"iter_{checkpoint.iteration}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            os.replace(tmp, path)  # atomic on POSIX
            logger.debug(f"Checkpoint saved: {path}")
        except OSError as exc:
            logger.warning(f"Failed to save checkpoint {path}: {exc}")
            # Non-fatal — the run continues without persistence.

    def load(self, session_name: str, iteration: int) -> Checkpoint | None:
        """Load a specific iteration's checkpoint."""
        path = os.path.join(self._session_dir(session_name), f"iter_{iteration}.json")
        return self._read(path)

    def load_latest(self, session_name: str) -> Checkpoint | None:
        """Load the highest-numbered checkpoint in a session."""
        d = self._session_dir(session_name)
        if not os.path.isdir(d):
            return None

        best: Checkpoint | None = None
        best_iter = -1
        for fname in os.listdir(d):
            m = re.match(r"iter_(\d+)\.json$", fname)
            if m:
                n = int(m.group(1))
                if n > best_iter:
                    cp = self._read(os.path.join(d, fname))
                    if cp is not None:
                        best = cp
                        best_iter = n
        return best

    @staticmethod
    def _read(path: str) -> Checkpoint | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return Checkpoint.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"Corrupt checkpoint {path}: {exc}")
            return None


# ── Sub-plan result ─────────────────────────────────────────────────────────

@dataclass
class SubPlanResult:
    """Outcome of running the full sub-plan for one listing."""

    iteration: int
    page: int
    success: bool
    data: str  # formatted extraction string or empty
    no_more_items: bool
    steps: int  # total steps across all micro-steps
    duration: float
    checkpoint: Checkpoint
    parse_failures: int = 0
    micro_results: dict[str, RunResult | None] = field(default_factory=dict)


# ── Sub-plan runner ─────────────────────────────────────────────────────────

class SubPlanRunner:
    """Executes the 4-step sub-plan pipeline for a single listing.

    Each micro-step gets a fresh GymRunner with focused intent, bounded
    step count, and optional grounding. Checkpoints are persisted after
    every micro-step so the run can resume on crash.

    Args:
        brain: Any object with a think() method (Brain protocol).
        env: GymEnvironment instance (browser stays on current page).
        grounding: Optional GroundingModel for CLICK refinement.
        on_step: Optional viewer callback fn(dict).
        checkpoint_dir: Base directory for checkpoint storage.
    """

    def __init__(
        self,
        brain: Any,
        env: Any,
        grounding: Any = None,
        on_step: Any = None,
        checkpoint_dir: str = "/data/checkpoints",
    ):
        self.brain = brain
        self.env = env
        self.grounding = grounding
        self.on_step = on_step
        self.store = CheckpointStore(base_dir=checkpoint_dir)

        # Accumulated learnings per micro-step, persisted across listings
        # so the model improves over the session.
        self.step_learnings: dict[str, list[str]] = {
            "FIND": [],
            "CLICK": [],
            "EXTRACT": [],
            "RETURN": [],
        }

    # ── Public API ──────────────────────────────────────────────────────

    def run_listing(
        self,
        iteration: int,
        page: int,
        page_iteration: int,
        checkpoint: Checkpoint | None = None,
    ) -> SubPlanResult:
        """Run the full FIND -> CLICK -> EXTRACT -> RETURN pipeline.

        Args:
            iteration: Global iteration number (across all pages).
            page: Current page number.
            page_iteration: 1-based listing index within this page.
            checkpoint: Resume from a prior checkpoint (optional).

        Returns:
            SubPlanResult with extraction data and updated checkpoint.
        """
        t0 = time.time()
        steps = 0

        if checkpoint is None:
            checkpoint = Checkpoint(
                iteration=iteration,
                page=page,
                step_name="",  # nothing completed yet
            )

        micro_results: dict[str, RunResult | None] = {}
        session_name = f"page{page}"

        for step in MICRO_STEPS:
            # Skip steps already completed in a prior run.
            if self._step_already_done(checkpoint, step.name):
                logger.info(
                    f"  [{step.name}] skipped (checkpoint past this step)"
                )
                continue

            # Build the focused intent for this micro-step.
            intent = self._build_intent(step, checkpoint, page_iteration)

            logger.info(
                f"  [{step.name}] starting (max_steps={step.max_steps}, "
                f"grounding={'ON' if step.grounding_enabled else 'OFF'})"
            )

            # Fresh runner per micro-step — bounded, focused, no URL navigation.
            runner = GymRunner(
                brain=self.brain,
                env=self.env,
                max_steps=step.max_steps,
                frames_per_inference=2,
                grounding=self.grounding if step.grounding_enabled else None,
                on_step=self.on_step,
            )

            result = runner.run(
                task=intent,
                task_id=f"iter{iteration}_{step.name}",
                # No start_url — browser stays on the current page.
            )
            micro_results[step.name] = result
            steps += result.total_steps

            # Update checkpoint from the result.
            checkpoint = self._update_checkpoint(checkpoint, step, result)
            checkpoint.timestamp = time.time()

            # Persist checkpoint after every micro-step (crash safety).
            try:
                self.store.save(session_name, checkpoint)
            except Exception as exc:
                logger.warning(f"  [{step.name}] checkpoint save failed: {exc}")

            # Check outcome and decide: continue, retry, skip, or abort.
            if self._step_succeeded(step, result):
                logger.info(
                    f"  [{step.name}] succeeded "
                    f"({result.total_steps} steps, {result.total_time:.1f}s)"
                )
                continue

            # Step failed — handle based on step type.
            summary = self._extract_done_summary(result)
            logger.warning(
                f"  [{step.name}] failed: {summary or result.termination_reason}"
            )

            action = self._handle_failure(step, result, checkpoint)

            if action == "abort":
                # No more listings on this page.
                logger.info(f"  [{step.name}] abort — no more listings")
                duration = time.time() - t0
                return SubPlanResult(
                    iteration=iteration,
                    page=page,
                    success=False,
                    data="",
                    no_more_items=True,
                    steps=steps,
                    duration=duration,
                    checkpoint=checkpoint,
                    micro_results=micro_results,
                )

            if action == "retry":
                # Gallery trap — escape and retry CLICK once more.
                logger.info(f"  [{step.name}] retrying after gallery escape")
                retry_runner = GymRunner(
                    brain=self.brain,
                    env=self.env,
                    max_steps=step.max_steps,
                    frames_per_inference=2,
                    grounding=self.grounding if step.grounding_enabled else None,
                    on_step=self.on_step,
                )
                retry_intent = self._build_intent(step, checkpoint, page_iteration)
                retry_intent += (
                    "\n\nCRITICAL: Your previous click opened a photo gallery. "
                    "Click the TITLE TEXT (Year Make Model), NOT the photo."
                )
                retry_result = retry_runner.run(
                    task=retry_intent,
                    task_id=f"iter{iteration}_{step.name}_retry",
                )
                micro_results[f"{step.name}_retry"] = retry_result
                steps += retry_result.total_steps

                checkpoint = self._update_checkpoint(checkpoint, step, retry_result)
                checkpoint.timestamp = time.time()
                try:
                    self.store.save(session_name, checkpoint)
                except Exception:
                    pass

                if self._step_succeeded(step, retry_result):
                    logger.info(f"  [{step.name}] retry succeeded")
                    # Record learning for future iterations.
                    self.step_learnings["CLICK"].append(
                        "Click the TITLE TEXT below the photo, not the photo itself."
                    )
                    continue
                else:
                    # Retry also failed — skip this listing entirely.
                    logger.warning(f"  [{step.name}] retry also failed, skipping listing")
                    self.step_learnings["CLICK"].append(
                        "Gallery trap persisted after retry. "
                        "Try clicking price text or 'View Details' link instead."
                    )
                    # Fall through to RETURN.

            if action == "skip":
                # Skip to RETURN — run the return step to get back to results.
                logger.info(f"  [{step.name}] skipping to RETURN")
                self.step_learnings[step.name].append(
                    f"{step.name} failed — skipped listing."
                )
                # Break inner loop; we still need to run RETURN below.
                break

        # Always ensure we return to results page, even after skip/failure.
        # If RETURN wasn't completed yet, run it now.
        if not checkpoint.back_on_results:
            return_step = MICRO_STEPS[3]  # RETURN
            if return_step.name not in micro_results:
                logger.info("  [RETURN] running cleanup return")
                return_runner = GymRunner(
                    brain=self.brain,
                    env=self.env,
                    max_steps=return_step.max_steps,
                    frames_per_inference=2,
                    on_step=self.on_step,
                )
                return_result = return_runner.run(
                    task=return_step.intent_template,
                    task_id=f"iter{iteration}_RETURN_cleanup",
                )
                micro_results["RETURN_cleanup"] = return_result
                steps += return_result.total_steps
                checkpoint.back_on_results = self._step_succeeded(
                    return_step, return_result
                )
                checkpoint.timestamp = time.time()
                try:
                    self.store.save(session_name, checkpoint)
                except Exception:
                    pass

        duration = time.time() - t0
        data = self._format_data(checkpoint) if checkpoint.extracted_data else ""
        success = bool(checkpoint.extracted_data)

        logger.info(
            f"  Sub-plan complete: success={success}, "
            f"steps={steps}, duration={duration:.1f}s"
        )

        return SubPlanResult(
            iteration=iteration,
            page=page,
            success=success,
            data=data,
            no_more_items=False,
            steps=steps,
            duration=duration,
            checkpoint=checkpoint,
            micro_results=micro_results,
        )

    # ── Intent construction ─────────────────────────────────────────────

    # Per-step context — only what's relevant for THAT step, not a generic preamble.
    STEP_CONTEXT = {
        "FIND": "The screen shows a search results page with boat listing cards. Each card has a boat photo with title text (Year Make Model) and price below it.",
        "CLICK": "You found a listing. Now click its TITLE TEXT (the text below the photo showing Year Make Model). Do NOT click the photo image.",
        "EXTRACT": "You are on a boat listing detail page. The title at the top shows Year Make Model and Price. Scroll down past the photos to read Description and Seller Notes for phone numbers.",
        "RETURN": "You finished reading the listing. Go back to the search results page.",
    }

    def _build_intent(
        self,
        step: SubPlanStep,
        checkpoint: Checkpoint,
        page_iteration: int,
    ) -> str:
        """Build the focused intent prompt for a micro-step.

        Each step gets only the context relevant to it:
        - FIND: "page shows listing cards"
        - CLICK: "click TITLE TEXT, listing is: {title}"
        - EXTRACT: "on detail page, scroll to description"
        - RETURN: "go back to search results"
        """
        ordinal = ORDINALS.get(page_iteration, f"#{page_iteration}")

        # Step-specific context (1 sentence, not a paragraph)
        context = self.STEP_CONTEXT.get(step.name, "")
        intent = f"{context}\n\n"

        # Fill template
        step_text = step.intent_template
        step_text = step_text.replace("{N}", str(page_iteration))
        step_text = step_text.replace("{ORDINAL}", ordinal)
        step_text = step_text.replace("{TITLE}", checkpoint.listing_title or "the listing")
        intent += step_text

        # Checkpoint context — only for steps that need it
        if step.name == "CLICK" and checkpoint.listing_title:
            intent += f"\n\nTarget listing: '{checkpoint.listing_title}'"
        elif step.name == "EXTRACT" and checkpoint.listing_title:
            intent += f"\n\nListing: {checkpoint.listing_title}"

        # Step-specific learnings only (last 3)
        learnings = self.step_learnings.get(step.name, [])
        if learnings:
            intent += "\n\nLEARNINGS:"
            for l in learnings[-3:]:
                intent += f"\n- {l}"

        return intent

    # ── Checkpoint update ───────────────────────────────────────────────

    def _update_checkpoint(
        self,
        checkpoint: Checkpoint,
        step: SubPlanStep,
        result: RunResult,
    ) -> Checkpoint:
        """Update checkpoint fields based on the micro-step outcome."""
        summary = self._extract_done_summary(result)
        checkpoint.step_name = step.name
        checkpoint.attempts += 1

        if step.name == "FIND":
            if summary and "FOUND:" in summary:
                checkpoint.found = True
                # Extract title from "FOUND: 2024 Grady-White Freedom 235"
                title_match = re.search(r"FOUND:\s*(.+)", summary)
                if title_match:
                    checkpoint.listing_title = title_match.group(1).strip()

        elif step.name == "CLICK":
            summary_lower = (summary or "").lower()
            if "gallery" in summary_lower:
                checkpoint.gallery_detected = True
                checkpoint.detail_page_loaded = False
            elif result.success or "detail page loaded" in summary_lower:
                checkpoint.detail_page_loaded = True
                checkpoint.gallery_detected = False

        elif step.name == "EXTRACT":
            if summary:
                parsed = self._parse_extraction(summary)
                if parsed:
                    checkpoint.extracted_data = parsed

        elif step.name == "RETURN":
            summary_lower = (summary or "").lower()
            if result.success or "back on results" in summary_lower:
                checkpoint.back_on_results = True

        return checkpoint

    # ── Failure handling ────────────────────────────────────────────────

    def _handle_failure(
        self,
        step: SubPlanStep,
        result: RunResult,
        checkpoint: Checkpoint,
    ) -> str:
        """Decide how to handle a micro-step failure.

        Returns:
            "retry" — escape gallery and retry the same step.
            "skip"  — skip to RETURN and move to next listing.
            "abort" — no more listings, signal pagination.
        """
        if step.name == "FIND":
            # FIND failed = no more listings on this page.
            return "abort"

        if step.name == "CLICK":
            if checkpoint.gallery_detected:
                self._escape_gallery()
                if checkpoint.attempts < 2:
                    return "retry"
                return "skip"
            return "skip"

        if step.name == "EXTRACT":
            # Extraction failed — run RETURN to go back, try next listing.
            return "skip"

        if step.name == "RETURN":
            # RETURN failed — try direct navigation as last resort.
            try:
                self.env.step(Action(ActionType.KEY_PRESS, {"keys": "alt+Left"}))
                time.sleep(1.5)
            except Exception as exc:
                logger.warning(f"  [RETURN] fallback navigation failed: {exc}")
            return "skip"

        return "skip"

    def _escape_gallery(self) -> None:
        """Escape a photo gallery overlay via keyboard shortcuts.

        Sends Escape to close the lightbox, then Alt+Left to go back
        to the page state before the gallery opened.
        """
        logger.info("  [gallery] escaping photo gallery")
        try:
            self.env.step(Action(ActionType.KEY_PRESS, {"keys": "Escape"}))
            time.sleep(0.5)
            self.env.step(Action(ActionType.KEY_PRESS, {"keys": "alt+Left"}))
            time.sleep(1.5)
        except Exception as exc:
            logger.warning(f"  [gallery] escape failed: {exc}")

    # ── Result parsing utilities ────────────────────────────────────────

    def _extract_done_summary(self, result: RunResult) -> str:
        """Find the done() action in the trajectory and return its summary."""
        for step in reversed(result.trajectory):
            if step.action.action_type == ActionType.DONE:
                return str(step.action.params.get("summary", ""))
        return ""

    @staticmethod
    def _parse_extraction(text: str) -> dict[str, str]:
        """Parse 'VIABLE | Year: X | Make: Y | ...' into a dict.

        Handles both pipe-separated and free-form patterns. Returns an
        empty dict if no meaningful data is found.
        """
        result: dict[str, str] = {}

        if "VIABLE" not in text.upper():
            return result

        # Split on pipe separators.
        parts = text.split("|")
        for part in parts:
            part = part.strip()
            # Match "Key: Value" pattern.
            m = re.match(r"^(\w[\w\s]*?):\s*(.+)$", part)
            if m:
                key = m.group(1).strip().lower()
                value = m.group(2).strip()
                if key in ("year", "make", "model", "price", "phone", "seller", "url", "type"):
                    result[key] = value

        return result

    @staticmethod
    def _format_data(checkpoint: Checkpoint) -> str:
        """Format checkpoint.extracted_data back into the canonical string.

        Output: 'VIABLE | Year: 2024 | Make: Grady-White | ...'
        """
        d = checkpoint.extracted_data
        if not d:
            return ""

        parts = ["VIABLE"]
        for key in ("year", "make", "model", "price", "phone", "seller", "url", "type"):
            if key in d:
                label = key.upper() if key == "url" else key.capitalize()
                parts.append(f"{label}: {d[key]}")
        return " | ".join(parts)

    # ── Step ordering and success checks ────────────────────────────────

    @staticmethod
    def _step_already_done(checkpoint: Checkpoint, step_name: str) -> bool:
        """Check if the checkpoint has already passed this step.

        A step is 'already done' if the checkpoint's step_name is
        strictly later in the pipeline ordering.
        """
        if not checkpoint.step_name:
            return False

        done_idx = _STEP_ORDER.get(checkpoint.step_name, -1)
        current_idx = _STEP_ORDER.get(step_name, -1)
        return done_idx > current_idx

    def _step_succeeded(self, step: SubPlanStep, result: RunResult) -> bool:
        """Check if the micro-step completed its objective.

        Uses success_signal for exact match, plus step-specific heuristics:
        - FIND: any summary with a year (2015, 2024) = found a listing
        - CLICK: "detail" or "loaded" in summary
        - EXTRACT: "VIABLE" or year+make in summary
        - RETURN: "results" or "back" in summary
        """
        summary = self._extract_done_summary(result)
        if not summary:
            return result.success  # Fall back to model's own success flag
        sl = summary.lower()

        # Exact signal match
        if step.success_signal.lower() in sl:
            return True

        # Step-specific heuristics
        if step.name == "FIND":
            # If summary contains a year, the model found a listing
            return bool(re.search(r'(?:19|20)\d{2}', summary))
        elif step.name == "CLICK":
            return "detail" in sl or "loaded" in sl or "listing" in sl
        elif step.name == "EXTRACT":
            return "viable" in sl or bool(re.search(r'(?:19|20)\d{2}', summary))
        elif step.name == "RETURN":
            return "result" in sl or "back" in sl or "listing" in sl

        return result.success
