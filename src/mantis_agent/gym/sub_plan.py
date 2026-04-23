"""Micro-step sub-plan pipeline for listing extraction.

Architecture (7 steps per listing):
  FIND      -> Holo3 locates the next listing card           (5 steps)
  CLICK     -> Holo3 clicks the title text, grounding=True   (3 steps)
  READ_URL  -> Claude reads URL from screenshot (API only)    (0 Holo3 steps)
  SCROLL    -> Holo3 scrolls past photos                     (5 steps)
  EXTRACT   -> Claude expands safe controls and reads screenshots (API only)
  RETURN    -> Holo3 presses Alt+Left                        (3 steps)
  PAGINATE  -> Holo3 scrolls to bottom, clicks Next          (10 steps)

Design principles:
  - Fresh GymRunner per Holo3 step (zero context accumulation)
  - 1-sentence intents (short, positive, specific)
  - ClaudeExtractor for EXTRACT + READ_URL (vision API, not Holo3 thinking)
  - Dedup by URL (READ_URL checks _seen_urls, skips to RETURN if duplicate)
  - Checkpoint saved after each micro-step (crash recovery)
  - PAGINATE is separate from the listing loop

Usage:
    from mantis_agent.gym.sub_plan import SubPlanRunner

    runner = SubPlanRunner(brain=brain, env=env, grounding=grounding,
                           extractor=extractor)
    result = runner.run_listing(iteration=1, page=1, page_iteration=1)
    # When page exhausted:
    runner.run_pagination()
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..actions import Action, ActionType
from .runner import GymRunner, RunResult

if TYPE_CHECKING:
    from ..extraction import ClaudeExtractor

logger = logging.getLogger(__name__)


# -- Checkpoint persistence ---------------------------------------------------

@dataclass
class Checkpoint:
    """Snapshot of a listing extraction run after each micro-step."""

    iteration: int
    page: int
    step_name: str  # last completed step
    url: str = ""
    dedup: bool = False
    extracted_data: dict = field(default_factory=dict)
    back_on_results: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "page": self.page,
            "step_name": self.step_name,
            "url": self.url,
            "dedup": self.dedup,
            "extracted_data": self.extracted_data,
            "back_on_results": self.back_on_results,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        return cls(
            iteration=d["iteration"],
            page=d["page"],
            step_name=d.get("step_name", ""),
            url=d.get("url", ""),
            dedup=d.get("dedup", False),
            extracted_data=d.get("extracted_data", {}),
            back_on_results=d.get("back_on_results", False),
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

    def load(self, session_name: str, iteration: int) -> Checkpoint | None:
        """Load a specific iteration's checkpoint."""
        path = os.path.join(
            self._session_dir(session_name), f"iter_{iteration}.json"
        )
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


# -- Sub-plan result ----------------------------------------------------------

@dataclass
class SubPlanResult:
    """Outcome of running the full sub-plan for one listing."""

    iteration: int
    page: int
    success: bool
    data: str  # formatted extraction string or empty
    no_more_items: bool
    steps: int  # total Holo3 steps across all micro-steps
    duration: float
    checkpoint: Checkpoint
    dedup: bool = False
    url: str = ""
    page_exhausted: bool = False
    micro_results: dict[str, RunResult | None] = field(default_factory=dict)


# -- Sub-plan runner ----------------------------------------------------------

class SubPlanRunner:
    """Executes the 7-step micro-step pipeline for a single listing.

    Each Holo3 step gets a fresh GymRunner with a 1-sentence intent,
    bounded step count, and optional grounding. Claude API handles
    READ_URL and EXTRACT (no Holo3 steps needed).

    Args:
        brain: Any object with a think() method (Brain protocol).
        env: GymEnvironment instance with screenshot() method.
        grounding: Optional GroundingModel for CLICK refinement.
        extractor: Optional ClaudeExtractor for READ_URL + EXTRACT.
        on_step: Optional viewer callback fn(dict).
        checkpoint_dir: Base directory for checkpoint storage.
    """

    def __init__(
        self,
        brain: Any,
        env: Any,
        grounding: Any = None,
        extractor: ClaudeExtractor | None = None,
        on_step: Any = None,
        checkpoint_dir: str = "/data/checkpoints",
    ):
        self.brain = brain
        self.env = env
        self.grounding = grounding
        self.extractor = extractor
        self.on_step = on_step
        self.store = CheckpointStore(base_dir=checkpoint_dir)
        self._seen_urls: set[str] = set()

    # -- Public API -----------------------------------------------------------

    def run_listing(
        self,
        iteration: int,
        page: int,
        page_iteration: int,
        checkpoint: Checkpoint | None = None,
    ) -> SubPlanResult:
        """Run the FIND -> CLICK -> READ_URL -> SCROLL -> EXTRACT -> RETURN pipeline.

        Args:
            iteration: Global iteration number (across all pages).
            page: Current page number.
            page_iteration: 1-based listing index within this page.
            checkpoint: Resume from a prior checkpoint (optional).

        Returns:
            SubPlanResult with extraction data and updated checkpoint.
        """
        t0 = time.time()
        total_steps = 0
        micro_results: dict[str, RunResult | None] = {}
        session_name = f"page{page}"

        if checkpoint is None:
            checkpoint = Checkpoint(
                iteration=iteration, page=page, step_name=""
            )

        # ── FIND + CLICK combined: click a listing title ────────────────
        # Combined because FIND alone ("look at the page") produces no action.
        # The model needs to actually CLICK something to advance.
        logger.info("  [FIND+CLICK] starting (max_steps=8, grounding=ON)")
        click_result = self._holo3_step(
            "Click a boat listing title. The title is blue text below a boat photo showing Year Make Model. "
            "done(success=true) after clicking.",
            max_steps=8,
            grounding=True,
        )
        micro_results["FIND_CLICK"] = click_result
        total_steps += click_result.total_steps
        checkpoint.step_name = "FIND_CLICK"
        self._save_checkpoint(session_name, checkpoint)

        if not click_result.success:
            # Could be: page exhausted OR gallery trap
            # Check if model mentioned no listings
            summary = ""
            for step in click_result.trajectory:
                if step.action.action_type.value == "done":
                    summary = str(step.action.params.get("summary", "")).lower()
            if "no more" in summary or "no listing" in summary or "footer" in summary:
                logger.info("  [FIND_CLICK] page exhausted")
                return SubPlanResult(
                    iteration=iteration, page=page, success=False, data="",
                    no_more_items=True, steps=total_steps,
                    duration=time.time() - t0, checkpoint=checkpoint,
                    page_exhausted=True, micro_results=micro_results,
                )
            # Gallery trap or other failure — recovery
            logger.warning("  [FIND_CLICK] failed — recovery")
            recovery = self._holo3_step("Press Escape then Alt+Left.", max_steps=3)
            micro_results["recovery"] = recovery
            total_steps += recovery.total_steps
            return SubPlanResult(
                iteration=iteration, page=page, success=False, data="",
                no_more_items=False, steps=total_steps,
                duration=time.time() - t0, checkpoint=checkpoint,
                micro_results=micro_results,
            )

        # ── READ_URL: Claude reads URL from screenshot (API only) ───────
        logger.info("  [READ_URL] extracting URL from screenshot")
        top_screenshot = self.env.screenshot()
        url = ""
        if self.extractor:
            url_data = self.extractor.extract(top_screenshot)
            url = url_data.url if url_data else ""
        checkpoint.url = url
        checkpoint.step_name = "READ_URL"
        self._save_checkpoint(session_name, checkpoint)
        logger.info(f"  [READ_URL] url={url[:80]}" if url else "  [READ_URL] no URL found")

        # ── DEDUP check ─────────────────────────────────────────────────
        if url and url in self._seen_urls:
            logger.info("  [DEDUP] duplicate URL, skipping to RETURN")
            checkpoint.dedup = True
            return_result = self._holo3_step(
                "Press Alt+Left to go back.",
                max_steps=3,
            )
            micro_results["RETURN_dedup"] = return_result
            total_steps += return_result.total_steps
            checkpoint.step_name = "RETURN"
            checkpoint.back_on_results = True
            self._save_checkpoint(session_name, checkpoint)
            return SubPlanResult(
                iteration=iteration,
                page=page,
                success=False,
                data="",
                no_more_items=False,
                steps=total_steps,
                duration=time.time() - t0,
                checkpoint=checkpoint,
                dedup=True,
                url=url,
                micro_results=micro_results,
            )
        if url:
            self._seen_urls.add(url)

        # ── SCROLL: Holo3 scrolls past photos ──────────────────────────
        logger.info("  [SCROLL] starting (max_steps=5)")
        scroll_result = self._holo3_step(
            "Scroll down 5 times.",
            max_steps=5,
        )
        micro_results["SCROLL"] = scroll_result
        total_steps += scroll_result.total_steps
        checkpoint.step_name = "SCROLL"
        self._save_checkpoint(session_name, checkpoint)

        # ── EXTRACT: Claude expands safe controls and reads screenshots ─
        logger.info("  [EXTRACT] reading data from screenshots")
        scrolled_screenshot = self.env.screenshot()
        data: Any = None
        if self.extractor:
            data = self._extract_listing_data_deep(top_screenshot, scrolled_screenshot)
        checkpoint.step_name = "EXTRACT"

        # Store extracted data in checkpoint for persistence
        if data and hasattr(data, "is_viable") and data.is_viable():
            checkpoint.extracted_data = {
                "year": getattr(data, "year", ""),
                "make": getattr(data, "make", ""),
                "model": getattr(data, "model", ""),
                "price": getattr(data, "price", ""),
                "phone": getattr(data, "phone", ""),
                "url": url or getattr(data, "url", ""),
                "seller": getattr(data, "seller", ""),
            }
        self._save_checkpoint(session_name, checkpoint)

        # ── RETURN: navigate back to results page ───────────────────────
        logger.info("  [RETURN] starting (max_steps=3)")
        return_result = self._holo3_step(
            "Press Alt+Left to go back.",
            max_steps=3,
        )
        micro_results["RETURN"] = return_result
        total_steps += return_result.total_steps
        checkpoint.step_name = "RETURN"
        checkpoint.back_on_results = True
        self._save_checkpoint(session_name, checkpoint)

        # ── Build result ────────────────────────────────────────────────
        duration = time.time() - t0
        success = data is not None and hasattr(data, "is_viable") and data.is_viable()
        data_str = data.to_summary() if success else ""

        logger.info(
            f"  Sub-plan complete: success={success}, "
            f"steps={total_steps}, duration={duration:.1f}s"
        )

        return SubPlanResult(
            iteration=iteration,
            page=page,
            success=success,
            data=data_str,
            no_more_items=False,
            steps=total_steps,
            duration=duration,
            checkpoint=checkpoint,
            url=url,
            micro_results=micro_results,
        )

    def _extract_listing_data_deep(self, top_screenshot: Any, scrolled_screenshot: Any = None) -> Any:
        """Capture top, expanded description, and lower detail viewports.

        This keeps the sub-plan path consistent with MicroPlanRunner: Claude
        detects safe expand/phone controls, the browser executes those clicks,
        and final extraction reads the full screenshot set.
        """
        if not self.extractor:
            return None

        screenshots: list[Any] = []
        labels: list[str] = []
        clicked_keys: set[str] = set()
        max_screenshots = 12
        max_viewports = 6

        def add_screenshot(shot: Any, label: str) -> None:
            if shot is not None and len(screenshots) < max_screenshots:
                screenshots.append(shot)
                labels.append(label)

        def capture(label: str) -> Any:
            if len(screenshots) >= max_screenshots:
                return None
            try:
                shot = self.env.screenshot()
                add_screenshot(shot, label)
                return shot
            except Exception as exc:
                logger.warning(f"  [deep-extract] screenshot failed: {exc}")
                return None

        add_screenshot(top_screenshot, "top/contact area before scroll")
        add_screenshot(scrolled_screenshot, "scrolled detail area before expansion")

        try:
            self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Home"}))
            time.sleep(1.5)
        except Exception:
            pass

        for viewport in range(max_viewports):
            shot = capture("top/contact area" if viewport == 0 else f"detail viewport {viewport + 1}")
            if shot is None:
                break

            target = self.extractor.find_listing_content_control(shot)
            if target:
                key = (
                    f"{target.get('action', '')}:{target.get('label', '').lower()}:"
                    f"{target['x'] // 25}:{target['y'] // 25}"
                )
                if key not in clicked_keys:
                    clicked_keys.add(key)
                    try:
                        self.env.step(Action(
                            action_type=ActionType.CLICK,
                            params={"x": target["x"], "y": target["y"]},
                        ))
                        time.sleep(2)
                        capture(
                            f"after {target.get('action', 'expand')} "
                            f"{target.get('label', '')[:40]}"
                        )
                    except Exception as exc:
                        logger.warning(f"  [deep-extract] reveal click failed: {exc}")

            if viewport < max_viewports - 1:
                try:
                    self.env.step(Action(action_type=ActionType.KEY_PRESS, params={"keys": "Page_Down"}))
                    time.sleep(1)
                except Exception:
                    break

        data = self.extractor.extract_multi(screenshots, labels=labels)
        if data and hasattr(data, "is_viable") and data.is_viable():
            return data

        if scrolled_screenshot is not None:
            return self.extractor.extract_full(top_screenshot, scrolled_screenshot)
        if screenshots:
            return self.extractor.extract(screenshots[-1])
        return None

    def run_pagination(self) -> bool:
        """Scroll to bottom and click the Next page button.

        Called by WorkflowRunner when a page is exhausted.
        Returns True if pagination succeeded.
        """
        logger.info("  [PAGINATE] starting (max_steps=10)")
        result = self._holo3_step(
            "Scroll to the bottom of the page. Click the Next page button.",
            max_steps=10,
        )
        if result.success:
            # Scroll to top of the new page so FIND starts clean
            try:
                self.env.step(
                    Action(
                        action_type=ActionType.KEY_PRESS,
                        params={"keys": "Home"},
                    )
                )
            except Exception as exc:
                logger.warning(f"  [PAGINATE] Home key failed: {exc}")
        logger.info(f"  [PAGINATE] success={result.success}")
        return result.success

    # -- Private helpers ------------------------------------------------------

    def _holo3_step(
        self,
        intent: str,
        max_steps: int = 5,
        grounding: bool = False,
    ) -> RunResult:
        """Run a single micro-step with a fresh GymRunner.

        Each call creates a new runner (zero context accumulation).
        The intent is a 1-sentence instruction for the CUA model.

        Args:
            intent: Short, specific, positive instruction.
            max_steps: Maximum steps for this micro-step.
            grounding: Whether to enable click grounding.

        Returns:
            RunResult from the bounded GymRunner execution.
        """
        runner = GymRunner(
            brain=self.brain,
            env=self.env,
            max_steps=max_steps,
            frames_per_inference=1,
            grounding=self.grounding if grounding else None,
            on_step=self.on_step,
        )
        task_id = intent[:20].replace(" ", "_")
        return runner.run(task=intent, task_id=task_id)

    def _save_checkpoint(self, session_name: str, checkpoint: Checkpoint) -> None:
        """Persist checkpoint to disk. Non-fatal on failure."""
        checkpoint.timestamp = time.time()
        try:
            self.store.save(session_name, checkpoint)
        except Exception as exc:
            logger.warning(f"Checkpoint save failed: {exc}")
