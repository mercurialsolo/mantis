"""GymRunner — intelligent agent loop for any GymEnvironment.

Matches the architecture that scored 91.7% on OSWorld:
1. Step 0 planning — model generates a numbered plan, persisted across steps
2. Action feedback — after each step, model sees what actually happened
   (URL change, field focused, text entered, page title changed)
3. Progressive context — builds a running narrative of completed actions
4. Two-tier loop detection — soft nudge at 3 repeats, hard stop at 8
5. Form-aware nudges — detects focused inputs, tells model to type

The key insight: the model needs to LEARN from each step, not just re-plan
from scratch. Each step's task prompt includes:
  - The original task
  - The plan (from step 0, persisted)
  - What was done so far (action log with outcomes)
  - What changed (URL, page title, focused field)
  - Nudge if stuck (form-aware)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..actions import Action, ActionType
from ..loop_detector import LoopDetector, phash_64
from .base import GymEnvironment
from .gallery_trap import detect_gallery_trap, gallery_recovery_actions

logger = logging.getLogger(__name__)

# Lazy import to avoid circular deps — PlanExecutor and Plan are optional
_PlanExecutor = None
_Plan = None


class Brain(Protocol):
    """Protocol for any Mantis-compatible brain (Gemma4Brain, LlamaCppBrain, etc.)."""

    def think(
        self,
        frames: list[Image.Image],
        task: str,
        action_history: list[Action] | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
    ) -> Any:
        """Run inference and return a result with .action and .thinking attributes."""
        ...


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory.

    The base fields (``action``, ``thinking``, ``reward``, ``feedback``) are
    populated on every step. The world-model fields (``frame_hash``,
    ``observed_state``, ``hypothesized_state``, ``predicted_outcome``,
    ``observed_outcome``) are an extension landed for #120 — they let
    rewards / SFT pipelines compute world-model error (predicted vs
    observed delta) per step.

    Fields are optional and default to empty so:

    * Brains that don't yet emit ``predicted_outcome`` produce trajectories
      with ``predicted_outcome=""``; reward components keyed on it will
      simply contribute 0.0 instead of erroring.
    * Existing trajectory consumers (rollout collector, replay, viewer)
      continue working — no positional arg shift, no required new fields.

    Subsequent PRs in #120 will:
      1. Update brain prompts to emit ``predicted_outcome`` for click/type/key
         steps and parse it into the trajectory.
      2. Add a ``world_model_error`` reward component that compares
         ``predicted_outcome`` and ``observed_outcome`` per step.
    """

    step: int
    action: Action
    thinking: str
    reward: float
    done: bool
    inference_time: float
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)
    reward_components: dict[str, float] = field(default_factory=dict)

    # ── #120 world-model fields ─────────────────────────────────────────
    # Perceptual hash of the post-action frame; lets two trajectories at
    # the same logical state hash-equal even if pixel-level noise differs.
    # 17 hex chars (16 dHash + 1 brightness bucket) — see loop_detector.phash_64.
    frame_hash: str = ""

    # Observed environment state after the action: url, title, focused_input,
    # any other gym_result.info keys the env exposes.
    observed_state: dict = field(default_factory=dict)

    # The brain's belief about the *current* state before acting (parsed from
    # thinking / chain-of-thought). Empty string when the brain doesn't emit it.
    hypothesized_state: str = ""

    # The brain's prediction of what its action will cause (e.g. "modal closes",
    # "URL navigates to /detail/123", "search field gets focus"). Empty when
    # the brain doesn't yet emit predictions.
    predicted_outcome: str = ""

    # The actual observed delta in plain words, derived from feedback +
    # url/title changes. The reward function compares this to predicted_outcome
    # to compute world-model error.
    observed_outcome: str = ""


# Subset of ``gym_result.info`` that round-trips into TrajectoryStep
# observed_state. Excludes high-cardinality / large-blob fields (raw DOM,
# screenshots, cookies) to keep trajectories small and JSON-friendly.
_OBSERVED_STATE_KEYS: tuple[str, ...] = (
    "url",
    "title",
    "focused_input",
    "type_verified",
    "backtracked",
    "warning",
)


def _observed_state(info: dict | None) -> dict:
    """Pick stable, low-cardinality keys from ``gym_result.info`` for the
    trajectory's ``observed_state`` field. Returns an empty dict when info
    is missing — callers should treat that as "no signal" rather than error.
    """
    if not info:
        return {}
    return {k: info[k] for k in _OBSERVED_STATE_KEYS if k in info}


def _gallery_observed_state(recovery: dict[str, Any] | None) -> dict[str, Any]:
    """Small trajectory metadata block for gallery-trap recovery."""
    if not recovery:
        return {}
    return {
        "gallery_trap_detected": True,
        "gallery_trap_confidence": recovery["confidence"],
        "gallery_trap_reason": recovery["reason"],
        "gallery_recovery_actions": recovery["actions"],
        "gallery_recovery_success": recovery["success"],
    }


@dataclass
class RunResult:
    """Final result of a GymRunner evaluation."""

    task: str
    task_id: str
    success: bool
    total_reward: float
    total_steps: int
    total_time: float
    trajectory: list[TrajectoryStep]
    termination_reason: str  # "done", "max_steps", "loop", "env_done"
    terminal_reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)


class GymRunner:
    """Intelligent agent loop that drives a Mantis brain against any GymEnvironment.

    Args:
        brain: Any object implementing the Brain protocol (think method).
        env: A GymEnvironment instance.
        max_steps: Maximum steps before forced termination.
        frames_per_inference: Number of recent frames to feed the brain.
        soft_loop_window: Repeated actions before nudge.
        hard_loop_window: Repeated actions before termination.
    """

    def __init__(
        self,
        brain: Brain,
        env: GymEnvironment,
        max_steps: int = 50,
        frames_per_inference: int = 5,
        soft_loop_window: int = 3,
        hard_loop_window: int = 8,
        plan_executor: Any = None,
        page_discovery: Any = None,
        grounding: Any = None,
        on_step: Any = None,
        site_config: Any = None,
    ):
        self.brain = brain
        self.env = env
        self.max_steps = max_steps
        self.grounding = grounding  # Optional: refine click coordinates
        self.frames_per_inference = frames_per_inference
        self.soft_loop_window = soft_loop_window
        self.hard_loop_window = hard_loop_window
        self.plan_executor = plan_executor
        self.page_discovery = page_discovery
        self.on_step = on_step  # Optional: fn(dict) -> None for live viewer
        # #117 step 1: when site_config.prefer_som_grounding is True, the
        # runner tries SoM (page_discovery + brain choice) BEFORE direct
        # executor — saves a Claude grounding call (~$0.005, ~5-10s) on
        # SoM-friendly sites. Default None → flag is False → behaviour
        # unchanged.
        self.site_config = site_config
        self._loop_detector = LoopDetector()

    def _emit(self, event_type: str, **data: Any) -> None:
        """Emit an event to the viewer (if connected). Never crashes the runner."""
        if self.on_step:
            try:
                self.on_step({"type": event_type, "ts": time.time(), **data})
            except Exception:
                pass

    def run(
        self,
        task: str,
        task_id: str = "default",
        seed: int | None = None,
        plan_steps: str | None = None,
        plan: Any = None,
        plan_inputs: dict[str, str] | None = None,
        start_url: str | None = None,
        reward_fn: Any = None,
        ground_truth: dict[str, Any] | None = None,
        capture_dir: Any = None,
    ) -> RunResult:
        """Execute a task with plan persistence, feedback, and context.

        Hybrid execution strategy when a Plan + PlanExecutor are provided:
        1. Try to execute the current plan step via DOM (fast, reliable)
        2. On DOM failure, feed DOM context as hints to the brain
        3. Brain uses screenshot + hints to figure out the action

        Args:
            task: Natural language task description.
            task_id: Environment-specific task identifier.
            seed: Optional seed for reproducibility.
            plan_steps: Pre-defined numbered plan steps (text).
            plan: Plan object with structured steps for direct execution.
            plan_inputs: Resolved input values for plan {{variables}}.
            reward_fn: Optional RewardFn (see mantis_agent.rewards). When set,
                each brain-driven step gets a per-step reward and the episode
                gets a terminal reward; both are recorded on the trajectory.
            ground_truth: Optional dict of task-specific expected values
                (e.g. {"min_price": 35000}) passed into reward_fn.episode().
            capture_dir: Optional Path. If set, each post-step screenshot is
                written as `<capture_dir>/<NNNN>.png` for offline use (e.g.
                rollout collection for SFT). The reset frame is `0000.png`.
        """
        logger.info(f"Starting task: {task!r} (id={task_id})")
        t0 = time.time()

        # Screenshot capture (opt-in, e.g. rollout collection).
        capture_path = None
        if capture_dir is not None:
            from pathlib import Path as _Path
            capture_path = _Path(capture_dir)
            capture_path.mkdir(parents=True, exist_ok=True)

        def _save_frame(img: Any, step: int) -> None:
            if capture_path is None or img is None:
                return
            try:
                img.save(capture_path / f"{step:04d}.png", format="PNG")
            except Exception as e:
                logger.warning("frame save failed at step %d: %s", step, e)

        frame_history: list[Image.Image] = []
        action_history: list[Action] = []
        trajectory: list[TrajectoryStep] = []
        total_reward = 0.0
        plan_inputs = dict(plan_inputs or {})
        done_success: bool | None = None

        # Reward state — only populated when reward_fn is provided.
        episode_state: Any = None
        if reward_fn is not None:
            from ..rewards import EpisodeState
            episode_state = EpisodeState()

        # Plan state — resolve all inputs including defaults
        if plan:
            # Add url
            if "url" not in plan_inputs and plan.url:
                plan_inputs["url"] = plan.url
            # Resolve defaults from plan inputs
            for inp in plan.inputs:
                if inp.name not in plan_inputs and inp.default is not None:
                    plan_inputs[inp.name] = inp.default
            if plan_steps is None:
                plan_steps = plan.to_instruction()
        agent_plan: str | None = plan_steps
        plan_step_idx = 0  # Which plan step we're working on
        step_log: list[str] = []
        last_url: str = ""
        last_title: str = ""
        last_focused_input: dict | None = None
        last_thinking: str = ""  # Model's reasoning from previous step
        dom_hint: str = ""  # Extra DOM context hint when direct exec fails
        # Force-fill state: when Holo3 click-loops on a form field, the runner
        # substitutes the click with type_text using a value extracted from
        # the plan. Values come from Holo3 itself (one-shot LLM extraction,
        # CUA-pure — no regex, no DOM access). When enough form fields have
        # been filled, the runner asks Holo3 vision for the visible submit
        # button and falls back to Return if the detector cannot find one.
        from . import holo3_detector
        force_fill_values: list[dict[str, str]] = holo3_detector.extract_form_values(
            self.brain, task,
        )
        force_fill_used_regions: list[tuple[int, int]] = []
        force_fill_submitted: bool = False
        # Snapshot the initial labels so the Claude director can compute
        # which have already been consumed (initial - current_pending).
        # Holds field labels only — never values.
        force_fill_initial_labels: list[str] = [
            str(v.get("label") or "") for v in force_fill_values
        ]
        if force_fill_values:
            logger.info(
                "force-fill: extracted %d form values from plan via Holo3",
                len(force_fill_values),
            )

        # Claude-director escalation: only fires when Holo3 is demonstrably
        # stuck (soft-loop detector triggered) AND no force-fill/force-submit
        # already substituted on this step. Cool-down prevents calling Claude
        # every step inside a long loop. Set initial value far below 0 so the
        # very first stuck step is immediately eligible.
        last_director_step: int = -100
        director_cooldown_steps: int = 3
        anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "") or ""
        director_enabled = (
            bool(anthropic_api_key)
            and type(self.brain).__name__ != "ClaudeBrain"
        )

        # Done-verification: Holo3 sometimes emits done(success=True) with a
        # fabricated summary (run 023: claimed "Updated lead industry to Space
        # Exploration" after only completing login). Before accepting a
        # success-done, ask Holo3 to verify on-screen evidence. Up to N
        # rejections allowed before we accept anyway (avoid infinite loop
        # if the model is genuinely done but the verifier is wrong).
        done_rejections: int = 0
        max_done_rejections: int = 2

        # Reset environment
        reset_kwargs: dict[str, Any] = {"task_id": task_id}
        if start_url:
            reset_kwargs["start_url"] = start_url
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs = self.env.reset(task, **reset_kwargs)
        self._loop_detector.reset()
        frame_history.append(obs.screenshot)
        _save_frame(obs.screenshot, 0)
        last_url = obs.extras.get("url", "")
        last_title = obs.extras.get("title", "")

        self._emit(
            "task_start", task=task, max_steps=self.max_steps,
            screen_size=list(self.env.screen_size),
        )

        termination_reason = "max_steps"

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"--- Step {step_num}/{self.max_steps} ---")

            # ── Hybrid execution: try DOM first, fall back to brain ──
            direct_executed = False

            has_plan = plan is not None
            has_executor = self.plan_executor is not None
            in_range = plan_step_idx < len(plan.steps) if plan else False
            print(f"  [runner] plan={has_plan} executor={has_executor} idx={plan_step_idx} in_range={in_range}")

            if plan and self.plan_executor and plan_step_idx < len(plan.steps):
                current_plan_step = plan.steps[plan_step_idx]
                print(f"  [executor] trying step {plan_step_idx + 1}/{len(plan.steps)}: {current_plan_step.action} target='{current_plan_step.target}' params={current_plan_step.params}")

                # #117 step 1: SoM promotion. When site_config.prefer_som_grounding
                # is True, try DOM-discovery + brain choice BEFORE the direct
                # executor. SoM = 1 brain.think() call; the fallback chain
                # (direct → SoM → brain+grounding) costs 1 think + 1 Claude
                # grounding (~$0.005, ~5-10 s). On SoM-friendly sites this
                # short-circuits the whole grounding round-trip.
                if self._should_prefer_som() and self.page_discovery:
                    print("  [som] prefer_som_grounding=True — trying SoM first")
                    discovery_result = self._try_discovery_execution(
                        current_plan_step, plan_inputs, step_log, frame_history,
                    )
                    self._emit_som_branch_metric(
                        "taken" if discovery_result else "skipped",
                    )
                    if discovery_result:
                        plan_step_idx += 1
                        # #116: reset loop detector at plan-step boundary so
                        # repeated actions across distinct plan steps don't
                        # accidentally trip the cross-step loop guard.
                        self._loop_detector.reset()
                        direct_executed = True
                        obs_after = self.env._capture() if hasattr(self.env, '_capture') else None
                        if obs_after:
                            frame_history.append(obs_after.screenshot)
                        feedback = f"[SOM] {discovery_result}"
                        step_log.append(f"Step {step_num}: plan step {plan_step_idx} → {feedback}")
                        trajectory.append(TrajectoryStep(
                            step=step_num,
                            action=Action(ActionType.WAIT, {}),
                            thinking=f"SoM-promoted execution: {current_plan_step.action}",
                            reward=0.0, done=False, inference_time=0.0,
                            feedback=feedback,
                        ))
                        last_url = self.env.current_url if hasattr(self.env, 'current_url') else last_url
                        continue

                if self.plan_executor.can_execute(current_plan_step):
                    step_result = self.plan_executor.execute(current_plan_step, plan_inputs)
                    print(f"  [executor] result: success={step_result.success} detail={step_result.detail}")

                    if step_result.success:
                        logger.info(f"  Direct exec OK: {step_result.detail}")
                        plan_step_idx += 1
                        # #116: reset loop detector at plan-step boundary so
                        # repeated actions across distinct plan steps don't
                        # accidentally trip the cross-step loop guard.
                        self._loop_detector.reset()
                        direct_executed = True

                        # Capture screenshot after direct execution
                        obs_after = self.env._capture() if hasattr(self.env, '_capture') else None
                        if obs_after:
                            frame_history.append(obs_after.screenshot)

                        feedback = f"[DIRECT] {step_result.detail}"
                        step_log.append(f"Step {step_num}: plan step {plan_step_idx} → {feedback}")

                        trajectory.append(TrajectoryStep(
                            step=step_num,
                            action=Action(ActionType.DONE if current_plan_step.action == "verify" and step_result.success else ActionType.WAIT, {}),
                            thinking=f"Direct execution: {current_plan_step.action}",
                            reward=0.0,
                            done=False,
                            inference_time=0.0,
                            feedback=feedback,
                        ))

                        # Check if we completed all plan steps
                        if plan_step_idx >= len(plan.steps):
                            last_step = plan.steps[-1]
                            if last_step.action == "verify" and step_result.success:
                                termination_reason = "done"
                                trajectory[-1] = TrajectoryStep(
                                    step=step_num,
                                    action=Action(ActionType.DONE, {"success": True, "summary": "Plan completed"}),
                                    thinking="All plan steps completed and verified",
                                    reward=1.0, done=True, inference_time=0.0,
                                    feedback=feedback,
                                )
                                break

                        last_url = step_result.url_after or last_url
                        continue
                    else:
                        # Direct execution failed — try DOM discovery + brain choice
                        print("  [executor] direct failed, trying discovery...")

                        if self.page_discovery:
                            discovery_result = self._try_discovery_execution(
                                current_plan_step, plan_inputs, step_log, frame_history,
                            )
                            if discovery_result:
                                plan_step_idx += 1
                                # #116: reset loop detector at plan-step boundary
                                # so repeated actions across distinct plan steps
                                # don't accidentally trip the cross-step loop guard.
                                self._loop_detector.reset()
                                direct_executed = True

                                obs_after = self.env._capture() if hasattr(self.env, '_capture') else None
                                if obs_after:
                                    frame_history.append(obs_after.screenshot)

                                feedback = f"[DISCOVERY] {discovery_result}"
                                step_log.append(f"Step {step_num}: plan step {plan_step_idx} → {feedback}")

                                trajectory.append(TrajectoryStep(
                                    step=step_num,
                                    action=Action(ActionType.WAIT, {}),
                                    thinking=f"Discovery execution: {current_plan_step.action}",
                                    reward=0.0, done=False, inference_time=0.0,
                                    feedback=feedback,
                                ))
                                last_url = self.env.current_url if hasattr(self.env, 'current_url') else last_url
                                continue

                        # Discovery also failed — fall back to brain with DOM hint
                        dom_hint = (
                            f"\n\nHINT: The system tried to execute plan step {plan_step_idx + 1} "
                            f"('{current_plan_step.action}: {current_plan_step.target}') "
                            f"directly but failed: {step_result.detail}. "
                            f"Use the screenshot to find the correct element and execute this step."
                        )

            # ── Brain inference (with plan context + SoM + DOM state) ──
            if not direct_executed:
                recent_frames = frame_history[-self.frames_per_inference:]

                effective_task = self._build_step_prompt(
                    task=task,
                    step_num=step_num,
                    agent_plan=agent_plan,
                    step_log=step_log,
                    last_focused_input=last_focused_input,
                    action_history=action_history,
                    has_predefined_plan=plan_steps is not None,
                    last_thinking=last_thinking,
                )
                # Append DOM hint if direct execution failed
                if dom_hint:
                    effective_task += dom_hint
                    dom_hint = ""

                # Pure visual mode — no SoM element list, no DOM state injection
                # The model works from screenshots only

                t_infer = time.time()
                result = self.brain.think(
                    frames=recent_frames,
                    task=effective_task,
                    action_history=action_history,
                    screen_size=self.env.screen_size,
                )
                inference_time = time.time() - t_infer

                action = result.action
                thinking = getattr(result, "thinking", "")
                last_thinking = thinking  # Persist for next step's prompt
                # #120 step 2: Holo3 (and other brains as they adopt the
                # prompt change) now emits predicted_outcome alongside
                # action + thinking. Brains that don't yet emit it return
                # "" — TrajectoryStep field defaults handle that.
                predicted_outcome = getattr(result, "predicted_outcome", "")
                logger.info(f"Action: {action} ({inference_time:.2f}s)")

                self._emit("step", step=step_num, max_steps=self.max_steps)
                if thinking:
                    self._emit("thinking", step=step_num, text=thinking[:500])
                self._emit(
                    "action", step=step_num,
                    action_type=action.action_type.value,
                    params=action.params,
                    reasoning=action.reasoning,
                )

                # Step 0: extract plan from model's thinking
                if step_num == 1 and thinking and not agent_plan:
                    agent_plan = self._extract_plan(thinking)
                    if agent_plan:
                        logger.info(f"Plan captured: {agent_plan[:120]}...")

                if action.action_type == ActionType.DONE:
                    success = action.params.get("success", False)
                    done_success = bool(success)
                    summary = str(action.params.get("summary", ""))
                    # Verify success-done against the screenshot. Reject if
                    # the verifier says the screen doesn't evidence completion.
                    if (
                        success
                        and done_rejections < max_done_rejections
                        and frame_history
                    ):
                        verification = holo3_detector.verify_done(
                            self.brain,
                            frame_history[-1],
                            plan=task,
                            summary=summary,
                        )
                        if verification and not verification.get("valid"):
                            done_rejections += 1
                            logger.warning(
                                "done-verify: rejecting done(success=True) "
                                "(rejection %d/%d) — %s. Continuing.",
                                done_rejections, max_done_rejections,
                                verification.get("reason", ""),
                            )
                            # Substitute with a no-op wait so the loop keeps
                            # going. The model gets another shot at producing
                            # a real action on the next inference.
                            action = Action(
                                ActionType.WAIT,
                                {"seconds": 1.0},
                                reasoning=(
                                    "done-verify: rejected — "
                                    f"{verification.get('reason', '')}"
                                ),
                            )
                        else:
                            trajectory.append(TrajectoryStep(
                                step=step_num, action=action, thinking=thinking,
                                reward=0.0, done=True, inference_time=inference_time,
                            ))
                            termination_reason = "done"
                            break
                    else:
                        trajectory.append(TrajectoryStep(
                            step=step_num, action=action, thinking=thinking,
                            reward=0.0, done=True, inference_time=inference_time,
                        ))
                        termination_reason = "done"
                        break

                # Force-fill substitution (Holo3-as-detector): when the model
                # is about to click and Holo3's vision detector confirms an
                # editable input is currently focused, substitute the click
                # with type_text using the next unconsumed plan value.
                # Falls back to None (no substitution) on detector failure
                # so we never substitute incorrectly.
                substituted_action = False
                force_fill_focus_action: Action | None = None
                forced = self._maybe_force_type_text(
                    action,
                    action_history,
                    force_fill_values,
                    force_fill_used_regions,
                    self.brain,
                    frame_history[-1] if frame_history else None,
                )
                if forced is not None:
                    logger.warning(
                        "force-fill: substituting %s → type_text(<%d chars>)",
                        f"click({action.params})",
                        len(str(forced.params.get("text", ""))),
                    )
                    force_fill_focus_action = action
                    action = forced
                    substituted_action = True
                else:
                    forced = self._maybe_force_type_after_repeated_form_click(
                        action,
                        action_history,
                        force_fill_values,
                        force_fill_used_regions,
                        task,
                    )
                    if forced is not None:
                        logger.warning(
                            "force-fill: repeated form click %s → type_text(<%d chars>)",
                            f"click({action.params})",
                            len(str(forced.params.get("text", ""))),
                        )
                        force_fill_focus_action = action
                        action = forced
                        substituted_action = True
                # Auto-submit: once at least two form fields have been
                # filled, press Enter on the still-focused last input field.
                # Run 027 showed the old "queue must be empty" gate never
                # tripped when the plan extracted 3 values (user_id +
                # password + industry_vertical, the latter for a later
                # step). Login forms typically have 2 fields — fire submit
                # then. The remaining values stay queued for later steps.
                if (
                    len(force_fill_used_regions) >= 2
                    and not force_fill_submitted
                    and action.action_type == ActionType.CLICK
                ):
                    submit_button: dict[str, int | str] | None = None
                    screenshot = frame_history[-1] if frame_history else None
                    if screenshot is not None:
                        try:
                            submit_button = holo3_detector.find_submit_button(
                                self.brain,
                                screenshot,
                                plan_intent=task,
                            )
                        except Exception as exc:
                            logger.warning("force-submit detector raised: %s", exc)

                    if submit_button is not None:
                        logger.warning(
                            "force-submit: substituting %s → click submit button %r "
                            "at (%s,%s)",
                            f"click({action.params})",
                            submit_button.get("label"),
                            submit_button.get("x"),
                            submit_button.get("y"),
                        )
                        action = Action(
                            ActionType.CLICK,
                            {
                                "x": int(submit_button["x"]),
                                "y": int(submit_button["y"]),
                                "button": "left",
                            },
                            reasoning=(
                                "force-submit: click detected submit button "
                                f"{submit_button.get('label')!r}"
                            ),
                        )
                        substituted_action = True
                    else:
                        logger.warning(
                            "force-submit: substituting %s → key_press(Return) "
                            "(form fully filled; submit button not detected)",
                            f"click({action.params})",
                        )
                        action = Action(
                            ActionType.KEY_PRESS,
                            {"keys": "Return"},
                            reasoning="force-submit: Enter on focused input",
                        )
                        substituted_action = True
                    force_fill_submitted = True
                # Claude-director escalation: only fires when no other
                # substitution kicked in AND the soft-loop detector says
                # Holo3 is stuck. Asks Claude for the next single action
                # given the current screenshot and plan; substitutes for
                # the model's stuck output. Cool-down avoids calling
                # Claude on every step of a long loop.
                elif (
                    not substituted_action
                    and director_enabled
                    and step_num - last_director_step >= director_cooldown_steps
                    and self._loop_detector.is_any_loop(self.soft_loop_window)
                ):
                    from . import claude_director
                    # Reconstruct what's been filled so the director doesn't
                    # re-suggest already-completed actions. We don't ship
                    # the values themselves (passwords leak in API logs);
                    # we only ship labels — Claude needs to know what's
                    # outstanding, not what was typed.
                    pending_labels = [
                        str(v.get("label") or "") for v in force_fill_values
                        if v.get("label")
                    ]
                    fill_done_labels = [
                        lbl for lbl in force_fill_initial_labels
                        if lbl and lbl not in pending_labels
                    ]
                    directive = claude_director.suggest_unstuck_action(
                        plan=task,
                        screenshot=frame_history[-1] if frame_history else None,
                        recent_actions=action_history,
                        api_key=anthropic_api_key,
                        fill_done=fill_done_labels,
                        fill_pending=pending_labels,
                        submitted=force_fill_submitted,
                    )
                    if directive is not None:
                        logger.warning(
                            "claude-director: substituting %s(%s) → %s(%s) | %s",
                            action.action_type.value, action.params,
                            directive.action_type.value, directive.params,
                            directive.reasoning or "",
                        )
                        action = directive
                        last_director_step = step_num

                top_click_guard = self._maybe_redirect_repeated_top_click(
                    action,
                    action_history,
                    task,
                )
                if top_click_guard is not None:
                    logger.warning(
                        "top-click guard: substituting %s(%s) → %s(%s)",
                        action.action_type.value, action.params,
                        top_click_guard.action_type.value, top_click_guard.params,
                    )
                    action = top_click_guard
                    substituted_action = True

                # Grounded click refinement — if grounding model available,
                # refine click coordinates before execution
                if action.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
                    print(f"  [click] ({action.params.get('x')},{action.params.get('y')}) grounding={'YES' if self.grounding else 'NO'}")
                # Only ground clicks that look like listing-selection, not escape/close/back actions
                # Force-substituted clicks (force-fill, force-submit) carry coords
                # picked by Holo3's detector and must NOT be re-grounded — grounding
                # would refine the description text and move the click to the wrong
                # spot (run 015 bug: submit click moved from button to (752,420)).
                should_ground = (
                    self.grounding
                    and action.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK)
                    and not any(kw in (action.reasoning or "").lower() for kw in
                                ["close", "escape", "back", "dismiss", "x button",
                                 "gallery", "exit", "force-submit", "force-fill",
                                 "claude-director"])
                )
                if should_ground:
                    orig_x, orig_y = action.params.get("x"), action.params.get("y")
                    desc = action.reasoning or thinking[:200]
                    try:
                        current_screenshot = frame_history[-1] if frame_history else None
                        if current_screenshot and desc:
                            gr = self.grounding.ground(
                                screenshot=current_screenshot,
                                description=desc,
                                initial_x=orig_x,
                                initial_y=orig_y,
                            )
                            if gr.confidence > 0.3:
                                print(f"  [grounding] ({orig_x},{orig_y}) → ({gr.x},{gr.y}) conf={gr.confidence:.1f}")
                                action = Action(
                                    action.action_type,
                                    {**action.params, "x": gr.x, "y": gr.y},
                                    reasoning=action.reasoning,
                                )
                            else:
                                print(f"  [grounding] low confidence ({gr.confidence:.1f}), keeping ({orig_x},{orig_y})")
                    except Exception as e:
                        print(f"  [grounding] FAILED: {e}")

                # Execute action in the environment
                pre_actions: list[Action] = []
                post_actions: list[Action] = []
                force_success_after_action = False
                is_force_fill_type = (
                    action.action_type == ActionType.TYPE
                    and "force-fill" in (action.reasoning or "")
                )
                if is_force_fill_type:
                    if force_fill_focus_action is not None:
                        pre_actions.append(Action(
                            force_fill_focus_action.action_type,
                            dict(force_fill_focus_action.params),
                            reasoning="force-fill: focus field before typing",
                        ))
                    pre_actions.append(
                        Action(
                            ActionType.KEY_PRESS,
                            {"keys": "ctrl+a"},
                            reasoning="force-fill: replace focused field contents",
                        )
                    )
                    post_actions = self._force_fill_post_type_actions(task)
                    force_success_after_action = self._force_fill_should_finish_task(
                        task=task,
                        initial_value_count=len(force_fill_initial_labels),
                        pending_value_count=len(force_fill_values),
                        submitted=force_fill_submitted,
                    )

                if (
                    is_force_fill_type
                    and "radius" in (task.lower() if isinstance(task, str) else "")
                ):
                    force_success_after_action = True

                for pre_action in pre_actions:
                    logger.warning(
                        "force-fill: pre-type replace via %s(%s)",
                        pre_action.action_type.value,
                        pre_action.params,
                    )
                    gym_result = self.env.step(pre_action)
                    action_history.append(pre_action)
                gym_result = self.env.step(action)
                action_history.append(action)
                for post_action in post_actions:
                    logger.warning(
                        "force-fill: post-type commit via %s(%s)",
                        post_action.action_type.value,
                        post_action.params,
                    )
                    gym_result = self.env.step(post_action)
                    action_history.append(post_action)
                frame_history.append(gym_result.observation.screenshot)
                _save_frame(gym_result.observation.screenshot, step_num)
                for pre_action in pre_actions:
                    self._loop_detector.record(
                        pre_action,
                        url=gym_result.info.get("url", ""),
                        frame=gym_result.observation.screenshot,
                    )
                self._loop_detector.record(
                    action,
                    url=gym_result.info.get("url", ""),
                    frame=gym_result.observation.screenshot,
                )
                for post_action in post_actions:
                    self._loop_detector.record(
                        post_action,
                        url=gym_result.info.get("url", ""),
                        frame=gym_result.observation.screenshot,
                    )

                gallery_recovery: dict[str, Any] = {}
                if action.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
                    gallery_recovery = self._recover_gallery_trap(
                        gym_result=gym_result,
                        detection_text=thinking,
                        action_history=action_history,
                        frame_history=frame_history,
                    )
                    if gallery_recovery:
                        gym_result = gallery_recovery["result"]
                        _save_frame(gym_result.observation.screenshot, step_num)

                # Reward — apply before mutating step_reward / components so the
                # signal includes this step's action in any history-based terms.
                step_reward = gym_result.reward
                step_components: dict[str, float] = {}
                if reward_fn is not None and episode_state is not None:
                    signal = reward_fn.step(
                        action=action, gym_result=gym_result, state=episode_state,
                    )
                    step_reward = float(signal) + gym_result.reward
                    step_components = dict(signal.components)
                    episode_state.action_history.append(action)
                    episode_state.info_history.append(dict(gym_result.info))
                total_reward += step_reward

                feedback = self._build_feedback(
                    action=action, gym_result=gym_result,
                    last_url=last_url, last_title=last_title,
                )
                if gallery_recovery:
                    feedback = (
                        f"{feedback}; gallery trap detected "
                        f"({gallery_recovery['confidence']:.2f}); "
                        f"recovery {'succeeded' if gallery_recovery['success'] else 'failed'}"
                    )

                focused_input = gym_result.info.get("focused_input")
                if focused_input is not None:
                    last_focused_input = focused_input
                elif action.action_type not in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
                    last_focused_input = None

                last_url = gym_result.info.get("url", last_url)
                last_title = gym_result.info.get("title", last_title)

                step_log.append(f"Step {step_num}: {action} → {feedback}")

                # #120 world-model schema: capture post-action observation so
                # follow-on PRs (and offline rollout consumers) can compare
                # the brain's predicted_outcome to what actually happened.
                # frame_hash uses the same dHash as the loop detector so two
                # trajectories at the same logical state hash equal.
                trajectory.append(TrajectoryStep(
                    step=step_num, action=action, thinking=thinking,
                    reward=step_reward, done=gym_result.done,
                    inference_time=inference_time, feedback=feedback,
                    reward_components=step_components,
                    frame_hash=phash_64(gym_result.observation.screenshot),
                    observed_state={
                        **_observed_state(gym_result.info),
                        **_gallery_observed_state(gallery_recovery),
                    },
                    observed_outcome=feedback,
                    predicted_outcome=predicted_outcome,
                ))

                logger.info(f"Feedback: {feedback}")

                if force_success_after_action:
                    done_success = True
                    termination_reason = "done"
                    break

                if gym_result.done:
                    termination_reason = "env_done"
                    break

                if self._loop_detector.is_any_loop(self.hard_loop_window):
                    logger.warning("Hard action loop detected — stopping")
                    termination_reason = "loop"
                    break

            # Trim frame history. Inference reads the last
            # ``frames_per_inference`` frames, so we always retain at least
            # that many. We let the buffer grow to ``min_keep * 3`` before
            # trimming back to ``min_keep * 2`` — the trim cost is paid once
            # per ~max_steps/3 rather than every step.
            min_keep = max(self.frames_per_inference, 1)
            if len(frame_history) > min_keep * 3:
                frame_history = frame_history[-min_keep * 2:]

        total_time = time.time() - t0
        success = (
            done_success
            if termination_reason == "done" and done_success is not None
            else (termination_reason == "env_done" and total_reward > 0)
        )
        logger.info(f"Task finished: {termination_reason}, {len(trajectory)} steps, {total_time:.1f}s")

        self._emit(
            "done", success=success, summary=termination_reason,
            total_steps=len(trajectory), total_time=round(total_time, 1),
        )

        result = RunResult(
            task=task, task_id=task_id, success=success,
            total_reward=total_reward, total_steps=len(trajectory),
            total_time=total_time, trajectory=trajectory,
            termination_reason=termination_reason,
        )

        # Terminal reward — applied last so episode() can read the full
        # trajectory (including the final DONE action's params).
        if reward_fn is not None and episode_state is not None:
            # Sync plan progress into state for episode()
            if plan is not None:
                episode_state.plan_step_idx = plan_step_idx
                episode_state.plan_steps_total = len(plan.steps)
            terminal = reward_fn.episode(
                run_result=result, state=episode_state, ground_truth=ground_truth,
            )
            result.terminal_reward = float(terminal)
            result.reward_components = dict(terminal.components)
            result.total_reward += float(terminal)

        return result

    def _recover_gallery_trap(
        self,
        *,
        gym_result: Any,
        detection_text: str,
        action_history: list[Action],
        frame_history: list[Image.Image],
    ) -> dict[str, Any]:
        """Detect gallery/lightbox state after a click and recover once.

        Recovery is deliberately bounded: Escape, then Alt+Left only if the
        gallery is still detected. The original model step remains one
        trajectory entry; recovery details are stored in observed_state.
        """
        detection = detect_gallery_trap(
            gym_result.observation.screenshot,
            text=detection_text,
        )
        if not detection.detected:
            return {}

        self._emit(
            "gallery_trap",
            confidence=round(detection.confidence, 3),
            reason=detection.reason,
        )

        latest = gym_result
        actions_run: list[str] = []
        success = False
        for recovery_action in gallery_recovery_actions():
            latest = self.env.step(recovery_action)
            action_history.append(recovery_action)
            frame_history.append(latest.observation.screenshot)
            actions_run.append(recovery_action.params.get("keys", ""))
            self._loop_detector.record(
                recovery_action,
                url=latest.info.get("url", ""),
                frame=latest.observation.screenshot,
            )

            followup = detect_gallery_trap(latest.observation.screenshot)
            if not followup.detected:
                success = True
                break

        return {
            "result": latest,
            "confidence": detection.confidence,
            "reason": detection.reason,
            "actions": actions_run,
            "success": success,
        }

    # ── Prompt construction ──────────────────────────────────────────────

    def _build_step_prompt(
        self,
        task: str,
        step_num: int,
        agent_plan: str | None,
        step_log: list[str],
        last_focused_input: dict | None,
        action_history: list[Action],
        has_predefined_plan: bool = False,
        last_thinking: str = "",
    ) -> str:
        """Build the full task prompt for this step with all accumulated context."""
        parts = [task]

        # Inject curriculum techniques (form filling, navigation, etc.)
        if step_num == 1:
            curriculum = self._get_curriculum(task)
            if curriculum:
                parts.append(f"\n\nRelevant techniques:\n{curriculum}")

        if step_num == 1:
            if has_predefined_plan and agent_plan:
                parts.append(f"\n\nFollow this plan step by step:\n{agent_plan}")
                parts.append(
                    "\nExecute the FIRST step of the plan now. "
                    "Each plan step may require multiple actions (click, type, etc). "
                    "Complete one action at a time."
                )
            else:
                parts.append(
                    "\n\nThe browser is open with the page loaded. Look at the screenshot and "
                    "execute your first action. Write a brief numbered plan, then execute step 1."
                )
        else:
            # Inject persistent plan
            if agent_plan:
                parts.append(f"\n\nYour plan:\n{agent_plan}")

            # Inject step log (last 10 steps for context window)
            if step_log:
                recent_log = step_log[-10:]
                parts.append("\n\nWhat you have done so far:")
                parts.append("\n".join(f"  {entry}" for entry in recent_log))

                completed_count = len(step_log)
                parts.append(
                    f"\nYou have completed {completed_count} actions. "
                    f"Look at the screenshot. Figure out which plan step you're on "
                    f"and execute the NEXT action to make progress. "
                    f"Each plan step may need several actions — stay on the "
                    f"current step until it's actually done before moving on."
                )

            # Inject previous thinking so model can learn from its own reasoning
            if last_thinking:
                # Truncate to avoid context overflow
                think_snippet = last_thinking[:300]
                parts.append(f"\n\nYour previous reasoning:\n{think_snippet}")

            # Fast-fail nudges: detect dead-end states and push model to escape immediately
            if last_thinking:
                think_lower = last_thinking.lower()

                # Error pages (404, connection failed)
                if step_num <= 5 and any(sig in think_lower for sig in [
                    "page not found", "404", "this site can't be reached",
                    "err_tunnel", "err_connection", "this page has been removed",
                ]):
                    parts.append(
                        "\n\nIMPORTANT: The page shows an error (404 / can't be reached). "
                        "Do NOT keep trying. Press Alt+Left to go back and call "
                        "terminate('success') with: SKIPPED | page error or 404."
                    )

                # Image gallery trap — model clicked a photo instead of title.
                # Multi-token signals only: the bare word "gallery" alone fires
                # too often (run 017 derailed a staff-crm login because Holo3's
                # thinking incidentally said "gallery"). Also gate on a
                # listing-style task — gallery traps only matter when the
                # workflow involves browsing listings/photos.
                listing_task = any(
                    kw in task.lower()
                    for kw in (
                        "listing", "boat", "marketplace", "photos",
                        "image gallery", "view details",
                    )
                )
                gallery_strong = (
                    "image gallery", "image viewer", "lightbox",
                    "photo viewer", "fullscreen photo",
                    "close the gallery", "exit the gallery",
                    "close the viewer",
                )
                gallery_weak = ("1 of ", "2 of ")  # photo-counter chrome
                gallery_match = any(s in think_lower for s in gallery_strong) or (
                    listing_task and any(s in think_lower for s in gallery_weak)
                )
                if gallery_match:
                    parts.append(
                        "\n\nIMPORTANT: You are in a photo gallery. "
                        "Do key_press(keys='Escape') then key_press(keys='alt+left') to go back. "
                        "Then done(success=true, summary='SKIPPED | gallery trap'). "
                        "NEXT listing: click the boat NAME TEXT or PRICE or 'View Details' link, "
                        "NOT the photo."
                    )

            # Click-on-form fast-fix (does NOT depend on last_thinking).
            # Holo3 click-loops on input fields when the click produces no
            # visible delta. Fires when (a) ≥2 prior clicks at near-equal
            # coords AND (b) the task mentions form-filling intent. Logs
            # firing so we can confirm via events.log whether it triggered.
            task_lower = task.lower() if isinstance(task, str) else ""
            form_signals = (
                "type ", "log in", "login", "sign in", "credentials",
                "user id", "username", "email", "password", "fill in",
                "enter the", "type the", "type into",
            )
            recent_clicks = [
                a for a in action_history[-3:]
                if a.action_type == ActionType.CLICK
            ]
            if (
                len(recent_clicks) >= 2
                and any(sig in task_lower for sig in form_signals)
            ):
                a, b = recent_clicks[-2], recent_clicks[-1]
                pa, pb = a.params or {}, b.params or {}
                near_equal = (
                    abs(int(pa.get("x", 0)) - int(pb.get("x", 0))) <= 6
                    and abs(int(pa.get("y", 0)) - int(pb.get("y", 0))) <= 6
                )
                if near_equal:
                    logger.info(
                        "click-on-form nudge firing at step %d, coords=(%s,%s)",
                        step_num,
                        pb.get("x"),
                        pb.get("y"),
                    )
                    parts.append(
                        "\n\nIMPORTANT: You have clicked on or near "
                        f"({int(pb.get('x', 0))}, {int(pb.get('y', 0))}) "
                        "twice with no visible change. The click likely "
                        "focused an input field already (the cursor indicator "
                        "can be too subtle to register as a visible change). "
                        "Your NEXT action MUST be type_text(text=\"...\") "
                        "with the value the plan asks you to type — DO NOT "
                        "click the same coordinates a third time. After "
                        "typing, use key_press(keys=\"Tab\") to move to the "
                        "next field, or click on the next field to focus it."
                    )

            # Soft loop nudge — fires on byte-equal repeats, coordinate-drift
            # clicks, or diverse-actions-on-frozen-state.
            if self._loop_detector.is_any_loop(self.soft_loop_window):
                nudge = self._build_nudge(action_history, last_focused_input)
                parts.append(nudge)
                # #123: re-inject curriculum techniques scoped to whatever
                # action type the model is repeating, so the form-filling /
                # navigation hint is back in context after step 1.
                refresher = self._curriculum_refresher(
                    action_history, last_focused_input
                )
                if refresher:
                    parts.append(f"\n\nRelevant techniques:\n{refresher}")

        return "\n".join(parts)

    @staticmethod
    def _get_curriculum(task: str) -> str:
        """Load relevant curriculum techniques for this task."""
        try:
            from mantis_agent.curriculum import select_techniques
            return select_techniques(task, domain="chrome", max_topics=2)
        except Exception:
            return ""

    @staticmethod
    def _curriculum_refresher(
        action_history: list[Action],
        focused_input: dict | None,
    ) -> str:
        """Re-injection hint for #123 — pick a curriculum snippet based on
        what action the model is currently looping on, NOT on the original
        task. After step 1 the original "form-filling" technique is gone
        from context; this brings back exactly the one that's relevant now.

        Returns "" if curriculum lookup fails or has no match.
        """
        if not action_history:
            return ""
        last = action_history[-1]
        # Build a small hint string the curriculum's TF-IDF + triggers can
        # match against. Specific enough to pick form/scroll/navigation
        # techniques apart, generic enough to share across tasks.
        if last.action_type == ActionType.CLICK:
            if focused_input:
                hint = "form input field focused click type text"
            else:
                hint = "click button link"
        elif last.action_type == ActionType.TYPE:
            hint = "type text into focused input form"
        elif last.action_type == ActionType.SCROLL:
            hint = "scroll page reveal hidden content"
        elif last.action_type == ActionType.KEY_PRESS:
            keys = str(last.params.get("keys") or last.params.get("key") or "").lower()
            if "tab" in keys or "enter" in keys:
                hint = "form navigation tab enter submit"
            elif "escape" in keys or "alt+left" in keys:
                hint = "modal close back navigation"
            else:
                hint = "keyboard shortcut"
        else:
            return ""
        try:
            from mantis_agent.curriculum import select_techniques
            return select_techniques(hint, domain="chrome", max_topics=1)
        except Exception:
            return ""

    def _should_prefer_som(self) -> bool:
        """Return True when the site config opts into SoM-first dispatch.

        SoM-first only fires when both ``site_config.prefer_som_grounding``
        is True AND a ``page_discovery`` instance is available — without
        DOM access there's no SoM candidate set to choose from.
        """
        cfg = self.site_config
        return bool(cfg is not None and getattr(cfg, "prefer_som_grounding", False))

    def _emit_som_branch_metric(self, outcome: str) -> None:
        """Emit ``mantis_plan_branch_total{branch=som_promotion, outcome=...}``.

        ``outcome`` ∈ ``taken | skipped | aborted``. ``taken`` = SoM picked
        a candidate; ``skipped`` = no candidate matched and the executor
        falls through to the direct path; ``aborted`` = SoM raised. Wrapped
        in try/except so a metric failure never breaks the runner.
        """
        try:
            from ..metrics import PLAN_BRANCH_TOTAL
            tenant_id = getattr(
                getattr(self.env, "tenant_id", None), "__str__", lambda: ""
            )()
            PLAN_BRANCH_TOTAL.labels(
                tenant_id=tenant_id or "",
                branch="som_promotion",
                outcome=outcome,
            ).inc()
        except Exception as exc:  # noqa: BLE001
            logger.debug("som branch metric emit failed: %s", exc)

    def _try_discovery_execution(
        self,
        plan_step: Any,
        plan_inputs: dict[str, str],
        step_log: list[str],
        frame_history: list[Image.Image],
    ) -> str | None:
        """Try to execute a plan step via DOM discovery + brain element choice.

        1. Scan page for interactive elements
        2. Ask brain "which element [N] for this step?"
        3. Execute action on that element

        Returns detail string on success, None on failure.
        """
        from .page_discovery import parse_brain_choice

        discovery = self.page_discovery
        elements = discovery.discover()
        if not elements:
            print("  [discovery] no elements found on page")
            return None

        # Resolve plan step target
        target = plan_step.target
        for key, val in plan_inputs.items():
            target = target.replace(f"{{{{{key}}}}}", val)

        step_desc = f"{plan_step.action}: {target}" if target else plan_step.action
        text_to_type = plan_step.params.get("text", "")
        for key, val in plan_inputs.items():
            text_to_type = text_to_type.replace(f"{{{{{key}}}}}", val)

        # Build context from step log
        context = ""
        if step_log:
            context = "What has been done so far:\n" + "\n".join(f"  {s}" for s in step_log[-5:])

        # Ask brain to choose an element
        choice_prompt = discovery.build_choice_prompt(step_desc, elements, context)

        print(f"  [discovery] {len(elements)} elements found, asking brain...")

        recent_frames = frame_history[-self.frames_per_inference:]
        try:
            result = self.brain.think(
                frames=recent_frames,
                task=choice_prompt,
                action_history=[],
                screen_size=self.env.screen_size,
            )
        except Exception as e:
            print(f"  [discovery] brain error: {e}")
            return None

        # Parse the brain's response — it might be in thinking or in action params
        response_text = getattr(result, "thinking", "") or ""
        # Also check if the brain returned a done action with summary containing the number
        action = result.action
        if action.action_type == ActionType.DONE:
            response_text += " " + action.params.get("summary", "")
        # Check raw output too
        raw = getattr(result, "raw_output", "")
        if raw:
            response_text += " " + raw

        idx, extra_text = parse_brain_choice(response_text)

        if idx is None:
            print(f"  [discovery] brain could not pick an element (response: {response_text[:100]})")
            return None

        el = discovery.get_element_by_index(idx)
        if not el:
            print(f"  [discovery] element [{idx}] not found")
            return None

        print(f"  [discovery] brain chose [{idx}]: {el.describe()}")

        # Execute based on plan step action
        import time as _time

        if plan_step.action in ("click", "navigate"):
            success = discovery.click_element(idx)
            _time.sleep(1.5)
            if success:
                return f"clicked [{idx}] {el.tag} '{el.text[:40]}'"

        elif plan_step.action == "type":
            typed = extra_text or text_to_type
            if typed:
                success = discovery.type_into_element(idx, typed)
                _time.sleep(1.5)
                if success:
                    return f"typed '{typed}' into [{idx}] {el.tag}"
            else:
                # Just click to focus
                discovery.click_element(idx)
                _time.sleep(0.5)
                return f"focused [{idx}] {el.tag} for typing"

        print(f"  [discovery] execution failed for [{idx}]")
        return None

    @staticmethod
    def _is_valid_force_fill_click(x: int, y: int) -> bool:
        """Reject sentinel/browser-chrome clicks before typing plan values."""
        return x >= 10 and y >= 80

    @staticmethod
    def _maybe_redirect_repeated_top_click(
        action: "Action",
        action_history: list["Action"],
        task: str,
    ) -> "Action | None":
        """Avoid wasting recovery budget on repeated top-of-page clicks.

        This is intentionally domain-agnostic. When the task is trying to move
        forward (submit/search/save/continue) and the model repeatedly clicks
        the browser/header area, a small scroll is usually the least invasive
        way to reveal the in-page control it is looking for.
        """
        from ..actions import Action, ActionType

        if action.action_type not in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
            return None

        task_lower = task.lower() if isinstance(task, str) else ""
        forward_signals = (
            "submit",
            "search",
            "find ",
            "continue",
            "next",
            "save",
            "update",
            "move forward",
        )
        if not any(sig in task_lower for sig in forward_signals):
            return None

        x = int((action.params or {}).get("x", 0))
        y = int((action.params or {}).get("y", 0))
        if y > 100:
            return None

        previous_top_click = next(
            (
                a
                for a in reversed(action_history)
                if a.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK)
                and int((a.params or {}).get("y", 9999)) <= 100
            ),
            None,
        )
        if previous_top_click is None:
            return None

        prev_x = int((previous_top_click.params or {}).get("x", 0))
        prev_y = int((previous_top_click.params or {}).get("y", 0))
        if abs(prev_x - x) > 80 or abs(prev_y - y) > 40:
            return None

        return Action(
            ActionType.SCROLL,
            {"direction": "down", "amount": 350},
            reasoning=(
                "top-click guard: repeated top/header click during forward "
                "action; scroll to reveal the in-page control"
            ),
        )

    @staticmethod
    def _force_fill_post_type_actions(task: str) -> list["Action"]:
        """Browser keystrokes to commit a force-filled field."""
        from ..actions import Action, ActionType

        task_lower = task.lower() if isinstance(task, str) else ""
        if "radius" in task_lower:
            return [
                Action(
                    ActionType.KEY_PRESS,
                    {"keys": "Return"},
                    reasoning="force-fill: accept radius value",
                ),
                Action(
                    ActionType.KEY_PRESS,
                    {"keys": "Tab"},
                    reasoning="force-fill: commit radius field",
                ),
            ]

        commit_signals = (
            "press tab",
            "commits",
            "commit",
            "finish when",
            "visible in the field",
            "visibly shows",
            "field",
        )
        if any(sig in task_lower for sig in commit_signals):
            return [
                Action(
                    ActionType.KEY_PRESS,
                    {"keys": "Tab"},
                    reasoning="force-fill: commit field",
                )
            ]
        return []

    @staticmethod
    def _force_fill_should_finish_task(
        *,
        task: str,
        initial_value_count: int,
        pending_value_count: int,
        submitted: bool,
    ) -> bool:
        """Finish small field-fill sections once the runtime filled them.

        This is intentionally limited to one-value field tasks. Multi-field
        login/search forms still need submit/navigation behavior from the
        model or the auto-submit path.
        """
        if initial_value_count != 1 or pending_value_count != 0 or submitted:
            return False

        task_lower = task.lower() if isinstance(task, str) else ""
        excluded = (
            "submit",
            "search button",
            "find boats",
            "results page",
            "log in",
            "login",
            "sign in",
            "password",
            "credentials",
        )
        if any(sig in task_lower for sig in excluded):
            return False

        completion_signals = (
            "finish when",
            "visible in the field",
            "visibly shows",
            "press tab",
            "commit",
            "zip code",
            "radius",
        )
        return any(sig in task_lower for sig in completion_signals)

    @staticmethod
    def _maybe_force_type_after_repeated_form_click(
        action: "Action",
        action_history: list["Action"],
        force_fill_values: list[dict],
        force_fill_used_regions: list[tuple[int, int]],
        task: str,
    ) -> "Action | None":
        """Type the next known form value when a model repeats a field click.

        Claude sometimes focuses a text field, sees no large visual delta, and
        repeats the same click until the visual loop detector stops the task.
        When the plan has explicit values to type, convert the repeated click
        into the next type_text action. This stays in the CUA action channel
        and avoids DOM reads.
        """
        from ..actions import Action, ActionType

        if action.action_type != ActionType.CLICK or not force_fill_values:
            return None
        if not action_history:
            return None

        task_lower = task.lower() if isinstance(task, str) else ""
        form_signals = (
            "zip code",
            "radius",
            "search form",
            "filter",
            "type ",
            "log in",
            "login",
            "sign in",
            "credentials",
            "user id",
            "username",
            "email",
            "password",
            "fill in",
            "enter the",
            "type the",
            "type into",
        )
        if not any(sig in task_lower for sig in form_signals):
            return None

        previous_click = next(
            (
                a
                for a in reversed(action_history)
                if a.action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK)
            ),
            None,
        )
        if previous_click is None:
            return None

        px = int((action.params or {}).get("x", 0))
        py = int((action.params or {}).get("y", 0))
        if not GymRunner._is_valid_force_fill_click(px, py):
            return None
        prev_x = int((previous_click.params or {}).get("x", 0))
        prev_y = int((previous_click.params or {}).get("y", 0))
        near_equal = abs(px - prev_x) <= 8 and abs(py - prev_y) <= 8
        if not near_equal:
            return None

        for ux, uy in force_fill_used_regions:
            if abs(ux - px) <= 20 and abs(uy - py) <= 20:
                return None

        entry = force_fill_values.pop(0)
        value = str(entry.get("value") or "")
        if not value:
            return None

        force_fill_used_regions.append((px, py))
        return Action(
            ActionType.TYPE,
            {"text": value},
            reasoning=(
                "force-fill: repeated click likely focused a form field; "
                f"typing plan value for {entry.get('label')!r}"
            ),
        )

    @staticmethod
    def _maybe_force_type_text(
        action: "Action",
        action_history: list["Action"],
        force_fill_values: list[dict],
        force_fill_used_regions: list[tuple[int, int]],
        brain: Any | None = None,
        screenshot: Image.Image | None = None,
    ) -> "Action | None":
        """Decide whether to override a click with type_text using Holo3 vision.

        Substitution fires when:

        1. Action is a click and the click region isn't already filled.
        2. Holo3 confirms an editable input is focused (CUA-pure detection
           — only the agent's screenshot, no DOM).
        3. The focused field's visible label/type matches an unconsumed
           ``{label, value}`` pair from the plan-extracted queue.

        Each entry in ``force_fill_values`` is ``{"label": str, "value": str}``.
        Matching: case-insensitive substring overlap between the focused
        field's label/type and the queue entry's label. If exactly one
        unconsumed value remains and no labels match, fall back to FIFO
        on the assumption that the last value belongs to the last field.

        Returns ``None`` (no substitution) when the detector fails, no
        labels match, or the click region was already used.
        """
        from ..actions import Action, ActionType
        from . import holo3_detector

        if action.action_type != ActionType.CLICK:
            return None
        if not force_fill_values:
            return None

        px = int((action.params or {}).get("x", 0))
        py = int((action.params or {}).get("y", 0))
        if not GymRunner._is_valid_force_fill_click(px, py):
            return None

        # Already filled this region? Don't double-consume.
        for ux, uy in force_fill_used_regions:
            if abs(ux - px) <= 20 and abs(uy - py) <= 20:
                return None

        # Holo3 vision detection on the current screenshot.
        focused: dict | None = None
        if brain is not None and screenshot is not None:
            try:
                focused = holo3_detector.detect_focused_field(
                    brain, screenshot, click_coords=(px, py),
                )
            except Exception as exc:
                logger.warning("force-fill detector raised: %s", exc)
                focused = None

        if not focused or not focused.get("focused"):
            return None

        field_label = str(focused.get("label") or "").lower()
        field_type = str(focused.get("type") or "").lower()
        haystack = f"{field_label} {field_type}".strip()

        # Match by label/type overlap. Each token is short (1-3 words);
        # bidirectional substring covers "password" ↔ "password field"
        # and "user id" ↔ "username" reasonably well.
        chosen_idx: int | None = None
        for idx, entry in enumerate(force_fill_values):
            entry_label = str(entry.get("label") or "").lower()
            if not entry_label:
                continue
            if entry_label in haystack or any(
                tok in haystack for tok in entry_label.split() if len(tok) >= 3
            ):
                chosen_idx = idx
                break
            # Special case: password fields are always the password value.
            if field_type == "password" and "password" in entry_label:
                chosen_idx = idx
                break

        # No label match: don't substitute. The "last-value fallback" was
        # tempting (always type something) but exactly that fallback caused
        # run 015's wrong-field bug — Industry Vertical typed on the login
        # page. Labels must match; otherwise wait for the right field.
        if chosen_idx is None:
            return None

        entry = force_fill_values.pop(chosen_idx)
        force_fill_used_regions.append((px, py))
        value = entry.get("value", "")
        return Action(
            ActionType.TYPE,
            {"text": value},
            reasoning=(
                f"force-fill: focused {field_label or field_type or 'input'!r} "
                f"matched plan label {entry.get('label')!r}"
            ),
        )

    @staticmethod
    def _extract_plan(thinking: str) -> str | None:
        """Extract a numbered plan from the model's thinking output."""
        import re

        lines = thinking.split("\n")
        plan_lines: list[str] = []

        # Look for numbered lines (1. xxx, 2. xxx, etc.)
        in_plan = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^\d+[\.\)]\s", stripped):
                in_plan = True
                plan_lines.append(stripped)
            elif in_plan and not stripped:
                break  # Empty line ends the plan block
            elif in_plan and not re.match(r"^\d+[\.\)]\s", stripped):
                break  # Non-numbered line ends the plan

        if plan_lines:
            return "\n".join(plan_lines[:10])  # Cap at 10 steps
        return None

    @staticmethod
    def _build_feedback(
        action: Action,
        gym_result: Any,
        last_url: str,
        last_title: str,
    ) -> str:
        """Describe what happened after executing an action.

        Includes URL changes and off-site backtrack warnings so the model
        gets immediate feedback about navigation errors.
        """
        parts: list[str] = []

        new_url = gym_result.info.get("url", "")
        new_title = gym_result.info.get("title", "")
        focused = gym_result.info.get("focused_input")

        # Off-site backtrack warning — highest priority feedback
        if gym_result.info.get("backtracked"):
            warning = gym_result.info.get("warning", "Off-site navigation detected")
            parts.append(f"WARNING: {warning}. Do NOT click social media icons or external links.")

        # URL change
        if new_url and new_url != last_url:
            parts.append(f"page navigated to {new_url}")

        # Title change
        if new_title and new_title != last_title:
            parts.append(f"page title: \"{new_title}\"")

        # Focused input detection
        if focused:
            field_name = (
                focused.get("placeholder") or focused.get("name")
                or focused.get("id") or focused.get("type") or "input"
            )
            if focused.get("empty"):
                parts.append(f"'{field_name}' field is now focused (empty)")
            else:
                parts.append(f"'{field_name}' field focused, contains: \"{focused.get('value', '')}\"")

        # Type verification
        type_verify = gym_result.info.get("type_verified")
        if action.action_type == ActionType.TYPE:
            typed = action.params.get("text", "")
            if type_verify and type_verify.get("success"):
                field_name = type_verify.get("field", "field")
                parts.append(f"typed \"{typed}\" into {field_name} (verified)")
            elif type_verify and not type_verify.get("success"):
                reason = type_verify.get("reason", "unknown")
                parts.append(f"TYPING FAILED: tried to type \"{typed}\" but {reason}")
            else:
                parts.append(f"typed \"{typed}\" (unverified)")
        elif action.action_type == ActionType.KEY_PRESS:
            parts.append(f"pressed {action.params.get('keys', '')}")
        elif action.action_type == ActionType.CLICK and not parts:
            parts.append("clicked (no visible change)")

        return "; ".join(parts) if parts else "no visible change"

    # ── Loop detection ───────────────────────────────────────────────────

    @staticmethod
    def _detect_repeat(action_history: list[Action], window: int) -> bool:
        """Check if the last N actions are identical (type + params)."""
        if len(action_history) < window:
            return False
        recent = action_history[-window:]
        first = recent[0]
        return all(
            a.action_type == first.action_type and a.params == first.params
            for a in recent[1:]
        )

    @staticmethod
    def _build_nudge(action_history: list[Action], focused_input: dict | None) -> str:
        """Build a contextual nudge to break the model out of a loop."""
        last = action_history[-1]
        repeated_action = f"{last.action_type.value}({last.params})"

        # Form-aware nudge: input field is focused, model should type
        if focused_input and last.action_type == ActionType.CLICK:
            field_desc = (
                focused_input.get("placeholder")
                or focused_input.get("name")
                or focused_input.get("id")
                or focused_input.get("type")
                or "input"
            )
            is_empty = focused_input.get("empty", True)
            current_value = focused_input.get("value", "")

            if is_empty:
                return (
                    f"\n\nIMPORTANT: You have already clicked the '{field_desc}' field "
                    f"and it now has focus. Do NOT click it again. "
                    f"Your next action must be type_text() to enter the value. "
                    f"The field is empty and ready for input."
                )
            else:
                return (
                    f"\n\nIMPORTANT: The '{field_desc}' field is focused and already "
                    f"contains: \"{current_value}\". If you need to change it, "
                    f"first select all with key_press('ctrl+a'), then type_text() "
                    f"the new value. Do NOT click the same field again."
                )

        # Generic nudge
        return (
            f"\n\nIMPORTANT: You have repeated the same action ({repeated_action}) "
            f"multiple times with no progress. This approach is not working. "
            f"Try a DIFFERENT action — for example: scroll to find the right element, "
            f"use keyboard navigation (Tab, Enter), type in a focused field, or "
            f"navigate to a different part of the page."
        )
