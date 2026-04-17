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
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment, GymObservation

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
    """A single step in the agent's trajectory."""

    step: int
    action: Action
    thinking: str
    reward: float
    done: bool
    inference_time: float
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)


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
    ):
        self.brain = brain
        self.env = env
        self.max_steps = max_steps
        self.frames_per_inference = frames_per_inference
        self.soft_loop_window = soft_loop_window
        self.hard_loop_window = hard_loop_window
        self.plan_executor = plan_executor
        self.page_discovery = page_discovery

    def run(
        self,
        task: str,
        task_id: str = "default",
        seed: int | None = None,
        plan_steps: str | None = None,
        plan: Any = None,
        plan_inputs: dict[str, str] | None = None,
        start_url: str | None = None,
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
        """
        logger.info(f"Starting task: {task!r} (id={task_id})")
        t0 = time.time()

        frame_history: list[Image.Image] = []
        action_history: list[Action] = []
        trajectory: list[TrajectoryStep] = []
        total_reward = 0.0
        plan_inputs = dict(plan_inputs or {})

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

        # Reset environment
        reset_kwargs: dict[str, Any] = {"task_id": task_id}
        if start_url:
            reset_kwargs["start_url"] = start_url
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs = self.env.reset(task, **reset_kwargs)
        frame_history.append(obs.screenshot)
        last_url = obs.extras.get("url", "")
        last_title = obs.extras.get("title", "")
        latest_obs = obs  # Track latest obs for SoM/DOM state injection

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

                if self.plan_executor.can_execute(current_plan_step):
                    step_result = self.plan_executor.execute(current_plan_step, plan_inputs)
                    print(f"  [executor] result: success={step_result.success} detail={step_result.detail}")

                    if step_result.success:
                        logger.info(f"  Direct exec OK: {step_result.detail}")
                        plan_step_idx += 1
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
                        print(f"  [executor] direct failed, trying discovery...")

                        if self.page_discovery:
                            discovery_result = self._try_discovery_execution(
                                current_plan_step, plan_inputs, step_log, frame_history,
                            )
                            if discovery_result:
                                plan_step_idx += 1
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
                logger.info(f"Action: {action} ({inference_time:.2f}s)")

                # Step 0: extract plan from model's thinking
                if step_num == 1 and thinking and not agent_plan:
                    agent_plan = self._extract_plan(thinking)
                    if agent_plan:
                        logger.info(f"Plan captured: {agent_plan[:120]}...")

                if action.action_type == ActionType.DONE:
                    success = action.params.get("success", False)
                    trajectory.append(TrajectoryStep(
                        step=step_num, action=action, thinking=thinking,
                        reward=0.0, done=True, inference_time=inference_time,
                    ))
                    termination_reason = "done"
                    break

                # Execute action in the environment
                gym_result = self.env.step(action)
                action_history.append(action)
                frame_history.append(gym_result.observation.screenshot)
                total_reward += gym_result.reward
                latest_obs = gym_result.observation  # Update for next step's SoM

                feedback = self._build_feedback(
                    action=action, gym_result=gym_result,
                    last_url=last_url, last_title=last_title,
                )

                focused_input = gym_result.info.get("focused_input")
                if focused_input is not None:
                    last_focused_input = focused_input
                elif action.action_type not in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
                    last_focused_input = None

                last_url = gym_result.info.get("url", last_url)
                last_title = gym_result.info.get("title", last_title)

                step_log.append(f"Step {step_num}: {action} → {feedback}")

                trajectory.append(TrajectoryStep(
                    step=step_num, action=action, thinking=thinking,
                    reward=gym_result.reward, done=gym_result.done,
                    inference_time=inference_time, feedback=feedback,
                ))

                logger.info(f"Feedback: {feedback}")

                if gym_result.done:
                    termination_reason = "env_done"
                    break

                if self._detect_repeat(action_history, self.hard_loop_window):
                    logger.warning("Hard action loop detected — stopping")
                    termination_reason = "loop"
                    break

            # Trim frame history
            if len(frame_history) > self.frames_per_inference * 3:
                frame_history = frame_history[-self.frames_per_inference * 2:]

        total_time = time.time() - t0
        success = termination_reason == "done" or (termination_reason == "env_done" and total_reward > 0)
        logger.info(f"Task finished: {termination_reason}, {len(trajectory)} steps, {total_time:.1f}s")

        return RunResult(
            task=task, task_id=task_id, success=success,
            total_reward=total_reward, total_steps=len(trajectory),
            total_time=total_time, trajectory=trajectory,
            termination_reason=termination_reason,
        )

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

            # Fast-fail nudge: if recent thinking mentions error pages, push to terminate
            if last_thinking and step_num <= 5:
                think_lower = last_thinking.lower()
                if any(sig in think_lower for sig in [
                    "page not found", "404", "this site can't be reached",
                    "err_tunnel", "err_connection", "this page has been removed",
                ]):
                    parts.append(
                        "\n\nIMPORTANT: The page shows an error (404 / can't be reached). "
                        "Do NOT keep trying. Press Alt+Left to go back and call "
                        "terminate('success') with: SKIPPED | page error or 404. "
                        "Move to the next listing."
                    )

            # Soft loop nudge
            if self._detect_repeat(action_history, self.soft_loop_window):
                nudge = self._build_nudge(action_history, last_focused_input)
                parts.append(nudge)

        return "\n".join(parts)

    @staticmethod
    def _get_curriculum(task: str) -> str:
        """Load relevant curriculum techniques for this task."""
        try:
            from mantis_agent.curriculum import select_techniques
            return select_techniques(task, domain="chrome", max_topics=2)
        except Exception:
            return ""

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
            print(f"  [discovery] no elements found on page")
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
