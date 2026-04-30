"""Open Reward Standard (ORS) HTTP adapter for any GymEnvironment.

Exposes a Mantis `GymEnvironment` over the ORS protocol so external RL
trainers and rollout clients can drive it without importing Mantis. This
is a pure transport layer — the reward signal still comes from a
mantis_agent.rewards.RewardFn the operator wires in.

Mapping
-------
    ORS Tool                           → Mantis Action
    -----------                          --------------
    click(x, y, button)                → ActionType.CLICK
    double_click(x, y)                 → ActionType.DOUBLE_CLICK
    type_text(text)                    → ActionType.TYPE
    key_press(keys)                    → ActionType.KEY_PRESS
    scroll(direction, amount, x, y)    → ActionType.SCROLL
    drag(start_x, start_y, end_x, ...) → ActionType.DRAG
    wait(seconds)                      → ActionType.WAIT
    terminate(success, summary)        → ActionType.DONE

    ORS ToolOutput
        blocks   = [ImageBlock(b64 screenshot)] + optional TextBlock
        reward   = float (per-step reward from RewardFn or 0.0)
        finished = bool (true on env-side done OR terminate tool)

Endpoints
---------
    GET  /                              health/info
    GET  /tasks                         list tasks (optional split filter)
    GET  /splits                        list splits
    POST /sessions                      open session for {task_id, prompt}
    GET  /sessions/{sid}/prompt         initial prompt + first observation
    POST /sessions/{sid}/tools/{name}   call a tool, returns ToolOutput
    DELETE /sessions/{sid}              close session

This is a thin shim — under ~300 LoC. The real reward and rollout logic
stays in `mantis_agent.rewards` and `training.rollout_collector`.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ..actions import Action, ActionType
from .base import GymEnvironment

logger = logging.getLogger(__name__)


# ── ORS payload shapes (minimal, matching the spec excerpts) ─────────────


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ImageBlock:
    image_url: str  # data: URI carrying a base64 PNG
    type: str = "image"


@dataclass
class ToolOutput:
    blocks: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Tool → Action translation ────────────────────────────────────────────


def _action_from_tool(name: str, args: dict[str, Any]) -> Action:
    """Map an ORS tool call to a Mantis Action.

    Unknown tool names fall back to a 1s WAIT — never raises, so a
    misbehaving client gets a no-op rather than a 500.
    """
    n = name.lower()
    if n == "click":
        return Action(ActionType.CLICK, dict(args))
    if n == "double_click":
        return Action(ActionType.DOUBLE_CLICK, dict(args))
    if n == "type_text":
        return Action(ActionType.TYPE, {"text": args.get("text", "")})
    if n == "key_press":
        keys = args.get("keys") or args.get("key", "")
        return Action(ActionType.KEY_PRESS, {"keys": keys})
    if n == "scroll":
        return Action(ActionType.SCROLL, dict(args))
    if n == "drag":
        return Action(ActionType.DRAG, dict(args))
    if n == "wait":
        return Action(ActionType.WAIT, {"seconds": args.get("seconds", 1.0)})
    if n in ("terminate", "submit_solution", "done"):
        return Action(ActionType.DONE, {
            "success": bool(args.get("success", True)),
            "summary": args.get("summary") or args.get("solution") or "",
        })
    logger.warning("unknown ORS tool '%s' — coerced to wait(1)", name)
    return Action(ActionType.WAIT, {"seconds": 1.0})


# ── Observation → Block translation ─────────────────────────────────────


def _img_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _observation_to_blocks(observation: Any) -> list[dict[str, Any]]:
    """Build ORS blocks from a GymObservation."""
    blocks: list[dict[str, Any]] = []
    img = getattr(observation, "screenshot", None)
    if isinstance(img, Image.Image):
        blocks.append({"type": "image", "image_url": _img_to_data_uri(img)})
    extras = getattr(observation, "extras", None)
    if extras:
        blocks.append({"type": "text", "text": json.dumps(extras, default=str)})
    return blocks


# ── Session state ────────────────────────────────────────────────────────


@dataclass
class _Session:
    """Server-side episode state — one per active client."""

    sid: str
    task: str
    task_id: str
    env: GymEnvironment
    reward_fn: Any = None
    ground_truth: dict[str, Any] | None = None
    state: Any = None  # rewards.EpisodeState, set when reward_fn is provided
    last_blocks: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Server ───────────────────────────────────────────────────────────────


try:
    from pydantic import BaseModel as _BaseModel
except ImportError:  # pragma: no cover
    _BaseModel = object  # type: ignore[misc,assignment]


class CreateSessionReq(_BaseModel):
    task_id: str | None = None
    task: str | None = None
    ground_truth: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ToolCallReq(_BaseModel):
    arguments: dict[str, Any] = {}


def make_app(
    env_factory: Callable[[], GymEnvironment],
    reward_factory: Callable[[], Any] | None = None,
    tasks_path: Path | None = None,
    system_prompt: str = "",
) -> Any:
    """Build a FastAPI app exposing ORS endpoints over `env_factory()`.

    Args:
        env_factory: zero-arg callable returning a fresh GymEnvironment per
            session. Reusing one env across clients would break isolation.
        reward_factory: optional callable returning a RewardFn per session.
            When omitted, every ToolOutput.reward is 0.0.
        tasks_path: JSON file describing the task catalogue. Each entry:
                {"task_id": "...", "prompt": "...", "split": "train",
                 "ground_truth": {...}, "metadata": {...}}
        system_prompt: prepended to the initial prompt response.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Mantis ORS Server", version="0.1.0")
    sessions: dict[str, _Session] = {}
    tasks: list[dict[str, Any]] = []
    if tasks_path and tasks_path.exists():
        tasks = json.loads(tasks_path.read_text())
        if isinstance(tasks, dict):
            tasks = tasks.get("tasks", [])
    splits: dict[str, list[str]] = {}
    for t in tasks:
        splits.setdefault(t.get("split", "train"), []).append(t["task_id"])

    def _resolve_task(req: CreateSessionReq) -> dict[str, Any]:
        if req.task_id and tasks:
            for t in tasks:
                if t["task_id"] == req.task_id:
                    return t
            raise HTTPException(404, f"unknown task_id {req.task_id}")
        if req.task:
            return {"task_id": req.task_id or "ad_hoc",
                    "prompt": req.task,
                    "ground_truth": req.ground_truth or {},
                    "metadata": req.metadata or {}}
        raise HTTPException(400, "must supply task_id or task")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "Mantis ORS Server",
            "tasks": len(tasks),
            "splits": list(splits.keys()),
            "active_sessions": len(sessions),
        }

    @app.get("/tasks")
    def list_tasks(split: str | None = None) -> JSONResponse:
        if split:
            filtered = [t for t in tasks if t.get("split", "train") == split]
            return JSONResponse(filtered)
        return JSONResponse(tasks)

    @app.get("/splits")
    def list_splits() -> dict[str, Any]:
        return {name: {"task_ids": ids} for name, ids in splits.items()}

    @app.post("/sessions")
    def open_session(req: CreateSessionReq) -> dict[str, Any]:
        task = _resolve_task(req)
        sid = uuid.uuid4().hex
        env = env_factory()
        reward_fn = reward_factory() if reward_factory else None

        # Reset env to first observation.
        obs = env.reset(task["prompt"], task_id=task["task_id"])
        blocks = _observation_to_blocks(obs)

        state = None
        if reward_fn is not None:
            from ..rewards import EpisodeState
            state = EpisodeState()

        sessions[sid] = _Session(
            sid=sid, task=task["prompt"], task_id=task["task_id"],
            env=env, reward_fn=reward_fn,
            ground_truth=task.get("ground_truth"),
            state=state, last_blocks=blocks,
            metadata=task.get("metadata", {}),
        )
        return {"session_id": sid, "task_id": task["task_id"],
                "prompt": task["prompt"], "initial_blocks": blocks}

    @app.get("/sessions/{sid}/prompt")
    def get_prompt(sid: str) -> dict[str, Any]:
        s = sessions.get(sid)
        if not s:
            raise HTTPException(404, "no such session")
        prefix = [{"type": "text", "text": system_prompt}] if system_prompt else []
        return {
            "task_id": s.task_id,
            "blocks": prefix + [{"type": "text", "text": s.task}] + s.last_blocks,
        }

    # response_model=None skips pydantic 2.13's TypeAdapter introspection of
    # the ToolOutput dataclass — FastAPI still serialises the return value via
    # the default jsonable_encoder, but doesn't try to build a forward-ref-aware
    # response schema (which fails on pydantic 2.13 for nested
    # `list[dict[str, Any]]` fields on a dataclass).
    @app.post("/sessions/{sid}/tools/{name}", response_model=None)
    def call_tool(sid: str, name: str, body: ToolCallReq) -> ToolOutput:
        s = sessions.get(sid)
        if not s:
            raise HTTPException(404, "no such session")
        if s.finished:
            raise HTTPException(409, "session already finished")

        action = _action_from_tool(name, body.arguments)

        # DONE never goes through env.step(); it terminates the episode.
        if action.action_type == ActionType.DONE:
            s.finished = True
            terminal_value = 0.0
            terminal_components: dict[str, float] = {}
            if s.reward_fn is not None and s.state is not None:
                from ..gym.runner import RunResult, TrajectoryStep
                trajectory = [TrajectoryStep(
                    step=1, action=action, thinking="", reward=0.0,
                    done=True, inference_time=0.0,
                )]
                fake_run = RunResult(
                    task=s.task, task_id=s.task_id, success=action.params.get("success", False),
                    total_reward=0.0, total_steps=1, total_time=0.0,
                    trajectory=trajectory, termination_reason="done",
                )
                signal = s.reward_fn.episode(
                    run_result=fake_run, state=s.state, ground_truth=s.ground_truth,
                )
                terminal_value = float(signal)
                terminal_components = dict(signal.components)
            return ToolOutput(
                blocks=[{"type": "text",
                         "text": f"terminated: {action.params.get('summary', '')}"}],
                reward=terminal_value, finished=True,
                metadata={"reward_components": terminal_components},
            )

        gym_result = s.env.step(action)

        step_value = float(gym_result.reward)
        step_components: dict[str, float] = {}
        if s.reward_fn is not None and s.state is not None:
            sig = s.reward_fn.step(action=action, gym_result=gym_result, state=s.state)
            step_value += float(sig)
            step_components = dict(sig.components)
            s.state.action_history.append(action)
            s.state.info_history.append(dict(gym_result.info))

        blocks = _observation_to_blocks(gym_result.observation)
        s.last_blocks = blocks
        if gym_result.done:
            s.finished = True
        return ToolOutput(
            blocks=blocks,
            reward=step_value,
            finished=gym_result.done,
            metadata={"info": gym_result.info, "reward_components": step_components},
        )

    @app.delete("/sessions/{sid}")
    def close_session(sid: str) -> dict[str, str]:
        s = sessions.pop(sid, None)
        if not s:
            raise HTTPException(404, "no such session")
        try:
            s.env.close()
        except Exception:
            pass
        return {"status": "closed"}

    return app


# ── CLI ─────────────────────────────────────────────────────────────────


_ENV_FACTORIES: dict[str, Callable[[], GymEnvironment]] = {}


def _register_env_factories() -> None:
    """Lazy-register env factories. Heavy adapters import only on demand."""
    def _playwright() -> GymEnvironment:
        from .playwright_env import PlaywrightGymEnv
        return PlaywrightGymEnv()
    _ENV_FACTORIES["playwright"] = _playwright

    def _gym_anything() -> GymEnvironment:
        from .gym_anything import GymAnythingAdapter
        return GymAnythingAdapter(env_dir=".")  # caller can override
    _ENV_FACTORIES["gym_anything"] = _gym_anything


_REWARD_FACTORIES: dict[str, Callable[[], Any]] = {}


def _register_reward_factories() -> None:
    def _plan_adherence() -> Any:
        from ..rewards import PlanAdherenceReward
        return PlanAdherenceReward()
    _REWARD_FACTORIES["plan_adherence"] = _plan_adherence

    def _boattrader() -> Any:
        from ..rewards import BoatTraderReward
        return BoatTraderReward()
    _REWARD_FACTORIES["boattrader"] = _boattrader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="playwright",
                        choices=list({"playwright", "gym_anything"}),
                        help="GymEnvironment adapter to expose")
    parser.add_argument("--reward", default="",
                        help="Reward fn name (plan_adherence | boattrader); empty disables")
    parser.add_argument("--tasks", default="",
                        help="Path to task catalogue JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    _register_env_factories()
    _register_reward_factories()

    env_factory = _ENV_FACTORIES[args.env]
    reward_factory = _REWARD_FACTORIES[args.reward] if args.reward else None
    tasks_path = Path(args.tasks) if args.tasks else None

    app = make_app(
        env_factory=env_factory,
        reward_factory=reward_factory,
        tasks_path=tasks_path,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
