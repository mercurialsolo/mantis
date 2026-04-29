"""Tests for the ORS server adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mantis_agent.actions import Action, ActionType
from mantis_agent.gym.base import GymEnvironment, GymObservation, GymResult
from mantis_agent.gym.ors_server import _action_from_tool, make_app
from mantis_agent.rewards import BoatTraderReward, PlanAdherenceReward


# ── stub environment ────────────────────────────────────────────────────


class StubEnv(GymEnvironment):
    """Records action history; returns a constant 8x8 PNG as observation."""

    def __init__(self, on_step_url: str = "https://www.boattrader.com/x/"):
        self._url = on_step_url
        self.actions: list[Action] = []
        self._closed = False

    def reset(self, task: str, **kwargs: Any) -> GymObservation:
        self.actions = []
        img = Image.new("RGB", (8, 8), "red")
        return GymObservation(screenshot=img, extras={"url": self._url})

    def step(self, action: Action) -> GymResult:
        self.actions.append(action)
        img = Image.new("RGB", (8, 8), "blue")
        return GymResult(
            observation=GymObservation(screenshot=img, extras={"url": self._url}),
            reward=0.0, done=False,
            info={"url": self._url, "title": "stub"},
        )

    def close(self) -> None:
        self._closed = True

    @property
    def screen_size(self) -> tuple[int, int]:
        return (8, 8)


# ── helpers ─────────────────────────────────────────────────────────────


def _make_client(reward: bool = False, tasks: list[dict] | None = None,
                 tmp_path: Path | None = None) -> TestClient:
    tasks_path = None
    if tasks is not None and tmp_path is not None:
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))
    app = make_app(
        env_factory=StubEnv,
        reward_factory=(lambda: PlanAdherenceReward()) if reward else None,
        tasks_path=tasks_path,
    )
    return TestClient(app)


# ── Action ↔ tool translation ───────────────────────────────────────────


def test_action_from_tool_click() -> None:
    a = _action_from_tool("click", {"x": 10, "y": 20})
    assert a.action_type == ActionType.CLICK
    assert a.params == {"x": 10, "y": 20}


def test_action_from_tool_terminate() -> None:
    a = _action_from_tool("terminate", {"success": True, "summary": "ok"})
    assert a.action_type == ActionType.DONE
    assert a.params["success"] is True
    assert a.params["summary"] == "ok"


def test_action_from_tool_unknown_falls_back_to_wait() -> None:
    a = _action_from_tool("not_a_real_tool", {})
    assert a.action_type == ActionType.WAIT


def test_action_from_tool_key_press_alias() -> None:
    a = _action_from_tool("key_press", {"key": "enter"})
    assert a.action_type == ActionType.KEY_PRESS
    assert a.params["keys"] == "enter"


# ── server endpoints ────────────────────────────────────────────────────


def test_root_returns_metadata() -> None:
    client = _make_client()
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data and "active_sessions" in data


def test_tasks_listing_with_split(tmp_path: Path) -> None:
    catalog = [
        {"task_id": "a", "prompt": "do A", "split": "train"},
        {"task_id": "b", "prompt": "do B", "split": "test"},
        {"task_id": "c", "prompt": "do C", "split": "train"},
    ]
    client = _make_client(tasks=catalog, tmp_path=tmp_path)

    all_ = client.get("/tasks").json()
    assert len(all_) == 3
    train = client.get("/tasks?split=train").json()
    assert {t["task_id"] for t in train} == {"a", "c"}
    splits = client.get("/splits").json()
    assert set(splits.keys()) == {"train", "test"}


def test_open_session_with_ad_hoc_task() -> None:
    client = _make_client()
    r = client.post("/sessions", json={"task": "extract one listing"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["task_id"] == "ad_hoc"
    # initial blocks should include the reset screenshot
    assert any(b["type"] == "image" for b in data["initial_blocks"])


def test_open_session_unknown_task_id_404(tmp_path: Path) -> None:
    catalog = [{"task_id": "a", "prompt": "x", "split": "train"}]
    client = _make_client(tasks=catalog, tmp_path=tmp_path)
    r = client.post("/sessions", json={"task_id": "missing"})
    assert r.status_code == 404


def test_tool_call_returns_observation_block_and_reward() -> None:
    client = _make_client(reward=True)
    sid = client.post("/sessions", json={"task": "demo"}).json()["session_id"]

    r = client.post(f"/sessions/{sid}/tools/click",
                    json={"arguments": {"x": 100, "y": 200}})
    assert r.status_code == 200
    out = r.json()
    assert any(b["type"] == "image" for b in out["blocks"])
    # PlanAdherenceReward gives +0.1 for a well-formed click on a known url
    # (no allowed_domains set, so off_site = 0). Format reward is the only
    # contribution, sums to 0.1.
    assert out["reward"] == pytest.approx(0.1)
    assert out["finished"] is False


def test_terminate_tool_finishes_session() -> None:
    client = _make_client()
    sid = client.post("/sessions", json={"task": "demo"}).json()["session_id"]

    r = client.post(f"/sessions/{sid}/tools/terminate",
                    json={"arguments": {"success": True, "summary": "all done"}})
    assert r.status_code == 200
    assert r.json()["finished"] is True

    # Subsequent tool calls return 409.
    r2 = client.post(f"/sessions/{sid}/tools/click",
                     json={"arguments": {"x": 1, "y": 2}})
    assert r2.status_code == 409


def test_terminate_with_boattrader_reward_grades_summary() -> None:
    """Terminal reward fires through the reward_fn.episode() hook."""
    app = make_app(
        env_factory=StubEnv,
        reward_factory=lambda: BoatTraderReward(),
    )
    client = TestClient(app)
    sid = client.post("/sessions",
                      json={"task": "extract listing",
                            "ground_truth": {"min_price": 35000}}).json()["session_id"]

    summary = "2018 Sea Ray 240 $42,500 https://www.boattrader.com/boat/x/"
    r = client.post(f"/sessions/{sid}/tools/terminate",
                    json={"arguments": {"success": True, "summary": summary}})
    out = r.json()
    assert out["finished"] is True
    assert out["reward"] == pytest.approx(1.0)
    assert "gate_passed" in out["metadata"]["reward_components"]


def test_session_close_removes_state() -> None:
    client = _make_client()
    sid = client.post("/sessions", json={"task": "demo"}).json()["session_id"]
    r = client.delete(f"/sessions/{sid}")
    assert r.status_code == 200
    # second close → 404
    r2 = client.delete(f"/sessions/{sid}")
    assert r2.status_code == 404


def test_get_prompt_includes_initial_blocks() -> None:
    client = _make_client()
    sid = client.post("/sessions", json={"task": "demo"}).json()["session_id"]
    p = client.get(f"/sessions/{sid}/prompt").json()
    # text block carrying the task + image block carrying the reset screenshot
    types = [b["type"] for b in p["blocks"]]
    assert "text" in types and "image" in types
