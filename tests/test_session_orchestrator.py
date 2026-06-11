"""Tests for the session orchestrator loop (Phase 1.5, #846, PR 2).

Drives :func:`run_session_loop` against an in-process ``dict`` so the
publish + close + TTL paths are exercised without Modal.
"""

from __future__ import annotations

from typing import Any

from mantis_agent.server.session_orchestrator import run_session_loop


def _now_factory(values: list[float]):
    """Returns a ``now()`` callable that pops the next pre-canned time."""

    def _now() -> float:
        return values.pop(0) if values else 9_999_999.0

    return _now


def _sleep_noop(_seconds: float) -> None:  # pragma: no cover — only invoked between checks
    return None


def test_publishes_initial_entry_and_returns_on_publish_only() -> None:
    d: dict[str, Any] = {}
    out = run_session_loop(
        session_id="sess_1",
        session_token="tok",
        init_payload={"tenant_id": "t1", "profile_id": "p", "run_id": "r"},
        ttl_seconds=10,
        session_dict=d,
        tunnel_url="https://x.modal.run",
        publish_only=True,
    )
    assert out["status"] == "ready"
    entry = d["sess_1"]
    assert entry["tunnel_url"] == "https://x.modal.run"
    assert entry["status"] == "ready"
    assert entry["session_token"] == "tok"
    assert entry["tenant_id"] == "t1"
    assert "started_at_ms" in entry


def test_loop_exits_on_close_requested() -> None:
    """A second poll picks up close_requested=True → loop exits clean."""
    d: dict[str, Any] = {}
    # Now: first call = 0.0 (start), then 0.5 (TTL check loop 1), 1.0 (loop 2 — close)
    now_seq = [0.0, 0.5, 1.0]
    polled = {"n": 0}

    def sleep_then_set_close(_secs: float) -> None:
        polled["n"] += 1
        if polled["n"] == 1:
            # Simulate the router calling DELETE → sets close_requested.
            cur = d["sess_1"]
            cur["close_requested"] = True
            d["sess_1"] = cur

    out = run_session_loop(
        session_id="sess_1",
        session_token="tok",
        init_payload={"tenant_id": "t1", "profile_id": "p", "run_id": "r"},
        ttl_seconds=3600,
        session_dict=d,
        tunnel_url="https://x.modal.run",
        sleep=sleep_then_set_close,
        now=_now_factory(now_seq),
    )
    assert out["status"] == "closed"
    assert d["sess_1"]["status"] == "closed"
    assert "terminal_at_ms" in d["sess_1"]


def test_loop_exits_on_ttl_expiry() -> None:
    d: dict[str, Any] = {}
    # Now sequence: start=0.0, loop check 1 (5.0 < 10), loop check 2 (15.0 > 10 → expired)
    now_seq = [0.0, 5.0, 15.0]
    out = run_session_loop(
        session_id="sess_t",
        session_token="tok",
        init_payload={},
        ttl_seconds=10,
        session_dict=d,
        tunnel_url="https://x.modal.run",
        sleep=_sleep_noop,
        now=_now_factory(now_seq),
    )
    assert out["status"] == "expired"
    assert d["sess_t"]["status"] == "expired"


def test_publish_failure_returns_publish_failed() -> None:
    """If the session_dict write raises, the loop returns publish_failed
    without crashing the container."""

    class _Boom(dict):
        def __setitem__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("network split")

    d = _Boom()
    out = run_session_loop(
        session_id="sess_x",
        session_token="tok",
        init_payload={},
        ttl_seconds=10,
        session_dict=d,
        tunnel_url="https://x.modal.run",
        publish_only=True,
    )
    assert out["status"] == "publish_failed"
    assert "network split" in out["error"]


def test_close_signal_survives_dict_update_race() -> None:
    """Concurrent writer (orchestrator vs router) — close still wins via
    sticky terminal state, and the second poll picks it up."""
    d: dict[str, Any] = {}
    polled = {"n": 0}
    # Now: start=0.0, then later checks.
    now_seq = [0.0, 0.5, 1.0]

    def sleep_then_race(_secs: float) -> None:
        polled["n"] += 1
        if polled["n"] == 1:
            # Concurrent: writer flips close_requested.
            d["sess_1"] = {"close_requested": True, "noise": "ok"}

    out = run_session_loop(
        session_id="sess_1",
        session_token="tok",
        init_payload={},
        ttl_seconds=3600,
        session_dict=d,
        tunnel_url="https://x.modal.run",
        sleep=sleep_then_race,
        now=_now_factory(now_seq),
    )
    assert out["status"] == "closed"


def test_init_payload_fields_propagate_to_entry() -> None:
    d: dict[str, Any] = {}
    run_session_loop(
        session_id="sess_p",
        session_token="tok",
        init_payload={
            "tenant_id": "alice",
            "profile_id": "prof_42",
            "run_id": "run_xyz",
            "start_url": "https://example.com",
        },
        ttl_seconds=3600,
        session_dict=d,
        tunnel_url="https://u.modal.run",
        publish_only=True,
    )
    entry = d["sess_p"]
    assert entry["tenant_id"] == "alice"
    assert entry["profile_id"] == "prof_42"
    assert entry["run_id"] == "run_xyz"
    # ``start_url`` isn't part of the public entry contract — kept on
    # the orchestrator side only.
    assert "start_url" not in entry
