"""``RunStatus.wall_time_breakdown()`` — convenience accessor for the
epic #362 Phase B surface.

Client code shouldn't have to defensively dereference
``status.summary["wall_time_breakdown"]`` — the accessor returns
``{}`` on pre-terminal runs (no summary) or on summaries that pre-date
Phase B, and a typed dict otherwise.
"""

from __future__ import annotations

from mantis_agent.api_schemas import RunStatus


def test_returns_empty_dict_when_no_summary() -> None:
    """Pre-terminal runs land here — ``summary`` is ``None`` until the
    runner writes the result envelope."""
    status = RunStatus(status="running", run_id="r1")
    assert status.wall_time_breakdown() == {}


def test_returns_empty_dict_when_summary_omits_key() -> None:
    """Pre-Phase-B summaries (or hosts that skipped the field) yield
    ``{}`` — caller can branch on truthiness without a KeyError."""
    status = RunStatus(
        status="succeeded", run_id="r1",
        summary={"total_time_s": 30, "steps_executed": 5},
    )
    assert status.wall_time_breakdown() == {}


def test_returns_typed_dict_when_breakdown_present() -> None:
    status = RunStatus(
        status="succeeded", run_id="r1",
        summary={
            "total_time_s": 100,
            "wall_time_breakdown": {
                "think": 40.5, "act": 5.0, "settle": 14.5, "overhead": 40.0,
            },
        },
    )
    bd = status.wall_time_breakdown()
    assert bd == {"think": 40.5, "act": 5.0, "settle": 14.5, "overhead": 40.0}
    # Values are floats — callers can do arithmetic without coercing.
    assert all(isinstance(v, float) for v in bd.values())


def test_coerces_int_seconds_to_float() -> None:
    """JSON over the wire may yield int seconds; accessor normalizes."""
    status = RunStatus(
        status="succeeded", run_id="r1",
        summary={"wall_time_breakdown": {"think": 40, "act": 5}},
    )
    bd = status.wall_time_breakdown()
    assert bd == {"think": 40.0, "act": 5.0}


def test_handles_non_dict_breakdown_defensively() -> None:
    """A misshapen summary (string / list under the key) must not
    raise — the accessor falls back to ``{}``."""
    status = RunStatus(
        status="succeeded", run_id="r1",
        summary={"wall_time_breakdown": "not a dict"},
    )
    assert status.wall_time_breakdown() == {}
