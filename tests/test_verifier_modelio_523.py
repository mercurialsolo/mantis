"""PR B-4 of #523: verifier-layer modelio capture.

Both `verify_gate` call sites in `gym/_runner_helpers.py` (lines
~544 and ~873) wrap their `runner.extractor.verify_gate(...)`
invocation in `publish_modelio_context(augur, "verifier",
step_index=index)`. The underlying `extractor._verify_client` uses
`AnthropicToolUseClient` so the capture fires automatically once
the context is published.
"""

from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("augur_sdk")


def test_runner_helpers_wraps_filter_gate_verify():
    """``ensure_results_filters`` calls ``verify_gate`` inside a
    ``publish_modelio_context(layer="verifier", step_index=index)``
    block. AST-light check: the source between the function def
    and the ``verify_gate`` call must contain a ``publish_modelio
    _context`` line with ``layer="verifier"``."""
    from mantis_agent.gym import _runner_helpers
    src = inspect.getsource(_runner_helpers.ensure_results_filters)
    assert "verify_gate" in src
    assert "publish_modelio_context" in src, (
        "ensure_results_filters must wrap verify_gate in publish_modelio_context"
    )
    assert re.search(r'layer\s*=\s*"verifier"', src), (
        "ensure_results_filters's publish_modelio_context must use layer='verifier'"
    )
    # The wrap must appear BEFORE the verify_gate call so the
    # contextvar is published when the LLM call fires.
    pub_idx = src.index("publish_modelio_context")
    call_idx = src.index("verify_gate(")
    assert pub_idx < call_idx, (
        "publish_modelio_context must precede verify_gate call"
    )


def test_runner_helpers_wraps_gate_step_verify():
    """``execute_step`` wraps the ``step.gate``-branch verify_gate
    call in publish_modelio_context. Same shape contract."""
    from mantis_agent.gym import _runner_helpers
    src = inspect.getsource(_runner_helpers.execute_step)
    assert "verify_gate" in src
    # Two distinct ``publish_modelio_context`` invocations could exist
    # in the function (one for each verify_gate call). At least one
    # must wrap a verify_gate.
    pub_count = src.count("publish_modelio_context")
    gate_count = src.count("verify_gate(")
    assert pub_count >= 1, (
        f"execute_step must wrap verify_gate in publish_modelio_context "
        f"(found 0 wraps for {gate_count} verify_gate calls)"
    )


def test_step_index_threaded_into_verifier_context():
    """The wrap must pass ``step_index=index`` so the resulting
    modelio record is tagged with the correct step (Mantis 0-based;
    the adapter bumps to 1-based at the SDK boundary)."""
    from mantis_agent.gym import _runner_helpers
    for fn in (_runner_helpers.ensure_results_filters, _runner_helpers.execute_step):
        src = inspect.getsource(fn)
        if "publish_modelio_context" not in src:
            continue
        # Match: step_index=index OR step_index=<some valid identifier>
        assert re.search(r"step_index\s*=\s*\w+", src), (
            f"{fn.__name__}: publish_modelio_context must pass step_index"
        )
