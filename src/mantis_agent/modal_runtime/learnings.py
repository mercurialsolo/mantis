"""Cross-run learnings persistence on the shared Modal volume.

A "learning" is an entry the OSWorld harness writes after a failed task
so the next agent run can read about prior attempts. Stored as JSON at
``/data/results/learnings.json`` on the shared volume.
"""

from __future__ import annotations

import json


def load_learnings(volume_path: str = "/data/results/learnings.json") -> list:
    """Load accumulated learnings from previous runs."""
    try:
        with open(volume_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def get_prior_learning(task_id: str, instruction: str, learnings: list) -> str:
    """Find relevant prior learnings for a task — by task ID or similar instruction."""
    relevant = []

    for entry in learnings:
        # Exact task match
        if entry.get("task_id") == task_id:
            relevant.append(entry)
            continue
        # Similar instruction (share 3+ words)
        prior_words = set(entry.get("instruction", "").lower().split())
        current_words = set(instruction.lower().split())
        overlap = prior_words & current_words - {
            "the", "a", "to", "in", "on", "my", "i", "can", "you", "help", "me", "is",
        }
        if len(overlap) >= 4:
            relevant.append(entry)

    if not relevant:
        return ""

    # Distill prior learnings into actionable advice
    advice = "\nPrior learnings from similar tasks:"
    for entry in relevant[-3:]:  # Last 3 relevant entries
        diag = entry.get("diagnosis", "")
        actions = entry.get("actions_tried", [])
        if diag:
            advice += f"\n- {diag}"
        if actions:
            advice += f" (failed approaches: {', '.join(a[:40] for a in actions)})"
    return advice
