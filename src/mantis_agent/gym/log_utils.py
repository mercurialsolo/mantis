"""Tiny logging helpers shared across the gym layer.

Lives outside ``_runner_helpers`` so it has no transitive imports and
can be safely consumed by modules that ``_runner_helpers`` itself
imports (e.g. :mod:`step_snapshot`). Keep this module dependency-free.
"""

from __future__ import annotations


def url_for_log(url: str, limit: int = 200) -> str:
    """Format ``url`` for log readability without silently mid-truncating.

    The previous ``url[:40]`` style silently truncated paths mid-segment,
    producing misleading log lines like ``/leads/1`` when the real URL
    ended in ``/leads/13`` (the truncation boundary cut off the trailing
    digit). Two log lines that disagree because of cropping look
    identical to a real verifier disagreement and burned debugging time
    on #209 Symptom 1 / Finding #2.

    Behaviour:
      • ``len(url) <= limit`` → return the URL untouched.
      • ``len(url) > limit``  → return ``url[:limit] + '…'`` so a reader
        can see truncation happened. The default ``200`` is large
        enough that any host + path the framework cares about for
        verify decisions is preserved verbatim; only long query
        strings or fragments hit the ellipsis path.

    Generic primitive — no domain knowledge. Use everywhere a URL is
    inlined into a log message that another log line might be compared
    against.
    """
    if not url:
        return ""
    if len(url) <= limit:
        return url
    return url[:limit] + "…"
