"""Loop guards — detect state repeats + off-source drift (#782).

Framework-level halt detectors used by `WorkflowRunner` (and any other
loop-driven runner). Both guards are stateful — they consume observed
(url, screenshot, text) tuples and emit a `LoopGuardHalt` when the
configured threshold is exceeded.

Detection model:

- **State-repeat** — fingerprint each iteration as
  `(url, sha256(screenshot_png), sha256(visible_text))`. If `K`
  consecutive fingerprints match, halt with `loop_stuck`.
- **Off-source drift** — pin a URL pattern at loop entry. If the
  active URL leaves the pattern for `M` consecutive observations
  without an explicit return, halt with `off_source_drift`.

Both guards are opt-in via `LoopConfig` fields (PR 5 adds them).
Default-off so existing plans keep their current behavior; recipe
authors / plan authors enable them per loop.

The guards do **not** know about runners or step handlers. The
integration layer (`WorkflowRunner._iter_loop`) calls `observe()`
once per iteration and checks the return value.

These guards are the framework-level analogue of the `cf_challenge`
auto-pause (#555 / `feedback_cf_challenge_diagnostic_shortcut`):
detect "we've left the intended page" and halt cleanly rather than
spinning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


HaltClass = str

LOOP_STUCK: HaltClass = "loop_stuck"
OFF_SOURCE_DRIFT: HaltClass = "off_source_drift"


@dataclass
class LoopGuardHalt:
    """Signal returned by a guard when its threshold trips.

    `halt_class` is one of the constants above. `reason` is a short
    human-readable description for logs / Augur tags. `evidence` is
    a guard-specific dict (e.g. consecutive fingerprints, off-pattern
    URL trace) used in tests and `/v1/runs/{id}/trace` (PR 6).
    """

    halt_class: HaltClass
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def fingerprint(url: str, screenshot_bytes: bytes, visible_text: str) -> str:
    """Stable 32-char hex digest of an iteration state.

    Reducing the page to one digest lets us compare consecutive
    iterations cheaply. Screenshot is included because pages with the
    same URL + text but different visual state (e.g. modal open / closed)
    are NOT the same iteration; URL alone is too narrow.
    """
    h = hashlib.sha256()
    h.update(url.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(screenshot_bytes or b"")
    h.update(b"\x00")
    h.update((visible_text or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()[:32]


def _matches_origin(url: str, pattern: str) -> bool:
    """Match a URL against a pinned-origin pattern.

    Patterns:
      - `https://example.com` — exact origin (path may extend).
      - `https://example.com/*` — origin + any path.
      - `https://example.com/foo/*` — origin + path prefix.
    Empty pattern → always matches (no pin → no drift possible).
    """
    if not pattern:
        return True
    if pattern.endswith("/*"):
        prefix = pattern[:-2]
        return url.startswith(prefix)
    if not url.startswith(pattern):
        return False
    if len(url) == len(pattern):
        return True
    return url[len(pattern)] in ("/", "?", "#")


@dataclass
class StateRepeatGuard:
    """Halt when `threshold` consecutive iterations have an identical
    state fingerprint.

    `threshold` of 0 disables the guard. Default 0 keeps the guard
    no-op until a plan opts in.

    Use case: stuck-in-back-nav / Chrome side-panel / blank-page loops
    that re-render the same screen indefinitely. The HN URL-collection
    user report (#785) called this out specifically.
    """

    threshold: int = 0
    _last_fp: str | None = None
    _streak: int = 0
    _streak_url: str = ""

    def observe(
        self, *, url: str, screenshot_bytes: bytes, visible_text: str
    ) -> LoopGuardHalt | None:
        if self.threshold <= 0:
            return None
        fp = fingerprint(url, screenshot_bytes, visible_text)
        if fp == self._last_fp:
            self._streak += 1
        else:
            self._streak = 1
            self._last_fp = fp
            self._streak_url = url
        if self._streak >= self.threshold:
            return LoopGuardHalt(
                halt_class=LOOP_STUCK,
                reason=f"{self._streak} consecutive iterations had identical state fingerprint",
                evidence={
                    "fingerprint": fp,
                    "url": self._streak_url,
                    "streak": self._streak,
                },
            )
        return None

    def reset(self) -> None:
        """Clear streak — call when the plan explicitly transitions to
        a new section (e.g. a `gate` step passes). Avoids spurious
        halts on legitimate loops that revisit the same page.
        """
        self._last_fp = None
        self._streak = 0
        self._streak_url = ""


@dataclass
class OffSourceDriftGuard:
    """Halt when the active URL has been off the pinned pattern for
    `step_budget` consecutive observations.

    `pinned_pattern` empty disables the guard. `step_budget` of 0
    disables the guard regardless of pattern.

    Use case: plan started on `https://news.ycombinator.com/*`, an
    accidental click navigated to a comments page, the runner kept
    scrolling — should halt instead of doing useful work on the
    wrong page.
    """

    pinned_pattern: str = ""
    step_budget: int = 0
    _off_streak: int = 0
    _off_urls: list[str] = field(default_factory=list)

    def observe(self, *, url: str) -> LoopGuardHalt | None:
        if not self.pinned_pattern or self.step_budget <= 0:
            return None
        if _matches_origin(url, self.pinned_pattern):
            self._off_streak = 0
            self._off_urls.clear()
            return None
        self._off_streak += 1
        # Keep last 4 off-pattern URLs for the evidence trail.
        self._off_urls.append(url)
        if len(self._off_urls) > 4:
            self._off_urls = self._off_urls[-4:]
        if self._off_streak >= self.step_budget:
            return LoopGuardHalt(
                halt_class=OFF_SOURCE_DRIFT,
                reason=(
                    f"{self._off_streak} consecutive steps off pinned origin "
                    f"{self.pinned_pattern!r}"
                ),
                evidence={
                    "pinned_pattern": self.pinned_pattern,
                    "off_streak": self._off_streak,
                    "off_urls_trail": list(self._off_urls),
                    "current_url": url,
                },
            )
        return None

    def reset(self) -> None:
        self._off_streak = 0
        self._off_urls.clear()


@dataclass
class LoopGuardSuite:
    """Compose both guards for a runner. Single observe() entry point."""

    state_repeat: StateRepeatGuard = field(default_factory=StateRepeatGuard)
    off_source: OffSourceDriftGuard = field(default_factory=OffSourceDriftGuard)

    def observe(
        self,
        *,
        url: str,
        screenshot_bytes: bytes = b"",
        visible_text: str = "",
    ) -> LoopGuardHalt | None:
        # Off-source is cheaper — check first. If both would trip, the
        # off-source halt is more diagnostic (tells operators *what*
        # went wrong, not just that nothing changed).
        halt = self.off_source.observe(url=url)
        if halt is not None:
            return halt
        return self.state_repeat.observe(
            url=url, screenshot_bytes=screenshot_bytes, visible_text=visible_text
        )

    def reset(self) -> None:
        self.state_repeat.reset()
        self.off_source.reset()


__all__ = [
    "LOOP_STUCK",
    "OFF_SOURCE_DRIFT",
    "LoopGuardHalt",
    "LoopGuardSuite",
    "OffSourceDriftGuard",
    "StateRepeatGuard",
    "fingerprint",
]
