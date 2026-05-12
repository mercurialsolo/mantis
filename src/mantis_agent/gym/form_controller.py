"""Runtime form controller — first-class form-filling capability for #301.

Consolidates the scattered ``force_fill_*`` state previously held on
``GymRunner.run``'s stack into a single object that owns:

  * ``pending_values``  — list of ``{"label", "value"}`` extracted from the
    plan via Holo3 vision; consumed FIFO or by label-match.
  * ``used_regions``    — coordinate regions already typed into, so the
    same field is never typed twice.
  * ``submitted``       — whether the auto-submit branch already pressed
    Return / clicked the visible submit button.
  * ``initial_labels``  — snapshot of the labels at episode start, used by
    the Claude director and by the done-acceptance gate to compare against
    the current pending list.

Behavioural responsibilities (per #301 acceptance):

  1. **Detect focused/target input** by DOM when available, otherwise by
     screenshot detector — delegates to ``holo3_detector.detect_focused_field``
     today; a future DOM-aware path can land behind the same surface.
  2. **Click/focus once** — substitution short-circuits a re-click on a
     field already in ``used_regions``.
  3. **Type via the strongest backend** — CDP ``Input.insertText`` first,
     paste second, raw xdotool last. Backend selection lives in
     :mod:`xdotool_env._cdp_insert_text`; the controller's job is to
     decide *when* to type, not *how*.
  4. **Verify the value landed** when DOM access is available — wired
     through ``gym_result.info["type_verified"]`` by the env adapter.
  5. **Submit with Enter** after the last credential/search field unless
     a submit target is explicitly required — delegates to the existing
     auto-submit logic in the runner; the controller exposes
     ``should_force_submit`` so the gate is in one place.
  6. **Update force-fill state** when an external director or fallback
     action moves focus — exposed as :meth:`mark_consumed_label` so values
     are not typed twice when something other than the controller's own
     substitution path advances the workflow.

The ablation toggle ``MANTIS_FORM_CONTROLLER=disabled`` keeps the runner
on the old static-method path (no controller object). Default ON; the
controller's behaviour is identical to the pre-refactor scattered code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from ..actions import Action

logger = logging.getLogger(__name__)


@dataclass
class FormController:
    """Owns the per-episode runtime form-filling state and decisions.

    Construct via :meth:`from_task` (which runs the Holo3 extractor) or
    directly with ``pending_values=[...]`` for tests / programmatic seeds.
    """

    pending_values: list[dict] = field(default_factory=list)
    used_regions: list[tuple[int, int]] = field(default_factory=list)
    submitted: bool = False
    initial_labels: list[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, brain: Any, task: str) -> "FormController":
        """Build a controller by extracting plan values via Holo3 vision.

        Mirrors the inline call previously at the top of ``GymRunner.run``.
        ``initial_labels`` is snapshotted before any consumption so the
        director / done-gate can compare ``initial - pending = consumed``.
        """
        from . import holo3_detector

        values = holo3_detector.extract_form_values(brain, task)
        labels = [str(v.get("label") or "") for v in values]
        return cls(
            pending_values=values,
            used_regions=[],
            submitted=False,
            initial_labels=labels,
        )

    # ── Read-only views ────────────────────────────────────────────────

    @property
    def has_pending(self) -> bool:
        return bool(self.pending_values)

    @property
    def pending_count(self) -> int:
        return len(self.pending_values)

    @property
    def initial_count(self) -> int:
        return len(self.initial_labels)

    @property
    def consumed_count(self) -> int:
        return self.initial_count - self.pending_count

    @property
    def pending_labels(self) -> list[str]:
        """Labels of values that haven't been consumed yet — for gate /
        director consumers (the done-acceptance gate uses this to detect
        the run-023 'claimed success with credentials still pending' case)."""
        return [str(v.get("label") or "") for v in self.pending_values]

    # ── External-mover hooks (#301 acceptance bullet 6) ────────────────

    def mark_consumed_label(self, label: str) -> bool:
        """Drop the first pending entry whose label matches ``label`` (case-
        insensitive substring, both directions). Returns True iff something
        was consumed.

        Use this when a director / fallback path types a value outside the
        controller's own substitution flow — without this hook the
        controller would try to type the same value again on a later step.
        """
        if not label:
            return False
        needle = label.lower().strip()
        if not needle:
            return False
        for idx, entry in enumerate(self.pending_values):
            entry_label = str(entry.get("label") or "").lower().strip()
            if not entry_label:
                continue
            if needle in entry_label or entry_label in needle:
                self.pending_values.pop(idx)
                return True
        return False

    def mark_used_region(self, x: int, y: int) -> None:
        """Record a coordinate as already-typed-into. ``maybe_substitute_*``
        skip clicks within ±20 px of any used region.
        """
        self.used_regions.append((int(x), int(y)))

    def mark_submitted(self) -> None:
        """Latch the submitted flag — auto-submit fires at most once per run."""
        self.submitted = True

    # ── Decision API (delegates to the static helpers on GymRunner) ────

    def maybe_substitute_click_with_type(
        self,
        action: "Action",
        action_history: list["Action"],
        brain: Any,
        screenshot: "Image.Image | None",
    ) -> "Action | None":
        """Holo3-vision-backed substitution: a click on a focused input
        becomes a ``type_text`` of the next matching plan value.

        Mirrors ``GymRunner._maybe_force_type_text``. The static helper
        stays for back-compat (existing tests call it directly); this
        method is the controller-shaped surface for new callers.

        Mutates ``self.pending_values`` and ``self.used_regions`` on
        successful substitution.
        """
        from .runner import GymRunner

        return GymRunner._maybe_force_type_text(
            action, action_history,
            self.pending_values, self.used_regions,
            brain, screenshot,
        )

    def maybe_substitute_repeated_click(
        self,
        action: "Action",
        action_history: list["Action"],
        task: str,
    ) -> "Action | None":
        """Geometric repeat-click substitution: when the brain re-clicks
        the same field twice on a form-shaped task, type the next plan
        value instead.

        Mirrors ``GymRunner._maybe_force_type_after_repeated_form_click``.
        """
        from .runner import GymRunner

        return GymRunner._maybe_force_type_after_repeated_form_click(
            action, action_history,
            self.pending_values, self.used_regions,
            task,
        )

    def should_finish_task(self, task: str) -> bool:
        """Whether the runner should auto-emit DONE for one-value field
        tasks the controller has fully consumed (zip code, radius, etc.)."""
        from .runner import GymRunner

        return GymRunner._force_fill_should_finish_task(
            task=task,
            initial_value_count=self.initial_count,
            pending_value_count=self.pending_count,
            submitted=self.submitted,
        )

    def finish_task_actions(self, task: str) -> list["Action"]:
        """The post-type commit sequence (Tab / Return) for one-value
        field tasks the controller just fully consumed."""
        from .runner import GymRunner

        return GymRunner._force_fill_post_type_actions(task)
