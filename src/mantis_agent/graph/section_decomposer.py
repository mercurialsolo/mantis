"""SectionDecomposer — group plan phases into Holo3-sized execution sections.

Holo3 passes 100% on isolated 3-8 step tasks but fails when instructions
are combined. This module groups verified PhaseNodes into logical sections,
each small enough for the executor, with explicit dependency chains.

Each section has:
  - precondition: what must be true before it runs
  - steps: 1-8 PhaseNodes (the work)
  - postcondition: what should be true after
  - depends_on: which sections must complete first

The dependency chain ensures sections execute in order:
  setup_section → gate_section → extraction_section (loops) → pagination_section
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .graph import PhaseNode, PhaseRole, RepeatMode

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSection:
    """A group of phases small enough for Holo3 to execute."""

    id: str
    name: str  # "setup", "gate", "extraction", "pagination"
    phases: list[PhaseNode] = field(default_factory=list)
    precondition: str = ""
    postcondition: str = ""
    depends_on: list[str] = field(default_factory=list)
    is_loop: bool = False  # True if this section repeats
    loop_source: str = ""  # Section ID that provides items

    def step_count(self) -> int:
        return sum(max(p.budget, 1) for p in self.phases)

    def summary(self) -> str:
        phase_ids = [p.id for p in self.phases]
        deps = f" (depends: {', '.join(self.depends_on)})" if self.depends_on else ""
        loop = " [LOOP]" if self.is_loop else ""
        return f"  {self.id:20s} {len(self.phases)} phases, ~{self.step_count()} steps{loop}{deps}"


class SectionDecomposer:
    """Group PhaseNodes into Holo3-sized execution sections."""

    def decompose(
        self,
        phases: dict[str, PhaseNode],
        max_steps_per_section: int = 15,
    ) -> list[ExecutionSection]:
        """Group phases into sections with dependency chains.

        Returns ordered list of ExecutionSections.
        """
        sections: list[ExecutionSection] = []

        # ── Collect by role ──
        setup_phases = [p for p in phases.values() if p.role == PhaseRole.SETUP and not p.gate]
        gate_phases = [p for p in phases.values() if p.gate or p.role == PhaseRole.GATE]
        extraction_phases = [
            p for p in phases.values()
            if p.role in (PhaseRole.ADMISSION, PhaseRole.EXTRACTION, PhaseRole.REJECTION)
            and p.repeat == RepeatMode.FOR_EACH
        ]
        return_phases = [p for p in phases.values() if p.role == PhaseRole.RETURN]
        pagination_phases = [p for p in phases.values() if p.role == PhaseRole.PAGINATION]

        # ── Setup section (navigate + filters) ──
        if setup_phases:
            # Split into sub-sections if too many steps
            current_batch: list[PhaseNode] = []
            batch_steps = 0
            batch_idx = 0

            for phase in setup_phases:
                phase_steps = max(phase.budget, 1)
                if batch_steps + phase_steps > max_steps_per_section and current_batch:
                    sid = f"setup_{batch_idx}" if batch_idx > 0 else "setup"
                    dep = [f"setup_{batch_idx - 1}"] if batch_idx > 0 else []
                    sections.append(ExecutionSection(
                        id=sid, name="setup",
                        phases=list(current_batch),
                        precondition="Browser ready" if batch_idx == 0 else "Previous setup complete",
                        postcondition="Setup steps applied",
                        depends_on=dep,
                    ))
                    current_batch = []
                    batch_steps = 0
                    batch_idx += 1
                current_batch.append(phase)
                batch_steps += phase_steps

            if current_batch:
                sid = f"setup_{batch_idx}" if batch_idx > 0 else "setup"
                dep = [f"setup_{batch_idx - 1}"] if batch_idx > 0 else []
                sections.append(ExecutionSection(
                    id=sid, name="setup",
                    phases=current_batch,
                    precondition="Browser ready" if batch_idx == 0 else "Previous setup complete",
                    postcondition="Setup steps applied",
                    depends_on=dep,
                ))

        # ── Gate section ──
        if gate_phases:
            last_setup = sections[-1].id if sections else ""
            sections.append(ExecutionSection(
                id="gate",
                name="gate",
                phases=gate_phases,
                precondition="All filters applied",
                postcondition="Filters verified active, page shows filtered results",
                depends_on=[last_setup] if last_setup else [],
            ))

        # ── Extraction section (the loop body) ──
        if extraction_phases or return_phases:
            loop_phases = list(extraction_phases) + list(return_phases)
            gate_dep = "gate" if any(s.id == "gate" for s in sections) else ""
            sections.append(ExecutionSection(
                id="extraction",
                name="extraction",
                phases=loop_phases,
                precondition="On filtered results page, candidate card visible",
                postcondition="Item data extracted, returned to results page",
                depends_on=[gate_dep] if gate_dep else [],
                is_loop=True,
                loop_source="discovery",
            ))

        # ── Pagination section ──
        if pagination_phases:
            sections.append(ExecutionSection(
                id="pagination",
                name="pagination",
                phases=pagination_phases,
                precondition="Current page exhausted (all items processed)",
                postcondition="Next results page loaded",
                depends_on=["extraction"],
                is_loop=True,
                loop_source="extraction",
            ))

        logger.info(
            "SectionDecomposer: %d phases -> %d sections",
            len(phases), len(sections),
        )
        for section in sections:
            logger.info(section.summary())

        return sections
