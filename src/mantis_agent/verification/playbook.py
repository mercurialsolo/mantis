"""Site-specific playbooks — learned knowledge from the Learning Phase.

A Playbook captures verified, site-specific knowledge:
- Which UI elements to click for each filter
- What the page looks like after each action
- Known traps and recovery actions
- Extraction patterns (scroll count, field locations)

Playbooks are JSON files stored on Modal volume at /data/playbooks/.
They're written by the LearningRunner and read by the WorkflowRunner.

Usage:
    # Save after learning
    store = PlaybookStore()
    store.save(playbook)

    # Load for execution
    playbook = store.load("boattrader.com")
    if playbook:
        print(f"Loaded {len(playbook.steps)} verified steps")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class PlaybookStep:
    """A single verified step in a playbook."""
    name: str                       # "apply_private_seller_filter"
    intent: str                     # "Click Private Seller option in sidebar"
    visual_target: str = ""         # "Text 'By Owner' in left sidebar under seller type"
    expected_outcome: str = ""      # "Page heading shows 'by owner', count < 10,000"
    failure_signal: str = ""        # "URL changed to /boats/condition-new/"
    recovery_action: str = ""       # "Navigate to /boats/by-owner/"
    max_attempts: int = 3
    confidence: float = 0.0         # 0-1, updated during learning
    attempts: int = 0               # Total attempts during learning
    successes: int = 0              # Successful attempts during learning

    def update_confidence(self, success: bool):
        """Update confidence based on a learning attempt."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.confidence = self.successes / max(self.attempts, 1)


@dataclass
class ExtractionPattern:
    """Learned pattern for extracting data from a detail page."""
    scrolls_to_description: int = 5     # How many scrolls to reach Description
    scrolls_to_phone: int = 6           # How many scrolls to reach phone area
    phone_location: str = ""            # "In Description section" or "Not visible"
    phone_format: str = ""              # "(305) 555-1234" or "none found"
    has_visible_phone: bool = False     # Whether phones appear at all
    detail_page_signals: list[str] = field(default_factory=list)  # ["boat specs", "seller info"]
    dealer_signals: list[str] = field(default_factory=list)       # ["More From This Dealer"]


@dataclass
class Playbook:
    """Site-specific knowledge from a learning run."""
    domain: str                                     # "boattrader.com"
    created_at: str = ""                            # ISO timestamp
    plan_hash: str = ""                             # Hash of the plan that generated this
    setup_steps: list[PlaybookStep] = field(default_factory=list)
    extraction_steps: list[PlaybookStep] = field(default_factory=list)
    extraction_pattern: ExtractionPattern = field(default_factory=ExtractionPattern)
    known_traps: list[str] = field(default_factory=list)
    page_signals: list[str] = field(default_factory=list)       # Expected signals on results page
    listings_per_page: int = 0                      # Learned average listings per page

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "domain": self.domain,
            "created_at": self.created_at,
            "plan_hash": self.plan_hash,
            "setup_steps": [
                {k: v for k, v in s.__dict__.items()} for s in self.setup_steps
            ],
            "extraction_steps": [
                {k: v for k, v in s.__dict__.items()} for s in self.extraction_steps
            ],
            "extraction_pattern": self.extraction_pattern.__dict__,
            "known_traps": self.known_traps,
            "page_signals": self.page_signals,
            "listings_per_page": self.listings_per_page,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Playbook:
        """Deserialize from dict."""
        pb = cls(
            domain=data.get("domain", ""),
            created_at=data.get("created_at", ""),
            plan_hash=data.get("plan_hash", ""),
            known_traps=data.get("known_traps", []),
            page_signals=data.get("page_signals", []),
            listings_per_page=data.get("listings_per_page", 0),
        )
        for s in data.get("setup_steps", []):
            pb.setup_steps.append(PlaybookStep(**s))
        for s in data.get("extraction_steps", []):
            pb.extraction_steps.append(PlaybookStep(**s))
        ep = data.get("extraction_pattern", {})
        if ep:
            pb.extraction_pattern = ExtractionPattern(**ep)
        return pb

    def summary(self) -> str:
        """Human-readable summary."""
        setup_conf = sum(s.confidence for s in self.setup_steps) / max(len(self.setup_steps), 1)
        extract_conf = sum(s.confidence for s in self.extraction_steps) / max(len(self.extraction_steps), 1)
        return (
            f"Playbook: {self.domain}\n"
            f"  Setup: {len(self.setup_steps)} steps (avg confidence {setup_conf:.0%})\n"
            f"  Extraction: {len(self.extraction_steps)} steps (avg confidence {extract_conf:.0%})\n"
            f"  Traps: {len(self.known_traps)}\n"
            f"  Listings/page: {self.listings_per_page}\n"
            f"  Phone visible: {self.extraction_pattern.has_visible_phone}\n"
            f"  Created: {self.created_at}"
        )


class PlaybookStore:
    """Persist and load playbooks from /data/playbooks/<domain>.json."""

    def __init__(self, base_path: str = "/data/playbooks"):
        self.base_path = base_path

    def _path(self, domain: str) -> str:
        safe = domain.replace(".", "_").replace("/", "_")
        return os.path.join(self.base_path, f"{safe}.json")

    def save(self, playbook: Playbook):
        """Save playbook to disk."""
        if not playbook.created_at:
            playbook.created_at = datetime.now(timezone.utc).isoformat()
        path = self._path(playbook.domain)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(playbook.to_dict(), f, indent=2)
            logger.info(f"Playbook saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save playbook: {e}")

    def load(self, domain: str) -> Playbook | None:
        """Load playbook from disk. Returns None if not found."""
        path = self._path(domain)
        try:
            with open(path) as f:
                data = json.load(f)
            pb = Playbook.from_dict(data)
            logger.info(f"Playbook loaded: {domain} ({len(pb.setup_steps)} setup, {len(pb.extraction_steps)} extract)")
            return pb
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Failed to load playbook: {e}")
            return None

    def exists(self, domain: str) -> bool:
        return os.path.exists(self._path(domain))
