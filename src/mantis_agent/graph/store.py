"""GraphStore — persist and load WorkflowGraphs.

Cache key: domain + objective_hash (different objectives on the same
domain get separate graphs).

Path: /data/graphs/{domain}_{objective_hash[:12]}.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from .graph import WorkflowGraph

logger = logging.getLogger(__name__)


class GraphStore:
    """Persist and load WorkflowGraphs from disk."""

    def __init__(self, base_path: str = "/data/graphs"):
        self.base_path = base_path

    def _path(self, domain: str, objective_hash: str) -> str:
        safe_domain = domain.replace(".", "_").replace("/", "_")
        return os.path.join(self.base_path, f"{safe_domain}_{objective_hash[:12]}.json")

    def save(self, graph: WorkflowGraph) -> str:
        """Save a WorkflowGraph to disk. Returns the file path."""
        if not graph.created_at:
            graph.created_at = datetime.now(timezone.utc).isoformat()
        path = self._path(graph.domain, graph.objective_hash)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            logger.info("WorkflowGraph saved: %s", path)
        except Exception as e:
            logger.warning("Failed to save WorkflowGraph: %s", e)
        return path

    def load(self, domain: str, objective_hash: str) -> WorkflowGraph | None:
        """Load a WorkflowGraph from disk. Returns None if not found."""
        path = self._path(domain, objective_hash)
        try:
            with open(path) as f:
                data = json.load(f)
            graph = WorkflowGraph.from_dict(data)
            logger.info(
                "WorkflowGraph loaded: %s (%d phases)",
                domain,
                len(graph.phases),
            )
            return graph
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("Failed to load WorkflowGraph: %s", e)
            return None

    def exists(self, domain: str, objective_hash: str) -> bool:
        return os.path.exists(self._path(domain, objective_hash))
