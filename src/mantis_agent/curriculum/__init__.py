"""Modular curriculum knowledge bank with embedding-based selection.

The curriculum is a collection of small, focused technique snippets, each
defined in its own file under ``techniques/``. The agent's prompt does
NOT include all of them — that would overwhelm a small model. Instead,
``select_techniques()`` picks the most relevant snippets for the current
task using a sparse-vector embedding (TF-IDF cosine similarity) over the
technique content, plus an explicit keyword-trigger boost.

Adding a new technique:
    1. Drop a new ``techniques/<name>.py`` file with the constants
       NAME, TAGS, TRIGGERS, ALWAYS, CONTENT.
    2. That's it. The loader auto-discovers it on next import.

Selection model:
    - Documents = each technique's (name + tags + triggers + content)
      flattened into a single text bag-of-words.
    - Query = (instruction + hint_text + domain) bag-of-words.
    - Score = TF-IDF cosine similarity, plus a +0.5 boost if any of the
      technique's regex triggers matches the query (this lets us pin
      strong domain signals like ``\\b132x?43\\b`` even when the
      vocabulary overlap with the technique content is small).
    - Topics with ``ALWAYS=True`` are always included regardless of score.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import List, Optional

from .tfidf import TFIDFIndex
from . import techniques as _techniques_pkg


# ── Technique discovery (cached at first call) ────────────────────────────────

_TECHNIQUES: Optional[List[dict]] = None
_INDEX: Optional[TFIDFIndex] = None
_INDEX_TOPICS: Optional[List[dict]] = None  # mirrors _INDEX position-by-position


def _flatten(t: dict) -> str:
    """Build the searchable bag-of-words document for a technique.

    We index ONLY (name + tags), not the full content. Tags are curated
    keywords that act as the canonical search vocabulary, while content is
    rich prose meant for the model — including it in the index dilutes the
    signal with common words and produces noisy false positives.
    """
    return " ".join([t["name"], " ".join(t.get("tags", []))])


def _load_techniques() -> List[dict]:
    """Auto-discover every module in the ``techniques`` subpackage."""
    techs: List[dict] = []
    for _, mod_name, _ in pkgutil.iter_modules(_techniques_pkg.__path__):
        mod = importlib.import_module(f"{__name__}.techniques.{mod_name}")
        techs.append(
            {
                "name": getattr(mod, "NAME", mod_name),
                "tags": list(getattr(mod, "TAGS", [])),
                "triggers": list(getattr(mod, "TRIGGERS", [])),
                "always": bool(getattr(mod, "ALWAYS", False)),
                "content": getattr(mod, "CONTENT", "").strip(),
            }
        )
    return techs


def _ensure_loaded() -> None:
    global _TECHNIQUES, _INDEX, _INDEX_TOPICS
    if _TECHNIQUES is None:
        _TECHNIQUES = _load_techniques()
        # Build the TF-IDF index over only the non-always topics so the
        # always-on ones don't dilute the IDF weights for the ranking pool.
        _INDEX_TOPICS = [t for t in _TECHNIQUES if not t["always"]]
        _INDEX = TFIDFIndex([_flatten(t) for t in _INDEX_TOPICS])


# ── Public API ────────────────────────────────────────────────────────────────

# Minimum combined score (TF-IDF cosine + trigger boost) for a non-always
# topic to be selected. Empirically, real matches score 0.15-0.35 and noise
# is 0.02-0.08 — 0.10 cleanly separates them.
MIN_RELEVANCE_SCORE = 0.10
# Boost added to a topic's score whenever any of its regex triggers matches
# the query. Larger than MIN_RELEVANCE so a single trigger guarantees inclusion.
TRIGGER_BOOST = 0.5


def select_techniques(
    instruction: str,
    hint_text: str = "",
    domain: str = "",
    max_topics: int = 3,
) -> str:
    """Pick the most relevant curriculum snippets for a task and return them.

    Args:
        instruction: The task instruction text.
        hint_text:   Any hint text already derived from the evaluator config.
        domain:      The OSWorld domain (os, chrome, vs_code, ...).
        max_topics:  Max number of non-always topics to include.

    Returns:
        Concatenated technique content (separated by blank lines), or an
        empty string if nothing relevant is found and there are no always-on
        techniques.
    """
    _ensure_loaded()
    assert _INDEX is not None and _INDEX_TOPICS is not None and _TECHNIQUES is not None

    query = " ".join(filter(None, [instruction, hint_text, domain]))

    # 1. TF-IDF semantic ranking — over-fetch so trigger boosts can promote
    #    topics that would otherwise fall outside the cut.
    ranked = _INDEX.query(query, top_k=len(_INDEX_TOPICS))
    scores = {i: s for s, i in ranked}

    # 2. Keyword trigger boost: any topic whose regex triggers match the query
    #    gets a +TRIGGER_BOOST nudge so explicit signals override weak cosine.
    for i, topic in enumerate(_INDEX_TOPICS):
        for pat in topic.get("triggers", []):
            try:
                if re.search(pat, query, re.IGNORECASE):
                    scores[i] = scores.get(i, 0.0) + TRIGGER_BOOST
                    break
            except re.error:
                continue

    # 3. Apply relevance threshold then pick top max_topics. Topics below
    #    threshold are excluded entirely — better to inject zero noise than
    #    pad to max_topics with weak matches.
    eligible = [(s, i) for i, s in scores.items() if s >= MIN_RELEVANCE_SCORE]
    eligible.sort(key=lambda x: -x[0])
    chosen = [_INDEX_TOPICS[i] for _, i in eligible[:max_topics]]

    # 4. Append always-on topics (these are unconditional and added on top)
    always = [t for t in _TECHNIQUES if t["always"]]

    final = chosen + always
    if not final:
        return ""
    return "\n\n".join(t["content"] for t in final)


def list_techniques() -> List[str]:
    """Return the names of all loaded techniques (for diagnostics)."""
    _ensure_loaded()
    assert _TECHNIQUES is not None
    return [t["name"] for t in _TECHNIQUES]


def reload() -> None:
    """Force a re-discovery of techniques. Useful for tests."""
    global _TECHNIQUES, _INDEX, _INDEX_TOPICS
    _TECHNIQUES = None
    _INDEX = None
    _INDEX_TOPICS = None
    _ensure_loaded()


__all__ = ["select_techniques", "list_techniques", "reload"]
