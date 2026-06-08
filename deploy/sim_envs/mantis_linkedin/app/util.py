"""Small utility helpers shared across the app."""

from __future__ import annotations

import re
from typing import Iterable

_HASHTAG_RE = re.compile(r"#(\w{2,40})")


def extract_hashtags(body: str) -> list[str]:
    """Return the unique hashtags found in ``body``, lower-cased, no leading #."""
    if not body:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for m in _HASHTAG_RE.finditer(body):
        tag = m.group(1).lower()
        if tag in seen:
            continue
        seen.add(tag)
        found.append(tag)
    return found


def truncate(text: str, *, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:1].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


def join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(t for t in tokens if t)
