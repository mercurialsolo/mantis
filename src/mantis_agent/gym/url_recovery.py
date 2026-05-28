"""URL rewrite recovery sources — plan-evolution Phase 1 (#705).

Consumed by :func:`mantis_agent.agentic_recovery.analyse_failure_and_recover`
when a navigate step fails with ``failure_class='bad_url'``. Each source
emits zero or more :class:`RewriteProposal` candidates; the recovery loop
picks the highest-confidence proposal and returns an ``edit_step``
RecoveryDecision that rewrites the step's intent + ``params.url``.

Phase 1 ships two sources (web_search deferred):

1. ``pattern_transform`` (~free) — pure-Python rule set for common URL
   drifts (slug normalisation, trailing slash, case folding). Confidence
   0.6 — pattern matches but doesn't validate semantic correctness.

2. ``page_links`` (~$0.001) — read ``a[href]`` from the currently-loaded
   page (homepage / last good page) via CDP; score by text overlap with
   the failed step's intent + the original URL's path tokens. Confidence
   0.7 — page-derived but heuristic.

Phase 1 does NOT validate proposals (no HEAD-200 check). A bad proposal
fails the next navigate, the recovery budget eventually exhausts, and
the run halts cleanly. Validation lands in Phase 2 (#706) once we have
real production data on how often pattern_transform produces invalid
URLs.

CUA-purity: ``page_links`` reads DOM only to *propose alternative URLs*.
The brain still has to navigate to the proposal and verify visually —
the URL is action-target, not grounding-derivation. See
``feedback_cua_no_dom_access.md`` for the boundary.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from .url_health import _normalize_netloc, expand_expected_domains

logger = logging.getLogger(__name__)

RewriteSource = Literal["pattern_transform", "page_links"]


@dataclass
class RewriteProposal:
    """A candidate URL rewrite + provenance.

    Recovery picks the highest-confidence proposal; ties broken by
    source order (pattern_transform before page_links).
    """

    new_url: str
    source: RewriteSource
    confidence: float           # 0.0–1.0
    notes: str = ""             # short human-readable rationale
    matched_link_text: str = "" # populated by page_links only


@dataclass
class _PageLink:
    text: str
    href: str


# ── pattern_transform ────────────────────────────────────────────────


def _pattern_transform(failed_url: str) -> list[RewriteProposal]:
    """Generate URL variants via common drift patterns.

    Each rule emits at most one proposal. Caller dedupes by `new_url`
    in the assembly step (multiple rules may converge on the same
    transform).
    """
    parsed = urlparse(failed_url)
    if not parsed.scheme or not parsed.netloc:
        return []

    proposals: list[RewriteProposal] = []
    path = parsed.path or "/"

    # Rule 1: trailing slash flip. /boats/by-owner ↔ /boats/by-owner/
    if path.endswith("/") and len(path) > 1:
        alt_path = path.rstrip("/")
        proposals.append(_make_proposal(
            parsed, alt_path,
            notes="strip-trailing-slash",
        ))
    elif not path.endswith("/") and "/" in path:
        proposals.append(_make_proposal(
            parsed, path + "/",
            notes="add-trailing-slash",
        ))

    # Rule 2: hyphen ↔ slash in slug. /boats/state-fl/ ↔ /boats/state/fl/
    # Apply only when the path contains a hyphen-joined slug that LOOKS
    # like two tokens (lowercase + 2-char state code is a strong tell on
    # marketplace plans; keep the rule conservative to avoid spurious
    # transforms on /style-guide etc.).
    for m in re.finditer(r"/([a-z]+)-([a-z]{2})(/|$)", path):
        before = path[: m.start()]
        after = path[m.end() - 1 :]  # keep trailing / or ""
        transformed = f"{before}/{m.group(1)}/{m.group(2)}{after}"
        if transformed != path:
            proposals.append(_make_proposal(
                parsed, transformed,
                notes=f"slug-split:{m.group(1)}-{m.group(2)}",
            ))
        # And the reverse: /state/fl ↔ /state-fl
    for m in re.finditer(r"/([a-z]+)/([a-z]{2})(/|$)", path):
        before = path[: m.start()]
        after = path[m.end() - 1 :]
        transformed = f"{before}/{m.group(1)}-{m.group(2)}{after}"
        if transformed != path:
            proposals.append(_make_proposal(
                parsed, transformed,
                notes=f"slug-merge:{m.group(1)}-{m.group(2)}",
            ))

    # Rule 3: drop www. or add www. — covers domain-level drift where
    # the canonical host moved between bare and www variants.
    netloc_no_port = parsed.netloc.split(":", 1)[0]
    port_suffix = ""
    if ":" in parsed.netloc:
        port_suffix = ":" + parsed.netloc.split(":", 1)[1]
    if netloc_no_port.startswith("www."):
        alt = parsed._replace(netloc=netloc_no_port[4:] + port_suffix)
        proposals.append(RewriteProposal(
            new_url=urlunparse(alt),
            source="pattern_transform",
            confidence=0.55,
            notes="strip-www",
        ))
    else:
        alt = parsed._replace(netloc="www." + netloc_no_port + port_suffix)
        proposals.append(RewriteProposal(
            new_url=urlunparse(alt),
            source="pattern_transform",
            confidence=0.55,
            notes="add-www",
        ))

    # Rule 4: case-fold query params. ?param=Value → ?param=value
    if parsed.query and parsed.query != parsed.query.lower():
        alt = parsed._replace(query=parsed.query.lower())
        proposals.append(RewriteProposal(
            new_url=urlunparse(alt),
            source="pattern_transform",
            confidence=0.5,
            notes="lower-query",
        ))

    # Dedupe by new_url, preserving first occurrence (highest confidence
    # since we appended in confidence order).
    seen: set[str] = set()
    deduped: list[RewriteProposal] = []
    for p in proposals:
        if p.new_url == failed_url or p.new_url in seen:
            continue
        seen.add(p.new_url)
        deduped.append(p)
    return deduped


def _make_proposal(
    parsed: Any, new_path: str, *, notes: str, confidence: float = 0.6,
) -> RewriteProposal:
    """Build a proposal that swaps just the path component."""
    new_url = urlunparse(parsed._replace(path=new_path))
    return RewriteProposal(
        new_url=new_url,
        source="pattern_transform",
        confidence=confidence,
        notes=notes,
    )


# ── page_links ────────────────────────────────────────────────────────


_LINK_EXTRACT_JS = (
    "Array.from(document.querySelectorAll('a[href]'))"
    ".slice(0, 200)"
    ".map(a => ({text: (a.innerText || a.title || a.ariaLabel || '').trim().slice(0, 120), "
    "href: a.href}))"
    ".filter(l => l.href && l.href.startsWith('http'))"
)


def _page_links(
    env: Any,
    failed_url: str,
    intent_text: str,
    *,
    max_links: int = 200,
    min_score: float = 0.25,
) -> list[RewriteProposal]:
    """Scan the currently-loaded page for links that match the intent.

    Degrades silently when env doesn't expose `cdp_evaluate` — many
    test envs and the local CLI fallback don't have CDP wired.
    """
    cdp = getattr(env, "cdp_evaluate", None)
    if not callable(cdp):
        return []

    try:
        raw = cdp(_LINK_EXTRACT_JS)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("page_links cdp_evaluate raised: %s", exc)
        return []

    links = _coerce_links(raw, max_links=max_links)
    if not links:
        return []

    expected = expand_expected_domains(failed_url)
    intent_tokens = _tokenize(intent_text)
    failed_tokens = _tokenize(_url_to_text(failed_url))
    target_tokens = intent_tokens | failed_tokens
    if not target_tokens:
        return []

    scored: list[tuple[float, _PageLink]] = []
    for link in links:
        link_netloc = _normalize_netloc(urlparse(link.href).netloc)
        if expected and link_netloc not in expected:
            # Off-domain link — skip. We don't follow cross-site
            # redirects from this source.
            continue
        link_tokens = _tokenize(link.text) | _tokenize(_url_to_text(link.href))
        if not link_tokens:
            continue
        overlap = len(target_tokens & link_tokens)
        score = overlap / max(len(target_tokens), 1)
        if score >= min_score and link.href != failed_url:
            scored.append((score, link))

    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])

    # Top 3 — confidence scales with score (0.5–0.75 range).
    proposals: list[RewriteProposal] = []
    for score, link in scored[:3]:
        confidence = round(min(0.75, 0.5 + score * 0.25), 3)
        proposals.append(RewriteProposal(
            new_url=link.href,
            source="page_links",
            confidence=confidence,
            notes=f"score={score:.2f}",
            matched_link_text=link.text[:80],
        ))
    return proposals


def _coerce_links(raw: Any, *, max_links: int) -> list[_PageLink]:
    if not isinstance(raw, list):
        return []
    out: list[_PageLink] = []
    for entry in raw[:max_links]:
        if not isinstance(entry, dict):
            continue
        href = str(entry.get("href", "") or "").strip()
        text = str(entry.get("text", "") or "").strip()
        if href:
            out.append(_PageLink(text=text, href=href))
    return out


# Common stopwords that hurt overlap scoring for URL-shaped strings
# (most marketplace URLs include `boats`, `by-owner`, etc. that aren't
# discriminating enough to drive a rewrite).
_STOPWORDS: frozenset[str] = frozenset({
    "http", "https", "www", "com", "org", "net",
    "the", "a", "an", "to", "for", "in", "of", "and", "or",
    "navigate", "go", "open", "visit", "page", "url",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase + alphanumeric word split + stopword strip.

    Keeps tokens ≥ 2 chars to avoid noise from single-char fragments.
    """
    if not text:
        return set()
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if len(t) >= 2 and t not in _STOPWORDS}


def _url_to_text(url: str) -> str:
    """Path + query tokens, hyphen / slash / underscore replaced with space."""
    parsed = urlparse(url)
    parts = [parsed.path or "", parsed.query or ""]
    return " ".join(parts).replace("-", " ").replace("/", " ").replace("_", " ")


# ── orchestrator ──────────────────────────────────────────────────────


@dataclass
class UrlRecoveryReport:
    """Surface what each source produced — used in logs + result.json."""

    proposals: list[RewriteProposal] = field(default_factory=list)
    sources_tried: list[str] = field(default_factory=list)
    sources_skipped: dict[str, str] = field(default_factory=dict)


def propose_url_rewrites(
    *,
    failed_url: str,
    failure_subclass: str,
    intent_text: str = "",
    env: Any = None,
) -> UrlRecoveryReport:
    """Run all sources, return ordered proposals + diagnostic report.

    Sources run in cheapness order; results are merged + sorted by
    confidence DESC. Empty proposal list means no source found a
    candidate — caller halts.

    Args:
        failed_url: The URL the navigate step tried to load.
        failure_subclass: `dns` / `not_found` / `wrong_domain` /
            `soft_404` — informs which sources are worth running.
            `blocked` is filtered upstream (handled by external_pause).
        intent_text: The plan step's intent prose; used by page_links
            for relevance scoring.
        env: Optional `GymEnvironment` for sources that need CDP
            (page_links). Absent → page_links skipped silently.
    """
    report = UrlRecoveryReport()

    if failure_subclass == "dns":
        # DNS failure → no page loaded → page_links can't run; pattern
        # transforms are the only useful source.
        report.sources_tried.append("pattern_transform")
        report.proposals.extend(_pattern_transform(failed_url))
        report.sources_skipped["page_links"] = "dns_no_page_loaded"
    else:
        # not_found / wrong_domain / soft_404 — both sources useful.
        report.sources_tried.append("pattern_transform")
        report.proposals.extend(_pattern_transform(failed_url))

        report.sources_tried.append("page_links")
        page_proposals = _page_links(env, failed_url, intent_text)
        if not page_proposals and not callable(getattr(env, "cdp_evaluate", None)):
            report.sources_skipped["page_links"] = "env_lacks_cdp_evaluate"
        report.proposals.extend(page_proposals)

    # Sort by confidence DESC; ties broken by source preference
    # (pattern_transform before page_links per spec).
    source_order = {"pattern_transform": 0, "page_links": 1}
    report.proposals.sort(
        key=lambda p: (-p.confidence, source_order.get(p.source, 99))
    )
    return report
