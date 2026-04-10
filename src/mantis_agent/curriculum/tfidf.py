"""Pure-Python TF-IDF embedder for curriculum technique selection.

This gives us a true sparse vector representation of each technique
(every word is a dimension, weighted by inverse document frequency)
without any heavy dependencies. Cosine similarity over these vectors
ranks topics by semantic relevance to a task instruction — much better
than naive keyword matching.

If we ever want denser embeddings (e.g. sentence-transformers), we can
swap this module out without touching the curriculum API. The
``TFIDFIndex.query`` interface returns ``(score, doc_index)`` pairs
which any embedding backend can produce.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple, Dict


def _tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokens. Numbers are kept (e.g. ``132x43``)."""
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFIndex:
    """Sparse TF-IDF index over a fixed corpus of documents.

    The corpus is fingerprinted at construction time so queries are O(N*K)
    where N is the corpus size and K is the average vocabulary overlap.
    For the curriculum (~10 documents) this is effectively constant time.
    """

    def __init__(self, documents: List[str]) -> None:
        self.documents = documents
        self._tokens: List[List[str]] = [_tokenize(d) for d in documents]

        # Document frequencies — how many docs contain each term
        df: Counter[str] = Counter()
        for tokens in self._tokens:
            df.update(set(tokens))

        n = max(len(documents), 1)
        # Smoothed IDF: log((N+1)/(df+1)) + 1
        self.idf: Dict[str, float] = {
            term: math.log((n + 1) / (freq + 1)) + 1.0
            for term, freq in df.items()
        }

        # Pre-compute document vectors so each query is just a dot product
        self._doc_vecs: List[Dict[str, float]] = [
            self._vectorize(tokens) for tokens in self._tokens
        ]
        self._doc_norms: List[float] = [
            math.sqrt(sum(v * v for v in vec.values()))
            for vec in self._doc_vecs
        ]

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        max_count = max(counts.values())
        # Plain normalized term frequency * IDF. We avoid the augmented
        # form (0.5 + 0.5 * tf) on purpose: that baseline gives every shared
        # term a non-trivial contribution and produces noisy false positives
        # in our small-corpus, short-query setting.
        return {
            term: (count / max_count) * self.idf.get(term, 0.0)
            for term, count in counts.items()
        }

    def query(self, text: str, top_k: int = 3) -> List[Tuple[float, int]]:
        """Return top-k (cosine_score, doc_index) pairs sorted by score desc.

        Documents with score == 0 (no shared vocabulary with the query) are
        excluded from the results. The caller decides what to do with the
        empty case.
        """
        q_tokens = _tokenize(text)
        if not q_tokens:
            return []
        q_vec = self._vectorize(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        scored: List[Tuple[float, int]] = []
        for i, (d_vec, d_norm) in enumerate(zip(self._doc_vecs, self._doc_norms)):
            if d_norm == 0:
                continue
            # Cosine similarity over the intersection of vocabularies
            common = set(q_vec).intersection(d_vec)
            if not common:
                continue
            dot = sum(q_vec[t] * d_vec[t] for t in common)
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


__all__ = ["TFIDFIndex"]
