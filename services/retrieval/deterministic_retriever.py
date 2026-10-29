"""
Day 2 — Deterministic retrieval prototype for the Regulatory Research Agent.

No embeddings, no ML model: pure keyword/metadata scoring, so the full
pipeline (Intake -> Risk -> Research -> ...) can be exercised end-to-end
on Day 2, before the embedding index exists (that's Day 3-5 on the M3
reduced schedule: start embedding index -> build embedding index +
metadata filtering -> hybrid keyword+vector retrieval).

Scoring is a simple normalized keyword-overlap score against each corpus
entry's text + topic fields, with an optional topic-hint boost. This is a
deliberately dependency-free stand-in for the future hybrid retriever —
same call signature, so Day 5's HybridRetriever can be swapped in behind
RegulatoryResearchAgent without changing the agent or its contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from agents.research_contracts import RetrievedChunk, Regulator

# Matches latin letters and Devanagari letters, so English and Hindi/
# code-mixed queries both tokenize sensibly.
_TOKEN_RE = re.compile(r"[a-zA-Z\u0900-\u097F]+")

# Small stopword list (English + a few common Hindi function words) so that
# generic connector words don't produce spurious keyword-overlap matches
# against unrelated corpus entries. This is deliberately minimal — a real
# stopword list / stemmer is exactly the kind of thing the Day 5 hybrid
# retriever would replace this with.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "your", "our", "is", "are", "be", "this", "that", "it", "as", "at",
    "no", "not", "into", "from", "by", "we", "you", "will",
    "है", "के", "का", "की", "और", "को", "में", "पर", "से", "यह",
}

_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "regulatory" / "sample_corpus.json"
)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


class DeterministicRetriever:
    """Keyword-overlap retriever over a small JSON regulatory corpus."""

    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = corpus_path or _DEFAULT_CORPUS_PATH
        self._corpus = self._load_corpus(self.corpus_path)

    @staticmethod
    def _load_corpus(path: Path) -> List[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Regulatory corpus not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(
        self,
        query_text: str,
        regulator_candidates: Optional[List[Regulator]] = None,
        topic_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        query_tokens = set(_tokenize(query_text))
        if topic_hint:
            query_tokens |= set(_tokenize(topic_hint))
        if not query_tokens:
            return []

        # Metadata filter before scoring, as the architecture doc recommends
        # ("Use metadata filtering before semantic search where possible.")
        candidates = self._corpus
        if regulator_candidates:
            candidates = [c for c in candidates if c["regulator"] in regulator_candidates]

        scored = []
        for entry in candidates:
            doc_tokens = set(_tokenize(entry["text"])) | set(_tokenize(entry.get("topic", "")))
            overlap = query_tokens & doc_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            if topic_hint and topic_hint.lower() in entry.get("topic", "").lower():
                score += 0.15
            scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: List[RetrievedChunk] = []
        for score, entry in scored[:top_k]:
            results.append(
                RetrievedChunk(
                    source_id=entry["source_id"],
                    regulator=entry["regulator"],
                    document_title=entry["document_title"],
                    section=entry["section"],
                    effective_date=entry["effective_date"],
                    text=entry["text"],
                    source_url=entry["source_url"],
                    topic=entry.get("topic", ""),
                    score=round(min(score, 1.0), 4),
                    publication_date=entry.get("publication_date"),
                    commencement_date=entry.get("commencement_date"),
                    commencement_note=entry.get("commencement_note"),
                )
            )
        return results
