"""
Day 5 - Hybrid keyword + vector retriever for RegLingua (fixed).

Fixes over the first version:
1. No early return when keyword search finds nothing. Hindi queries share no
   tokens with the English corpus, so keyword results are empty; the vector
   side must still run (it is the multilingual path).
2. Chunk-level scores are collapsed to source level with MAX, not
   "last chunk wins" dict overwrite.
3. Fusion uses Reciprocal Rank Fusion (RRF), which does not depend on the
   keyword (overlap fraction) and vector (cosine) scores being on the same
   scale.
4. Source-level aggregation (one result per source_id) so top_k is not
   filled by many chunks of the same document.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from agents.research_contracts import RetrievedChunk, Regulator
from services.retrieval.deterministic_retriever import DeterministicRetriever
from services.embeddings.vector_index import RegulatoryVectorIndex

RRF_K = 60


class HybridRetriever:
    """Combine keyword and semantic retrieval via reciprocal rank fusion."""

    KEYWORD_WEIGHT = 1.0
    VECTOR_WEIGHT = 1.0
    CANDIDATE_POOL = 400  # chunks pulled from each retriever before collapsing

    def __init__(
        self,
        keyword_retriever: Optional[DeterministicRetriever] = None,
        vector_index: Optional[RegulatoryVectorIndex] = None,
    ):
        self.keyword_retriever = keyword_retriever or DeterministicRetriever()
        self.vector_index = vector_index or RegulatoryVectorIndex()
        if self.vector_index.embeddings is None:
            self.vector_index.build()

    @staticmethod
    def _collapse_to_sources(ranked: List[tuple]) -> List[tuple]:
        """ranked: [(source_id, score, payload)] best-first -> best chunk per source."""
        seen: Dict[str, tuple] = {}
        order: List[str] = []
        for source_id, score, payload in ranked:
            if source_id not in seen:
                seen[source_id] = (source_id, score, payload)
                order.append(source_id)
        return [seen[s] for s in order]

    def retrieve(
        self,
        query_text: str,
        regulator_candidates: Optional[List[Regulator]] = None,
        topic_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not query_text.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        pool = max(top_k, self.CANDIDATE_POOL)

        # 1. Keyword side (may legitimately be empty, e.g. Hindi query).
        keyword_results = self.keyword_retriever.retrieve(
            query_text=query_text,
            regulator_candidates=regulator_candidates,
            topic_hint=topic_hint,
            top_k=pool,
        )
        keyword_ranked = self._collapse_to_sources(
            [(c.source_id, c.score, c) for c in keyword_results]
        )

        # 2. Vector side - always runs.
        vector_regulator = (
            regulator_candidates[0]
            if regulator_candidates and len(regulator_candidates) == 1
            else None
        )
        vector_results = self.vector_index.search(
            query=query_text,
            top_k=pool,
            regulator=vector_regulator,
            topic=topic_hint,
        )
        # If multiple regulators were given, filter vector hits to them.
        if regulator_candidates and len(regulator_candidates) > 1:
            vector_results = [
                r for r in vector_results if r["regulator"] in regulator_candidates
            ]
        vector_ranked = self._collapse_to_sources(
            [(r["source_id"], r["score"], r) for r in vector_results]
        )

        if not keyword_ranked and not vector_ranked:
            return []

        # 3. Reciprocal rank fusion over source-level rankings.
        fused: Dict[str, float] = {}
        chunks: Dict[str, RetrievedChunk] = {}

        for rank, (sid, _score, chunk) in enumerate(keyword_ranked, start=1):
            fused[sid] = fused.get(sid, 0.0) + self.KEYWORD_WEIGHT / (RRF_K + rank)
            chunks[sid] = chunk

        for rank, (sid, vscore, doc) in enumerate(vector_ranked, start=1):
            fused[sid] = fused.get(sid, 0.0) + self.VECTOR_WEIGHT / (RRF_K + rank)
            if sid not in chunks:
                chunks[sid] = RetrievedChunk(
                    source_id=doc["source_id"],
                    regulator=doc["regulator"],
                    document_title=doc["document_title"],
                    section=doc["section"],
                    effective_date=doc["effective_date"],
                    text=doc["text"],
                    source_url=doc["source_url"],
                    topic=doc.get("topic", ""),
                    score=vscore,
                )

        # Normalise fused score to 0..1 (max possible = both lists rank 1).
        max_possible = (self.KEYWORD_WEIGHT + self.VECTOR_WEIGHT) / (RRF_K + 1)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        out = []
        for sid, f in ordered:
            c = chunks[sid]
            out.append(
                RetrievedChunk(
                    source_id=c.source_id,
                    regulator=c.regulator,
                    document_title=c.document_title,
                    section=c.section,
                    effective_date=c.effective_date,
                    text=c.text,
                    source_url=c.source_url,
                    topic=c.topic,
                    score=round(min(f / max_possible, 1.0), 4),
                )
            )
        return out
