"""
Day 5 — Hybrid keyword + vector retriever for RegLingua.

Combines:
- deterministic keyword retrieval
- multilingual vector retrieval

The two scores are normalized and combined into a single hybrid score.
Metadata filtering is delegated to the underlying retrievers where possible.
"""

from __future__ import annotations

from typing import List, Optional

from agents.research_contracts import RetrievedChunk, Regulator
from services.retrieval.deterministic_retriever import DeterministicRetriever
from services.embeddings.vector_index import RegulatoryVectorIndex


class HybridRetriever:
    """Combine keyword and semantic retrieval into one ranked result set."""

    KEYWORD_WEIGHT = 0.5
    VECTOR_WEIGHT = 0.5

    def __init__(
        self,
        keyword_retriever: Optional[DeterministicRetriever] = None,
        vector_index: Optional[RegulatoryVectorIndex] = None,
    ):
        self.keyword_retriever = (
            keyword_retriever or DeterministicRetriever()
        )
        self.vector_index = vector_index or RegulatoryVectorIndex()

        if self.vector_index.embeddings is None:
            self.vector_index.build()

    def retrieve(
        self,
        query_text: str,
        regulator_candidates: Optional[List[Regulator]] = None,
        topic_hint: Optional[str] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """Retrieve and rerank documents using keyword + vector scores."""

        if not query_text.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # 1. Keyword retrieval
        keyword_results = self.keyword_retriever.retrieve(
            query_text=query_text,
            regulator_candidates=regulator_candidates,
            topic_hint=topic_hint,
            top_k=max(top_k, 10),
        )

        # 2. Vector retrieval
        vector_results = self.vector_index.search(
            query=query_text,
            top_k=max(top_k, 10),
            regulator=(
                regulator_candidates[0]
                if regulator_candidates and len(regulator_candidates) == 1
                else None
            ),
            topic=topic_hint,
        )

        # 3. Convert both result sets into lookup dictionaries.
        keyword_scores = {
            result.source_id: result.score
            for result in keyword_results
        }

        vector_scores = {
            result["source_id"]: result["score"]
            for result in vector_results
        }

        # 4. Consider every document retrieved by either method.
        source_ids = set(keyword_scores) | set(vector_scores)

        vector_documents = {
            result["source_id"]: result
            for result in vector_results
        }

        keyword_documents = {
            result.source_id: result
            for result in keyword_results
        }

        ranked_results = []

        for source_id in source_ids:
            keyword_score = keyword_scores.get(source_id, 0.0)
            vector_score = vector_scores.get(source_id, 0.0)

            hybrid_score = (
                self.KEYWORD_WEIGHT * keyword_score
                + self.VECTOR_WEIGHT * vector_score
            )

            # Prefer the keyword result because it is already a
            # RetrievedChunk with the contract expected downstream.
            if source_id in keyword_documents:
                chunk = keyword_documents[source_id]
            else:
                document = vector_documents[source_id]

                chunk = RetrievedChunk(
                    source_id=document["source_id"],
                    regulator=document["regulator"],
                    document_title=document["document_title"],
                    section=document["section"],
                    effective_date=document["effective_date"],
                    text=document["text"],
                    source_url=document["source_url"],
                    topic=document.get("topic", ""),
                    score=vector_score,
                )

            ranked_results.append((hybrid_score, chunk))

        # 5. Rerank using the combined score.
        ranked_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results = []

        for hybrid_score, chunk in ranked_results[:top_k]:
            results.append(
                RetrievedChunk(
                    source_id=chunk.source_id,
                    regulator=chunk.regulator,
                    document_title=chunk.document_title,
                    section=chunk.section,
                    effective_date=chunk.effective_date,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    topic=chunk.topic,
                    score=round(min(hybrid_score, 1.0), 4),
                )
            )

        return results