"""
Vector index for RegLingua regulatory documents.

Day 4:
- Loads the regulatory corpus.
- Generates multilingual embeddings.
- Stores embeddings together with regulatory metadata.
- Supports metadata filtering before vector ranking.
- Provides cosine-similarity search.

The index is intentionally kept simple so the Day 5 hybrid retriever
can combine it with keyword retrieval and reranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from services.embeddings.encoder import MultilingualEncoder


_DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "regulatory"
    / "sample_corpus.json"
)


class RegulatoryVectorIndex:
    """In-memory vector index over regulatory corpus entries."""

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        encoder: Optional[MultilingualEncoder] = None,
    ):
        self.corpus_path = corpus_path or _DEFAULT_CORPUS_PATH
        self.encoder = encoder or MultilingualEncoder()

        self.documents: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build(self) -> None:
        """Load the corpus and create an embedding for every document."""

        with open(self.corpus_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        texts = [
            f"{document.get('topic', '')}. {document['text']}"
            for document in self.documents
        ]

        vectors = self.encoder.encode(texts)
        self.embeddings = np.asarray(vectors, dtype=np.float32)

    def _filter_indices(
        self,
        regulator: Optional[Union[str, List[str]]] = None,
        topic: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[int]:
        """Return document indices matching the supplied metadata filters."""

        candidate_indices = list(range(len(self.documents)))

        if regulator:
            allowed_regulators = (
                {regulator} if isinstance(regulator, str) else set(regulator)
            )
            candidate_indices = [
                index
                for index in candidate_indices
                if self.documents[index]["regulator"] in allowed_regulators
            ]

        if topic:
            topic_lower = topic.lower()
            candidate_indices = [
                index
                for index in candidate_indices
                if topic_lower in self.documents[index].get("topic", "").lower()
            ]

        if source_id:
            candidate_indices = [
                index
                for index in candidate_indices
                if self.documents[index]["source_id"] == source_id
            ]

        return candidate_indices

    def search(
        self,
        query: str,
        top_k: int = 5,
        regulator: Optional[Union[str, List[str]]] = None,
        topic: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Return the most semantically similar regulatory documents.

        Metadata filtering is performed before vector ranking.
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if self.embeddings is None:
            raise RuntimeError("Index has not been built. Call build() first.")

        candidate_indices = self._filter_indices(
            regulator=regulator,
            topic=topic,
            source_id=source_id,
        )

        if not candidate_indices:
            return []

        query_vector = np.asarray(
            self.encoder.encode_one(query),
            dtype=np.float32,
        )

        scores = self.embeddings @ query_vector

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = dict(self.documents[index])
            result["score"] = round(float(scores[index]), 4)
            results.append(result)

        return results