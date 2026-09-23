"""
Multilingual embedding encoder for RegLingua.

Day 3:
- Loads a multilingual SentenceTransformer model.
- Converts regulatory text into dense vector embeddings.
- Keeps embedding generation separate from retrieval/indexing.
"""

from __future__ import annotations

from typing import List, Sequence

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEncoder:
    """Generate multilingual sentence embeddings for regulatory text."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode multiple texts into dense embedding vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def encode_one(self, text: str) -> List[float]:
        """Encode a single text into one dense embedding vector."""
        if not text.strip():
            raise ValueError("text must not be empty")

        return self.encode([text])[0]