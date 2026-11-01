"""
DAY 10 (M2): Multilingual-embedding risk classifier.

Frozen sentence embeddings (default: MultilingualEncoder, M3-owned, read-only)
-> LogisticRegression(class_weight="balanced"). Implements RiskClassifier so it
is a drop-in for the TF-IDF baseline; the baseline stays the deterministic
fallback and is never modified.

Artifacts: only the fitted LR + class list + encoder *name* are pickled; encoder
weights are never saved. A sidecar JSON (<path>.meta.json) records library
versions. The encoder is imported lazily, so this module imports without
sentence_transformers installed.

Encoder protocol: any object with `encode(Sequence[str]) -> 2-D array-like`
and optionally a `model_name` attribute.
"""
from __future__ import annotations

import json
import pickle
import platform
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from services.classifier.contracts import RiskLevel, RiskPrediction
from services.classifier.interface import RiskClassifier

DEFAULT_ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _lib_versions() -> dict:
    import numpy
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        from importlib.metadata import version

        versions["sentence-transformers"] = version("sentence-transformers")
    except Exception:
        versions["sentence-transformers"] = None
    return versions


def meta_path_for(path: str) -> Path:
    return Path(str(path) + ".meta.json")


class EmbeddingLogisticClassifier(RiskClassifier):
    """Frozen multilingual embeddings + balanced Logistic Regression."""

    def __init__(self, encoder: Optional[Any] = None, C: float = 1.0, max_iter: int = 1000):
        self._encoder = encoder
        self._encoder_name: Optional[str] = (
            getattr(encoder, "model_name", None) if encoder is not None else None
        ) or (None if encoder is not None else DEFAULT_ENCODER_NAME)
        if encoder is not None and self._encoder_name is None:
            self._encoder_name = type(encoder).__name__
        self.model = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
        self._is_fitted = False
        self._classes: List[str] = []

    # ---- encoder handling -------------------------------------------------
    def _get_encoder(self):
        if self._encoder is None:
            try:
                from services.embeddings.encoder import MultilingualEncoder
            except ImportError as exc:  # sentence_transformers missing
                raise RuntimeError(
                    "No encoder available: pass `encoder=` to "
                    "EmbeddingLogisticClassifier, or install sentence-transformers "
                    f"(pip install sentence-transformers) for the default "
                    f"'{DEFAULT_ENCODER_NAME}'. Original error: {exc}"
                ) from exc
            self._encoder = MultilingualEncoder(self._encoder_name or DEFAULT_ENCODER_NAME)
        return self._encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._get_encoder().encode(list(texts)), dtype=float)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "EmbeddingLogisticClassifier is not trained yet. Call fit() or "
                "load() a saved model."
            )

    # ---- RiskClassifier ---------------------------------------------------
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "EmbeddingLogisticClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty dataset")
        self.model.fit(self._embed(texts), list(labels))
        self._classes = [str(c) for c in self.model.classes_]
        self._is_fitted = True
        return self

    def _to_prediction(self, proba_row: np.ndarray) -> RiskPrediction:
        idx = int(proba_row.argmax())
        return RiskPrediction(
            risk_level=self._classes[idx],
            confidence=float(min(max(proba_row[idx], 0.0), 1.0)),
            risk_factors=[],
        )

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    def predict_proba_batch(self, texts: Sequence[str]) -> np.ndarray:
        """(n, n_classes) probabilities, columns ordered as `classes`."""
        self._require_fitted()
        if len(texts) == 0:
            return np.zeros((0, len(self._classes)))
        return self.model.predict_proba(self._embed(texts))

    def predict(self, text: str) -> RiskPrediction:
        self._require_fitted()
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> List[RiskPrediction]:
        self._require_fitted()
        if len(texts) == 0:
            return []
        proba = self.model.predict_proba(self._embed(texts))
        return [self._to_prediction(row) for row in proba]

    def save(self, path: str) -> None:
        self._require_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "classes": self._classes,
                    "encoder_name": self._encoder_name,
                },
                f,
            )
        meta = {
            "classifier": "EmbeddingLogisticClassifier",
            "encoder_name": self._encoder_name,
            "encoder_weights_saved": False,
            "classes": self._classes,
            "library_versions": _lib_versions(),
        }
        meta_path_for(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: str) -> "EmbeddingLogisticClassifier":
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self._classes = artifact["classes"]
        self._encoder_name = artifact["encoder_name"]
        self._is_fitted = True
        return self
