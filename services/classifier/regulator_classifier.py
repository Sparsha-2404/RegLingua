"""
RegLingua — Regulator Classifier
====================================
Owned by: M2 (classifier service), added as part of the M4 Day 3-5
integration pass.

`services.classifier.baseline.TfidfLogisticClassifier` is single-output: it
predicts `risk_level` only. `services.classifier.contracts.RiskPrediction`
does have a `regulator_candidates` field, but the real baseline classifier
never populates it (see `INTEGRATION_NOTES.md`, "Regulator attribution
narrowed" — a known, documented scope gap at Day 1-2, expected to close once
a real classifier is trained on M1's dataset, which does carry per-example
SEBI/RBI/DPDP labels).

Rather than turn the risk classifier into a multi-task model (a bigger
change than the Day 3-5 M4 gate needs), this is a second, independent
TF-IDF + Logistic Regression classifier with the exact same shape as
`TfidfLogisticClassifier`, trained on the same text but against the
`regulator` column instead of `risk_level`. `RiskAgentAdapter` (M4-owned)
calls it only to fill `regulator_candidates` when the risk classifier
itself doesn't supply any — see the adapter for how the two are combined.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class RegulatorClassifier:
    """Predicts which regulator (SEBI / RBI / DPDP) a piece of fintech
    product text is most relevant to. Not trained until `fit()` or
    `load()` is called."""

    def __init__(self, max_features: int = 20_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._is_fitted = False
        self._classes: List[str] = []

    def fit(self, texts: Sequence[str], regulators: Sequence[str]) -> "RegulatorClassifier":
        if len(texts) != len(regulators):
            raise ValueError("texts and regulators must be the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty dataset")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, regulators)
        self._classes = list(self.model.classes_)
        self._is_fitted = True
        return self

    def predict(self, text: str, top_n: int = 1) -> List[Tuple[str, float]]:
        """Returns up to `top_n` (regulator, confidence) pairs, highest
        confidence first."""
        if not self._is_fitted:
            raise RuntimeError(
                "RegulatorClassifier is not trained yet. Call fit() or load() first."
            )
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        ranked = sorted(zip(self._classes, proba), key=lambda pair: pair[1], reverse=True)
        return [(label, float(score)) for label, score in ranked[:top_n]]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "model": self.model, "classes": self._classes},
                f,
            )

    def load(self, path: str) -> "RegulatorClassifier":
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        self.vectorizer = artifact["vectorizer"]
        self.model = artifact["model"]
        self._classes = artifact["classes"]
        self._is_fitted = True
        return self
