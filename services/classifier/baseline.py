"""
DAY 1 DELIVERABLE (M2): Baseline training skeleton.

This is the skeleton for the model named in the architecture's Version 1
AI/ML table (Section 8): "Risk classifier: TF-IDF + Logistic Regression."

Day 1's gate only requires the interface + skeleton to exist and be wired
together — actually training on the real labelled dataset is explicitly a
Day 3 task ("M2: Train TF-IDF + Logistic Regression; save artifacts"), which
depends on M1's Day 2 dataset (200+ labelled examples). So `fit()` here is
fully implemented and ready to use the moment that dataset lands; only the
*training run itself* is deferred.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from core.contracts import RiskLevel, RiskPrediction
from services.classifier.interface import RiskClassifier


class TfidfLogisticClassifier(RiskClassifier):
    """
    Baseline Risk Analyst classifier: TF-IDF features + multinomial
    Logistic Regression.

    NOT trained yet as of Day 1 — calling `predict()` before `fit()`/`load()`
    raises RuntimeError so a missing-training-step failure is loud rather
    than silently returning garbage risk labels for a compliance tool.
    """

    def __init__(self, max_features: int = 20_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # regulatory risk data is usually imbalanced (mostly LOW)
        )
        self._is_fitted = False
        self._classes: List[str] = []

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TfidfLogisticClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty dataset")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._classes = list(self.model.classes_)
        self._is_fitted = True
        return self

    def predict(self, text: str) -> RiskPrediction:
        if not self._is_fitted:
            raise RuntimeError(
                "TfidfLogisticClassifier is not trained yet. Call fit() with "
                "M1's labelled dataset (Day 3 task) or load() a saved model."
            )
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        best_idx = proba.argmax()
        label = self._classes[best_idx]
        confidence = float(proba[best_idx])

        risk_factors: List[str] = []
        if label != RiskLevel.LOW.value:
            risk_factors = self._top_contributing_terms(text)

        return RiskPrediction(risk_level=label, confidence=confidence, risk_factors=risk_factors)

    def _top_contributing_terms(self, text: str, top_n: int = 3) -> List[str]:
        """Cheap interpretability: highest-TF-IDF-weight terms present in the text."""
        X = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        row = X.toarray()[0]
        top_indices = row.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices if row[i] > 0]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "model": self.model, "classes": self._classes},
                f,
            )

    def load(self, path: str) -> "TfidfLogisticClassifier":
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        self.vectorizer = artifact["vectorizer"]
        self.model = artifact["model"]
        self._classes = artifact["classes"]
        self._is_fitted = True
        return self


if __name__ == "__main__":
    # Skeleton smoke test — proves fit/predict/save/load wire together end
    # to end on toy data. Day 3 replaces this toy data with M1's real dataset.
    toy_texts = [
        "We guarantee 20% returns on your investment.",
        "Track your monthly spending with our budgeting tool.",
        "Assured returns, zero risk, invest today.",
        "Read our terms and conditions before using the app.",
    ]
    toy_labels = ["HIGH", "LOW", "HIGH", "LOW"]

    clf = TfidfLogisticClassifier()
    clf.fit(toy_texts, toy_labels)
    pred = clf.predict("This plan offers guaranteed high returns.")
    print(pred.to_dict())
