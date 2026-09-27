"""
DAY 1 DELIVERABLE (M2): Classifier interface.

"Create classifier interface and baseline training skeleton." (Day 1 plan)

Defines the contract that ANY risk classifier implementation must satisfy.
The Day 3 baseline (TF-IDF + Logistic Regression, see baseline.py) and the
Day 10 fine-tuned multilingual transformer both plug into this same
interface, so the rest of the pipeline (Supervisor, RiskAgent) never has to
change when the model underneath is swapped — this is what lets "Risk
classifier is a tool" (Architecture Review, row 3) actually hold.

Day 1 gate: "All modules have interfaces even if implementations are
mocks." -> MockRiskClassifier below satisfies that gate without requiring
any trained model or dataset to exist yet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from services.classifier.contracts import RiskPrediction


class RiskClassifier(ABC):
    """Abstract interface for any fintech-text risk classifier (English or Hindi)."""

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "RiskClassifier":
        """Train the classifier on labelled (text, risk_level) pairs. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: str) -> RiskPrediction:
        """Return a single RiskPrediction for one piece of product text."""
        raise NotImplementedError

    def predict_batch(self, texts: Sequence[str]) -> List[RiskPrediction]:
        """Default batch implementation — subclasses may override for a vectorized speedup."""
        return [self.predict(t) for t in texts]

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist trained weights/artifacts to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> "RiskClassifier":
        """Load trained weights/artifacts from disk. Returns self."""
        raise NotImplementedError


class MockRiskClassifier(RiskClassifier):
    """
    Deterministic, dependency-free stand-in for the real classifier.

    Used to prove the end-to-end pipeline (Intake -> Risk -> ... -> Output)
    before any model is trained (Day 1-2 gate). Flags a small fixed list of
    known SEBI-risk keywords/phrases so the Supervisor and downstream agents
    have something real to route on during Day 1-2 integration testing.
    """

    _HIGH_RISK_KEYWORDS = [
        "guaranteed",
        "guarantee",
        "assured return",
        "personalized investment",
        "risk-free",
        "zero risk",
    ]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "MockRiskClassifier":
        # No-op: the mock does not learn from data — that's the point of a mock.
        return self

    def predict(self, text: str) -> RiskPrediction:
        lowered = text.lower()
        hits = [kw for kw in self._HIGH_RISK_KEYWORDS if kw in lowered]
        if hits:
            return RiskPrediction(
                risk_level="HIGH",
                confidence=0.6,
                risk_factors=hits,
                regulator_candidates=["SEBI"],
            )
        return RiskPrediction(risk_level="LOW", confidence=0.85, risk_factors=[])

    def save(self, path: str) -> None:
        pass  # nothing to persist for a rule-based mock

    def load(self, path: str) -> "MockRiskClassifier":
        return self
