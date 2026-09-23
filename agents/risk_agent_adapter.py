"""
RiskAgentAdapter — Day 1/2 integration glue between M2 and M4.
================================================================

M2's classifier service (services/classifier/) speaks its own vocabulary:
`RiskLevel.LOW/MEDIUM/HIGH/NEEDS_REVIEW`, free-form `regulator_candidates`
strings, and a `RiskPrediction` dataclass — because M2 built it to be
testable standalone, before the Supervisor's schema existed (Day 1 gate:
"all modules have interfaces even if implementations are mocks").

M4's Supervisor speaks a coarser, case-scoped vocabulary instead:
`core.schemas.RiskLabel.COMPLIANT / RISKY / AMBIGUOUS`, a single
`Regulator` enum, and reads/writes everything through one `CaseState`
object.

This module is the ONLY place that translates between the two. Neither
M2's classifier code nor M4's Supervisor/graph code needs to know the
other's types exist.
"""

from __future__ import annotations

from typing import Optional

from core.schemas import CaseState, EventType, Regulator, RiskLabel
from services.classifier.contracts import RiskLevel, RiskPrediction
from services.classifier.interface import MockRiskClassifier, RiskClassifier
from services.classifier.preprocessing import normalize_for_classifier

# M2's four-way risk vocabulary collapses onto M4's three-way one. NEEDS_REVIEW
# (M2's low-confidence abstention) maps to AMBIGUOUS rather than COMPLIANT so
# the Supervisor still routes it into the evidence/explain/rewrite path
# instead of silently letting a low-confidence case through.
_RISK_LEVEL_TO_LABEL = {
    RiskLevel.HIGH: RiskLabel.RISKY,
    RiskLevel.MEDIUM: RiskLabel.AMBIGUOUS,
    RiskLevel.LOW: RiskLabel.COMPLIANT,
    RiskLevel.NEEDS_REVIEW: RiskLabel.AMBIGUOUS,
}

_STR_TO_REGULATOR = {
    "SEBI": Regulator.SEBI,
    "RBI": Regulator.RBI,
    "DPDP": Regulator.DPDP,
}


class RiskAgentAdapter:
    """Implements `agents.contracts.RiskAgent` on top of M2's classifier.

    Day 1-2: defaults to `MockRiskClassifier` (M2's dependency-free keyword
    mock), so the Supervisor graph runs against M2's real interface/mock
    instead of a second, duplicate mock maintained inside M4's code.

    Day 3+: once M1's labelled dataset lands and M2 trains
    `TfidfLogisticClassifier`, pass a loaded instance in via
    `RiskAgentAdapter(classifier=trained_clf)` — nothing else in the
    Supervisor or graph wiring needs to change.
    """

    def __init__(self, classifier: Optional[RiskClassifier] = None) -> None:
        self.classifier = classifier or MockRiskClassifier()

    def classify(self, state: CaseState) -> CaseState:
        normed = normalize_for_classifier(state.input_text)
        prediction: RiskPrediction = self.classifier.predict(normed["normalized_text"])

        state.risk_label = _RISK_LEVEL_TO_LABEL[prediction.risk_level]
        state.classifier_confidence = prediction.confidence
        state.risk_phrase = prediction.risk_factors[0] if prediction.risk_factors else None

        regulator_candidates = prediction.regulator_candidates or []
        state.regulator = (
            _STR_TO_REGULATOR.get(regulator_candidates[0], Regulator.GENERAL)
            if regulator_candidates
            else Regulator.GENERAL
        )

        state.log(
            EventType.RISK_CLASSIFIED,
            agent_name=type(self.classifier).__name__,
            risk_level=prediction.risk_level.value,
            confidence=prediction.confidence,
            risk_factors=list(prediction.risk_factors),
        )
        return state
