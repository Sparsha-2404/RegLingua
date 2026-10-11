"""
RegLingua — Verification Policy (Day 8)
============================================
Owned by: M1 ("Define verification policy and failure cases"), implemented
as an M4 gap-filler alongside the real VerifierAgent since it's the direct
input the Verifier's deterministic checks run against, and no data-team
config for this existed yet.

This is the "policy engine" half of Day 8's `M3: Add evidence/constraint
checks` + `M4: Implement VerifierAgent and PASS/FAIL contract` — kept as
plain, explicit Python data rather than a rules DSL, per the architecture
doc's Phase A philosophy (deterministic, inspectable, no hidden state)
before any of this is a candidate for an LLM policy agent.

A rewrite is verified compliant only if it clears ALL of:
  1. Classifier re-check: the (real or mock) RiskAgent classifies the
     rewritten text as COMPLIANT (independent of whatever the Rewriter
     claims about itself).
  2. No banned phrase: none of the still-risky phrases for the case's
     regulator survive in the rewritten text.
  3. Required disclaimer present (SEBI/RBI only): the rewrite must contain
     at least one of the regulator's accepted disclaimer phrasings.

Any failing check produces a specific `FailureReason` and a human-readable
message fed back to the Rewriter as `state.verifier_feedback` — the
Rewriter never marks itself successful; only `VerifierAgent.verify()`
(agents/verifier_agent.py) sets `state.verified_compliant`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureReason(str, Enum):
    CLASSIFIER_STILL_RISKY = "classifier_still_risky"
    BANNED_PHRASE_PRESENT = "banned_phrase_present"
    MISSING_REQUIRED_DISCLAIMER = "missing_required_disclaimer"
    LOW_REVERIFICATION_CONFIDENCE = "low_reverification_confidence"


# Phrases that must never survive a rewrite, independent of regulator --
# these are the core "guaranteed outcome" / "zero risk" claims the whole
# project exists to catch.
GLOBAL_BANNED_PHRASES: list[str] = [
    "guarantee",
    "guaranteed",
    "assured return",
    "zero risk",
    "risk-free",
    "risk free",
    "100% safe",
    "no risk",
]

# Regulator-specific additions.
REGULATOR_BANNED_PHRASES: dict[str, list[str]] = {
    "SEBI": ["highest returns", "double your money", "sure-shot profit"],
    "RBI": ["instant approval with zero risk", "no questions asked loan", "hidden charges waived forever"],
    "DPDP": ["share your data with anyone", "sell your data", "no consent required"],
}

# At least one of these must appear for a rewrite to count as carrying the
# regulator's required disclaimer. Deliberately permissive (substring, not
# exact match) since the Rewriter may phrase it a few different ways.
REQUIRED_DISCLAIMER_PHRASES: dict[str, list[str]] = {
    "SEBI": [
        "market risk",
        "market-linked",
        "subject to market",
        "past performance does not indicate future results",
        "past performance is not indicative of future results",
    ],
    "RBI": [
        "subject to credit approval",
        "terms and conditions apply",
        "subject to eligibility",
    ],
    "DPDP": [
        "with your consent",
        "as permitted under applicable law",
        "in accordance with applicable data protection law",
    ],
}

# Regulators for which a disclaimer is mandatory (General/no-regulator cases
# aren't held to a specific disclaimer phrasing).
DISCLAIMER_REQUIRED_FOR = {"SEBI", "RBI", "DPDP"}

# Below this, the Verifier's own re-classification isn't trusted enough to
# certify PASS even if the label came back COMPLIANT.
REVERIFICATION_CONFIDENCE_FLOOR = 0.3


@dataclass
class VerificationVerdict:
    passed: bool
    failure_reasons: list[FailureReason] = field(default_factory=list)
    feedback: Optional[str] = None
    reclassified_label: Optional[str] = None
    reclassified_confidence: Optional[float] = None


def _banned_phrases_for(regulator: Optional[str]) -> list[str]:
    phrases = list(GLOBAL_BANNED_PHRASES)
    if regulator and regulator in REGULATOR_BANNED_PHRASES:
        phrases += REGULATOR_BANNED_PHRASES[regulator]
    return phrases


def find_banned_phrase(text: str, regulator: Optional[str]) -> Optional[str]:
    """Returns the first banned phrase found (case-insensitive substring),
    or None if the text is clean."""
    lowered = text.lower()
    for phrase in _banned_phrases_for(regulator):
        if phrase in lowered:
            return phrase
    return None


def has_required_disclaimer(text: str, regulator: Optional[str]) -> bool:
    if not regulator or regulator not in DISCLAIMER_REQUIRED_FOR:
        return True  # not applicable -- General/unset regulator
    lowered = text.lower()
    return any(phrase in lowered for phrase in REQUIRED_DISCLAIMER_PHRASES.get(regulator, []))


def evaluate(
    rewritten_text: str,
    regulator: Optional[str],
    reclassified_label: str,
    reclassified_confidence: float,
) -> VerificationVerdict:
    """Runs every Day 8 check and combines them into one verdict. Called by
    `agents/verifier_agent.py` after it re-classifies the rewrite; this
    function itself never calls the classifier -- keeping the policy pure
    and independently unit-testable."""
    reasons: list[FailureReason] = []

    if reclassified_label != "Compliant":
        reasons.append(FailureReason.CLASSIFIER_STILL_RISKY)
    elif reclassified_confidence < REVERIFICATION_CONFIDENCE_FLOOR:
        reasons.append(FailureReason.LOW_REVERIFICATION_CONFIDENCE)

    banned = find_banned_phrase(rewritten_text, regulator)
    if banned:
        reasons.append(FailureReason.BANNED_PHRASE_PRESENT)

    if not has_required_disclaimer(rewritten_text, regulator):
        reasons.append(FailureReason.MISSING_REQUIRED_DISCLAIMER)

    if not reasons:
        return VerificationVerdict(
            passed=True,
            reclassified_label=reclassified_label,
            reclassified_confidence=reclassified_confidence,
        )

    feedback_parts = []
    if FailureReason.CLASSIFIER_STILL_RISKY in reasons:
        feedback_parts.append(f"Re-classified as '{reclassified_label}', not Compliant.")
    if FailureReason.LOW_REVERIFICATION_CONFIDENCE in reasons:
        feedback_parts.append(f"Re-classification confidence too low ({reclassified_confidence:.2f}).")
    if banned:
        feedback_parts.append(f"Still contains a banned phrase: '{banned}'.")
    if FailureReason.MISSING_REQUIRED_DISCLAIMER in reasons:
        feedback_parts.append(f"Missing a required {regulator} disclaimer.")

    return VerificationVerdict(
        passed=False,
        failure_reasons=reasons,
        feedback=" ".join(feedback_parts),
        reclassified_label=reclassified_label,
        reclassified_confidence=reclassified_confidence,
    )
