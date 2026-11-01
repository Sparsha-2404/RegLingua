"""Day 8 (M2): independent risk re-check.

Pure function ``recheck(original, rewrite, classifier, regulator=None)``.
Both texts are re-classified from scratch with the supplied classifier; the
only inputs are the two strings, so no rewriter-supplied claim (rationale,
"compliant" flag, attempt number, ...) can influence the result.

``classifier.predict(text)`` must return an object with ``risk_level``
(str/enum: LOW, MEDIUM, HIGH, NEEDS_REVIEW) and ``confidence`` (float).
Callers that need text normalisation wrap the classifier (see the Day 7
``_NormalizingClassifier`` adapter); this module stays dependency-free apart
from the read-only, stdlib-only ``core.verification_policy`` phrase list.
NEEDS_REVIEW is ranked like MEDIUM, matching ``agents/risk_agent_adapter``
(NEEDS_REVIEW -> AMBIGUOUS).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from core.verification_policy import REVERIFICATION_CONFIDENCE_FLOOR, find_banned_phrase

_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "NEEDS_REVIEW": 1}


class RecheckFlag(str, Enum):
    NOOP_REWRITE = "NOOP_REWRITE"
    WORSENED = "WORSENED"
    BLIND_ORIGINAL = "BLIND_ORIGINAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True)
class RecheckResult:
    before_label: str
    before_confidence: float
    after_label: str
    after_confidence: float
    rewrite_unchanged: bool
    direction: str  # "reduced" | "same" | "worsened"
    original_classifier_blind: bool
    low_confidence_after: bool
    flags: tuple[RecheckFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "flags"}
        d["flags"] = [f.value for f in self.flags]
        return d


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _label(prediction: Any) -> str:
    level = getattr(prediction, "risk_level", prediction)
    level = getattr(level, "value", level)
    value = str(level).upper()
    if value not in _RANK:
        raise ValueError(f"Unsupported risk label: {value!r}")
    return value


def recheck(
    original: str,
    rewrite: str,
    classifier: Any,
    regulator: Optional[str] = None,
    confidence_floor: float = REVERIFICATION_CONFIDENCE_FLOOR,
) -> RecheckResult:
    if not isinstance(original, str) or not isinstance(rewrite, str):
        raise TypeError("recheck takes plain strings only; rewriter objects/claims are not accepted")

    before = classifier.predict(original)
    after = classifier.predict(rewrite)
    before_label, after_label = _label(before), _label(after)
    before_conf, after_conf = float(before.confidence), float(after.confidence)

    unchanged = normalize_whitespace(original) == normalize_whitespace(rewrite)
    delta = _RANK[after_label] - _RANK[before_label]
    direction = "reduced" if delta < 0 else "worsened" if delta > 0 else "same"
    blind = find_banned_phrase(original, regulator) is not None and before_label == "LOW"
    low_conf = after_conf < confidence_floor

    flags = []
    if unchanged:
        flags.append(RecheckFlag.NOOP_REWRITE)
    if direction == "worsened":
        flags.append(RecheckFlag.WORSENED)
    if blind:
        flags.append(RecheckFlag.BLIND_ORIGINAL)
    if low_conf:
        flags.append(RecheckFlag.LOW_CONFIDENCE)

    return RecheckResult(
        before_label=before_label,
        before_confidence=before_conf,
        after_label=after_label,
        after_confidence=after_conf,
        rewrite_unchanged=unchanged,
        direction=direction,
        original_classifier_blind=blind,
        low_confidence_after=low_conf,
        flags=tuple(flags),
    )
