"""Day 9 (M2): confidence / abstention logic for the risk classifier.

Pure, dependency-light (stdlib + the read-only phrase list in core.verification_policy;
no imports from agents/). ``decide`` maps a probability vector + text to ACCEPT / REVIEW /
ABSTAIN with machine-readable reasons and the signals it used.

Only these signals are used (all measured in scripts/analyze_confidence_signals.py):
max probability, top-2 margin, entropy, language (optional per-language floor) and the
conflict signal (classifier says LOW but the policy phrase list finds a banned phrase).
Prompt-1 finding: no signal beat raw confidence on held-out, so the margin and entropy
rules ship DISABLED (None); thresholds are config, to be chosen on non-held-out rows only.
Defaults for the two floors mirror agents/supervisor.py (0.35 / 0.7) and are NOT tuned.

OOV ratio is not used: it needs the vectorizer vocabulary (not available to a pure
function) and it was not informative on held-out (AUROC 0.446).

Action semantics (``route_hint`` maps to M4's route_after_classification names):
  ABSTAIN  max_prob < abstain_floor            -> "abstain"
  REVIEW   any review reason fired             -> "needs_review"
  ACCEPT   no reason fired; LOW -> "compliant", MEDIUM/HIGH -> "needs_review"
           (the supervisor sends every non-compliant label down the same route).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from core.verification_policy import find_banned_phrase

LABELS = ("LOW", "MEDIUM", "HIGH", "NEEDS_REVIEW")
# Tie-break order: riskiest first, so a tie never favours LOW.
_TIE_ORDER = {"HIGH": 3, "MEDIUM": 2, "NEEDS_REVIEW": 1, "LOW": 0}
_SUM_TOLERANCE = 1e-3


class Action(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"


class Reason(str, Enum):
    ACCEPTED = "ACCEPTED"
    BELOW_ABSTAIN_FLOOR = "BELOW_ABSTAIN_FLOOR"
    CONFLICT_BANNED_PHRASE_LOW = "CONFLICT_BANNED_PHRASE_LOW"
    LOW_BELOW_COMPLIANT_FLOOR = "LOW_BELOW_COMPLIANT_FLOOR"
    LOW_MARGIN = "LOW_MARGIN"
    HIGH_ENTROPY = "HIGH_ENTROPY"
    CLASSIFIER_NEEDS_REVIEW = "CLASSIFIER_NEEDS_REVIEW"


@dataclass(frozen=True)
class AbstentionConfig:
    abstain_floor: float = 0.35                  # max_prob <  floor -> ABSTAIN (supervisor: 0.35)
    compliant_floor: float = 0.7                 # LOW needs max_prob >= floor to ACCEPT (supervisor: 0.7)
    margin_floor: Optional[float] = None         # margin < floor -> REVIEW; None = rule off
    entropy_ceiling: Optional[float] = None      # entropy (nats) > ceiling -> REVIEW; None = rule off
    language_abstain_floors: dict = field(default_factory=dict)  # per-language override of abstain_floor

    def __post_init__(self) -> None:
        def unit(name: str, v: float) -> None:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be a number in [0, 1], got {v!r}")

        unit("abstain_floor", self.abstain_floor)
        unit("compliant_floor", self.compliant_floor)
        if self.abstain_floor > self.compliant_floor:
            raise ValueError("abstain_floor must be <= compliant_floor")
        if self.margin_floor is not None:
            unit("margin_floor", self.margin_floor)
        if self.entropy_ceiling is not None and not (
            isinstance(self.entropy_ceiling, (int, float)) and self.entropy_ceiling >= 0
        ):
            raise ValueError("entropy_ceiling must be a number >= 0 or None")
        for lang, v in self.language_abstain_floors.items():
            unit(f"language_abstain_floors[{lang!r}]", v)

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AbstentionConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(data))

    def to_json(self, path: Union[str, Path, None] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "AbstentionConfig":
        """Load from a JSON string or a path to a JSON file."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8-sig")
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class AbstentionDecision:
    action: Action
    reasons: list
    signals: dict

    @property
    def route_hint(self) -> str:
        if self.action is Action.ABSTAIN:
            return "abstain"
        if self.action is Action.ACCEPT and self.signals["label"] == "LOW":
            return "compliant"
        return "needs_review"

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "reasons": [r.value for r in self.reasons],
                "signals": dict(self.signals), "route_hint": self.route_hint}


def entropy(probs: Mapping[str, float]) -> float:
    """Shannon entropy in nats."""
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def top2_margin(probs: Mapping[str, float]) -> float:
    ordered = sorted(probs.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


def normalize_probabilities(probabilities: Mapping[Any, float]) -> dict[str, float]:
    """Validate and normalise keys to upper-case label strings (enum keys accepted)."""
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    out: dict[str, float] = {}
    for key, p in probabilities.items():
        label = str(getattr(key, "value", key)).upper()
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
        if label in out:
            raise ValueError(f"duplicate label {label!r}")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or math.isnan(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"probability for {label} must be a number in [0, 1], got {p!r}")
        out[label] = float(p)
    if abs(sum(out.values()) - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"probabilities must sum to 1 (got {sum(out.values()):.6f})")
    return out


def decide(
    probabilities: Mapping[Any, float],
    text: str,
    regulator: Optional[str] = None,
    language: Optional[str] = None,
    config: Optional[AbstentionConfig] = None,
) -> AbstentionDecision:
    cfg = config if config is not None else AbstentionConfig()
    probs = normalize_probabilities(probabilities)
    label = max(probs, key=lambda k: (probs[k], _TIE_ORDER[k]))
    max_prob = probs[label]
    margin, ent = top2_margin(probs), entropy(probs)
    banned = find_banned_phrase(text, regulator) if label == "LOW" else None
    conflict = banned is not None
    abstain_floor = cfg.language_abstain_floors.get(language, cfg.abstain_floor) if language else cfg.abstain_floor

    signals = {"label": label, "max_prob": max_prob, "margin": margin, "entropy": ent, "conflict": conflict,
               "banned_phrase": banned, "language": language, "regulator": regulator,
               "abstain_floor_used": abstain_floor, "compliant_floor": cfg.compliant_floor}

    reasons: list[Reason] = []
    if max_prob < abstain_floor:
        reasons.append(Reason.BELOW_ABSTAIN_FLOOR)
        if conflict:
            reasons.append(Reason.CONFLICT_BANNED_PHRASE_LOW)
        return AbstentionDecision(Action.ABSTAIN, reasons, signals)

    if conflict:
        reasons.append(Reason.CONFLICT_BANNED_PHRASE_LOW)
    if label == "NEEDS_REVIEW":
        reasons.append(Reason.CLASSIFIER_NEEDS_REVIEW)
    if label == "LOW" and max_prob < cfg.compliant_floor:
        reasons.append(Reason.LOW_BELOW_COMPLIANT_FLOOR)
    if cfg.margin_floor is not None and margin < cfg.margin_floor:
        reasons.append(Reason.LOW_MARGIN)
    if cfg.entropy_ceiling is not None and ent > cfg.entropy_ceiling:
        reasons.append(Reason.HIGH_ENTROPY)

    if reasons:
        return AbstentionDecision(Action.REVIEW, reasons, signals)
    return AbstentionDecision(Action.ACCEPT, [Reason.ACCEPTED], signals)
