"""Day 12 (M2): two sanity baselines every real model must beat. Neither is trained on the eval rows
in any way that could leak: Majority is fit on the TRAIN POOL labels only; the rule baseline is fixed.

Both expose ``predict(texts, regulators=None) -> list[str]`` of LOW/MEDIUM/HIGH labels.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence

from services.evaluation.classification_metrics import LABELS


class MajorityClassBaseline:
    """Always predicts the most frequent training label (ties -> the first of LOW, MEDIUM, HIGH)."""
    name = "majority_class"

    def __init__(self) -> None:
        self.label: Optional[str] = None

    def fit(self, y_train: Sequence[str]) -> "MajorityClassBaseline":
        if not len(y_train):
            raise ValueError("cannot fit on an empty label list")
        counts = Counter(y_train)
        self.label = max(LABELS, key=lambda l: (counts.get(l, 0), -LABELS.index(l)))
        return self

    def predict(self, texts: Sequence[str], regulators: Optional[Sequence[str]] = None) -> list[str]:
        if self.label is None:
            raise RuntimeError("fit() first")
        return [self.label] * len(texts)


class BannedPhraseRuleBaseline:
    """HIGH if any banned phrase from core/verification_policy (read-only use) matches the text for the
    given regulator, else LOW. Never predicts MEDIUM. English-only phrase lists: Hindi/Hinglish text is
    expected to fall through to LOW. ``regulators`` should be the row's true regulator (oracle input)."""
    name = "banned_phrase_rule"

    def __init__(self, matcher: Optional[Callable[[str, Optional[str]], Optional[str]]] = None) -> None:
        if matcher is None:
            from core.verification_policy import find_banned_phrase  # stdlib-only module; read-only use
            matcher = find_banned_phrase
        self._match = matcher

    def predict(self, texts: Sequence[str], regulators: Optional[Sequence[str]] = None) -> list[str]:
        regs = regulators if regulators is not None else [None] * len(texts)
        if len(regs) != len(texts):
            raise ValueError("texts and regulators must have equal length")
        return ["HIGH" if self._match(t, r) else "LOW" for t, r in zip(texts, regs)]
