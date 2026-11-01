"""Day 8 (M2): tests for services/verification/independent_recheck.py.

All cases are SYNTHETIC (hand-written strings + a FAKE keyword classifier);
none of the labels/confidences here come from the trained model.
FAKE rule: "guarantee"/"zero risk" -> HIGH 0.9; "highest returns" -> MEDIUM 0.8;
"unsure" -> LOW 0.1 (low confidence); anything else -> LOW 0.9.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.verification.independent_recheck import RecheckFlag, recheck


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        t = text.lower()
        if "guarantee" in t or "zero risk" in t:
            return SimpleNamespace(risk_level="HIGH", confidence=0.9)
        if "highest returns" in t:
            return SimpleNamespace(risk_level="MEDIUM", confidence=0.8)
        if "unsure" in t:
            return SimpleNamespace(risk_level="LOW", confidence=0.1)
        if "review me" in t:
            return SimpleNamespace(risk_level="NEEDS_REVIEW", confidence=0.5)
        return SimpleNamespace(risk_level="LOW", confidence=0.9)


RISKY = "We guarantee 20% returns."
SAFE = "Returns are subject to market risks."


def test_clean_reduction_has_no_flags():
    r = recheck(RISKY, SAFE, FakeClassifier(), "SEBI")
    assert (r.before_label, r.after_label, r.direction) == ("HIGH", "LOW", "reduced")
    assert r.flags == () and not r.rewrite_unchanged and not r.original_classifier_blind


def test_noop_rewrite_flag_exact_and_whitespace_only():
    assert recheck(RISKY, RISKY, FakeClassifier()).rewrite_unchanged
    r = recheck(RISKY, "  We guarantee   20% returns.\n", FakeClassifier())
    assert r.rewrite_unchanged and RecheckFlag.NOOP_REWRITE in r.flags
    r2 = recheck(RISKY, RISKY.lower(), FakeClassifier())  # case change is a real edit
    assert not r2.rewrite_unchanged


def test_worsened_flag():
    r = recheck(SAFE, SAFE + " We guarantee it.", FakeClassifier())
    assert (r.before_label, r.after_label, r.direction) == ("LOW", "HIGH", "worsened")
    assert RecheckFlag.WORSENED in r.flags and RecheckFlag.NOOP_REWRITE not in r.flags


def test_blind_original_flag_needs_banned_phrase_and_low_label():
    blind_text = "This plan is risk-free for everyone."  # banned phrase, fake says LOW
    r = recheck(blind_text, "Plan details apply.", FakeClassifier())
    assert r.before_label == "LOW" and r.original_classifier_blind
    assert RecheckFlag.BLIND_ORIGINAL in r.flags
    # banned phrase but classifier already HIGH -> not blind
    assert not recheck(RISKY, SAFE, FakeClassifier()).original_classifier_blind
    # LOW and no banned phrase -> not blind
    assert not recheck(SAFE, SAFE + "!", FakeClassifier()).original_classifier_blind


def test_blind_original_uses_regulator_specific_phrases():
    text = "Our fund has sure-shot profit."  # SEBI-only phrase; fake says LOW
    assert recheck(text, "ok", FakeClassifier(), "SEBI").original_classifier_blind
    assert not recheck(text, "ok", FakeClassifier(), None).original_classifier_blind


def test_low_confidence_flag_and_custom_floor():
    r = recheck(RISKY, "I am unsure about this.", FakeClassifier())
    assert r.low_confidence_after and RecheckFlag.LOW_CONFIDENCE in r.flags
    assert not recheck(RISKY, SAFE, FakeClassifier()).low_confidence_after
    assert recheck(RISKY, SAFE, FakeClassifier(), confidence_floor=0.95).low_confidence_after


def test_reclassifies_both_texts_from_scratch():
    c = FakeClassifier()
    recheck(RISKY, SAFE, c)
    assert c.calls == [RISKY, SAFE]


def test_rewriter_claims_are_rejected_not_read():
    claim = SimpleNamespace(rewritten_text=SAFE, compliant=True, rationale="all clear")
    with pytest.raises(TypeError):
        recheck(RISKY, claim, FakeClassifier())


def test_needs_review_ranks_like_medium():
    r = recheck(RISKY, "review me", FakeClassifier())
    assert r.after_label == "NEEDS_REVIEW" and r.direction == "reduced"
    assert recheck("review me", RISKY, FakeClassifier()).direction == "worsened"


def test_unknown_label_raises():
    class Bad:
        def predict(self, t):
            return SimpleNamespace(risk_level="CRITICAL", confidence=1.0)

    with pytest.raises(ValueError):
        recheck("a", "b", Bad())


def test_to_dict_flags_are_strings():
    d = recheck(RISKY, RISKY, FakeClassifier()).to_dict()
    assert d["flags"] == ["NOOP_REWRITE"] and d["direction"] == "same"
