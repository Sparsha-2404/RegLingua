"""Day 7 (M2): unit tests for services/evaluation/rewrite_eval.py.

Uses FAKE classifier and rewriter only - no sklearn model, no agents, no
pydantic, no embeddings - so it runs anywhere (including HF-less sandboxes).

Fake classifier rule (deterministic, keyword-based):
    "guarantee" / "zero risk" in text -> HIGH
    "highest returns"                 -> MEDIUM
    anything else                     -> LOW
"""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from services.evaluation.rewrite_eval import (
    RISK_RANK,
    aggregate_counts,
    aggregate_phrase_removed,
    evaluate_pre_post,
    format_rate,
    risk_label_value,
    risk_rank,
    risk_transition,
    summary_lines,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeClassifier:
    """Keyword classifier; records every text it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, text: str) -> str:
        self.calls.append(text)
        lowered = text.lower()
        if "guarantee" in lowered or "zero risk" in lowered:
            return "HIGH"
        if "highest returns" in lowered:
            return "MEDIUM"
        return "LOW"


class ReducingRewriter:
    """Always strips the risky phrasing."""

    def rewrite(self, text: str) -> str:
        return (
            text.replace("guarantee", "aim for")
            .replace("zero risk", "market risk")
            .replace("highest returns", "competitive returns")
        )


class IdentityRewriter:
    """Never changes anything."""

    def rewrite(self, text: str) -> str:
        return text


class WorseningRewriter:
    """Makes things worse by injecting risky language."""

    def rewrite(self, text: str) -> str:
        return text + " We guarantee it with zero risk."


def _run(rewriter, texts):
    classifier = FakeClassifier()
    results = [evaluate_pre_post(t, classifier, rewriter) for t in texts]
    return classifier, results


HIGH_TEXT = "We guarantee 20% returns."
MEDIUM_TEXT = "Our fund offers the highest returns."
LOW_TEXT = "Investments are subject to market risks."


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_risk_rank_mapping_is_ordinal():
    assert RISK_RANK == {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    assert risk_rank("LOW") < risk_rank("MEDIUM") < risk_rank("HIGH")


def test_risk_label_accepts_case_enum_and_prediction_like_objects():
    class Level(Enum):
        HIGH = "HIGH"

    assert risk_label_value("high") == "HIGH"
    assert risk_label_value(Level.HIGH) == "HIGH"
    assert risk_label_value(SimpleNamespace(risk_level=Level.HIGH)) == "HIGH"
    assert risk_label_value(SimpleNamespace(risk_level="medium")) == "MEDIUM"


@pytest.mark.parametrize("bad", ["CRITICAL", "", "unknown", None])
def test_risk_label_rejects_unsupported_labels(bad):
    with pytest.raises(ValueError):
        risk_label_value(bad)


@pytest.mark.parametrize(
    "before,after,direction",
    [
        ("HIGH", "LOW", "reduced"),
        ("HIGH", "MEDIUM", "reduced"),
        ("MEDIUM", "LOW", "reduced"),
        ("HIGH", "HIGH", "unchanged"),
        ("MEDIUM", "MEDIUM", "unchanged"),
        ("LOW", "LOW", "unchanged"),
        ("LOW", "MEDIUM", "worsened"),
        ("LOW", "HIGH", "worsened"),
        ("MEDIUM", "HIGH", "worsened"),
    ],
)
def test_risk_transition_all_nine_pairs(before, after, direction):
    assert risk_transition(before, after) == {
        "before_label": before,
        "after_label": after,
        "direction": direction,
    }


def test_aggregate_counts_totals_and_buckets():
    transitions = [
        {"direction": "reduced"},
        {"direction": "reduced"},
        {"direction": "unchanged"},
        {"direction": "worsened"},
    ]
    assert aggregate_counts(transitions) == {
        "total": 4,
        "reduced": 2,
        "unchanged": 1,
        "worsened": 1,
    }


def test_aggregate_counts_empty_and_invalid_direction():
    assert aggregate_counts([]) == {"total": 0, "reduced": 0, "unchanged": 0, "worsened": 0}
    with pytest.raises(ValueError):
        aggregate_counts([{"direction": "sideways"}])


def test_format_rate_matches_legacy_script_format_and_handles_empty():
    assert format_rate(2, 10) == "2/10 (20%)"
    assert format_rate(10, 10) == "10/10 (100%)"
    assert format_rate(0, 10) == "0/10 (0%)"
    assert format_rate(0, 0) == "0/0 (n/a)"  # legacy script would ZeroDivisionError


def test_summary_lines_exact_legacy_text():
    counts = {"total": 10, "reduced": 2, "unchanged": 8, "worsened": 0}
    assert summary_lines(counts, 10) == [
        "Risk reduced:   2/10 (20%)",
        "Risk unchanged: 8/10 (80%)",
        "Risk worsened:  0/10 (0%)",
        "Named banned phrase removed by attempt 1: 10/10 (100%)",
    ]
    assert len(summary_lines(counts)) == 3  # phrase line omitted when not measured


# --------------------------------------------------------------------------
# Scenario: always-reduces
# --------------------------------------------------------------------------
def test_always_reduces_rewriter():
    _, results = _run(ReducingRewriter(), [HIGH_TEXT, MEDIUM_TEXT])
    assert [(r["before_label"], r["after_label"], r["direction"]) for r in results] == [
        ("HIGH", "LOW", "reduced"),
        ("MEDIUM", "LOW", "reduced"),
    ]
    assert aggregate_counts(results) == {"total": 2, "reduced": 2, "unchanged": 0, "worsened": 0}


# --------------------------------------------------------------------------
# Scenario: never-changes
# --------------------------------------------------------------------------
def test_never_changes_rewriter():
    _, results = _run(IdentityRewriter(), [HIGH_TEXT, MEDIUM_TEXT, LOW_TEXT])
    assert [r["direction"] for r in results] == ["unchanged"] * 3
    assert all(r["before_label"] == r["after_label"] for r in results)
    assert all(r["rewritten_text"] == r["original_text"] for r in results)
    assert aggregate_counts(results) == {"total": 3, "reduced": 0, "unchanged": 3, "worsened": 0}


# --------------------------------------------------------------------------
# Scenario: worsens
# --------------------------------------------------------------------------
def test_worsening_rewriter():
    _, results = _run(WorseningRewriter(), [LOW_TEXT, MEDIUM_TEXT])
    assert [(r["before_label"], r["after_label"], r["direction"]) for r in results] == [
        ("LOW", "HIGH", "worsened"),
        ("MEDIUM", "HIGH", "worsened"),
    ]
    assert aggregate_counts(results) == {"total": 2, "reduced": 0, "unchanged": 0, "worsened": 2}


# --------------------------------------------------------------------------
# Scenario: original already LOW
# --------------------------------------------------------------------------
def test_already_low_original_cannot_count_as_reduced():
    # Even a "good" rewriter cannot reduce a LOW row: it must land in
    # 'unchanged', otherwise the reduction rate would be inflated.
    _, results = _run(ReducingRewriter(), [LOW_TEXT])
    (result,) = results
    assert result["before_label"] == "LOW"
    assert result["after_label"] == "LOW"
    assert result["direction"] == "unchanged"
    assert aggregate_counts(results)["reduced"] == 0


def test_already_low_mixed_with_reducible_rows_dilutes_rate_honestly():
    _, results = _run(ReducingRewriter(), [HIGH_TEXT, LOW_TEXT, LOW_TEXT, MEDIUM_TEXT])
    counts = aggregate_counts(results)
    assert counts == {"total": 4, "reduced": 2, "unchanged": 2, "worsened": 0}
    assert summary_lines(counts)[0] == "Risk reduced:   2/4 (50%)"


# --------------------------------------------------------------------------
# evaluate_pre_post orchestration contract
# --------------------------------------------------------------------------
def test_evaluate_pre_post_structured_result_and_call_order():
    classifier, (result,) = _run(ReducingRewriter(), [HIGH_TEXT])
    assert result == {
        "original_text": HIGH_TEXT,
        "rewritten_text": "We aim for 20% returns.",
        "before_label": "HIGH",
        "after_label": "LOW",
        "direction": "reduced",
    }
    # predict(original) first, then predict(rewrite) - exactly two calls.
    assert classifier.calls == [HIGH_TEXT, "We aim for 20% returns."]
    assert "phrase_removed" not in result  # only present when a hook is supplied


def test_phrase_removed_hook_recorded_and_aggregated():
    classifier = FakeClassifier()

    def removed(text: str) -> bool:
        return "guarantee" not in text.lower()

    results = [
        evaluate_pre_post(HIGH_TEXT, classifier, ReducingRewriter(), phrase_removed_fn=removed),
        evaluate_pre_post(HIGH_TEXT, classifier, IdentityRewriter(), phrase_removed_fn=removed),
    ]
    assert [r["phrase_removed"] for r in results] == [True, False]
    assert aggregate_phrase_removed(results) == 1
    assert aggregate_phrase_removed([]) == 0


def test_prediction_object_classifier_supported():
    class ObjClassifier:
        def predict(self, text: str):
            level = "HIGH" if "guarantee" in text else "LOW"
            return SimpleNamespace(risk_level=SimpleNamespace(value=level))

    result = evaluate_pre_post(HIGH_TEXT, ObjClassifier(), ReducingRewriter())
    assert (result["before_label"], result["after_label"], result["direction"]) == (
        "HIGH",
        "LOW",
        "reduced",
    )
