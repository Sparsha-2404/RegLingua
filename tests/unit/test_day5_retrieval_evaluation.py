"""
Tests proving the Day 5 (M2) evaluation harness's metric math is correct,
BEFORE any real retrieval score from it is trusted.

Day 5 gate: "EvidenceAgent returns selected chunks with source IDs."
M2's Day 5 task: "Evaluate retrieval queries against known relevant
sources."

These tests use small, synthetic fake retrievers and a small, synthetic
eval CSV fixture -- deliberately NOT the real 308-chunk KB or the real
516-row regulator_topic_eval_m2.csv -- so the metric logic in
services/retrieval/evaluation.py is checked in isolation from any
data-quality issue in the real dataset. A failure here means the
implementation is wrong, not the test: these tests encode the correct
definition of Recall@K / Precision@K for this evaluation.
"""
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from services.retrieval.evaluation import evaluate_retriever, load_eval_rows  # noqa: E402


@dataclass
class FakeChunk:
    """Minimal stand-in for RetrievedChunk -- only .source_id is used by
    the evaluation harness."""
    source_id: str


class _BaseFakeRetriever:
    """Shared no-op constructor; subclasses only need .retrieve()."""

    def retrieve(self, query_text, regulator_candidates=None, topic_hint=None, top_k=5):
        raise NotImplementedError


class PerfectRetriever(_BaseFakeRetriever):
    """Always returns the correct source_id first."""

    def retrieve(self, query_text, regulator_candidates=None, topic_hint=None, top_k=5):
        return [FakeChunk("CORRECT_ID")][:top_k]


class UselessRetriever(_BaseFakeRetriever):
    """Always returns results, but never the correct one."""

    def retrieve(self, query_text, regulator_candidates=None, topic_hint=None, top_k=5):
        return [FakeChunk(f"WRONG_ID_{i}") for i in range(top_k)]


class RankThreeRetriever(_BaseFakeRetriever):
    """Returns the correct source_id at position 3 (index 2), wrong before it."""

    def retrieve(self, query_text, regulator_candidates=None, topic_hint=None, top_k=5):
        results = [FakeChunk("WRONG_ID_1"), FakeChunk("WRONG_ID_2"), FakeChunk("CORRECT_ID")]
        results += [FakeChunk(f"WRONG_ID_{i}") for i in range(3, top_k)]
        return results[:top_k]


class EmptyRetriever(_BaseFakeRetriever):
    """Always returns no results at all."""

    def retrieve(self, query_text, regulator_candidates=None, topic_hint=None, top_k=5):
        return []


ONE_EVAL_ROW = [
    {
        "query_id": "Q1",
        "true_regulator": "SEBI",
        "language": "en",
        "expected_source_id": "CORRECT_ID",
        "text": "some query text",
    }
]

K_VALUES = (1, 3, 5)


def test_perfect_retriever_scores_100_percent():
    results = evaluate_retriever(PerfectRetriever(), ONE_EVAL_ROW, k_values=K_VALUES)
    assert results[1]["recall"] == 1.0
    assert results[3]["recall"] == 1.0
    assert results[5]["recall"] == 1.0
    assert results[1]["precision"] == 1.0


def test_useless_retriever_scores_zero():
    results = evaluate_retriever(UselessRetriever(), ONE_EVAL_ROW, k_values=K_VALUES)
    for k in K_VALUES:
        assert results[k]["recall"] == 0.0
        assert results[k]["precision"] == 0.0


def test_rank_3_hit_scores_correctly():
    results = evaluate_retriever(RankThreeRetriever(), ONE_EVAL_ROW, k_values=K_VALUES)
    assert results[1]["recall"] == 0.0
    assert results[3]["recall"] == 1.0
    assert results[5]["recall"] == 1.0
    # Found at rank 3, not rank 1 -- precision@3 reflects the rank, not
    # just "found somewhere within top 3".
    assert results[3]["precision"] == pytest.approx(1 / 3)


def test_empty_retriever_results_scores_zero():
    results = evaluate_retriever(EmptyRetriever(), ONE_EVAL_ROW, k_values=K_VALUES)
    for k in K_VALUES:
        assert results[k]["recall"] == 0.0
        assert results[k]["precision"] == 0.0


def test_load_eval_rows_filters_blank_expected_source_id(tmp_path):
    csv_path = tmp_path / "fixture_eval.csv"
    fieldnames = [
        "query_id", "original_id", "text", "language",
        "true_regulator", "true_topic", "topic_in_kb", "expected_source_id",
    ]
    rows = [
        {"query_id": "Q1", "original_id": "ID1", "text": "a", "language": "en",
         "true_regulator": "SEBI", "true_topic": "t", "topic_in_kb": "True",
         "expected_source_id": "SRC1"},
        {"query_id": "Q2", "original_id": "ID2", "text": "b", "language": "en",
         "true_regulator": "RBI", "true_topic": "t", "topic_in_kb": "False",
         "expected_source_id": ""},
        {"query_id": "Q3", "original_id": "ID3", "text": "c", "language": "hi",
         "true_regulator": "DPDP", "true_topic": "t", "topic_in_kb": "True",
         "expected_source_id": "SRC3"},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_eval_rows(csv_path)
    assert len(loaded) == 2
    assert {r["query_id"] for r in loaded} == {"Q1", "Q3"}
