"""
Tests proving the M4 Day 3-5 gates are actually met, against the real
artifacts M1/M2/M3 delivered through Day 3 (not the Day 1-2 mocks):

Day 3 gate: "Real text -> real risk prediction -> CaseState."
Day 4 gate: "Query retrieves source-linked regulatory chunks."
Day 5 gate: "EvidenceAgent returns selected chunks with source IDs."

These are skipped (not failed) when the trained artifacts haven't been
built locally yet — `models/` is gitignored. Run, from the repo root:
    python scripts/extract_m1_labeled_csv.py
    python scripts/train_baseline_classifier.py
    python scripts/train_regulator_classifier.py
before running this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.agent_factory import (  # noqa: E402
    M1_REGULATORY_KB_PATH,
    REGULATOR_MODEL_PATH,
    RISK_MODEL_PATH,
    build_agents,
)
from agents.supervisor import build_supervisor_graph  # noqa: E402
from agents.tools import build_tool_registry  # noqa: E402
from core.schemas import CaseState, EventType  # noqa: E402

pytestmark = pytest.mark.skipif(
    not RISK_MODEL_PATH.exists(),
    reason=(
        f"{RISK_MODEL_PATH} not found. Run scripts/extract_m1_labeled_csv.py "
        "then scripts/train_baseline_classifier.py first."
    ),
)


@pytest.fixture(scope="module")
def production_agents():
    return build_agents(use_trained_artifacts=True)


# ---------------------------------------------------------------------------
# Day 3 — "Real text -> real risk prediction -> CaseState."
# ---------------------------------------------------------------------------

def test_production_agents_load_real_classifier_not_mock(production_agents):
    from services.classifier.interface import MockRiskClassifier

    assert not isinstance(production_agents["risk_agent"].classifier, MockRiskClassifier)


def test_real_classifier_flows_into_case_state(production_agents):
    state = CaseState(input_text="We guarantee 20% assured returns on your investment, absolutely risk-free.")
    result = production_agents["risk_agent"].classify(state)

    assert result.risk_label is not None
    assert 0.0 <= result.classifier_confidence <= 1.0
    # MockRiskClassifier's fingerprint is one of exactly two fixed
    # confidences (0.6 / 0.85) -- a real probabilistic model essentially
    # never lands on them, so this proves real inference actually ran.
    assert result.classifier_confidence not in {0.6, 0.85}
    assert result.trace and result.trace[0].agent_name == "TfidfLogisticClassifier"


@pytest.mark.skipif(not REGULATOR_MODEL_PATH.exists(), reason="Run scripts/train_regulator_classifier.py first.")
def test_regulator_gap_filler_populates_regulator_when_classifier_does_not(production_agents):
    # M2's real TfidfLogisticClassifier never sets regulator_candidates
    # (see services/classifier/regulator_classifier.py's docstring) -- this
    # is what closes that gap for a clearly SEBI-flavoured statement.
    state = CaseState(input_text="Our platform guarantees fixed high returns on mutual fund investments.")
    result = production_agents["risk_agent"].classify(state)
    assert result.regulator is not None


def test_full_production_graph_resolves_end_to_end(production_agents):
    """The Day 3-5 production Supervisor graph -- real classifier, real M1
    KB -- runs a case to completion without raising."""
    graph = build_supervisor_graph(production_agents)
    state = CaseState(input_text="We guarantee 20% assured returns on your investment, absolutely risk-free.")
    state.log(EventType.CASE_STARTED, agent_name="Supervisor")
    result = graph.invoke(state)
    result = result if isinstance(result, CaseState) else CaseState(**result)

    assert result.resolution_status in {"resolved", "escalated"}
    assert result.risk_label is not None


# ---------------------------------------------------------------------------
# Day 4 — "Query retrieves source-linked regulatory chunks."
# ---------------------------------------------------------------------------

def test_research_tool_returns_source_linked_chunks(production_agents):
    registry = build_tool_registry(production_agents)
    result = registry.call("regulatory_research", query="guaranteed returns on mutual fund investments")

    assert result["results"], "expected at least one retrieved chunk"
    for chunk in result["results"]:
        assert chunk["source_id"], "every retrieved chunk must carry a source_id"


def test_research_tool_filters_by_regulator(production_agents):
    registry = build_tool_registry(production_agents)
    result = registry.call(
        "regulatory_research",
        query="data privacy consent",
        regulator_candidates=["DPDP"],
    )
    assert result["results"]
    assert all(r["regulator"] == "DPDP" for r in result["results"])


def test_research_tool_is_listed_with_schema(production_agents):
    registry = build_tool_registry(production_agents)
    names = {t["name"] for t in registry.list()}
    assert "regulatory_research" in names


# ---------------------------------------------------------------------------
# Day 5 — "EvidenceAgent returns selected chunks with source IDs."
# ---------------------------------------------------------------------------

def test_evidence_agent_against_real_m1_kb_returns_clause_ids():
    agents = build_agents(use_trained_artifacts=True, corpus_path=M1_REGULATORY_KB_PATH)
    registry = build_tool_registry(agents)

    result = registry.call(
        "evidence_gathering",
        text="We guarantee assured returns on mutual fund investments.",
        regulator_hint="SEBI",
    )
    evidence = result["evidence"]
    assert evidence, "expected at least one evidence item from the real M1 KB"
    for item in evidence:
        assert item["clause_id"], "every evidence item must carry a clause_id (source ID)"
