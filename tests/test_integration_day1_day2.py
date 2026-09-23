"""
Day 1-2 CROSS-TEAM INTEGRATION TEST — the actual point of this merge.

Everything else in tests/ proves one member's module works in isolation
(that's what M2/M3/M4 already had). This file proves the merged system
works end to end, and specifically that M1's real, ingested regulatory
data (extracted from RegLingua_M1_FINAL_v2.xlsx — 516 labelled examples,
308 regulatory-KB chunks) can be swapped in for M2's/M3's original demo
fixtures without touching agents/supervisor.py, agents/contracts.py, or
core/schemas.py.

Per the architecture doc: "Do not wait until Day 11 for the first
cross-team integration." This test is that first integration point.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_factory import M1_REGULATORY_KB_PATH, build_agents  # noqa: E402
from core.schemas import CaseState, EventType, Language, ResolutionStatus  # noqa: E402
from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
M1_LABELED_CSV = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"


# ---------------------------------------------------------------------------
# M1's data assets load and have the shape the rest of the system expects
# ---------------------------------------------------------------------------

def test_m1_labeled_dataset_loads_and_is_substantial():
    with open(M1_LABELED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 500  # M1 delivered 516 examples across SEBI/RBI/DPDP
    assert {"text", "risk_level"}.issubset(rows[0].keys())
    languages = {r["language"] for r in rows}
    assert {"en", "hi", "hi-en"}.issubset(languages)
    risk_levels = {r["risk_level"] for r in rows}
    assert risk_levels == {"LOW", "MEDIUM", "HIGH"}


def test_m1_regulatory_kb_loads_and_covers_all_three_regulators():
    import json

    with open(M1_REGULATORY_KB_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    assert len(chunks) >= 300  # M1 delivered 308 chunks
    regulators = {c["regulator"] for c in chunks}
    assert regulators == {"SEBI", "RBI", "DPDP"}
    assert all(c["text"] and c["source_id"] for c in chunks)


# ---------------------------------------------------------------------------
# The Supervisor graph, wired via build_agents(), runs against M1's real
# regulatory corpus instead of M3's small demo fixture
# ---------------------------------------------------------------------------

def test_supervisor_pipeline_against_m1_real_corpus_sebi_case():
    agents = build_agents(corpus_path=M1_REGULATORY_KB_PATH)

    state = CaseState(input_text="We guarantee assured returns on your investment.")
    state.log(EventType.CASE_STARTED, agent_name="Supervisor")

    lang = agents["language_detector"].detect(state.input_text)
    state.language = Language.HI if lang == "hi" else Language.EN

    state = agents["risk_agent"].classify(state)
    assert state.risk_label.value == "Risky"
    assert state.regulator.value == "SEBI"

    from core.schemas import EvidenceRequest

    response = agents["evidence_agent"].gather(
        EvidenceRequest(
            case_id=state.case_id,
            flagged_text=state.input_text,
            language=state.language,
            regulator_hint=state.regulator,
        )
    )
    # Retrieval must come from M1's real KB, not M3's 10-entry demo fixture
    # — that fixture doesn't have this many SEBI-tagged chunks.
    assert response.evidence, "expected at least one evidence item from M1's real corpus"
    assert all("SEBI" in item.clause_id or "SEBI" in item.source_document for item in response.evidence)


def test_full_run_case_resolves_end_to_end():
    """Uses the module-level Supervisor (M3's small demo corpus, wired by
    default in agents/agent_factory.py) to prove the whole graph — M2's
    classifier, M3's retriever, M4's mocked explain/rewrite/verify loop —
    runs start to finish without manual intervention."""
    from agents.supervisor import run_case

    result = run_case("We offer guaranteed returns of 12% annually.")
    assert result.risk_label.value == "Risky"
    assert result.regulator.value == "SEBI"
    assert result.resolution_status == ResolutionStatus.RESOLVED
    event_types = [e.event_type.value for e in result.trace]
    assert event_types[0] == "case_started"
    for expected in ("language_detected", "risk_classified", "evidence_retrieved", "rewrite_proposed", "verification_run"):
        assert expected in event_types


def test_compliant_text_short_circuits_without_touching_evidence_or_rewrite():
    from agents.supervisor import run_case

    result = run_case("Our app helps you track your monthly budget.")
    assert result.resolution_status == ResolutionStatus.RESOLVED
    assert result.rewrite_attempts == 0
    event_types = [e.event_type.value for e in result.trace]
    assert "evidence_retrieved" not in event_types
    assert "rewrite_proposed" not in event_types


if __name__ == "__main__":
    test_m1_labeled_dataset_loads_and_is_substantial()
    test_m1_regulatory_kb_loads_and_covers_all_three_regulators()
    test_supervisor_pipeline_against_m1_real_corpus_sebi_case()
    test_full_run_case_resolves_end_to_end()
    test_compliant_text_short_circuits_without_touching_evidence_or_rewrite()
    print("\nAll Day 1-2 cross-team (M1+M2+M3+M4) integration checks passed.")
