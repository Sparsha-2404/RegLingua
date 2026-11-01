"""Day 10-14 (M4-owned): resilience layer, intake guard, supervisor failure routing,
API/UI, contract freeze, canonical demo cases. Depends only on M4 code + Day 3-9 agents."""
import json
import time
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

from agents.supervisor import build_supervisor_graph, run_case, validate_input
from core import versions
from core.resilience import StructuredOutputError, call_with_resilience
from core.schemas import CaseState, ResolutionStatus


# ---------------- resilience ----------------
def test_resilience_success_first_try():
    o = call_with_resilience(lambda: 42)
    assert o.ok and o.value == 42 and o.status == "success" and o.attempts == 1


def test_resilience_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    o = call_with_resilience(flaky, max_retries=1)
    assert o.ok and o.status == "retried" and o.value == "ok"


def test_resilience_uses_fallback_after_failures():
    def bad():
        raise RuntimeError("down")

    o = call_with_resilience(bad, fallback=lambda: "safe", max_retries=1)
    assert o.ok and o.used_fallback and o.value == "safe" and o.status == "fallback"


def test_resilience_total_failure_is_not_ok():
    def bad():
        raise RuntimeError("down")

    o = call_with_resilience(bad, fallback=bad, max_retries=0)
    assert not o.ok and o.status == "failed" and o.error


def test_resilience_timeout():
    o = call_with_resilience(lambda: time.sleep(1.0), timeout_s=0.1, max_retries=0)
    assert not o.ok


def test_resilience_validation_failure_triggers_fallback():
    def v(x):
        if not x:
            raise StructuredOutputError("empty")

    o = call_with_resilience(lambda: "", fallback=lambda: "fine", validate=v, max_retries=0)
    assert o.ok and o.used_fallback and o.value == "fine"


# ---------------- intake guard ----------------
@pytest.mark.parametrize("text", ["", "   ", "hi", "ok sure", "!!! ???", "x" * 6000])
def test_invalid_inputs_are_rejected(text):
    assert validate_input(text) is not None


def test_invalid_input_routes_to_needs_review():
    r = run_case("hi", use_trained_artifacts=True)
    assert r.resolution_status == ResolutionStatus.NEEDS_REVIEW
    assert r.risk_label is None and not r.rewrite_candidate


def test_valid_input_passes_guard():
    assert validate_input("We guarantee assured returns on this scheme") is None


def test_trace_records_every_stage():
    r = run_case("We guarantee 100% safe returns with no risk on this mutual fund, invest now.", use_trained_artifacts=True)
    kinds = {e.event_type.value for e in r.trace}
    assert {"language_detected", "evidence_retrieved", "explanation_generated", "rewrite_proposed"} <= kinds


def test_agent_failure_degrades_to_needs_review_not_crash():
    from agents.agent_factory import build_agents

    agents = build_agents(use_trained_artifacts=True, production=True)

    class Boom:
        def gather(self, *_):
            raise RuntimeError("retriever down")

    agents["evidence_agent"] = Boom()
    agents["fallback_evidence_agent"] = Boom()
    g = build_supervisor_graph(agents, compliant_floor=0.7, abstain_floor=0.0)
    out = g.invoke(CaseState(input_text="We guarantee 100% safe returns with no risk on this fund"))
    state = CaseState(**out) if isinstance(out, dict) else out
    assert state.resolution_status == ResolutionStatus.NEEDS_REVIEW
    assert any(e.event_type.value == "agent_error" for e in state.trace)


# ---------------- API ----------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from apps.api.main import app

    return TestClient(app)


def test_api_version_matches_registry(client):
    v = client.get("/api/version").json()
    assert v["contract_version"] == versions.CONTRACT_VERSION and v["release"] == versions.RELEASE


def test_api_analyze_shape(client):
    d = client.post("/api/analyze", json={"text": "We guarantee 100% safe returns with no risk on this fund"}).json()
    for k in ("case_id", "risk_label", "evidence", "trace", "versions", "resolution_status", "disclaimer"):
        assert k in d


def test_api_analyze_bad_input_does_not_500(client):
    r = client.post("/api/analyze", json={"text": "x"})
    assert r.status_code == 200 and r.json()["resolution_status"] == "needs_review"


def test_api_trace_roundtrip(client):
    cid = client.post("/api/analyze", json={"text": "We guarantee 100% safe returns with no risk"}).json()["case_id"]
    assert client.get(f"/scan/{cid}/trace").status_code == 200
    assert client.get("/scan/nope/trace").status_code == 404


def test_ui_and_cases_served(client):
    assert "RegLingua" in client.get("/").text
    assert len(client.get("/api/cases").json()["cases"]) >= 5


# ---------------- contract freeze (Day 11) ----------------
def test_contract_matches_frozen_snapshot():
    from scripts.freeze_contract import build

    frozen = json.loads((ROOT / "core" / "contract_snapshot.json").read_text())
    assert build() == frozen, (
        "Agent contract changed. If intentional: bump core.versions.CONTRACT_VERSION and run "
        "`python3 scripts/freeze_contract.py`."
    )


def test_canonical_cases_all_run_without_exception():
    cases = json.loads((ROOT / "data" / "demo" / "canonical_cases.json").read_text(encoding="utf-8"))
    for c in cases:
        r = run_case(c["text"], use_trained_artifacts=True)
        assert r.resolution_status != ResolutionStatus.PENDING, c["id"]
