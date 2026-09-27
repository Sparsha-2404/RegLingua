"""
RegLingua — FastAPI Endpoint
===============================
Owned by: M4
Day 1 deliverable: the endpoint skeleton, wired to run_case() once Day 2's
pipeline existed.

Day 3-5 additions (M4):
  - /scan now runs against the production graph (real Day 3 classifier +
    real Day 4-5 M1 knowledge base) by default — set
    `"use_real_classifier": false` in the request body to fall back to the
    original Day 1-2 mock pipeline instead.
  - /tools, /tools/{name}/invoke expose the Day 4 ResearchAgent tool
    interface (and the risk/evidence tools alongside it) directly over
    HTTP, independent of a full /scan run — see agents/tools.py.
  - /health reports whether the real trained artifacts are actually
    loaded, so "is this the mock or the real pipeline?" is answerable
    without reading logs.

Run locally with (from the repo root):
    uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.agent_factory import REGULATOR_MODEL_PATH, RISK_MODEL_PATH, build_agents
from agents.supervisor import run_case
from agents.tools import ToolRegistry, build_tool_registry
from core.schemas import CaseState

app = FastAPI(
    title="RegLingua API",
    description="Multi-agent compliance scanner for Indian fintech product text (SEBI / RBI / DPDP).",
    version="0.1.0",
)

# Built lazily (not at import time) so importing this module never requires
# the trained-artifact files to exist, and cached because loading the pickled
# TF-IDF vectorizers + M1's 308-chunk KB on every request would be wasteful.
_tool_registry: Optional[ToolRegistry] = None


def _get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = build_tool_registry(build_agents(use_trained_artifacts=True))
    return _tool_registry


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Product text to scan (English or Hindi).")
    max_rewrite_iterations: int = Field(3, ge=1, le=10)
    use_real_classifier: bool = Field(
        True,
        description=(
            "True (default): run against the production graph — M2's real "
            "trained classifier + M1's real regulatory knowledge base. "
            "False: run against the original Day 1-2 deterministic-mock "
            "pipeline instead."
        ),
    )


class ScanResponse(BaseModel):
    case_id: str
    language: Optional[str]
    risk_label: Optional[str]
    regulator: Optional[str]
    reason: Optional[str]
    clause_cited: Optional[str]
    rewrite_candidate: Optional[str]
    verified_compliant: Optional[bool]
    resolution_status: str
    rewrite_attempts: int


class ToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "production_artifacts": {
            "risk_classifier_trained": RISK_MODEL_PATH.exists(),
            "regulator_classifier_trained": REGULATOR_MODEL_PATH.exists(),
        },
    }


@app.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest) -> ScanResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result: CaseState = run_case(
        request.text,
        max_rewrite_iterations=request.max_rewrite_iterations,
        use_trained_artifacts=request.use_real_classifier,
    )

    return ScanResponse(
        case_id=result.case_id,
        language=result.language,
        risk_label=result.risk_label,
        regulator=result.regulator,
        reason=result.reason,
        clause_cited=result.clause_cited,
        rewrite_candidate=result.rewrite_candidate,
        verified_compliant=result.verified_compliant,
        resolution_status=result.resolution_status,
        rewrite_attempts=result.rewrite_attempts,
    )


@app.get("/scan/{case_id}/trace")
def get_trace(case_id: str) -> dict:
    # Stub only: a per-case store lands alongside the demo UI, so /scan is
    # currently stateless. Documenting the contract now so the UI has a
    # stable endpoint to build the agent-trace panel against.
    raise HTTPException(status_code=501, detail="Case trace store not implemented yet.")


# ---------------------------------------------------------------------------
# Day 4-5 (M4): tool interface, exposed directly over HTTP.
# ---------------------------------------------------------------------------

@app.get("/tools")
def list_tools() -> dict:
    """Day 4 gate support: lists the callable tools (regulatory_research,
    risk_classification, evidence_gathering) with their JSON-schema input
    shape — the same shape a future AI-planning Supervisor would be handed."""
    return {"tools": _get_tool_registry().list()}


@app.post("/tools/{name}/invoke")
def invoke_tool(name: str, request: ToolInvokeRequest) -> dict:
    """Day 4 gate: "Query retrieves source-linked regulatory chunks" —
    call with name="regulatory_research", arguments={"query": "..."}.
    Day 5 gate: "EvidenceAgent returns selected chunks with source IDs" —
    call with name="evidence_gathering", arguments={"text": "..."}.
    """
    try:
        return _get_tool_registry().call(name, **request.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=f"Bad arguments for tool '{name}': {exc}") from exc
