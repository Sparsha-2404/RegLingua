"""
RegLingua — FastAPI Endpoint
===============================
Owned by: M4
Day 1 deliverable: the endpoint skeleton, wired to run_case() once Day 2's
mocked Supervisor pipeline existed.

Run locally with:
    uvicorn reglingua_supervisor.main:app --reload
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .schemas import CaseState
from .supervisor import run_case

app = FastAPI(
    title="RegLingua API",
    description="Multi-agent compliance scanner for Indian fintech product text (SEBI / RBI / DPDP).",
    version="0.1.0",
)


class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Product text to scan (English or Hindi).")
    max_rewrite_iterations: int = Field(3, ge=1, le=10)


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest) -> ScanResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    result: CaseState = run_case(request.text, max_rewrite_iterations=request.max_rewrite_iterations)

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
    # Stub only: a per-case store lands alongside the Streamlit chat
    # interface (Phase 8 / session state work), so /scan is currently
    # stateless. Documenting the contract now so app/app.py has a stable
    # endpoint to build the agent-trace panel against.
    raise HTTPException(status_code=501, detail="Case trace store not implemented yet.")
