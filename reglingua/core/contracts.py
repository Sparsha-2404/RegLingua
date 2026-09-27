"""
Shared contracts for the Regulatory Research Agent (M3 — reduced scope: retrieval,
embeddings, hybrid search only).

Day 1 deliverable: "Define Research agent contract."
Gate: all modules have interfaces even if implementations are mocks.

Every other agent (Supervisor, Risk Analyst, Evidence Agent) that talks to the
Regulatory Research Agent should depend only on the types in this file, never
on internal retrieval implementation details (deterministic keyword search
today, hybrid BM25 + vector search from Day 5 onward). That's what lets M3
swap the retriever underneath without breaking anyone downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
import json

RiskLabel = Literal["LOW", "MEDIUM", "HIGH"]
Regulator = Literal["SEBI", "RBI", "DPDP"]
AgentStatus = Literal["success", "partial", "failure"]


@dataclass
class RiskPrediction:
    """Mirrors the Risk Analyst agent's output (Agent 2 in the architecture doc)."""
    label: RiskLabel
    confidence: float
    factors: List[str] = field(default_factory=list)


@dataclass
class RegulatoryResearchInput:
    """
    Input contract for Agent 3 — Regulatory Researcher.

    Per the architecture doc:
        Input: Text + risk + regulator/topic candidate
    """
    case_id: str
    normalized_text: str
    language: str                       # "en" | "hi" | "hi-en" (code-mixed)
    risk_prediction: RiskPrediction
    regulator_candidates: List[Regulator] = field(default_factory=list)
    topic_hint: Optional[str] = None
    top_k: int = 5

    def validate(self) -> None:
        if not self.normalized_text.strip():
            raise ValueError("normalized_text must not be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        for reg in self.regulator_candidates:
            if reg not in ("SEBI", "RBI", "DPDP"):
                raise ValueError(f"Unknown regulator candidate: {reg}")


@dataclass
class RetrievedChunk:
    """One retrieved regulatory passage, matching the metadata schema in the architecture doc."""
    source_id: str
    regulator: Regulator
    document_title: str
    section: str
    effective_date: str
    text: str
    source_url: str
    topic: str
    score: float


@dataclass
class RegulatoryResearchResult:
    retrieval_results: List[RetrievedChunk] = field(default_factory=list)
    regulator_candidates: List[Regulator] = field(default_factory=list)


@dataclass
class RegulatoryResearchOutput:
    """
    Output contract for Agent 3, matching the "Structured Agent Output" shape
    from the architecture doc: agent / status / confidence / result /
    evidence_refs / next_action.
    """
    agent: str
    status: AgentStatus
    confidence: float
    result: RegulatoryResearchResult
    evidence_refs: List[str] = field(default_factory=list)
    next_action: str = "evidence_selection"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
