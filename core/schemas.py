"""
RegLingua — Core Schemas
=========================
Owned by: M4 (Platform/Orchestration Engineer)
Day 1 deliverable: CaseState, event schema, Evidence/Explainer/Rewriter
request/response models.

These are the single source of truth for what flows through the graph.
Every agent — whether owned by M2, M3, or M4, whether mocked (Day 2) or a
real model/LLM call (Day 9) — reads and writes against these models. That's
what lets the Supervisor's routing logic stay untouched when a mock is
swapped for a real implementation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskLabel(str, Enum):
    COMPLIANT = "Compliant"
    RISKY = "Risky"
    AMBIGUOUS = "Ambiguous"


class Regulator(str, Enum):
    SEBI = "SEBI"
    RBI = "RBI"
    DPDP = "DPDP"
    GENERAL = "General"


class Language(str, Enum):
    EN = "en"
    HI = "hi"
    CODE_MIXED = "hi-en"


class ResolutionStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    NEEDS_REVIEW = "needs_review"  # Day 9: abstention -- low confidence or no
    # evidence found. Distinct from ESCALATED (which means "the
    # rewrite/verify loop ran and failed"): NEEDS_REVIEW means the system
    # deliberately declined to guess rather than force a call it isn't
    # confident in (architecture doc Section 15: "abstain instead of
    # inventing a regulation").


class EventType(str, Enum):
    CASE_STARTED = "case_started"
    LANGUAGE_DETECTED = "language_detected"
    RISK_CLASSIFIED = "risk_classified"
    EVIDENCE_RETRIEVED = "evidence_retrieved"
    EXPLANATION_GENERATED = "explanation_generated"
    REWRITE_PROPOSED = "rewrite_proposed"
    VERIFICATION_RUN = "verification_run"
    ITERATION_RETRY = "iteration_retry"
    ESCALATED = "escalated"
    ABSTAINED = "abstained"  # Day 9: routed to NEEDS_REVIEW instead of
    # forcing a low-confidence call or fabricating a citation.
    CASE_RESOLVED = "case_resolved"
    AGENT_ERROR = "agent_error"


# ---------------------------------------------------------------------------
# Evidence contract (M4-owned; moved from M3 per the revised ownership table)
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    clause_id: str
    source_document: str
    clause_text: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class EvidenceRequest(BaseModel):
    case_id: str
    flagged_text: str
    language: Language
    regulator_hint: Optional[Regulator] = None
    top_k: int = 5


class EvidenceResponse(BaseModel):
    case_id: str
    evidence: list[EvidenceItem]
    retrieval_method: Literal["vector", "hybrid", "keyword"] = "hybrid"


# ---------------------------------------------------------------------------
# Explainer contract (M4-owned)
# ---------------------------------------------------------------------------

class ExplainerRequest(BaseModel):
    case_id: str
    flagged_text: str
    risk_label: RiskLabel
    evidence: list[EvidenceItem]


class ExplainerResponse(BaseModel):
    case_id: str
    reason: str
    clause_cited: str
    risk_phrase: str
    grounded: bool = True  # False if the explainer had no solid evidence to cite


# ---------------------------------------------------------------------------
# Rewriter contract (M4-owned)
# ---------------------------------------------------------------------------

class RewriteRequest(BaseModel):
    case_id: str
    flagged_text: str
    clause_cited: str
    reason: str
    prior_attempt: Optional[str] = None
    verifier_feedback: Optional[str] = None
    attempt_number: int = 1


class RewriteResponse(BaseModel):
    case_id: str
    rewritten_text: str
    attempt_number: int
    rationale: str


# ---------------------------------------------------------------------------
# Event schema — every node in the graph appends one of these to the trace
# ---------------------------------------------------------------------------

class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    event_type: EventType
    agent_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# CaseState — the state object threaded through the LangGraph StateGraph
# ---------------------------------------------------------------------------

class CaseState(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Input
    input_text: str
    language: Optional[Language] = None

    # RiskAgent output (M2-owned classifier; mocked for now)
    risk_label: Optional[RiskLabel] = None
    regulator: Optional[Regulator] = None
    classifier_confidence: Optional[float] = None

    # EvidenceAgent / ExplainerAgent output (M4-owned)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reason: Optional[str] = None
    clause_cited: Optional[str] = None
    risk_phrase: Optional[str] = None

    # RewriterAgent <-> VerifierAgent loop
    rewrite_candidate: Optional[str] = None
    rewrite_attempts: int = 0
    max_rewrite_iterations: int = 3
    verifier_feedback: Optional[str] = None
    verified_compliant: Optional[bool] = None

    # Bookkeeping
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    trace: list[AgentEvent] = Field(default_factory=list)

    def log(self, event_type: EventType, agent_name: str, **payload: Any) -> None:
        """Append a trace event. Called by every node in the Supervisor graph
        so the full agent history is inspectable per case (used later by the
        Streamlit UI's agent trace panel)."""
        self.trace.append(
            AgentEvent(
                case_id=self.case_id,
                event_type=event_type,
                agent_name=agent_name,
                payload=payload,
            )
        )
