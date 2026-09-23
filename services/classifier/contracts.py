"""
Shared typed contracts for the NLP Regulatory Intelligence System.

These are the data structures that flow between agents/services, as defined
in the multi-agent architecture blueprint (Section 3 "Agent Responsibilities
& Contracts" and Section 5 "Shared Case State & Audit Trail"). Keeping them
in one place means every module speaks the same schema regardless of whether
a stage is currently a deterministic function (Phase A) or an LLM-backed
agent (Phase B/C) — the Day 1 architecture-review recommendation.

Integration note (Day 1-2 merge of M2/M3/M4): this file was originally
`core/contracts.py` inside M2's own standalone `nlp-regulatory-agents/`
package. M4's `core/schemas.py` (CaseState etc.) is the system-wide state
contract that the Supervisor and every agent-to-agent call actually speaks,
so on merge this module was relocated to `services/classifier/contracts.py`
to make clear it is the classifier service's own internal input/output
type (RiskPrediction), not a second competing "core" schema.
`agents/risk_agent_adapter.py` is the only place that translates between
this file's `RiskLevel` and `core.schemas.RiskLabel`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    NEEDS_REVIEW = "NEEDS_REVIEW"  # low-confidence abstention, see Section 15


@dataclass
class RiskPrediction:
    """Output contract of the Risk Analyst agent / classifier (Section 3, Agent 2)."""

    risk_level: RiskLevel
    confidence: float
    risk_factors: List[str] = field(default_factory=list)
    regulator_candidates: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.risk_level, str):
            self.risk_level = RiskLevel(self.risk_level)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "confidence": round(self.confidence, 4),
            "risk_factors": list(self.risk_factors),
            "regulator_candidates": list(self.regulator_candidates),
        }


@dataclass
class AgentOutput:
    """
    Generic structured-output envelope every agent returns (Section 3), e.g.:

        {
          "agent": "risk_analyst",
          "status": "success",
          "confidence": 0.86,
          "result": {...},
          "evidence_refs": [],
          "next_action": "regulatory_research"
        }
    """

    agent: str
    status: str  # "success" | "error" | "needs_review"
    confidence: float
    result: dict
    evidence_refs: List[str] = field(default_factory=list)
    next_action: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "result": self.result,
            "evidence_refs": list(self.evidence_refs),
            "next_action": self.next_action,
        }
