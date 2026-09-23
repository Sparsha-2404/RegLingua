"""
RegLingua — Agent Contracts (Day 1 draft)
============================================
Owned by: M4

Each contract is a `Protocol`, not a base class — any callable satisfies it
as long as the method signature matches. That's deliberate: it means a mock
(Day 2), a real MuRIL classifier (M2), a FAISS-backed retriever (M3), or a
Gemini-backed rewriter can all be dropped in without the Supervisor's graph
wiring changing at all.

Ownership per the revised table:
  - RiskAgent               M2 (Classifier)
  - ResearchAgent            M3, reduced scope — retrieval/embeddings/hybrid search only
  - EvidenceAgent            M4 — moved from M3
  - ExplainerAgent           M4 — moved from M3, evidence-grounded
  - RewriterAgent            M4 — moved from M3, constrained rewrite
  - VerifierAgent            M4 — independent risk re-check + evidence/constraint checks
  - Supervisor               M4 — owns all wiring/state/routing
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas import (
    CaseState,
    EvidenceRequest,
    EvidenceResponse,
    ExplainerRequest,
    ExplainerResponse,
    RewriteRequest,
    RewriteResponse,
)


@runtime_checkable
class LanguageDetectorAgent(Protocol):
    def detect(self, text: str) -> str:
        """Return an ISO-ish language code, e.g. 'en' or 'hi'."""
        ...


@runtime_checkable
class RiskAgent(Protocol):
    """Classifier Agent — M2-owned. Contract lives here so the Supervisor
    (M4) can type-check and mock against it before the real MuRIL+LoRA model
    exists."""

    def classify(self, state: CaseState) -> CaseState:
        """Read state.input_text; write risk_label, regulator,
        classifier_confidence (and risk_phrase, if applicable) back onto the
        state and return it."""
        ...


@runtime_checkable
class ResearchAgent(Protocol):
    """Regulatory Researcher agent — M3-owned, reduced scope: retrieval,
    embeddings, hybrid search only. EvidenceAgent (below) wraps this."""

    def research(self, request: EvidenceRequest) -> EvidenceResponse:
        ...


@runtime_checkable
class EvidenceAgent(Protocol):
    """M4-owned. Case-scoped wrapper around ResearchAgent's raw hits."""

    def gather(self, request: EvidenceRequest) -> EvidenceResponse:
        ...


@runtime_checkable
class ExplainerAgent(Protocol):
    """M4-owned. Turns cited evidence into a plain-language, evidence-grounded
    explanation. Must not fabricate a clause when evidence is empty — see
    ExplainerResponse.grounded."""

    def explain(self, request: ExplainerRequest) -> ExplainerResponse:
        ...


@runtime_checkable
class RewriterAgent(Protocol):
    """M4-owned. Proposes a compliant rewrite given the flagged text and
    cited clause. Must accept `verifier_feedback` on retries (attempt_number
    > 1) and use it — not just regenerate the same candidate."""

    def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        ...


@runtime_checkable
class VerifierAgent(Protocol):
    """M4-owned. Independently re-checks a rewrite candidate (typically by
    re-running it through RiskAgent) plus any evidence/constraint checks, and
    writes verified_compliant / verifier_feedback back onto the state."""

    def verify(self, state: CaseState) -> CaseState:
        ...
