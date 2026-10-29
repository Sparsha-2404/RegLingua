"""
RegLingua — ExplainerAgent (Day 6)
=======================================
Owned by: M3/M4 per the plan ("M3: Implement EvidenceAgent and
evidence-grounded Explainer"; EvidenceAgent already landed at Day 1-2, so
this file is the Explainer half, built by M4 alongside the rest of Day 6-9).

Gate: "System produces an explanation tied to explicit evidence references."

Upgrades over `agents.mock_agents.MockExplainerAgent`:
  - Cites every retrieved clause the request carries (not just the first),
    each with its clause_id and source document, so the explanation reads
    as "conflicts with X per [clause_id, clause_id]" rather than a single
    hardcoded reference.
  - Genuinely abstains when there's no evidence to cite -- the old mock's
    "conflicts with UNKNOWN: No matching clause found." asserted a
    violation it couldn't actually back up; this version says plainly that
    no supporting regulation was found (`grounded=False`), matching the
    architecture doc's "abstain instead of inventing a regulation".
  - Compliant-labelled cases get a short, still-evidence-aware "no issue
    found" explanation instead of always assuming something is wrong.

Kept as a Version-1, deterministic template (per the plan's Section 8
AI/ML architecture table: "Explanation: Evidence-grounded template" for
Version 1, "LLM explanation agent" as the later upgrade) -- swapping in an
LLM later only means replacing this class; the contract
(ExplainerRequest -> ExplainerResponse) doesn't change.
"""

from __future__ import annotations

from core.schemas import EvidenceItem, ExplainerRequest, ExplainerResponse, RiskLabel


class ExplainerAgent:
    def explain(self, request: ExplainerRequest) -> ExplainerResponse:
        if request.risk_label == RiskLabel.COMPLIANT:
            return self._explain_compliant(request)
        if not request.evidence:
            return self._explain_ungrounded(request)
        return self._explain_grounded(request)

    # -- Grounded: at least one evidence item was retrieved -------------

    def _explain_grounded(self, request: ExplainerRequest) -> ExplainerResponse:
        top = request.evidence[0]
        others = request.evidence[1:3]  # cite up to 2 more, keep it readable

        citation = self._format_citation(top)
        reason = (
            f"The statement \"{request.flagged_text}\" appears {request.risk_label.value.lower()} "
            f"because it conflicts with {citation}."
        )
        if others:
            extra = ", ".join(self._format_citation(item, short=True) for item in others)
            reason += f" Related guidance: {extra}."

        return ExplainerResponse(
            case_id=request.case_id,
            reason=reason,
            clause_cited=top.clause_id,
            risk_phrase=request.flagged_text,
            grounded=True,
        )

    # -- Ungrounded: risk flagged, but no supporting evidence found ------

    def _explain_ungrounded(self, request: ExplainerRequest) -> ExplainerResponse:
        reason = (
            f"The statement \"{request.flagged_text}\" was flagged as "
            f"{request.risk_label.value.lower()}, but no specific regulatory clause could be "
            "retrieved to support the citation. Flagging on classifier signal alone; "
            "route for manual review rather than treat this as a confirmed violation."
        )
        return ExplainerResponse(
            case_id=request.case_id,
            reason=reason,
            clause_cited="UNSUPPORTED",
            risk_phrase=request.flagged_text,
            grounded=False,
        )

    # -- Compliant: no issue, but still note if evidence was checked -----

    def _explain_compliant(self, request: ExplainerRequest) -> ExplainerResponse:
        reason = f"No compliance issue found in \"{request.flagged_text}\"."
        if request.evidence:
            reason += f" Checked against {request.evidence[0].source_document} and found no conflict."
        return ExplainerResponse(
            case_id=request.case_id,
            reason=reason,
            clause_cited=request.evidence[0].clause_id if request.evidence else "N/A",
            risk_phrase=request.flagged_text,
            grounded=bool(request.evidence),
        )

    @staticmethod
    def _format_citation(item: EvidenceItem, short: bool = False) -> str:
        if short:
            return f"{item.clause_id} ({item.source_document})"
        return f"{item.clause_id} ({item.source_document}): \"{item.clause_text}\""
