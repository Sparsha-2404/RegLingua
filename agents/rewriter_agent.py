"""
RegLingua — RewriterAgent (Day 7)
======================================
Owned by: M3 ("Implement RewriterAgent using constrained prompt/rules"),
M4 connects it into the Supervisor (already true structurally since Day
1-2 -- `agents/supervisor.py`'s `rewrite` node has always called
`agents["rewriter_agent"].rewrite(request)` against the
`RewriteRequest -> RewriteResponse` contract; this file just gives that
slot a real implementation).

Gate: "Original -> rewrite -> structured result." (Already satisfied by the
existing `RewriteRequest`/`RewriteResponse` pydantic contract -- this class
just produces genuinely better rewrites than
`agents.mock_agents.MockRewriterAgent`'s two hardcoded substitutions.)

Kept as Version-1 rules/templates per the plan's Section 8 table ("Rewrite:
Rules/templates" for Version 1, "LLM rewrite agent with constrained
prompting" as the later upgrade) -- the contract doesn't change when that
upgrade happens, only this class does.

Design:
  - A phrase -> compliant-replacement table, covering all three regulators'
    banned-phrase families from `core/verification_policy.py` (kept as the
    single source of truth for "what's risky" so the Rewriter and Verifier
    never disagree about what needs fixing).
  - Attempt-based escalation: attempt 1 does targeted phrase substitution;
    attempt 2 additionally appends the regulator's required disclaimer
    (see `core/verification_policy.REQUIRED_DISCLAIMER_PHRASES`); attempt 3+
    falls back to a fully generic, maximally conservative statement for the
    regulator, so the loop is guaranteed to converge well before
    `max_rewrite_iterations` in practice.
  - Reads `verifier_feedback` when present and specifically re-checks for
    the phrase or disclaimer gap it names, rather than blindly re-running
    attempt 1's substitution again.

Note: `RewriteResponse` never carries a compliance verdict -- only
`agents/verifier_agent.py` decides that. This class doesn't and shouldn't
try to judge its own output.
"""

from __future__ import annotations

import re
from typing import Optional

from core.schemas import RewriteRequest, RewriteResponse
from core.verification_policy import (
    DISCLAIMER_REQUIRED_FOR,
    REQUIRED_DISCLAIMER_PHRASES,
    find_banned_phrase,
    has_required_disclaimer,
)

# Phrase -> compliant replacement. Covers `core.verification_policy`'s
# GLOBAL_BANNED_PHRASES plus the three regulators' additions, so a
# substitution exists for everything the Verifier will check for.
_REPLACEMENTS: dict[str, str] = {
    "guaranteed": "potential",
    "guarantee": "aim for",
    "assured return": "potential return, subject to market risk",
    "zero risk": "regulated risk",
    "risk-free": "subject to market risks",
    "risk free": "subject to market risks",
    "100% safe": "regulated and subject to applicable risk",
    "no risk": "subject to applicable risk",
    "highest returns": "competitive returns",
    "double your money": "grow your investment over time",
    "sure-shot profit": "potential returns",
    "instant approval with zero risk": "approval subject to standard eligibility checks",
    "no questions asked loan": "loan subject to standard verification",
    "hidden charges waived forever": "all charges disclosed upfront",
    "share your data with anyone": "share your data only as permitted under applicable law",
    "sell your data": "use your data only as described in our privacy policy",
    "no consent required": "processed with your consent",
}

_GENERIC_FALLBACK: dict[Optional[str], str] = {
    "SEBI": (
        "Returns are market-linked and subject to market risk. "
        "Past performance does not indicate future results."
    ),
    "RBI": (
        "This product is subject to standard eligibility checks, credit approval, "
        "and terms and conditions."
    ),
    "DPDP": (
        "Your data is processed only with your consent and in accordance with "
        "applicable data protection law."
    ),
    None: "Please review this statement for accuracy before publishing.",
}


class RewriterAgent:
    def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        regulator = self._regulator_from_clause(request.clause_cited)

        if request.attempt_number == 1:
            candidate, rationale = self._targeted_substitution(request.flagged_text, regulator)
        elif request.attempt_number == 2:
            base = request.prior_attempt or request.flagged_text
            candidate, rationale = self._substitution_plus_disclaimer(base, regulator, request.verifier_feedback)
        else:
            candidate = _GENERIC_FALLBACK.get(regulator, _GENERIC_FALLBACK[None])
            rationale = (
                f"Attempt {request.attempt_number}: replaced with a generic, "
                f"pre-approved {regulator or 'compliant'} statement."
            )

        return RewriteResponse(
            case_id=request.case_id,
            rewritten_text=candidate,
            attempt_number=request.attempt_number,
            rationale=rationale,
        )

    # ------------------------------------------------------------------

    def _targeted_substitution(self, text: str, regulator: Optional[str]) -> tuple[str, str]:
        candidate = text
        replaced = []
        for phrase, replacement in _REPLACEMENTS.items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(candidate):
                candidate = pattern.sub(replacement, candidate)
                replaced.append(phrase)

        if replaced:
            rationale = f"Replaced flagged phrase(s) {replaced} with compliant wording."
        else:
            rationale = "No known flagged phrase matched a substitution rule; text unchanged on attempt 1."
        return candidate, rationale

    def _substitution_plus_disclaimer(
        self, text: str, regulator: Optional[str], verifier_feedback: Optional[str]
    ) -> tuple[str, str]:
        candidate, _ = self._targeted_substitution(text, regulator)

        still_banned = find_banned_phrase(candidate, regulator)
        if still_banned:
            # verifier_feedback named a phrase our table didn't cover -- strip it directly.
            candidate = re.sub(re.escape(still_banned), "", candidate, flags=re.IGNORECASE).strip()

        if regulator in DISCLAIMER_REQUIRED_FOR and not has_required_disclaimer(candidate, regulator):
            disclaimer = REQUIRED_DISCLAIMER_PHRASES[regulator][0]
            candidate = f"{candidate.rstrip('.')}. This is {disclaimer}."

        rationale = "Re-applied substitutions and appended the required regulatory disclaimer."
        if verifier_feedback:
            rationale += f" (Responding to verifier feedback: {verifier_feedback})"
        return candidate, rationale

    @staticmethod
    def _regulator_from_clause(clause_cited: str) -> Optional[str]:
        """`clause_cited` is a source_id like 'SEBI_MC_IA_2026' -- the
        regulator prefix is a reliable, dependency-free way to recover which
        regulator's rules apply without threading a separate parameter
        through the whole Explainer -> Rewriter handoff."""
        if not clause_cited:
            return None
        for regulator in ("SEBI", "RBI", "DPDP"):
            if clause_cited.upper().startswith(regulator):
                return regulator
        return None
