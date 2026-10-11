# Day 8 — Verification Policy & Failure Cases

Owned by: M1 ("Define verification policy and failure cases"). Implemented
as `core/verification_policy.py`, consumed by `agents/verifier_agent.py`
(the Verifier) and `agents/rewriter_agent.py` (the Rewriter — reads the same
banned-phrase/disclaimer definitions so the two agents never disagree about
what "compliant" means).

## What counts as a PASS

A rewrite is verified compliant only if **all three** checks pass:

| # | Check | What it catches |
|---|-------|------------------|
| 1 | Independent classifier re-check | Re-runs the (real or mock) RiskAgent on the rewritten text from scratch. Must return `Compliant` with confidence ≥ `REVERIFICATION_CONFIDENCE_FLOOR` (0.3). |
| 2 | No banned phrase | None of `GLOBAL_BANNED_PHRASES` (guarantee, zero risk, risk-free, ...) or the case's regulator's additional banned phrases survive in the rewrite. |
| 3 | Required disclaimer present | For SEBI/RBI/DPDP cases, at least one of that regulator's accepted disclaimer phrasings (e.g. "subject to market risk" for SEBI) appears in the rewrite. Not required for General/unset regulator. |

## Failure reasons

Each failing check produces one `FailureReason`:

- `CLASSIFIER_STILL_RISKY` — re-classification didn't come back Compliant.
- `LOW_REVERIFICATION_CONFIDENCE` — came back Compliant, but too unsure to trust.
- `BANNED_PHRASE_PRESENT` — a specific banned phrase is named in the feedback.
- `MISSING_REQUIRED_DISCLAIMER` — the regulator's disclaimer wasn't found.

All of a case's failure reasons (there can be more than one) are combined
into `state.verifier_feedback`, a single human-readable string the
`RewriterAgent` reads on its next attempt.

## Why the Rewriter can't grade its own work

`RewriteResponse` (`core/schemas.py`) has exactly three fields:
`rewritten_text`, `attempt_number`, `rationale` — there is no
`is_compliant` or `success` field for the Rewriter to set. `state.
verified_compliant` is written in exactly one place in the codebase:
`VerifierAgent.verify()`. This is what the Day 8 gate ("Rewriter cannot
declare its own success; Verifier decides") means literally, not just by
convention.

## Extending the policy

Both `GLOBAL_BANNED_PHRASES` / `REGULATOR_BANNED_PHRASES` and
`REQUIRED_DISCLAIMER_PHRASES` are plain Python lists/dicts in
`core/verification_policy.py` — add a phrase there and both the Rewriter's
substitution table (`agents/rewriter_agent.py`'s `_REPLACEMENTS`, kept
separate since a phrase needs a *replacement*, not just a ban) and the
Verifier's check pick it up. There's deliberately no separate config file or
DSL yet — see the architecture doc's Phase A philosophy: plain, inspectable
Python before any of this becomes an LLM policy agent.
