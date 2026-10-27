"""
RegLingua — Agent / prompt / contract version registry (Day 13, M3)
=======================================================================
"Agent/tool documentation and prompt/version tracking."

One dict per concern, bumped by hand whenever behaviour changes. The
version strings are surfaced in two places so a result can always be tied
back to the exact code that produced it:
  - `GET /api/version`
  - every `/api/analyze` response (`versions` block)

`CONTRACT_VERSION` is frozen at Day 11 (see `tests/test_contract_freeze.py`,
which fails if `CaseState` or the agent request/response models drift from
`core/contract_snapshot.json` without this number being bumped).
"""

from __future__ import annotations

CONTRACT_VERSION = "1.0.0"
RELEASE = "v1.0.0"

AGENT_VERSIONS: dict[str, str] = {
    "supervisor": "1.0.0",       # deterministic state machine, conditional routing, bounded retries
    "intake": "1.0.0",           # language detection + input validation guard
    "risk_analyst": "1.1.0",     # TF-IDF+LR with rule-based keyword guardrail (Day 10)
    "regulator_classifier": "1.0.0",
    "regulatory_researcher": "1.1.0",  # hybrid BM25 + vector + rerank (Day 10)
    "evidence": "1.0.0",
    "explainer": "1.1.0",        # evidence-grounded template; optional LLM variant (Day 10)
    "rewriter": "1.1.0",         # rules/templates
    "verifier": "1.0.0",         # classifier re-check + policy checks
}

PROMPT_VERSIONS: dict[str, str] = {
    # Only the optional LLM explainer uses a prompt; deterministic agents
    # have none. Bump when the text in agents/llm_agents.py changes.
    "llm_explainer": "1.0.0",
}

MODEL_VERSIONS: dict[str, str] = {
    "risk_classifier": "tfidf-logreg-baseline-v1",
    "regulator_classifier": "tfidf-logreg-v1",
    "vector_encoder": "tfidf-char-ngram (offline) | paraphrase-multilingual-MiniLM-L12-v2 (optional)",
}


def snapshot() -> dict:
    return {
        "release": RELEASE,
        "contract_version": CONTRACT_VERSION,
        "agents": dict(AGENT_VERSIONS),
        "prompts": dict(PROMPT_VERSIONS),
        "models": dict(MODEL_VERSIONS),
    }
