# Agent contracts (frozen at v1.0.0)

Source of truth: `agents/contracts.py` (Protocols) and `core/schemas.py` (pydantic). A machine-readable snapshot is
`core/contract_snapshot.json`; to change a contract, bump `core/versions.CONTRACT_VERSION`, run
`python3 scripts/freeze_contract.py`, and say so in the commit.

| Stage | Agent | Input | Output | Deterministic fallback |
|---|---|---|---|---|
| Intake | LanguageAgentAdapter + `validate_input` | text | `language`, or NEEDS_REVIEW if empty / < 3 words / > 5000 chars | none (reject) |
| Risk | RiskAgentAdapter over RuleAugmentedClassifier(TF-IDF+LR) | CaseState | `risk_label`, `classifier_confidence`, `risk_phrase`, `regulator` | un-augmented baseline |
| Evidence | EvidenceAgentAdapter over BridgedHybridRetriever (`services/retrieval/bridged_retriever.py`; M3's `hybrid_retriever.py` is untouched) (BM25 + char-ngram vector, RRF, feature rerank, cross-lingual bridge) | `EvidenceRequest` | `EvidenceResponse` (clause id, source, text, score) | DeterministicRetriever |
| Explain | ExplainerAgent (template) or LLMExplainerAgent (optional) | `ExplainerRequest` | reason + `clause_cited`, validated by `core/grounding.py` | template explainer |
| Rewrite | RewriterAgent (rules, EN/HI/Hinglish) | `RewriteRequest` | `rewritten_text` | generic safe rewrite on attempt 3 |
| Verify | VerifierAgent (re-classify + policy) | CaseState | `verified_compliant`, `verifier_feedback` | none: fails closed |

## Supervisor routing
`detect_language -> (invalid? needs_review) -> classify -> {compliant | needs_review | gather_evidence}`
`gather_evidence -> {explain | needs_review} -> rewrite -> verify -> {resolve | rewrite (<= max iterations) | escalate}`

- Production floors: abstain < 0.40, auto-compliant >= 0.70 (`PRODUCTION_*` in `agents/supervisor.py`). Defaults for the Day 1-9 mock graph: 0.35 / 0.70.
- Every agent call goes through `core/resilience.call_with_resilience`: timeout -> retry (default 1) -> fallback -> validate structured output -> NEEDS_REVIEW. Non-routine outcomes are logged as `AGENT_ERROR` trace events.
- Resolution statuses: `resolved`, `escalated`, `needs_review`, `abstained`.

## Guardrails
- The rule layer can only raise risk. Negated claims ("do not guarantee") are not flagged by rules.
- The verifier is the final decider; there is no fallback verifier by design.
- Explanations must cite a clause present in the retrieved evidence, <= 1200 chars, with no invented numbered provisions.

## Versioning
`GET /api/version` returns release, contract, per-agent, prompt and model versions; every `/api/analyze` response carries the same block.
