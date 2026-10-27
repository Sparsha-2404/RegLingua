# M4 Day 10-14 notes

Because other members' work stopped at Day 3, the M1-M3 columns for Day 10-14 were built here too.

## What was built
- **Day 10 AI augmentation:** hybrid retriever + feature reranker + cross-lingual QueryBridge; optional grounded LLM explainer (`agents/llm_agents.py`, behind fallback); multilingual verification policy with negation handling; rule guardrail; grammar-safe multilingual rewriter.
- **Day 10-11 integration:** `core/resilience.py` (timeout/retry/fallback/validate), intake guard, per-mode floors, failure -> NEEDS_REVIEW, `REGLINGUA_DEMO_MODE`, contract freeze + test, `scripts/bootstrap_demo.py`.
- **Day 12:** `scripts/evaluate_day12.py`, `docs/evaluation.md`.
- **Day 13:** `/api/analyze`, `/api/version`, `/api/cases`, HTML UI, docs, Dockerfile/compose, version registry.
- **Day 14:** feature freeze at v1.0.0; `scripts/rehearse_demo.py` (3 runs x 6 cases, all stable) and `docs/demo_fallback_capture.json`.

## Known limitations (read before the demo)
1. **Weak classifier.** 62% accuracy; 28% of HIGH-risk held-out cases are auto-cleared as Compliant. The rule layer helps only on slogan-style text and is not measured on a representative set.
2. **Day 9 correction.** The Day 6-9 hand-off did not flag that the real classifier labelled an obviously risky statement Compliant and that compliant text could be rewritten with nonsense disclaimers. The rule layer, 0.40 production floor and intake guard address the worst of it; they do not fix the model.
3. **Verifier inherits the classifier.** Hedged text can be flagged Risky and escalated (canonical case C6).
4. **Retrieval relevance is document-level**, and the bridge gains are in-distribution (see `docs/evaluation.md`).
5. **Not tested here:** Docker build (no Docker in the sandbox), `AnthropicClient` against the live API, the sentence-transformers encoder and its test (could not be installed; run with `--ignore=tests/unit/test_embeddings.py`), no transformer comparison.
6. **Pickles are sklearn-version fragile**; run `scripts/bootstrap_demo.py` after checkout.
7. Case store for `/scan/{id}/trace` is in-memory only.
8. Two M2 Day 1-2 tests need `data/raw/sample_labeled_examples.csv` (gitignored) and fail in a clean checkout without it.
