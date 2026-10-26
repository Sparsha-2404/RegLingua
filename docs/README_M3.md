# M3 - Regulatory Research Agent

## Scope

M3 implements the regulatory research and retrieval layer. Under the revised ownership table, Evidence, Explainer, and Rewriter responsibilities belong to M4. Existing files alone do not prove that all M4 work is complete.

## Implemented Components

### 1. Research Agent Contract

The research agent uses structured input/output contracts for case identifiers, normalized text, language, regulator candidates, optional topic hints, and retrieval settings. Results include source IDs, regulator, document metadata, text, source URL, topic, and relevance score.

### 2. Deterministic Keyword Retrieval

The keyword retriever supports regulator filtering, Devanagari-aware tokenization, stopword handling, and topic hints.

### 3. Multilingual Embeddings and Vector Index

The embedding encoder generates multilingual sentence embeddings. The vector index loads the JSON corpus, creates embeddings, supports metadata filters, and ranks candidates using cosine similarity.

### 4. Hybrid Retrieval

`services/retrieval/hybrid_retriever.py` combines keyword and vector retrieval using weights of 0.5 each. It merges candidates, computes combined scores, reranks results, and returns the top-k results.

### 5. Research-Agent Integration and Abstention

`agents/regulatory_research_agent.py` calls the hybrid retriever. If no results are returned or the top score is below 0.1, the agent returns `failure`, zero confidence, no evidence references, and `needs_review`. Scores of at least 0.5 are marked `success`; other accepted results are marked `partial`.

The 0.1 abstention threshold is provisional and requires calibration against a larger labelled dataset.

## Corpus and Evaluation

The current `data/regulatory/sample_corpus.json` contains 10 synthetic regulatory entries covering SEBI, RBI, and DPDP topics, including a Hindi entry. This development corpus is not a substitute for a verified, up-to-date collection of official regulatory sources.

Run retrieval evaluation with:

```powershell
python tests/evaluation/evaluate_retrieval.py
```

The evaluation currently contains six manually labelled English, code-mixed, and Hindi queries. The latest observed run reported:

- Mean Precision@3: 0.333
- Mean Recall@3: 1.000
- Mean Reciprocal Rank: 1.000

These preliminary results are limited by the small corpus and evaluation set. The evaluation script writes `tests/evaluation/retrieval_results.json`, including a generation timestamp.

## Relevant Tests

- `tests/test_research_agent.py`
- `tests/unit/test_embeddings.py`
- `tests/unit/test_hybrid_retriever.py`
- `tests/unit/test_research_agent.py`
- `tests/test_integration_day1_day2.py`
- `tests/test_pipeline.py`
- `tests/test_m4_day6_9.py`
- `tests/unit/test_day1_day2_classifier.py`
- `tests/unit/test_day3_baseline_training.py`

The latest broad regression run passed 61 tests with nine scikit-learn version warnings. Saved model artifacts were created under a different scikit-learn version than the current runtime; a consistent dependency and artifact workflow is still needed.

## Remaining Work

1. Expand labelled retrieval evaluation with relevant and irrelevant queries across supported languages.
2. Calibrate the abstention threshold using validation data.
3. Verify retrieval against a larger, authoritative regulatory corpus when ingestion is ready.
4. Review evidence handoff contracts with the M4 owners.
5. Keep this README aligned with code and rerun regression tests before committing.
