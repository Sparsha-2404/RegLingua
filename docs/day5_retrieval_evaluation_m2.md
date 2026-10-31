# Day 5 Retrieval Evaluation (M2)

"Evaluate retrieval queries against known relevant sources." Metric correctness is verified by `tests/unit/test_day5_retrieval_evaluation.py` (5/5 passing against synthetic fake retrievers) before any of these numbers were trusted. The hybrid retriever fix is covered by `tests/unit/test_hybrid_retriever_fix.py` (3 tests, fake keyword retriever and fake vector index).

**Setup:** all 510 rows of `data/evaluation/regulator_topic_eval_m2.csv` with a verified `expected_source_id`, run against the real 308-chunk KB (9 distinct `source_id`s). Each query is run with `regulator_candidates=[true_regulator]`, so this measures retrieval *given the correct regulator is already known*, not regulator classification. Evaluation is at **source-document level**, not chunk level.

## Summary: three retrievers compared

| Retriever | Recall@1 | Recall@3 | Recall@5 | Misses at k=5 |
|---|---|---|---|---|
| Keyword only (`DeterministicRetriever`) | 0.5059 | 0.6196 | 0.6725 | 167 / 510 |
| Hybrid, first version | 0.3039 | 0.7059 | 0.7137 | 146 / 510 |
| **Hybrid, final (`CANDIDATE_POOL = 400`)** | **0.6725** | **0.9961** | **1.0000** | **0 / 510** |

Final hybrid Precision@K: 0.6725 (k=1), 0.3320 (k=3), 0.2000 (k=5). With exactly one relevant source per query, Precision@K equals Recall@K divided by K, so it carries no extra information. Recall@1 is the meaningful headline number.

## Final hybrid results by regulator

| Regulator | Recall@1 | Recall@3 | Recall@5 | n |
|---|---|---|---|---|
| DPDP | 0.6802 | 1.0000 | 1.0000 | 172 |
| RBI | 0.6765 | 1.0000 | 1.0000 | 170 |
| SEBI | 0.6607 | 0.9881 | 1.0000 | 168 |

## Final hybrid results by language

| Language | Recall@1 | Recall@3 | Recall@5 | n |
|---|---|---|---|---|
| en | 0.6835 | 1.0000 | 1.0000 | 237 |
| hi | 0.6371 | 0.9916 | 1.0000 | 237 |
| hi-en | 0.8333 | 1.0000 | 1.0000 | 36 |

Hindi Recall@5 went from 0.346 (keyword only) and 0.384 (first hybrid) to 1.000 (final hybrid). The multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) works; the first hybrid version was preventing it from running on Hindi queries. The `hi-en` Recall@1 of 0.8333 is on only 36 queries, so it should not be read as a real advantage over the other groups.

## Findings: bugs in the first hybrid version

The first hybrid evaluation was worse than keyword-only at rank 1 and returned **empty results** for Hindi queries (the failure listing showed blank `retrieved_source_ids`). Tracing `HybridRetriever.retrieve()` found four problems:

1. **Early return skipped the vector search (main cause of the Hindi failure).** The code had `if not keyword_results: return []` before the vector search. Hindi queries share no tokens with the English-only KB, so keyword search returned nothing and the multilingual embedding model was never called. Fix: always run the vector search; keyword results may legitimately be empty.
2. **Chunk scores overwritten instead of maximised.** Scores were stored as `{source_id: score}` over 308 chunks from 9 sources, so the last chunk seen for a source overwrote better ones. Fix: collapse to one entry per source keeping the best-ranked chunk.
3. **Incomparable score scales.** The 50/50 blend added a keyword overlap fraction to a cosine similarity. Fix: Reciprocal Rank Fusion (k=60), which depends only on ranks.
4. **Small sources crowded out of the candidate pool.** With `CANDIDATE_POOL = 50`, large sources (about 70 chunks each) could fill all 50 slots and a small source never reached fusion. This caused the single remaining miss, `EVALQ0076` (Hindi, "profit is guaranteed from our investment recommendations", expected `SEBI_SPOT_SCAM`). Fix: `CANDIDATE_POOL = 400`, larger than the 308-chunk KB. Re-run result: 0 / 510 misses at k=5. The label was correct; this was a retrieval bug.

An earlier version of this document predicted the original merge logic already mitigated crowding-out. That prediction was wrong: the early return and the overwrite bug dominated, and the pool size mattered as well.

## Keyword-only baseline (for reference)

| Language | Recall@1 | Recall@3 | Recall@5 | n |
|---|---|---|---|---|
| en | 0.6878 | 0.8861 | 0.9536 | 237 |
| hi | 0.2743 | 0.3038 | 0.3460 | 237 |
| hi-en | 0.8333 | 0.9444 | 0.9722 | 36 |

0 of 308 KB chunks contain Devanagari text (the sources are official English documents), so a keyword-overlap retriever cannot match Hindi queries. That is the expected limit of the baseline, not a tokenizer bug.

## Limitations

- **Oracle regulator filter.** Each query is restricted to the true regulator, leaving only 2 to 4 candidate sources. Recall@3 and Recall@5 are therefore close to ceiling by construction. Recall@1 is the informative number. End-to-end performance with a *predicted* regulator has not been measured.
- **Source-level, not chunk-level.** A hit means the right document was retrieved, not that the right passage was. The EvidenceAgent needs the right passage, so chunk-level evaluation is still open.
- **Small KB (9 sources).** Results may not transfer to a larger corpus. `CANDIDATE_POOL = 400` is sized to the current 308-chunk KB and should be revisited if the KB grows substantially.
- **Recall@1 headroom.** About one third of queries (33%) still rank a wrong source first. Likely next steps: a learned reranker and chunk-level evidence selection.

## Reproduce

```
python scripts\evaluate_retrieval_recall_precision.py            # keyword baseline
python scripts\evaluate_retrieval_recall_precision.py --hybrid   # final hybrid
python scripts\analyze_retrieval_errors.py
python -m pytest tests/unit -v
```

Result files: `data/evaluation/retrieval_eval_results_m2_keyword.csv`, `data/evaluation/retrieval_eval_results_m2_hybrid.csv`.
