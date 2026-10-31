# Day 6 (M2): Risk / Evidence Consistency

Plan line for M2: "Validate risk/evidence consistency." Team gate: "System produces an explanation tied to explicit evidence references."

**Bottom line.** M2's validation is complete and every number below comes from real runs on the 516-row evaluation set. The team gate is **not met by the pipeline as it runs today** (251 of 516 statements, including 185 of 240 Hindi, get no explanation) because `agent_factory.py` injects the keyword retriever. Running the same production pipeline with the fixed `HybridRetriever` swapped in (in-process, no files edited) takes that to **2 of 516** (the two abstentions). The remaining weakness is the risk classifier, not retrieval.

## 1. Data and setup

- `data/evaluation/risk_evidence_eval_m2.csv`: Day 4 eval set (516 rows: 240 en, 240 hi, 36 hi-en) joined to M1's risk labels on **normalised text + language** (IDs repeat across language variants) and to M1's evidence QA on `original_id`. 516 in, 516 out, 0 unmatched, 0 duplicate or conflicting keys. 6 rows (the 3 statements M1 marked REVIEW, in two languages each) have no `expected_source_id`: kept and flagged, excluded from retrieval metrics (510 eligible).
- Models: `risk_classifier_tfidf_logreg.pkl` (committed) and `regulator_classifier.pkl` (trained locally; held-out accuracy 0.942 on 104 rows). Both use the same split (stratified on `risk_level`, test 0.2, seed 42): **104 held-out rows**.
- Everything is measured through the real `RiskAgentAdapter`, `agents.supervisor.route_after_classification`, `EvidenceAgentAdapter` and `run_case(use_trained_artifacts=True)`. Routing and rank are recomputed and cross-checked against the stored Prompt 2 results: 0 route mismatches, 0 rank mismatches.

## 2. Retrieval under real pipeline conditions (predicted regulator, risk-phrase query)

Day 5 measured retrieval with the true regulator and the full statement (hybrid: Recall@1 0.6725, Recall@5 1.0000). The Supervisor does neither: it filters by the **predicted** regulator and queries with `risk_phrase or input_text`, where `risk_phrase` is the classifier's top TF-IDF term.

| Config (510 rows) | Retriever | Query | R@1 | R@3 | R@5 | MRR | No evidence |
|---|---|---|---|---|---|---|---|
| A (**today**) | keyword | risk phrase | 0.3784 | 0.4392 | 0.4745 | 0.4129 | 47.84% |
| B | hybrid | risk phrase | 0.6431 | 0.9725 | 0.9922 | 0.8085 | 0% |
| C | hybrid | full statement | 0.6627 | 0.9882 | 0.9922 | 0.8245 | 0% |
| D | keyword | full statement | 0.4980 | 0.6137 | 0.6667 | 0.5613 | 28.82% |

Hindi Recall@5: A 0.2068, D 0.3376, B 0.9873, C 0.9873. Hindi no-evidence rate: A 76.8%, D 62.0%, B/C 0%.

- The **retriever** is the dominant factor (A to B: +0.52 Recall@5). The **query** matters little once the retriever is hybrid (B to C: +0.02 at rank 1, 0 at rank 5).
- **Regulator prediction** costs under one point: accuracy 98.6% (all rows) / 93.3% (held-out). 4 rows had a wrong specific regulator, all true RBI (expected `RBI_DL_DIRECTIONS_2025`), predicted DPDP (3) or SEBI (1); the true regulator was second in all four with margins of 0.05 to 0.18. The 0.4 fallback to GENERAL does not catch them (wrong-class probability 0.42 to 0.51).
- Supervisor routes (516): `needs_review` 514, `abstain` 2, `compliant` **0**.

## 3. Risk / evidence consistency (config C evidence, top-3 = supported)

| Category | All 516 | Held-out 104 |
|---|---|---|
| consistent (flagged, truly risky, verified source in top-3) | 269 | 41 |
| flagged-wrong-evidence | 5 | 3 |
| unverifiable (no M1 source) | 6 | 0 |
| false-alarm (compliant statement flagged) | 23 | 20 |
| missed-risk-review (risky predicted Compliant, still reviewed) | 14 | 14 |
| missed-risk (risky auto-resolved) | 0 | 0 |
| ambiguous-predicted-compliant | 1 | 1 |
| correct-clear | 196 | 23 |
| abstained | 2 | 2 |

Verification-policy checks (document-level support; see limitations):

| Check | All | Held-out |
|---|---|---|
| VP01 evidence present for flagged cases | 100% (303) | 100% (64) |
| VP02 verified source in top-3 / at rank 1 | 98.2% / 64.6% | 93.2% / 65.9% |
| VP04 predicted risk agrees with M1 | 91.6% | **61.8%** |
| VP04 under-flagged / over-flagged | 3.3% / 5.1% | 16.7% / 21.6% |
| True-HIGH predicted Compliant | 14 / 213 (6.6%) | **14 / 43 (32.6%)** |
| True-HIGH auto-resolved with no review | 0 / 213 | 0 / 43 |

All 14 missed risks are held-out rows: the 91.6% overall agreement is inflated by training rows. Held-out figures are still probably optimistic (see limitations).

**Confidence is weakly informative.** Almost all predictions fall between 0.35 and 0.70; no Compliant prediction reaches the 0.7 fast-path bar. Held-out agreement is 55.2% for confidence 0.35 to 0.50 and 72.1% for 0.50 to 0.70. **Lowering the 0.7 "compliant" threshold is unsafe**: on held-out rows, a threshold of 0.40 would clear 34 statements, 13 of them truly risky (61.8% precision); 0.50 would clear 13, of which 7 are truly risky.

## 4. End-to-end gate: explanation tied to evidence

The production pipeline, statement by statement, judged against M1's verified source.

| Outcome (516) | Keyword (today) | Hybrid (factory override) |
|---|---|---|
| grounded, cites the verified source | 78 | **168** |
| grounded, cites a different source | 32 | 106 |
| false-alarm grounded (compliant statement explained as a conflict) | 11 | 23 |
| unverifiable | 1 | 6 |
| compliant explanation | 143 | 211 |
| **no explanation** | **251** | **2** |

- **Root cause confirmed.** The only change between the two columns is the retriever. Keyword: all 251 no-explanation rows had zero evidence, 0 pipeline errors, all routed `needs_review`; 185 were Hindi (77% of Hindi). Hybrid: no-explanation falls to the 2 abstentions, Hindi to 0.
- Hybrid, by language: en 240 / hi 240 / hi-en 36. Citation accuracy (verified source cited, over the 274 explanations that can be checked): **61.3% overall (en 60.9%, hi 62.2%, hi-en 59.3%)**. The language gap is gone. The keyword figure (78 of 110, 70.9%) is not comparable: it covers only rows where keyword search found anything, mostly English.
- Every flagged explanation cites a clause that is in its evidence and names it in the text (100%).
- **100% of flagged explanations quote a fragment** (the classifier's top term) as "the statement", never the statement itself.
- The Explainer cites the top-ranked result. The verified source is first only about 61 to 65% of the time but is in the top 3 for 97 to 99%.
- 14 truly risky statements (all held-out) receive a **"compliant" explanation**; 23 compliant statements receive a grounded explanation of a conflict.

### Cross-checks (two independent tools agree exactly)

| Gate (hybrid) | Consistency analysis |
|---|---|
| flagged explanations 168 + 106 + 23 + 6 = 303 | VP01 pass = 303 |
| judged for citation 168 + 106 = 274 | consistent 269 + wrong-evidence 5 = 274 |
| compliant explanation 211 | correct-clear 196 + missed-risk-review 14 + ambiguous-predicted-compliant 1 = 211 |
| no explanation 2 | abstained 2 |
| false-alarm grounded 23 | false-alarm 23 |
| unverifiable 6 | unverifiable 6 |

The keyword gate run also matches Prompt 2 config A: 251 statements with no evidence against about 244 predicted from 47.84% of 510 (the 6 rows without an expected source were not in that 510).

## 5. Findings and recommended actions

**M3 / M4 (pipeline)**
1. **Inject the fixed `HybridRetriever` in `agents/agent_factory.py`** (it currently injects `DeterministicRetriever`). Measured effect: no-explanation 251 to 2, Hindi 185 to 0, grounded-correct 78 to 168. `EvidenceAgentAdapter` also hardcodes `retrieval_method="keyword"`. Note the fixed retriever's score is a normalised rank-fusion value, not a similarity.
2. **Quote the statement, not the classifier's top term, in the explanation** (100% of flagged explanations currently quote a fragment). The full-statement query is also slightly better for retrieval (R@1 0.643 to 0.663).
3. **Citation:** the Explainer cites the top result. Verified-source citation is about 61% (a conservative figure: M1 labels a single source per statement, and other sources in the same regulator may also support it). Consider citing the top-3 or reranking.
4. From code reading, not measured here: `EvidenceItem` carries no `section` or `source_url`, which VP01/VP03 in M1's verification policy require.
5. Optional: pass the top-2 regulators when the margin is small (recovers all 4 regulator misses; the retriever already accepts a list).
6. `reglingua_supervisor/tests/test_pipeline.py::test_compliant_text_short_circuits` fails about 28% of runs (unseeded `uniform(0.6, 0.95)` against a 0.7 cutoff).

**M1 (data)**
7. DPDP001 to DPDP004 (8 rows) cite `DPDP_RULES_2025` in the example sheets but `DPDP_ACT_2023` in the evidence QA sheet. The eval follows the evidence QA. Please confirm which is authoritative.
8. `data/raw/labeled_examples_m1.csv` was the short 3-column version while the model card says it keeps full metadata; regenerate with `scripts/extract_m1_labeled_csv.py` (it expects `RegLingua_M1_FINAL_v2.xlsx` at the repo root).

**M2 (classifier)**
9. Risk classifier is the weak link: held-out agreement 61.8%, 14 of 43 risky statements predicted Compliant. Next: a grouped split by `original_id` for honest numbers, then the Day 9 abstention logic and the Day 10 multilingual comparison. The committed risk model was saved under scikit-learn 1.8.0 and loads under 1.9.1 with warnings; retrain or pin.

## 6. Limitations

- **Train/test leakage.** Translations of one statement share an `original_id` and can sit on opposite sides of the row-level split, so held-out numbers are still optimistic. Held-out n = 104 is small.
- **Document-level support.** "Expected source in top-3" shows the right document was retrieved, not that the retrieved passage supports the claim. Passage-level support and explanation correctness need passage text and human review.
- **Single expected source.** Citation accuracy treats M1's one source as the only valid one.
- The labelled language is passed to retrieval (the Supervisor's language detection is skipped because no retriever uses it).
- The hybrid gate run swaps the retriever in-process; the factory file was not changed. The keyword gate figures come from the earlier full run of the same script.
- A "wrong regulator" or "wrong source" label is relative to M1's labels, not legal ground truth. This is decision-support evaluation, not legal certification.

## 7. Reproduce

```
python scripts\extract_m1_labeled_csv.py
python scripts\train_regulator_classifier.py            # set PYTHONPATH=%CD% on Windows
python scripts\build_risk_evidence_eval.py --strict
python scripts\evaluate_pipeline_retrieval.py           # config A, B, C, D
python scripts\analyze_risk_evidence_consistency.py
python scripts\check_day6_gate.py                       # pipeline as it is today
python scripts\check_day6_gate.py --retriever hybrid    # pipeline with the fixed HybridRetriever
python -m pytest tests/unit -v
```

Outputs: `data/evaluation/risk_evidence_eval_m2.csv`, `pipeline_retrieval_results_m2.csv`, `day6_consistency/`, `day6_gate_check_m2.csv`, `day6_gate_check_m2_hybrid.csv`.
