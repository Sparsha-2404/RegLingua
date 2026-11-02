# Demo talk track and Q&A (M2)

Decision support / research software, not legal certification. Every number is read from the file named in brackets by `scripts\build_talk_track_m2.py` (bash: `python3 scripts/build_talk_track_m2.py`). Question 4 depends on the scenario results (PENDING until the local run); question 9 on a clean-machine check (not yet done).

## 1. M2 script (60-90 seconds; 180 words; time it in the table below, do not rely on the word count)

> Hi, I'm M2. My part is the risk classifier and its evaluation. We use a deliberately simple baseline, TF-IDF with logistic regression, so everything is inspectable and runs offline. It reads a statement in English, Hindi or code-mixed text and labels it low, medium or high risk, and that label starts the pipeline's routing. I want to be upfront about what it is. It is decision support, not legal advice and not a compliance certificate. On a frozen test set it is right about 62% of the time, well above majority-class guessing at 42%, but it still calls some risky statements low, so a low label is never treated as approval. Hindi is weaker than English, and it misses some banned phrases, which is why an independent verifier and human review sit behind it. We protect the evaluation by splitting on the original statement, so translations never leak between training and test, and we report confidence intervals. Now let's see it on a live example. If the live run fails, I will switch to the recorded output and say so.

## 2. Rehearsal timings (fill in)

| Rehearsal | Date | Start | End | Total (s) | Within 90 s? | Stumbles / notes |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

## 3. Say / do not say

**Say**

- 'decision support' / 'triage', and 'a baseline model'
- the confidence intervals, not only point estimates
- 'LOW means review lightly, not approved'
- 'recorded output, not live' whenever the fallback is used
- 'chosen demonstration; the typical rate is on the slide' for every scenario
- 'we have not verified this on a clean machine' until the sign-off says PASS

**Do not say**

- 'compliant', 'approved', 'certified' or 'legal advice'
- 'accurate', 'reliable' or 'AI verifies compliance' without the interval and the caveat
- 'understands Hindi' (it is weaker than English and the Hindi coverage is thin)
- 'the embedding model is deployed' (it is not evaluated on the frozen split)
- any number that is not in the files cited here; if asked, say 'I will check the file'
- 'typical' about a chosen scenario
- 'fresh checkout tested' or 'all tests pass' before the sign-off records a real local run

## 4. The 15 hardest examiner questions

**Q1. Why TF-IDF and logistic regression?**

It was the planned deterministic baseline: simple, inspectable, trains in seconds and runs offline [PLAN: architecture PDF section 8]. On the frozen final test it scores accuracy 0.621 [0.524, 0.717] against 0.417 for majority class and 0.427 for a banned-phrase rule [final_eval_results_m2.csv]. We claim it is a baseline, not a finished model.

**Q2. A frozen embedding model beat it on Day 10. Why not use that one?**

On 104 held-out rows (leak-free refit) accuracy rose from 0.529 to 0.769 with a CI that excludes 0 [day10_embedding_vs_baseline_m2.csv]. But that is a different split from the frozen Day 12 test, it was never run on it, and it needs a cached download; on 6 code-mixed rows it was inconclusive. So the demo uses the TF-IDF model we can evaluate end to end.

**Q3. Why is the rewrite risk reduction so low?**

In the rewrite evaluation the text changed in only 15 of 305 rows (290 unchanged) and the classifier label moved down in 2 [rewrite_eval_results_m2.csv]. Where a banned phrase applied, it was removed in 15 of 15 rows. A lexical classifier mostly reacts to swapped phrases, so a low reduction rate says rewrites rarely fire and the label rarely moves; it does not prove the rewrites are good, and the verifier, not the classifier, decides PASS.

**Q4. Why do safe statements still gather evidence?**

We do not yet have a recorded number: the safe scenario's evidence count is in demo_scenarios_m2.csv, which is PENDING the local run. A likely mechanism: the default compliant floor is 0.70 [abstention.py] and no out-of-fold probability reaches 0.70 [classification_error_analysis_m2.txt], so a LOW prediction is routed to review rather than cleared. Say it plainly: LOW means 'review lightly', not 'approved'.

**Q5. How is Hindi handled?**

By the same model, trained on 240 Hindi rows that are translations of the English ones [audit_eval_data_m2.txt]. It is weaker: final-test accuracy 0.562 [0.417, 0.708] for Hindi against 0.708 [0.583, 0.833] for English, and in the matched out-of-fold comparison the EN minus HI gap is 0.146 [0.089, 0.201] [final_eval_results_m2.csv]. The default tokenizer does not keep 3286 of 3697 Devanagari words whole (89%) [classification_error_analysis_m2.txt]; we think that matters but did not test a fix.

**Q6. What are classifier-blind cases?**

Statements with a banned phrase that the classifier still labels LOW: 7 Day 7 originals out of 305 (95% CI 2-12) [model_usefulness_m2.csv]. On our hand-written probes the documented-convention cases fail 10/11, including banned phrases 5/6 and the architecture PDF's own 'guarantee 20% returns' example 1/1 [adversarial_results_m2.csv]. That is why the banned-phrase check must stay independent of the classifier.

**Q7. How do you protect against data leakage?**

We split by original_id so an English statement and its Hindi translation always land on the same side, merge near-duplicates, and freeze one split with seed 42: 413 train / 103 test rows, 0 original_ids straddling [final_split_m2.meta.json]. The Day 3 row-level split was leaky: 76 of its 104 unseen rows had a sibling in training, and the committed model saw 78 of the 103 final-test rows, so we never score that file [audit_eval_data_m2.txt]. The final test is scored once; tuning used out-of-fold predictions.

**Q8. What is the false-compliant rate and why emphasise it?**

It is the share of truly MEDIUM or HIGH statements predicted LOW, the costly error because a LOW label can let a risky statement through. It is 0.350 [0.210, 0.492] on the final test and 0.308 [0.246, 0.370] out-of-fold; the majority baseline is 1.000 by construction [final_eval_results_m2.csv]. So we say: the model prioritises review, it must never clear content.

**Q9. Does it work offline?**

The TF-IDF model needs no download and the runners set HF_HUB_OFFLINE=1; the classifier selector falls back from the embedding model to TF-IDF to a rule-based mock [model_selection.py]. But we have not verified the full demo on a clean machine: in the review sandbox the readiness check said GATE NOT MET because langgraph and sentence_transformers were missing [demo_readiness_day13_m2.txt]. That is why the recorded fallback exists, and it is labelled RECORDED OUTPUT.

**Q10. Is this legal advice?**

No. It is decision-support and research software; the labels are project-level annotations, not categories issued by any regulator, and a LOW label does not mean a statement is compliant or approved [annotation_policy_m1.md]. A person must review every result.

**Q11. 62% accuracy is low. Why should anyone trust it?**

They should not rely on it alone, and we say so. Accuracy is 0.621 [0.524, 0.717] and macro-F1 0.621 [0.514, 0.712] on 103 rows [final_eval_results_m2.csv]; it is useful as triage and clearly above the sanity baselines, not as a gate. We report the intervals so nobody over-reads the point estimates.

**Q12. Is the confidence score calibrated? What does abstention do?**

No. Out-of-fold, 310 predictions sit below 0.50 with a 41.9% error rate and 97 sit in 0.50-0.60 at 40.2%; none reaches 0.70 [classification_error_analysis_m2.txt]. The Day 9 balanced operating point let 2 false-compliant outcomes through out of 7 LOW accepts and permissive 10 of 17; conservative accepts no LOW [abstention_operating_points_m2.csv], on held-out rows that overlap training by original_id, so treat them as optimistic.

**Q13. Your test set is tiny. Are conclusions reliable?**

Only the large effects are. The final test has 103 rows from 55 independent groups, MEDIUM 17 and code-mixed 7 rows [final_split_m2.meta.json]; most slice comparisons have overlapping CIs. The pooled out-of-fold results use 3 repeats of the same 413 rows, so they are more stable but not independent.

**Q14. Can someone fool the classifier?**

Yes. Typos changed the prediction in 19/204 final-test variants, and appended text changed it in 56/618 variants, but a neutral appended sentence did so in 74/618, so this is general instability, not a proven injection weakness [adversarial_results_m2.csv]. Hand-written banned-phrase probes mostly get through, which is why the policy checks are separate.

**Q15. How reliable are the labels?**

One labelling pass by M1 with no second-annotator agreement measured [day12_risk_classification_eval_m2.md]. We flagged 15 rows where the model strongly disagrees [suspected_label_errors_m2.csv]; on the English and code-mixed ones the given labels looked right and the model was wrong, so we have no confirmed label error. M1 should adjudicate the Hindi rows.

## 5. Known weaknesses we state up front (ordered by severity, M2's judgement)

1. False-compliant errors: the model clears 0.350 [0.210, 0.492] of risky statements as LOW on the final test (0.308 [0.246, 0.370] out-of-fold) [final_eval_results_m2.csv]. It cannot be a gate.
2. Blind spots: banned phrases, disclaimers and negation; documented hand-written probes fail 10/11, including the PDF's own example [adversarial_results_m2.csv].
3. Confidence carries almost no signal and no LOW prediction reaches the 0.70 floor out-of-fold [classification_error_analysis_m2.txt]; abstention thresholds are not risk calibration.
4. Hindi is weaker than English at equal content (EN minus HI 0.146 [0.089, 0.201] out-of-fold); code-mixed is effectively unevaluated (7 final-test rows) [final_eval_results_m2.csv, final_split_m2.meta.json].
5. The rewrite step barely moves the classifier label (2 of 305 reduced); verification relies on the same classifier family plus separate policy checks [rewrite_eval_results_m2.csv].
6. Small data and a single labelling pass: 103 final-test rows, wide CIs, no second annotator [final_split_m2.meta.json].
7. The committed model file is contaminated for the final test; reported numbers describe the re-fitted recipe, not that file [audit_eval_data_m2.txt].
8. Demo readiness is not verified on a clean machine; the committed canonical-run file holds only import errors, and the scenario results are PENDING a local run [demo_readiness_day13_m2.txt, canonical_run_m2.csv].
9. The embedding classifier has not been evaluated on the frozen split [day10_embedding_vs_baseline_m2.csv].
