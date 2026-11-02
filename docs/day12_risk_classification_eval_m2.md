# Day 12 (M2): Risk Classification Metrics and Error Analysis

**Bottom line.** The TF-IDF + Logistic Regression baseline is clearly better than the sanity baselines but is **not a reliable
risk gate**. On the frozen final test (103 rows, 55 original_id groups, scored once) it reaches
accuracy 0.621 [0.524, 0.717] and macro-F1 0.621 [0.514, 0.712]; majority class is 0.417
accuracy and the banned-phrase rule 0.427. The dangerous error, a MEDIUM/HIGH statement predicted LOW
(**false-compliant**), is 0.350 [0.210, 0.492] on the final test and 0.308 [0.246, 0.370] out-of-fold. Hindi is weaker than
English in the matched out-of-fold comparison (OOF accuracy gap EN minus HI 0.146 [0.089, 0.201]); on the small final test the gap has the same sign but its CI includes 0. Predicted
probabilities carry almost no confidence signal (see section 5), and on the hand-written probes it fails 5 of 6 banned-phrase cases, all 3 disclaimer-wrapped cases and the plain negation case
(section 6; tiny n). Treat it as a baseline for the Day 10/11 comparison, not as a decision-maker; keep the verifier's banned-phrase check independent of it.

Final test CIs are wide (n = 103); read differences between slices as directional unless the CI excludes 0.

## 1. Protocol and leakage statement

- **Data.** `risk_evidence_eval_m2.csv`, 516 rows / 276 original_ids. The "final labelled evaluation set" from M1 is
  `RegLingua_M1_FINAL.xlsx`; the data audit (`data/evaluation/audit_eval_data_m2.txt`) found its 516 rows are exact matches of the CSV, so
  **no separate M1 final set exists beyond it** and the 516-row file is used. No label was changed.
- **Grouping.** Leak groups = original_id (EN/HI translations share one) plus 3 near-duplicate links (merged). 274 groups.
- **Frozen split.** `scripts/make_final_split.py`, seed 42, best of 2000 GroupShuffleSplit candidates (#771) chosen only by
  label/language/regulator balance, never by model results. Train pool 413 rows, final test 103 rows. Verified: original_ids straddling = 0,
  leak groups straddling = 0. Split SHA-256 `379491205811b3b0...`; data file hashes are in `results/MANIFEST_m2.json`.
- **Use-once.** Models are fit on the train pool only and `final_test` is scored once per run behind an `OnceGuard`; reported final numbers come from that single run.
  A second run in a scratch copy (verification only) reproduced all three output files byte-for-byte (SHA-256 equal to the manifest).
- **Tuning and error analysis** use out-of-fold (OOF) predictions on the train pool only: repeated StratifiedGroupKFold, 5 folds x 3 repeats
  (`random_state = 42 + repeat`), models re-fitted per fold. The committed `models/risk_classifier_tfidf_logreg.pkl` is **never scored here**: the audit shows it was trained on 412 of
  the 516 rows (78 of the final-test rows), so scoring it would be contaminated.
- **Intervals.** Percentile 95% CIs from 2000 bootstrap resamples of whole leak groups, identical resamples across models for paired differences. Seed 42 throughout.
- **Disclosed protocol caveats.** (1) 25 of the 103 final-test rows were in the Day 5-10 `held_out` set that earlier analysis looked at. (2) The adversarial suite
  was run on final-test *texts* (labels not read) and six banned-phrase probes (B01-B06) were added after its first run. (3) The split files are untracked in the archive I reviewed, so git cannot prove the
  split was frozen before analysis; commit `final_split_m2.csv` and its meta now. (4) The embedding classifier was not run (encoder not cached; section 8).

## 2. Overall results

Final test (103 rows, scored once):

| model | accuracy | macro-F1 | false-compliant rate |
|---|---:|---:|---:|
| majority | 0.417 [0.283, 0.549] | 0.196 [0.147, 0.236] | 1.000 [1.000, 1.000] |
| banned-phrase rule | 0.427 [0.294, 0.557] | 0.213 [0.158, 0.268] | 0.983 [0.944, 1.000] |
| TF-IDF+LogReg | 0.621 [0.524, 0.717] | 0.621 [0.514, 0.712] | 0.350 [0.210, 0.492] |

Train-pool OOF (413 rows x 3 repeats pooled = 1239 predictions):

| model | accuracy | macro-F1 | false-compliant rate |
|---|---:|---:|---:|
| majority | 0.431 [0.366, 0.498] | 0.201 [0.179, 0.222] | 1.000 [1.000, 1.000] |
| banned-phrase rule | 0.433 [0.369, 0.500] | 0.212 [0.187, 0.237] | 0.983 [0.965, 0.996] |
| TF-IDF+LogReg | 0.602 [0.552, 0.650] | 0.632 [0.579, 0.679] | 0.308 [0.246, 0.370] |

False-compliant rate = share of truly MEDIUM/HIGH rows predicted LOW. The majority baseline is 1.000 by construction. The banned-phrase rule uses the row's true regulator (oracle input) and English-only phrase lists.

Paired differences (same resamples), final test:

| TF-IDF minus banned-phrase rule | difference [95% CI] | bootstrap p |
|---|---:|---:|
| accuracy | 0.194 [0.040, 0.346] | 0.0130 |
| macro_f1 | 0.408 [0.289, 0.508] | 0.0000 |
| false_compliant_rate | -0.633 [-0.778, -0.483] | 0.0000 |

| TF-IDF minus majority | difference [95% CI] | bootstrap p |
|---|---:|---:|
| accuracy | 0.204 [0.050, 0.353] | 0.0090 |
| macro_f1 | 0.425 [0.309, 0.524] | 0.0000 |
| false_compliant_rate | -0.650 [-0.790, -0.508] | 0.0000 |

Out-of-fold (more data, same direction): accuracy 0.169 [0.081, 0.257] and macro-F1 0.420 [0.358, 0.474] versus the banned-phrase rule.

## 3. Per-class, language and regulator (TF-IDF+LogReg, final test)

| class | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| LOW | 43 | 0.588 [0.404, 0.759] | 0.698 [0.548, 0.840] | 0.638 [0.488, 0.756] |
| MEDIUM | 17 | 0.667 [0.400, 0.900] | 0.588 [0.393, 0.812] | 0.625 [0.428, 0.778] |
| HIGH | 43 | 0.649 [0.441, 0.829] | 0.558 [0.373, 0.730] | 0.600 [0.418, 0.737] |

Confusion matrix, final test (rows true, columns predicted):

|  | pred LOW | pred MEDIUM | pred HIGH |
|---|---:|---:|---:|
| true LOW | 30 | 2 | 11 |
| true MEDIUM | 5 | 10 | 2 |
| true HIGH | 16 | 3 | 24 |

By language:

| language | n | TF-IDF acc | TF-IDF macro-F1 | TF-IDF false-compliant | majority acc |
|---|---:|---:|---:|---:|---:|
| en | 48 | 0.708 [0.583, 0.833] | 0.720 [0.556, 0.848] | 0.333 [0.150, 0.519] | 0.438 |
| hi | 48 | 0.562 [0.417, 0.708] | 0.532 [0.337, 0.703] | 0.444 [0.250, 0.643] | 0.438 |
| hi-en | 7 | 0.429 [0.143, 0.857] | 0.200 [0.083, 0.462] | 0.000 [0.000, 0.000] | 0.143 |

By regulator:

| regulator | n | TF-IDF acc | TF-IDF macro-F1 | TF-IDF false-compliant | majority acc |
|---|---:|---:|---:|---:|---:|
| SEBI | 32 | 0.594 [0.419, 0.765] | 0.542 [0.323, 0.684] | 0.375 [0.077, 0.684] | 0.500 |
| RBI | 37 | 0.649 [0.459, 0.811] | 0.642 [0.404, 0.801] | 0.318 [0.087, 0.588] | 0.405 |
| DPDP | 34 | 0.618 [0.441, 0.778] | 0.655 [0.485, 0.798] | 0.364 [0.176, 0.545] | 0.353 |

hi-en has n = 7 (7 groups): its numbers are anecdotal. Regulator differences have heavily overlapping CIs; no regulator gap is established.

Same slices on the train-pool OOF (larger n):

| language | n | TF-IDF acc | TF-IDF macro-F1 | TF-IDF false-compliant | majority acc |
|---|---:|---:|---:|---:|---:|
| en | 576 | 0.681 [0.623, 0.735] | 0.714 [0.651, 0.766] | 0.237 [0.170, 0.310] | 0.443 |
| hi | 576 | 0.535 [0.472, 0.597] | 0.572 [0.492, 0.639] | 0.414 [0.339, 0.498] | 0.443 |
| hi-en | 87 | 0.529 [0.368, 0.690] | 0.460 [0.291, 0.609] | 0.127 [0.015, 0.267] | 0.276 |

## 4. English vs Hindi matched comparison

Matched = the same original_id in EN and HI (translation pairs), so content and label are identical and only the language differs.

Final test:

| metric | EN | HI | EN minus HI |
|---|---:|---:|---:|
| accuracy | 0.708 [0.583, 0.833] | 0.562 [0.417, 0.708] | 0.146 [-0.042, 0.312] |
| macro_f1 | 0.720 [0.556, 0.848] | 0.532 [0.337, 0.703] | 0.188 [-0.069, 0.434] |

discordant pairs: en right/hi wrong=13, en wrong/hi right=6; exact two-sided binomial (exact McNemar p = 0.1671; 48 pairs). Same sign as OOF but the CI includes 0: **not established on the final test alone**.

Train-pool OOF (pairs pooled over 3 repeats; CIs resample leak groups):

| metric | EN | HI | EN minus HI |
|---|---:|---:|---:|
| accuracy | 0.681 [0.622, 0.736] | 0.535 [0.470, 0.595] | 0.146 [0.089, 0.201] |
| macro_f1 | 0.714 [0.654, 0.768] | 0.572 [0.495, 0.639] | 0.141 [0.075, 0.217] |

discordant pairs: en right/hi wrong=36, en wrong/hi right=12; exact two-sided binomial (exact McNemar p = 0.0007; 192 pairs). Here the gap is established: **Hindi is worse than English at the same content** (the `verdict`/`note` text in the CSV is boilerplate that says "CI excludes 0" even when it does not; read the CI). Likely cause: default `TfidfVectorizer` tokenization splits Devanagari words at vowel signs (section 5, item 5); not tested here as a fix.

## 5. Error analysis (train-pool OOF only; single seed-42 5-fold run, `classification_error_analysis_m2.txt`)

That run gives accuracy 0.588 [0.528, 0.643], macro-F1 0.617 [0.556, 0.670], false-compliant 0.328 [0.258, 0.402] (a different resampling from the 3-repeat pooled numbers in section 2). 170/413 rows are errors.

1. **Two dominant errors, equal size.** LOW->HIGH 71 (42% of errors; 40% of true LOW) and HIGH->LOW 65 (38%; 38% of true HIGH). MEDIUM->LOW 12, MEDIUM->HIGH 10, HIGH->MEDIUM 7, LOW->MEDIUM 5.
   False-compliant 77/235 risky rows (32.8%); false-risky 76/178 LOW rows (42.7%). False-compliant is the worse error here: the verifier passes a rewrite when the classifier says LOW.
2. **Negation / polarity.** Compliant statements containing "not/do not/never" are predicted HIGH (e.g. "We do not guarantee returns..." is labeled LOW, predicted HIGH). The negation cue is present in 21% of LOW->HIGH errors versus 7% of correctly classified LOW rows.
   Minimal pairs differing only by "not" exist in the data with opposite labels (section 9).
3. **Confidence is uninformative.** 407 of 413 OOF predictions have max probability below 0.6; none reach 0.7. Error rate is 41.9% in the <0.50 band and 40.2% in 0.50-0.60 (CIs overlap). Confidence thresholds cannot separate good from bad predictions for this model.
4. **Slices (OOF error rate).** en 34.9% [28.4, 41.9], hi 46.4% [39.3, 53.4], hi-en 48.3% [31.0, 65.5]; SEBI 44.3%, DPDP 41.3%, RBI 37.8% (CIs overlap). MEDIUM is the best-classified class by error rate (33.8%) but has the fewest rows (65).
5. **Tokenization.** 3,286 of 3,697 Devanagari words (89%) are not kept as a single token by the default pattern (vowel signs are not `\w`), producing fragments such as the top-weighted error terms. Pronoun-like fragments (e.g. the token for "we") dominate LOW->HIGH error terms: lexical shortcuts rather than regulatory content.

## 6. Adversarial results (TF-IDF fitted on the train pool; `adversarial_results_m2.csv`)

Fail = the model's behavior violates the stated expectation (invariance: prediction changed; directional: moved the wrong way). Invariance/directional checks compare the model with itself, not with gold labels. CIs resample clusters.

| check | final-test population | result [95% CI] |
|---|---|---|
| typo invariance | all final test | 19/204 = 0.093 [0.038, 0.157] |
| typo invariance | HI only | 5/39 = 0.128 [0.000, 0.308] |
| punctuation / emoji invariance | all final test | 0/303 = 0.000 [0.000, 0.000]; 0/309 = 0.000 [0.000, 0.000] |
| append "guarantee" must not lower risk | all final test | 3/103 = 0.029 [0.000, 0.066] |
| appended injection text keeps prediction | all final test | 56/618 = 0.091 [0.049, 0.139] |
| **control**: appended neutral text keeps prediction | all final test | 74/618 = 0.120 [0.077, 0.165] |

The injection check does not fail more often than its neutral-text control (CIs overlap), so the sensitivity to appended text is generic instability, not demonstrated injection susceptibility. Casing and whitespace perturbations are mostly trivial after preprocessing (counts in the CSV).

Hand-written probes (33 cases, small n; expectation kinds from `adversarial_classifier_tests_m2.py`):

| expectation kind | fail |
|---|---|
| strict (unambiguous) | 0/10 |
| documented (project convention, e.g. banned phrase must not be LOW) | 10/11 |
| judgement (reasonable people could differ; reported separately) | 7/12 |

Documented failures include the architecture PDF's own example ("We guarantee 20% returns...", expected HIGH, anchor probe 1/1), banned phrases 5/6, disclaimer-wrapped banned phrases 3/3 and the plain English negation probe 1/1. Latin-script Hindi probes: 4/4 (judgement).

## 7. Limitations

- **Small n.** 103 final-test rows (MEDIUM 17, hi-en 7); only 55 independent groups. CIs are wide; slice comparisons are mostly inconclusive.
- **Label noise / single labelling pass.** Single labelling pass, no second-annotator agreement measured; true label noise is unknown. The 15 rows in `suspected_label_errors_m2.csv` are model-disagreement candidates, not confirmed errors (section 9).
- **Frozen splits.** One split drawn once; the single final-test number depends on that draw. OOF CIs reflect resampling, not alternative splits.
- **One classifier family.** Only TF-IDF+LogReg (plus trivial baselines) is evaluated here; no hyperparameter search was run. The embedding classifier is not in these numbers.
- **Near-parallel data.** Many rows are translations or minimal negation pairs; results may not transfer to unconstrained real-world copy. hi-en rows are singleton groups, so semantic siblings are not ruled out.
- **Baselines are weak by design.** The banned-phrase rule is English-only and oracle-regulator; beating it is a low bar.
- **Adversarial** variants are sampled, and hand-written cases are 33 items chosen by the author.
- **Environment.** The committed manifest records scikit-learn 1.9.1 / Python 3.13.18; the scratch reproduction ran scikit-learn 1.8.0 / Python 3.12.3 and still produced byte-identical result CSVs.

## 8. Reproduction

```bash
python scripts/run_m2_final_evaluation.py --seed 42 && python scripts/build_day12_doc_m2.py
```

The first command reads the frozen split (never re-draws it) and rewrites `final_eval_*_m2.csv` and `results/MANIFEST_m2.json`; the output hashes must equal the manifest. Optional local-only step for the embedding classifier (needs internet once, then offline):

```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"
python scripts/run_m2_final_evaluation.py --seed 42 --embedding on
```

Error analysis and adversarial suite: `python scripts/analyze_classification_errors.py` and `python scripts/adversarial_classifier_tests_m2.py`.
Re-running the final evaluation re-scores the frozen test set; do it only to verify reproducibility, never to choose models or settings.

## 9. Findings for M1 / M3 / M4

**M1 (data).**
- `suspected_label_errors_m2.csv` lists 15 rows where the fold model disagrees with the given label with high probability gap. I read the 6 English/code-mixed rows: for 5 of them the given label looks **correct** and the model is wrong (negated compliant statements labeled LOW; "make it difficult"/"ignore valid requests" labeled HIGH); the code-mixed row EVALQ0148 (HIGH vs predicted MEDIUM) is a judgement call. **No label error is confirmed.** Hindi rows were not adjudicated by me; please review the 9 Hindi rows (all `m1_decision` fields are blank).
- 3 near-duplicate pairs have different labels (EVALQ0373/0374, 0379/0380, 0449/0450). These are "does / does not" minimal pairs, so the differing labels look intentional and correct; keep them in the same leak group (the split does).
- MEDIUM is the thin class (82 rows; 17 in the final test) and 9 of 27 risk x language x regulator cells have fewer than 10 rows. More MEDIUM and hi-en examples would tighten every CI above.
- A second annotator on a sample would turn "label noise unknown" into a measured number.

**M3 (agents/retrieval).**
- Do not use the TF-IDF risk label to decide whether evidence is needed for LOW-confidence cases: no OOF probability reached 0.7, and error rate does not fall with confidence.
- Hindi risk predictions are less reliable than English; Hindi cases deserve retrieval/explanation even when the classifier says LOW.

**M4 (orchestration/verifier/API).**
- The classifier fails every documented banned-phrase and disclaimer probe and the PDF's own HIGH example. Verifier check 1 (pass when the classifier says LOW) must not be the only gate; keep the banned-phrase and policy checks independent of the classifier.
- Day 9 abstention thresholds were selected on earlier OOF probabilities; with max-probability mostly below 0.6 here, check what share of traffic they actually accept before presenting them as risk control.
- Routing on `NEEDS_REVIEW` should be treated as the default for Hindi/hi-en and negation-heavy inputs until a better model exists.
