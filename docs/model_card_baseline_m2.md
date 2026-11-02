# Model Card: RegLingua Risk Classifier (TF-IDF + Logistic Regression), final M2 version

Card date: 2026-11-02 (Day 13). Owner: M2. Source IDs in square brackets (e.g. [S1]) point to the Sources table at the end; every figure is read from the listed file.

> **Decision-support / research software. Not legal advice and not compliance certification.** Labels are project-level annotation labels, not categories issued by any regulator [S13]. A human must review every output before it is relied on.

**Bottom line.** This is a small, simple baseline. On the frozen final test (103 rows, scored once) it reaches accuracy 0.621 [0.524, 0.717] and macro-F1 0.621 [0.514, 0.712], clearly above majority-class (0.417 [0.283, 0.549] accuracy) and a banned-phrase rule (0.427 [0.294, 0.557]) [S1]. It still clears a truly MEDIUM/HIGH statement as LOW 0.350 [0.210, 0.492] of the time on that test and 0.308 [0.246, 0.370] out-of-fold [S1]. Its confidence scores do not separate right from wrong predictions, Hindi is weaker than English, and it fails negation and banned-phrase probes. Use it to prioritise human review, never to clear content.

## Changes since Day 3

- **Day 3 held-out metrics removed.** They came from a stratified row-level 80/20 split that was not grouped by `original_id`; the audit shows 76 of the 104 unseen rows have an EN/HI sibling in training, so those numbers were leakage-inflated [S5]. They are replaced by results on a frozen group split, below [S1, S4].
- **Day 3 per-class story withdrawn.** Day 3 presented MEDIUM as the best-classified class. With group-aware evaluation the per-class CIs overlap and MEDIUM has only 17 test rows (Table 2) [S1].
- **Corrected successor claim.** Day 3 said Day 10 would replace this model with a fine-tuned multilingual transformer. Day 10 actually evaluated a *frozen* multilingual-embedding encoder + LogReg (not fine-tuned) on a different split; it has not been run on the Day 12 frozen split, and the demo path still loads this TF-IDF pickle [S10, S19].
- **Corrected configuration.** The vectorizer also uses `sublinear_tf=True` and the classifier `max_iter=1000`, omitted on Day 3 [S14].
- **Corrected provenance wording.** The extraction script names `RegLingua_M1_FINAL_v2.xlsx`; the workbook in the repo is `data/evaluation/RegLingua_M1_FINAL.xlsx`, and the audit found all 516 of its rows in the evaluation CSV, which is row-for-row identical (text and label) to the training CSV [S5].
- **Added:** false-compliant rate, 95% CIs, sanity baselines, language/regulator slices, calibration and abstention, robustness, fallback behaviour, ethics and failure modes.
- **Important scope note.** The committed pickle was trained on 412 of the 516 rows, including 78 of the 103 final-test rows, so it cannot be scored fairly on the final test [S5, S11]. All Day 12 numbers describe the **training recipe re-fitted on the 413-row train pool**, not the committed file.

## 1. Model details

- **Architecture.** `TfidfLogisticClassifier`: TF-IDF (`max_features=20000`, `ngram_range=(1,2)`, `sublinear_tf=True`) feeding `LogisticRegression(max_iter=1000, class_weight="balanced")`; three classes LOW / MEDIUM / HIGH; input is the classifier-normalised text [S14].
- **Version.** "Baseline v1" (label used in this card; Day 3 recipe, unchanged). Evaluation of the recipe: Day 12.
- **Artifact.** `models/risk_classifier_tfidf_logreg.pkl`, SHA-256 `220c734878c1b083ef6854176b0a93aba52a5a9fc2767507af82649ce2d92d13`, committed 2026-09-24 [S21], pickled with scikit-learn 1.8.0, 412 training rows [S11].
- **Library versions.** Models manifest environment: Python 3.13.15, numpy 2.1.3, scikit-learn 1.9.1, sentence-transformers 6.2.0, langgraph 1.2.18 [S11]. Day 12 evaluation run: Python 3.13.18, numpy 2.5.3, scikit-learn 1.9.1 [S3].
- **Seeds / resampling.** Seed 42; 2000 bootstrap resamples of whole leak groups; 5-fold x 3-repeat StratifiedGroupKFold on the train pool [S3].

## 2. Intended use and users

Analysts, compliance researchers and the RegLingua demo pipeline, to **triage** short fintech statements (marketing/disclosure style) in English, Hindi and Hindi-English code-mixed text into LOW / MEDIUM / HIGH review priority, alongside retrieved regulatory evidence, an explanation, a suggested rewrite and an independent verification step. The label supports prioritisation and discussion; it is one input among several.

## 3. Out-of-scope uses

- **Legal advice or compliance certification of any kind.** A LOW label does not mean a statement is compliant or approved [S13].
- Automatically clearing, publishing or blocking content without human review.
- Regulators other than SEBI / RBI / DPDP, other languages or scripts, long documents, contracts, or non-fintech text; none were evaluated.
- Using the predicted probability as a calibrated measure of risk or certainty (section 7).
- Adversarial or high-stakes settings where a deliberately crafted statement must be caught (section 8).

## 4. Factors

- **Languages.** en, hi, hi-en. Corpus: 240 EN / 240 HI translation pairs share an `original_id` (240 groups of 2), plus 36 hi-en singleton rows [S5].
- **Regulators.** SEBI, RBI, DPDP; the classifier sees text only, regulator is an evaluation factor [S5, S14].
- **Final-test composition:** 43 LOW / 17 MEDIUM / 43 HIGH; en 48, hi 48, hi-en 7; SEBI 32, RBI 37, DPDP 34 [S4].

## 5. Metrics and why they were chosen

- **False-compliant rate (primary safety metric).** Share of truly MEDIUM/HIGH rows predicted LOW. A risky statement cleared as LOW is the costly error, because the verifier can pass a rewrite when the classifier says LOW. Lower is better. The majority baseline scores 1.000 by construction [S1, S6].
- **Macro-F1.** Equal weight per class, so the small MEDIUM class is not hidden by LOW/HIGH [S1].
- **Accuracy.** Reported for comparability with earlier days; the class mix is close to balanced for LOW/HIGH only.
- **Per-class precision/recall/F1, confusion matrix, language and regulator slices.** To locate where errors concentrate [S1, S2].
- **95% CIs.** Percentile bootstrap over whole leak groups; paired differences use identical resamples, so translation pairs never straddle a resample [S3].

## 6. Evaluation data and split protocol

- **Data.** 516 rows, 276 original_id groups, labels balanced as LOW 221, HIGH 213, MEDIUM 82 [S5].
- **Frozen split.** Seed 42; one of 2000 candidate group splits chosen by label/language/regulator balance only; train pool 413 rows / final test 103 rows; original_ids straddling = 0, leak groups straddling = 0 [S4].
- **Use-once.** Models are fit on the train pool only; the final test is scored once per run; tuning and error analysis use out-of-fold predictions on the train pool only [S3, S18].
- **Disclosed caveats.** 25 of the 103 final-test rows were in the Day 5-10 held-out set analysed earlier [S5]; the split files were untracked in the reviewed archive, so git cannot prove the freeze date [S18].

## 7. Quantitative results

Table 1. Overall, with 95% CIs [S1]. False-compliant = truly MEDIUM/HIGH predicted LOW.

**Final test (n = 103, scored once)**

| model | accuracy | macro-F1 | false-compliant |
|---|---:|---:|---:|
| majority class | 0.417 [0.283, 0.549] | 0.196 [0.147, 0.236] | 1.000 [1.000, 1.000] |
| banned-phrase rule | 0.427 [0.294, 0.557] | 0.213 [0.158, 0.268] | 0.983 [0.944, 1.000] |
| TF-IDF + LogReg | 0.621 [0.524, 0.717] | 0.621 [0.514, 0.712] | 0.350 [0.210, 0.492] |

**Train-pool out-of-fold (n = 1239 predictions = 413 rows x 3 repeats)**

| model | accuracy | macro-F1 | false-compliant |
|---|---:|---:|---:|
| majority class | 0.431 [0.366, 0.498] | 0.201 [0.179, 0.222] | 1.000 [1.000, 1.000] |
| banned-phrase rule | 0.433 [0.369, 0.500] | 0.212 [0.187, 0.237] | 0.983 [0.965, 0.996] |
| TF-IDF + LogReg | 0.602 [0.552, 0.650] | 0.632 [0.579, 0.679] | 0.308 [0.246, 0.370] |

The banned-phrase rule uses the true regulator and English-only phrase lists (an oracle-input sanity baseline) [S3]. Paired TF-IDF minus banned-phrase rule on the final test: accuracy 0.194 [0.040, 0.346], macro-F1 0.408 [0.289, 0.508], false-compliant -0.633 [-0.778, -0.483]; versus majority: accuracy 0.204 [0.050, 0.353] [S1].

Table 2. Per class, final test [S1]

| class | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| LOW | 43 | 0.588 [0.404, 0.759] | 0.698 [0.548, 0.840] | 0.638 [0.488, 0.756] |
| MEDIUM | 17 | 0.667 [0.400, 0.900] | 0.588 [0.393, 0.812] | 0.625 [0.428, 0.778] |
| HIGH | 43 | 0.649 [0.441, 0.829] | 0.558 [0.373, 0.730] | 0.600 [0.418, 0.737] |

Table 3. Confusion matrix, final test (rows true, columns predicted) [S2]

|  | pred LOW | pred MEDIUM | pred HIGH |
|---|---:|---:|---:|
| true LOW | 30 | 2 | 11 |
| true MEDIUM | 5 | 10 | 2 |
| true HIGH | 16 | 3 | 24 |

Table 4. By language, final test [S1]

| language | n | accuracy | macro-F1 | false-compliant |
|---|---:|---:|---:|---:|
| en | 48 | 0.708 [0.583, 0.833] | 0.720 [0.556, 0.848] | 0.333 [0.150, 0.519] |
| hi | 48 | 0.562 [0.417, 0.708] | 0.532 [0.337, 0.703] | 0.444 [0.250, 0.643] |
| hi-en | 7 | 0.429 [0.143, 0.857] | 0.200 [0.083, 0.462] | 0.000 [0.000, 0.000] |

Table 5. By regulator, final test [S1]

| regulator | n | accuracy | macro-F1 | false-compliant |
|---|---:|---:|---:|---:|
| SEBI | 32 | 0.594 [0.419, 0.765] | 0.542 [0.323, 0.684] | 0.375 [0.077, 0.684] |
| RBI | 37 | 0.649 [0.459, 0.811] | 0.642 [0.404, 0.801] | 0.318 [0.087, 0.588] |
| DPDP | 34 | 0.618 [0.441, 0.778] | 0.655 [0.485, 0.798] | 0.364 [0.176, 0.545] |

hi-en has n = 7 and is anecdotal; regulator CIs overlap heavily, so no regulator difference is established.

**English vs Hindi, matched translation pairs** [S1]. Final test (48 pairs): accuracy EN 0.708 [0.583, 0.833] vs HI 0.562 [0.417, 0.708], difference 0.146 [-0.042, 0.312] (exact McNemar p = 0.167): not established on this small set. Out-of-fold (576 pooled pair predictions): EN 0.681 [0.622, 0.736] vs HI 0.535 [0.470, 0.595], difference 0.146 [0.089, 0.201]: **Hindi is worse at the same content.**

**Error pattern** (train-pool OOF, single seed-42 run) [S6]: LOW to HIGH 71 errors and HIGH to LOW 65 errors dominate; false-compliant 77/235 = 32.8% of risky rows; false-risky 76/178 = 42.7% of LOW rows. Compliant statements containing a negation cue are over-flagged HIGH (cue in 15/71, 21% of LOW-to-HIGH errors vs 7/102, 7% of correct LOW rows). The default tokenizer does not keep 3286/3697 (89%) of Devanagari words whole, which plausibly weakens Hindi (a hypothesis, not tested).

**Embedding candidate (not the model of record).** Day 10 compared a frozen multilingual encoder + LogReg with the TF-IDF recipe on 104 held-out rows after removing translation leakage from training (336 train rows). This is a different split from Day 12, so it must not be compared with Tables 1-5 or taken as evidence for the demo model [S10]:

| metric | n | TF-IDF | embedding | difference [95% CI] | verdict |
|---|---:|---:|---:|---:|---:|
| accuracy, all | 104 | 0.529 | 0.769 | 0.240 [0.144, 0.340] | improvement |
| macro-F1, all | 104 | 0.551 | 0.760 | 0.209 [0.119, 0.304] | improvement |
| accuracy, en | 43 | 0.558 | 0.767 | 0.209 [0.093, 0.342] | improvement |
| accuracy, hi | 55 | 0.509 | 0.818 | 0.309 [0.167, 0.451] | improvement |
| accuracy, hi-en | 6 | 0.500 | 0.333 | -0.167 [-0.546, 0.000] | no significant difference |

The encoder is a download-dependent optional dependency and is absent from this review environment; it was not run on the frozen split (command in section 12) [S17].

## 8. Calibration and abstention behaviour

- **Confidence is weakly informative.** Out-of-fold, 310 of the predictions have maximum probability below 0.50 (error rate 41.9%) and 97 sit in 0.50-0.60 (error rate 40.2%); 0 predictions fall in the 0.70-0.80 band and 0 at 0.80 or above, so none reaches 0.70 [S6]. Error rates in the two bands are statistically indistinguishable.
- **Calibration.** Day 10 expected calibration error (leak-free refit, n = 104): TF-IDF 0.085 [0.046, 0.189]; embedding 0.058 [0.043, 0.160]; difference embedding minus TF-IDF -0.027 [-0.119, 0.086]: no detectable difference [S9]. Day 9 found no calibration method with a demonstrated improvement over raw probability (see `docs/day9_confidence_abstention_m2.md`).
- **Abstention rule.** ABSTAIN if max probability < `abstain_floor`; REVIEW if any review reason fires (for example a LOW prediction below `compliant_floor`, or the optional margin rule); otherwise ACCEPT, where an accepted LOW routes as compliant and an accepted MEDIUM/HIGH routes to needs_review. Defaults 0.35 / 0.70 mirror the supervisor and are **not tuned** [S16].
- **Consequence.** With the 0.70 default compliant floor and no out-of-fold probability reaching 0.70 [S6], no LOW prediction would be accepted as compliant; LOW predictions go to review.
- **Operating points** (Day 9, selected on non-held-out rows, reported on 104 held-out rows). Nominal only: Day 9 documents that 76 of the 104 held-out rows share an original_id with training rows, so these overstate performance [S8]:

| operating point | abstain / compliant floor | coverage [95% CI] | accuracy on accepted | false-compliant / LOW accepted | review | abstain |
|---|---:|---:|---:|---:|---:|---:|
| conservative | 0.46 / 1.00 | 0.385 [0.298, 0.471] | 0.750 | 0/0 | 22 | 42 |
| balanced | 0.43 / 0.54 | 0.538 [0.442, 0.635] | 0.714 | 2/7 | 22 | 26 |
| permissive | 0.43 / 0.49 | 0.635 [0.538, 0.721] | 0.636 | 10/17 | 12 | 26 |
| supervisor_default (reference) | 0.35 / 0.70 | 0.615 [0.519, 0.712] | 0.625 | 0/0 | 38 | 2 |

Balanced and permissive settings let false-compliant outcomes through; the conservative point avoids auto-accepting LOW entirely. Recommended default: conservative or the review-everything-LOW behaviour above.

## 9. Robustness results

Perturbations of final-test texts; model fitted on the train pool; "fail" = prediction changes (invariance) or moves the wrong way (directional). Invariance checks compare the model with itself, not with gold labels [S7].

| check | result [95% CI] |
|---|---:|
| typo invariance (all final test) | 19/204 = 0.093 [0.038, 0.157] |
| typo invariance (Hindi only) | 5/39 = 0.128 [0.000, 0.308] |
| punctuation invariance | 0/303 = 0.000 [0.000, 0.000] |
| emoji invariance | 0/309 = 0.000 [0.000, 0.000] |
| append "guarantee" must not lower risk | 3/103 = 0.029 [0.000, 0.066] |
| appended injection text keeps prediction | 56/618 = 0.091 [0.049, 0.139] |
| control: appended neutral text keeps prediction | 74/618 = 0.120 [0.077, 0.165] |

The injection check does not fail more often than its neutral control (CIs overlap): sensitivity to appended text is generic instability. Hand-written probes (small, author-chosen): strict 0/10 fail; documented-convention 10/11 fail (banned phrases 5/6, disclaimer-wrapped banned phrases 3/3, plain English negation 1/1, the architecture PDF's own "guarantee 20% returns" example 1/1); judgement 7/12 fail [S7].

## 10. Fallback behaviour

- **Production path.** The pipeline loads the committed TF-IDF pickle directly [S19].
- **Selector (`load_best_available_classifier`).** Prefers the embedding classifier only if its artifact, library versions and locally cached encoder all check out; otherwise TF-IDF baseline; if that artifact is unusable, the rule-based `MockRiskClassifier`. It never raises for a missing optional piece [S15].
- **Abstention and routing.** Low-confidence or low-margin predictions become ABSTAIN/REVIEW (section 8) [S16].
- **Independent verifier.** The Day 8 recheck flags no-op rewrites, worsened risk and blind originals, but it uses the same TF-IDF family and shares its blind spots (`docs/day8_independent_recheck_m2.md`).
- **Readiness status in the review sandbox.** Artifact hashes matched; the offline encoder and the end-to-end smoke request could not run (`sentence_transformers` and `langgraph` not installed); the demo gate is **not verified here** and must be confirmed on a clean machine [S17].

## 11. Ethical considerations and known failure modes

- **Over-trust risk.** A LOW label can be mistaken for approval. Keep the disclaimer visible; never auto-clear.
- **Classifier-blind cases.** Statements containing banned phrases can still be labelled LOW: 7 Day 7 originals out of 305 were labelled LOW despite containing a banned phrase (95% CI 2-12; leak-free refit; only a small subset of the originals contain a banned phrase, see `docs/day10_model_comparison_m2.md`) [S9, S20]. Banned-phrase and policy checks must stay independent of the classifier.
- **Negation.** Negated compliant statements are over-flagged HIGH (section 7) [S6], and the plain English negation probe fails (section 9) [S7].
- **Hindi coverage.** Weaker than English at equal content (out-of-fold gap established, section 7) [S1]; Hindi-English code-mixed coverage is 7 final-test rows and effectively unevaluated [S4]. Hindi cases should default to human review.
- **Label noise.** Single labelling pass; no second-annotator agreement measured. 15 rows are flagged for M1 review as model/label disagreements; none is a confirmed error, and a reviewer's reading of the English/code-mixed rows found the given labels plausible [S12, S18].
- **Class imbalance.** MEDIUM is the thinnest class (17 test rows); its metrics are the least stable [S4].
- **Data limits.** Near-parallel statements (translations, minimal negation pairs) may not represent unconstrained real-world copy [S5].

## 12. Caveats, recommendations and reproduction

1. Use only as triage; require human review of every LOW and every Hindi/hi-en result.
2. Keep verifier banned-phrase/policy checks independent of this classifier.
3. Do not present predicted probabilities as calibrated; prefer the conservative abstention behaviour.
4. Before any "model of record" change to the embedding classifier, run it once on the frozen split with the command below and rebuild this card.
5. Follow-up for M2: train a pool-only artifact under a new file name for evidence parity (the committed pickle is contaminated for the final test).

Commands, Windows Command Prompt (repository root, virtual environment active):

```bat
python -m pip install -r requirements.txt
python scripts\prepare_demo_artifacts.py
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python scripts\check_demo_readiness.py
python scripts\run_m2_final_evaluation.py --seed 42
python scripts\build_model_card_m2.py
```

Bash equivalents:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_demo_artifacts.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/check_demo_readiness.py
python scripts/run_m2_final_evaluation.py --seed 42
python scripts/build_model_card_m2.py
```

One online run is needed first to cache the encoder (optional; for the embedding classifier): `python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"`, then `python scripts\run_m2_final_evaluation.py --seed 42 --embedding on` (bash: use `/`). Do not run `prepare_demo_artifacts.py --force` for routine setup; it can overwrite committed artifacts. Re-running the final evaluation re-scores the frozen test set: use it to verify reproducibility, never to choose models.

## Sources

| ID | file | supports |
|---|---:|---:|
| S1 | `data/evaluation/final_eval_results_m2.csv` | Day 12 metrics, 95% cluster-bootstrap CIs, paired differences, EN-HI matched comparison (final test + train-pool OOF) |
| S2 | `data/evaluation/final_eval_confusions_m2.csv` | Day 12 confusion counts |
| S3 | `results/MANIFEST_m2.json` | Day 12 run manifest: seed, bootstrap count, CV design, data/output hashes, library versions |
| S4 | `data/evaluation/final_split_m2.meta.json` | Frozen split: rows, groups, leakage verification, test composition |
| S5 | `data/evaluation/audit_eval_data_m2.txt` | Data audit: file identity, groups, label balance, committed-model training contamination |
| S6 | `data/evaluation/classification_error_analysis_m2.txt` | Day 12 OOF error analysis (single seed-42 5-fold run): confusion pairs, confidence bands, tokenization |
| S7 | `data/evaluation/adversarial_results_m2.csv` | Day 12 robustness: invariance, directional, injection/control, hand-written probes (meta: adversarial_results_m2.meta.json) |
| S8 | `data/evaluation/abstention_operating_points_m2.csv` | Day 9 abstention operating points (held-out rows; leakage caveat in text) |
| S9 | `data/evaluation/model_usefulness_m2.csv` | Day 10 ECE and classifier-blind counts (leak-free refit; embedding vs baseline) |
| S10 | `data/evaluation/day10_embedding_vs_baseline_m2.csv` | Day 10 baseline vs embedding comparison (leak-free, 104 held-out rows, not the Day 12 frozen split) |
| S11 | `models/MANIFEST_m2.json` | Committed artifact hashes, training-row counts, library versions of the artifacts and environment |
| S12 | `data/evaluation/suspected_label_errors_m2.csv` | Rows flagged for M1 review (model/label disagreement; not confirmed errors) |
| S13 | `docs/annotation_policy_m1.md` | M1 label definitions (read-only) |
| S14 | `services/classifier/baseline.py` | Architecture and hyperparameters |
| S15 | `services/classifier/model_selection.py` | Classifier fallback chain (embedding -> TF-IDF -> rule-based mock) |
| S16 | `services/classifier/abstention.py` | Abstention decision rule and default floors |
| S17 | `data/evaluation/demo_readiness_day13_m2.txt` | Real readiness-check output captured in the review sandbox (status text only) |
| S18 | `docs/day12_risk_classification_eval_m2.md` | Day 12 findings text, incl. M1 label-review reading (a reviewer's reading, not a measured number) |
| S19 | `agents/agent_factory.py (M4-owned, read-only)` | Line 105 loads the committed TF-IDF pickle directly; the selector in S15 is not on this path |
| S21 | `git history of models/risk_classifier_tfidf_logreg.pkl` | Artifact commit date: `git log -1 --format=%ad --date=short -- models/risk_classifier_tfidf_logreg.pkl` |
| S20 | `docs/day10_model_comparison_m2.md` | Day 10 text on the classifier-blind subset (banned-phrase originals) |
