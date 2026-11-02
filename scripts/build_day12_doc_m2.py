"""Day 12 (M2): render docs/day12_risk_classification_eval_m2.md from the committed result files.

    python scripts/build_day12_doc_m2.py

Every table cell is read from data/evaluation/*.csv|json (no estimated numbers). Prose numbers that are not tables
are pulled from the same files or quoted from classification_error_analysis_m2.txt (single seed-42 OOF run).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "evaluation"
OUT = ROOT / "docs" / "day12_risk_classification_eval_m2.md"

R = pd.read_csv(D / "final_eval_results_m2.csv")
CONF = pd.read_csv(D / "final_eval_confusions_m2.csv")
ADV = pd.read_csv(D / "adversarial_results_m2.csv")
SPLIT = json.loads((D / "final_split_m2.meta.json").read_text(encoding="utf-8"))
MAN = json.loads((ROOT / "results" / "MANIFEST_m2.json").read_text(encoding="utf-8"))
MODELS = ["majority_class", "banned_phrase_rule", "tfidf_logreg"]
SHORT = {"majority_class": "majority", "banned_phrase_rule": "banned-phrase rule", "tfidf_logreg": "TF-IDF+LogReg"}


def pick(phase, scope, group, model, metric, comparator=None):
    d = R[(R.phase == phase) & (R.scope == scope) & (R.group == group) & (R.model == model) & (R.metric == metric)]
    if comparator is not None:
        d = d[d.comparator == comparator]
    return d.iloc[0]


def ci(row, nd=3):
    return f"{row.value:.{nd}f} [{row.ci_lo:.{nd}f}, {row.ci_hi:.{nd}f}]"


def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|\n"
    return s + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def headline(phase, scope, group):
    return [ci(pick(phase, scope, group, m, k)) for m in MODELS for k in ("accuracy", "macro_f1", "false_compliant_rate")]


def overall_table(phase):
    rows = []
    for m in MODELS:
        rows.append([SHORT[m]] + [ci(pick(phase, "overall", "all", m, k)) for k in ("accuracy", "macro_f1", "false_compliant_rate")])
    return table(["model", "accuracy", "macro-F1", "false-compliant rate"], rows)


def slice_table(phase, scope, groups):
    rows = []
    for g in groups:
        n = int(pick(phase, scope, g, "tfidf_logreg", "accuracy").n)
        rows.append([g, n] + [ci(pick(phase, scope, g, "tfidf_logreg", k)) for k in ("accuracy", "macro_f1", "false_compliant_rate")]
                    + [f"{pick(phase, scope, g, 'majority_class', 'accuracy').value:.3f}"])
    return table([scope, "n", "TF-IDF acc", "TF-IDF macro-F1", "TF-IDF false-compliant", "majority acc"], rows)


def class_table(phase):
    rows = []
    for c in ["LOW", "MEDIUM", "HIGH"]:
        rows.append([c, int(pick(phase, "class", c, "tfidf_logreg", "recall").n)]
                    + [ci(pick(phase, "class", c, "tfidf_logreg", k)) for k in ("precision", "recall", "f1")])
    return table(["class", "support", "precision", "recall", "F1"], rows)


def confusion(phase, model):
    c = CONF[(CONF.phase == phase) & (CONF.model == model)]
    rows = []
    for t in ["LOW", "MEDIUM", "HIGH"]:
        rows.append([f"true {t}"] + [int(c[(c.true_label == t) & (c.pred_label == p)]["count"].iloc[0]) for p in ["LOW", "MEDIUM", "HIGH"]])
    return table(["", "pred LOW", "pred MEDIUM", "pred HIGH"], rows)


def paired(phase, comp):
    rows = []
    for k in ("accuracy", "macro_f1", "false_compliant_rate"):
        r = R[(R.phase == phase) & (R.scope == "paired_overall") & (R.model == "tfidf_logreg") & (R.comparator == comp) & (R.metric == k)].iloc[0]
        rows.append([k, ci(r), f"{r.p_value:.4f}"])
    return table([f"TF-IDF minus {SHORT[comp]}", "difference [95% CI]", "bootstrap p"], rows)


def en_hi(phase):
    rows = []
    for k in ("accuracy", "macro_f1"):
        e, h, d = (R[(R.phase == phase) & (R.scope == "paired_en_hi") & (R.model == "tfidf_logreg") & (R.group == g) & (R.metric == k)].iloc[0]
                   for g in ("en", "hi", "en_minus_hi"))
        rows.append([k, ci(e), ci(h), ci(d)])
    p = R[(R.phase == phase) & (R.scope == "paired_en_hi") & (R.model == "tfidf_logreg") & (R.group == "mcnemar_exact")].iloc[0]
    n = int(p.n)
    return table(["metric", "EN", "HI", "EN minus HI"], rows), f"{p.note} (exact McNemar p = {p.p_value:.4f}; {n} pairs)"


def adv(pop, test):
    r = ADV[(ADV.population == pop) & (ADV.test == test)].iloc[0]
    if r.n_effective == 0:
        return "n/a (no effective variants)"
    return f"{int(r.n_fail)}/{int(r.n_effective)} = {r.fail_rate:.3f} [{r.fail_lo:.3f}, {r.fail_hi:.3f}]"


def adv_hw(test, kind):
    r = ADV[(ADV.population == "handwritten") & (ADV.test == test) & (ADV.expectation_kind == kind)].iloc[0]
    return f"{int(r.n_fail)}/{int(r.n_effective)}"


tf_ft = lambda k: pick("final_test", "overall", "all", "tfidf_logreg", k)  # noqa: E731
tf_cv = lambda k: pick("cv_oof", "overall", "all", "tfidf_logreg", k)  # noqa: E731
en_hi_ft_tab, en_hi_ft_note = en_hi("final_test")
en_hi_cv_tab, en_hi_cv_note = en_hi("cv_oof")
vt = SPLIT["verification"]

doc = f"""# Day 12 (M2): Risk Classification Metrics and Error Analysis

**Bottom line.** The TF-IDF + Logistic Regression baseline is clearly better than the sanity baselines but is **not a reliable
risk gate**. On the frozen final test ({SPLIT['test_rows']} rows, {SPLIT['test_groups']} original_id groups, scored once) it reaches
accuracy {ci(tf_ft('accuracy'))} and macro-F1 {ci(tf_ft('macro_f1'))}; majority class is {pick('final_test','overall','all','majority_class','accuracy').value:.3f}
accuracy and the banned-phrase rule {pick('final_test','overall','all','banned_phrase_rule','accuracy').value:.3f}. The dangerous error, a MEDIUM/HIGH statement predicted LOW
(**false-compliant**), is {ci(tf_ft('false_compliant_rate'))} on the final test and {ci(tf_cv('false_compliant_rate'))} out-of-fold. Hindi is weaker than
English in the matched out-of-fold comparison (OOF accuracy gap EN minus HI {ci(R[(R.phase=='cv_oof')&(R.scope=='paired_en_hi')&(R.model=='tfidf_logreg')&(R.group=='en_minus_hi')&(R.metric=='accuracy')].iloc[0])}); on the small final test the gap has the same sign but its CI includes 0. Predicted
probabilities carry almost no confidence signal (see section 5), and on the hand-written probes it fails 5 of 6 banned-phrase cases, all 3 disclaimer-wrapped cases and the plain negation case
(section 6; tiny n). Treat it as a baseline for the Day 10/11 comparison, not as a decision-maker; keep the verifier's banned-phrase check independent of it.

Final test CIs are wide (n = {SPLIT['test_rows']}); read differences between slices as directional unless the CI excludes 0.

## 1. Protocol and leakage statement

- **Data.** `risk_evidence_eval_m2.csv`, 516 rows / 276 original_ids. The "final labelled evaluation set" from M1 is
  `RegLingua_M1_FINAL.xlsx`; the data audit (`data/evaluation/audit_eval_data_m2.txt`) found its 516 rows are exact matches of the CSV, so
  **no separate M1 final set exists beyond it** and the 516-row file is used. No label was changed.
- **Grouping.** Leak groups = original_id (EN/HI translations share one) plus 3 near-duplicate links (merged). {SPLIT['groups']} groups.
- **Frozen split.** `scripts/make_final_split.py`, seed {SPLIT['seed']}, best of {SPLIT['n_candidates']} GroupShuffleSplit candidates (#{SPLIT['chosen_candidate']}) chosen only by
  label/language/regulator balance, never by model results. Train pool {vt['train_pool_rows']} rows, final test {vt['final_test_rows']} rows. Verified: original_ids straddling = {vt['original_ids_straddling']},
  leak groups straddling = {vt['leak_groups_straddling']}. Split SHA-256 `{MAN['split']['hash_sha256'][:16]}...`; data file hashes are in `results/MANIFEST_m2.json`.
- **Use-once.** Models are fit on the train pool only and `final_test` is scored once per run behind an `OnceGuard`; reported final numbers come from that single run.
  A second run in a scratch copy (verification only) reproduced all three output files byte-for-byte (SHA-256 equal to the manifest).
- **Tuning and error analysis** use out-of-fold (OOF) predictions on the train pool only: repeated StratifiedGroupKFold, 5 folds x 3 repeats
  (`random_state = 42 + repeat`), models re-fitted per fold. The committed `models/risk_classifier_tfidf_logreg.pkl` is **never scored here**: the audit shows it was trained on 412 of
  the 516 rows (78 of the final-test rows), so scoring it would be contaminated.
- **Intervals.** Percentile 95% CIs from {MAN['n_boot']} bootstrap resamples of whole leak groups, identical resamples across models for paired differences. Seed 42 throughout.
- **Disclosed protocol caveats.** (1) 25 of the 103 final-test rows were in the Day 5-10 `held_out` set that earlier analysis looked at. (2) The adversarial suite
  was run on final-test *texts* (labels not read) and six banned-phrase probes (B01-B06) were added after its first run. (3) The split files are untracked in the archive I reviewed, so git cannot prove the
  split was frozen before analysis; commit `final_split_m2.csv` and its meta now. (4) The embedding classifier was not run (encoder not cached; section 8).

## 2. Overall results

Final test ({SPLIT['test_rows']} rows, scored once):

{overall_table('final_test')}
Train-pool OOF (413 rows x 3 repeats pooled = {int(tf_cv('accuracy').n)} predictions):

{overall_table('cv_oof')}
False-compliant rate = share of truly MEDIUM/HIGH rows predicted LOW. The majority baseline is 1.000 by construction. The banned-phrase rule uses the row's true regulator (oracle input) and English-only phrase lists.

Paired differences (same resamples), final test:

{paired('final_test', 'banned_phrase_rule')}
{paired('final_test', 'majority_class')}
Out-of-fold (more data, same direction): accuracy {ci(R[(R.phase=='cv_oof')&(R.scope=='paired_overall')&(R.comparator=='banned_phrase_rule')&(R.metric=='accuracy')].iloc[0])} and macro-F1 {ci(R[(R.phase=='cv_oof')&(R.scope=='paired_overall')&(R.comparator=='banned_phrase_rule')&(R.metric=='macro_f1')].iloc[0])} versus the banned-phrase rule.

## 3. Per-class, language and regulator (TF-IDF+LogReg, final test)

{class_table('final_test')}
Confusion matrix, final test (rows true, columns predicted):

{confusion('final_test', 'tfidf_logreg')}
By language:

{slice_table('final_test', 'language', ['en', 'hi', 'hi-en'])}
By regulator:

{slice_table('final_test', 'regulator', ['SEBI', 'RBI', 'DPDP'])}
hi-en has n = 7 (7 groups): its numbers are anecdotal. Regulator differences have heavily overlapping CIs; no regulator gap is established.

Same slices on the train-pool OOF (larger n):

{slice_table('cv_oof', 'language', ['en', 'hi', 'hi-en'])}
## 4. English vs Hindi matched comparison

Matched = the same original_id in EN and HI (translation pairs), so content and label are identical and only the language differs.

Final test:

{en_hi_ft_tab}
{en_hi_ft_note}. Same sign as OOF but the CI includes 0: **not established on the final test alone**.

Train-pool OOF (pairs pooled over 3 repeats; CIs resample leak groups):

{en_hi_cv_tab}
{en_hi_cv_note}. Here the gap is established: **Hindi is worse than English at the same content** (the `verdict`/`note` text in the CSV is boilerplate that says "CI excludes 0" even when it does not; read the CI). Likely cause: default `TfidfVectorizer` tokenization splits Devanagari words at vowel signs (section 5, item 5); not tested here as a fix.

## 5. Error analysis (train-pool OOF only; single seed-42 5-fold run, `classification_error_analysis_m2.txt`)

That run gives accuracy 0.588 [0.528, 0.643], macro-F1 0.617 [0.556, 0.670], false-compliant 0.328 [0.258, 0.402] (a different resampling from the 3-repeat pooled numbers in section 2). 170/413 rows are errors.

1. **Two dominant errors, equal size.** LOW->HIGH 71 (42% of errors; 40% of true LOW) and HIGH->LOW 65 (38%; 38% of true HIGH). MEDIUM->LOW 12, MEDIUM->HIGH 10, HIGH->MEDIUM 7, LOW->MEDIUM 5.
   False-compliant 77/235 risky rows (32.8%); false-risky 76/178 LOW rows (42.7%). False-compliant is the worse error here: the verifier passes a rewrite when the classifier says LOW.
2. **Negation / polarity.** Compliant statements containing "not/do not/never" are predicted HIGH (e.g. "We do not guarantee returns..." is labeled LOW, predicted HIGH). The negation cue is present in 21% of LOW->HIGH errors versus 7% of correctly classified LOW rows.
   Minimal pairs differing only by "not" exist in the data with opposite labels (section 9).
3. **Confidence is uninformative.** 407 of 413 OOF predictions have max probability below 0.6; none reach 0.7. Error rate is 41.9% in the <0.50 band and 40.2% in 0.50-0.60 (CIs overlap). Confidence thresholds cannot separate good from bad predictions for this model.
4. **Slices (OOF error rate).** en 34.9% [28.4, 41.9], hi 46.4% [39.3, 53.4], hi-en 48.3% [31.0, 65.5]; SEBI 44.3%, DPDP 41.3%, RBI 37.8% (CIs overlap). MEDIUM is the best-classified class by error rate (33.8%) but has the fewest rows (65).
5. **Tokenization.** 3,286 of 3,697 Devanagari words (89%) are not kept as a single token by the default pattern (vowel signs are not `\\w`), producing fragments such as the top-weighted error terms. Pronoun-like fragments (e.g. the token for "we") dominate LOW->HIGH error terms: lexical shortcuts rather than regulatory content.

## 6. Adversarial results (TF-IDF fitted on the train pool; `adversarial_results_m2.csv`)

Fail = the model's behavior violates the stated expectation (invariance: prediction changed; directional: moved the wrong way). Invariance/directional checks compare the model with itself, not with gold labels. CIs resample clusters.

| check | final-test population | result [95% CI] |
|---|---|---|
| typo invariance | all final test | {adv('final_test','invariance_typo')} |
| typo invariance | HI only | {adv('final_test:hi','invariance_typo')} |
| punctuation / emoji invariance | all final test | {adv('final_test','invariance_punctuation')}; {adv('final_test','invariance_emoji')} |
| append "guarantee" must not lower risk | all final test | {adv('final_test','directional_append_guarantee')} |
| appended injection text keeps prediction | all final test | {adv('final_test','mft_injection_appended')} |
| **control**: appended neutral text keeps prediction | all final test | {adv('final_test','control_appended_neutral')} |

The injection check does not fail more often than its neutral-text control (CIs overlap), so the sensitivity to appended text is generic instability, not demonstrated injection susceptibility. Casing and whitespace perturbations are mostly trivial after preprocessing (counts in the CSV).

Hand-written probes (33 cases, small n; expectation kinds from `adversarial_classifier_tests_m2.py`):

| expectation kind | fail |
|---|---|
| strict (unambiguous) | {adv_hw('mft_ALL','strict')} |
| documented (project convention, e.g. banned phrase must not be LOW) | {adv_hw('mft_ALL','documented')} |
| judgement (reasonable people could differ; reported separately) | {adv_hw('mft_ALL','judgement')} |

Documented failures include the architecture PDF's own example ("We guarantee 20% returns...", expected HIGH, anchor probe {adv_hw('mft_anchor','documented')}), banned phrases {adv_hw('mft_banned','documented')}, disclaimer-wrapped banned phrases {adv_hw('mft_disclaimer','documented')} and the plain English negation probe {adv_hw('mft_negation','documented')}. Latin-script Hindi probes: {adv_hw('mft_latin_hindi','judgement')} (judgement).

## 7. Limitations

- **Small n.** {SPLIT['test_rows']} final-test rows (MEDIUM {SPLIT['final_test_by_risk']['MEDIUM']}, hi-en {SPLIT['final_test_by_language']['hi-en']}); only {SPLIT['test_groups']} independent groups. CIs are wide; slice comparisons are mostly inconclusive.
- **Label noise / single labelling pass.** Single labelling pass, no second-annotator agreement measured; true label noise is unknown. The 15 rows in `suspected_label_errors_m2.csv` are model-disagreement candidates, not confirmed errors (section 9).
- **Frozen splits.** One split drawn once; the single final-test number depends on that draw. OOF CIs reflect resampling, not alternative splits.
- **One classifier family.** Only TF-IDF+LogReg (plus trivial baselines) is evaluated here; no hyperparameter search was run. The embedding classifier is not in these numbers.
- **Near-parallel data.** Many rows are translations or minimal negation pairs; results may not transfer to unconstrained real-world copy. hi-en rows are singleton groups, so semantic siblings are not ruled out.
- **Baselines are weak by design.** The banned-phrase rule is English-only and oracle-regulator; beating it is a low bar.
- **Adversarial** variants are sampled, and hand-written cases are 33 items chosen by the author.
- **Environment.** The committed manifest records scikit-learn {MAN['library_versions']['scikit-learn']} / Python {MAN['library_versions']['python']}; the scratch reproduction ran scikit-learn 1.8.0 / Python 3.12.3 and still produced byte-identical result CSVs.

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
"""
OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({len(doc.splitlines())} lines)")
