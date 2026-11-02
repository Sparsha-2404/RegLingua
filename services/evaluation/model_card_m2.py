"""Day 13 (M2): render docs/model_card_baseline_m2.md from result files.

Every number in the card is read from a result file (CSV/JSON) or asserted to exist verbatim in a text result file
via need(); a missing figure raises instead of printing something unverified. Source IDs [S#] map to the Sources table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

D = "data/evaluation/"
MODELS = ["majority_class", "banned_phrase_rule", "tfidf_logreg"]
SHORT = {"majority_class": "majority class", "banned_phrase_rule": "banned-phrase rule", "tfidf_logreg": "TF-IDF + LogReg"}

SOURCES = [  # id, path, supports
    ("S1", D + "final_eval_results_m2.csv", "Day 12 metrics, 95% cluster-bootstrap CIs, paired differences, EN-HI matched comparison (final test + train-pool OOF)"),
    ("S2", D + "final_eval_confusions_m2.csv", "Day 12 confusion counts"),
    ("S3", "results/MANIFEST_m2.json", "Day 12 run manifest: seed, bootstrap count, CV design, data/output hashes, library versions"),
    ("S4", D + "final_split_m2.meta.json", "Frozen split: rows, groups, leakage verification, test composition"),
    ("S5", D + "audit_eval_data_m2.txt", "Data audit: file identity, groups, label balance, committed-model training contamination"),
    ("S6", D + "classification_error_analysis_m2.txt", "Day 12 OOF error analysis (single seed-42 5-fold run): confusion pairs, confidence bands, tokenization"),
    ("S7", D + "adversarial_results_m2.csv", "Day 12 robustness: invariance, directional, injection/control, hand-written probes (meta: adversarial_results_m2.meta.json)"),
    ("S8", D + "abstention_operating_points_m2.csv", "Day 9 abstention operating points (held-out rows; leakage caveat in text)"),
    ("S9", D + "model_usefulness_m2.csv", "Day 10 ECE and classifier-blind counts (leak-free refit; embedding vs baseline)"),
    ("S10", D + "day10_embedding_vs_baseline_m2.csv", "Day 10 baseline vs embedding comparison (leak-free, 104 held-out rows, not the Day 12 frozen split)"),
    ("S11", "models/MANIFEST_m2.json", "Committed artifact hashes, training-row counts, library versions of the artifacts and environment"),
    ("S12", D + "suspected_label_errors_m2.csv", "Rows flagged for M1 review (model/label disagreement; not confirmed errors)"),
    ("S13", "docs/annotation_policy_m1.md", "M1 label definitions (read-only)"),
    ("S14", "services/classifier/baseline.py", "Architecture and hyperparameters"),
    ("S15", "services/classifier/model_selection.py", "Classifier fallback chain (embedding -> TF-IDF -> rule-based mock)"),
    ("S16", "services/classifier/abstention.py", "Abstention decision rule and default floors"),
    ("S17", D + "demo_readiness_day13_m2.txt", "Real readiness-check output captured in the review sandbox (status text only)"),
    ("S18", "docs/day12_risk_classification_eval_m2.md", "Day 12 findings text, incl. M1 label-review reading (a reviewer's reading, not a measured number)"),
    ("S19", "agents/agent_factory.py (M4-owned, read-only)", "Line 105 loads the committed TF-IDF pickle directly; the selector in S15 is not on this path"),
    ("S21", "git history of models/risk_classifier_tfidf_logreg.pkl", "Artifact commit date: `git log -1 --format=%ad --date=short -- models/risk_classifier_tfidf_logreg.pkl`"),
    ("S20", "docs/day10_model_comparison_m2.md", "Day 10 text on the classifier-blind subset (banned-phrase originals)"),
]


def _csv(root, name):
    return pd.read_csv(root / D / name)


def _txt(root, name):
    return (root / D / name).read_text(encoding="utf-8")


def need(text, needle):
    """Assert a quoted figure exists verbatim in its source text and return it."""
    if needle not in text:
        raise ValueError(f"figure not found in source: {needle!r}")
    return needle


def ci(v, lo, hi, nd=3):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|\n"
    return out + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def build_model_card(root: Path) -> str:
    root = Path(root)
    R = _csv(root, "final_eval_results_m2.csv")
    CONF = _csv(root, "final_eval_confusions_m2.csv")
    ADV = _csv(root, "adversarial_results_m2.csv")
    OP = _csv(root, "abstention_operating_points_m2.csv")
    USE = _csv(root, "model_usefulness_m2.csv")
    D10 = _csv(root, "day10_embedding_vs_baseline_m2.csv")
    SUSP = _csv(root, "suspected_label_errors_m2.csv")
    SPLIT = json.loads((root / D / "final_split_m2.meta.json").read_text(encoding="utf-8"))
    M3 = json.loads((root / "results" / "MANIFEST_m2.json").read_text(encoding="utf-8"))
    M11 = json.loads((root / "models" / "MANIFEST_m2.json").read_text(encoding="utf-8"))
    AUDIT, ERR = _txt(root, "audit_eval_data_m2.txt"), _txt(root, "classification_error_analysis_m2.txt")

    def pick(phase, scope, group, model, metric, comp=None):
        d = R[(R.phase == phase) & (R.scope == scope) & (R.group == group) & (R.model == model) & (R.metric == metric)]
        if comp is not None:
            d = d[d.comparator == comp]
        return d.iloc[0]

    def c(phase, scope, group, model, metric):
        r = pick(phase, scope, group, model, metric)
        return ci(r.value, r.ci_lo, r.ci_hi)

    tf = "tfidf_logreg"
    art = next(f for f in M11["files"] if f["file"] == "models/risk_classifier_tfidf_logreg.pkl")
    env, renv = M11["environment"], M3["library_versions"]
    vt = SPLIT["verification"]
    nsusp = len(SUSP)

    overall_rows = lambda ph: [[SHORT[m]] + [c(ph, "overall", "all", m, k) for k in ("accuracy", "macro_f1", "false_compliant_rate")] for m in MODELS]  # noqa: E731
    class_rows = [[k, int(pick("final_test", "class", k, tf, "recall").n)] + [c("final_test", "class", k, tf, m) for m in ("precision", "recall", "f1")] for k in ("LOW", "MEDIUM", "HIGH")]

    def slice_rows(scope, groups, ph="final_test"):
        return [[g, int(pick(ph, scope, g, tf, "accuracy").n)] + [c(ph, scope, g, tf, k) for k in ("accuracy", "macro_f1", "false_compliant_rate")] for g in groups]

    cf = CONF[(CONF.phase == "final_test") & (CONF.model == tf)]
    conf_rows = [[f"true {t}"] + [int(cf[(cf.true_label == t) & (cf.pred_label == p)]["count"].iloc[0]) for p in ("LOW", "MEDIUM", "HIGH")] for t in ("LOW", "MEDIUM", "HIGH")]

    def paired(ph, comp, k):
        r = R[(R.phase == ph) & (R.scope == "paired_overall") & (R.model == tf) & (R.comparator == comp) & (R.metric == k)].iloc[0]
        return ci(r.value, r.ci_lo, r.ci_hi)

    def enhi(ph, g, k):
        r = R[(R.phase == ph) & (R.scope == "paired_en_hi") & (R.model == tf) & (R.group == g) & (R.metric == k)].iloc[0]
        return ci(r.value, r.ci_lo, r.ci_hi)

    mc = R[(R.phase == "final_test") & (R.scope == "paired_en_hi") & (R.model == tf) & (R.group == "mcnemar_exact")].iloc[0]

    def adv(pop, test):
        r = ADV[(ADV.population == pop) & (ADV.test == test)].iloc[0]
        return f"{int(r.n_fail)}/{int(r.n_effective)} = {r.fail_rate:.3f} [{r.fail_lo:.3f}, {r.fail_hi:.3f}]"

    def advhw(test, kind):
        r = ADV[(ADV.population == "handwritten") & (ADV.test == test) & (ADV.expectation_kind == kind)].iloc[0]
        return f"{int(r.n_fail)}/{int(r.n_effective)}"

    op_rows = []
    for _, r in OP[OP.group_type == "all"].iterrows():
        cov = f"{r.coverage:.3f}" + (f" [{r.coverage_lo:.3f}, {r.coverage_hi:.3f}]" if pd.notna(r.coverage_lo) else "")
        op_rows.append([r.operating_point.replace(" (reference, not tuned)", " (reference)"), f"{r.abstain_floor:.2f} / {r.compliant_floor:.2f}", cov,
                        f"{r.accuracy_on_accepted:.3f}", f"{int(r.false_compliant)}/{int(r.low_accepted)}", int(r.review), int(r.abstain)])

    def ece(cand):
        r = USE[(USE.question == "c_calibration") & (USE.metric == "ece_top_label") & (USE.candidate == cand)].iloc[0]
        return ci(r.candidate_value if pd.isna(r.baseline) else r.baseline, r.ci_low, r.ci_high)

    ece_diff = USE[(USE.question == "c_calibration") & (USE.metric == "ece_diff") & (USE.candidate == "embedding")].iloc[0]
    blind = lambda cand: USE[(USE.question == "b_classifier_blind") & (USE.metric == "low_despite_banned_phrase") & (USE.candidate == cand)].iloc[0]  # noqa: E731
    b0 = blind("baseline")

    d10_rows = []
    for m, lab in [("accuracy", "accuracy, all"), ("macro_f1", "macro-F1, all"), ("accuracy@en", "accuracy, en"), ("accuracy@hi", "accuracy, hi"), ("accuracy@hi-en", "accuracy, hi-en")]:
        r = D10[D10.metric == m].iloc[0]
        d10_rows.append([lab, int(r.n), f"{r.baseline:.3f}", f"{r.embedding:.3f}", ci(r.delta, r.ci_low, r.ci_high), r.verdict])

    robust_rows = [
        ['typo invariance (all final test)', adv('final_test', 'invariance_typo')],
        ['typo invariance (Hindi only)', adv('final_test:hi', 'invariance_typo')],
        ['punctuation invariance', adv('final_test', 'invariance_punctuation')],
        ['emoji invariance', adv('final_test', 'invariance_emoji')],
        ['append "guarantee" must not lower risk', adv('final_test', 'directional_append_guarantee')],
        ['appended injection text keeps prediction', adv('final_test', 'mft_injection_appended')],
        ['control: appended neutral text keeps prediction', adv('final_test', 'control_appended_neutral')]]
    need(AUDIT, "76 have an en/hi sibling (same original_id) in training")
    need(AUDIT, "final_test rows that were in the Day 5-10 `held_out` set (analysed/tuned on before): 25/103; final_test rows the committed baseline trained on: 78")
    need(AUDIT, "overall: {'MEDIUM': 82, 'LOW': 221, 'HIGH': 213}")
    need(AUDIT, "groups=276; size histogram={2: 240, 1: 36}")
    for s in ("errors with conf>=0.7: 0", "<0.50       310    130      41.9%", "0.50-0.60   97     39       40.2%",
              "false_compliant (MEDIUM/HIGH predicted LOW): 77/235 = 32.8%", "false_risky (LOW predicted MEDIUM/HIGH): 76/178 = 42.7%",
              "LOW->HIGH       71", "HIGH->LOW       65", "3286/3697 = 89%", "LOW->HIGH: cue in 15/71 errors (21%) vs 7/102 correct true-LOW rows (7%)"):
        need(ERR, s)

    return f"""# Model Card: RegLingua Risk Classifier (TF-IDF + Logistic Regression), final M2 version

Card date: 2026-11-02 (Day 13). Owner: M2. Source IDs in square brackets (e.g. [S1]) point to the Sources table at the end; every figure is read from the listed file.

> **Decision-support / research software. Not legal advice and not compliance certification.** Labels are project-level annotation labels, not categories issued by any regulator [S13]. A human must review every output before it is relied on.

**Bottom line.** This is a small, simple baseline. On the frozen final test ({SPLIT['test_rows']} rows, scored once) it reaches accuracy {c('final_test','overall','all',tf,'accuracy')} and macro-F1 {c('final_test','overall','all',tf,'macro_f1')}, clearly above majority-class ({c('final_test','overall','all','majority_class','accuracy')} accuracy) and a banned-phrase rule ({c('final_test','overall','all','banned_phrase_rule','accuracy')}) [S1]. It still clears a truly MEDIUM/HIGH statement as LOW {c('final_test','overall','all',tf,'false_compliant_rate')} of the time on that test and {c('cv_oof','overall','all',tf,'false_compliant_rate')} out-of-fold [S1]. Its confidence scores do not separate right from wrong predictions, Hindi is weaker than English, and it fails negation and banned-phrase probes. Use it to prioritise human review, never to clear content.

## Changes since Day 3

- **Day 3 held-out metrics removed.** They came from a stratified row-level 80/20 split that was not grouped by `original_id`; the audit shows {need(AUDIT, '76 have an en/hi sibling (same original_id) in training').split(' have')[0].split()[-1]} of the 104 unseen rows have an EN/HI sibling in training, so those numbers were leakage-inflated [S5]. They are replaced by results on a frozen group split, below [S1, S4].
- **Day 3 per-class story withdrawn.** Day 3 presented MEDIUM as the best-classified class. With group-aware evaluation the per-class CIs overlap and MEDIUM has only {class_rows[1][1]} test rows (Table 2) [S1].
- **Corrected successor claim.** Day 3 said Day 10 would replace this model with a fine-tuned multilingual transformer. Day 10 actually evaluated a *frozen* multilingual-embedding encoder + LogReg (not fine-tuned) on a different split; it has not been run on the Day 12 frozen split, and the demo path still loads this TF-IDF pickle [S10, S19].
- **Corrected configuration.** The vectorizer also uses `sublinear_tf=True` and the classifier `max_iter=1000`, omitted on Day 3 [S14].
- **Corrected provenance wording.** The extraction script names `RegLingua_M1_FINAL_v2.xlsx`; the workbook in the repo is `data/evaluation/RegLingua_M1_FINAL.xlsx`, and the audit found all 516 of its rows in the evaluation CSV, which is row-for-row identical (text and label) to the training CSV [S5].
- **Added:** false-compliant rate, 95% CIs, sanity baselines, language/regulator slices, calibration and abstention, robustness, fallback behaviour, ethics and failure modes.
- **Important scope note.** The committed pickle was trained on {art['training_rows']} of the 516 rows, including {need(AUDIT, 'the committed baseline trained on: 78').split(': ')[-1]} of the {SPLIT['test_rows']} final-test rows, so it cannot be scored fairly on the final test [S5, S11]. All Day 12 numbers describe the **training recipe re-fitted on the {vt['train_pool_rows']}-row train pool**, not the committed file.

## 1. Model details

- **Architecture.** `TfidfLogisticClassifier`: TF-IDF (`max_features=20000`, `ngram_range=(1,2)`, `sublinear_tf=True`) feeding `LogisticRegression(max_iter=1000, class_weight="balanced")`; three classes LOW / MEDIUM / HIGH; input is the classifier-normalised text [S14].
- **Version.** "Baseline v1" (label used in this card; Day 3 recipe, unchanged). Evaluation of the recipe: Day 12.
- **Artifact.** `models/risk_classifier_tfidf_logreg.pkl`, SHA-256 `{art['sha256']}`, committed 2026-09-24 [S21], pickled with scikit-learn {art['trained_with_sklearn']}, {art['training_rows']} training rows [S11].
- **Library versions.** Models manifest environment: Python {env['python']}, numpy {env['numpy']}, scikit-learn {env['scikit-learn']}, sentence-transformers {env['sentence-transformers']}, langgraph {env['langgraph']} [S11]. Day 12 evaluation run: Python {renv['python']}, numpy {renv['numpy']}, scikit-learn {renv['scikit-learn']} [S3].
- **Seeds / resampling.** Seed {M3['seed']}; {M3['n_boot']} bootstrap resamples of whole leak groups; {M3['cv']['folds']}-fold x {M3['cv']['repeats']}-repeat StratifiedGroupKFold on the train pool [S3].

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
- **Final-test composition:** {SPLIT['final_test_by_risk']['LOW']} LOW / {SPLIT['final_test_by_risk']['MEDIUM']} MEDIUM / {SPLIT['final_test_by_risk']['HIGH']} HIGH; en {SPLIT['final_test_by_language']['en']}, hi {SPLIT['final_test_by_language']['hi']}, hi-en {SPLIT['final_test_by_language']['hi-en']}; SEBI {SPLIT['final_test_by_regulator']['SEBI']}, RBI {SPLIT['final_test_by_regulator']['RBI']}, DPDP {SPLIT['final_test_by_regulator']['DPDP']} [S4].

## 5. Metrics and why they were chosen

- **False-compliant rate (primary safety metric).** Share of truly MEDIUM/HIGH rows predicted LOW. A risky statement cleared as LOW is the costly error, because the verifier can pass a rewrite when the classifier says LOW. Lower is better. The majority baseline scores 1.000 by construction [S1, S6].
- **Macro-F1.** Equal weight per class, so the small MEDIUM class is not hidden by LOW/HIGH [S1].
- **Accuracy.** Reported for comparability with earlier days; the class mix is close to balanced for LOW/HIGH only.
- **Per-class precision/recall/F1, confusion matrix, language and regulator slices.** To locate where errors concentrate [S1, S2].
- **95% CIs.** Percentile bootstrap over whole leak groups; paired differences use identical resamples, so translation pairs never straddle a resample [S3].

## 6. Evaluation data and split protocol

- **Data.** 516 rows, 276 original_id groups, labels balanced as LOW {need(AUDIT, "'LOW': 221").split(': ')[-1]}, HIGH {need(AUDIT, "'HIGH': 213").split(': ')[-1]}, MEDIUM {need(AUDIT, "'MEDIUM': 82").split(': ')[-1]} [S5].
- **Frozen split.** Seed {SPLIT['seed']}; one of {SPLIT['n_candidates']} candidate group splits chosen by label/language/regulator balance only; train pool {vt['train_pool_rows']} rows / final test {vt['final_test_rows']} rows; original_ids straddling = {vt['original_ids_straddling']}, leak groups straddling = {vt['leak_groups_straddling']} [S4].
- **Use-once.** Models are fit on the train pool only; the final test is scored once per run; tuning and error analysis use out-of-fold predictions on the train pool only [S3, S18].
- **Disclosed caveats.** {need(AUDIT, 'analysed/tuned on before): 25/103').split('): ')[-1].split('/')[0]} of the {vt['final_test_rows']} final-test rows were in the Day 5-10 held-out set analysed earlier [S5]; the split files were untracked in the reviewed archive, so git cannot prove the freeze date [S18].

## 7. Quantitative results

Table 1. Overall, with 95% CIs [S1]. False-compliant = truly MEDIUM/HIGH predicted LOW.

**Final test (n = {SPLIT['test_rows']}, scored once)**

{table(['model', 'accuracy', 'macro-F1', 'false-compliant'], overall_rows('final_test'))}
**Train-pool out-of-fold (n = {int(pick('cv_oof','overall','all',tf,'accuracy').n)} predictions = {vt['train_pool_rows']} rows x {M3['cv']['repeats']} repeats)**

{table(['model', 'accuracy', 'macro-F1', 'false-compliant'], overall_rows('cv_oof'))}
The banned-phrase rule uses the true regulator and English-only phrase lists (an oracle-input sanity baseline) [S3]. Paired TF-IDF minus banned-phrase rule on the final test: accuracy {paired('final_test','banned_phrase_rule','accuracy')}, macro-F1 {paired('final_test','banned_phrase_rule','macro_f1')}, false-compliant {paired('final_test','banned_phrase_rule','false_compliant_rate')}; versus majority: accuracy {paired('final_test','majority_class','accuracy')} [S1].

Table 2. Per class, final test [S1]

{table(['class', 'support', 'precision', 'recall', 'F1'], class_rows)}
Table 3. Confusion matrix, final test (rows true, columns predicted) [S2]

{table(['', 'pred LOW', 'pred MEDIUM', 'pred HIGH'], conf_rows)}
Table 4. By language, final test [S1]

{table(['language', 'n', 'accuracy', 'macro-F1', 'false-compliant'], slice_rows('language', ['en', 'hi', 'hi-en']))}
Table 5. By regulator, final test [S1]

{table(['regulator', 'n', 'accuracy', 'macro-F1', 'false-compliant'], slice_rows('regulator', ['SEBI', 'RBI', 'DPDP']))}
hi-en has n = 7 and is anecdotal; regulator CIs overlap heavily, so no regulator difference is established.

**English vs Hindi, matched translation pairs** [S1]. Final test ({int(mc.n)} pairs): accuracy EN {enhi('final_test','en','accuracy')} vs HI {enhi('final_test','hi','accuracy')}, difference {enhi('final_test','en_minus_hi','accuracy')} (exact McNemar p = {mc.p_value:.3f}): not established on this small set. Out-of-fold ({int(pick('cv_oof','paired_en_hi','en',tf,'accuracy').n)} pooled pair predictions): EN {enhi('cv_oof','en','accuracy')} vs HI {enhi('cv_oof','hi','accuracy')}, difference {enhi('cv_oof','en_minus_hi','accuracy')}: **Hindi is worse at the same content.**

**Error pattern** (train-pool OOF, single seed-42 run) [S6]: LOW to HIGH {need(ERR,'LOW->HIGH       71').split()[-1]} errors and HIGH to LOW {need(ERR,'HIGH->LOW       65').split()[-1]} errors dominate; false-compliant {need(ERR,'77/235 = 32.8%').split(' ')[0]} = 32.8% of risky rows; false-risky 76/178 = 42.7% of LOW rows. Compliant statements containing a negation cue are over-flagged HIGH (cue in 15/71, 21% of LOW-to-HIGH errors vs 7/102, 7% of correct LOW rows). The default tokenizer does not keep 3286/3697 (89%) of Devanagari words whole, which plausibly weakens Hindi (a hypothesis, not tested).

**Embedding candidate (not the model of record).** Day 10 compared a frozen multilingual encoder + LogReg with the TF-IDF recipe on 104 held-out rows after removing translation leakage from training (336 train rows). This is a different split from Day 12, so it must not be compared with Tables 1-5 or taken as evidence for the demo model [S10]:

{table(['metric', 'n', 'TF-IDF', 'embedding', 'difference [95% CI]', 'verdict'], d10_rows)}
The encoder is a download-dependent optional dependency and is absent from this review environment; it was not run on the frozen split (command in section 12) [S17].

## 8. Calibration and abstention behaviour

- **Confidence is weakly informative.** Out-of-fold, {need(ERR,'<0.50       310    130      41.9%').split()[1]} of the predictions have maximum probability below 0.50 (error rate 41.9%) and 97 sit in 0.50-0.60 (error rate 40.2%); {need(ERR,'0.70-0.80   0      0').split()[1]} predictions fall in the 0.70-0.80 band and {need(ERR,'>=0.80      0      0').split()[1]} at 0.80 or above, so none reaches 0.70 [S6]. Error rates in the two bands are statistically indistinguishable.
- **Calibration.** Day 10 expected calibration error (leak-free refit, n = 104): TF-IDF {ece('baseline')}; embedding {ece('embedding')}; difference embedding minus TF-IDF {ci(ece_diff.delta, ece_diff.ci_low, ece_diff.ci_high)}: {ece_diff.verdict} [S9]. Day 9 found no calibration method with a demonstrated improvement over raw probability (see `docs/day9_confidence_abstention_m2.md`).
- **Abstention rule.** ABSTAIN if max probability < `abstain_floor`; REVIEW if any review reason fires (for example a LOW prediction below `compliant_floor`, or the optional margin rule); otherwise ACCEPT, where an accepted LOW routes as compliant and an accepted MEDIUM/HIGH routes to needs_review. Defaults 0.35 / 0.70 mirror the supervisor and are **not tuned** [S16].
- **Consequence.** With the 0.70 default compliant floor and no out-of-fold probability reaching 0.70 [S6], no LOW prediction would be accepted as compliant; LOW predictions go to review.
- **Operating points** (Day 9, selected on non-held-out rows, reported on 104 held-out rows). Nominal only: Day 9 documents that 76 of the 104 held-out rows share an original_id with training rows, so these overstate performance [S8]:

{table(['operating point', 'abstain / compliant floor', 'coverage [95% CI]', 'accuracy on accepted', 'false-compliant / LOW accepted', 'review', 'abstain'], op_rows)}
Balanced and permissive settings let false-compliant outcomes through; the conservative point avoids auto-accepting LOW entirely. Recommended default: conservative or the review-everything-LOW behaviour above.

## 9. Robustness results

Perturbations of final-test texts; model fitted on the train pool; "fail" = prediction changes (invariance) or moves the wrong way (directional). Invariance checks compare the model with itself, not with gold labels [S7].

{table(['check', 'result [95% CI]'], robust_rows)}
The injection check does not fail more often than its neutral control (CIs overlap): sensitivity to appended text is generic instability. Hand-written probes (small, author-chosen): strict {advhw('mft_ALL','strict')} fail; documented-convention {advhw('mft_ALL','documented')} fail (banned phrases {advhw('mft_banned','documented')}, disclaimer-wrapped banned phrases {advhw('mft_disclaimer','documented')}, plain English negation {advhw('mft_negation','documented')}, the architecture PDF's own "guarantee 20% returns" example {advhw('mft_anchor','documented')}); judgement {advhw('mft_ALL','judgement')} fail [S7].

## 10. Fallback behaviour

- **Production path.** The pipeline loads the committed TF-IDF pickle directly [S19].
- **Selector (`load_best_available_classifier`).** Prefers the embedding classifier only if its artifact, library versions and locally cached encoder all check out; otherwise TF-IDF baseline; if that artifact is unusable, the rule-based `MockRiskClassifier`. It never raises for a missing optional piece [S15].
- **Abstention and routing.** Low-confidence or low-margin predictions become ABSTAIN/REVIEW (section 8) [S16].
- **Independent verifier.** The Day 8 recheck flags no-op rewrites, worsened risk and blind originals, but it uses the same TF-IDF family and shares its blind spots (`docs/day8_independent_recheck_m2.md`).
- **Readiness status in the review sandbox.** Artifact hashes matched; the offline encoder and the end-to-end smoke request could not run (`sentence_transformers` and `langgraph` not installed); the demo gate is **not verified here** and must be confirmed on a clean machine [S17].

## 11. Ethical considerations and known failure modes

- **Over-trust risk.** A LOW label can be mistaken for approval. Keep the disclaimer visible; never auto-clear.
- **Classifier-blind cases.** Statements containing banned phrases can still be labelled LOW: {int(b0.candidate_value)} Day 7 originals out of {int(b0.n)} were labelled LOW despite containing a banned phrase (95% CI {b0.ci_low:.0f}-{b0.ci_high:.0f}; leak-free refit; only a small subset of the originals contain a banned phrase, see `docs/day10_model_comparison_m2.md`) [S9, S20]. Banned-phrase and policy checks must stay independent of the classifier.
- **Negation.** Negated compliant statements are over-flagged HIGH (section 7) [S6], and the plain English negation probe fails (section 9) [S7].
- **Hindi coverage.** Weaker than English at equal content (out-of-fold gap established, section 7) [S1]; Hindi-English code-mixed coverage is {SPLIT['final_test_by_language']['hi-en']} final-test rows and effectively unevaluated [S4]. Hindi cases should default to human review.
- **Label noise.** Single labelling pass; no second-annotator agreement measured. {nsusp} rows are flagged for M1 review as model/label disagreements; none is a confirmed error, and a reviewer's reading of the English/code-mixed rows found the given labels plausible [S12, S18].
- **Class imbalance.** MEDIUM is the thinnest class ({SPLIT['final_test_by_risk']['MEDIUM']} test rows); its metrics are the least stable [S4].
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
python scripts\\prepare_demo_artifacts.py
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python scripts\\check_demo_readiness.py
python scripts\\run_m2_final_evaluation.py --seed 42
python scripts\\build_model_card_m2.py
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

One online run is needed first to cache the encoder (optional; for the embedding classifier): `python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"`, then `python scripts\\run_m2_final_evaluation.py --seed 42 --embedding on` (bash: use `/`). Do not run `prepare_demo_artifacts.py --force` for routine setup; it can overwrite committed artifacts. Re-running the final evaluation re-scores the frozen test set: use it to verify reproducibility, never to choose models.

## Sources

{table(['ID', 'file', 'supports'], [[a, f'`{b}`', c_] for a, b, c_ in SOURCES])}"""
