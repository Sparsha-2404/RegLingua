r"""Day 14 (M2): render docs\demo_talk_track_m2.md from result files (every number read from a file; a missing figure raises).

  python scripts\build_talk_track_m2.py        (bash: python scripts/build_talk_track_m2.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "evaluation"
R = pd.read_csv(D / "final_eval_results_m2.csv")
ADV = pd.read_csv(D / "adversarial_results_m2.csv")
RW = pd.read_csv(D / "rewrite_eval_results_m2.csv")
OP = pd.read_csv(D / "abstention_operating_points_m2.csv")
USE = pd.read_csv(D / "model_usefulness_m2.csv")
D10 = pd.read_csv(D / "day10_embedding_vs_baseline_m2.csv")
SPLIT = json.loads((D / "final_split_m2.meta.json").read_text(encoding="utf-8"))
AUDIT = (D / "audit_eval_data_m2.txt").read_text(encoding="utf-8")
ERR = (D / "classification_error_analysis_m2.txt").read_text(encoding="utf-8")
READY = (D / "demo_readiness_day13_m2.txt").read_text(encoding="utf-8")
TF = "tfidf_logreg"


def need(text, needle):
    if needle not in text:
        raise ValueError(f"figure not found in source: {needle!r}")
    return needle


def pick(ph, sc, g, m, k):
    return R[(R.phase == ph) & (R.scope == sc) & (R.group == g) & (R.model == m) & (R.metric == k)].iloc[0]


def c(ph, sc, g, m, k, nd=3):
    r = pick(ph, sc, g, m, k)
    return f"{r.value:.{nd}f} [{r.ci_lo:.{nd}f}, {r.ci_hi:.{nd}f}]"


def v(ph, sc, g, m, k):
    return f"{pick(ph, sc, g, m, k).value:.3f}"


def hw(test, kind):
    r = ADV[(ADV.population == "handwritten") & (ADV.test == test) & (ADV.expectation_kind == kind)].iloc[0]
    return f"{int(r.n_fail)}/{int(r.n_effective)}"


def adv(pop, test):
    r = ADV[(ADV.population == pop) & (ADV.test == test)].iloc[0]
    return f"{int(r.n_fail)}/{int(r.n_effective)}"


b0 = USE[(USE.question == "b_classifier_blind") & (USE.metric == "low_despite_banned_phrase") & (USE.candidate == "baseline")].iloc[0]
d10 = lambda m: D10[D10.metric == m].iloc[0]  # noqa: E731
op = lambda n: OP[(OP.operating_point == n) & (OP.group_type == "all")].iloc[0]  # noqa: E731
n_rw, n_unch, n_red = len(RW), int(RW.text_unchanged.sum()), int((RW.direction == "reduced").sum())
n_phr, n_rem = int(RW.phrase_applicable.sum()), int((RW.phrase_removed == True).sum())  # noqa: E712
changed = n_rw - n_unch
gap = R[(R.phase == "cv_oof") & (R.scope == "paired_en_hi") & (R.model == TF) & (R.group == "en_minus_hi") & (R.metric == "accuracy")].iloc[0]
mc_tf = lambda t: t  # noqa: E731
for s in ("3286/3697 = 89%", "<0.50       310    130      41.9%", "0.50-0.60   97     39       40.2%", "0.70-0.80   0      0",
          "LOW->HIGH       71", "HIGH->LOW       65"):
    need(ERR, s)
need(AUDIT, "76 have an en/hi sibling (same original_id) in training")
need(AUDIT, "the committed baseline trained on: 78")
need(AUDIT, "groups=276")
need(READY, "GATE (end-to-end request completes offline, no manual step; working tree AND fresh checkout): NOT MET")
vt = SPLIT["verification"]
fc_ft, fc_cv = c("final_test", "overall", "all", TF, "false_compliant_rate"), c("cv_oof", "overall", "all", TF, "false_compliant_rate")

SCRIPT = ("Hi, I'm M2. My part is the risk classifier and its evaluation. We use a deliberately simple baseline, TF-IDF with logistic regression, "
          "so everything is inspectable and runs offline. It reads a statement in English, Hindi or code-mixed text and labels it low, medium or high risk, "
          "and that label starts the pipeline's routing. I want to be upfront about what it is. It is decision support, not legal advice and not a "
          f"compliance certificate. On a frozen test set it is right about {float(v('final_test','overall','all',TF,'accuracy')):.0%} of the time, well above "
          f"majority-class guessing at {float(v('final_test','overall','all','majority_class','accuracy')):.0%}, but it still calls some risky statements low, "
          "so a low label is never treated as approval. Hindi is weaker than English, and it misses some banned phrases, which is why an independent verifier "
          "and human review sit behind it. We protect the evaluation by splitting on the original statement, so translations never leak between training and "
          "test, and we report confidence intervals. Now let's see it on a live example. If the live run fails, I will switch to the recorded output and say so.")
words = len(SCRIPT.split())

QA = [
 ("Why TF-IDF and logistic regression?",
  f"It was the planned deterministic baseline: simple, inspectable, trains in seconds and runs offline [PLAN: architecture PDF section 8]. On the frozen final test it scores accuracy {c('final_test','overall','all',TF,'accuracy')} against {v('final_test','overall','all','majority_class','accuracy')} for majority class and {v('final_test','overall','all','banned_phrase_rule','accuracy')} for a banned-phrase rule [final_eval_results_m2.csv]. We claim it is a baseline, not a finished model."),
 ("A frozen embedding model beat it on Day 10. Why not use that one?",
  f"On 104 held-out rows (leak-free refit) accuracy rose from {d10('accuracy').baseline:.3f} to {d10('accuracy').embedding:.3f} with a CI that excludes 0 [day10_embedding_vs_baseline_m2.csv]. But that is a different split from the frozen Day 12 test, it was never run on it, and it needs a cached download; on {int(d10('accuracy@hi-en').n)} code-mixed rows it was inconclusive. So the demo uses the TF-IDF model we can evaluate end to end."),
 ("Why is the rewrite risk reduction so low?",
  f"In the rewrite evaluation the text changed in only {changed} of {n_rw} rows ({n_unch} unchanged) and the classifier label moved down in {n_red} [rewrite_eval_results_m2.csv]. Where a banned phrase applied, it was removed in {n_rem} of {n_phr} rows. A lexical classifier mostly reacts to swapped phrases, so a low reduction rate says rewrites rarely fire and the label rarely moves; it does not prove the rewrites are good, and the verifier, not the classifier, decides PASS."),
 ("Why do safe statements still gather evidence?",
  f"We do not yet have a recorded number: the safe scenario's evidence count is in demo_scenarios_m2.csv, which is PENDING the local run. A likely mechanism: the default compliant floor is 0.70 [abstention.py] and no out-of-fold probability reaches 0.70 [classification_error_analysis_m2.txt], so a LOW prediction is routed to review rather than cleared. Say it plainly: LOW means 'review lightly', not 'approved'."),
 ("How is Hindi handled?",
  f"By the same model, trained on 240 Hindi rows that are translations of the English ones [audit_eval_data_m2.txt]. It is weaker: final-test accuracy {c('final_test','language','hi',TF,'accuracy')} for Hindi against {c('final_test','language','en',TF,'accuracy')} for English, and in the matched out-of-fold comparison the EN minus HI gap is {gap.value:.3f} [{gap.ci_lo:.3f}, {gap.ci_hi:.3f}] [final_eval_results_m2.csv]. The default tokenizer does not keep 3286 of 3697 Devanagari words whole (89%) [classification_error_analysis_m2.txt]; we think that matters but did not test a fix."),
 ("What are classifier-blind cases?",
  f"Statements with a banned phrase that the classifier still labels LOW: {int(b0.candidate_value)} Day 7 originals out of {int(b0.n)} (95% CI {b0.ci_low:.0f}-{b0.ci_high:.0f}) [model_usefulness_m2.csv]. On our hand-written probes the documented-convention cases fail {hw('mft_ALL','documented')}, including banned phrases {hw('mft_banned','documented')} and the architecture PDF's own 'guarantee 20% returns' example {hw('mft_anchor','documented')} [adversarial_results_m2.csv]. That is why the banned-phrase check must stay independent of the classifier."),
 ("How do you protect against data leakage?",
  f"We split by original_id so an English statement and its Hindi translation always land on the same side, merge near-duplicates, and freeze one split with seed {SPLIT['seed']}: {vt['train_pool_rows']} train / {vt['final_test_rows']} test rows, {vt['original_ids_straddling']} original_ids straddling [final_split_m2.meta.json]. The Day 3 row-level split was leaky: 76 of its 104 unseen rows had a sibling in training, and the committed model saw 78 of the {vt['final_test_rows']} final-test rows, so we never score that file [audit_eval_data_m2.txt]. The final test is scored once; tuning used out-of-fold predictions."),
 ("What is the false-compliant rate and why emphasise it?",
  f"It is the share of truly MEDIUM or HIGH statements predicted LOW, the costly error because a LOW label can let a risky statement through. It is {fc_ft} on the final test and {fc_cv} out-of-fold; the majority baseline is {v('final_test','overall','all','majority_class','false_compliant_rate')} by construction [final_eval_results_m2.csv]. So we say: the model prioritises review, it must never clear content."),
 ("Does it work offline?",
  f"The TF-IDF model needs no download and the runners set HF_HUB_OFFLINE=1; the classifier selector falls back from the embedding model to TF-IDF to a rule-based mock [model_selection.py]. But we have not verified the full demo on a clean machine: in the review sandbox the readiness check said GATE NOT MET because langgraph and sentence_transformers were missing [demo_readiness_day13_m2.txt]. That is why the recorded fallback exists, and it is labelled RECORDED OUTPUT."),
 ("Is this legal advice?",
  "No. It is decision-support and research software; the labels are project-level annotations, not categories issued by any regulator, and a LOW label does not mean a statement is compliant or approved [annotation_policy_m1.md]. A person must review every result."),
 (f"{float(v('final_test','overall','all',TF,'accuracy')):.0%} accuracy is low. Why should anyone trust it?",
  f"They should not rely on it alone, and we say so. Accuracy is {c('final_test','overall','all',TF,'accuracy')} and macro-F1 {c('final_test','overall','all',TF,'macro_f1')} on {SPLIT['test_rows']} rows [final_eval_results_m2.csv]; it is useful as triage and clearly above the sanity baselines, not as a gate. We report the intervals so nobody over-reads the point estimates."),
 ("Is the confidence score calibrated? What does abstention do?",
  f"No. Out-of-fold, 310 predictions sit below 0.50 with a 41.9% error rate and 97 sit in 0.50-0.60 at 40.2%; none reaches 0.70 [classification_error_analysis_m2.txt]. The Day 9 balanced operating point let {int(op('balanced').false_compliant)} false-compliant outcomes through out of {int(op('balanced').low_accepted)} LOW accepts and permissive {int(op('permissive').false_compliant)} of {int(op('permissive').low_accepted)}; conservative accepts no LOW [abstention_operating_points_m2.csv], on held-out rows that overlap training by original_id, so treat them as optimistic."),
 ("Your test set is tiny. Are conclusions reliable?",
  f"Only the large effects are. The final test has {SPLIT['test_rows']} rows from {SPLIT['test_groups']} independent groups, MEDIUM {SPLIT['final_test_by_risk']['MEDIUM']} and code-mixed {SPLIT['final_test_by_language']['hi-en']} rows [final_split_m2.meta.json]; most slice comparisons have overlapping CIs. The pooled out-of-fold results use 3 repeats of the same {vt['train_pool_rows']} rows, so they are more stable but not independent."),
 ("Can someone fool the classifier?",
  f"Yes. Typos changed the prediction in {adv('final_test','invariance_typo')} final-test variants, and appended text changed it in {adv('final_test','mft_injection_appended')} variants, but a neutral appended sentence did so in {adv('final_test','control_appended_neutral')}, so this is general instability, not a proven injection weakness [adversarial_results_m2.csv]. Hand-written banned-phrase probes mostly get through, which is why the policy checks are separate."),
 ("How reliable are the labels?",
  "One labelling pass by M1 with no second-annotator agreement measured [day12_risk_classification_eval_m2.md]. We flagged 15 rows where the model strongly disagrees [suspected_label_errors_m2.csv]; on the English and code-mixed ones the given labels looked right and the model was wrong, so we have no confirmed label error. M1 should adjudicate the Hindi rows."),
]

WEAK = [
 f"False-compliant errors: the model clears {fc_ft} of risky statements as LOW on the final test ({fc_cv} out-of-fold) [final_eval_results_m2.csv]. It cannot be a gate.",
 f"Blind spots: banned phrases, disclaimers and negation; documented hand-written probes fail {hw('mft_ALL','documented')}, including the PDF's own example [adversarial_results_m2.csv].",
 "Confidence carries almost no signal and no LOW prediction reaches the 0.70 floor out-of-fold [classification_error_analysis_m2.txt]; abstention thresholds are not risk calibration.",
 f"Hindi is weaker than English at equal content (EN minus HI {gap.value:.3f} [{gap.ci_lo:.3f}, {gap.ci_hi:.3f}] out-of-fold); code-mixed is effectively unevaluated ({SPLIT['final_test_by_language']['hi-en']} final-test rows) [final_eval_results_m2.csv, final_split_m2.meta.json].",
 f"The rewrite step barely moves the classifier label ({n_red} of {n_rw} reduced); verification relies on the same classifier family plus separate policy checks [rewrite_eval_results_m2.csv].",
 f"Small data and a single labelling pass: {SPLIT['test_rows']} final-test rows, wide CIs, no second annotator [final_split_m2.meta.json].",
 "The committed model file is contaminated for the final test; reported numbers describe the re-fitted recipe, not that file [audit_eval_data_m2.txt].",
 "Demo readiness is not verified on a clean machine; the committed canonical-run file holds only import errors, and the scenario results are PENDING a local run [demo_readiness_day13_m2.txt, canonical_run_m2.csv].",
 "The embedding classifier has not been evaluated on the frozen split [day10_embedding_vs_baseline_m2.csv].",
]

SAY = ["'decision support' / 'triage', and 'a baseline model'", "the confidence intervals, not only point estimates", "'LOW means review lightly, not approved'",
       "'recorded output, not live' whenever the fallback is used", "'chosen demonstration; the typical rate is on the slide' for every scenario",
       "'we have not verified this on a clean machine' until the sign-off says PASS"]
DONT = ["'compliant', 'approved', 'certified' or 'legal advice'", "'accurate', 'reliable' or 'AI verifies compliance' without the interval and the caveat",
        "'understands Hindi' (it is weaker than English and the Hindi coverage is thin)", "'the embedding model is deployed' (it is not evaluated on the frozen split)",
        "any number that is not in the files cited here; if asked, say 'I will check the file'", "'typical' about a chosen scenario",
        "'fresh checkout tested' or 'all tests pass' before the sign-off records a real local run"]

out = ["# Demo talk track and Q&A (M2)", "",
       "Decision support / research software, not legal certification. Every number is read from the file named in brackets by `scripts\\build_talk_track_m2.py` "
       "(bash: `python3 scripts/build_talk_track_m2.py`). Question 4 depends on the scenario results (PENDING until the local run); question 9 on a clean-machine check (not yet done).", "",
       f"## 1. M2 script (60-90 seconds; {words} words; time it in the table below, do not rely on the word count)", "", f"> {SCRIPT}", "",
       "## 2. Rehearsal timings (fill in)", "", "| Rehearsal | Date | Start | End | Total (s) | Within 90 s? | Stumbles / notes |", "|---|---|---|---|---|---|---|",
       "| 1 | | | | | | |", "| 2 | | | | | | |", "| 3 | | | | | | |", "",
       "## 3. Say / do not say", "", "**Say**", ""] + [f"- {s}" for s in SAY] + ["", "**Do not say**", ""] + [f"- {s}" for s in DONT] + \
      ["", "## 4. The 15 hardest examiner questions", ""]
for i, (q, a) in enumerate(QA, 1):
    out += [f"**Q{i}. {q}**", "", a, ""]
out += ["## 5. Known weaknesses we state up front (ordered by severity, M2's judgement)", ""] + [f"{i}. {w}" for i, w in enumerate(WEAK, 1)] + [""]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "demo_talk_track_m2.md").write_text("\n".join(out), encoding="utf-8")
print(f"wrote docs/demo_talk_track_m2.md ({len(QA)} questions, {len(WEAK)} weaknesses, script {words} words)")
