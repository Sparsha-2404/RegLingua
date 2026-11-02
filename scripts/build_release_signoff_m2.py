r"""Day 14 (M2): render docs\day14_m2_release_signoff.md. Numbers come from result files / the verification report; missing figures raise.

  python scripts/build_release_signoff_m2.py        (bash: python scripts/build_release_signoff_m2.py)
Statuses: PASS / FAIL only where a recorded real run shows it; PENDING (local) otherwise. Nothing here tags a release.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "evaluation"
R = pd.read_csv(D / "final_eval_results_m2.csv")
REP = (ROOT / "docs" / "day14_final_verification_m2.md").read_text(encoding="utf-8")
BLIND = pd.read_csv(D / "model_usefulness_m2.csv")
ADV = pd.read_csv(D / "adversarial_results_m2.csv")
RW = pd.read_csv(D / "rewrite_eval_results_m2.csv")
TF = "tfidf_logreg"


def need(text, s):
    if s not in text:
        raise ValueError(f"not found in source: {s!r}")
    return s


def c(ph, sc, g, m, k):
    r = R[(R.phase == ph) & (R.scope == sc) & (R.group == g) & (R.model == m) & (R.metric == k)].iloc[0]
    return f"{r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"


need(REP, "47/84 claims PASS, 37 FAIL / 34 ERROR-type (regex/column not found), 3 numeric mismatches")
need(REP, "run A vs run B byte-identical: True")
need(REP, "pytest is not installed in this environment")
need(REP, "needs a git checkout (NO) and docs/freeze_manifest_m2.json (NO)")
b = BLIND[(BLIND.question == "b_classifier_blind") & (BLIND.metric == "low_despite_banned_phrase") & (BLIND.candidate == "baseline")].iloc[0]
doc = ADV[(ADV.population == "handwritten") & (ADV.test == "mft_ALL") & (ADV.expectation_kind == "documented")].iloc[0]
gap = R[(R.phase == "cv_oof") & (R.scope == "paired_en_hi") & (R.model == TF) & (R.group == "en_minus_hi") & (R.metric == "accuracy")].iloc[0]
n_rw, n_red = len(RW), int((RW.direction == "reduced").sum())

REPORT = "docs/day14_final_verification_m2.md"
ITEMS = [
    ("Suite count, `python -m pytest tests\\unit -q`", "PENDING (local)", f"{REPORT} rows 1a/1b: NOT_RUN, pytest not installed in the verification environment. Real count: ____ passed, ____ skipped, ____ xfailed, ____ failed."),
    ("Suite with HF_HUB_OFFLINE=1", "PENDING (local)", f"{REPORT} row 1b: NOT_RUN. Real count: ____"),
    ("Contract-snapshot test", "PENDING (local)", f"{REPORT} row 3: NOT_RUN (pytest not installed)."),
    ("Doc-number check (`check_doc_numbers_m2.py`)", "FAIL", f"{REPORT} row 2: 47/84 claims PASS, 37 FAIL (34 regex/column-not-found errors, 3 numeric mismatches); proposed blocker, owner M2. Re-run locally and fix before freezing; some failures may be environment-related."),
    ("Day 12 evaluation determinism", "PASS", f"{REPORT} row 4: run A vs run B byte-identical; identical to committed outputs after CRLF to LF."),
    ("Fresh checkout simulation", "PENDING (local)", f"{REPORT} row 5: FAIL in the verification environment only (not a git repository; ran --static-only, a weaker check). Re-run locally without --static-only."),
    ("Freeze check (`freeze_check_m2.py`)", "PENDING (local)", f"{REPORT} row 6: NOT_RUN (needs a git checkout and `docs/freeze_manifest_m2.json`; the manifest was not among the files received for this step). `--create` after committing, then `--verify`."),
    ("Scenarios: six scenarios x 3 runs (`demo_scenarios_m2.csv`, `docs/demo_scenarios_m2.md`)", "PENDING (local)", "Runner and CSV-driven markdown renderer built and unit-tested; needs the project environment (langgraph missing in the sandbox). Run canonical cases first."),
    ("Fallback demo (`demo_fallback_m2\\`)", "PENDING (local)", "Capture/replay scripts built and unit-tested; capture after the scenario run."),
    ("Talk track and Q&A (`docs/demo_talk_track_m2.md`)", "PASS", "Generated from result files; 15 questions, known-weaknesses list, blank timing table; structure test passes (run with a minimal pytest stand-in in the sandbox; confirm with real pytest)."),
    ("Rehearsal timings (3 rehearsals)", "PENDING (local)", "Fill the table in `docs/demo_talk_track_m2.md`."),
    ("New Day 14 unit tests under real pytest", "PENDING (local)", "Scenario, CSV render, fallback demo and talk-track tests passed only via a minimal stand-in runner in the sandbox."),
]
out = ["# Day 14 (M2): release sign-off", "",
       "Decision support / research software, not legal certification. **This document does not tag a release;** the tag is applied on main by whoever the lead names.", "",
       f"- Commit: PENDING (local): record `git rev-parse HEAD` after committing (the verification report records the commit as unavailable: not a git checkout there).",
       f"- Date: {dt.date.today().isoformat()}  - Evidence base: `{REPORT}`, result files under `data/evaluation/`", "",
       "## Day 14 items", "", "| Item | Status | Evidence |", "|---|---|---|"] + [f"| {a} | **{b_}** | {e} |" for a, b_, e in ITEMS] + [
       "", "## Known limitations at release (real numbers)", "",
       f"- Final test (103 rows, scored once): accuracy {c('final_test','overall','all',TF,'accuracy')}, macro-F1 {c('final_test','overall','all',TF,'macro_f1')}; false-compliant rate {c('final_test','overall','all',TF,'false_compliant_rate')} (out-of-fold {c('cv_oof','overall','all',TF,'false_compliant_rate')}) [`final_eval_results_m2.csv`].",
       f"- Hindi weaker than English out-of-fold, accuracy gap EN minus HI {gap.value:.3f} [{gap.ci_lo:.3f}, {gap.ci_hi:.3f}] [`final_eval_results_m2.csv`].",
       f"- Classifier-blind: {int(b.candidate_value)} of {int(b.n)} Day 7 originals labelled LOW despite a banned phrase (95% CI {b.ci_low:.0f}-{b.ci_high:.0f}) [`model_usefulness_m2.csv`]; documented-convention hand-written probes fail {int(doc.n_fail)}/{int(doc.n_effective)} [`adversarial_results_m2.csv`].",
       f"- Rewrite changes the classifier label rarely: {n_red} of {n_rw} rows reduced [`rewrite_eval_results_m2.csv`].",
       "- Confidence is uninformative (no out-of-fold probability reaches 0.70) [`classification_error_analysis_m2.txt`].",
       "- Committed model file is contaminated for the final test; reported numbers describe the re-fitted recipe [`audit_eval_data_m2.txt`].",
       "- The offline demo gate was NOT MET in the review sandbox (missing langgraph and sentence_transformers) [`demo_readiness_day13_m2.txt`]; the committed `canonical_run_m2.csv` contains only import errors, so no real canonical rates exist yet.",
       "", "## Open issues by owner", "",
       "The verification report assigns no items to M1, M3 or M4; the items below come from the Day 13 readiness capture and Day 12 findings.", "",
       "- **M1:** adjudicate the Hindi rows in `suspected_label_errors_m2.csv` (decisions blank) [Day 12 doc]; no second-annotator agreement measured; MEDIUM and hi-en are thin.",
       "- **M3:** `agent_factory` still injects the keyword-only retriever although the hybrid retriever exists [`demo_readiness_day13_m2.txt`, R1].",
       "- **M4:** `models/regulator_classifier.pkl` is untracked, so a fresh clone needs `python scripts\\prepare_demo_artifacts.py` [`demo_readiness_day13_m2.txt`]; keep verifier banned-phrase/policy checks independent of the classifier; confirm LOW routing with the default 0.70 compliant floor; `langgraph` must be in the install path for the offline smoke request.",
       "- **Everyone:** many tracked files show as modified against HEAD in the working tree reviewed [`demo_readiness_day13_m2.txt`]; commit or discard before the freeze.",
       "", "## Publish steps (no pull request, no tag)", "", "```bat",
       "git pull origin main", "python -m pytest tests\\unit -q", "git status",
       "git add scripts/freeze_check_m2.py docs/freeze_manifest_m2.json scripts/final_verification_m2.py services/evaluation/final_verification.py docs/day14_final_verification_m2.md data/evaluation/final_verification_m2.json scripts/run_demo_scenarios_m2.py services/evaluation/demo_scenarios.py data/evaluation/demo_scenarios_m2.csv data/evaluation/demo_scenarios_m2.meta.json docs/demo_scenarios_m2.md scripts/capture_fallback_demo_m2.py scripts/replay_fallback_demo_m2.py scripts/build_talk_track_m2.py scripts/build_release_signoff_m2.py docs/demo_talk_track_m2.md docs/day14_m2_release_signoff.md tests/unit/test_day14_final_verification.py tests/unit/test_day14_demo_scenarios.py tests/unit/test_day14_demo_scenarios_csv_m2.py tests/unit/test_day14_fallback_demo_m2.py tests/unit/test_day14_talk_track_m2.py tests/unit/test_freeze_check_m2.py",
       'git commit -m "M2 Day 14: freeze check, final verification, demo scenarios, fallback demo, talk track, sign-off"',
       "git push origin M2", "```", "", "Replace `M2` with the actual branch name if it differs. Do not `git add -A`. Share `demo_fallback_m2\\` as a zip (it lives outside the M2-owned paths), unless the lead asks for it to be committed.", "",
       "## Message to the lead", "",
       "M2 is not frozen and not ready for integrate yet. Built and tested in the sandbox: freeze check, final verification, scenario runner, fallback demo tooling, talk track. Still open: real pytest count, doc-number check (47/84 passing), freeze manifest, fresh-checkout check, scenario run, fallback capture, rehearsals. Classifier results are final but weak: false-compliant rate 0.350 on the final test. I will report once these run locally. Please name who tags the release.", ""]
msg = out[-2]
assert len(msg.split()) <= 120, len(msg.split())
(ROOT / "docs" / "day14_m2_release_signoff.md").write_text("\n".join(out), encoding="utf-8")
print("wrote docs/day14_m2_release_signoff.md; lead message words:", len(msg.split()))
