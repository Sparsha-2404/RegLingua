"""Day 9 (M2): which per-prediction signals predict that the risk classifier is correct?

Protocol (never tunes on held-out):
  * Eval set = 516 rows of data/evaluation/risk_evidence_eval_m2.csv; the held-out flag
    (104 rows) comes from pipeline_retrieval_results_m2.csv (the classifier's test split).
  * The committed model was trained on the 412 non-held-out rows, so its predictions there are
    in-sample. For FITTING (signal direction, language score, combined model, calibrators) the
    non-held-out rows therefore use out-of-fold predictions: 5-fold GroupKFold by original_id
    (parallel en/hi/hi-en variants never straddle a fold), fold-trained TF-IDF+LogReg with the
    committed hyper-parameters. Held-out rows are scored once with the COMMITTED model.
  * Calibration (isotonic / sigmoid) is fitted on non-held-out only, kept in memory; nothing is
    saved and models/risk_classifier_tfidf_logreg.pkl is only read.
  * All headline numbers are on held-out rows; bootstrap 95% CIs, seed 42, 2000 resamples.

Usage (repo root): python scripts/analyze_confidence_signals.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core.verification_policy import find_banned_phrase  # noqa: E402
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402
from services.evaluation.confidence_signals import (  # noqa: E402
    auroc, bootstrap, ece, fit_direction, oov_ratio, percentile_ci, proba_signals,
)

EVAL = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
HELD = REPO_ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
MODEL = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
OUT = REPO_ROOT / "data" / "evaluation" / "confidence_signals_m2.csv"
FOLDS, LANGS = 5, ("en", "hi", "hi-en")
pc = lambda x: "n/a" if x is None else f"{x*100:.1f}%"  # noqa: E731
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731


def base_pipeline():
    return Pipeline([("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)),
                     ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def load_rows():
    held = {r["query_id"]: r["held_out"] == "True"
            for r in csv.DictReader(open(HELD, newline="", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(EVAL, newline="", encoding="utf-8-sig")))
    for r in rows:
        r["held_out"] = held[r["query_id"]]
        r["norm"] = normalize_for_classifier(r["text"])["normalized_text"]
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    rows = load_rows()
    tr = [i for i, r in enumerate(rows) if not r["held_out"]]
    ho = [i for i, r in enumerate(rows) if r["held_out"]]
    X = np.array([r["norm"] for r in rows], dtype=object)
    y = np.array([r["true_risk_level"] for r in rows])
    groups = np.array([r["original_id"] for r in rows])
    print(f"rows={len(rows)} non-held-out={len(tr)} held-out={len(ho)} | held-out original_ids also in "
          f"non-held-out: {len(set(groups[ho]) & set(groups[tr]))}/{len(set(groups[ho]))}")

    committed = TfidfLogisticClassifier().load(str(MODEL))
    classes = list(committed._classes)

    # ---- probabilities: OOF on non-held-out (fit rows), committed model on held-out (report rows)
    P = np.zeros((len(rows), len(classes)))
    oov = np.zeros(len(rows))
    splits = list(GroupKFold(n_splits=FOLDS).split(X[tr], y[tr], groups[tr]))
    for a, b in splits:
        m = TfidfLogisticClassifier().fit(list(X[tr][a]), list(y[tr][a]))
        assert list(m._classes) == classes
        ib = [tr[k] for k in b]
        P[ib] = m.model.predict_proba(m.vectorizer.transform(X[ib]))
        oov[ib] = [oov_ratio(t, m.vectorizer) for t in X[ib]]
    P[ho] = committed.model.predict_proba(committed.vectorizer.transform(X[ho]))
    oov[ho] = [oov_ratio(t, committed.vectorizer) for t in X[ho]]
    pred = np.array(classes)[P.argmax(1)]
    correct = (pred == y).astype(float)
    sig = proba_signals(P)

    # sanity: committed model reproduces stored Day 6 confidences on all 516 rows
    Pall = committed.model.predict_proba(committed.vectorizer.transform(X))
    stored = {r["query_id"]: float(r["confidence"]) for r in csv.DictReader(open(HELD, newline="", encoding="utf-8-sig"))}
    dmax = max(abs(Pall[i].max() - stored[rows[i]["query_id"]]) for i in range(len(rows)))
    print(f"sanity: committed-model max-prob vs stored Day 6 confidence, max abs diff = {dmax:.2e}")
    ins = float((np.array(classes)[Pall[tr].argmax(1)] == y[tr]).mean())
    print(f"sanity: committed model IN-SAMPLE accuracy on non-held-out = {ins:.3f} (why fit rows use out-of-fold)")

    conflict = np.zeros(len(rows))
    banned = [""] * len(rows)
    for i, r in enumerate(rows):
        bp = find_banned_phrase(r["text"], r["true_regulator"])
        if pred[i] == "LOW" and bp:
            conflict[i], banned[i] = 1.0, bp

    print(f"fit-rows descriptive: conflicts among non-held-out OOF predictions = {int(conflict[tr].sum())}")

    # language score = OOF accuracy per language on non-held-out rows (fit rows only)
    lang_acc = {l: correct[[i for i in tr if rows[i]["language"] == l]].mean() for l in LANGS}
    lang_score = np.array([lang_acc[r["language"]] for r in rows])
    feats = np.column_stack([sig["max_prob"], sig["margin"], -sig["entropy"], -oov, -conflict,
                             *[[float(r["language"] == l) for r in rows] for l in LANGS]])
    scaler = StandardScaler().fit(feats[tr])
    comb = LogisticRegression(max_iter=1000).fit(scaler.transform(feats[tr]), correct[tr])
    comb_score = comb.predict_proba(scaler.transform(feats))[:, 1]

    scores = {"max_prob (raw confidence)": sig["max_prob"], "margin (top-2)": sig["margin"],
              "entropy": sig["entropy"], "oov_ratio": oov, "language": lang_score,
              "conflict (LOW + banned phrase)": conflict, "combined (logreg, all signals)": comb_score}
    yh = correct[ho]

    # ---- (c) AUROC: direction fitted on non-held-out OOF, scored on held-out
    dirs = {k: fit_direction(correct[tr], s[tr]) if k != "combined (logreg, all signals)" else 1
            for k, s in scores.items()}
    raw = "max_prob (raw confidence)"
    stats = {k: (lambda idx, s=dirs[k] * scores[k][ho]: auroc(yh[idx], s[idx])) for k in scores}
    boot = bootstrap(stats, len(ho))
    point = {k: auroc(yh, dirs[k] * scores[k][ho]) for k in scores}
    table = []
    for k in scores:
        lo, hi, _ = percentile_ci(boot[k])
        dlo, dhi, nv = percentile_ci(boot[k] - boot[raw]) if k != raw else (None, None, 0)
        table.append((k, dirs[k], point[k], lo, hi, dlo, dhi))
    table.sort(key=lambda t: -t[2])

    # ---- (a)/(b) calibration, all scored on held-out
    cc = {}
    for m in ("isotonic", "sigmoid"):
        cc[m] = CalibratedClassifierCV(base_pipeline(), method=m, cv=splits).fit(X[tr], y[tr])
    ctrl = np.mean([c.estimator.predict_proba(X[ho]) for c in cc["isotonic"].calibrated_classifiers_], axis=0)
    cands = {"raw (committed model)": P[ho], "5-fold ensemble, uncalibrated": ctrl,
             "isotonic (CV on non-held-out)": cc["isotonic"].predict_proba(X[ho]),
             "sigmoid (CV on non-held-out)": cc["sigmoid"].predict_proba(X[ho])}
    cal = {}
    for name, Q in cands.items():
        cal[name] = (Q.max(1), (np.array(classes)[Q.argmax(1)] == y[ho]).astype(float))
    ece_stats = {n: (lambda idx, c=c, k=k: ece(c[idx], k[idx])[0]) for n, (c, k) in cal.items()}
    eb = bootstrap(ece_stats, len(ho))
    rawn = "raw (committed model)"

    # ---- report (held-out only)
    print(f"\nHELD-OUT ONLY (n={len(ho)}; correct={int(yh.sum())}, accuracy={yh.mean():.3f}; "
          f"conflicts={int(conflict[ho].sum())})")
    e0, bins = ece(*cal[rawn])
    c0 = percentile_ci(eb[rawn])
    print(f"\n(a) Raw max-prob reliability: ECE={e0:.3f} [{f3(c0[0])}-{f3(c0[1])}]  mean conf={cal[rawn][0].mean():.3f}"
          f"  accuracy={cal[rawn][1].mean():.3f}")
    for lo, hi, n, mc, acc in bins:
        print(f"    [{lo:.1f},{hi:.1f}) n={n:<3} mean_conf={mc:.3f} acc={acc:.3f}")
    print("\n(b) Calibration, held-out ECE (lower is better) [95% CI] | paired diff vs raw [95% CI] | accuracy")
    cal_rows = []
    for n, (c, k) in cal.items():
        lo, hi, _ = percentile_ci(eb[n])
        dl, dh, _ = percentile_ci(eb[n] - eb[rawn]) if n != rawn else (None, None, 0)
        e, _ = ece(c, k)
        cal_rows.append((n, e, lo, hi, dl, dh, k.mean()))
        print(f"  {n:34} ECE={e:.3f} [{f3(lo)}-{f3(hi)}] | diff [{f3(dl)}, {f3(dh)}] | acc={k.mean():.3f}")
    print("\n(c) AUROC for 'classifier is correct' (direction fitted on non-held-out OOF; scored on held-out)")
    print(f"  {'signal':34} dir  AUROC [95% CI]        diff vs raw [95% CI]  beats raw?")
    beats = []
    for k, d, a, lo, hi, dl, dh in table:
        if len(np.unique(scores[k][ho])) == 1:
            print(f"  {k:34} {d:+d}   not evaluable: constant on held-out (conflicts={int(conflict[ho].sum())})")
            continue
        b = dl is not None and dl > 0
        beats.append(k) if b else None
        print(f"  {k:34} {d:+d}   {a:.3f} [{f3(lo)}-{f3(hi)}]   "
              + (f"[{f3(dl)}, {f3(dh)}]  {'YES' if b else 'no'}" if k != raw else "(baseline)"))
    better = [n for n, e, lo, hi, dl, dh, acc in cal_rows if dh is not None and dh < 0]
    print("\nVERDICT (b):", ("calibration with ECE gain whose paired CI is below 0: " + ", ".join(better)) if better
          else "no calibration method has a paired ECE-difference CI entirely below 0 vs raw (not shown to help).")
    print("VERDICT (c):", ("signals whose paired-AUROC gain over raw confidence has a CI above 0: " + ", ".join(beats))
          if beats else f"No signal beats raw confidence: every paired-AUROC difference vs raw max-prob has a 95% CI "
          f"that includes 0 or is negative (n={len(ho)} held-out rows).")

    cols = ["query_id", "original_id", "split", "language", "true_regulator", "true_risk", "pred_label", "correct",
            "prob_source", "max_prob", "margin", "entropy", "oov_ratio", "conflict", "banned_phrase",
            *[f"p_{c}" for c in classes], "max_prob_isotonic", "max_prob_sigmoid"]
    iso, sig_c = cands["isotonic (CV on non-held-out)"].max(1), cands["sigmoid (CV on non-held-out)"].max(1)
    hpos = {i: j for j, i in enumerate(ho)}
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for i, r in enumerate(rows):
            h = i in hpos
            w.writerow([r["query_id"], r["original_id"], "held_out" if h else "non_held_out", r["language"],
                        r["true_regulator"], y[i], pred[i], int(correct[i]), "committed" if h else "oof_fold",
                        f"{sig['max_prob'][i]:.6f}", f"{sig['margin'][i]:.6f}", f"{sig['entropy'][i]:.6f}",
                        f"{oov[i]:.4f}", int(conflict[i]), banned[i], *[f"{p:.6f}" for p in P[i]],
                        f"{iso[hpos[i]]:.6f}" if h else "", f"{sig_c[hpos[i]]:.6f}" if h else ""])
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)} ({len(rows)} rows; calibrated columns filled for held-out only)")


if __name__ == "__main__":
    main()
