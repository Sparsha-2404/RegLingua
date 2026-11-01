"""
DAY 10 (M2): model usefulness - TF-IDF baseline vs multilingual embedding vs 50/50 ensemble.

(a) macro-F1 change on hi / hi-en vs en; (b) classifier-blind count on the 305 Day 7 originals
(LOW despite a banned phrase, core.verification_policy read-only); (c) ECE on held-out;
(d) ensemble = mean of the two models' probabilities; (e) can each model safely auto-accept? Day 9
abstention thresholds are RE-SELECTED per model (same grid/constraints) on that model's own out-of-fold
probabilities for the leak-free train rows, then reported on held-out with CIs (no pass/fail, no gating).
Protocol = compare_risk_models.py: leak-free split (336 train / 104 held-out), C on train only,
paired cluster bootstrap seed 42, verdict by CI-excludes-0 only. Day 7 originals that are train
rows use 5-fold out-of-fold probabilities (never in-sample).

Run locally (real numbers):
    pip install sentence-transformers
    python scripts\\evaluate_model_usefulness.py
Smoke test only (NOT for reporting): python scripts\\evaluate_model_usefulness.py --fake-encoder
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from core.verification_policy import find_banned_phrase  # noqa: E402  (read-only)
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.embedding_classifier import EmbeddingLogisticClassifier  # noqa: E402
from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402
from services.evaluation.embedding_comparison import CachingEncoder, load_eval_rows  # noqa: E402
from services.evaluation.abstention_eval import decide_rows, frange, per_row_flags, summarize  # noqa: E402
from services.evaluation.model_comparison import accuracy, macro_f1  # noqa: E402
from services.evaluation.model_usefulness import (  # noqa: E402
    BETTER, NODIFF, SEED, WORSE, align_proba, banned_low_flags, bootstrap_dict, ci,
    abstention_boot_stats, abstention_rows, ece_top, ensemble_probs, probs_to_labels, recommend,
    reselect_operating_points, select_embedding_C, verdict,
)

D = REPO_ROOT / "data" / "evaluation"
PATHS = dict(eval_path=D / "risk_evidence_eval_m2.csv", held_path=D / "pipeline_retrieval_results_m2.csv",
             day7_path=D / "rewrite_eval_results_m2.csv", abst_path=D / "abstention_thresholds_m2.json",
             day9_ops_path=D / "abstention_operating_points_m2.csv",
             preds_path=D / "model_comparison_predictions_m2.csv", cost_path=D / "model_comparison_m2.csv")
LANGS = ("en", "hi", "hi-en")
CANDS = ("embedding", "ensemble")
FAKE_BANNER = "[FAKE-ENCODER SMOKE TEST - NOT FOR REPORTING]"
COLS = ["mode", "question", "slice", "metric", "candidate", "baseline", "candidate_value", "delta",
        "ci_low", "ci_high", "n", "verdict", "note"]


class FakeEncoder:
    model_name = "FAKE-hash-encoder"

    def encode(self, texts):
        return np.array([[b / 255 for b in hashlib.md5(t.encode()).digest()] for t in texts])


def _tfidf_proba(m, texts):
    return m.model.predict_proba(m.vectorizer.transform(list(texts)))


def run(raw_encoder, fake=False, out_dir=D, n_boot=2000, grid=(0.1, 1.0, 10.0, 100.0), verbose=True, **paths) -> dict:
    warnings.filterwarnings("ignore")
    P_ = {**PATHS, **paths}
    mode, pre = ("FAKE_SMOKE", "FAKE_SMOKE_") if fake else ("real", "")
    rows = load_eval_rows(Path(P_["eval_path"]), Path(P_["held_path"]))
    held = np.array([r["held_out"] for r in rows])
    X = np.array([r["norm"] for r in rows], dtype=object)
    raw = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([r["true_risk_level"] for r in rows], dtype=str)
    oid = np.array([r["original_id"] for r in rows], dtype=str)
    lang = np.array([r["language"] for r in rows], dtype=str)
    reg = np.array([r["true_regulator"] for r in rows], dtype=str)
    qid = np.array([r["query_id"] for r in rows], dtype=str)
    ho = np.where(held)[0]
    held_ids = set(oid[ho])
    tr = np.array([i for i in np.where(~held)[0] if oid[i] not in held_ids])
    assert not (held_ids & set(oid[tr]))

    enc = CachingEncoder(raw_encoder)
    best_C, _ = select_embedding_C(enc, X[tr], y[tr], oid[tr], grid)
    base = TfidfLogisticClassifier().fit(list(X[tr]), list(y[tr]))
    emb = EmbeddingLogisticClassifier(encoder=enc, C=best_C).fit(list(X[tr]), list(y[tr]))
    classes = list(emb.classes)
    assert list(base._classes) == classes, "class order mismatch"

    def probs(texts):
        Pb, Pe = _tfidf_proba(base, texts), emb.predict_proba_batch(list(texts))
        return {"baseline": Pb, "embedding": Pe, "ensemble": ensemble_probs(Pb, Pe)}

    # ---------------- held-out: (a) (c) (d) overall ----------------
    P = probs(X[ho])
    pred = {m: probs_to_labels(P[m], classes) for m in P}
    yh, lh, gh = y[ho], lang[ho], oid[ho]
    yidx = None

    def stat_fn(idx):
        out, f1, ec, ac = {}, {}, {}, {}
        yy, LL = yh[idx], lh[idx]
        for m in P:
            pm = pred[m][idx]
            ec[m] = ece_top(P[m][idx], yy, classes)
            ac[m] = accuracy(yy, pm)
            f1[(m, "all")] = macro_f1(yy, pm)
            for l in LANGS:
                k = LL == l
                f1[(m, l)] = macro_f1(yy[k], pm[k]) if k.any() else float("nan")
            out[f"{m}|ece_abs"] = ec[m]
            out[f"{m}|f1_abs|all"] = f1[(m, "all")]
            for l in LANGS:
                out[f"{m}|f1_abs|{l}"] = f1[(m, l)]
        for c in CANDS:
            out[f"{c}|f1|all"] = f1[(c, "all")] - f1[("baseline", "all")]
            out[f"{c}|acc|all"] = ac[c] - ac["baseline"]
            out[f"{c}|ece"] = ec[c] - ec["baseline"]
            d = {l: f1[(c, l)] - f1[("baseline", l)] for l in LANGS}
            for l in LANGS:
                out[f"{c}|f1|{l}"] = d[l]
            out[f"{c}|did|hi"] = d["hi"] - d["en"]
            out[f"{c}|did|hi-en"] = d["hi-en"] - d["en"]
        return out

    point = stat_fn(np.arange(len(ho)))
    draws = bootstrap_dict(stat_fn, gh, n_boot, SEED)
    nlang = {l: int((lh == l).sum()) for l in LANGS}

    out_rows, V = [], {c: {} for c in CANDS}

    def add(q, sl, met, cand, b, c, d, lo, hi, n, vd="", note=""):
        r = [mode, q, sl, met, cand, b, c, d, lo, hi, n, vd, note]
        out_rows.append([round(v, 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                         ("" if isinstance(v, (float, np.floating)) else v) for v in r])

    n = len(ho)
    for c in CANDS:
        for key, q, sl, met, hib, vkey in [
            ("f1|all", "overall", "all", "macro_f1", True, "overall_macro_f1"),
            ("acc|all", "overall", "all", "accuracy", True, "overall_accuracy"),
        ]:
            lo, hi = ci(draws[f"{c}|{key}"]); vd = verdict(lo, hi, hib)
            bv = point[f"baseline|f1_abs|all"] if met == "macro_f1" else accuracy(yh, pred["baseline"])
            cv = point[f"{c}|f1_abs|all"] if met == "macro_f1" else accuracy(yh, pred[c])
            add(q, sl, met, c, bv, cv, point[f"{c}|{key}"], lo, hi, n, vd)
            V[c][vkey] = vd
        for l in LANGS:
            lo, hi = ci(draws[f"{c}|f1|{l}"]); vd = verdict(lo, hi, True)
            add("a_language", l, "macro_f1", c, point[f"baseline|f1_abs|{l}"], point[f"{c}|f1_abs|{l}"],
                point[f"{c}|f1|{l}"], lo, hi, nlang[l], vd, "" if nlang[l] >= 15 else "small n: CI unreliable")
            V[c][f"lang_{l}"] = vd
        for l in ("hi", "hi-en"):
            lo, hi = ci(draws[f"{c}|did|{l}"]); vd = verdict(lo, hi, True)
            add("a_vs_english", l, "macro_f1_gain_minus_en_gain", c, "", "", point[f"{c}|did|{l}"], lo, hi, nlang[l], vd,
                "positive = candidate helps this language more than English")
            V[c][f"did_{l}"] = vd
    for m in P:
        lo, hi = ci(draws[f"{m}|ece_abs"])
        add("c_calibration", "all", "ece_top_label", m, "", point[f"{m}|ece_abs"], "", lo, hi, n, "", "lower is better")
    for c in CANDS:
        lo, hi = ci(draws[f"{c}|ece"]); vd = verdict(lo, hi, higher_is_better=False)
        add("c_calibration", "all", "ece_diff", c, point["baseline|ece_abs"], point[f"{c}|ece_abs"], point[f"{c}|ece"], lo, hi, n, vd,
            "lower is better")
        V[c]["ece"] = vd

    # ---------------- (b) Day 7 originals: LOW despite banned phrase (never in-sample) ----------------
    d7 = list(csv.DictReader(open(P_["day7_path"], newline="", encoding="utf-8-sig")))
    key = {X[i]: i for i in range(len(X))}
    d7_norm = [normalize_for_classifier(r["original_text"])["normalized_text"] for r in d7]
    d7_idx = [key.get(t, -1) for t in d7_norm]
    in_tr = {int(i): k for k, i in enumerate(tr)}
    # OOF probs for train rows (5-fold by original_id), fold models refit; unseen rows use the full leak-free models
    oof = {m: np.zeros((len(tr), len(classes))) for m in P}
    for a, b in GroupKFold(n_splits=min(5, len(set(oid[tr].tolist())))).split(X[tr], y[tr], oid[tr]):
        mb = TfidfLogisticClassifier().fit(list(X[tr][a]), list(y[tr][a]))
        me = EmbeddingLogisticClassifier(encoder=enc, C=best_C).fit(list(X[tr][a]), list(y[tr][a]))
        Pb = align_proba(_tfidf_proba(mb, X[tr][b]), mb._classes, classes)
        Pe = align_proba(me.predict_proba_batch(list(X[tr][b])), me.classes, classes)
        oof["baseline"][b], oof["embedding"][b], oof["ensemble"][b] = Pb, Pe, ensemble_probs(Pb, Pe)
    P7_full = probs(np.array(d7_norm, dtype=object))
    P7 = {m: np.array([oof[m][in_tr[i]] if i in in_tr else P7_full[m][k] for k, i in enumerate(d7_idx)]) for m in P}
    n_oof = sum(i in in_tr for i in d7_idx)
    lab7 = {m: probs_to_labels(P7[m], classes) for m in P}
    banned = [find_banned_phrase(r["original_text"], r["regulator"]) for r in d7]
    flags = {m: banned_low_flags(lab7[m], banned) for m in P}
    g7 = np.array([oid[i] if i >= 0 else f"curated:{r['row_id']}" for i, r in zip(d7_idx, d7)])
    nb = int(sum(b is not None for b in banned))

    def stat7(idx):
        o = {f"{m}|blind": flags[m][idx].sum() for m in P}
        for c in CANDS:
            o[f"{c}|blind_diff"] = flags[c][idx].sum() - flags["baseline"][idx].sum()
        return o

    d7draw = bootstrap_dict(stat7, g7, n_boot, SEED)
    for m in P:
        lo, hi = ci(d7draw[f"{m}|blind"])
        add("b_classifier_blind", "day7_originals", "low_despite_banned_phrase", m, "", int(flags[m].sum()), "", lo, hi, len(d7), "",
            f"of {nb} originals with a banned phrase (n_total={len(d7)}; {n_oof} train rows scored out-of-fold, rest unseen)")
    for c in CANDS:
        lo, hi = ci(d7draw[f"{c}|blind_diff"]); vd = verdict(lo, hi, higher_is_better=False)
        add("b_classifier_blind", "day7_originals", "blind_count_diff", c, int(flags["baseline"].sum()), int(flags[c].sum()),
            int(flags[c].sum() - flags["baseline"].sum()), lo, hi, len(d7), vd, f"only {nb} rows can be blind: CI is coarse; lower is better")
        V[c]["banned_low"] = vd
    stored = sum(int(r["before_label"] == "LOW" and b is not None) for r, b in zip(d7, banned))
    add("b_classifier_blind", "day7_originals", "low_despite_banned_phrase", "committed_tfidf_stored", "", stored, "", "", "", len(d7), "",
        "REFERENCE ONLY: committed model's Day 7 before_label; in-sample for its 412 training rows")

    # ---------------- (e) abstention: thresholds RE-SELECTED per model (Day 9 procedure) ----------------
    # Frozen Day 9 thresholds were tuned on the committed TF-IDF model's out-of-fold probabilities and do not
    # transfer to another model's probability scale, so every model gets its own selection: same grid, same
    # constraints, on ITS OWN out-of-fold probabilities for the leak-free train rows (never held-out). Held-out
    # is report-only (frozen after selection): coverage / accuracy-on-accepted / false-compliant with cluster-
    # bootstrap CIs plus the exact one-sided upper bound (ucb95) on the false-compliant rate among accepted LOW.
    # No pass/fail on held-out: it neither gates the recommendation nor is compared with a scaled budget.
    abst = json.load(open(P_["abst_path"], encoding="utf-8"))
    meta = abst["meta"]
    min_acc, max_ucb = meta["min_accuracy_on_accepted"], meta["max_fc_upper95_for_compliant_acceptance"]
    fractions = meta["false_compliant_budget_fraction"]
    a_vals, c_vals = frange(*meta["grid"]["abstain_floor"]), frange(*meta["grid"]["compliant_floor"])
    day9 = {}
    if Path(P_["day9_ops_path"]).exists():
        for r in csv.DictReader(open(P_["day9_ops_path"], newline="", encoding="utf-8-sig")):
            if r["group_type"] == "all":
                day9[r["operating_point"]] = r
    ntr = len(tr)
    e_summary = {m: {} for m in P}
    for m in P:
        sel_rows = abstention_rows(oof[m], classes, raw[tr], reg[tr], lang[tr], y[tr])
        held_rows = abstention_rows(P[m], classes, raw[ho], reg[ho], lh, yh, held_out=True)
        picks = reselect_operating_points(sel_rows, fractions, min_acc, max_ucb, a_vals, c_vals)
        for op, pk in picks.items():
            cfg, sel = pk["config"], pk["selection"]
            if cfg is None:
                add("e_abstention", op, "selection_feasible", m, "", "", "", "", "", ntr, "infeasible",
                    f"no (abstain_floor, compliant_floor) meets fc<={pk['budget']} and acc>={min_acc} on the {ntr} out-of-fold train rows")
                e_summary[m][op] = "infeasible"
                continue
            snote = f"re-selected on {m}'s own out-of-fold probs, {ntr} leak-free train rows; budget(train)={pk['budget']}"
            add("e_abstention", op, "abstain_floor", m, "", cfg.abstain_floor, "", "", "", ntr, "", snote)
            add("e_abstention", op, "compliant_floor", m, "", cfg.compliant_floor, "", "", "", ntr, "",
                snote + ("; compliant acceptance switched off" if cfg.compliant_floor >= 1.0 else ""))
            add("e_abstention", op, "selection_coverage", m, "", sel["coverage"], "", "", "", ntr, "", f"accepted={sel['accepted']}")
            add("e_abstention", op, "selection_accuracy_on_accepted", m, "", sel["accuracy_on_accepted"], "", "", "", sel["accepted"], "", f"min_acc={min_acc}")
            add("e_abstention", op, "selection_false_compliant", m, "", sel["false_compliant"], "", "", "", sel["low_accepted"], "",
                f"budget(train)={pk['budget']}; low_accepted={sel['low_accepted']}")
            add("e_abstention", op, "selection_fc_ucb95", m, "", sel["fc_upper95_among_low_accepted"], "", "", "", sel["low_accepted"],
                "safe" if pk["safe_for_compliant_acceptance"] else "not safe",
                f"exact one-sided 95% upper bound on false-compliant rate among accepted LOW; safe = low_accepted>0 and ucb<={max_ucb}")
            hflags = {k: np.array(v) for k, v in per_row_flags(held_rows, decide_rows(held_rows, cfg)).items()}
            hs = summarize({k: v.tolist() for k, v in hflags.items()})
            hb = bootstrap_dict(lambda idx, f=hflags: abstention_boot_stats(f, idx), gh, n_boot, SEED)
            (clo, chi), (alo, ahi), (flo, fhi) = ci(hb["cov"]), ci(hb["acc"]), ci(hb["fc"])
            hnote = f"held-out (report-only, frozen thresholds a={cfg.abstain_floor:.2f} c={cfg.compliant_floor:.2f}); 95% cluster-bootstrap CI by original_id"
            add("e_abstention", op, "heldout_coverage", m, "", hs["coverage"], "", clo, chi, n, "", hnote + f"; accepted={hs['accepted']} review={hs['review']} abstain={hs['abstain']}")
            add("e_abstention", op, "heldout_accuracy_on_accepted", m, "", hs["accuracy_on_accepted"] if hs["accuracy_on_accepted"] is not None else "",
                "", alo, ahi, hs["accepted"], "", hnote + ("" if hs["accepted"] else "; nothing accepted") + ("; CI undefined: too few resamples with accepted rows" if np.isnan(alo) and hs["accepted"] else ""))
            add("e_abstention", op, "heldout_false_compliant", m, "", hs["false_compliant"], "", flo, fhi, hs["low_accepted"], "",
                hnote + f"; count of accepted-LOW whose true label is not LOW, of {hs['low_accepted']} LOW accepted")
            add("e_abstention", op, "heldout_fc_ucb95", m, "", hs["fc_upper95_among_low_accepted"], "", "", "", hs["low_accepted"], "",
                "exact one-sided 95% upper bound on the false-compliant rate among accepted LOW (1.0 = nothing to bound)")
            e_summary[m][op] = (f"a={cfg.abstain_floor:.2f} c={cfg.compliant_floor:.2f} sel(cov={sel['coverage']:.2f} FC={sel['false_compliant']}/{sel['low_accepted']}LOW "
                                f"safe={pk['safe_for_compliant_acceptance']}) held(cov={hs['coverage']:.2f} FC={hs['false_compliant']}/{hs['low_accepted']}LOW)")
    for op in fractions:
        if op in day9:
            d = day9[op]
            add("e_abstention", op, "heldout_coverage", "day9_committed_tfidf_ref", "", d["coverage"], "", d.get("coverage_lo", ""), d.get("coverage_hi", ""), d["n"], "",
                "REFERENCE: stored Day 9 held-out (committed model trained on 412 rows, thresholds tuned on its out-of-fold probs)")
            add("e_abstention", op, "heldout_accuracy_on_accepted", "day9_committed_tfidf_ref", "", d["accuracy_on_accepted"], "", d.get("acc_lo", ""), d.get("acc_hi", ""), d["accept"])
            add("e_abstention", op, "heldout_false_compliant", "day9_committed_tfidf_ref", "", d["false_compliant"], "", d.get("fc_lo", ""), d.get("fc_hi", ""), d["low_accepted"])

    # ---------------- recommendation ----------------
    rec = recommend(V)   # abstention is reported per model in (e) but does not gate the recommendation
    cost_note = ""
    if Path(P_["cost_path"]).exists():
        for r in csv.DictReader(open(P_["cost_path"], encoding="utf-8")):
            if r["scope"] == "cost" and r["metric"] in ("latency_ms_mean", "saved_model_kb", "encoder_fp32_mb"):
                cost_note += f" {r['metric']}: base={r['baseline']} emb={r['embedding']};"
    note = ("rule: a candidate is default only if no slice is 'worse' and at least one key slice (overall macro-F1, ECE, banned-LOW) is "
            "'better'; else the deterministic TF-IDF baseline stays default. Abstention (e) is reported per model with re-selected "
            "thresholds and does not gate this rule."
            + (" Cost from model_comparison_m2.csv:" + cost_note if cost_note else ""))
    add("recommendation", "default", "model", rec["default"], "", "", "", "", "", "", "", note)
    add("recommendation", "fallback", "model", rec["fallback"], "", "", "", "", "", "", "", "")

    # ---------------- consistency with Prompt 3 predictions ----------------
    cons = ""
    if Path(P_["preds_path"]).exists():
        pr = {r["query_id"]: r for r in csv.DictReader(open(P_["preds_path"], encoding="utf-8"))}
        if pr and all(q in pr for q in qid[ho]):
            mb = np.mean([pr[q]["baseline_pred"] == p for q, p in zip(qid[ho], pred["baseline"])])
            me_ = np.mean([pr[q]["embedding_pred"] == p for q, p in zip(qid[ho], pred["embedding"])])
            cons = f"agreement with model_comparison_predictions_m2.csv: baseline={mb:.3f} embedding={me_:.3f}"
            add("consistency", "all", "agreement_with_prompt3_predictions", "baseline/embedding", mb, me_, "", "", "", n, "", cons)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pre}model_usefulness_m2.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(out_rows)
    if verbose:
        say = (lambda *a: print(FAKE_BANNER, *a)) if fake else print
        say(f"train={len(tr)} held-out={n} best_C={best_C} day7_originals={len(d7)} banned={nb} {cons}")
        for c in CANDS:
            say(f"{c}: " + "; ".join(f"{k}={v}" for k, v in V[c].items()))
        for m in P:
            for op, line in e_summary[m].items():
                say(f"abstention[{m}/{op}]: {line}")
        say(f"RECOMMENDATION default={rec['default']} fallback={rec['fallback']}")
        say("wrote", path)
    return {"rows": out_rows, "verdicts": V, "recommendation": rec, "path": str(path), "best_C": best_C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-encoder", action="store_true", help="smoke test; outputs prefixed FAKE_SMOKE_; NOT for reporting")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=D)
    a = ap.parse_args()
    if a.fake_encoder:
        run(FakeEncoder(), fake=True, out_dir=a.out_dir, n_boot=a.n_boot); return
    try:
        from services.embeddings.encoder import MultilingualEncoder  # M3-owned, read-only
        enc = MultilingualEncoder()
    except ImportError as exc:
        sys.exit(f"Real mode needs sentence-transformers ({exc}).\nRun locally:\npip install sentence-transformers\n"
                 "python scripts\\evaluate_model_usefulness.py")
    run(enc, out_dir=a.out_dir, n_boot=a.n_boot)


if __name__ == "__main__":
    main()
