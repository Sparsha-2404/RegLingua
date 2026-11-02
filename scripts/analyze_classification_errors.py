"""Day 12 (M2): error analysis on OUT-OF-FOLD predictions from the train pool only.

The frozen final_test rows (data/evaluation/final_split_m2.csv) are never loaded into any model, metric or table.
Group-aware stratified 5-fold CV (StratifiedGroupKFold, shuffle, seed 42) over leak groups (original_id merged with
near-duplicate links, exactly as make_final_split.py) so translations never straddle folds. Fresh models per fold;
committed artifacts are neither read nor overwritten. No label is changed.

Usage (repo root):  python scripts/analyze_classification_errors.py [--model tfidf|embedding] [--suspect-conf 0.70]
  embedding needs the multilingual encoder (Hugging Face) -> run locally; outputs get an _embedding suffix.
Writes: data/evaluation/classification_errors_m2.csv, suspected_label_errors_m2.csv, classification_error_analysis_m2.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.preprocessing import normalize_for_classifier  # noqa: E402
from services.evaluation import classification_metrics as cm  # noqa: E402
from services.evaluation import data_audit as da  # noqa: E402
from services.evaluation import error_analysis as ea  # noqa: E402

SEED, N_FOLDS, N_BOOT, TOP_PAIRS, N_EXAMPLES, N_TERMS = 42, 5, 2000, 3, 10, 12
D = REPO_ROOT / "data" / "evaluation"
EVAL, SPLIT, META = D / "risk_evidence_eval_m2.csv", D / "final_split_m2.csv", D / "final_split_m2.meta.json"


def load_train_pool():
    raw = EVAL.read_bytes()
    meta = json.loads(META.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != meta["eval_sha256"]:
        sys.exit("eval file differs from the one the frozen split was drawn on; refusing to continue")
    with open(EVAL, newline="", encoding="utf-8-sig") as f:
        ev = list(csv.DictReader(f))
    with open(SPLIT, newline="", encoding="utf-8") as f:
        split = {r["row_id"]: r["split"] for r in csv.DictReader(f)}
    assert [r["query_id"] for r in ev] == list(split), "split rows do not match eval rows"
    # leak groups over ALL rows (identical recipe to make_final_split.py), then keep the train pool only
    exact, near = da.find_duplicates([r["text"] for r in ev], 0.9)
    link = [(g[0], j) for g in exact for j in g[1:]] + [(i, j) for i, j, _ in near]
    groups = da.leak_groups([r["original_id"] for r in ev], link)
    conflicts = defaultdict(list)
    for i, j in link:
        if ev[i]["true_risk_level"] != ev[j]["true_risk_level"]:
            conflicts[ev[i]["query_id"]].append(ev[j]["query_id"])
            conflicts[ev[j]["query_id"]].append(ev[i]["query_id"])
    keep = [i for i, r in enumerate(ev) if split[r["query_id"]] == "train_pool"]
    pool = [ev[i] | {"group": groups[i]} for i in keep]
    assert not {r["query_id"] for r in pool} & {k for k, v in split.items() if v == "final_test"}
    assert len({r["group"] for r in pool} & {groups[i] for i, r in enumerate(ev) if split[r["query_id"]] == "final_test"}) == 0
    return pool, conflicts, meta


def oof_predict(pool, model: str):
    """Return (P[n,3] in cm.LABELS order, fold[n], contrib: {pair: {term: sum}}, term_docs: {pair: Counter})."""
    y = np.array([r["true_risk_level"] for r in pool])
    g = np.array([r["group"] for r in pool])
    texts = [normalize_for_classifier(r["text"])["normalized_text"] for r in pool]
    P, fold = np.zeros((len(pool), 3)), np.zeros(len(pool), dtype=int)
    contrib, term_docs = defaultdict(dict), defaultdict(Counter)
    for k, (tr, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(texts, y, g)):
        assert not set(g[tr]) & set(g[te])
        if model == "tfidf":
            clf = TfidfLogisticClassifier().fit([texts[i] for i in tr], list(y[tr]))
            X, classes = clf.vectorizer.transform([texts[i] for i in te]), list(clf.model.classes_)
            proba, coef, names = clf.model.predict_proba(X), clf.model.coef_, clf.vectorizer.get_feature_names_out()
        else:
            from services.classifier.embedding_classifier import EmbeddingLogisticClassifier
            clf = EmbeddingLogisticClassifier().fit([texts[i] for i in tr], list(y[tr]))
            proba, classes = clf.predict_proba_batch([texts[i] for i in te]), list(clf.classes())
        for a, i in enumerate(te):
            P[i] = [proba[a][classes.index(l)] for l in cm.LABELS]
            fold[i] = k
            pred = cm.LABELS[int(P[i].argmax())]
            if model == "tfidf" and pred != y[i]:
                pair = f"{y[i]}->{pred}"
                ea.add_contrib(contrib[pair], X[a], coef[classes.index(pred)], coef[classes.index(y[i])], names)
                x = X[a].tocoo()
                term_docs[pair].update(names[j] for j in x.col)
    return P, fold, contrib, term_docs


def fmt_rows(rows, cols):
    return ["  " + "  ".join(f"{str(c):<{w}}" for c, w in zip(r, cols)) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tfidf", "embedding"], default="tfidf")
    ap.add_argument("--suspect-conf", type=float, default=None,
                    help="min confidence for a suspected label error (default: 90th percentile of all OOF confidences)")
    args = ap.parse_args()
    NEG = re.compile(r"(?<!\w)(not|no|never|without|cannot|nahi|nahin|bina)(?!\w)|नहीं|नही|बिना")
    DEV = re.compile(r"[\u0900-\u097F]")
    suf = "" if args.model == "tfidf" else "_embedding"
    pool, conflicts, meta = load_train_pool()
    P, fold, contrib, term_docs = oof_predict(pool, args.model)

    recs = []
    for r, p, f in zip(pool, P, fold):
        pred, conf = cm.LABELS[int(p.argmax())], float(p.max())
        t = r["true_risk_level"]
        recs.append({"row_id": r["query_id"], "original_id": r["original_id"], "group": r["group"], "language": r["language"],
                     "regulator": r["true_regulator"], "text": r["text"], "true": t, "pred": pred, "conf": conf,
                     "p_true": float(p[cm.LABELS.index(t)]), "p_LOW": float(p[0]), "p_MEDIUM": float(p[1]),
                     "p_HIGH": float(p[2]), "fold": int(f), "kind": ea.error_kind(t, pred),
                     "pair": f"{t}->{pred}" if t != pred else "", "band": ea.confidence_band(conf)})
    yt, yp = [r["true"] for r in recs], [r["pred"] for r in recs]
    errs = [r for r in recs if r["kind"]]
    L = [f"CLASSIFICATION ERROR ANALYSIS (M2, Day 12) -- model={args.model}, OUT-OF-FOLD on the train pool only",
         f"data: risk_evidence_eval_m2.csv (516 rows; same rows as RegLingua_M1_FINAL.xlsx per the data audit) -> train_pool "
         f"{len(recs)} rows / {len({r['group'] for r in recs})} leak groups; final_test ({meta['test_rows']} rows) untouched",
         f"protocol: StratifiedGroupKFold(k={N_FOLDS}, shuffle, seed={SEED}) by leak group; bootstrap {N_BOOT}x resampling leak groups, "
         f"95% percentile CI; sklearn {sklearn.__version__}; eval sha256 {meta['eval_sha256'][:12]}", ""]
    ci = lambda m: cm.bootstrap_ci(yt, yp, m, n_boot=N_BOOT, seed=SEED, cluster_ids=[r["group"] for r in recs])
    L.append("1. OVERALL (OOF)")
    for name, m in cm.METRICS.items():
        c = ci(m)
        L.append(f"  {name}: {c['point']:.3f}  [{c['lo']:.3f}, {c['hi']:.3f}]")
    mat = cm.confusion_matrix(yt, yp)
    L += ["  confusion (rows=true, cols=pred; LOW MEDIUM HIGH): " + " | ".join(f"{l}: {row}" for l, row in zip(cm.LABELS, mat)),
          f"  errors: {len(errs)}/{len(recs)}", ""]

    pairs = ea.top_pairs(yt, yp)
    L.append("2. TOP CONFUSION PAIRS (true->pred)")
    L += fmt_rows([(f"{t}->{p}", n, f"{n / len(errs):.0%} of errors", f"{n / yt.count(t):.0%} of true {t}") for t, p, n in pairs],
                  (14, 5, 16, 18))
    kinds = Counter(r["kind"] for r in errs)
    n_risky, n_low = sum(t != "LOW" for t in yt), yt.count("LOW")
    L += ["", "3. FALSE-COMPLIANT vs FALSE-RISKY",
          f"  false_compliant (MEDIUM/HIGH predicted LOW): {kinds['false_compliant']}/{n_risky} = {kinds['false_compliant'] / n_risky:.1%} of risky rows"
          f"  (HIGH->LOW {Counter(r['pair'] for r in errs)['HIGH->LOW']}, MEDIUM->LOW {Counter(r['pair'] for r in errs)['MEDIUM->LOW']})",
          f"  false_risky (LOW predicted MEDIUM/HIGH): {kinds['false_risky']}/{n_low} = {kinds['false_risky'] / n_low:.1%} of LOW rows",
          f"  severity-only: under_severity HIGH->MEDIUM {kinds['under_severity']}, over_severity MEDIUM->HIGH {kinds['over_severity']}",
          "  WORSE for this project: false-compliant. A risky claim cleared as LOW is published/shipped un-reviewed, and the "
          "verifier (core/verification_policy.py check 1) passes a rewrite when the classifier says LOW, so one false LOW can also "
          "wave through a bad rewrite. A false-risky costs only analyst review time.", ""]

    L.append("4. ERROR RATE vs CONFIDENCE BAND (max predicted probability)")
    bt = ea.slice_table(recs, lambda r: r["band"], ea.BAND_ORDER, N_BOOT, SEED)
    L += fmt_rows([(b["slice"], b["n"], b["errors"], f"{b['error_rate']:.1%}" if b["n"] else "-",
                    f"[{b['lo']:.1%}, {b['hi']:.1%}]" if b["n"] else "-", f"FC {b['false_compliant']}") for b in bt], (10, 5, 7, 8, 16, 6))
    L += [f"  errors with conf>={t}: {sum(r['conf'] >= t for r in errs)}" for t in (0.6, 0.7, 0.8, 0.9)] + [""]

    L.append("5. ERRORS BY SLICE (error rate [95% CI]; FC = false-compliant / risky rows; FR = false-risky / LOW rows)")
    slices = {"language": lambda r: r["language"], "regulator": lambda r: r["regulator"], "true_risk": lambda r: r["true"],
              "language x regulator": lambda r: f"{r['language']}|{r['regulator']}"}
    for title, key in slices.items():
        L.append(f" by {title}")
        L += fmt_rows([(s["slice"], s["n"], s["errors"], f"{s['error_rate']:.1%}", f"[{s['lo']:.1%}, {s['hi']:.1%}]",
                        f"FC {s['false_compliant']}/{s['n_risky']}", f"FR {s['false_risky']}/{s['n_low']}")
                       for s in ea.slice_table(recs, key, None, N_BOOT, SEED)], (14, 5, 5, 7, 16, 9, 9))
    L.append("")

    top = [f"{t}->{p}" for t, p, _ in pairs[:TOP_PAIRS]]
    ana = TfidfLogisticClassifier().vectorizer.build_analyzer()
    L.append(f"6. MOST DISCRIMINATIVE TF-IDF TERMS BEHIND THE TOP {TOP_PAIRS} ERROR TYPES")
    if args.model != "tfidf":
        L.append("  n/a for the embedding model (no term features).")
    else:
        L.append("  score = sum over that type's misclassified docs of tfidf * (coef[pred]-coef[true]) from the fold model that "
                 "made the error; n_err = error docs containing the term; n_ok = correctly-classified docs of the TRUE class containing it.")
        sets = {r["query_id"]: set(ana(normalize_for_classifier(r["text"])["normalized_text"])) for r in pool}
        ok_by_class = {c: [sets[x["row_id"]] for x in recs if x["true"] == c and not x["kind"]] for c in cm.LABELS}
        for pair in top:
            t = pair.split("->")[0]
            best = sorted(contrib[pair].items(), key=lambda kv: -kv[1])[:N_TERMS]
            L.append(f" {pair} ({Counter(r['pair'] for r in errs)[pair]} errors)")
            L += fmt_rows([(w, f"{s:.3f}", term_docs[pair][w], sum(w in d for d in ok_by_class[t])) for w, s in best],
                          (28, 8, 6, 6))
    L.append("")
    cue = [bool(NEG.search(r["text"].lower())) for r in recs]
    L += ["6b. DIAGNOSTICS (descriptive, same OOF rows)",
          "  polarity-cue lexicon: not|no|never|without|cannot|nahi|nahin|bina|नहीं|नही|बिना"]
    for flag, name in ((True, "with cue"), (False, "no cue")):
        sub = [(r["kind"] != "") for r, c in zip(recs, cue) if c == flag]
        L.append(f"  error rate {name}: {sum(sub)}/{len(sub)} = {sum(sub) / len(sub):.1%}")
    for pair in top:
        t = pair.split("->")[0]
        e_c = [c for r, c in zip(recs, cue) if r["pair"] == pair]
        o_c = [c for r, c in zip(recs, cue) if r["true"] == t and not r["kind"]]
        L.append(f"  {pair}: cue in {sum(e_c)}/{len(e_c)} errors ({sum(e_c) / len(e_c):.0%}) vs {sum(o_c)}/{len(o_c)} correct true-{t} rows ({sum(o_c) / max(len(o_c), 1):.0%})")
    words = [w.strip(".,;:!?()\"'।") for r in pool if DEV.search(r["text"]) for w in r["text"].split() if DEV.search(w)]
    words = [w for w in words if len(w) > 1]
    split_w = sum(re.findall(r"(?u)\b\w\w+\b", w) != [w] for w in words)
    L += [f"  tokenization: default TfidfVectorizer token_pattern does not keep {split_w}/{len(words)} = {split_w / len(words):.0%} of "
          "Devanagari words as one token (vowel signs are not \\w) -- the fragments seen in section 6 (e.g. नह, गकर)."]
    L.append("")

    L.append(f"7. {N_EXAMPLES} REPRESENTATIVE EXAMPLES PER TOP ERROR TYPE (round-robin over language x regulator, most confident first, "
             "distinct original_id first)")
    rank = {}
    for pair in top:
        L.append(f" {pair}")
        for n, e in enumerate(ea.representative([r for r in errs if r["pair"] == pair], N_EXAMPLES), 1):
            rank[e["row_id"]] = (pair, n)
            L.append(f"  {n:>2}. {e['row_id']} {e['original_id']} {e['language']}/{e['regulator']} conf={e['conf']:.2f} "
                     f"| {e['text'][:110]}")
    L.append("")

    # ---- CSVs ----
    grp = defaultdict(list)
    for r in recs:
        grp[r["original_id"]].append(r)
    cols = ["row_id", "original_id", "language", "regulator", "text", "true_label", "pred_label", "error_pair", "error_kind",
            "confidence", "confidence_band", "p_LOW", "p_MEDIUM", "p_HIGH", "p_true_label", "fold", "model",
            "top_error_type_example_rank", "top_error_type"]
    with open(D / f"classification_errors_m2{suf}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in sorted(errs, key=lambda e: (-e["conf"], e["row_id"])):
            pr = rank.get(e["row_id"], ("", ""))
            w.writerow([e["row_id"], e["original_id"], e["language"], e["regulator"], e["text"], e["true"], e["pred"], e["pair"],
                        e["kind"], f"{e['conf']:.4f}", e["band"], f"{e['p_LOW']:.4f}", f"{e['p_MEDIUM']:.4f}", f"{e['p_HIGH']:.4f}",
                        f"{e['p_true']:.4f}", e["fold"], args.model, pr[1], pr[0]])
    thr = args.suspect_conf if args.suspect_conf is not None else float(np.percentile([r["conf"] for r in recs], 90))
    sus = [e for e in errs if e["conf"] >= thr]
    scols = ["row_id", "original_id", "language", "regulator", "text", "given_label", "model_label", "model_confidence",
             "p_given_label", "versions_in_group", "versions_misclassified_same_way", "near_dup_label_conflict_with",
             "m1_decision", "m1_notes"]
    with open(D / f"suspected_label_errors_m2{suf}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(scols)
        for e in sorted(sus, key=lambda e: (-e["conf"], e["row_id"])):
            sib = grp[e["original_id"]]
            w.writerow([e["row_id"], e["original_id"], e["language"], e["regulator"], e["text"], e["true"], e["pred"],
                        f"{e['conf']:.4f}", f"{e['p_true']:.4f}", len(sib), sum(s["pred"] == e["pred"] and s["true"] == e["true"] for s in sib),
                        ";".join(conflicts.get(e["row_id"], [])), "", ""])
    n_sib = sum(any(s["row_id"] != e["row_id"] and s["pred"] == e["pred"] and s["true"] == e["true"] for s in grp[e["original_id"]]) for e in sus)
    multi = [v for v in grp.values() if len(v) > 1]
    both = sum(all(x["kind"] and x["pred"] == v[0]["pred"] for x in v) for v in multi)
    L += ["8. SUSPECTED LABEL ERRORS (for M1 review; NO label changed)",
          f"  rule: OOF error with model confidence >= {thr:.3f} ({'user-set' if args.suspect_conf is not None else '90th percentile of all OOF confidences; max OOF confidence is %.3f, so a fixed 0.7/0.8 cut-off selects nothing' % max(r['conf'] for r in recs)}): {len(sus)} rows ({len({e['original_id'] for e in sus})} original_ids); "
          f"{n_sib} of them have a translation sibling mislabelled the same way.",
          f"  all-versions-wrong-the-same-way: {both}/{len(multi)} multi-version original_ids (any confidence)",
          f"  rows also in a near-duplicate pair with conflicting labels: {sum(bool(conflicts.get(e['row_id'])) for e in sus)}",
          "  caution: a high-confidence disagreement is a REVIEW CANDIDATE, not evidence the label is wrong -- the model can be confidently wrong "
          "(e.g. keyword shortcuts).", ""]
    (D / f"classification_error_analysis_m2{suf}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:4]), f"\nerrors={len(errs)}/{len(recs)} suspects={len(sus)}", sep="\n")
    print(f"wrote classification_errors_m2{suf}.csv, suspected_label_errors_m2{suf}.csv, classification_error_analysis_m2{suf}.txt")


if __name__ == "__main__":
    main()
