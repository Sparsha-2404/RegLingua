"""
DAY 10 (M2): final baseline (TF-IDF+LogReg) vs multilingual-embedding+LogReg comparison.

Protocol: leak-free split (train = non-held-out rows minus rows sharing an original_id with
a held-out row; test = Day 6 held-out rows), identical for both models; embedding C chosen by
5-fold GroupKFold on train rows only; paired cluster bootstrap (seed 42) for the macro-F1
difference; McNemar for accuracy; model size and mean per-prediction latency.

Real mode (needs sentence-transformers + Hugging Face download), run locally:
    pip install sentence-transformers
    python scripts\\compare_risk_models.py
Smoke test only (NOT for reporting): python scripts\\compare_risk_models.py --fake-encoder
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tempfile
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402
from services.classifier.embedding_classifier import EmbeddingLogisticClassifier, _lib_versions  # noqa: E402
from services.evaluation.embedding_comparison import CachingEncoder, load_eval_rows  # noqa: E402
from services.evaluation.model_comparison import (  # noqa: E402
    accuracy, confusion_matrix, macro_f1, mcnemar, metrics_by_language, metrics_by_regulator,
    paired_bootstrap_macro_f1_diff, per_class_f1,
)

EVAL = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
HELD = REPO_ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"
OUT_DIR = REPO_ROOT / "data" / "evaluation"
COMMITTED = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
C_GRID = (0.1, 1.0, 10.0, 100.0)
LABELS = ("HIGH", "MEDIUM", "LOW")
SEED, N_BOOT, WARMUP = 42, 2000, 3
COMMANDS = ("pip install sentence-transformers\n"
            "python scripts\\compare_risk_models.py")
FAKE_BANNER = "[FAKE-ENCODER SMOKE TEST - NOT FOR REPORTING]"


class FakeEncoder:
    """Deterministic hash embeddings. Smoke test only; carries no linguistic signal."""
    model_name = "FAKE-hash-encoder"

    def encode(self, texts):
        return np.array([[b / 255 for b in hashlib.md5(t.encode()).digest()] for t in texts])


def select_C(enc, X, y, groups, grid=C_GRID):
    scores = {}
    for C in grid:
        oof = np.empty(len(y), dtype=object)
        for a, b in GroupKFold(n_splits=min(5, len(set(groups.tolist())))).split(X, y, groups):
            m = EmbeddingLogisticClassifier(encoder=enc, C=C).fit(list(X[a]), list(y[a]))
            oof[b] = [p.risk_level.value for p in m.predict_batch(list(X[b]))]
        scores[C] = macro_f1(y, oof.astype(str), labels=LABELS)
    return max(grid, key=lambda c: (round(scores[c], 12), -c)), scores


def _latency_ms(model, texts):
    for t in texts[:WARMUP]:
        model.predict(t)
    ts = []
    for t in texts:
        s = time.perf_counter()
        model.predict(t)
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


def _encoder_size_mb(raw_encoder):
    """fp32 parameter-size estimate of the (unsaved) encoder; None if not introspectable."""
    try:
        return sum(p.numel() for p in raw_encoder.model.parameters()) * 4 / 1e6
    except Exception:  # noqa: BLE001
        return None


def run(raw_encoder, fake=False, eval_path=EVAL, held_path=HELD, out_dir=OUT_DIR, n_boot=N_BOOT,
        grid=C_GRID, committed_path=COMMITTED, verbose=True) -> dict:
    warnings.filterwarnings("ignore")
    mode = "FAKE_SMOKE" if fake else "real"
    pre = "FAKE_SMOKE_" if fake else ""
    say = (lambda *a: print(FAKE_BANNER, *a)) if fake else print

    rows = load_eval_rows(Path(eval_path), Path(held_path))
    held = np.array([r["held_out"] for r in rows])
    X = np.array([r["norm"] for r in rows], dtype=object)
    y = np.array([r["true_risk_level"] for r in rows], dtype=str)
    oid = np.array([r["original_id"] for r in rows], dtype=str)
    lang = np.array([r["language"] for r in rows], dtype=str)
    reg = np.array([r["true_regulator"] for r in rows], dtype=str)
    qid = np.array([r["query_id"] for r in rows], dtype=str)
    ho = np.where(held)[0]
    held_ids = set(oid[ho])
    tr_all = np.where(~held)[0]
    tr = np.array([i for i in tr_all if oid[i] not in held_ids])
    assert not (held_ids & set(oid[tr])) and not (set(tr) & set(ho)), "split leak"

    enc = CachingEncoder(raw_encoder)
    best_C, cv = select_C(enc, X[tr], y[tr], oid[tr], grid)
    base = TfidfLogisticClassifier().fit(list(X[tr]), list(y[tr]))
    emb = EmbeddingLogisticClassifier(encoder=enc, C=best_C).fit(list(X[tr]), list(y[tr]))
    pb_obj, pe_obj = base.predict_batch(list(X[ho])), emb.predict_batch(list(X[ho]))
    pb = np.array([p.risk_level.value for p in pb_obj], dtype=str)
    pe = np.array([p.risk_level.value for p in pe_obj], dtype=str)
    yt = y[ho]

    # ---- size + latency (artifacts in a temp dir; encoder weights are never saved) ----
    with tempfile.TemporaryDirectory() as td:
        bp, ep = Path(td) / "base.pkl", Path(td) / "emb.pkl"
        base.save(str(bp)); emb.save(str(ep))
        size_base_kb, size_emb_kb = bp.stat().st_size / 1024, ep.stat().st_size / 1024
        lat_emb_model = EmbeddingLogisticClassifier(encoder=raw_encoder).load(str(ep))  # RAW encoder: no cache
    lat_b = _latency_ms(base, list(X[ho]))
    lat_e = _latency_ms(lat_emb_model, list(X[ho]))
    lat_labels_match = all(lat_emb_model.predict(t).risk_level.value == l for t, l in zip(X[ho][:20], pe[:20]))
    committed_kb = Path(committed_path).stat().st_size / 1024 if Path(committed_path).exists() else None
    enc_mb = _encoder_size_mb(raw_encoder)

    # ---- metrics ----
    M = {}
    for name, p in (("baseline", pb), ("embedding", pe)):
        M[name] = {"accuracy": accuracy(yt, p), "macro_f1": macro_f1(yt, p, LABELS),
                   "per_class": per_class_f1(yt, p, LABELS), "cm": confusion_matrix(yt, p, LABELS),
                   "lang": metrics_by_language(yt, p, lang[ho], LABELS),
                   "reg": metrics_by_regulator(yt, p, reg[ho], LABELS)}
    boot = paired_bootstrap_macro_f1_diff(yt, pb, pe, oid[ho], n_boot=n_boot, seed=SEED)
    mc = mcnemar(yt, pb, pe)
    lang_boot = {}
    for L in ("en", "hi", "hi-en"):
        m = lang[ho] == L
        if m.sum() >= 2:
            lang_boot[L] = paired_bootstrap_macro_f1_diff(yt[m], pb[m], pe[m], oid[ho][m], n_boot=n_boot, seed=SEED)

    # ---- outputs ----
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{pre}model_comparison_predictions_m2.csv"
    sum_path = out_dir / f"{pre}model_comparison_m2.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "query_id", "original_id", "language", "regulator", "true_risk", "baseline_pred",
                    "baseline_conf", "embedding_pred", "embedding_conf", "baseline_correct", "embedding_correct"])
        for k, i in enumerate(ho):
            w.writerow([mode, qid[i], oid[i], lang[i], reg[i], yt[k], pb[k], round(pb_obj[k].confidence, 4),
                        pe[k], round(pe_obj[k].confidence, 4), int(pb[k] == yt[k]), int(pe[k] == yt[k])])

    R = []
    def add(scope, group, metric, n="", b="", e="", d="", lo="", hi="", note=""):
        R.append([mode, scope, group, metric, n, b, e, d, lo, hi, note])
    n = len(ho)
    for met in ("accuracy", "macro_f1"):
        add("overall", "all", met, n, M["baseline"][met], M["embedding"][met], M["embedding"][met] - M["baseline"][met])
    for c in LABELS:
        for met in ("precision", "recall", "f1"):
            add("per_class", c, met, M["baseline"]["per_class"][c]["support"],
                M["baseline"]["per_class"][c][met], M["embedding"]["per_class"][c][met],
                M["embedding"]["per_class"][c][met] - M["baseline"]["per_class"][c][met])
    for scope, key in (("language", "lang"), ("regulator", "reg")):
        for g, v in M["baseline"][key].items():
            ve = M["embedding"][key][g]
            for met in ("accuracy", "macro_f1"):
                add(scope, g, met, v["n"], v[met], ve[met], ve[met] - v[met], note="descriptive; no CI" if scope == "regulator" else "")
    add("difference", "all", "macro_f1_bootstrap", n, boot["macro_f1_a"], boot["macro_f1_b"], boot["diff"],
        boot["ci_low"], boot["ci_high"], boot["verdict"])
    for L, b in lang_boot.items():
        add("difference", L, "macro_f1_bootstrap", b["n"], b["macro_f1_a"], b["macro_f1_b"], b["diff"],
            b["ci_low"], b["ci_high"], b["verdict"])
    add("difference", "all", "mcnemar_p_exact", n, mc["only_a_correct"], mc["only_b_correct"], "", "", "",
        f"p={mc['p_exact']:.4g}; baseline/embedding columns = only-baseline/only-embedding correct")
    add("cost", "all", "saved_model_kb", "", size_base_kb, size_emb_kb, size_emb_kb - size_base_kb,
        note="embedding file = LR only; encoder weights NOT included")
    add("cost", "all", "encoder_fp32_mb", "", 0, enc_mb if enc_mb is not None else "", "", note="downloaded separately; blank if not introspectable")
    add("cost", "all", "latency_ms_mean", n, lat_b[0], lat_e[0], lat_e[0] - lat_b[0], note="single predict() calls, uncached, after warmup")
    add("cost", "all", "latency_ms_p95", n, lat_b[1], lat_e[1], lat_e[1] - lat_b[1])
    add("protocol", "all", "split", "", len(tr), len(ho), "", "", "",
        f"leak-free; baseline col=train rows, embedding col=held-out rows; dropped {len(tr_all)-len(tr)}; best_C={best_C}; "
        f"seed={SEED}; n_boot={n_boot}; encoder={enc.model_name}; libs={_lib_versions()}; committed_pkl_kb={committed_kb}")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "scope", "group", "metric", "n", "baseline", "embedding", "delta", "ci_low", "ci_high", "note"])
        for r in R:
            w.writerow([round(v, 4) if isinstance(v, (float, np.floating)) else v for v in r])

    result = {"mode": mode, "train": len(tr), "held_out": len(ho), "best_C": best_C, "bootstrap": boot,
              "mcnemar": mc, "metrics": M, "latency_ms_mean": {"baseline": lat_b[0], "embedding": lat_e[0]},
              "size_kb": {"baseline": size_base_kb, "embedding": size_emb_kb}, "latency_labels_match": lat_labels_match,
              "paths": [str(pred_path), str(sum_path)]}
    if verbose:
        say(f"train={len(tr)} (dropped {len(tr_all)-len(tr)}) held-out={len(ho)} best_C={best_C}")
        say(f"{'':<14}{'baseline':>10}{'embedding':>11}")
        for met in ("accuracy", "macro_f1"):
            say(f"{met:<14}{M['baseline'][met]:>10.3f}{M['embedding'][met]:>11.3f}")
        say(f"macro-F1 diff (emb-base) = {boot['diff']:+.3f}  95% CI [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]  -> {boot['verdict']}")
        say(f"McNemar exact p={mc['p_exact']:.4g} (only-base-correct={mc['only_a_correct']}, only-emb-correct={mc['only_b_correct']})")
        say(f"size KB base/emb={size_base_kb:.1f}/{size_emb_kb:.1f}; latency ms mean base/emb={lat_b[0]:.2f}/{lat_e[0]:.2f}")
        say("wrote:", pred_path, sum_path)
    return result


def main():
    ap = argparse.ArgumentParser(epilog="Real mode commands:\n" + COMMANDS,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake-encoder", action="store_true",
                    help="smoke test with a fake encoder; outputs prefixed FAKE_SMOKE_ and NOT for reporting")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    if a.fake_encoder:
        run(FakeEncoder(), fake=True, n_boot=a.n_boot, out_dir=a.out_dir)
        return
    try:
        from services.embeddings.encoder import MultilingualEncoder  # M3-owned, read-only
        enc = MultilingualEncoder()
    except ImportError as exc:
        sys.exit(f"Real mode needs sentence-transformers ({exc}).\nRun locally:\n{COMMANDS}")
    run(enc, fake=False, n_boot=a.n_boot, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
