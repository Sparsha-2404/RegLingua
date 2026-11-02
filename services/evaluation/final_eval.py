"""Day 12 (M2): final risk-classification evaluation core (used by scripts/run_m2_final_evaluation.py).

Protocol (all pinned by tests/unit/test_day12_determinism.py):
  * The frozen split (data/evaluation/final_split_m2.csv: train_pool / final_test, made by leak group =
    original_id + near-duplicate links) is READ, never re-drawn. final_test labels are not touched until the
    single final scoring step; cross-validation, error analysis inputs and any tuning use the train pool only.
  * Train pool: repeated StratifiedGroupKFold (5 folds x 3 repeats, random_state = seed + repeat, groups = leak
    groups, stratified by risk label). Models are re-fitted per fold: majority class, banned-phrase rule (fixed),
    TF-IDF+LogReg (never the committed .pkl) and, if an encoder is supplied, the embedding classifier (C picked
    by an inner grouped CV on that fold's training rows only).
  * Final: fit on the whole train pool once, score final_test once (guarded; a second call raises).
  * CIs: percentile 95% cluster bootstrap (whole leak groups resampled), seed fixed, identical resamples for
    every model so differences are paired. Metrics follow services/evaluation/classification_metrics.py
    (cross-checked against it in the tests). No timestamps or paths in outputs -> byte-identical re-runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.preprocessing import normalize_for_classifier
from services.evaluation import data_audit as da
from services.evaluation.classification_metrics import LABELS, mcnemar
from services.evaluation.final_split import verify_split
from services.evaluation.model_comparison import predict_labels
from services.evaluation.sanity_baselines import BannedPhraseRuleBaseline, MajorityClassBaseline

SEED, N_BOOT, FOLDS, REPEATS, SMALL_N = 42, 2000, 5, 3, 15
IDX = {l: i for i, l in enumerate(LABELS)}
CLASS_METRICS = ("precision", "recall", "f1")
NAMES = ["accuracy", "macro_f1", "false_compliant_rate"] + [f"{c}_{m}" for c in LABELS for m in CLASS_METRICS]
HEADLINE = ("accuracy", "macro_f1", "false_compliant_rate")
LOWER_IS_BETTER = {"false_compliant_rate"}
BASE_MODELS = ("majority_class", "banned_phrase_rule", "tfidf_logreg")
EMB = "embedding_logreg"
C_GRID = (0.1, 1.0, 10.0, 100.0)
COLS = ["phase", "scope", "group", "model", "comparator", "metric", "value", "ci_lo", "ci_hi", "n", "n_valid", "p_value",
        "verdict", "note"]
NAN = float("nan")


# ------------------------------------------------------------------ metrics from a 3x3 confusion matrix
def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(y * 3 + p, minlength=9).reshape(3, 3)


def metrics_from_confusion(C: np.ndarray) -> np.ndarray:
    """Vector in NAMES order; same conventions as classification_metrics (zero_division=0; macro over classes
    present in truth or prediction; false-compliant = truly MEDIUM/HIGH predicted LOW, NaN if none)."""
    n = C.sum()
    tp = np.diag(C).astype(float)
    row, col = C.sum(1).astype(float), C.sum(0).astype(float)
    prec = np.divide(tp, col, out=np.zeros(3), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros(3), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    present = (row + col) > 0
    risky = C[1:, :].sum()
    out = [tp.sum() / n if n else NAN, f1[present].mean() if n else NAN, C[1:, 0].sum() / risky if risky else NAN]
    for i in range(3):
        out += [prec[i], rec[i], f1[i]]
    return np.array(out, dtype=float)


def _ci(v: np.ndarray) -> tuple[float, float, int]:
    v = v[~np.isnan(v)]
    if v.size == 0:
        return NAN, NAN, 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)


def cluster_bootstrap(y: np.ndarray, preds: dict, clusters: Sequence[str], n_boot: int, seed: int):
    """Resample whole clusters; every model sees the same resamples. Returns (point, samples):
    point[model] -> vector over NAMES, samples[model] -> (n_boot, len(NAMES))."""
    cl = np.asarray(clusters, dtype=object)
    members = [np.where(cl == u)[0] for u in sorted(set(cl.tolist()))]
    k, rng = len(members), np.random.default_rng(seed)
    point = {m: metrics_from_confusion(confusion(y, p)) for m, p in preds.items()}
    samples = {m: np.empty((n_boot, len(NAMES))) for m in preds}
    for b in range(n_boot):
        ix = np.concatenate([members[j] for j in rng.integers(0, k, k)])
        for m, p in preds.items():
            samples[m][b] = metrics_from_confusion(confusion(y[ix], p[ix]))
    return point, samples


def verdict(lo: float, hi: float, lower_better: bool = False) -> str:
    """Project rule: improvement/degradation only if the CI of (A - B) excludes 0."""
    if math.isnan(lo) or math.isnan(hi):
        return "undetermined"
    if lo > 0:
        return "degradation" if lower_better else "improvement"
    if hi < 0:
        return "improvement" if lower_better else "degradation"
    return "no significant difference"


def _f(x: float) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.6f}"


class Rows:
    def __init__(self) -> None:
        self.data: list = []

    def add(self, phase, scope, group, model, metric, value, lo="", hi="", n="", n_valid="", comparator="", p="", vd="", note=""):
        self.data.append([phase, scope, group, model, comparator, metric, _f(value) if value != "" else "", _f(lo) if lo != "" else "",
                          _f(hi) if hi != "" else "", n, n_valid, _f(p) if p != "" else "", vd, note])


def evaluate_phase(rows: Rows, confs: list, phase: str, y: np.ndarray, preds: dict, lang: np.ndarray, reg: np.ndarray,
                   clusters: np.ndarray, n_boot: int, seed: int, models: Sequence[str]) -> None:
    """overall + per-class + per-language + per-regulator metrics with CIs, paired model comparisons, McNemar."""
    def block(scope, group, mask, metrics, note_small=True):
        yy, pp, cc = y[mask], {m: preds[m][mask] for m in models}, clusters[mask]
        if len(yy) == 0:
            return None
        point, samp = cluster_bootstrap(yy, pp, cc, n_boot, seed)
        small = f"small n (<{SMALL_N}): CI unreliable" if note_small and len(yy) < SMALL_N else ""
        for m in models:
            for name in metrics:
                j = NAMES.index(name)
                lo, hi, nv = _ci(samp[m][:, j])
                rows.add(phase, scope, group, m, name, point[m][j], lo, hi, len(yy), nv, note=small)
        return point, samp

    allm = np.ones(len(y), dtype=bool)
    res = block("overall", "all", allm, NAMES[:3])
    point, samp = res
    for m in models:                                         # per-class P/R/F1 from the same bootstrap draws
        for c in LABELS:
            for met in CLASS_METRICS:
                name, j = f"{c}_{met}", NAMES.index(f"{c}_{met}")
                lo, hi, nv = _ci(samp[m][:, j])
                rows.add(phase, "class", c, m, met, point[m][j], lo, hi, int((y == IDX[c]).sum()), nv, note=f"n = true support of {c}")
    _paired(rows, phase, "overall", "all", point, samp, models, len(y))
    for scope, arr in (("language", lang), ("regulator", reg)):
        for g in sorted(set(arr.tolist())):
            mask = arr == g
            r = block(scope, g, mask, NAMES[:3])
            if r:
                _paired(rows, phase, scope, g, r[0], r[1], models, int(mask.sum()))
    for m in models:                                          # overall confusion matrix
        C = confusion(y, preds[m])
        for t in LABELS:
            for p in LABELS:
                confs.append([phase, m, t, p, int(C[IDX[t], IDX[p]])])
    for a, b in _pairs(models):                               # reference only: ignores clustering
        mc = mcnemar(_s(y), _s(preds[a]), _s(preds[b]))
        rows.add(phase, "paired_mcnemar", "all", a, "exact_p", mc["exact_p"], n=len(y), comparator=b, p=mc["exact_p"],
                 note=f"b={mc['b']} ({a} right, {b} wrong) c={mc['c']}; row-level, ignores group clustering: reference only")


def _s(a: np.ndarray) -> list:
    return [LABELS[i] for i in a]


def _pairs(models: Sequence[str]) -> list:
    out = []
    if "tfidf_logreg" in models:
        out += [("tfidf_logreg", "banned_phrase_rule"), ("tfidf_logreg", "majority_class")]
    if EMB in models:
        out += [(EMB, "tfidf_logreg"), (EMB, "banned_phrase_rule")]
    return out


def _paired(rows: Rows, phase: str, scope: str, group: str, point: dict, samp: dict, models, n: int) -> None:
    for a, b in _pairs(models):
        for name in HEADLINE:
            j = NAMES.index(name)
            d = samp[a][:, j] - samp[b][:, j]
            lo, hi, nv = _ci(d)
            v = d[~np.isnan(d)]
            p = min(1.0, 2 * min(float((v <= 0).mean()), float((v >= 0).mean()))) if v.size else NAN
            rows.add(phase, f"paired_{scope}", group, a, name, point[a][j] - point[b][j], lo, hi, n, nv, comparator=b, p=p,
                     vd=verdict(lo, hi, name in LOWER_IS_BETTER),
                     note="difference A - B; paired cluster bootstrap on identical resamples; p is a bootstrap estimate"
                          + ("; lower is better" if name in LOWER_IS_BETTER else ""))


def en_hi_pairs(oid: Sequence[str], lang: Sequence[str]) -> list:
    """[(row index en, row index hi)] for original_ids with exactly one en and one hi row."""
    by: dict = {}
    for i, (o, l) in enumerate(zip(oid, lang)):
        by.setdefault(o, {}).setdefault(l, []).append(i)
    return [(v["en"][0], v["hi"][0]) for o, v in sorted(by.items()) if len(v.get("en", [])) == 1 and len(v.get("hi", [])) == 1]


def matched_en_hi(rows: Rows, phase: str, y: np.ndarray, preds: dict, pair_en: np.ndarray, pair_hi: np.ndarray, pair_cluster: np.ndarray,
                  mc_sel: np.ndarray, n_boot: int, seed: int, models: Sequence[str], note: str) -> None:
    """Matched English-vs-Hindi comparison on original_id pairs: accuracy / macro-F1 for each language and
    their difference (en - hi) with a pair-cluster bootstrap CI, plus an exact McNemar test on discordant pairs
    (en right & hi wrong vs en wrong & hi right) taken from the pairs selected by ``mc_sel``."""
    cl = np.asarray(pair_cluster, dtype=object)
    members = [np.where(cl == u)[0] for u in sorted(set(cl.tolist()))]
    rng, k = np.random.default_rng(seed), len(members)
    draws = [np.concatenate([members[j] for j in rng.integers(0, k, k)]) for _ in range(n_boot)]
    for m in models:
        p = preds[m]
        met = lambda sel: (metrics_from_confusion(confusion(y[pair_en[sel]], p[pair_en[sel]])),       # noqa: E731
                           metrics_from_confusion(confusion(y[pair_hi[sel]], p[pair_hi[sel]])))
        all_sel = np.arange(len(pair_en))
        (e0, h0) = met(all_sel)
        samples = np.array([[*(x[:2] for x in met(sel))] for sel in draws]).reshape(n_boot, 2, 2)    # [draw, lang, (acc, f1)]
        for j, name in ((0, "accuracy"), (1, "macro_f1")):
            for lab, pt, col in (("en", e0[j], 0), ("hi", h0[j], 1)):
                lo, hi, nv = _ci(samples[:, col, j])
                rows.add(phase, "paired_en_hi", lab, m, name, pt, lo, hi, len(all_sel), nv, note=note)
            d = samples[:, 0, j] - samples[:, 1, j]
            lo, hi, nv = _ci(d)
            rows.add(phase, "paired_en_hi", "en_minus_hi", m, name, e0[j] - h0[j], lo, hi, len(all_sel), nv, vd=verdict(lo, hi), note=note +
                     "; difference en - hi; CI excludes 0 -> language gap")
        en_ok = (p[pair_en[mc_sel]] == y[pair_en[mc_sel]])
        hi_ok = (p[pair_hi[mc_sel]] == y[pair_hi[mc_sel]])
        b, c = int((en_ok & ~hi_ok).sum()), int((~en_ok & hi_ok).sum())
        nd = b + c
        pv = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(min(b, c) + 1)) / 2 ** nd)
        rows.add(phase, "paired_en_hi", "mcnemar_exact", m, "p_value", pv, n=int(mc_sel.sum() if mc_sel.dtype == bool else len(mc_sel)), p=pv,
                 note=f"discordant pairs: en right/hi wrong={b}, en wrong/hi right={c}; exact two-sided binomial")


# ------------------------------------------------------------------ models
class Ctx:
    def __init__(self, raw, norm, ystr, reg, groups, seed, encoder):
        self.raw, self.norm, self.ystr, self.reg, self.groups, self.seed, self.enc = raw, norm, ystr, reg, groups, seed, encoder
        self.rule = BannedPhraseRuleBaseline()
        self.chosen_C: list = []


def select_C(ctx: Ctx, a: np.ndarray, grid=C_GRID) -> float:
    """Inner grouped CV over C on the rows ``a`` ONLY (never test rows); ties -> smaller C."""
    from services.classifier.embedding_classifier import EmbeddingLogisticClassifier

    y = np.array([IDX[s] for s in ctx.ystr[a]])
    scores = {}
    for c in grid:
        oof = np.zeros(len(a), dtype=int)
        for tr, va in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=ctx.seed).split(a, y, ctx.groups[a]):
            m = EmbeddingLogisticClassifier(encoder=ctx.enc, C=c).fit(list(ctx.norm[a][tr]), list(ctx.ystr[a][tr]))
            oof[va] = [IDX[s] for s in predict_labels(m, list(ctx.norm[a][va]))]
        scores[c] = float(metrics_from_confusion(confusion(y, oof))[1])
    return max(grid, key=lambda c: (round(scores[c], 12), -c))


def fit_predict(name: str, ctx: Ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if name == "majority_class":
        out = MajorityClassBaseline().fit(list(ctx.ystr[a])).predict(list(ctx.raw[b]))
    elif name == "banned_phrase_rule":
        out = ctx.rule.predict(list(ctx.raw[b]), list(ctx.reg[b]))                 # TRUE regulator (oracle input)
    elif name == "tfidf_logreg":
        out = predict_labels(TfidfLogisticClassifier().fit(list(ctx.norm[a]), list(ctx.ystr[a])), list(ctx.norm[b]))
    elif name == EMB:
        from services.classifier.embedding_classifier import EmbeddingLogisticClassifier

        c = select_C(ctx, a)
        ctx.chosen_C.append(c)
        out = predict_labels(EmbeddingLogisticClassifier(encoder=ctx.enc, C=c).fit(list(ctx.norm[a]), list(ctx.ystr[a])), list(ctx.norm[b]))
    else:
        raise ValueError(name)
    return np.array([IDX[s] for s in out], dtype=int)


class OnceGuard:
    """The frozen final_test may be scored exactly once per run."""

    def __init__(self, name: str) -> None:
        self.name, self.used = name, False

    def use(self) -> None:
        if self.used:
            raise RuntimeError(f"{self.name} was already scored; the frozen test set may be used exactly once")
        self.used = True


# ------------------------------------------------------------------ driver
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, header: Sequence[str], data: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(data)


def load_inputs(eval_path: Path, split_path: Path) -> dict:
    ev = list(csv.DictReader(open(eval_path, newline="", encoding="utf-8-sig")))
    sp = {r["row_id"]: r for r in csv.DictReader(open(split_path, newline="", encoding="utf-8"))}
    if set(sp) != {r["query_id"] for r in ev}:
        raise ValueError("split rows and eval rows do not cover the same query_ids")
    oid = [r["original_id"] for r in ev]
    exact, near = da.find_duplicates([r["text"] for r in ev], 0.9)
    link = [(g[0], j) for g in exact for j in g[1:]] + [(i, j) for i, j, _ in near]
    groups = da.leak_groups(oid, link)
    split_rows = [{"row_id": r["query_id"], "original_id": r["original_id"], "split": sp[r["query_id"]]["split"]} for r in ev]
    ver = verify_split(split_rows, groups)
    if ver["original_ids_straddling"] or ver["leak_groups_straddling"]:
        raise RuntimeError(f"frozen split leaks across train/test: {ver}")
    return {"ev": ev, "groups": np.array(groups, dtype=object), "split": np.array([r["split"] for r in split_rows]), "verification": ver}


def run_final_evaluation(eval_path: Path, split_path: Path, out_dir: Path, results_dir: Path, seed: int = SEED, n_boot: int = N_BOOT,
                         folds: int = FOLDS, repeats: int = REPEATS, encoder: Optional[Any] = None, encoder_name: str = "",
                         embedding_note: str = "", expected_eval_sha: Optional[str] = None, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    eval_path, split_path, out_dir, results_dir = map(Path, (eval_path, split_path, out_dir, results_dir))
    if expected_eval_sha and _sha(eval_path) != expected_eval_sha:
        raise RuntimeError("eval file does not match the checksum recorded with the frozen split")
    inp = load_inputs(eval_path, split_path)
    ev = inp["ev"]
    raw = np.array([r["text"] for r in ev], dtype=object)
    norm = np.array([normalize_for_classifier(r["text"])["normalized_text"] for r in ev], dtype=object)
    ystr = np.array([r["true_risk_level"] for r in ev], dtype=object)
    y = np.array([IDX[s] for s in ystr], dtype=int)
    lang = np.array([r["language"] for r in ev], dtype=object)
    reg = np.array([r["true_regulator"] for r in ev], dtype=object)
    oid = np.array([r["original_id"] for r in ev], dtype=object)
    qid = np.array([r["query_id"] for r in ev], dtype=object)
    groups = inp["groups"]
    tr = np.where(inp["split"] == "train_pool")[0]
    te = np.where(inp["split"] == "final_test")[0]
    models = list(BASE_MODELS) + ([EMB] if encoder is not None else [])
    ctx = Ctx(raw, norm, ystr, reg, groups, seed, None)
    if encoder is not None:
        from services.evaluation.embedding_comparison import CachingEncoder

        ctx.enc = CachingEncoder(encoder)

    # ---- phase 1: repeated group-stratified CV on the TRAIN POOL only (final_test is not read here)
    oof = {m: np.full((repeats, len(tr)), -1, dtype=int) for m in models}
    fold_id = np.zeros((repeats, len(tr)), dtype=int)
    for r in range(repeats):
        for f, (a, b) in enumerate(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r).split(tr, y[tr], groups[tr])):
            fold_id[r, b] = f
            for m in models:
                oof[m][r, b] = fit_predict(m, ctx, tr[a], tr[b])
    assert all((v >= 0).all() for v in oof.values()), "every train-pool row must get an out-of-fold prediction"
    cv_C = list(ctx.chosen_C)

    rows, confs = Rows(), []
    pooled = lambda arr: arr.reshape(-1)                                                         # noqa: E731  (repeat-major)
    yy, ll, rr, gg = (np.tile(x[tr], repeats) for x in (y, lang, reg, groups))
    pp = {m: pooled(oof[m]) for m in models}
    evaluate_phase(rows, confs, "cv_oof", yy, pp, ll, rr, gg, n_boot, seed, models)
    for m in models:                                                                             # spread across the repeats
        for r in range(repeats):
            v = metrics_from_confusion(confusion(y[tr], oof[m][r]))
            for j, name in enumerate(HEADLINE):
                rows.add("cv_oof", "per_repeat", f"repeat{r}", m, name, v[j], n=len(tr), note="point estimate, one CV repeat")
    pe = np.array([(a + r * len(tr), b + r * len(tr)) for r in range(repeats) for a, b in en_hi_pairs(oid[tr], lang[tr])]).reshape(-1, 2)
    if len(pe):
        pair_cl = np.array([oid[tr][int(a) % len(tr)] for a in pe[:, 0]], dtype=object)
        mc = np.array([i < len(pe) // repeats for i in range(len(pe))])                          # McNemar: repeat 0 only
        matched_en_hi(rows, "cv_oof", yy, pp, pe[:, 0], pe[:, 1], pair_cl, mc, n_boot, seed, models,
                      f"{len(pe) // repeats} en/hi original_id pairs x {repeats} CV repeats pooled; McNemar uses repeat 0 only")
    oof_rows = [[m, r, int(fold_id[r, i]), qid[tr[i]], oid[tr[i]], lang[tr[i]], reg[tr[i]], ystr[tr[i]], LABELS[oof[m][r, i]]]
                for m in models for r in range(repeats) for i in range(len(tr))]

    # ---- phase 2: fit on the whole train pool once, score the frozen final_test ONCE
    guard = OnceGuard("final_test")
    guard.use()                                                                                  # a second use raises
    final = {}
    ctx.chosen_C = []
    for m in models:
        final[m] = fit_predict(m, ctx, tr, te)
    final_C = list(ctx.chosen_C)
    evaluate_phase(rows, confs, "final_test", y[te], final, lang[te], reg[te], groups[te], n_boot, seed, models)
    pe_t = np.array(en_hi_pairs(oid[te], lang[te])).reshape(-1, 2)
    if len(pe_t):
        matched_en_hi(rows, "final_test", y[te], final, pe_t[:, 0], pe_t[:, 1], oid[te][pe_t[:, 0]], np.ones(len(pe_t), dtype=bool), n_boot, seed,
                      models, f"{len(pe_t)} en/hi original_id pairs in final_test")

    out_dir, results_dir = Path(out_dir), Path(results_dir)
    p_res, p_conf, p_oof = (out_dir / n for n in ("final_eval_results_m2.csv", "final_eval_confusions_m2.csv", "final_eval_oof_predictions_m2.csv"))
    _write(p_res, COLS, rows.data)
    _write(p_conf, ["phase", "model", "true_label", "pred_label", "count"], confs)
    _write(p_oof, ["model", "repeat", "fold", "row_id", "original_id", "language", "regulator", "true_label", "pred_label"], oof_rows)

    def _v(pkg):
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    split_hash = hashlib.sha256("\n".join(f"{q},{s}" for q, s in sorted(zip(qid.tolist(), inp["split"].tolist()))).encode()).hexdigest()
    manifest = {
        "script": "scripts/run_m2_final_evaluation.py", "seed": seed, "n_boot": n_boot,
        "cv": {"folds": folds, "repeats": repeats, "splitter": "StratifiedGroupKFold(shuffle=True, random_state=seed+repeat)",
               "groups": "leak groups (original_id + near-duplicate links)", "stratified_by": "true_risk_level"},
        "bootstrap": "percentile 95% CI, whole leak groups resampled, identical resamples across models",
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": _v("scikit-learn"),
                             "sentence-transformers": _v("sentence-transformers")},
        "data_files_sha256": {eval_path.name: _sha(eval_path), split_path.name: _sha(split_path)},
        "split": {"hash_sha256": split_hash, "train_pool_rows": int(len(tr)), "final_test_rows": int(len(te)),
                  "verification": inp["verification"], "final_test_scored_once": True},
        "models": models,
        "model_notes": {"banned_phrase_rule": "uses the row's TRUE regulator (oracle input); English-only phrase lists",
                        "tfidf_logreg": "re-fitted per fold and on the train pool; committed .pkl never loaded",
                        "majority_class": "fit on training-fold labels only"},
        "embedding": ({"status": "included", "encoder": encoder_name, "selected_C_per_cv_fit": cv_C, "selected_C_final": final_C[0],
                       "C_selection": "inner StratifiedGroupKFold(3) on each fit's training rows only, grid " + str(list(C_GRID))}
                      if encoder is not None else {"status": "skipped", "reason": embedding_note}),
        "outputs_sha256": {p.name: _sha(p) for p in (p_res, p_conf, p_oof)},
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "MANIFEST_m2.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": rows.data, "manifest": manifest, "paths": [str(p_res), str(p_conf), str(p_oof)], "models": models}
