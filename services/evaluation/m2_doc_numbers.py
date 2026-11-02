"""
DAY 13 (M2): the numbers ledger behind docs/snippets_for_team_m2.md and docs/README_M2.md.

Every number that appears in those documents is computed HERE from result files in data/evaluation
(real runs made earlier) and written to data/evaluation/m2_doc_numbers.json with its source file(s) and
derivation. Documents are rendered from the ledger, never typed by hand. Seed 42, 2000 resamples,
percentile 95% CIs. Missing inputs -> the key is simply absent (documents then say "not available").
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from services.evaluation.confidence_signals import ece
from services.evaluation.model_comparison import accuracy, macro_f1

SEED, N_BOOT = 42, 2000
D = "data/evaluation/"


def _boot_ci(stat: Callable[[np.ndarray], float], n: int, groups=None, n_boot: int = N_BOOT, seed: int = SEED):
    g = np.arange(n) if groups is None else np.asarray(groups, dtype=object)
    members = [np.where(g == u)[0] for u in sorted(set(g.tolist()))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        vals.append(stat(idx))
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))


def _put(L: dict, key: str, value, source, how: str) -> None:
    L[key] = {"value": value, "source": [source] if isinstance(source, str) else list(source), "derivation": how}


def _put_ci(L, key, value, ci, source, how):
    _put(L, key, float(value), source, how)
    _put(L, key + ".ci_low", ci[0], source, how + " (95% CI lower, cluster bootstrap seed 42, 2000 resamples)")
    _put(L, key + ".ci_high", ci[1], source, how + " (95% CI upper)")


def _read(root: Path, rel: str) -> Optional[pd.DataFrame]:
    p = Path(root) / rel
    return pd.read_csv(p, encoding="utf-8-sig") if p.is_file() else None


def _classifier(L: dict, root: Path) -> None:
    f = D + "confidence_signals_m2.csv"
    df = _read(root, f)
    if df is None:
        return
    for name, sub in (("heldout", df[df.split == "held_out"]), ("oof", df[df.split != "held_out"])):
        y, p, g = sub.true_risk.to_numpy(str), sub.pred_label.to_numpy(str), sub.original_id.to_numpy(str)
        conf, ok = sub.max_prob.to_numpy(float), sub.correct.to_numpy(float)
        how = f"rows with split {'== held_out' if name == 'heldout' else '!= held_out (out-of-fold predictions)'}"
        _put(L, f"clf.{name}.n", int(len(sub)), f, how)
        _put(L, f"clf.{name}.prob_source", ", ".join(sorted(sub.prob_source.astype(str).unique())), f, "unique prob_source values")
        _put_ci(L, f"clf.{name}.accuracy", accuracy(y, p), _boot_ci(lambda i: accuracy(y[i], p[i]), len(y), g), f, how + "; accuracy(true_risk, pred_label)")
        _put_ci(L, f"clf.{name}.macro_f1", macro_f1(y, p), _boot_ci(lambda i: macro_f1(y[i], p[i]), len(y), g), f, how + "; macro-F1")
        _put_ci(L, f"clf.{name}.ece", ece(conf, ok)[0], _boot_ci(lambda i: ece(conf[i], ok[i])[0], len(y), g), f, how + "; top-label ECE, 10 equal-width bins, max_prob vs correct")
        for lang in ("en", "hi", "hi-en"):
            m = sub.language.to_numpy(str) == lang
            if m.any():
                _put(L, f"clf.{name}.n.{lang}", int(m.sum()), f, how + f"; language == {lang}")
                yy, pp, gg = y[m], p[m], g[m]
                _put_ci(L, f"clf.{name}.accuracy.{lang}", accuracy(yy, pp), _boot_ci(lambda i: accuracy(yy[i], pp[i]), len(yy), gg), f, how + f"; language == {lang}")
    held = set(df[df.split == "held_out"].original_id)
    train = set(df[df.split != "held_out"].original_id)
    _put(L, "leak.heldout_ids", len(held), f, "distinct original_id among held-out rows")
    _put(L, "leak.heldout_ids_with_partner_in_train", len(held & train), f, "held-out original_ids that also occur among non-held-out (training-pool) rows")


def _abstention(L: dict, root: Path) -> None:
    f = D + "abstention_operating_points_m2.csv"
    df = _read(root, f)
    if df is not None:
        for _, r in df[df.group_type == "all"].iterrows():
            op = str(r["operating_point"]).split()[0]
            for col, key in (("n", "n"), ("accept", "accepted"), ("review", "review"), ("abstain", "abstain"), ("coverage", "coverage"),
                             ("coverage_lo", "coverage.ci_low"), ("coverage_hi", "coverage.ci_high"),
                             ("accuracy_on_accepted", "accuracy_on_accepted"), ("acc_lo", "accuracy_on_accepted.ci_low"),
                             ("acc_hi", "accuracy_on_accepted.ci_high"), ("false_compliant", "false_compliant"),
                             ("abstain_floor", "abstain_floor"), ("compliant_floor", "compliant_floor")):
                v = r.get(col)
                if v is not None and pd.notna(v):
                    _put(L, f"abst.{op}.{key}", float(v), f, f"row operating_point={r['operating_point']}, group_type=all, column {col}")
    j = Path(root) / D / "abstention_thresholds_m2.json"
    if j.is_file():
        meta = json.loads(j.read_text(encoding="utf-8"))
        _put(L, "abst.verdict_text", meta.get("verdict", ""), D + "abstention_thresholds_m2.json", "field 'verdict' (quoted verbatim)")
    s = Path(root) / "agents" / "supervisor.py"
    if s.is_file():
        t = s.read_text(encoding="utf-8")
        for name, key in (("_ABSTENTION_CONFIDENCE_FLOOR", "supervisor.abstain_floor"), ("_COMPLIANT_CONFIDENCE_FLOOR", "supervisor.compliant_floor")):
            m = re.search(rf"^{name}\s*=\s*([0-9.]+)", t, re.M)
            if m:
                _put(L, key, float(m.group(1)), "agents/supervisor.py", f"constant {name} (read-only)")


def _retrieval(L: dict, root: Path) -> None:
    for name, f in (("keyword", D + "retrieval_eval_results_m2_keyword.csv"), ("hybrid", D + "retrieval_eval_results_m2_hybrid.csv")):
        df = _read(root, f)
        if df is None:
            continue
        _put(L, f"retr.{name}.n", int(len(df)), f, "number of query rows")
        for k in (1, 3, 5):
            h = df[f"hit_at_{k}"].to_numpy(float)
            _put_ci(L, f"retr.{name}.recall@{k}", h.mean(), _boot_ci(lambda i: h[i].mean(), len(h)), f, f"mean of hit_at_{k}; row-level bootstrap")
        for lang in ("en", "hi", "hi-en"):
            m = (df.language == lang).to_numpy()
            if m.any():
                h = df.loc[m, "hit_at_1"].to_numpy(float)
                _put(L, f"retr.{name}.n.{lang}", int(m.sum()), f, f"language == {lang}")
                _put_ci(L, f"retr.{name}.recall@1.{lang}", h.mean(), _boot_ci(lambda i: h[i].mean(), len(h)), f, f"mean hit_at_1, language == {lang}")
    f = D + "day6_consistency/risk_evidence_consistency_all.csv"
    df = _read(root, f)
    if df is not None:
        av = df[df.expected_source_available.astype(str) == "True"]
        for sub, name in ((av, "all"), (av[av.held_out.astype(str) == "True"], "heldout")):
            for col, key in (("expected_source_top1", "top1"), ("expected_source_in_top3", "top3")):
                x = (sub[col].astype(str) == "True").to_numpy(float)
                _put(L, f"pipe.{name}.{key}", float(x.mean()), f, f"share of rows with {col} True among expected_source_available rows ({name})")
            _put(L, f"pipe.{name}.n", int(len(sub)), f, f"rows with expected_source_available True ({name})")
    f = D + "pipeline_retrieval_results_m2.csv"
    df = _read(root, f)
    if df is not None:
        h = df[df.held_out.astype(str) == "True"]
        x = (h.predicted_regulator.astype(str) == h.true_regulator.astype(str)).to_numpy(float)
        _put(L, "pipe.regulator_acc_heldout", float(x.mean()), f, "predicted_regulator == true_regulator over held-out rows")
        _put(L, "pipe.regulator_acc_heldout.n", int(len(h)), f, "held-out rows")


def _day10(L: dict, root: Path) -> None:
    f = D + "model_comparison_m2.csv"
    df = _read(root, f)
    if df is None or "mode" not in df or set(df["mode"]) != {"real"}:
        return          # absent, or a FAKE_SMOKE file: never cited
    r = df[(df.scope == "difference") & (df.group == "all") & (df.metric == "macro_f1_bootstrap")]
    if len(r):
        r = r.iloc[0]
        for c in ("baseline", "embedding", "delta", "ci_low", "ci_high"):
            _put(L, f"cmp.macro_f1.{c}", float(r[c]), f, f"row scope=difference, metric=macro_f1_bootstrap, column {c}")
        _put(L, "cmp.macro_f1.verdict", str(r["note"]), f, "note column of the same row")
        _put(L, "cmp.held_out_n", int(r["n"]), f, "n of the same row")


def collect_numbers(root) -> Dict[str, dict]:
    root = Path(root)
    L: Dict[str, dict] = {}
    for fn in (_classifier, _abstention, _retrieval, _day10):
        fn(L, root)
    return L


def write_ledger(root, ledger: Dict[str, dict]) -> Path:
    p = Path(root) / D / "m2_doc_numbers.json"
    p.write_text(json.dumps(ledger, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
