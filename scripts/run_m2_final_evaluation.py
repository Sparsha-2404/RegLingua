"""Day 12 (M2): single-command final risk-classification evaluation.

    python scripts/run_m2_final_evaluation.py --seed 42
    python scripts/run_m2_final_evaluation.py --seed 42 --embedding on     # fail instead of skip if the encoder is unavailable

Reads the FROZEN split data/evaluation/final_split_m2.csv (never re-draws it). Train pool: repeated group-stratified
5-fold CV (3 repeats) of majority / banned-phrase rule / TF-IDF+LogReg (re-trained per fold, never the committed .pkl)
and the embedding classifier when its encoder is cached locally. Then one fit on the whole train pool and one scoring of
final_test. Writes data/evaluation/final_eval_{results,confusions,oof_predictions}_m2.csv and results/MANIFEST_m2.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.final_eval import N_BOOT, run_final_evaluation  # noqa: E402

D = REPO_ROOT / "data" / "evaluation"
LOCAL_CMD = ("pip install sentence-transformers\n"
             "  python -c \"from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')\"   # cache once, online\n"
             "  python scripts/run_m2_final_evaluation.py --seed 42 --embedding on")


def build_encoder(mode: str):
    """(encoder|None, name, note). Local files only: never downloads."""
    if mode == "off":
        return None, "", "disabled with --embedding off"
    try:
        from services.classifier.model_selection import DEFAULT_ENCODER, LocalOnlyEncoder, encoder_is_cached

        if not encoder_is_cached(DEFAULT_ENCODER):
            raise RuntimeError(f"encoder '{DEFAULT_ENCODER}' is not in the local Hugging Face cache")
        enc = LocalOnlyEncoder(DEFAULT_ENCODER)
        enc.encode(["smoke test"])
        return enc, DEFAULT_ENCODER, ""
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        if mode == "on":
            sys.exit(f"--embedding on but the encoder is unavailable ({note}).\nRun locally:\n  {LOCAL_CMD}")
        print(f"NOTE: embedding classifier SKIPPED ({note}).\nTo include it, run locally:\n  {LOCAL_CMD}")
        return None, "", note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--embedding", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--eval", type=Path, default=D / "risk_evidence_eval_m2.csv")
    ap.add_argument("--split", type=Path, default=D / "final_split_m2.csv")
    ap.add_argument("--out-dir", type=Path, default=D)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    a = ap.parse_args()
    meta_path = a.split.with_suffix(".meta.json")
    expected = json.loads(meta_path.read_text(encoding="utf-8"))["eval_sha256"] if meta_path.exists() else None
    enc, name, note = build_encoder(a.embedding)
    res = run_final_evaluation(a.eval, a.split, a.out_dir, a.results_dir, seed=a.seed, n_boot=a.n_boot, encoder=enc,
                               encoder_name=name, embedding_note=note, expected_eval_sha=expected)
    show(res["rows"], res["models"])
    print("wrote:", *res["paths"], str(a.results_dir / "MANIFEST_m2.json"), sep="\n  ")


def show(rows: list, models: list) -> None:
    """Compact console summary: headline metrics and the TF-IDF vs rule-baseline verdict."""
    for phase in ("cv_oof", "final_test"):
        print(f"\n[{phase}] overall (95% cluster-bootstrap CI)")
        for m in models:
            vals = {r[5]: r for r in rows if r[0] == phase and r[1] == "overall" and r[3] == m}
            print(f"  {m:20}" + "  ".join(f"{k}={vals[k][6]} [{vals[k][7]},{vals[k][8]}]" for k in ("accuracy", "macro_f1", "false_compliant_rate")))
        for r in rows:
            if r[0] == phase and r[1] == "paired_overall" and r[3] == "tfidf_logreg" and r[4] == "banned_phrase_rule":
                print(f"  TF-IDF - rule baseline  {r[5]:22} {r[6]} [{r[7]},{r[8]}] -> {r[12]}")


if __name__ == "__main__":
    main()
