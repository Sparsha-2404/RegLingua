"""Day 12 (M2) data audit: duplicates, original_id grouping, balance, baseline training provenance,
train/eval overlap, and (if present) verification of the frozen split. Read-only: touches no model
output and no labels. Writes data/evaluation/audit_eval_data_m2.{txt,json}.

Usage (repo root):  python scripts/audit_eval_data.py [--semantic]
  --semantic  (needs Hugging Face weights; not runnable offline) also lists hi-en rows whose nearest
              en/hi neighbour in ANOTHER original_id has cosine >= 0.85 -> audit_semantic_neighbors_m2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation import data_audit as da  # noqa: E402
from services.evaluation.final_split import verify_split  # noqa: E402

D = REPO_ROOT / "data" / "evaluation"
EVAL = D / "risk_evidence_eval_m2.csv"
RAW = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
BACKUP = REPO_ROOT / "data" / "raw" / "labeled_examples_m1_short_backup.csv"
XLSX = D / "RegLingua_M1_FINAL.xlsx"
SPLIT = D / "final_split_m2.csv"
BASELINE_PKL = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
HELD_FLAGS = D / "pipeline_retrieval_results_m2.csv"  # carries the Day 5-10 `held_out` flag per query_id
SEED, THR = 42, 0.9
OTHER_FILES = {  # name -> (path, text columns)
    "rewrite_eval_m2.csv": (D / "rewrite_eval_m2.csv", ["text"]),
    "rewrite_eval_set_m1.csv": (D / "rewrite_eval_set_m1.csv", ["original_text"]),
    "canonical_cases_m2.csv": (D / "canonical_cases_m2.csv", ["text"]),
    "verifier_known_cases_m2.csv": (D / "verifier_known_cases_m2.csv", ["original", "rewrite"]),
}
OUT: list[str] = []
REPORT: dict = {}


def say(s: str = "") -> None:
    print(s)
    OUT.append(s)


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def xlsx_sheet_rows(sheet: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    it = wb[sheet].iter_rows(values_only=True)
    head = [str(h).strip() if h is not None else "" for h in next(it)]
    rows = [dict(zip(head, r)) for r in it if any(c not in (None, "") for c in r)]
    return rows


def table(title: str, pairs) -> None:
    c = Counter(pairs)
    rows = sorted({a for a, _ in c})
    cols = sorted({b for _, b in c})
    say(f"  {title}: " + " | ".join(f"{r}: " + ",".join(f"{k}={c[(r, k)]}" for k in cols) for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", action="store_true")
    args = ap.parse_args()

    ev = read_csv(EVAL)
    n = len(ev)
    text = [r["text"] for r in ev]
    oid = [r["original_id"] for r in ev]
    lang = [r["language"] for r in ev]
    reg = [r["true_regulator"] for r in ev]
    lab = [r["true_risk_level"] for r in ev]

    # ---- 1. inventory + provenance of eval rows --------------------------------------------------
    say(f"== 1. FILES  (eval = {EVAL.name}, {n} rows, {len(set(r['query_id'] for r in ev))} unique query_ids, "
        f"{len(set(oid))} original_ids)")
    raw, bak = read_csv(RAW), read_csv(BACKUP)
    same_raw = [(a["text"].strip(), a["risk_level"].strip().upper()) for a in raw] == \
               [(t.strip(), l) for t, l in zip(text, lab)]
    say(f"  raw training CSV ({RAW.name}): {len(raw)} rows; row-for-row identical (text+label, same order) to eval file: {same_raw}")
    say(f"  short_backup CSV identical to raw training CSV: {raw == bak}")
    xl_note = "openpyxl missing -> skipped"
    xl_texts: list[str] = []
    try:
        for sh in ("SEBI", "RBI", "DPDP"):
            xl_texts += [str(r.get("TEXT") or "") for r in xlsx_sheet_rows(sh)]
        m = da.cross_matches(text, xl_texts, THR)
        ex = sum(k == "exact" for k, _, _ in m)
        xl_note = (f"{len(xl_texts)} rows in SEBI/RBI/DPDP sheets; {ex} exact + "
                   f"{sum(k == 'near' for k, _, _ in m)} near-only matches into eval; "
                   f"eval rows not in xlsx: {n - len({i for k, i, _ in m if k == 'exact'})}")
    except Exception as e:  # noqa: BLE001
        xl_note = f"xlsx read failed: {e!r}"
    say(f"  M1 xlsx ({XLSX.name}): {xl_note}")
    say("  -> no separate 'final labelled evaluation set' beyond the xlsx; the 516-row CSV is its flattened export.")
    REPORT["files"] = {"eval_rows": n, "raw_identical_to_eval": same_raw, "backup_identical_to_raw": raw == bak,
                       "xlsx": xl_note}

    # ---- 2. duplicates ---------------------------------------------------------------------------
    say(f"\n== 2. DUPLICATES (normalised text exact; char-trigram Jaccard >= {THR} near)")
    exact, near = da.find_duplicates(text, THR)
    say(f"  within eval: exact groups={len(exact)} (rows={sum(map(len, exact))}); near pairs={len(near)}")
    link = [(g[0], j) for g in exact for j in g[1:]] + [(i, j) for i, j, _ in near]
    cross_id = [(i, j) for i, j in link if oid[i] != oid[j]]
    conflict = [(i, j) for i, j in link if lab[i] != lab[j]]
    say(f"  of those links: different original_id={len(cross_id)}; label conflict={len(conflict)}")
    for i, j in list(dict.fromkeys(cross_id + conflict))[:5]:
        say(f"    e.g. {ev[i]['query_id']}({oid[i]},{lab[i]}) ~ {ev[j]['query_id']}({oid[j]},{lab[j]})")
    for name, (p, cols) in OTHER_FILES.items():
        if not p.exists():
            say(f"  {name}: not present")
            continue
        rows = read_csv(p)
        for c in cols:
            if c not in (rows[0] if rows else {}):
                continue
            ts = [r[c] for r in rows if r.get(c)]
            ex_o, nr_o = da.find_duplicates(ts, THR)
            m = da.cross_matches(text, ts, THR)
            kinds = Counter(k for k, _, _ in m)
            say(f"  {name}[{c}]: {len(ts)} texts | within-file exact groups={len(ex_o)}, near pairs={len(nr_o)} | "
                f"vs eval: exact={kinds['exact']} near-only={kinds['near']} none={kinds['none']}")
            REPORT.setdefault("cross_file", {})[f"{name}[{c}]"] = dict(kinds)
    REPORT["duplicates"] = {"exact_groups": len(exact), "near_pairs": len(near),
                            "links_cross_original_id": len(cross_id), "links_label_conflict": len(conflict)}

    # ---- 3. original_id grouping -----------------------------------------------------------------
    say("\n== 3. original_id GROUPING")
    by = defaultdict(list)
    for i, o in enumerate(oid):
        by[o].append(i)
    sizes = Counter(len(v) for v in by.values())
    combos = Counter("+".join(sorted(lang[i] for i in v)) for v in by.values())
    say(f"  groups={len(by)}; size histogram={dict(sizes)}; language combos={dict(combos)}")
    lab_conf = sum(len({lab[i] for i in v}) > 1 for v in by.values())
    reg_conf = sum(len({reg[i] for i in v}) > 1 for v in by.values())
    pref_bad = sum(not o.upper().startswith(reg[i].upper()) for o, v in by.items() for i in v[:1])
    say(f"  groups with label conflict between versions={lab_conf}; regulator conflict={reg_conf}; "
        f"id prefix != regulator={pref_bad}")
    en_hi = [v for v in by.values() if sorted(lang[i] for i in v) == ["en", "hi"]]
    say(f"  en+hi pairs sharing one original_id: {len(en_hi)}  -> translations ARE grouped (never split by a group split)")
    hien = [i for i in range(n) if lang[i] == "hi-en"]
    others = [i for i in range(n) if lang[i] != "hi-en"]
    sims = []
    for i in hien:
        gi = da.trigrams(da.normalize_text(text[i]))
        sims.append(max(da.jaccard(gi, da.trigrams(da.normalize_text(text[j]))) for j in others if oid[j] != oid[i]))
    say(f"  hi-en rows={len(hien)} each in its own singleton group; max lexical trigram sim to any en/hi row of another "
        f"group: max={max(sims):.2f}, rows>=0.5: {sum(s >= 0.5 for s in sims)}  (lexical only; semantic siblings not ruled out)")
    REPORT["grouping"] = {"groups": len(by), "sizes": dict(sizes), "combos": dict(combos), "label_conflict_groups": lab_conf,
                          "hi_en_rows": len(hien), "hi_en_max_lexical_sim": round(max(sims), 3)}

    # ---- 4. balance ------------------------------------------------------------------------------
    say("\n== 4. LABEL BALANCE (rows)")
    say(f"  overall: {dict(Counter(lab))}")
    table("risk x language", zip(lab, lang))
    table("risk x regulator", zip(lab, reg))
    table("language x regulator", zip(lang, reg))
    small = [(k, c) for k, c in Counter(zip(lab, lang, reg)).items() if c < 10]
    say(f"  risk x language x regulator cells with <10 rows: {len(small)} of {len(set(zip(lab, lang, reg)))}")

    # ---- 5. baseline provenance ------------------------------------------------------------------
    say("\n== 5. COMMITTED BASELINE: what it was trained on")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from services.classifier.dataset import load_dataset
    examples = load_dataset(str(RAW))
    y = [e.risk_level for e in examples]
    tr_ix, te_ix = train_test_split(list(range(len(examples))), test_size=0.2, random_state=42, stratify=y)
    say("  script: scripts/train_baseline_classifier.py -> load_dataset(data/raw/labeled_examples_m1.csv) -> "
        "train_test_split_examples(test_size=0.2, random_state=42, stratified on risk_level; NOT grouped)")
    say(f"  reproduced split: {len(tr_ix)} train / {len(te_ix)} test of {len(examples)} rows")
    verdict = "pickle not found"
    if BASELINE_PKL.exists():
        import pickle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = pickle.load(open(BASELINE_PKL, "rb"))["vectorizer"]
        mine = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True).fit(
            [examples[i].normalized_text for i in tr_ix])
        full = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True).fit(
            [e.normalized_text for e in examples])
        jt = len(set(mine.vocabulary_) & set(vec.vocabulary_)) / len(set(mine.vocabulary_) | set(vec.vocabulary_))
        ja = len(set(full.vocabulary_) & set(vec.vocabulary_)) / len(set(full.vocabulary_) | set(vec.vocabulary_))
        verdict = (f"vocabulary Jaccard vs pickle: refit on the 412 train rows={jt:.3f}, refit on all 516 rows={ja:.3f} "
                   f"-> pickle matches the 80% train split, not the full file (exact equality not provable: CSV/pickle "
                   f"have no recorded data hash and the pickle (Sep 24) predates the only committed CSV (Sep 27))")
    say(f"  pickle check: {verdict}")
    held = {r["query_id"]: r["held_out"] == "True" for r in read_csv(HELD_FLAGS)}
    held_flags = [held.get(r["query_id"]) for r in ev]
    test_set = set(te_ix)
    match = all(held_flags[i] == (i in test_set) for i in range(n))
    say(f"  reproduced test rows == Day 5-10 `held_out` flag ({sum(bool(h) for h in held_flags)} rows): {match}")
    seen = [i for i in tr_ix]
    sib = [i for i in te_ix if any(oid[j] == oid[i] and j in set(tr_ix) for j in by[oid[i]])]
    say(f"  eval rows (all {n} come from the training CSV) seen in baseline training: {len(seen)} ({len(seen) / n:.0%}); "
        f"never seen: {len(te_ix)}")
    say(f"  of the {len(te_ix)} unseen rows, {len(sib)} have an en/hi sibling (same original_id) in training "
        f"-> translation leakage even for the 'held-out' rows")
    say("  embedding model (models/*minilm.pkl): meta.json records no training rows; "
        "scripts/evaluate_day10_model_comparison_m2.py trains on rows with held_out==False "
        "(then drops rows sharing an original_id with held-out).")
    say("  => scoring ANY committed model on the 516 rows is contaminated (80% seen); "
        "only freshly trained models on the train_pool can be scored on final_test.")
    REPORT["baseline"] = {"train_rows_seen": len(seen), "unseen_rows": len(te_ix), "unseen_with_sibling_in_train": len(sib),
                          "pickle_check": verdict, "split_reproduces_held_out_flag": match}

    # ---- 6. frozen split verification ------------------------------------------------------------
    say("\n== 6. FROZEN SPLIT")
    if SPLIT.exists():
        sp = {r["row_id"]: r for r in read_csv(SPLIT)}
        rows = [{"row_id": r["query_id"], "original_id": r["original_id"], "split": sp[r["query_id"]]["split"]} for r in ev]
        groups = da.leak_groups(oid, link)
        v = verify_split(rows, groups)
        test_ix = [i for i, r in enumerate(rows) if r["split"] == "final_test"]
        old_seen = sum(bool(held_flags[i]) for i in test_ix)
        say(f"  {SPLIT.name}: {v}")
        say(f"  final_test rows that were in the Day 5-10 `held_out` set (analysed/tuned on before): {old_seen}/{len(test_ix)}; "
            f"final_test rows the committed baseline trained on: {sum(i in set(tr_ix) for i in test_ix)}")
        for nm, col in (("risk", lab), ("language", lang), ("regulator", reg)):
            c_all, c_te = Counter(col), Counter(col[i] for i in test_ix)
            say(f"  test share by {nm}: " + ", ".join(f"{k}={c_te[k] / c_all[k]:.0%}" for k in sorted(c_all)))
        REPORT["split"] = v | {"final_test_in_old_held_out": old_seen}
    else:
        say(f"  {SPLIT.name} not created yet (run scripts/make_final_split.py)")

    # ---- optional semantic neighbours (needs HF) -------------------------------------------------
    if args.semantic:
        import numpy as np
        from services.embeddings.encoder import MultilingualEncoder
        E = np.array(MultilingualEncoder().encode(text))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        S = E @ E.T
        out = D / "audit_semantic_neighbors_m2.csv"
        k = 0
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "original_id", "neighbor_query_id", "neighbor_original_id", "cosine"])
            for i in hien:
                cand = [j for j in others if oid[j] != oid[i]]
                j = max(cand, key=lambda x: S[i, x])
                if S[i, j] >= 0.85:
                    w.writerow([ev[i]["query_id"], oid[i], ev[j]["query_id"], oid[j], round(float(S[i, j]), 4)])
                    k += 1
        say(f"\n== SEMANTIC: hi-en rows with a >=0.85 cosine neighbour in another original_id: {k}/{len(hien)} -> {out.name}")

    (D / "audit_eval_data_m2.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    (D / "audit_eval_data_m2.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
