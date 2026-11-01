"""Integrity checks for the M1 data. `python scripts/validate_dataset.py` exits non-zero on any failure."""
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import *

def run():
    errs = []
    df, kb, src = load_labelled(), load_kb(), load_sources()
    ev = pd.read_csv(DATA/"regulatory"/"evidence_qa.csv", dtype=str, keep_default_na=False)
    rw = pd.read_csv(DATA/"evaluation"/"rewrite_eval.csv", dtype=str, keep_default_na=False)
    vc = pd.read_csv(DATA/"evaluation"/"verifier_test_cases.csv", dtype=str, keep_default_na=False)
    gold = load_retrieval_gold(None)

    if df.duplicated(["REGULATOR","ID","LANGUAGE"]).any(): errs.append("duplicate (REGULATOR,ID,LANGUAGE)")
    if not set(df.LANGUAGE) <= {"en","hi","hi-en"}: errs.append("unexpected LANGUAGE codes")
    if not set(df["RISK LEVEL"]) <= {"Low","Medium","High"}: errs.append("unexpected RISK LEVEL values")
    for c in ["SOURCE_ID","DOCUMENT_TITLE","SECTION","EFFECTIVE_DATE","SOURCE_URL","TOPIC","TEXT"]:
        if (df[c].str.strip()=="").any(): errs.append(f"blank values in {c}")
    if (df.TOPIC != df.TOPIC.str.strip()).any(): errs.append("TOPIC has leading/trailing spaces")
    bad = ~df.EFFECTIVE_DATE.str.match(r"^(\d{2}-\d{2}-\d{4}|—)")
    if bad.any(): errs.append(f"EFFECTIVE_DATE does not start with DD-MM-YYYY (or —) in {bad.sum()} rows")
    if not set(df.SOURCE_ID) <= set(src.SOURCE_ID): errs.append("labelled SOURCE_ID missing from regulatory_sources")
    if not set(kb.SOURCE_ID) <= set(src.SOURCE_ID): errs.append("KB SOURCE_ID missing from regulatory_sources")
    if kb.CHUNK_ID.duplicated().any(): errs.append("duplicate CHUNK_ID")
    # translation pairs: same risk + same split
    for (reg, i), g in df.groupby(["REGULATOR","ID"]):
        if g["RISK LEVEL"].nunique() > 1: errs.append(f"risk mismatch across languages: {reg} {i}")
        if g.SPLIT.nunique() > 1: errs.append(f"split leakage across translations: {reg} {i}")
    # evidence QA covers every unique ID and is fully verified
    if set(ev.EVIDENCE_STATUS) != {"VERIFIED"}: errs.append("EVIDENCE_QA has non-VERIFIED rows")
    if not set(df.ID) <= set(ev.ID): errs.append("labelled IDs missing from EVIDENCE_QA")
    # retrieval gold points at real chunks
    ids = set(kb.CHUNK_ID)
    for s in gold.GOLD_CHUNK_IDS:
        if not s or any(c not in ids for c in s.split("|")): errs.append("retrieval_gold references unknown chunk"); break
    if len(gold) != len(df): errs.append("retrieval_gold should have one row per labelled example")
    for name, t in [("rewrite_eval", rw), ("verifier_test_cases", vc)]:
        if not set(t.SOURCE_ID) <= set(src.SOURCE_ID): errs.append(f"{name}: unknown SOURCE_ID")
    if set(vc.EXPECTED_VERDICT) - {"PASS","FAIL","REVIEW"}: errs.append("bad EXPECTED_VERDICT")
    return errs

if __name__ == "__main__":
    e = run()
    print("OK - all data integrity checks passed" if not e else "FAILED:\n- " + "\n- ".join(e))
    sys.exit(1 if e else 0)
