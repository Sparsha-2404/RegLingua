"""Day-10 helper: auto-check M3's explanations against gold evidence.
Input CSV columns: CASE_ID, AI_EXPLANATION, CITED_CHUNK_IDS (pipe-separated)
Usage: python scripts/validate_explanations.py outputs.csv > flags.csv
Auto flags are a first pass only; M1 still fills the reviewed columns in EXPLANATION_VALIDATION."""
import re, sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DATA, load_kb

REF = re.compile(r"(Regulation|Para|Paragraph|Section|Rule)s?\s+(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)", re.I)

def main(path):
    kb = load_kb().set_index("CHUNK_ID")
    tpl = pd.read_csv(DATA/"evaluation"/"explanation_validation_template.csv", dtype=str, keep_default_na=False).set_index("CASE_ID")
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for _, r in out.iterrows():
        flags, cited = [], [c for c in r["CITED_CHUNK_IDS"].split("|") if c]
        gold = tpl.loc[r["CASE_ID"], "GOLD_CHUNK_IDS"].split("|") if r["CASE_ID"] in tpl.index else []
        if not cited: flags.append("NO_CITATION")
        unknown = [c for c in cited if c not in kb.index]
        if unknown: flags.append("UNKNOWN_CHUNK:" + ",".join(unknown))
        if cited and gold and not set(cited) & set(gold): flags.append("CITES_NO_GOLD")
        known = [c for c in cited if c in kb.index]
        ctx = " ".join(kb.loc[c, "TEXT"] + " " + kb.loc[c, "SECTION"] for c in known).lower()
        for lab, num in REF.findall(r["AI_EXPLANATION"]):
            if num.lower() not in ctx: flags.append(f"UNSUPPORTED_REF:{lab} {num}")
        for q in re.findall(r"[\"“]([^\"”]{40,})[\"”]", r["AI_EXPLANATION"]):
            if q.lower().strip() not in ctx: flags.append("QUOTE_NOT_IN_SOURCE")
        hard = any(f.startswith(("UNKNOWN_CHUNK","UNSUPPORTED_REF","QUOTE_NOT_IN_SOURCE")) for f in flags)
        verdict = "FAIL" if hard else "REVIEW" if flags else "PASS_CANDIDATE"
        rows.append({"CASE_ID": r["CASE_ID"], "AUTO_FLAGS": ";".join(flags), "AUTO_VERDICT": verdict})
    res = pd.DataFrame(rows); res.to_csv(sys.stdout, index=False); return res

if __name__ == "__main__":
    main(sys.argv[1])
