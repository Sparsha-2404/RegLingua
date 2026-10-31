"""
DAY 6 GATE CHECK (M2): run real statements through the PRODUCTION pipeline and
judge whether the explanation is tied to the right evidence.

For each row of data/evaluation/risk_evidence_eval_m2.csv this calls
agents.supervisor.run_case(text, use_trained_artifacts=True), i.e. the real
classifier, the retriever agent_factory.py injects today, the real Explainer,
Rewriter and Verifier, then compares the cited clause with M1's verified
source. It changes nothing in the pipeline.

Usage (from the repo root):
    python scripts/check_day6_gate.py --sample 6 --show 6    # eyeball a few full explanations
    python scripts/check_day6_gate.py                        # all 516 rows
    python scripts/check_day6_gate.py --limit 40

    # What the pipeline WOULD do if agent_factory.py injected the fixed HybridRetriever.
    # The factory file is not edited: its retriever constructor is swapped in-process.
    python scripts/check_day6_gate.py --retriever hybrid
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.dataset_join import read_csv_rows  # noqa: E402
from services.evaluation.gate_check import enrich, format_gate_report, summarize_gate  # noqa: E402

JOINED = ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
OUT = ROOT / "data" / "evaluation" / "day6_gate_check_m2.csv"
OUT_HYBRID = ROOT / "data" / "evaluation" / "day6_gate_check_m2_hybrid.csv"
KB_PATH = ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"

_PATCH = {"installed": False, "built": 0}


def install_hybrid_retriever() -> None:
    """Make the production factory build the fixed HybridRetriever instead of the
    DeterministicRetriever it injects today. agents/agent_factory.py is NOT edited:
    only its module-level name `DeterministicRetriever` is replaced for this
    process, and it must happen before the first run_case() (the Supervisor is cached)."""
    import agents.agent_factory as factory
    from services.embeddings.vector_index import RegulatoryVectorIndex
    from services.retrieval.deterministic_retriever import DeterministicRetriever
    from services.retrieval.hybrid_retriever import HybridRetriever

    def build(corpus_path=None):
        kb = Path(corpus_path) if corpus_path else KB_PATH
        _PATCH["built"] += 1
        return HybridRetriever(
            keyword_retriever=DeterministicRetriever(corpus_path=kb),
            vector_index=RegulatoryVectorIndex(corpus_path=kb),
        )

    factory.DeterministicRetriever = build
    _PATCH["installed"] = True

COLUMNS = [
    "query_id", "language", "true_regulator", "true_risk_level", "expected_source_id",
    "predicted_risk_label", "confidence", "regulator", "n_evidence", "evidence_ids",
    "clause_cited", "risk_phrase", "reason", "resolution_status", "outcome",
    "cited_in_evidence", "reason_mentions_clause", "quoted_is_fragment",
    "quoted_fragment_words", "error",
]


def run_one(row: dict) -> dict:
    from agents.supervisor import run_case

    rec = {
        "query_id": row["query_id"], "language": row["language"],
        "true_regulator": row["true_regulator"], "true_risk_level": row["true_risk_level"],
        "expected_source_id": row["expected_source_id"], "input_text": row["text"],
        "error": "",
    }
    try:
        state = run_case(row["text"], use_trained_artifacts=True)
    except Exception as exc:  # recorded, never silent
        rec.update({"error": f"{type(exc).__name__}: {exc}", "evidence_ids": [], "n_evidence": 0})
        return rec

    rec.update({
        "predicted_risk_label": state.risk_label.value if state.risk_label else "",
        "confidence": round(float(state.classifier_confidence or 0.0), 4),
        "regulator": state.regulator.value if state.regulator else "",
        "evidence_ids": [e.clause_id for e in state.evidence],
        "n_evidence": len(state.evidence),
        "clause_cited": state.clause_cited or "",
        "risk_phrase": state.risk_phrase or "",
        "reason": state.reason or "",
        "resolution_status": state.resolution_status.value if state.resolution_status else "",
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="first N rows")
    ap.add_argument("--sample", type=int, default=None, help="random sample of N rows (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print the first N full explanations")
    ap.add_argument("--retriever", choices=["keyword", "hybrid"], default="keyword",
                    help="keyword = the pipeline as it is today (default); hybrid = as if the "
                         "factory injected the fixed HybridRetriever (needs the embedding model).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_path = args.out or (OUT_HYBRID if args.retriever == "hybrid" else OUT)
    if args.retriever == "hybrid":
        install_hybrid_retriever()

    if not JOINED.exists():
        raise SystemExit(f"Missing {JOINED}\n  -> run: python scripts/build_risk_evidence_eval.py")
    _, rows = read_csv_rows(JOINED)
    if args.sample:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    elif args.limit:
        rows = rows[: args.limit]

    records = [enrich(run_one(r)) for r in rows]
    if args.retriever == "hybrid" and not _PATCH["built"]:
        raise SystemExit("The hybrid override did not take effect (the factory never built a retriever "
                         "through the patched name), so these results are NOT hybrid. Not saving.")
    print(f"Retriever in the pipeline for this run: {args.retriever}"
          + (f" (factory built {_PATCH['built']} retriever(s) via the override)" if args.retriever == "hybrid" else "")
          + "\n")

    if args.show:
        out = ROOT / "gate_check_examples.txt"   # UTF-8 file: Hindi survives the Windows console
        with open(out, "w", encoding="utf-8") as f:
            for r in records[: args.show]:
                f.write(f"{r['query_id']} [{r['language']}] true={r['true_risk_level']} "
                        f"predicted={r.get('predicted_risk_label', '')} conf={r.get('confidence', '')}\n"
                        f"  statement:  {r['input_text']}\n"
                        f"  expected:   {r['expected_source_id'] or '(none)'}\n"
                        f"  evidence:   {r['evidence_ids'][:3]}\n"
                        f"  cited:      {r.get('clause_cited', '')}\n"
                        f"  explanation: {r.get('reason', '')}\n"
                        f"  outcome:    {r['outcome']}\n\n")
        print(f"Wrote {min(args.show, len(records))} full explanations to {out}\n")

    errors = [r for r in records if r["error"]]
    print(format_gate_report(summarize_gate(records)))
    if errors:
        print(f"\n!! {len(errors)} row(s) raised an error. First: {errors[0]['query_id']}: {errors[0]['error']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row["evidence_ids"] = ";".join(r.get("evidence_ids", []))
            w.writerow(row)
    print(f"\nSaved {len(records)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
