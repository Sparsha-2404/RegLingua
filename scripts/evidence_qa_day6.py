"""
RegLingua — Day 6 Evidence QA
=================================
Owned by: M1 ("Evidence QA") + M2 ("Validate risk/evidence consistency").

A lightweight, dependency-free consistency check rather than a full audit
tool: for a sample of M1's real labelled dataset, retrieves evidence via the
real `RegulatoryResearchAgent` (M1's 308-chunk KB) using each example's
*true* regulator label as a hint, and reports how often retrieval actually
returns a chunk for the *same* regulator. A low match rate here would mean
the KB or retriever needs attention before trusting the Explainer's
citations; a high one is the Day 6 gate's evidence, not just the code path.

Usage (from repo root, after scripts/extract_m1_labeled_csv.py):
    python scripts/evidence_qa_day6.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from agents.agent_factory import M1_REGULATORY_KB_PATH
from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.research_contracts import RegulatoryResearchInput, RiskPrediction
from services.retrieval.deterministic_retriever import DeterministicRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
SAMPLE_SIZE = 60
SEED = 42


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"{DATASET_CSV} not found. Run scripts/extract_m1_labeled_csv.py first.")

    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.Random(SEED).shuffle(rows)
    sample = rows[:SAMPLE_SIZE]

    agent = RegulatoryResearchAgent(retriever=DeterministicRetriever(corpus_path=M1_REGULATORY_KB_PATH))

    matches = 0
    no_results = 0
    for i, row in enumerate(sample):
        request = RegulatoryResearchInput(
            case_id=f"qa-{i}",
            normalized_text=row["text"],
            language=row.get("language", "en"),
            risk_prediction=RiskPrediction(label=row["risk_level"], confidence=1.0, factors=[]),
            regulator_candidates=[row["regulator"]],
            top_k=3,
        )
        output = agent.run(request)
        chunks = output.result.retrieval_results
        if not chunks:
            no_results += 1
            continue
        if any(c.regulator == row["regulator"] for c in chunks):
            matches += 1

    print(f"Sampled {len(sample)} labelled examples (seed={SEED}).")
    print(f"Retrieval returned >=1 chunk matching the true regulator: {matches}/{len(sample)} ({matches / len(sample):.1%})")
    print(f"Retrieval returned no chunks at all: {no_results}/{len(sample)} ({no_results / len(sample):.1%})")


if __name__ == "__main__":
    main()
