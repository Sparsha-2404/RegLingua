"""
DAY 6 DELIVERABLE (M2): evidence retrieval measured the way the Supervisor
actually runs it, not the Day 5 oracle way.

Thin CLI wrapper around services.evaluation.pipeline_retrieval (the metric
and attribution logic lives there and is unit-tested with fakes). This
script only wires in the REAL components:

    RiskAgentAdapter(TfidfLogisticClassifier, RegulatorClassifier)   -> predicted regulator, risk_phrase
    agents.supervisor.route_after_classification                     -> needs_review / compliant / abstain
    EvidenceAgentAdapter(RegulatoryResearchAgent(retriever=...))     -> evidence, built from the same
                                                                        EvidenceRequest the Supervisor builds

Configs (the first three are the Day 6 prompt; D isolates the query effect):
    A  keyword retriever + risk_phrase query   <- what agent_factory.py / the Supervisor do today
    B  fixed HybridRetriever + risk_phrase query
    C  fixed HybridRetriever + full-statement query
    D  keyword retriever + full-statement query

Prerequisites (run from the repo root):
    python scripts/extract_m1_labeled_csv.py
    python scripts/train_baseline_classifier.py        (or the committed models/risk_classifier_tfidf_logreg.pkl)
    python scripts/train_regulator_classifier.py       (-> models/regulator_classifier.pkl)
    python scripts/build_risk_evidence_eval.py

Usage:
    python scripts/evaluate_pipeline_retrieval.py
    python scripts/evaluate_pipeline_retrieval.py --configs A,D        # keyword only, no model download
    python scripts/evaluate_pipeline_retrieval.py --limit 20           # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.evaluation.dataset_join import normalize_text, read_csv_rows  # noqa: E402
from services.evaluation.pipeline_retrieval import (  # noqa: E402
    QUERY_FULL_TEXT,
    QUERY_RISK_PHRASE,
    ClassificationOutcome,
    Hit,
    RetrievalConfig,
    evaluate_pipeline,
    format_report,
    write_per_row_csv,
)

JOINED_PATH = REPO_ROOT / "data" / "evaluation" / "risk_evidence_eval_m2.csv"
LABELED_PATH = REPO_ROOT / "data" / "raw" / "labeled_examples_m1.csv"
KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"
RISK_MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
REGULATOR_MODEL_PATH = REPO_ROOT / "models" / "regulator_classifier.pkl"
OUT_PATH = REPO_ROOT / "data" / "evaluation" / "pipeline_retrieval_results_m2.csv"

CONFIG_SPECS = {
    # name: (retriever kind, query mode, description)
    "A": ("keyword", QUERY_RISK_PHRASE,
          "keyword retriever + risk_phrase query (what the Supervisor / agent_factory.py do today)"),
    "B": ("hybrid", QUERY_RISK_PHRASE, "fixed HybridRetriever + risk_phrase query"),
    "C": ("hybrid", QUERY_FULL_TEXT, "fixed HybridRetriever + full-statement query"),
    "D": ("keyword", QUERY_FULL_TEXT, "keyword retriever + full-statement query"),
}


def require_file(path: Path, how_to_create: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {path}\n  -> {how_to_create}")


def held_out_texts() -> set[str]:
    """Texts neither classifier trained on: the same split both training
    scripts use (stratified on risk_level, test_size=0.2, random_state=42)."""
    from services.classifier.dataset import load_dataset, train_test_split_examples

    examples = load_dataset(str(LABELED_PATH))
    _, test = train_test_split_examples(examples, test_size=0.2, random_state=42)
    return {normalize_text(ex.text) for ex in test}


def build_classify():
    """Real RiskAgentAdapter + the Supervisor's own routing function."""
    from agents.risk_agent_adapter import RiskAgentAdapter
    from agents.supervisor import route_after_classification
    from core.schemas import CaseState, Language
    from services.classifier.baseline import TfidfLogisticClassifier
    from services.classifier.regulator_classifier import RegulatorClassifier

    # Load explicitly and fail loudly: agent_factory silently falls back to
    # mocks when a model file is missing, which would invalidate every number.
    risk_agent = RiskAgentAdapter(
        classifier=TfidfLogisticClassifier().load(str(RISK_MODEL_PATH)),
        regulator_classifier=RegulatorClassifier().load(str(REGULATOR_MODEL_PATH)),
    )

    def classify(row):
        state = CaseState(input_text=row["text"], language=Language(row["language"]))
        state = risk_agent.classify(state)
        return ClassificationOutcome(
            risk_label=state.risk_label.value,
            confidence=float(state.classifier_confidence or 0.0),
            regulator=state.regulator.value,
            risk_phrase=state.risk_phrase,
            route=route_after_classification(state),
        )

    return classify


def build_retrievers(kinds: set[str]) -> dict:
    from services.retrieval.deterministic_retriever import DeterministicRetriever

    retrievers = {}
    if "keyword" in kinds:
        retrievers["keyword"] = DeterministicRetriever(corpus_path=KB_PATH)
    if "hybrid" in kinds:
        from services.embeddings.vector_index import RegulatoryVectorIndex
        from services.retrieval.hybrid_retriever import HybridRetriever

        retrievers["hybrid"] = HybridRetriever(
            keyword_retriever=DeterministicRetriever(corpus_path=KB_PATH),
            vector_index=RegulatoryVectorIndex(corpus_path=KB_PATH),
        )
    return retrievers


def make_retrieve(retriever):
    """Go through EvidenceAgentAdapter with the same EvidenceRequest the
    Supervisor's gather_evidence node builds (default top_k=5)."""
    from agents.evidence_agent_adapter import EvidenceAgentAdapter
    from agents.regulatory_research_agent import RegulatoryResearchAgent
    from core.schemas import EvidenceRequest, Language, Regulator

    adapter = EvidenceAgentAdapter(research_agent=RegulatoryResearchAgent(retriever=retriever))

    def retrieve(query_text, regulator, language, query_id):
        hint = Regulator.GENERAL if regulator == "GENERAL" else Regulator(regulator)
        request = EvidenceRequest(
            case_id=query_id,
            flagged_text=query_text,
            language=Language(language),
            regulator_hint=hint,
        )
        response = adapter.gather(request)
        return [Hit(item.clause_id, item.similarity_score) for item in response.evidence]

    return retrieve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="A,B,C,D",
                        help="Comma-separated subset of A,B,C,D (default: all). "
                             "B and C need the one-time embedding-model download.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke test).")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    names = [c.strip().upper() for c in args.configs.split(",") if c.strip()]
    unknown = [n for n in names if n not in CONFIG_SPECS]
    if unknown:
        raise SystemExit(f"Unknown config(s) {unknown}; choose from {sorted(CONFIG_SPECS)}")

    require_file(JOINED_PATH, "run: python scripts/build_risk_evidence_eval.py")
    require_file(LABELED_PATH, "run: python scripts/extract_m1_labeled_csv.py")
    require_file(RISK_MODEL_PATH, "run: python scripts/train_baseline_classifier.py")
    require_file(REGULATOR_MODEL_PATH,
                 "run: python scripts/train_regulator_classifier.py (set PYTHONPATH=%CD% first on Windows)")
    require_file(KB_PATH, "regulatory KB missing")

    _, rows = read_csv_rows(JOINED_PATH)
    if args.limit:
        rows = rows[: args.limit]

    retrievers = build_retrievers({CONFIG_SPECS[n][0] for n in names})
    configs = [
        RetrievalConfig(name=n, description=CONFIG_SPECS[n][2], query_mode=CONFIG_SPECS[n][1],
                        retrieve=make_retrieve(retrievers[CONFIG_SPECS[n][0]]))
        for n in names
    ]

    held = held_out_texts()
    print(f"Rows: {len(rows)} | configs: {','.join(names)} | held-out texts: {len(held)}\n")

    result = evaluate_pipeline(
        rows, build_classify(), configs,
        held_out_keys=held, key_fn=normalize_text,
    )
    print(format_report(result, {c.name: c.description for c in configs}))

    write_per_row_csv(result, args.out)
    print(f"\nSaved {len(result.per_row)} per-row results to {args.out}")
    print(
        "\nNotes: (1) the labelled language is passed to retrieval; the Supervisor's language-detection\n"
        "node is skipped because no retriever uses it. (2) 'held-out' rows were not used to train either\n"
        "classifier, but English/Hindi/hi-en versions of one statement can sit on opposite sides of the\n"
        "split, so held-out numbers are still somewhat optimistic. (3) A row counts toward retrieval\n"
        "metrics only if it has an expected_source_id (510 of 516)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
