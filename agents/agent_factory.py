"""
RegLingua — Agent Factory
============================
Owned by: M4 (wiring), consuming M1/M2/M3 components.

Single place that assembles the concrete agent roster the Supervisor's
graph calls into. This is the direct replacement for the old
`agents.mock_agents.build_mock_agents()` factory: as of the Day 1-2
integration merge, RiskAgent, EvidenceAgent and the language detector are
real M2/M3 components (via adapters); Explainer, Rewriter and Verifier
remain mocks until Day 6-8 per the 14-day plan.

Nothing in `agents/supervisor.py` changed to make this swap — it already
only depends on the `agents.contracts` Protocols, which is exactly the
point of defining agent contracts on Day 1 before any real implementation
existed.

Day 3-5 update (M4): M2 has now actually trained and committed
`models/risk_classifier_tfidf_logreg.pkl` (see `scripts/
train_baseline_classifier.py` and `docs/model_card_baseline_m2.md`) and M1's
real 308-chunk knowledge base has been the default regulatory corpus since
the Day 1-2 merge. `build_agents(use_trained_artifacts=True)` is the M4
switch that wires the Supervisor to these real artifacts instead of the
Day 1-2 mocks — see its docstring below for exactly what changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.evidence_agent_adapter import EvidenceAgentAdapter
from agents.explainer_agent import ExplainerAgent as RealExplainerAgent
from agents.language_agent_adapter import LanguageAgentAdapter
from agents.mock_agents import MockExplainerAgent, MockRewriterAgent, MockVerifierAgent
from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.rewriter_agent import RewriterAgent as RealRewriterAgent
from agents.risk_agent_adapter import RiskAgentAdapter
from agents.verifier_agent import VerifierAgent as RealVerifierAgent
from services.classifier.baseline import TfidfLogisticClassifier
from services.classifier.interface import MockRiskClassifier, RiskClassifier
from services.classifier.regulator_classifier import RegulatorClassifier
from core.config import (
    LABELED_EXAMPLES_PATH,
    QUERY_BRIDGE_PATH,
    get_settings,
)
try:  # optional, non-M4 components (rule guardrail, bridged retriever): degrade gracefully if absent
    from services.classifier.rule_augmented import RuleAugmentedClassifier
except ImportError:  # pragma: no cover
    RuleAugmentedClassifier = None
from services.retrieval.deterministic_retriever import DeterministicRetriever
try:
    from services.retrieval.bridged_retriever import BridgedHybridRetriever
    from services.retrieval.query_bridge import load_default_bridge
except ImportError:  # pragma: no cover
    BridgedHybridRetriever = None
    load_default_bridge = None

REPO_ROOT = Path(__file__).resolve().parents[1]

# M1's real, ingested regulatory knowledge base (308 chunks across
# SEBI/RBI/DPDP, extracted from RegLingua_M1_FINAL_v2.xlsx). Used in place
# of M3's original 10-entry demo corpus once you want the pipeline running
# against real regulatory text instead of the smoke-test fixture.
M1_REGULATORY_KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"

# M2's Day 3 trained artifact (see scripts/train_baseline_classifier.py).
# Gitignored — a fresh checkout needs to run
# `scripts/extract_m1_labeled_csv.py` then `scripts/train_baseline_classifier.py`
# to regenerate it locally before `use_trained_artifacts=True` can load it.
RISK_MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"

# M4's Day 3 gap-filler (see services/classifier/regulator_classifier.py's
# module docstring for why this exists): M2's TfidfLogisticClassifier is
# single-task (risk_level only) and never populates
# RiskPrediction.regulator_candidates. Built by
# `scripts/train_regulator_classifier.py`, from the same
# data/raw/labeled_examples_m1.csv M2's extraction script produces (it
# already carries per-example SEBI/RBI/DPDP labels).
REGULATOR_MODEL_PATH = REPO_ROOT / "models" / "regulator_classifier.pkl"


def build_agents(
    classifier: Optional[RiskClassifier] = None,
    regulator_classifier: Optional[RegulatorClassifier] = None,
    corpus_path: Optional[Path] = None,
    use_trained_artifacts: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Build the full agent roster used by the Supervisor graph.

    Args:
        classifier: M2's risk classifier. Defaults to `MockRiskClassifier`
            (Day 1-2 dependency-free keyword mock) unless
            `use_trained_artifacts` says otherwise.
        regulator_classifier: fills `regulator_candidates` when `classifier`
            doesn't supply any — see `agents/risk_agent_adapter.py`.
        corpus_path: regulatory corpus for M3's retriever. Defaults to
            M3's small demo corpus (`data/regulatory/sample_corpus.json`,
            used by M3's own unit tests) unless `use_trained_artifacts` says
            otherwise. Pass `M1_REGULATORY_KB_PATH` explicitly to run
            against M1's real 308-chunk knowledge base regardless.
        use_trained_artifacts: Day 3-5 production switch. When True and
            `classifier`/`regulator_classifier` weren't passed explicitly,
            auto-loads `models/risk_classifier_tfidf_logreg.pkl` /
            `models/regulator_classifier.pkl` if present (falling back to
            the Day 1-2 mocks with a note on stderr if they aren't — e.g. a
            fresh checkout that hasn't run the training scripts yet), and
            defaults `corpus_path` to `M1_REGULATORY_KB_PATH`. Also swaps in
            the real Day 6-8 `ExplainerAgent`/`RewriterAgent`/`VerifierAgent`
            (agents/explainer_agent.py, rewriter_agent.py, verifier_agent.py)
            in place of the Day 1-2 mocks. Kept opt-in rather than the new
            default so the existing Day 1-2 unit/integration tests (which
            assert the deterministic mocks' exact behaviour) keep passing
            unchanged; `apps/api/main.py` and `agents/supervisor.py`'s
            production graph both opt in.
    """
    if classifier is None and use_trained_artifacts:
        if RISK_MODEL_PATH.exists():
            classifier = TfidfLogisticClassifier().load(str(RISK_MODEL_PATH))
        else:
            print(
                f"[agent_factory] {RISK_MODEL_PATH} not found — falling back to "
                "MockRiskClassifier. Run scripts/extract_m1_labeled_csv.py then "
                "scripts/train_baseline_classifier.py to use the real classifier.",
            )

    if regulator_classifier is None and use_trained_artifacts and REGULATOR_MODEL_PATH.exists():
        regulator_classifier = RegulatorClassifier().load(str(REGULATOR_MODEL_PATH))

    if corpus_path is None and use_trained_artifacts:
        corpus_path = M1_REGULATORY_KB_PATH

    base_classifier = classifier or MockRiskClassifier()
    fallback_risk_agent = None
    if production:
        # Day 10: rule guardrail (can only raise risk) around the weak ML
        # baseline. The un-augmented baseline stays as the resilience fallback.
        fallback_risk_agent = RiskAgentAdapter(
            classifier=base_classifier, regulator_classifier=regulator_classifier
        )
        if RuleAugmentedClassifier is not None:
            base_classifier = RuleAugmentedClassifier(base_classifier)

    risk_agent = RiskAgentAdapter(
        classifier=base_classifier,
        regulator_classifier=regulator_classifier,
    )

    deterministic = DeterministicRetriever(corpus_path=corpus_path) if corpus_path else DeterministicRetriever()
    retriever: object = deterministic
    fallback_evidence_agent = None
    if production and BridgedHybridRetriever is not None:
        try:
            bridge = load_default_bridge(LABELED_EXAMPLES_PATH, QUERY_BRIDGE_PATH)
            retriever = BridgedHybridRetriever(corpus_path=corpus_path, bridge=bridge)
            fallback_evidence_agent = EvidenceAgentAdapter(
                research_agent=RegulatoryResearchAgent(retriever=deterministic)
            )
        except Exception as exc:  # noqa: BLE001 - never fail startup on an optional upgrade
            print(f"[agent_factory] hybrid retriever unavailable ({exc}); using deterministic retriever.")
            retriever = deterministic
    research_agent = RegulatoryResearchAgent(retriever=retriever)
    evidence_agent = EvidenceAgentAdapter(research_agent=research_agent)

    if use_trained_artifacts:
        explainer_agent: object = RealExplainerAgent()
        rewriter_agent: object = RealRewriterAgent()
        verifier_agent: object = RealVerifierAgent(risk_agent=risk_agent)
    else:
        explainer_agent = MockExplainerAgent()
        rewriter_agent = MockRewriterAgent()
        verifier_agent = MockVerifierAgent(risk_agent=risk_agent)

    fallback_explainer_agent = None
    if production:
        fallback_explainer_agent = explainer_agent
        settings = get_settings()
        if settings.llm_allowed:
            try:
                from agents.llm_agents import AnthropicClient, LLMExplainerAgent

                explainer_agent = LLMExplainerAgent(AnthropicClient())
            except Exception as exc:  # noqa: BLE001
                print(f"[agent_factory] LLM explainer unavailable ({exc}); using deterministic explainer.")
                fallback_explainer_agent = None

    roster = {
        "language_detector": LanguageAgentAdapter(),
        "risk_agent": risk_agent,
        "research_agent": research_agent,
        "evidence_agent": evidence_agent,
        "explainer_agent": explainer_agent,
        "rewriter_agent": rewriter_agent,
        "verifier_agent": verifier_agent,
    }
    if production:
        roster["fallback_risk_agent"] = fallback_risk_agent
        roster["fallback_evidence_agent"] = fallback_evidence_agent
        roster["fallback_explainer_agent"] = fallback_explainer_agent
    return roster
