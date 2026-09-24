"""
DAY 3 DELIVERABLE (M2): Integration proof.

Gate: "Real text -> real risk prediction -> CaseState."

Loads the trained artifact from models/risk_classifier_tfidf_logreg.pkl, wraps it
in agents.risk_agent_adapter.RiskAgentAdapter (M4's integration point, built
Day 1-2 to accept exactly this), and runs it against real fintech sentences
that are NOT in the training data — proving the pipeline now produces real
model-driven predictions on CaseState instead of MockRiskClassifier's
keyword rules.

Usage:
    python scripts/demo_trained_risk_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.risk_agent_adapter import RiskAgentAdapter  # noqa: E402
from core.schemas import CaseState  # noqa: E402
from services.classifier.baseline import TfidfLogisticClassifier  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"

# Held-out sentences, deliberately NOT present in data/raw/labeled_examples_m1.csv.
DEMO_SENTENCES = [
    "Our app promises you will double your money within six months, guaranteed.",
    "We share your loan repayment history with the credit bureaus as required by law.",
    "aapka paisa hamare paas bilkul surakshit hai aur hum ise kisi tarah ka risk nahi dete",
]


def main() -> None:
    print(f"Loading trained classifier from {MODEL_PATH} ...")
    clf = TfidfLogisticClassifier().load(str(MODEL_PATH))
    agent = RiskAgentAdapter(classifier=clf)
    print("RiskAgentAdapter wired with the trained TfidfLogisticClassifier.\n")

    for sentence in DEMO_SENTENCES:
        state = CaseState(input_text=sentence)
        state = agent.classify(state)

        print("-" * 60)
        print(f"Input text:  {sentence}")
        print(f"risk_label:              {state.risk_label}")
        print(f"classifier_confidence:   {state.classifier_confidence:.4f}")
        print(f"regulator:               {state.regulator}")
        print(f"risk_phrase:             {state.risk_phrase}")
        print(f"trace events logged:     {len(state.trace)}")

    print("-" * 60)
    print("Gate met: real text -> real risk prediction -> CaseState.")


if __name__ == "__main__":
    main()
