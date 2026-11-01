"""Regenerate core/contract_snapshot.json. Run ONLY when deliberately changing
the agent contract, and bump core.versions.CONTRACT_VERSION in the same commit."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import schemas, versions  # noqa: E402

MODELS = ["CaseState", "EvidenceItem", "EvidenceRequest", "EvidenceResponse",
          "ExplainerRequest", "ExplainerResponse", "RewriteRequest", "RewriteResponse", "AgentEvent"]


def build() -> dict:
    return {
        "contract_version": versions.CONTRACT_VERSION,
        "models": {n: sorted(getattr(schemas, n).model_fields) for n in MODELS},
        "enums": {n: sorted(m.value for m in getattr(schemas, n))
                  for n in ("RiskLabel", "Regulator", "Language", "ResolutionStatus", "EventType")},
    }


if __name__ == "__main__":
    (ROOT / "core" / "contract_snapshot.json").write_text(json.dumps(build(), indent=2, sort_keys=True), encoding="utf-8")
    print("wrote core/contract_snapshot.json")
