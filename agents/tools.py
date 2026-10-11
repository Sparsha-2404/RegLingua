"""
RegLingua — Agent Tool Interface (Day 4, M4)
=================================================
Owned by: M4

Day 4 plan line: "M4: Add ResearchAgent tool interface."
Gate: "Query retrieves source-linked regulatory chunks."

Phase A of the architecture doc builds every stage as a deterministic
function/class behind a stable contract; Phase B replaces selected stages
with AI agents "one by one" (Section 4). For that swap to be possible
without rewriting the Supervisor, every stage needs to be reachable through
one uniform shape — name in, structured JSON out — the same way an LLM
planner calls a tool. This module is that seam.

A `Tool` wraps one agent method behind:
  - a stable `name` (what a planner/route table refers to it by)
  - a `description` (what an LLM tool-call planner would read)
  - an `input_schema` (JSON-schema-shaped dict — informational today, but
    exactly what an OpenAI/Anthropic tool-use `tools` block or LangGraph
    ToolNode expects verbatim later, so no reshaping is needed at that
    point)
  - a single `run(**kwargs) -> dict` entrypoint that returns
    JSON-serializable data, regardless of whether the thing underneath
    speaks dataclasses (M3's `RegulatoryResearchAgent`), pydantic
    (M4's `EvidenceAgent`), or a plain dict.

`ToolRegistry` collects a named set of tools and is what `apps/api/main.py`
exposes over HTTP (`/tools`, `/tools/{name}/invoke`) and what a future
AI-planning Supervisor (Section 9's "Supervisor can be upgraded to an AI
planning agent") would be handed as its callable tool list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from agents.evidence_agent_adapter import EvidenceAgentAdapter
from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.research_contracts import RegulatoryResearchInput, RiskPrediction
from agents.risk_agent_adapter import RiskAgentAdapter
from core.schemas import CaseState, EvidenceRequest, Language, Regulator


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    run: Callable[..., dict]


class ToolRegistry:
    """Name -> Tool lookup, plus a uniform `call()` entrypoint."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def call(self, name: str, **kwargs: Any) -> dict:
        return self.get(name).run(**kwargs)


# ---------------------------------------------------------------------------
# Tool factories — one per Phase-A-deterministic stage that Phase B would
# eventually replace with (or route through) an AI agent.
# ---------------------------------------------------------------------------

def make_research_tool(research_agent: RegulatoryResearchAgent) -> Tool:
    """Day 4 gate: "Query retrieves source-linked regulatory chunks."

    Callable directly with just a query string — the case/risk-prediction
    scaffolding `RegulatoryResearchInput` normally needs is filled with
    permissive defaults so this tool is usable standalone (e.g. from the
    API or a notebook), not only from inside a running case.
    """

    def run(
        query: str,
        regulator_candidates: Optional[list[str]] = None,
        topic_hint: Optional[str] = None,
        top_k: int = 5,
        case_id: str = "adhoc",
    ) -> dict:
        request = RegulatoryResearchInput(
            case_id=case_id,
            normalized_text=query,
            language="en",
            risk_prediction=RiskPrediction(label="HIGH", confidence=1.0, factors=[]),
            regulator_candidates=regulator_candidates or [],
            topic_hint=topic_hint,
            top_k=top_k,
        )
        output = research_agent.run(request)
        # `RegulatoryResearchOutput.to_json()` returns a JSON *string*
        # (for logging the agent_trace verbatim); a Tool must return a
        # JSON-serializable dict, and callers want a flat "results" list
        # rather than reaching through .result.retrieval_results, so this
        # reshapes rather than calling to_json() directly.
        return {
            "status": output.status,
            "confidence": output.confidence,
            "results": [asdict(chunk) for chunk in output.result.retrieval_results],
        }

    return Tool(
        name="regulatory_research",
        description=(
            "Retrieve regulatory passages (SEBI/RBI/DPDP) relevant to a query. "
            "Returns source-linked chunks (source_id, regulator, document_title, "
            "section, text, score) — abstains (empty results, status='failure') "
            "rather than inventing a citation when nothing matches."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to find supporting regulation for."},
                "regulator_candidates": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["SEBI", "RBI", "DPDP"]},
                    "description": "Optional metadata filter, applied before scoring.",
                },
                "topic_hint": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        run=run,
    )


def make_risk_classification_tool(risk_agent: RiskAgentAdapter) -> Tool:
    """Wraps M2's RiskAgent (via the M4 adapter) as a standalone tool."""

    def run(text: str) -> dict:
        probe = risk_agent.classify(CaseState(input_text=text))
        return {
            "risk_label": probe.risk_label.value if probe.risk_label else None,
            "regulator": probe.regulator.value if probe.regulator else None,
            "confidence": probe.classifier_confidence,
            "risk_phrase": probe.risk_phrase,
        }

    return Tool(
        name="risk_classification",
        description="Classify fintech product text as Compliant/Risky/Ambiguous and infer the likely regulator.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        run=run,
    )


def make_evidence_tool(evidence_agent: EvidenceAgentAdapter) -> Tool:
    """Day 5 gate: "EvidenceAgent returns selected chunks with source IDs."

    Case-scoped wrapper around the research tool (see
    `agents/evidence_agent_adapter.py`) — this is the tool a Supervisor
    would call once a case already has a risk prediction, as opposed to
    `regulatory_research`, which is usable standalone.
    """

    def run(
        text: str,
        language: str = "en",
        regulator_hint: Optional[str] = None,
        top_k: int = 5,
        case_id: str = "adhoc",
    ) -> dict:
        request = EvidenceRequest(
            case_id=case_id,
            flagged_text=text,
            language=Language(language),
            regulator_hint=Regulator(regulator_hint) if regulator_hint else None,
            top_k=top_k,
        )
        response = evidence_agent.gather(request)
        return response.model_dump()

    return Tool(
        name="evidence_gathering",
        description=(
            "Case-scoped evidence lookup: given flagged text, returns selected "
            "regulatory clauses (each with a clause_id / source ID) supporting a "
            "compliance explanation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "language": {"type": "string", "enum": ["en", "hi", "hi-en"], "default": "en"},
                "regulator_hint": {"type": "string", "enum": ["SEBI", "RBI", "DPDP", "General"]},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["text"],
        },
        run=run,
    )


def build_tool_registry(agents: dict[str, object]) -> ToolRegistry:
    """Build the standard tool registry from an `agent_factory.build_agents()`
    roster. `agents` must contain at least `research_agent`, `risk_agent` and
    `evidence_agent`."""
    registry = ToolRegistry()
    registry.register(make_research_tool(agents["research_agent"]))
    registry.register(make_risk_classification_tool(agents["risk_agent"]))
    registry.register(make_evidence_tool(agents["evidence_agent"]))
    return registry
