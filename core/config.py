"""
RegLingua — Runtime configuration (Day 10-11, M4)
=====================================================
Single place for the knobs that decide *how* the system runs, so nothing
else in the codebase reads environment variables directly.

Everything defaults to **demo mode**: fully offline, deterministic, using
only local cached artifacts (`models/*.pkl`, `data/regulatory/*.json`).
No network call is ever needed for a complete end-to-end run -- that is the
Day 11 gate ("Add demo mode with local cached sources/models") and the
plan's definition of done ("The project runs in demo mode without depending
on a live external service").

Environment variables (all optional):

    REGLINGUA_DEMO_MODE        "1" (default) | "0"
                               1 -> never call an external LLM, even if a
                                    key is present in the environment.
    REGLINGUA_LLM_ENABLED      "0" (default) | "1"
                               Opt in to the LLM-backed Explainer. Ignored
                               while demo mode is on.
    ANTHROPIC_API_KEY          Only read by agents/llm_agents.py when the
                               two flags above allow it.
    REGLINGUA_AGENT_TIMEOUT_S  per-agent-call timeout, default 5.0
    REGLINGUA_AGENT_RETRIES    retries after the first failure, default 1
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

RISK_MODEL_PATH = MODELS_DIR / "risk_classifier_tfidf_logreg.pkl"
REGULATOR_MODEL_PATH = MODELS_DIR / "regulator_classifier.pkl"
M1_REGULATORY_KB_PATH = DATA_DIR / "regulatory" / "regulatory_kb_m1.json"
LABELED_EXAMPLES_PATH = DATA_DIR / "raw" / "labeled_examples_m1.csv"
QUERY_BRIDGE_PATH = DATA_DIR / "processed" / "query_bridge_m1.json"
CANONICAL_CASES_PATH = DATA_DIR / "demo" / "canonical_cases.json"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    demo_mode: bool
    llm_enabled: bool
    agent_timeout_s: float
    agent_max_retries: int

    @property
    def llm_allowed(self) -> bool:
        """An external LLM may be called only when explicitly enabled AND
        demo mode is off."""
        return self.llm_enabled and not self.demo_mode


def get_settings() -> Settings:
    """Read settings fresh from the environment on every call so tests (and
    operators) can flip a flag without reloading modules."""
    return Settings(
        demo_mode=_env_bool("REGLINGUA_DEMO_MODE", True),
        llm_enabled=_env_bool("REGLINGUA_LLM_ENABLED", False),
        agent_timeout_s=_env_float("REGLINGUA_AGENT_TIMEOUT_S", 5.0),
        agent_max_retries=max(0, _env_int("REGLINGUA_AGENT_RETRIES", 1)),
    )
