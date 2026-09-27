"""
LanguageAgentAdapter — Day 1/2 integration glue between M2 and M4.
=====================================================================

M4's original Day 2 mock (`MockLanguageDetector`) and M2's Day 2
`services.classifier.preprocessing.detect_language` did the exact same
thing independently: check for Devanagari Unicode code points. Rather than
ship two copies of the same rule, the Supervisor's `detect_language` node
now calls this thin adapter around M2's real implementation — the one that
also backs the classifier's own preprocessing step, so both stages agree on
what "language: hi" means for a given piece of text.
"""

from __future__ import annotations

from services.classifier.preprocessing import detect_language


class LanguageAgentAdapter:
    """Implements `agents.contracts.LanguageDetectorAgent` on top of M2's
    preprocessing module."""

    def detect(self, text: str) -> str:
        return detect_language(text)
