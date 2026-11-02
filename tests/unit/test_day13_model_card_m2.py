"""Day 13 (M2): the model card is generated from result files and stays in sync with them."""
import re
from pathlib import Path

import pytest

from services.evaluation.model_card_m2 import SOURCES, build_model_card, need

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "model_card_baseline_m2.md"


def test_committed_card_matches_regeneration():
    assert CARD.read_text(encoding="utf-8") == build_model_card(ROOT), "run: python scripts/build_model_card_m2.py"


def test_required_sections_and_disclaimer():
    t = CARD.read_text(encoding="utf-8")
    assert "Not legal advice and not compliance certification" in t
    for h in ("## Changes since Day 3", "## 1. Model details", "## 2. Intended use", "## 3. Out-of-scope", "## 4. Factors",
              "## 5. Metrics", "## 6. Evaluation data", "## 7. Quantitative results", "## 8. Calibration", "## 9. Robustness",
              "## 10. Fallback", "## 11. Ethical", "## 12. Caveats", "## Sources"):
        assert h in t, h


def test_every_cited_source_id_is_defined_and_files_exist():
    t = CARD.read_text(encoding="utf-8")
    defined = {sid for sid, _, _ in SOURCES}
    used = set(re.findall(r"S\d+", t.split("## Sources")[0]))
    assert used <= defined
    for sid, path, _ in SOURCES:
        p = path.split(" ")[0]
        if p.startswith(("data/", "results/", "models/", "docs/", "services/")) and "history" not in path:
            assert (ROOT / p).exists(), f"{sid}: {p}"


def test_need_rejects_missing_figure():
    with pytest.raises(ValueError):
        need("abc", "xyz")
