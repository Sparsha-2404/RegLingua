"""Render docs/model_card_baseline_m2.md from result files (no hand-typed numbers).

    python scripts\build_model_card_m2.py        (bash: python scripts/build_model_card_m2.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.evaluation.model_card_m2 import build_model_card  # noqa: E402

out = ROOT / "docs" / "model_card_baseline_m2.md"
out.write_text(build_model_card(ROOT), encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)}")
