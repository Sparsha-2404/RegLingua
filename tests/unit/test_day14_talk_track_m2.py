"""Day 14 (M2): talk track is generated from result files and contains the required parts."""
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs" / "demo_talk_track_m2.md"


def test_talk_track_structure():
    t = DOC.read_text(encoding="utf-8")
    assert "not legal certification" in t and "| 1 | | | | | | |" in t and "| 3 | | | | | | |" in t
    assert t.count("\n**Q") + t.startswith("**Q") == 15
    for h in ("## 3. Say / do not say", "## 5. Known weaknesses"):
        assert h in t
    qa = t.split("## 4.")[1].split("## 5.")[0]
    assert qa.count("[") >= 15  # every answer cites a source in brackets
