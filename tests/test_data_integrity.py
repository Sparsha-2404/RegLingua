import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_dataset import run
from data_utils import load_labelled

def test_integrity():
    assert run() == []

def test_splits_cover_all_languages():
    t = load_labelled("test")
    assert {"en", "hi", "hi-en"} <= set(t.LANGUAGE)
    assert {"Low", "Medium", "High"} <= set(t["RISK LEVEL"])
