import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_rewrite_failures import build_categories, inspect_grammar


def test_detects_doubled_function_word():
    reasons = inspect_grammar("This is an investment.", "This is is an investment.")
    assert any(reason.startswith("doubled_function_word:is") for reason in reasons)


def test_detects_truncated_tail():
    reasons = inspect_grammar("A complete sentence is here.", "This is subject to")
    assert any(reason.startswith("possible_truncated_tail:") for reason in reasons)


def test_detects_length_ratio():
    reasons = inspect_grammar("one two three four five six", "one")
    assert any(reason.startswith("length_ratio_out_of_range:") for reason in reasons)


def test_flags_unchanged_and_high_after_label():
    rows = [{
        "row_id": "X1", "original_text": "Original statement",
        "rewritten_text": "Original statement", "before_label": "HIGH",
        "after_label": "HIGH",
    }]
    result = build_categories(rows)
    assert len(result["unchanged"]) == 1
    assert len(result["still_medium_high"]) == 1


def test_grammar_flag_does_not_require_classifier_label():
    rows = [{
        "row_id": "X2", "original_text": "We provide a complete statement",
        "rewritten_text": "We provide a subject to market risks invest",
        "before_label": "HIGH", "after_label": "LOW",
    }]
    result = build_categories(rows)
    assert len(result["grammar_suspicious"]) == 1


def test_category_logic_retains_all_flags_before_top_five_export():
    rows = [
        {"row_id": str(i), "original_text": "Same", "rewritten_text": "Same",
         "before_label": "HIGH", "after_label": "HIGH"}
        for i in range(8)
    ]
    result = build_categories(rows)
    assert len(result["unchanged"]) == 8
