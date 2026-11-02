"""Day 13 (M2): numbers ledger, renderers and README consistency (synthetic data; no real results needed)."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.evaluation import m2_doc_numbers as N
from services.evaluation import m2_docs as M

ROOT = Path(__file__).resolve().parents[2]


def _synthetic(tmp_path):
    d = tmp_path / "data" / "evaluation"; d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 60
    lab = rng.choice(["HIGH", "LOW", "MEDIUM"], n)
    pred = np.where(rng.random(n) < 0.7, lab, rng.choice(["HIGH", "LOW", "MEDIUM"], n))
    mp = rng.uniform(0.35, 0.9, n)
    pd.DataFrame({"query_id": [f"q{i}" for i in range(n)], "original_id": [f"o{i // 2}" for i in range(n)],
                  "split": ["held_out" if i < 20 else "non_held_out" for i in range(n)],
                  "language": ["en" if i % 2 == 0 else "hi" for i in range(n)], "true_risk": lab, "pred_label": pred,
                  "correct": (lab == pred).astype(int), "prob_source": ["committed" if i < 20 else "oof_fold" for i in range(n)],
                  "max_prob": mp}).to_csv(d / "confidence_signals_m2.csv", index=False)
    pd.DataFrame({"operating_point": ["balanced", "permissive", "conservative"], "group_type": ["all"] * 3, "group": ["all"] * 3,
                  "n": [20] * 3, "accept": [10, 12, 6], "review": [5] * 3, "abstain": [5, 3, 9], "coverage": [.5, .6, .3],
                  "coverage_lo": [.3] * 3, "coverage_hi": [.7] * 3, "accuracy_on_accepted": [.7, .6, .8], "acc_lo": [.5] * 3,
                  "acc_hi": [.9] * 3, "false_compliant": [1, 2, 0], "abstain_floor": [.43, .43, .46], "compliant_floor": [.54, .49, 1.0]}
                 ).to_csv(d / "abstention_operating_points_m2.csv", index=False)
    (d / "abstention_thresholds_m2.json").write_text(json.dumps({"verdict": "synthetic verdict"}))
    for name in ("keyword", "hybrid"):
        pd.DataFrame({"query_id": range(30), "language": ["en", "hi", "hi-en"] * 10, "hit_at_1": rng.integers(0, 2, 30),
                      "hit_at_3": 1, "hit_at_5": 1}).to_csv(d / f"retrieval_eval_results_m2_{name}.csv", index=False)
    return tmp_path


def test_ledger_is_deterministic_and_every_entry_has_source(tmp_path):
    root = _synthetic(tmp_path)
    a, b = N.collect_numbers(root), N.collect_numbers(root)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)          # seed 42 -> identical
    assert all(e["source"] and e["derivation"] for e in a.values())
    assert a["clf.heldout.n"]["value"] == 20 and a["clf.oof.n"]["value"] == 40
    lo, v, hi = (a[f"clf.heldout.accuracy{s}"]["value"] for s in (".ci_low", "", ".ci_high"))
    assert lo <= v <= hi and a["abst.balanced.accepted"]["value"] == 10 and a["abst.verdict_text"]["value"] == "synthetic verdict"
    assert "retr.hybrid.recall@1.hi-en" in a and "leak.heldout_ids_with_partner_in_train" in a


def test_missing_inputs_give_absent_keys_not_made_up_values(tmp_path):
    (tmp_path / "data" / "evaluation").mkdir(parents=True)
    assert N.collect_numbers(tmp_path) == {}
    text, missing = M.render_snippets({})
    assert M.NA in text and "clf.heldout.accuracy" in missing and "%" not in text.split("## Sources")[0]   # no formatted value invented


def test_fake_smoke_comparison_file_is_never_cited(tmp_path):
    d = tmp_path / "data" / "evaluation"; d.mkdir(parents=True)
    row = {"mode": "FAKE_SMOKE", "scope": "difference", "group": "all", "metric": "macro_f1_bootstrap", "n": 5, "baseline": .5,
           "embedding": .9, "delta": .4, "ci_low": .1, "ci_high": .7, "note": "improvement"}
    pd.DataFrame([row]).to_csv(d / "model_comparison_m2.csv", index=False)
    assert "cmp.macro_f1.delta" not in N.collect_numbers(tmp_path)
    pd.DataFrame([{**row, "mode": "real"}]).to_csv(d / "model_comparison_m2.csv", index=False)
    assert N.collect_numbers(tmp_path)["cmp.macro_f1.delta"]["value"] == 0.4


def test_rendered_snippets_use_ledger_values_and_list_sources(tmp_path):
    led = N.collect_numbers(_synthetic(tmp_path))
    text, missing = M.render_snippets(led)
    acc = led["clf.heldout.accuracy"]["value"]
    assert f"{100 * acc:.1f}%" in text and "## Sources" in text and "confidence_signals_m2.csv" in text
    assert M.DISCLAIMER in text and "decision-support" in text
    assert "no embedding-model" in text                                                    # A4 fallback paragraph
    used_keys = re.findall(r"\| `([^`]+)` \| `", text.split("## Sources")[1])
    assert "clf.heldout.accuracy" in used_keys and all(k in led for k in used_keys)


def test_script_table_statuses():
    meta = [{"script": "a.py", "purpose": "p", "inputs": "i", "outputs": "o"}, {"script": "b.py", "purpose": "p", "inputs": "i", "outputs": ""},
            {"script": "c.py", "purpose": "p", "inputs": "i", "outputs": "o"}, {"script": "d.py", "purpose": "p", "inputs": "i", "outputs": "o"}]
    rt = {"a.py": {"status": "ok", "seconds": "1.5", "reproduces": "byte-identical"}, "b.py": {"status": "skipped", "reason": "needs HF"},
          "c.py": {"status": "exit_nonzero", "exit_code": 1, "last_line": "boom"}}
    t = M.script_table(meta, rt)
    assert "1.5 s" in t and "byte-identical" in t and "not run: needs HF" in t and "failed here (exit 1): boom" in t and "not measured" in t


# ---- README consistency (documented commands must reference real files) ----
def test_readme_commands_reference_existing_scripts_and_has_both_shells():
    t = (ROOT / "docs" / "README_M2.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"scripts[\\/]([A-Za-z0-9_]+\.py)", t))
    assert refs and all((ROOT / "scripts" / r).is_file() for r in refs), [r for r in refs if not (ROOT / "scripts" / r).is_file()]
    assert "Command Prompt" in t and "bash" in t and "set HF_HUB_OFFLINE=1" in t and "export HF_HUB_OFFLINE=1" in t
    assert "<!-- BEGIN:M2_SCRIPT_TABLE -->" in t and "<!-- END:M2_SCRIPT_TABLE -->" in t
    for h in ("What M2 owns", "Quick start", "Expected outputs", "Troubleshooting", "Limitations"):
        assert h in t
    assert "decision-support" in t and "not certify legal compliance" in t
    code = re.sub(r"(?s)## 5\. Troubleshooting.*?## 6\.", "", t)                      # python3 may be named only in troubleshooting
    assert "python3" not in code.replace("`python3`", "")


def test_m2_script_docstrings_and_help_use_python_not_python3():
    for f in sorted((ROOT / "scripts").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(src)) or ""
        assert not re.search(r"(^|\s)python3\s", doc), f.name
