"""Day 10 (M2) Part 2 runner: end-to-end with a FAKE encoder on synthetic CSVs (no HF)."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HI = ["guaranteed returns zero risk", "assured profit guaranteed income", "risk free guaranteed gains"]
ME = ["investment advice for your goals", "consult expert for plans", "advice on financial goals"]
LO = ["track monthly spending budget", "read terms and conditions", "view account statement summary"]
KW = {"guaranteed", "assured", "risk", "zero", "free", "profit", "returns", "gains"}
MID = {"advice", "consult", "expert", "goals", "plans", "financial", "investment"}


class FakeEncoder:
    model_name = "fake-enc"

    def encode(self, texts):
        out = []
        for t in texts:
            w = t.lower().split()
            h = hashlib.md5(t.encode()).digest()
            out.append([sum(x in KW for x in w), sum(x in MID for x in w)] + [b / 255 * 0.05 for b in h[:3]])
        return np.array(out)


def _load():
    spec = importlib.util.spec_from_file_location(
        "ev2", Path(__file__).resolve().parents[2] / "scripts" / "evaluate_day10_model_comparison_m2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # must import without sentence_transformers
    return m


def test_run_end_to_end_leak_free_and_strict_json(tmp_path):
    m = _load()
    rows, regs = [], ["RBI", "SEBI", "DPDP"]
    for i in range(72):  # 36 originals x (en, hi) translation pairs
        pool, lab = [(HI, "HIGH"), (ME, "MEDIUM"), (LO, "LOW")][(i // 2) % 3]
        rows.append({"query_id": f"q{i}", "original_id": f"o{i // 2}", "text": f"{pool[(i // 2) % 3]} v{i // 2}x{i % 2}",
                     "language": "en" if i % 2 == 0 else "hi", "true_regulator": regs[(i // 2) % 3],
                     "true_risk_level": lab})
    ev, held = tmp_path / "e.csv", tmp_path / "h.csv"
    with open(ev, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(held, "w", newline="", encoding="utf-8") as f:  # hold out the 'hi' half of 12 pairs
        w = csv.writer(f); w.writerow(["query_id", "held_out"])
        for i in range(72):
            w.writerow([f"q{i}", "True" if (i % 2 == 1 and i // 2 < 12) else "False"])

    rep = m.run(FakeEncoder(), eval_path=ev, held_path=held, out_dir=tmp_path, n_boot=60,
                grid=(1.0,), verbose=False)
    p = rep["protocol"]
    assert p["held_out_rows"] == 12 and p["dropped_train_rows"] == 12 and p["train_rows"] == 48
    assert p["held_out_original_ids_in_train"] == 0 and p["leak_free"] and p["seed"] == 42
    for k in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "by_language", "by_regulator"):
        assert k in rep["metrics"]["baseline"] and k in rep["metrics"]["embedding"]
    assert set(rep["metrics"]["embedding"]["by_regulator"]) == {"RBI", "SEBI", "DPDP"}
    assert rep["paired_macro_f1_bootstrap_embedding_minus_baseline"]["verdict"] in (
        "improvement", "degradation", "no significant difference")
    assert "p_exact" in rep["mcnemar_paired_correctness"]
    # strict JSON (no NaN tokens; absent language 'hi-en' has n=0 -> null)
    txt = (tmp_path / "day10_model_comparison_m2.json").read_text()
    assert "NaN" not in txt
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert rep["metrics"]["baseline"]["by_language"]["hi-en"]["n"] == 0
    assert (tmp_path / "day10_model_comparison_m2.csv").exists()
