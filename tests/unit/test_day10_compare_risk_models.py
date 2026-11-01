"""Day 10 (M2): scripts/compare_risk_models.py with FAKE encoder only (no HF)."""
import csv
import importlib.util
from pathlib import Path

import numpy as np

HI = ["guaranteed returns zero risk", "assured profit guaranteed income", "risk free guaranteed gains"]
ME = ["investment advice for your goals", "consult expert for plans", "advice on financial goals"]
LO = ["track monthly spending budget", "read terms and conditions", "view account statement summary"]
KW, MID = {"guaranteed", "assured", "risk", "zero", "free", "profit", "returns", "gains"}, {"advice", "consult", "expert", "goals", "plans"}


class Enc:
    model_name = "test-enc"

    def encode(self, texts):
        return np.array([[sum(w in KW for w in t.split()), sum(w in MID for w in t.split()), len(t) % 7 / 100]
                         for t in texts])


def _mod():
    s = importlib.util.spec_from_file_location("crm", Path(__file__).resolve().parents[2] / "scripts" / "compare_risk_models.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)  # imports without sentence_transformers
    return m


def _data(tmp):
    rows, regs = [], ["RBI", "SEBI", "DPDP"]
    for i in range(72):
        pool, lab = [(HI, "HIGH"), (ME, "MEDIUM"), (LO, "LOW")][(i // 2) % 3]
        rows.append({"query_id": f"q{i}", "original_id": f"o{i // 2}", "text": f"{pool[(i // 2) % 3]} v{i // 2}x{i % 2}",
                     "language": "en" if i % 2 == 0 else "hi", "true_regulator": regs[(i // 2) % 3], "true_risk_level": lab})
    ev, held = tmp / "e.csv", tmp / "h.csv"
    with open(ev, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(held, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_id", "held_out"])
        for i in range(72):
            w.writerow([f"q{i}", "True" if (i % 2 == 1 and i // 2 < 12) else "False"])
    return ev, held


def _rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def test_fake_mode_is_labelled_and_outputs_complete(tmp_path):
    m = _mod(); ev, held = _data(tmp_path)
    r = m.run(Enc(), fake=True, eval_path=ev, held_path=held, out_dir=tmp_path, n_boot=50, grid=(1.0,), verbose=False)
    pred, summ = tmp_path / "FAKE_SMOKE_model_comparison_predictions_m2.csv", tmp_path / "FAKE_SMOKE_model_comparison_m2.csv"
    assert pred.exists() and summ.exists()
    assert not (tmp_path / "model_comparison_m2.csv").exists()          # never writes the real filenames
    P, S = _rd(pred), _rd(summ)
    assert len(P) == 12 == r["held_out"] and {x["mode"] for x in P + S} == {"FAKE_SMOKE"}
    assert "text" not in P[0] and {"baseline_pred", "embedding_pred", "baseline_conf", "embedding_conf"} <= set(P[0])
    keys = {(x["scope"], x["group"], x["metric"]) for x in S}
    for k in [("overall", "all", "macro_f1"), ("language", "en", "accuracy"), ("regulator", "RBI", "macro_f1"),
              ("difference", "all", "macro_f1_bootstrap"), ("difference", "all", "mcnemar_p_exact"),
              ("cost", "all", "saved_model_kb"), ("cost", "all", "latency_ms_mean"), ("per_class", "HIGH", "f1")]:
        assert k in keys, k


def test_real_mode_filenames_split_and_cost_metrics(tmp_path):
    m = _mod(); ev, held = _data(tmp_path)
    r = m.run(Enc(), fake=False, eval_path=ev, held_path=held, out_dir=tmp_path, n_boot=50, grid=(1.0,), verbose=False)
    assert (tmp_path / "model_comparison_predictions_m2.csv").exists() and (tmp_path / "model_comparison_m2.csv").exists()
    assert r["train"] == 48 and r["held_out"] == 12                      # leak-free: 12 partner rows dropped
    assert r["size_kb"]["baseline"] > 0 and r["size_kb"]["embedding"] > 0
    assert r["latency_ms_mean"]["baseline"] > 0 and r["latency_ms_mean"]["embedding"] > 0
    assert r["latency_labels_match"]                                    # uncached latency model == evaluated model
    assert {x["mode"] for x in _rd(tmp_path / "model_comparison_m2.csv")} == {"real"}
    assert r["bootstrap"]["verdict"] in ("improvement", "degradation", "no significant difference")


def test_real_mode_without_sentence_transformers_prints_commands(monkeypatch, capsys):
    import sys, pytest
    m = _mod()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.delitem(sys.modules, "services.embeddings.encoder", raising=False)
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert "pip install sentence-transformers" in str(e.value) and "scripts\\compare_risk_models.py" in str(e.value)
