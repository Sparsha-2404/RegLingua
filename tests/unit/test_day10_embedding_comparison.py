"""Day 10 (M2): comparison utilities + end-to-end run with a FAKE encoder (no HF)."""
import csv
import hashlib
import json

import numpy as np

from services.evaluation.embedding_comparison import (
    CachingEncoder, discordant_counts, paired_bootstrap, verdict,
)

HI = ["guaranteed returns zero risk", "assured profit guaranteed income", "risk free guaranteed gains"]
LO = ["track monthly spending budget", "read terms and conditions", "view account statement summary"]
_HIGHW = {"guaranteed", "assured", "risk", "zero", "free", "profit", "returns", "gains"}


class FakeEncoder:
    model_name = "fake-enc"
    calls = 0

    def encode(self, texts):
        FakeEncoder.calls += len(texts)
        out = []
        for t in texts:
            w = t.lower().split()
            hi = sum(x in _HIGHW for x in w)
            h = hashlib.md5(t.encode()).digest()
            out.append([hi, len(w) - hi] + [b / 255 * 0.05 for b in h[:3]])
        return np.array(out)


def test_verdict_rules():
    assert verdict(0.01, 0.2) == "improvement"
    assert verdict(-0.2, -0.01) == "degradation"
    assert verdict(-0.1, 0.1) == "no significant difference"
    assert verdict(float("nan"), 0.1) == "undetermined"


def _toy(n=60):
    y = np.array(["HIGH", "LOW"] * (n // 2))
    g = np.array([f"g{i//2}" for i in range(n)])
    lang = np.array(["en", "hi"] * (n // 2))
    return y, g, lang


def test_identical_predictions_zero_delta_not_improvement():
    y, g, lang = _toy()
    r = paired_bootstrap(y, y, y, g, lang, n_boot=200)
    assert r["accuracy"]["delta"] == 0 and r["accuracy"]["verdict"] == "no significant difference"


def test_clear_improvement_excludes_zero_and_is_seeded():
    y, g, lang = _toy(120)
    bad = np.where(np.arange(120) % 2 == 0, "LOW", "LOW")  # baseline: always LOW
    r1 = paired_bootstrap(y, bad, y, g, lang, n_boot=300)
    r2 = paired_bootstrap(y, bad, y, g, lang, n_boot=300)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)  # NaN-safe
    assert r1["accuracy"]["verdict"] == "improvement" and r1["accuracy"]["ci_low"] > 0
    assert r1["high_recall"]["verdict"] == "improvement"
    assert "accuracy@en" in r1 and "macro_f1@hi" in r1


def test_discordant_counts():
    d = discordant_counts(["A", "A", "B", "B"], ["A", "B", "B", "A"], ["A", "A", "A", "B"])
    assert d == {"both_correct": 1, "only_baseline_correct": 1, "only_embedding_correct": 2, "both_wrong": 0}


def test_caching_encoder_encodes_each_text_once():
    FakeEncoder.calls = 0
    c = CachingEncoder(FakeEncoder())
    c.encode(["a b", "c d", "a b"])
    c.encode(["a b", "c d"])
    assert FakeEncoder.calls == 2 and c.model_name == "fake-enc"


def test_end_to_end_run_with_fake_encoder(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "cmp", Path(__file__).resolve().parents[2] / "scripts" / "compare_baseline_vs_multilingual.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)

    ev, held = tmp_path / "eval.csv", tmp_path / "held.csv"
    rows = []
    for i in range(48):
        hi = i % 2 == 0
        base = (HI if hi else LO)[i % 3]
        rows.append({"query_id": f"q{i}", "original_id": f"o{i//2}", "text": f"{base} variant {i}",
                     "language": "en", "true_risk_level": "HIGH" if hi else "LOW"})
    with open(ev, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(held, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_id", "held_out"])
        for i in range(48):
            w.writerow([f"q{i}", "True" if i >= 38 else "False"])

    art = tmp_path / "emb.pkl"
    committed = tmp_path / "committed.pkl"
    committed.write_bytes(b"keep")
    res = cmp.run(FakeEncoder(), eval_path=ev, held_path=held, out_dir=tmp_path, artifact_path=art,
                  n_boot=100, committed_path=committed, grid=(0.1, 1.0))
    p = res["protocol"]
    # q38..q47 held out -> originals o19..o23; q36,q37 (o18) not held; o19's partner q38 is held too.
    assert p["leak_free"] and p["held_out_original_ids_in_train"] == 0
    assert p["train_rows"] + p["dropped_train_rows"] == 38 and p["held_out_rows"] == 10
    assert res["protocol"]["seed"] == 42
    assert (tmp_path / "day10_embedding_vs_baseline_m2.json").exists()
    assert (tmp_path / "day10_embedding_vs_baseline_m2.csv").exists()
    assert art.exists() and (tmp_path / "emb.pkl.meta.json").exists()
    assert json.loads((tmp_path / "emb.pkl.meta.json").read_text())["encoder_weights_saved"] is False
    assert committed.read_bytes() == b"keep"  # never touched
    # refusing to overwrite the committed path
    import pytest
    with pytest.raises(ValueError):
        cmp.run(FakeEncoder(), eval_path=ev, held_path=held, out_dir=tmp_path, artifact_path=committed,
                n_boot=20, committed_path=committed, grid=(1.0,))


def test_leak_free_drops_translation_pair_rows_and_leaky_keeps_them(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "cmp2", Path(__file__).resolve().parents[2] / "scripts" / "compare_baseline_vs_multilingual.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    ev, held = tmp_path / "e.csv", tmp_path / "h.csv"
    rows = []
    for i in range(40):  # pairs: q(2k), q(2k+1) share original o k, different language
        hi = (i // 2) % 2 == 0
        rows.append({"query_id": f"q{i}", "original_id": f"o{i//2}",
                     "text": f"{(HI if hi else LO)[i % 3]} v{i}", "language": "en" if i % 2 == 0 else "hi",
                     "true_risk_level": "HIGH" if hi else "LOW"})
    with open(ev, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(held, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_id", "held_out"])
        for i in range(40):  # hold out the 'hi' member of 4 pairs -> its 'en' partner is a leak in train
            w.writerow([f"q{i}", "True" if i in (1, 5, 9, 13) else "False"])
    kw = dict(eval_path=ev, held_path=held, out_dir=tmp_path, artifact_path=None, n_boot=50, grid=(1.0,),
              committed_path=tmp_path / "none.pkl")
    lf = cmp.run(FakeEncoder(), leak_free=True, **kw)["protocol"]
    lk = cmp.run(FakeEncoder(), leak_free=False, **kw)["protocol"]
    assert lf["dropped_train_rows"] == 4 and lf["train_rows"] == 32 and lf["held_out_original_ids_in_train"] == 0
    assert lk["dropped_train_rows"] == 0 and lk["train_rows"] == 36 and lk["held_out_original_ids_in_train"] == 4
    assert (tmp_path / "day10_embedding_vs_baseline_m2.json").exists()
    assert (tmp_path / "day10_embedding_vs_baseline_m2_leaky.json").exists()
