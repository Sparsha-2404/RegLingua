"""Day 12 (M2): the final evaluation is deterministic and respects the frozen-split protocol (small synthetic fixture)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from services.evaluation import classification_metrics as cm
from services.evaluation import final_eval as fe

KW = dict(n_boot=30, folds=3, repeats=2, verbose=False)
RISK_WORDS = {"HIGH": "guaranteed assured returns zero risk", "MEDIUM": "may vary depends market conditions", "LOW": "disclosure policy standard notice"}
REGS = ("SEBI", "RBI", "DPDP")


def make_fixture(tmp_path: Path, name: str = "fx", flip_test_labels: bool = False):
    """60 original_ids x (en, hi) + 12 hi-en rows; every 4th id is final_test. Returns (eval_path, split_path)."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}x{j}" for i in range(40) for j in range(5)]
    rows, split = [], []

    def add(oid, lang, label, test):
        words = " ".join(rng.choice(vocab, 8, replace=False))
        text = f"{RISK_WORDS[label]} {words}" if lang != "hi" else f"{words} {RISK_WORDS[label].split()[0]}"
        qid = f"Q{len(rows):04d}"
        lab = {"LOW": "HIGH", "HIGH": "LOW", "MEDIUM": "MEDIUM"}[label] if (test and flip_test_labels) else label
        rows.append({"query_id": qid, "original_id": oid, "text": text, "language": lang, "true_regulator": REGS[len(rows) % 3],
                     "true_risk_level": lab})
        split.append({"row_id": qid, "original_id": oid, "split": "final_test" if test else "train_pool", "seed": 42})

    for i in range(60):
        label = ("HIGH", "MEDIUM", "LOW")[i % 3]
        for lang in ("en", "hi"):
            add(f"O{i:03d}", lang, label, i % 4 == 0)
    for i in range(12):
        add(f"C{i:03d}", "hi-en", ("HIGH", "LOW")[i % 2], i % 4 == 0)
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    ep, sp = d / "eval.csv", d / "split.csv"
    for p, data in ((ep, rows), (sp, split)):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return ep, sp


def run(tmp_path, tag, eval_path, split_path, **kw):
    out, res = tmp_path / tag / "out", tmp_path / tag / "results"
    r = fe.run_final_evaluation(eval_path, split_path, out, res, **{**KW, **kw})
    return out, res, r


def blobs(out, res):
    names = ("final_eval_results_m2.csv", "final_eval_confusions_m2.csv", "final_eval_oof_predictions_m2.csv")
    return [(out / n).read_bytes() for n in names] + [(res / "MANIFEST_m2.json").read_bytes()]


class FakeEncoder:
    model_name = "fake-encoder"

    def encode(self, texts):
        return np.array([[b / 255 for b in hashlib.md5(t.encode()).digest()] for t in texts])


def test_two_runs_produce_identical_outputs(tmp_path):
    ep, sp = make_fixture(tmp_path)
    a = blobs(*run(tmp_path, "a", ep, sp)[:2])
    b = blobs(*run(tmp_path, "b", ep, sp)[:2])
    assert a == b and all(len(x) > 100 for x in a)
    m = json.loads(a[3])
    assert m["seed"] == 42 and m["split"]["final_test_scored_once"] and m["embedding"]["status"] == "skipped"
    assert set(m["outputs_sha256"]) == {"final_eval_results_m2.csv", "final_eval_confusions_m2.csv", "final_eval_oof_predictions_m2.csv"}
    assert m["outputs_sha256"]["final_eval_results_m2.csv"] == hashlib.sha256(a[0]).hexdigest()


def test_embedding_branch_is_deterministic_and_selects_C_on_training_rows(tmp_path):
    ep, sp = make_fixture(tmp_path)
    x = run(tmp_path, "a", ep, sp, encoder=FakeEncoder(), encoder_name="fake-encoder")
    y = run(tmp_path, "b", ep, sp, encoder=FakeEncoder(), encoder_name="fake-encoder")
    assert blobs(*x[:2]) == blobs(*y[:2]) and fe.EMB in x[2]["models"]
    emb = x[2]["manifest"]["embedding"]
    assert emb["status"] == "included" and len(emb["selected_C_per_cv_fit"]) == KW["folds"] * KW["repeats"] and emb["selected_C_final"] in fe.C_GRID


def test_final_test_labels_never_influence_cross_validation(tmp_path):
    ep1, sp1 = make_fixture(tmp_path, "one")
    ep2, sp2 = make_fixture(tmp_path, "two", flip_test_labels=True)
    o1, _, r1 = run(tmp_path, "a", ep1, sp1)
    o2, _, r2 = run(tmp_path, "b", ep2, sp2)
    assert (o1 / "final_eval_oof_predictions_m2.csv").read_bytes() == (o2 / "final_eval_oof_predictions_m2.csv").read_bytes()
    assert [r for r in r1["rows"] if r[0] == "cv_oof"] == [r for r in r2["rows"] if r[0] == "cv_oof"]
    assert [r for r in r1["rows"] if r[0] == "final_test"] != [r for r in r2["rows"] if r[0] == "final_test"]   # test split does react


def test_oof_has_every_train_row_per_repeat_and_groups_never_straddle_folds(tmp_path):
    ep, sp = make_fixture(tmp_path)
    out, _, _ = run(tmp_path, "a", ep, sp)
    oof = list(csv.DictReader(open(out / "final_eval_oof_predictions_m2.csv", encoding="utf-8")))
    n_train = sum(r["split"] == "train_pool" for r in csv.DictReader(open(sp, encoding="utf-8")))
    for m in fe.BASE_MODELS:
        for rep in range(KW["repeats"]):
            sel = [r for r in oof if r["model"] == m and r["repeat"] == str(rep)]
            assert len(sel) == n_train and len({r["row_id"] for r in sel}) == n_train
            folds_of = {}
            for r in sel:
                folds_of.setdefault(r["original_id"], set()).add(r["fold"])
            assert all(len(v) == 1 for v in folds_of.values())          # an en/hi pair never straddles CV folds
    train_ids = {r["row_id"] for r in csv.DictReader(open(sp, encoding="utf-8")) if r["split"] == "train_pool"}
    assert {r["row_id"] for r in oof} == train_ids                       # final_test rows never appear in OOF


def test_fast_metrics_match_reference_module():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y, p = rng.integers(0, 3, 40), rng.integers(0, 3, 40)
        v = dict(zip(fe.NAMES, fe.metrics_from_confusion(fe.confusion(y, p))))
        ys, ps = [cm.LABELS[i] for i in y], [cm.LABELS[i] for i in p]
        assert v["accuracy"] == pytest.approx(cm.accuracy(ys, ps)) and v["macro_f1"] == pytest.approx(cm.macro_f1(ys, ps))
        assert v["false_compliant_rate"] == pytest.approx(cm.false_compliant_rate(ys, ps), nan_ok=True)
        prf = cm.per_class_prf(ys, ps)
        for c in cm.LABELS:
            for k in fe.CLASS_METRICS:
                assert v[f"{c}_{k}"] == pytest.approx(prf[c][k])


def test_guards_and_helpers(tmp_path):
    g = fe.OnceGuard("final_test")
    g.use()
    with pytest.raises(RuntimeError, match="exactly once"):
        g.use()
    assert fe.verdict(0.01, 0.2) == "improvement" and fe.verdict(-0.2, -0.01) == "degradation"
    assert fe.verdict(-0.1, 0.1) == "no significant difference" and fe.verdict(float("nan"), 1) == "undetermined"
    assert fe.verdict(-0.3, -0.1, lower_better=True) == "improvement"
    assert fe.en_hi_pairs(["a", "a", "b", "c", "c"], ["en", "hi", "hi-en", "en", "en"]) == [(0, 1)]
    ep, sp = make_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="checksum"):
        run(tmp_path, "x", ep, sp, expected_eval_sha="0" * 64)


def test_leaky_split_is_rejected(tmp_path):
    ep, sp = make_fixture(tmp_path)
    rows = list(csv.DictReader(open(sp, encoding="utf-8")))
    rows[0]["split"], rows[1]["split"] = "final_test", "train_pool"          # the en/hi pair of O000 now straddles
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(RuntimeError, match="leaks"):
        run(tmp_path, "x", ep, sp)


def test_committed_pickle_is_never_loaded(tmp_path, monkeypatch):
    from services.classifier.baseline import TfidfLogisticClassifier

    def boom(*a, **k):
        raise AssertionError("the committed .pkl must not be loaded")
    monkeypatch.setattr(TfidfLogisticClassifier, "load", boom)
    ep, sp = make_fixture(tmp_path)
    assert "tfidf_logreg" in run(tmp_path, "a", ep, sp)[2]["models"]


def test_result_rows_have_expected_scopes(tmp_path):
    ep, sp = make_fixture(tmp_path)
    rows = run(tmp_path, "a", ep, sp)[2]["rows"]
    scopes = {(r[0], r[1]) for r in rows}
    for ph in ("cv_oof", "final_test"):
        assert {(ph, s) for s in ("overall", "class", "language", "regulator", "paired_overall", "paired_en_hi", "paired_mcnemar")} <= scopes
    assert ("cv_oof", "per_repeat") in scopes
