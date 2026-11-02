"""Day 13 (M2): scripts/check_doc_numbers_m2.py on a tiny fixture."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("chk", ROOT / "scripts" / "check_doc_numbers_m2.py")
chk = importlib.util.module_from_spec(_s); _s.loader.exec_module(chk)

CSV = "pred,truth,grp,v\nA,A,x,0.5\nA,B,x,0.25\nB,B,y,0.125\nB,A,y,1\n"


def build(tmp, doc, checks):
    (tmp / "docs").mkdir(exist_ok=True)
    (tmp / "d.csv").write_text(CSV, encoding="utf-8")
    (tmp / "d.json").write_text(json.dumps({"a": {"b": [0.4567]}}), encoding="utf-8")
    (tmp / "docs" / "x.md").write_text(doc, encoding="utf-8")
    m = tmp / "map.json"
    m.write_text(json.dumps({"checks": [dict(c, document="docs/x.md") for c in checks]}), encoding="utf-8")
    return ["--root", str(tmp), "--map", str(m)]


def ck(id_, rx, src="d.csv", fmt="number", **kw):
    return {"id": id_, "regex": rx, "source": src, "format": fmt, **kw}


def test_pass_rounding_ops_and_json(tmp_path, capsys):
    doc = "agree 0.50. count 4 rows. share 50%. mean 0.469 here. f1 0.5. json 0.46."
    checks = [ck("agree", r"agree ([0-9.]+)", select={"op": "agree", "column": "pred", "other": "truth"}),
              ck("count", r"count (\d+)", fmt="int", select={"op": "count"}),
              ck("share", r"share (\d+)%", fmt="percent", select={"op": "share", "column": "grp", "equals": "x"}),
              ck("mean", r"mean ([0-9.]+)", select={"op": "mean", "column": "v"}),
              ck("f1", r"f1 ([0-9.]+)", select={"op": "f1", "column": "pred", "other": "truth", "label": "A"}),
              ck("json", r"json ([0-9.]+)", src="d.json", key="a.b.0")]
    assert chk.main(build(tmp_path, doc, checks)) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") >= 6 and "6/6 claims PASS, 0 FAIL" in out


def test_mismatch_exits_nonzero(tmp_path, capsys):
    rc = chk.main(build(tmp_path, "count 5 rows", [ck("count", r"count (\d+)", fmt="int", select={"op": "count"})]))
    assert rc == 1 and "FAIL" in capsys.readouterr().out


def test_pattern_not_found_and_ambiguous_first(tmp_path, capsys):
    checks = [ck("nf", r"nothing ([0-9.]+)", select={"op": "count"}),
              ck("amb", r"x ([0-9.]+)", select={"op": "first", "column": "v", "where": {"grp": "x"}})]
    assert chk.main(build(tmp_path, "x 0.5", checks)) == 1
    out = capsys.readouterr().out
    assert out.count("ERROR") == 2 and "exactly one" in out
