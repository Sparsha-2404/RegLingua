"""Day 13 (M2): helpers of scripts/simulate_fresh_checkout_m2.py and the offline policy in tests/unit/conftest.py."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sim = _load("scripts/simulate_fresh_checkout_m2.py", "sim_fc_m2")
cf = _load("tests/unit/conftest.py", "conftest_m2")


def test_first_error_prefers_pytest_then_traceback():
    assert sim.first_error("x\nFAILED tests/a.py::t - boom\nTraceback (most recent call last):\n  File x\nValueError: v").startswith("FAILED tests/a.py::t")
    assert sim.first_error("ok\nTraceback (most recent call last):\n  File \"a.py\", line 1\nModuleNotFoundError: No module named 'q'") == "ModuleNotFoundError: No module named 'q'"
    assert sim.first_error("models/x.pkl   FAIL   sha256 differs").startswith("models/x.pkl")
    assert sim.first_error("") == ""


def test_pytest_summary_counts_and_skip_reasons():
    out = "SKIPPED [2] tests/unit/a.py:3: needs network\n= 1 failed, 5 passed, 2 skipped in 1.20s =\n"
    s = sim.pytest_summary(out)
    assert s["counts"] == {"failed": 1, "passed": 5, "skipped": 2} and s["skip_reasons"] == {"needs network": 2}


def test_network_error_signatures_and_offline_env(monkeypatch):
    assert cf.is_network_error("OSError: We couldn't connect to 'https://huggingface.co' to load this file")
    assert not cf.is_network_error("AssertionError: 1 != 2")
    cf._cache.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert cf.offline_reason().startswith("offline: HF_HUB_OFFLINE")
    cf._cache.clear()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_static_checks_flag_untracked_import_undeclared_package_and_abs_path(tmp_path):
    def g(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "services/evaluation").mkdir(parents=True)
    (tmp_path / "requirements.txt").write_text("numpy>=1\n")
    (tmp_path / "services/__init__.py").write_text("")
    (tmp_path / "services/evaluation/__init__.py").write_text("")
    (tmp_path / "services/evaluation/helper.py").write_text("X = 1\n")                       # stays untracked
    (tmp_path / "services/evaluation/main.py").write_text(
        "import numpy, pandas\nfrom services.evaluation import helper\nP = 'C:\\\\Users\\\\bob\\\\x.csv'\nD = 'data/in.csv'\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/in.csv").write_text("a\n")                                            # referenced, untracked
    g("init", "-q")
    g("add", "requirements.txt", "services/__init__.py", "services/evaluation/__init__.py", "services/evaluation/main.py")
    st = sim.static_checks(tmp_path)
    assert [i["what"] for i in st["untracked_imports"]] == ["services/evaluation/helper.py"]
    assert [p["what"] for p in st["undeclared_packages"]] == ["pandas"]
    assert [r["what"] for r in st["referenced_untracked"]] == ["data/in.csv"]
    assert len(st["abs_paths_code"]) == 1 and sim.high_findings(st) == 4
