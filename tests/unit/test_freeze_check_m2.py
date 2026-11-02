"""Day 14 (M2): freeze_check_m2.py against throw-away git repos (no network, no real project files)."""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "freeze_check_m2.py"
_spec = importlib.util.spec_from_file_location("freeze_check_m2", SCRIPT)
fc = importlib.util.module_from_spec(_spec)
sys.modules["freeze_check_m2"] = fc
_spec.loader.exec_module(fc)

FILES = {  # owned (O) / not owned (-)
    "services/evaluation/a.py": "x = 1\n",                      # O code
    "services/classifier/b.py": "y = 2\n",                      # O code
    "services/classifier/regulator_classifier.py": "m4 = 1\n",  # excluded (M4)
    "scripts/x_m2.py": "print('x')\n",                          # O code
    "scripts/train_regulator_classifier.py": "m4 = 1\n",        # excluded (M4)
    "tests/unit/test_a.py": "def test_a():\n    pass\n",        # O code
    "docs/foo_m2.md": "# foo\n",                                # O docs
    "docs/day1_m2_handoff.md": "# h\n",                         # O docs
    "docs/other.md": "# not ours\n",                            # -
    "data/evaluation/listed.csv": "a,b\n1,2\n",                 # O results (listed in the model manifest)
    "models/MANIFEST_m2.json": json.dumps({"files": [{"file": "data/evaluation/listed.csv"}, {"file": "models/model.pkl"}]}),  # O
    "agents/m4.py": "z = 3\n", "README.md": "r\n", "requirements.txt": "numpy\nnot-a-real-pkg-xyz>=1\n",
    "data/other.csv": "q\n1\n", "models/model.pkl": "pkl", "models/stray.pkl": "pkl2",
}
OWNED = {"services/evaluation/a.py", "services/classifier/b.py", "scripts/x_m2.py", "tests/unit/test_a.py", "docs/foo_m2.md",
         "docs/day1_m2_handoff.md", "data/evaluation/listed.csv", "models/MANIFEST_m2.json"}


def g(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def make_repo(tmp_path, big=False):
    if not shutil.which("git"):
        raise unittest.SkipTest("git not available")
    repo = Path(tmp_path) / "repo"
    repo.mkdir()
    g(repo, "init", "-q")
    for k, v in (("user.email", "t@t"), ("user.name", "t"), ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
        g(repo, "config", k, v)
    for p, c in FILES.items():
        f = repo / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(c, encoding="utf-8", newline="")
    if big:
        (repo / "big.bin").write_bytes(b"\0" * (5 * 1024 * 1024 + 1))
    g(repo, "add", "-A")
    g(repo, "commit", "-q", "-m", "init")
    return repo


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fc.main([str(a) for a in argv])
    return rc, buf.getvalue()


def fake(tmp_path, summary, rc=0):
    script = Path(tmp_path) / "fake_suite.py"
    script.write_text(f"print({summary!r})\nraise SystemExit({rc})\n", encoding="utf-8")
    parts = [sys.executable, str(script)]
    return subprocess.list2cmdline(parts) if os.name == "nt" else " ".join(__import__("shlex").quote(x) for x in parts)


def freeze(repo, tmp_path, summary="7 passed, 1 skipped in 0.10s"):
    rc, out = run("--create", "--root", repo, "--tests-cmd", fake(tmp_path, summary))
    assert rc == 0, out
    g(repo, "add", "-A")
    g(repo, "commit", "-q", "-m", "freeze")
    return json.loads((repo / fc.MANIFEST_REL).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ pure helpers
def test_ownership_globs_and_exclusions():
    o = lambda p: fc.is_owned(p, fc.OWNED_GLOBS, fc.OWNED_EXCLUDE)  # noqa: E731
    assert all(o(p) for p in OWNED | {"services/verification/deep/x.py", "README_M2_Day1_Day2.md"})
    assert not any(o(p) for p in ["agents/m4.py", "docs/other.md", "scripts/train_regulator_classifier.py", "scripts/notes.txt",
                                  "services/classifier/regulator_classifier.py", "core/contracts.py", fc.MANIFEST_REL, "data/raw/x.csv"])
    assert o("scripts/anything.py") and not o("scripts/sub/x.py")  # single-star does not cross directories


def test_hash_normalises_text_line_endings_but_not_binary():
    assert fc.hash_bytes(b"a\r\nb\r\n") == fc.hash_bytes(b"a\nb\n") != fc.hash_bytes(b"a\nb")
    assert fc.hash_bytes(b"\0a\r\n") != fc.hash_bytes(b"\0a\n")


def test_category():
    assert [fc.category(p) for p in ["a/b.py", "data/evaluation/x.csv", "models/MANIFEST_m2.json", "docs/x_m2.md", "x.txt"]] == \
           ["CODE", "RESULTS", "RESULTS", "DOCS", "OTHER"]


def test_parse_pytest_summary():
    p = fc.parse_pytest_summary
    assert p("....\n262 passed, 2 skipped, 2 xfailed in 12.34s (0:00:12)") | {} == p("262 passed, 2 skipped, 2 xfailed in 1s") | {}
    x = p("262 passed, 2 skipped, 2 xfailed in 12.34s")
    assert (x["passed"], x["skipped"], x["xfailed"], x["total_tests"], x["errors"]) == (262, 2, 2, 266, 0)
    y = p("===== 1 failed, 5 passed, 3 errors in 0.5s =====")
    assert (y["failed"], y["passed"], y["errors"], y["total_tests"]) == (1, 5, 3, 6)
    assert p("no tests ran in 0.01s")["total_tests"] == 0 and p("")["total_tests"] == 0


# ------------------------------------------------------------------------------------------ --create
def test_create_records_commit_branch_files_tests_and_libraries(tmp_path):
    repo = make_repo(tmp_path)
    head, branch = g(repo, "rev-parse", "HEAD"), g(repo, "rev-parse", "--abbrev-ref", "HEAD")
    rc, out = run("--create", "--root", repo, "--tests-cmd", fake(tmp_path, "7 passed, 1 skipped in 0.10s"))
    assert rc == 0, out
    m = json.loads((repo / fc.MANIFEST_REL).read_text(encoding="utf-8"))
    assert m["git"]["commit"] == head and m["git"]["branch"] == branch and m["date"] == m["created_utc"][:10]
    assert set(m["files"]) == OWNED and m["file_count"] == len(OWNED)
    assert fc.MANIFEST_REL not in m["files"] and "agents/m4.py" not in m["files"]
    assert m["tests"]["status"] == "ok" and m["tests"]["total_tests"] == 8 and m["tests"]["passed"] == 7 and m["tests"]["skipped"] == 1
    assert m["libraries"]["not-a-real-pkg-xyz"] is None and "numpy" in m["libraries"] and m["python"]
    assert m["files"]["services/evaluation/a.py"] == fc.hash_bytes(b"x = 1\n")
    assert m["excluded_paths"] and "pkl_csv_not_in_model_manifest" in m["hygiene_at_freeze"]


def test_create_skip_tests_records_not_run_never_a_number(tmp_path):
    repo = make_repo(tmp_path)
    assert run("--create", "--skip-tests", "--root", repo)[0] == 0
    t = json.loads((repo / fc.MANIFEST_REL).read_text(encoding="utf-8"))["tests"]
    assert t["status"] == "not_run" and t["total_tests"] is None


def test_create_refuses_uncommitted_m2_changes_unless_allowed(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "services/evaluation/a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "data/evaluation/new.csv").write_text("n\n", encoding="utf-8")
    rc, out = run("--create", "--skip-tests", "--root", repo)
    assert rc == 2 and "modified services/evaluation/a.py" in out and "untracked data/evaluation/new.csv" in out
    assert not (repo / fc.MANIFEST_REL).exists()
    assert run("--create", "--skip-tests", "--allow-dirty", "--root", repo)[0] == 0
    m = json.loads((repo / fc.MANIFEST_REL).read_text(encoding="utf-8"))
    assert "modified services/evaluation/a.py" in m["git"]["uncommitted_m2_paths_at_freeze"]


def test_dirty_check_ignores_line_ending_only_changes_and_non_m2_edits(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "docs/foo_m2.md").write_bytes(b"# foo\r\n")   # same text, CRLF
    (repo / "agents/m4.py").write_text("z = 99\n", encoding="utf-8")
    assert fc.dirty_owned(repo, fc.OWNED_GLOBS, fc.OWNED_EXCLUDE) == []


def test_create_refuses_failing_or_unrunnable_suite(tmp_path):
    repo = make_repo(tmp_path)
    rc, out = run("--create", "--root", repo, "--tests-cmd", fake(tmp_path, "1 failed, 2 passed in 0.1s", rc=1))
    assert rc == 2 and "REFUSED" in out and not (repo / fc.MANIFEST_REL).exists()
    bad = subprocess.list2cmdline([sys.executable, "-c", "import module_that_does_not_exist_xyz"]) if os.name == "nt" else \
        f"{sys.executable} -c 'import module_that_does_not_exist_xyz'"
    assert run("--create", "--root", repo, "--tests-cmd", bad)[0] == 2
    rc, _ = run("--create", "--root", repo, "--allow-failing-tests", "--tests-cmd", fake(tmp_path, "1 failed, 2 passed in 0.1s", rc=1))
    t = json.loads((repo / fc.MANIFEST_REL).read_text(encoding="utf-8"))["tests"]
    assert rc == 0 and t["status"] == "failed" and t["failed"] == 1 and t["total_tests"] == 3


# ------------------------------------------------------------------------------------------ --verify
def test_verify_clean_after_freeze_commit_and_ignores_manifest_commit(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    rc, out = run("--verify", "--root", repo)
    assert rc == 0 and "no drift" in out and "(+1 commits)" in out and "CLEAN" in out


def test_verify_detects_changed_added_removed_and_flags_code_and_results(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    (repo / "services/evaluation/a.py").write_text("x = 2\n", encoding="utf-8")                 # changed code
    (repo / "data/evaluation/listed.csv").write_text("a,b\n9,9\n", encoding="utf-8")            # changed results
    (repo / "docs/foo_m2.md").unlink()                                                          # removed doc
    (repo / "services/classifier/new.py").write_text("n = 1\n", encoding="utf-8")               # added + tracked
    g(repo, "add", "services/classifier/new.py")
    (repo / "data/evaluation/loose.csv").write_text("u\n", encoding="utf-8")                    # added, untracked
    rc, out = run("--verify", "--root", repo, "--full")
    assert rc == 1 and "DRIFT: 2 changed, 2 added, 1 removed" in out
    for needle in ["CHANGED           [CODE   ] services/evaluation/a.py", "CHANGED           [RESULTS] data/evaluation/listed.csv",
                   "REMOVED           [DOCS   ] docs/foo_m2.md", "ADDED             [CODE   ] services/classifier/new.py",
                   "ADDED (untracked) [RESULTS] data/evaluation/loose.csv", "M2 CODE OR RESULTS CHANGED SINCE THE FREEZE"]:
        assert needle in out, needle


def test_verify_docs_only_drift_is_drift_but_not_code_flagged(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    (repo / "docs/foo_m2.md").write_text("# changed\n", encoding="utf-8")
    rc, out = run("--verify", "--root", repo)
    assert rc == 1 and "[DOCS" in out and "M2 CODE OR RESULTS" not in out


def test_verify_ignores_non_m2_changes_but_reports_them(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    (repo / "agents/m4.py").write_text("z = 100\n", encoding="utf-8")
    (repo / "docs/other.md").write_text("# edited\n", encoding="utf-8")
    rc, out = run("--verify", "--root", repo)
    assert rc == 0 and "2 non-M2 file(s) differ" in out and "agents/m4.py" in out


def test_verify_line_ending_conversion_is_not_drift(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    for p in OWNED - {"models/MANIFEST_m2.json"}:
        f = repo / p
        f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    assert run("--verify", "--root", repo)[0] == 0


def test_verify_untracked_m2_file_known_at_freeze_is_not_new_drift(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "data/evaluation/scratch.csv").write_text("s\n", encoding="utf-8")
    assert run("--create", "--skip-tests", "--allow-dirty", "--root", repo)[0] == 0
    g(repo, "add", fc.MANIFEST_REL)
    g(repo, "commit", "-q", "-m", "freeze")
    assert run("--verify", "--root", repo)[0] == 0
    (repo / "data/evaluation/scratch2.csv").write_text("s\n", encoding="utf-8")
    assert run("--verify", "--root", repo)[0] == 1


def test_verify_run_tests_flags_count_change_and_accepts_same(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path, "7 passed, 1 skipped in 0.10s")
    assert run("--verify", "--run-tests", "--root", repo, "--tests-cmd", fake(tmp_path, "7 passed, 1 skipped in 0.2s"))[0] == 0
    rc, out = run("--verify", "--run-tests", "--root", repo, "--tests-cmd", fake(tmp_path, "8 passed, 1 skipped in 0.2s"))
    assert rc == 1 and "DIFFERENT" in out


def test_verify_without_manifest_or_outside_repo_exits_3(tmp_path):
    repo = make_repo(tmp_path)
    assert run("--verify", "--root", repo)[0] == 3
    plain = Path(tmp_path) / "not_a_repo"
    plain.mkdir()
    assert run("--verify", "--root", plain)[0] == 3


def test_cli_exit_status_is_nonzero_on_drift(tmp_path):
    repo = make_repo(tmp_path)
    freeze(repo, tmp_path)
    ok = subprocess.run([sys.executable, str(SCRIPT), "--verify", "--root", str(repo)], capture_output=True)
    assert ok.returncode == 0
    (repo / "scripts/x_m2.py").write_text("print('changed')\n", encoding="utf-8")
    bad = subprocess.run([sys.executable, str(SCRIPT), "--verify", "--root", str(repo)], capture_output=True)
    assert bad.returncode == 1 and b"DRIFT" in bad.stdout


# ------------------------------------------------------------------------------------------ hygiene
def test_hygiene_oversize_untracked_stray_and_unlisted_pkl_csv(tmp_path):
    repo = make_repo(tmp_path, big=True)
    (repo / "scratch_notes.txt").write_text("tmp", encoding="utf-8")        # untracked + stray root
    (repo / "data/evaluation/loose.csv").write_text("u\n", encoding="utf-8")  # untracked in an M2 path
    h = fc.hygiene(repo, fc.OWNED_GLOBS, fc.OWNED_EXCLUDE)
    assert [p for p, _ in h["over_5mb"]] == ["big.bin"] and h["over_5mb"][0][1] > fc.SIZE_LIMIT
    assert set(h["untracked"]) == {"scratch_notes.txt", "data/evaluation/loose.csv"}
    assert h["untracked_in_m2_paths"] == ["data/evaluation/loose.csv"]
    assert h["stray_root_files"] == ["big.bin", "scratch_notes.txt"]  # README.md / requirements.txt are allowed
    assert h["pkl_csv_not_in_model_manifest"] == ["data/other.csv", "models/stray.pkl"]  # listed.csv / model.pkl are listed
    assert h["model_manifest_present"]


def test_hygiene_without_model_manifest_lists_every_pkl_csv(tmp_path):
    repo = make_repo(tmp_path)
    g(repo, "rm", "-q", "models/MANIFEST_m2.json")
    h = fc.hygiene(repo, fc.OWNED_GLOBS, fc.OWNED_EXCLUDE)
    assert not h["model_manifest_present"] and "data/evaluation/listed.csv" in h["pkl_csv_not_in_model_manifest"]


def test_hygiene_never_changes_exit_code(tmp_path):
    repo = make_repo(tmp_path, big=True)
    freeze(repo, tmp_path)
    (repo / "scratch.tmp").write_text("x", encoding="utf-8")
    rc, out = run("--verify", "--root", repo)
    assert rc == 0 and "stray files in repo root" in out and "big.bin" in out
