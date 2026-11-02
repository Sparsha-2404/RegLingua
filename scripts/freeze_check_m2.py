#!/usr/bin/env python3
"""Day 14 (M2): freeze manifest + drift check for M2-owned files. Tooling only: no new model feature or metric.

  python scripts/freeze_check_m2.py --create    # run the full test suite, write docs/freeze_manifest_m2.json
  python scripts/freeze_check_m2.py --verify    # compare the working tree to the manifest; exit 1 on drift

What is frozen: every TRACKED file matching OWNED_GLOBS (minus OWNED_EXCLUDE), keyed by repo-relative path with a sha256 of
the working-tree bytes (text files CRLF->LF normalised so Windows/Linux checkouts agree; binary files hashed raw). The
manifest also records commit, branch, date, the full-suite test count and library versions. The globs/exclusions used are stored
in the manifest and --verify uses THOSE, not the defaults below.
Exit codes: 0 ok | 1 drift (added/changed/removed M2 file, or test-count change with --run-tests) | 2 --create refused
(uncommitted M2 changes, or the suite did not pass) | 3 usage/manifest problem. Hygiene findings (files >5 MB, untracked files,
stray root files, .pkl/.csv missing from models/MANIFEST_m2.json) are reported as warnings and never change the exit code.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA = 1
MANIFEST_REL = "docs/freeze_manifest_m2.json"
MODEL_MANIFEST_REL = "models/MANIFEST_m2.json"
SIZE_LIMIT = 5 * 1024 * 1024
OWNED_GLOBS = ["services/classifier/**", "services/evaluation/**", "services/verification/**", "scripts/*.py", "tests/unit/**",
               "docs/*_m2*.md", "data/evaluation/**", MODEL_MANIFEST_REL, "README_M2*.md"]
# files that live under an M2 glob but are declared as another owner's (headers checked on the Day 13 snapshot)
OWNED_EXCLUDE = {
    "scripts/train_regulator_classifier.py": "header: 'Owned by: M4' (regulator gap-filler)",
    "services/classifier/regulator_classifier.py": "M4 gap-filler module (see scripts/train_regulator_classifier.py)",
    "scripts/extract_m1_labeled_csv.py": "M1 labelled-data extractor",
    "scripts/evidence_qa_day6.py": "header: co-owned by M1 + M2 (excluded to avoid false drift from M1 edits)",
}
ROOT_ALLOW = ["README*.md", "CONTRIBUTING.md", "LICENSE*", "CHANGELOG*.md", "requirements*.txt", "pytest.ini", "pyproject.toml",
              "setup.cfg", "setup.py", "tox.ini", "Makefile", "Dockerfile", ".dockerignore", "docker-compose.yml", ".gitignore",
              ".gitattributes", ".env.example", ".pre-commit-config.yaml", "conftest.py", "MANIFEST.in"]
CORE_PACKAGES = ["pytest", "scikit-learn", "numpy", "pandas", "scipy", "joblib", "sentence-transformers", "torch", "transformers",
                 "langgraph", "fastapi", "pydantic", "langdetect"]
CODE_EXT = {".py"}


# ---------------------------------------------------------------------------------------------- git helpers
def git(root, *args, check=True, binary=False):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.decode('utf-8', 'replace').strip()}")
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")


def repo_root(start) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel").strip())


def zsplit(s: str) -> list[str]:
    return [x for x in s.split("\0") if x]


def tracked_files(root) -> list[str]:
    return zsplit(git(root, "ls-files", "-z"))


def untracked_files(root) -> list[str]:
    return zsplit(git(root, "ls-files", "-z", "--others", "--exclude-standard"))


def head_files(root) -> list[str]:
    return zsplit(git(root, "ls-tree", "-r", "--name-only", "-z", "HEAD"))


# ---------------------------------------------------------------------------------------------- ownership + hashing
def glob_re(g: str):
    out, i = [], 0
    while i < len(g):
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            out.append("[^/]*" if g[i] == "*" else "[^/]" if g[i] == "?" else re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def is_owned(path: str, globs, exclude) -> bool:
    return path not in exclude and path != MANIFEST_REL and any(glob_re(g).match(path) for g in globs)


def hash_bytes(data: bytes) -> str:
    if b"\0" not in data[:8192]:  # text (git's own binary heuristic): make CRLF/LF checkouts hash identically
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def category(path: str) -> str:
    if path.startswith("data/evaluation/") or path == MODEL_MANIFEST_REL:
        return "RESULTS"
    if os.path.splitext(path)[1] in CODE_EXT:
        return "CODE"
    return "DOCS" if path.endswith(".md") else "OTHER"


def owned_hashes(root, globs, exclude) -> dict[str, str]:
    out = {}
    for p in tracked_files(root):
        f = Path(root) / p
        if is_owned(p, globs, exclude) and f.is_file():
            out[p] = hash_bytes(f.read_bytes())
    return out


def dirty_owned(root, globs, exclude) -> list[str]:
    """Owned paths whose working tree differs from HEAD (content, ignoring line endings), or that are staged-only, deleted
    or untracked. Returned as 'status path'."""
    root = Path(root)
    head = set(head_files(root))
    cur = {p for p in tracked_files(root) if is_owned(p, globs, exclude)}
    out = []
    for p in sorted(cur | {h for h in head if is_owned(h, globs, exclude)}):
        f = root / p
        if not f.is_file():
            out.append(f"deleted {p}")
        elif p not in head:
            out.append(f"staged-new {p}")
        else:
            blob = git(root, "show", f"HEAD:{p}", binary=True)
            if hash_bytes(blob) != hash_bytes(f.read_bytes()):
                out.append(f"modified {p}")
    out += [f"untracked {p}" for p in untracked_files(root) if is_owned(p, globs, exclude)]
    return out


# ---------------------------------------------------------------------------------------------- tests + environment
def split_cmd(s: str) -> list[str]:
    if os.name == "nt":  # shlex(posix=False) keeps quotes and backslashes; strip the quotes
        return [t[1:-1] if len(t) > 1 and t[0] == t[-1] == '"' else t for t in shlex.split(s, posix=False)]
    return shlex.split(s)


_COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected|warnings?)")


def parse_pytest_summary(text: str) -> dict:
    counts = {k: 0 for k in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors", "deselected")}
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        found = _COUNT.findall(line)
        if found and (" in " in line or "no tests ran" in line or "=" in line):
            for n, k in found:
                k = "errors" if k.startswith("error") else k
                if k in counts:
                    counts[k] = int(n)
            break
    counts["total_tests"] = sum(counts[k] for k in ("passed", "failed", "skipped", "xfailed", "xpassed"))
    return counts


def run_tests(root, cmd: list[str], timeout=3600) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, timeout=timeout)
        out = (p.stdout + b"\n" + p.stderr).decode("utf-8", "replace")
        rc = p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        out, rc = f"{type(e).__name__}: {e}", -1
    res = parse_pytest_summary(out)
    parsed = res["total_tests"] > 0 or res["errors"] > 0
    res.update(command=" ".join(cmd), returncode=rc, duration_s=round(time.time() - t0, 1),
               status="ok" if rc == 0 and parsed else "failed" if parsed else "error",
               output_tail=[l for l in out.splitlines() if l.strip()][-3:])
    return res


def library_versions(root) -> dict:
    names = list(CORE_PACKAGES)
    req = Path(root) / "requirements.txt"
    if req.is_file():
        for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9_.\-]+)", line)
            if m and not line.lstrip().startswith(("#", "-")):
                names.append(m.group(1))
    out = {}
    for n in sorted(set(names), key=str.lower):
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


# ---------------------------------------------------------------------------------------------- hygiene
def listed_in_model_manifest(root) -> set[str] | None:
    f = Path(root) / MODEL_MANIFEST_REL
    if not f.is_file():
        return None
    found = set()
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.lower().endswith((".pkl", ".csv")):
            found.add(x.replace("\\", "/"))
    walk(json.loads(f.read_text(encoding="utf-8-sig")))
    return found


def hygiene(root, globs, exclude) -> dict:
    root = Path(root)
    tracked = tracked_files(root)
    big = [(p, (root / p).stat().st_size) for p in tracked if (root / p).is_file() and (root / p).stat().st_size > SIZE_LIMIT]
    untracked = untracked_files(root)
    stray = sorted(e.name for e in os.scandir(root) if e.is_file() and not any(fnmatch.fnmatch(e.name, a) for a in ROOT_ALLOW))
    listed = listed_in_model_manifest(root)
    unlisted = sorted(p for p in tracked if p.lower().endswith((".pkl", ".csv")) and (listed is None or p not in listed))
    return {"over_5mb": sorted(big), "untracked": sorted(untracked),
            "untracked_in_m2_paths": sorted(p for p in untracked if is_owned(p, globs, exclude)),
            "stray_root_files": stray, "model_manifest_present": listed is not None, "pkl_csv_not_in_model_manifest": unlisted}


def print_hygiene(h: dict, limit: int | None) -> None:
    def show(title, items, fmt=str):
        print(f"  {title}: {len(items) if items else 'none'}")
        for x in (items if limit is None else items[:limit]):
            print(f"      {fmt(x)}")
        if limit is not None and len(items) > limit:
            print(f"      ... +{len(items) - limit} more (--full lists all)")
    print("hygiene (warnings only):")
    show("tracked files > 5 MB", h["over_5mb"], lambda x: f"{x[0]} ({x[1] / 1048576:.1f} MB)")
    show("untracked files", h["untracked"], lambda x: x + ("   [M2 path]" if x in h["untracked_in_m2_paths"] else ""))
    show("stray files in repo root", h["stray_root_files"])
    if not h["model_manifest_present"]:
        print(f"  {MODEL_MANIFEST_REL} not found: every tracked .pkl/.csv counts as unlisted")
    show("tracked .pkl/.csv not listed in models/MANIFEST_m2.json", h["pkl_csv_not_in_model_manifest"])


# ---------------------------------------------------------------------------------------------- modes
def cmd_create(root: Path, a) -> int:
    globs, exclude = OWNED_GLOBS, OWNED_EXCLUDE
    try:
        commit = git(root, "rev-parse", "HEAD").strip()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 3
    dirty = dirty_owned(root, globs, exclude)
    if dirty and not a.allow_dirty:
        print(f"REFUSED: {len(dirty)} M2-owned path(s) are not committed; commit them first (or --allow-dirty to record them):")
        print("  " + "\n  ".join(dirty[:15]) + (f"\n  ... +{len(dirty) - 15} more" if len(dirty) > 15 else ""))
        return 2
    if a.skip_tests:
        tests = {"status": "not_run", "reason": "--skip-tests", "total_tests": None}
    else:
        cmd = split_cmd(a.tests_cmd) if a.tests_cmd else [sys.executable, "-m", "pytest", "-q"]
        print(f"running full suite: {' '.join(cmd)} (this can take a while)")
        tests = run_tests(root, cmd)
        if tests["status"] != "ok" and not a.allow_failing_tests:
            print(f"REFUSED: suite status={tests['status']} (rc={tests['returncode']}): " + "; ".join(tests["output_tail"]))
            print("  fix/skip the cause, or --allow-failing-tests to record the real (non-passing) result")
            return 2
    files = owned_hashes(root, globs, exclude)
    now = datetime.now(timezone.utc)
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    man = {"schema": SCHEMA, "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "date": now.strftime("%Y-%m-%d"),
           "git": {"commit": commit, "branch": branch, "detached_head": branch == "HEAD", "uncommitted_m2_paths_at_freeze": dirty},
           "owned_globs": globs, "excluded_paths": exclude,
           "hash_note": "sha256 of working-tree bytes; text files CRLF->LF normalised, binary files raw",
           "file_count": len(files), "files": files, "tests": tests,
           "python": platform.python_version(), "platform": platform.platform(), "libraries": library_versions(root),
           "hygiene_at_freeze": hygiene(root, globs, exclude)}
    out = root / (a.manifest or MANIFEST_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    t = tests["total_tests"]
    print(f"WROTE {out.relative_to(root).as_posix()}: commit {commit[:10]} branch {branch}, {len(files)} M2 files, tests: "
          f"{tests['status']}" + (f" ({t} tests: {tests['passed']} passed, {tests['skipped']} skipped, {tests['xfailed']} xfailed, "
                                  f"{tests['failed']} failed, {tests['errors']} errors)" if t is not None else ""))
    print("next: git add docs/freeze_manifest_m2.json && git commit  (the manifest records the commit BEFORE this one)")
    print_hygiene(man["hygiene_at_freeze"], None if a.full else a.limit)
    return 0


def cmd_verify(root: Path, a) -> int:
    mf = root / (a.manifest or MANIFEST_REL)
    if not mf.is_file():
        print(f"ERROR: {mf} not found; run --create first")
        return 3
    man = json.loads(mf.read_text(encoding="utf-8"))
    globs, exclude = man["owned_globs"], man["excluded_paths"]
    frozen = man["files"]
    now = owned_hashes(root, globs, exclude)
    changed = sorted(p for p in frozen if p in now and now[p] != frozen[p])
    removed = sorted(p for p in frozen if p not in now)
    added = sorted(p for p in now if p not in frozen)
    known = set(man["git"].get("uncommitted_m2_paths_at_freeze", []))
    new_untracked = sorted(p for p in untracked_files(root) if is_owned(p, globs, exclude)
                           and f"untracked {p}" not in known)
    commit = man["git"]["commit"]
    try:
        ahead = git(root, "rev-list", "--count", f"{commit}..HEAD").strip()
        head = git(root, "rev-parse", "--short=10", "HEAD").strip()
        outside = [p for p in zsplit(git(root, "diff", "--name-only", "-z", commit))
                   if p != mf.relative_to(root).as_posix() and not is_owned(p, globs, exclude)]
    except RuntimeError:
        ahead, head, outside = "?", "?", None
    print(f"freeze: commit {commit[:10]} branch {man['git']['branch']} on {man['date']}; HEAD now {head} (+{ahead} commits)")
    print(f"M2 files: {len(frozen)} frozen, {len(now)} tracked now")
    drift = bool(changed or removed or added or new_untracked)
    items = [("CHANGED", p) for p in changed] + [("REMOVED", p) for p in removed] + [("ADDED", p) for p in added] + \
            [("ADDED (untracked)", p) for p in new_untracked]
    if drift:
        print(f"DRIFT: {len(changed)} changed, {len(added) + len(new_untracked)} added, {len(removed)} removed")
        for kind, p in (items if a.full else items[:max(a.limit, 40)]):
            print(f"  {kind:<17} [{category(p):<7}] {p}")
        if any(category(p) in ("CODE", "RESULTS") for _, p in items):
            print("  *** M2 CODE OR RESULTS CHANGED SINCE THE FREEZE: re-run the unit suite, doc-number check and contract-snapshot test ***")
    else:
        print("no drift: every frozen M2 file is byte-identical (CRLF-normalised) and nothing was added or removed")
    if outside is not None:
        print(f"info: {len(outside)} non-M2 file(s) differ from the freeze commit (not checked here)"
              + (": " + ", ".join(outside[:6]) + (" ..." if len(outside) > 6 else "") if outside else ""))
    t = man["tests"]
    if a.run_tests:
        cmd = split_cmd(a.tests_cmd) if a.tests_cmd else [sys.executable, "-m", "pytest", "-q"]
        r = run_tests(root, cmd)
        same = t.get("total_tests") is not None and all(r[k] == t[k] for k in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors"))
        print(f"tests re-run: {r['total_tests']} tests ({r['passed']} passed, {r['failed']} failed, {r['errors']} errors) vs frozen "
              f"{t.get('total_tests')} -> {'same' if same else 'DIFFERENT'}")
        drift = drift or not same
    elif t.get("total_tests") is not None:
        print(f"tests (frozen, not re-run; use --run-tests): {t['total_tests']} tests, {t['passed']} passed, {t['skipped']} skipped, "
              f"{t['xfailed']} xfailed, {t['failed']} failed, {t['errors']} errors, status {t['status']}")
    else:
        print(f"tests: no frozen count (status {t['status']}); re-create with the suite run to record one")
    libs_now = library_versions(root)
    diff = [f"{k} {man['libraries'].get(k)} -> {libs_now.get(k)}" for k in sorted(set(libs_now) | set(man["libraries"]))
            if man["libraries"].get(k) != libs_now.get(k)]
    if diff or man["python"] != platform.python_version():
        print(f"info: environment differs from freeze env (python {man['python']} -> {platform.python_version()}): "
              + "; ".join(diff[:8]) + (" ..." if len(diff) > 8 else ""))
    print_hygiene(hygiene(root, globs, exclude), None if a.full else a.limit)
    print("RESULT: " + ("DRIFT (exit 1)" if drift else "CLEAN (exit 0)"))
    return 1 if drift else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--create", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--root", default=".", help="any path inside the git repo (default: cwd)")
    ap.add_argument("--manifest", help=f"manifest path relative to the repo root (default {MANIFEST_REL})")
    ap.add_argument("--tests-cmd", help="suite command (default: python -m pytest -q)")
    ap.add_argument("--skip-tests", action="store_true", help="--create: do not run the suite; records status not_run")
    ap.add_argument("--allow-dirty", action="store_true", help="--create: record even with uncommitted M2 paths")
    ap.add_argument("--allow-failing-tests", action="store_true", help="--create: record a non-passing suite result")
    ap.add_argument("--run-tests", action="store_true", help="--verify: re-run the suite and compare counts")
    ap.add_argument("--limit", type=int, default=10, help="max items per list (default 10)")
    ap.add_argument("--full", action="store_true", help="print every list in full")
    a = ap.parse_args(argv)
    try:
        root = repo_root(a.root)
    except (RuntimeError, OSError) as e:
        print(f"ERROR: not a git repository: {e}")
        return 3
    return cmd_create(root, a) if a.create else cmd_verify(root, a)


if __name__ == "__main__":
    sys.exit(main())
