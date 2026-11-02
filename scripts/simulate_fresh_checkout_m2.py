"""
DAY 13 (M2): fresh-checkout simulation.  Gate: a clean clone runs the demo with the documented commands.

Copies ONLY `git ls-files` (tracked/indexed files, working-tree content) into a temp dir, creates a clean venv there,
installs requirements.txt, then runs:
  1 python scripts/prepare_demo_artifacts.py --verify      (M2 artifact preparation; override: --prepare-args)
  2 python -m pytest tests/unit                            (M2 test suite; network tests skip when offline, see tests/unit/conftest.py)
  3 smoke request (agents.supervisor.run_case, canonical text, trained artifacts)
  4 `<script> --help` for every tracked scripts/*.py       (scripts with no argparse are NOT run: --help would execute them)
Static checks always run (even with --static-only or when venv/pip is impossible): untracked / gitignored / missing
dependencies, undeclared packages, hard-coded absolute paths, tests that touch Hugging Face/network.

Writes docs/fresh_checkout_report_m2.md and data/evaluation/fresh_checkout_results_m2.json (every number in the report
comes from that run). Exit code 0 only if every step PASSes and no HIGH static finding exists.
Steps run with HF_HUB_OFFLINE=1 by default (the demo gate is offline); --online lifts that.

  Windows cmd:  python scripts\\simulate_fresh_checkout_m2.py [--static-only] [--online] [--keep] [--pip-arg ARG]
  bash:         python3 scripts/simulate_fresh_checkout_m2.py [--static-only] [--online] [--keep] [--pip-arg ARG]
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_REL = "docs/fresh_checkout_report_m2.md"
JSON_REL = "data/evaluation/fresh_checkout_results_m2.json"
CANONICAL_TEXT = "This mutual fund offers guaranteed 20% returns with zero risk."
PY_SCOPE = ("services/classifier/", "services/evaluation/", "services/verification/", "tests/unit/")  # + scripts/*.py
OWNED_PREFIXES = PY_SCOPE + ("scripts/", "data/evaluation/", "models/", "results/")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", ".ipynb_checkpoints"}

DIST = {"sklearn": "scikit-learn", "yaml": "pyyaml", "PIL": "pillow", "dotenv": "python-dotenv", "bs4": "beautifulsoup4",
        "cv2": "opencv-python", "sentence_transformers": "sentence-transformers"}
TRANSITIVE = {"scipy": "scikit-learn", "joblib": "scikit-learn", "threadpoolctl": "scikit-learn", "torch": "sentence-transformers",
              "transformers": "sentence-transformers", "huggingface_hub": "sentence-transformers", "tqdm": "sentence-transformers",
              "starlette": "fastapi", "typing_extensions": "pydantic", "pydantic_core": "pydantic", "click": "uvicorn",
              "langchain_core": "langgraph", "_pytest": "pytest", "pluggy": "pytest"}
NET_PATTERNS = re.compile(r"SentenceTransformer\(|from_pretrained\(|snapshot_download|hf_hub_download|MultilingualEncoder\(|"
                          r"LocalOnlyEncoder\(|requests\.(get|post)|urlopen\(|httpx\.|huggingface\.co")
HERMETIC_HINT = re.compile(r"hf_network|importorskip|monkeypatch\.setattr\(socket|FakeEncoder|class Enc\b")
ABS_RE = re.compile(r"(?<![\w])[A-Za-z]:\\(?!\\n)[\w$.\-]|(?<![\w:/.\-])[A-Za-z]:/(?:Users|Documents|Projects|Desktop|work|dev|home)/|"
                    r"(?<![\w:/.\-])/(?:home|Users|mnt|root|workspace|var/folders|tmp)/[\w.\-]+")
PATH_RE = re.compile(r"^(?:\.?/)?(?:data|models|docs|results|notebooks|scripts|services|core|agents|apps|tests)/[\w\-./]+\.[A-Za-z0-9]{1,6}$")

SMOKE_CODE = r"""
import json, sys, time
root, text = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
t = time.time()
from agents.supervisor import run_case
s = run_case(text, use_trained_artifacts=True)
g = lambda v: getattr(v, "value", v)
print("SMOKE_JSON " + json.dumps({
    "resolution_status": g(s.resolution_status), "risk_label": g(s.risk_label), "regulator": g(s.regulator),
    "n_evidence": len(s.evidence), "verified_compliant": s.verified_compliant, "seconds": round(time.time() - t, 2)}))
"""
TERMINAL = {"resolved", "escalated", "needs_review"}


# ------------------------------------------------------------------ helpers
def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=str(root), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def git_list(root: Path, *args: str) -> list:
    p = git(root, *args, "-z")
    return [x for x in p.stdout.split("\0") if x] if p.returncode == 0 else []


def keep(rel: str) -> bool:
    return not (set(Path(rel).parts) & SKIP_DIRS) and not rel.endswith((".pyc", ".pyo"))


def first_error(text: str) -> str:
    """First meaningful error line of a step's combined output ('' if none)."""
    lines = [l.rstrip() for l in (text or "").splitlines()]
    for l in lines:
        if re.match(r"^(FAILED|ERROR) \S+", l) or re.match(r"^FAIL\s+\S+", l) or re.search(r"^\S+\s+FAIL\s+\S", l):
            return l[:240]
    for i, l in enumerate(lines):
        if l.startswith("Traceback"):
            for m in lines[i + 1:]:
                if m and not m[0].isspace() and not m.startswith("Traceback"):
                    return m[:240]
    for l in lines:
        if re.match(r"^E\s+\S", l) or re.search(r"(?i)\b(error|exception)\b.*:", l):
            return l.strip()[:240]
    tail = [l for l in lines if l.strip()]
    return tail[-1].strip()[:240] if tail else ""


def pytest_summary(out: str) -> dict:
    last = ""
    for l in out.splitlines():
        if re.search(r"\b(passed|failed|skipped|error|errors|no tests ran)\b.* in [\d.]+s", l):
            last = l.strip("= ").strip()
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)", last)}
    reasons = Counter()
    for l in out.splitlines():
        m = re.match(r"SKIPPED \[(\d+)\] \S+: (.*)", l)
        if m:
            reasons[m.group(2).strip()[:120]] += int(m.group(1))
    return {"line": last, "counts": counts, "skip_reasons": dict(reasons)}


def run(name: str, cmd: list, cwd: Path, env: dict, timeout: int) -> dict:
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", f"timeout after {timeout}s"
    except OSError as exc:
        rc, out, err = -1, "", f"{type(exc).__name__}: {exc}"
    secs = round(time.perf_counter() - t, 2)
    return {"step": name, "status": "PASS" if rc == 0 else "FAIL", "seconds": secs, "returncode": rc,
            "first_error": "" if rc == 0 else first_error(err + "\n" + out), "stdout": out, "stderr": err}


def skipped(name: str, why: str) -> dict:
    return {"step": name, "status": "SKIP", "seconds": 0.0, "returncode": None, "first_error": why, "stdout": "", "stderr": ""}


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ------------------------------------------------------------------ static checks
def parse_requirements(root: Path) -> set:
    names = set()
    f = root / "requirements.txt"
    if f.is_file():
        for l in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", l.split("#")[0])
            if m:
                names.add(m.group(1).lower().replace("_", "-"))
    return names


def py_scope(root: Path, tracked: set) -> list:
    """M2 python files: tracked or on disk (untracked ones are exactly what we must flag)."""
    found = {t for t in tracked if t.endswith(".py")}
    for pre in PY_SCOPE + ("scripts/",):
        d = root / pre
        if d.is_dir():
            found |= {p.relative_to(root).as_posix() for p in d.rglob("*.py")}
    return sorted(f for f in found if keep(f) and (f.startswith(PY_SCOPE) or (f.startswith("scripts/") and f.count("/") == 1)))


def module_file(root: Path, parts: list):
    """Resolve dotted module parts to an existing repo file/dir (relative posix) or None."""
    base = "/".join(parts)
    for cand in (f"{base}.py", f"{base}/__init__.py"):
        if (root / cand).is_file():
            return cand
    return base if (root / base).is_dir() else None


def optional_import(node, parents) -> bool:
    n = node
    while n in parents:
        n = parents[n]
        if isinstance(n, ast.Try) and any(h.type is None or "Error" in ast.dump(h.type) or "Exception" in ast.dump(h.type) for h in n.handlers):
            return True
    return False


def analyse_python(root: Path, files: list, tracked: set, ignored: set, reqs: set) -> dict:
    local = {p.name for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")} | \
            {p.stem for p in root.glob("*.py")}
    stdlib = getattr(sys, "stdlib_module_names", set())
    script_mods = {p.stem for p in (root / "scripts").glob("*.py")} if (root / "scripts").is_dir() else set()
    out = {"imports": [], "packages": [], "path_refs": [], "abs_paths": []}
    seen, queue = set(), list(files)
    while queue:
        rel = queue.pop()
        if rel in seen or not (root / rel).is_file():
            continue
        seen.add(rel)
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            out["imports"].append({"file": rel, "line": exc.lineno or 0, "what": f"SyntaxError: {exc.msg}", "kind": "SYNTAX_ERROR"})
            continue
        parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        in_scope = rel in files
        for node in ast.walk(tree):
            mods = []   # (dotted parts, required): `from a.b import c` -> a.b required, a.b.c only if it is a submodule
            if isinstance(node, ast.Import):
                mods = [(a.name.split("."), True) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                base = node.module.split(".")
                mods = [(base, True)] + [(base + [a.name], False) for a in node.names if a.name != "*"]
            for parts, required in mods:
                top = parts[0]
                if top in stdlib or top == "__future__":
                    continue
                if top in local or top in script_mods:
                    target = module_file(root, parts) if top in local else (f"scripts/{top}.py" if len(parts) == 1 else None)
                    if target is None:
                        if required:
                            out["imports"].append({"file": rel, "line": node.lineno, "what": ".".join(parts), "kind": "MISSING_IMPORT"})
                        continue
                    if target.endswith(".py"):
                        if target not in tracked:
                            k = "GITIGNORED_IMPORT" if target in ignored else "UNTRACKED_IMPORT"
                            out["imports"].append({"file": rel, "line": node.lineno, "what": target, "kind": k})
                        queue.append(target)
                elif in_scope and required:
                    dist = DIST.get(top, top.replace("_", "-").lower())
                    if dist in reqs:
                        continue
                    k = "TRANSITIVE_ONLY" if TRANSITIVE.get(top) in reqs else "UNDECLARED_PACKAGE"
                    out["packages"].append({"file": rel, "line": node.lineno, "what": top, "kind": k,
                                            "optional": optional_import(node, parents), "via": TRANSITIVE.get(top, "")})
            if in_scope and rel != "scripts/simulate_fresh_checkout_m2.py" and isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip().replace("\\", "/")
                if PATH_RE.match(s):
                    out["path_refs"].append({"file": rel, "line": node.lineno, "what": s.lstrip("./") if s.startswith("./") else s.lstrip("/")})
                if ABS_RE.search(node.value) and "\n" not in node.value[:200] and len(node.value) < 300:
                    out["abs_paths"].append({"file": rel, "line": node.lineno, "what": node.value.strip()[:100], "where": "code"})
    # absolute paths in comments / docstrings of M2 python are caught above only for short strings; scan raw lines too
    for rel in files:
        for n, l in enumerate((root / rel).read_text(encoding="utf-8", errors="replace").splitlines() if (root / rel).is_file() else [], 1):
            s = l.lstrip()
            if s.startswith("#") and ABS_RE.search(l):
                out["abs_paths"].append({"file": rel, "line": n, "what": s[:100], "where": "comment"})
    return out


def text_abs_paths(root: Path, tracked: set) -> list:
    cands = set()
    for pat in ("docs/*.md", "data/evaluation/*.json", "data/evaluation/*.txt", "models/*.json"):
        cands |= {p.relative_to(root).as_posix() for p in root.glob(pat)}
    res = []
    for rel in sorted(cands):
        if not (re.search(r"(^|[_/])m2([_.]|$)", rel) or rel.startswith(("models/", "data/evaluation/"))):
            continue
        p = root / rel
        if p.stat().st_size > 2_000_000 or rel == REPORT_REL:
            continue
        for n, l in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = ABS_RE.search(l)
            if m:
                res.append({"file": rel, "line": n, "what": l.strip()[:100], "where": "tracked" if rel in tracked else "untracked"})
                break   # first hit per file keeps the report short
    return res


def network_tests(root: Path) -> list:
    res = []
    for p in sorted((root / "tests/unit").glob("test_*.py")):
        t = p.read_text(encoding="utf-8", errors="replace")
        hits = sorted({m.group(0) for m in NET_PATTERNS.finditer(t)})
        if hits:
            res.append({"file": p.relative_to(root).as_posix(), "patterns": hits, "marked": "hf_network" in t,
                        "hermetic_hint": bool(HERMETIC_HINT.search(t))})
    return res


def static_checks(root: Path) -> dict:
    tracked = [t for t in git_list(root, "ls-files") if keep(t)]
    tset = set(tracked)
    others = [x for x in git_list(root, "ls-files", "--others", "--exclude-standard") if keep(x)]
    ignored = [x for x in git_list(root, "ls-files", "--others", "--ignored", "--exclude-standard") if keep(x)]
    iset = set(ignored)
    head = set(git_list(root, "ls-tree", "-r", "--name-only", "HEAD"))
    reqs = parse_requirements(root)
    files = py_scope(root, tset)
    an = analyse_python(root, files, tset, iset, reqs)
    # path literals -> classify
    refs = {}
    for r in an["path_refs"]:
        refs.setdefault(r["what"], []).append(f"{r['file']}:{r['line']}")
    ref_untracked, ref_ignored, ref_absent = [], [], []
    for path, where in sorted(refs.items()):
        on_disk = (root / path).is_file()
        if path in tset:
            continue
        item = {"what": path, "file": where[0], "n_refs": len(where)}
        if on_disk and path in iset:
            ref_ignored.append(item)
        elif on_disk:
            ref_untracked.append(item)
        else:
            ref_absent.append(item)
    referenced = set(refs) | {i["what"] for i in an["imports"]}
    m2_untracked = [x for x in others if x.startswith(OWNED_PREFIXES) or re.search(r"(^|[_/])m2([_.]|$)", x)]
    m2_untracked = [x for x in m2_untracked if x not in referenced]
    m2_ignored = [x for x in ignored if x.startswith(OWNED_PREFIXES) and x not in referenced]
    return {
        "n_tracked": len(tracked), "n_py_scope": len(files),
        "untracked_imports": [i for i in an["imports"] if i["kind"] == "UNTRACKED_IMPORT"],
        "gitignored_imports": [i for i in an["imports"] if i["kind"] == "GITIGNORED_IMPORT"],
        "missing_imports": [i for i in an["imports"] if i["kind"] in ("MISSING_IMPORT", "SYNTAX_ERROR")],
        "undeclared_packages": [p for p in an["packages"] if p["kind"] == "UNDECLARED_PACKAGE"],
        "transitive_packages": [p for p in an["packages"] if p["kind"] == "TRANSITIVE_ONLY"],
        "referenced_untracked": ref_untracked, "referenced_gitignored": ref_ignored, "referenced_absent": ref_absent,
        "abs_paths_code": an["abs_paths"], "abs_paths_text": text_abs_paths(root, tset),
        "untracked_m2_unreferenced": m2_untracked, "gitignored_m2_unreferenced": m2_ignored,
        "modified_tracked": git_list(root, "diff", "--name-only"),
        "staged_not_committed": sorted(set(tracked) - head) if head else [],
        "tracked_but_deleted": [t for t in tracked if not (root / t).is_file()],
        "network_tests": network_tests(root),
        "scripts_untracked": [f for f in files if f.startswith("scripts/") and f not in tset],
        "scripts_tracked": [f for f in files if f.startswith("scripts/") and f in tset],
        "tracked": tracked, "head": head, "as_if_extra": [x for x in others if not x.endswith(".pkl")],
    }


def high_findings(s: dict) -> int:
    return sum(len(s[k]) for k in ("untracked_imports", "gitignored_imports", "missing_imports", "referenced_untracked",
                                   "referenced_gitignored", "abs_paths_code")) + \
        len([p for p in s["undeclared_packages"] if not p["optional"]])


# ------------------------------------------------------------------ dynamic steps
def has_cli(path: Path) -> bool:
    t = path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"\bargparse\b|\bclick\b|\btyper\b|\bfire\b", t))


def smoke_step(py: Path, tmp: Path, env: dict, timeout: int) -> dict:
    r = run("smoke request (canonical text, trained artifacts)", [str(py), "-c", SMOKE_CODE, str(tmp), CANONICAL_TEXT], tmp, env, timeout)
    if r["status"] == "FAIL":
        return r
    line = next((l for l in r["stdout"].splitlines() if l.startswith("SMOKE_JSON ")), None)
    why = None
    if line is None:
        why = "no SMOKE_JSON in output"
    else:
        d = json.loads(line[11:])
        r["data"] = d
        if d["resolution_status"] not in TERMINAL:
            why = f"request did not finish (resolution_status={d['resolution_status']})"
        elif "falling back" in r["stderr"] and "Mock" in r["stderr"]:
            why = "risk model missing -> silently fell back to MockRiskClassifier"
        elif d.get("regulator") in (None, "None") and not (tmp / "models/regulator_classifier.pkl").is_file():
            why = "models/regulator_classifier.pkl absent -> no regulator candidates"
    if why:
        r["status"], r["first_error"] = "FAIL", why
    return r


def dynamic_steps(root: Path, a, st: dict) -> tuple:
    steps, extra = [], {}
    tmp = Path(tempfile.mkdtemp(prefix="reglingua_fresh_m2_"))
    try:
        gone = 0
        copy_list = st["tracked"] + (st["as_if_extra"] if a.as_if_committed else [])
        for rel in copy_list:
            src = root / rel
            if not src.is_file():
                gone += 1
                continue
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / rel)
        extra["n_files_copied"] = len(copy_list) - gone
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
        env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        if not a.online:
            env.update(OFFLINE_ENV)
        venv = tmp / ".venv"
        r = run("create clean venv" + (" (+system site-packages)" if a.system_site_packages else ""),
                [sys.executable, "-m", "venv", *(["--system-site-packages"] if a.system_site_packages else []), str(venv)], tmp, env, 600)
        steps.append(r)
        py = venv_python(venv)
        ok = r["status"] == "PASS"
        if ok and a.system_site_packages:
            steps.append(skipped("pip install -r requirements.txt", "NOT RUN (--system-site-packages): uses packages already installed here, so this is NOT a clean-install result"))
        elif ok:
            r = run("pip install -r requirements.txt", [str(py), "-m", "pip", "install", "-r", "requirements.txt", *a.pip_arg], tmp, env, a.timeout_pip)
            steps.append(r)
            ok = r["status"] == "PASS"
            if ok:
                pl = run("pip list", [str(py), "-m", "pip", "list", "--format=json"], tmp, env, 120)
                try:
                    extra["n_packages_installed"] = len(json.loads(pl["stdout"]))
                except ValueError:
                    pass
        why = "venv/pip step failed (see above); run the local command in the report"
        sc = lambda n: skipped(n, why)
        if not ok:
            steps += [sc("M2 artifact preparation"), sc("M2 test suite"), sc("smoke request"), sc("--help of every M2 script")]
            return steps, extra
        steps.append(run("M2 artifact preparation", [str(py), "scripts/prepare_demo_artifacts.py", *a.prepare_args], tmp, env, a.timeout_step)
                     if (tmp / "scripts/prepare_demo_artifacts.py").is_file() else skipped("M2 artifact preparation", "scripts/prepare_demo_artifacts.py not tracked"))
        r = run("M2 test suite (tests/unit)", [str(py), "-m", "pytest", "tests/unit", "-q", "-rs", "-p", "no:cacheprovider", "--no-header", *a.pytest_arg],
                tmp, env, a.timeout_step)
        r["pytest"] = pytest_summary(r["stdout"])
        steps.append(r)
        steps.append(smoke_step(py, tmp, env, a.timeout_step))
        helps = []
        for rel in st["scripts_tracked"] + (st["scripts_untracked"] if a.as_if_committed else []):
            name = f"--help {rel}"
            if not (tmp / rel).is_file():
                continue
            if not has_cli(tmp / rel) and not a.run_no_argparse:
                helps.append({**skipped(name, "no argparse: --help would execute the script; not run"), "status": "WARN"})
                continue
            h = run(name, [str(py), rel, "--help"], tmp, env, a.timeout_help)
            if h["status"] == "PASS" and "usage" not in h["stdout"].lower():
                h["status"], h["first_error"] = "FAIL", "exit 0 but no usage text (--help ran the script?)"
            helps.append(h)
        for rel in ([] if a.as_if_committed else st["scripts_untracked"]):
            helps.append({"step": f"--help {rel}", "status": "FAIL", "seconds": 0.0, "returncode": None, "stdout": "", "stderr": "",
                          "first_error": "script exists locally but is untracked: absent from a fresh checkout"})
        extra["help"] = helps
        steps.append(None)  # placeholder: expanded in the report
        return steps, extra
    finally:
        if a.keep:
            extra["kept"] = str(tmp)
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ report
def cell(s: str, n: int = 200) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").replace("`", "'")[:n]


def lst(items, fmt, cap=None) -> list:
    rows = [fmt(i) for i in items]
    return rows[:cap] + ([f"... +{len(rows) - cap} more (full list in {JSON_REL})"] if cap and len(rows) > cap else [])


def render(root: Path, a, st: dict, steps: list, extra: dict, high: int, verdict: str, head: str) -> str:
    L = []
    L += ["# Fresh-checkout simulation (M2)", "",
          f"Generated by `scripts/simulate_fresh_checkout_m2.py` on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
          f"({platform.system()} {platform.release()}, Python {platform.python_version()}); git HEAD `{head}`; "
          f"{st['n_tracked']} tracked files" + (f", {extra['n_files_copied']} copied" if "n_files_copied" in extra else "") + ". "
          f"Raw results: `{JSON_REL}`. Mode: {'static checks only' if a.static_only else ('online' if a.online else 'offline (HF_HUB_OFFLINE=1)')}{'; AS-IF-COMMITTED: untracked non-ignored files (not *.pkl) were copied too' if a.as_if_committed else ''}{'; venv inherits installed packages, pip install NOT run (sandbox has no PyPI access) - not a clean-install result' if a.system_site_packages else ''}. "
          "Copy = `git ls-files` (index) with working-tree content; a teammate's clone gets HEAD, so files that are only staged still need a commit.", "",
          f"**Verdict: {verdict}** ({high} HIGH static finding(s); steps below).", "",
          "Decision-support / research software, not legal certification.", ""]
    L += ["## Steps", "", "| # | Step | Status | Time (s) | First error |", "|---|---|---|---|---|"]
    n = 0
    flat = []
    for s in steps:
        if s is None:
            flat += extra.get("help", [])
        else:
            flat.append(s)
    if a.static_only:
        flat = [skipped("venv, pip install, prepare, tests, smoke, --help", "--static-only")]
    for s in flat:
        if s["step"].startswith("--help "):
            continue
        n += 1
        extra_txt = ""
        if "pytest" in s:
            extra_txt = f" [{s['pytest']['line']}]"
        L.append(f"| {n} | {cell(s['step'])} | {s['status']} | {s['seconds']} | {cell(s['first_error'] + extra_txt)} |")
    hs = extra.get("help", [])
    if hs:
        c = Counter(h["status"] for h in hs)
        n += 1
        L.append(f"| {n} | `--help` of {len(hs)} scripts | {'PASS' if c['FAIL'] == 0 and c['WARN'] == 0 else 'FAIL' if c['FAIL'] else 'WARN'} | "
                 f"{round(sum(h['seconds'] for h in hs), 2)} | {c['PASS']} PASS, {c['FAIL']} FAIL, {c['WARN']} WARN (per-script table below) |")
    ps = next((s.get("pytest") for s in flat if s.get("pytest")), None)
    if ps and ps["skip_reasons"]:
        L += ["", "Skipped tests (pytest `-rs`), by reason:", ""] + [f"- {v} x {k}" for k, v in sorted(ps["skip_reasons"].items(), key=lambda kv: -kv[1])]
    if hs:
        L += ["", "### `--help` per script", "", "| Script | Status | Time (s) | First error |", "|---|---|---|---|"]
        L += [f"| `{h['step'][7:]}` | {h['status']} | {h['seconds']} | {cell(h['first_error'])} |" for h in hs]
    L += ["", "## Static findings", "", "| Finding | Count | Severity |", "|---|---|---|"]
    rows = [("Untracked module imported by M2 code", "untracked_imports", "HIGH"), ("Gitignored module imported", "gitignored_imports", "HIGH"),
            ("Missing module / syntax error", "missing_imports", "HIGH"), ("Referenced data/model file exists locally but is untracked", "referenced_untracked", "HIGH"),
            ("Referenced file is gitignored", "referenced_gitignored", "HIGH"), ("Hard-coded absolute path in M2 code/comments", "abs_paths_code", "HIGH"),
            ("Third-party package not in requirements.txt", "undeclared_packages", "HIGH (optional imports excluded from verdict)"),
            ("Package only transitively installed", "transitive_packages", "INFO"), ("Path literal absent everywhere (output or typo; verify)", "referenced_absent", "INFO"),
            ("Absolute path in docs / data / manifest", "abs_paths_text", "WARN"), ("Untracked M2-owned file, not referenced (commit it)", "untracked_m2_unreferenced", "WARN"),
            ("Gitignored M2-owned file, not referenced", "gitignored_m2_unreferenced", "INFO"), ("Tracked file modified vs HEAD", "modified_tracked", "WARN"),
            ("Staged but not committed", "staged_not_committed", "WARN"), ("Tracked but deleted on disk", "tracked_but_deleted", "HIGH")]
    for label, k, sev in rows:
        L.append(f"| {label} | {len(st[k])} | {sev} |")
    P = lambda i: f"`{i['file']}:{i['line']}` {cell(i['what'], 90)}"
    sections = [
        ("Untracked / gitignored / missing imports", st["untracked_imports"] + st["gitignored_imports"] + st["missing_imports"], lambda i: f"- {P(i)} ({i['kind']})", None),
        ("Referenced files that are untracked or gitignored (inputs or outputs; absent in a clone)", st["referenced_untracked"] + st["referenced_gitignored"],
         lambda i: f"- `{i['what']}` (first ref `{i['file']}`, {i['n_refs']} ref(s))", None),
        ("Packages not in requirements.txt", st["undeclared_packages"] + st["transitive_packages"],
         lambda i: f"- {P(i)} {'[optional import] ' if i['optional'] else ''}{('(transitive via ' + i['via'] + ')') if i['via'] else '(NOT installed by requirements.txt)'}", None),
        ("Hard-coded absolute paths", st["abs_paths_code"] + st["abs_paths_text"], lambda i: f"- {P(i)} [{i['where']}]", None),
        ("Untracked M2-owned files not referenced by code", st["untracked_m2_unreferenced"], lambda i: f"- `{i}`", None),
        ("Path literals absent everywhere", st["referenced_absent"], lambda i: f"- `{i['what']}` (`{i['file']}`)", 40),
        ("Modified / staged / deleted tracked files", [("modified", x) for x in st["modified_tracked"]] + [("staged-not-committed", x) for x in st["staged_not_committed"]] +
         [("deleted", x) for x in st["tracked_but_deleted"]], lambda i: f"- {i[0]}: `{i[1]}`", None),
    ]
    for title, items, fmt, cap in sections:
        if items:
            L += ["", f"### {title}", ""] + lst(items, fmt, cap)
    L += ["", "### Tests that touch Hugging Face / network (static scan of tests/unit)", ""]
    nt = st["network_tests"]
    if nt:
        L += ["| Test file | Patterns | `hf_network` marker | Fakes/hermetic hint |", "|---|---|---|---|"]
        L += [f"| `{t['file']}` | {cell(', '.join(t['patterns']))} | {'yes' if t['marked'] else 'no'} | {'yes' if t['hermetic_hint'] else 'NO'} |" for t in nt]
    else:
        L.append("None found.")
    L += ["", "Policy: `tests/unit/conftest.py` registers `@pytest.mark.hf_network` (skipped with a reason when offline) and turns an unmarked "
          "network/HF-offline error into a skip listed at the end of the run. Assertion failures and missing packages are never masked."]
    L += ["", "## Run it locally", "", "Windows Command Prompt:", "", "```", "git clone <repo-url> RegLingua && cd RegLingua",
          "python scripts\\simulate_fresh_checkout_m2.py", "python scripts\\simulate_fresh_checkout_m2.py --online   (allow network)",
          "python scripts\\simulate_fresh_checkout_m2.py --as-if-committed   (what-if: also copy untracked files, except *.pkl)",
          "python scripts\\simulate_fresh_checkout_m2.py --static-only", "```", "", "bash:", "", "```",
          "git clone <repo-url> RegLingua && cd RegLingua", "python3 scripts/simulate_fresh_checkout_m2.py",
          "python3 scripts/simulate_fresh_checkout_m2.py --online", "python3 scripts/simulate_fresh_checkout_m2.py --as-if-committed",
          "python3 scripts/simulate_fresh_checkout_m2.py --static-only", "```", "",
          "Run it from your working repo after `git add`-ing every file listed above (the simulation reads the index; only committed files reach a clone). "
          "Limits: path literals are found only as whole string constants (`\"data/x.csv\"`), not built from pieces; `--help` of scripts without argparse is not run."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_ROOT), help="repo to simulate (default: this repo)")
    ap.add_argument("--static-only", action="store_true", help="skip venv/pip/steps; static checks only")
    ap.add_argument("--online", action="store_true", help="do not set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE for the steps")
    ap.add_argument("--system-site-packages", action="store_true",
                    help="no network/PyPI: venv inherits installed packages, pip install is skipped; verdict can never be PASS")
    ap.add_argument("--as-if-committed", action="store_true",
                    help="what-if: also copy untracked, non-ignored files (except *.pkl), i.e. what `git add -A` would stage; verdict can never be PASS")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder and print its path")
    ap.add_argument("--pip-arg", action="append", default=[], help="extra pip install argument (repeatable), e.g. --pip-arg=--index-url=URL")
    ap.add_argument("--prepare-args", nargs="*", default=["--verify"], help="args for prepare_demo_artifacts.py (default: --verify)")
    ap.add_argument("--pytest-arg", action="append", default=[], help="extra pytest argument (repeatable), e.g. --pytest-arg=--continue-on-collection-errors")
    ap.add_argument("--run-no-argparse", action="store_true", help="also run scripts without argparse with --help (executes them)")
    ap.add_argument("--timeout-pip", type=int, default=3000)
    ap.add_argument("--timeout-step", type=int, default=900)
    ap.add_argument("--timeout-help", type=int, default=120)
    ap.add_argument("--report", default=REPORT_REL)
    ap.add_argument("--json", default=JSON_REL)
    a = ap.parse_args()
    root = Path(a.repo).resolve()
    if git(root, "rev-parse", "--is-inside-work-tree").returncode != 0:
        print("not a git repository: `git ls-files` needed", file=sys.stderr)
        return 2
    st = static_checks(root)
    steps, extra = ([], {}) if a.static_only else dynamic_steps(root, a, st)
    flat = [s for s in steps if s is not None] + extra.get("help", [])
    failed = [s for s in flat if s["status"] in ("FAIL",)]
    high = high_findings(st)
    if a.static_only:
        verdict = "STATIC ONLY - NOT A GATE RESULT" + (" (HIGH findings present)" if high else "")
    else:
        verdict = "PASS" if not failed and high == 0 and not any(s["status"] == "SKIP" for s in flat) else "NOT READY"
        if a.as_if_committed or a.system_site_packages:
            tags = " + ".join(t for t, on in (("WHAT-IF: untracked files copied", a.as_if_committed), ("NOT A CLEAN INSTALL", a.system_site_packages)) if on)
            verdict = f"PARTIAL - NOT A GATE RESULT ({tags}; {len(failed)} failed step(s), {high} HIGH static finding(s))"
    head = (git(root, "rev-parse", "--short", "HEAD").stdout.strip() or "none")
    rep = render(root, a, st, steps, extra, high, verdict, head)
    rp, jp = root / a.report, root / a.json
    for p in (rp, jp):
        p.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(rep, encoding="utf-8")
    slim = [{**{k: v for k, v in s.items() if k not in ("stdout", "stderr")}, "output_tail": (s["stderr"] + "\n" + s["stdout"]).strip()[-600:]} for s in flat]
    jp.write_text(json.dumps({"generated": dt.datetime.now().isoformat(timespec="seconds"), "head": head, "verdict": verdict, "mode": vars(a),
                              "steps": slim, "extra": {k: v for k, v in extra.items() if k != "help"},
                              "static": {k: v for k, v in st.items() if k not in ("tracked", "head")}}, indent=1, default=str), encoding="utf-8")
    c = Counter(s["status"] for s in flat)
    print(f"verdict: {verdict}; steps: {dict(c)}; HIGH static findings: {high}\nreport: {rp}\njson:   {jp}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
