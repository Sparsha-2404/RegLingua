"""
DAY 11 (M2): demo-readiness helpers (manifest, git fresh-checkout simulation, offline smoke request).

Everything environment-touching is injectable (runner, git). Nothing here edits M3/M4 files; the
supervisor/agent_factory are only *executed* (in a subprocess) or *read as text*.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

SEED = 42
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MANIFEST_REL = "models/MANIFEST_m2.json"
CANONICAL_TEXT = "This mutual fund offers guaranteed 20% returns with zero risk."
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONHASHSEED": "0"}
TERMINAL = {"resolved", "escalated", "needs_review"}


@dataclass(frozen=True)
class Item:
    file: str
    role: str
    required: bool          # required for the demo gate (False = optional / training input only)
    produced_by: str = ""   # how to (re)create it, shown in reports


ITEMS: List[Item] = [
    Item("models/risk_classifier_tfidf_logreg.pkl", "risk classifier (committed baseline)", True,
         "scripts/train_baseline_classifier.py (committed; --force to retrain)"),
    Item("models/regulator_classifier.pkl", "regulator classifier (local, NOT committed)", True,
         "python scripts\\prepare_demo_artifacts.py (scripts/train_regulator_classifier.py alone fails: no regulator column in committed CSV)"),
    Item("data/regulatory/regulatory_kb_m1.json", "M1 knowledge base (retrieval corpus)", True, "committed (M1)"),
    Item("data/raw/labeled_examples_m1.csv", "training input for the classifiers", False, "committed (M1)"),
    Item("models/risk_classifier_embedding_logreg_minilm.pkl", "embedding classifier (optional)", False,
         "prepare_demo_artifacts.py --with-embedding"),
    Item("models/risk_classifier_embedding_logreg_minilm.pkl.meta.json", "embedding sidecar (optional)", False,
         "written next to the embedding artifact"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": sys.version.split()[0]}
    for pkg in ("scikit-learn", "numpy", "sentence-transformers", "langgraph"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def pickle_sklearn_version(path: Path) -> Optional[str]:
    """scikit-learn version a pickle was WRITTEN with. sklearn drops it on load, but warns when it differs from
    the installed one (InconsistentVersionWarning.original_sklearn_version); no warning => same as installed."""
    import pickle
    import warnings

    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            pickle.load(open(path, "rb"))
        for w in ws:
            v = getattr(w.message, "original_sklearn_version", None)
            if v:
                return str(v)
        return env_versions()["scikit-learn"]
    except Exception:  # noqa: BLE001
        return None


def build_entry(root: Path, item: Item, training_rows=None, recipe: str = "", tracked: Optional[bool] = None) -> Optional[dict]:
    p = Path(root) / item.file
    if not p.is_file():
        return None
    v = env_versions()
    e = {"file": item.file, "sha256": sha256_file(p), "size_bytes": p.stat().st_size, "role": item.role,
         "required": item.required, "git_tracked": tracked, "installed_sklearn": v["scikit-learn"],
         "installed_numpy": v["numpy"], "training_rows": training_rows, "training_recipe": recipe}
    if item.file.endswith(".pkl"):
        e["trained_with_sklearn"] = pickle_sklearn_version(p)
    return e


def write_manifest(root: Path, entries: Sequence[dict], seed: int = SEED) -> Path:
    path = Path(root) / MANIFEST_REL
    doc = {"generated_by": "scripts/prepare_demo_artifacts.py", "seed": seed, "environment": env_versions(),
           "files": sorted(entries, key=lambda e: e["file"])}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # no timestamp: deterministic
    return path


def read_manifest(root: Path) -> Optional[dict]:
    p = Path(root) / MANIFEST_REL
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_files(root: Path, manifest: Optional[dict]) -> List[dict]:
    """One row per ITEM: status PASS / WARN / FAIL with reason. Manifest hash must match."""
    by = {e["file"]: e for e in (manifest or {}).get("files", [])}
    inst = env_versions()["scikit-learn"]
    rows = []
    for it in ITEMS:
        p = Path(root) / it.file
        if not p.is_file():
            rows.append({"file": it.file, "status": "FAIL" if it.required else "WARN",
                         "detail": ("missing" if it.required else "missing (optional)") + f"; create: {it.produced_by}"})
            continue
        e = by.get(it.file)
        if manifest is None:
            rows.append({"file": it.file, "status": "FAIL", "detail": f"{MANIFEST_REL} missing; run scripts/prepare_demo_artifacts.py"})
        elif e is None:
            rows.append({"file": it.file, "status": "FAIL" if it.required else "WARN", "detail": "not in manifest; rerun prepare"})
        elif sha256_file(p) != e["sha256"]:
            rows.append({"file": it.file, "status": "FAIL", "detail": "sha256 differs from manifest (file changed after prepare)"})
        else:
            tw = e.get("trained_with_sklearn")
            if tw and inst and tw != inst:
                rows.append({"file": it.file, "status": "WARN",
                             "detail": f"hash OK; pickled with scikit-learn {tw}, installed {inst} (version warning on load)"})
            else:
                rows.append({"file": it.file, "status": "PASS", "detail": "hash matches manifest"})
    return rows


# ---------------------------------------------------------------- smoke request (subprocess, offline)
_SMOKE_CODE = r"""
import json, sys, time
root, text = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
t = time.time()
from agents.supervisor import run_case
s = run_case(text, use_trained_artifacts=True)
g = lambda v: getattr(v, "value", v)
print("SMOKE_JSON " + json.dumps({
    "resolution_status": g(s.resolution_status), "risk_label": g(s.risk_label), "regulator": g(s.regulator),
    "classifier_confidence": s.classifier_confidence, "n_evidence": len(s.evidence),
    "rewrite_attempts": s.rewrite_attempts, "verified_compliant": s.verified_compliant,
    "trace_events": len(s.trace), "seconds": round(time.time() - t, 2)}))
"""


@dataclass
class SmokeResult:
    status: str                       # PASS / DEGRADED / FAIL
    reasons: List[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    stderr_tail: str = ""


def run_smoke_subprocess(root: Path, text: str = CANONICAL_TEXT, timeout: int = 300) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(OFFLINE_ENV)
    try:
        p = subprocess.run([sys.executable, "-c", _SMOKE_CODE, str(root), text], cwd=str(root), env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def classify_failure(stderr: str) -> str:
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr)
    if m:
        return f"missing package '{m.group(1)}' (pip install -r requirements.txt)"
    m = re.search(r"FileNotFoundError: .*?'([^']+)'", stderr) or re.search(r"FileNotFoundError: (.+)", stderr)
    if m:
        return f"missing file {m.group(1).strip()}"
    last = [l for l in stderr.strip().splitlines() if l.strip()]
    return last[-1][:200] if last else "no error output"


def evaluate_smoke(raw: dict, root: Path, missing_required: Sequence[str] = ()) -> SmokeResult:
    err = raw.get("stderr", "")
    if raw.get("returncode") != 0:
        return SmokeResult("FAIL", [classify_failure(err)], stderr_tail="\n".join(err.strip().splitlines()[-3:]))
    line = next((l for l in raw.get("stdout", "").splitlines() if l.startswith("SMOKE_JSON ")), None)
    if line is None:
        return SmokeResult("FAIL", ["no SMOKE_JSON in output"], stderr_tail=err[-300:])
    data = json.loads(line[len("SMOKE_JSON "):])
    if data["resolution_status"] not in TERMINAL:
        return SmokeResult("FAIL", [f"request did not finish (resolution_status={data['resolution_status']})"], data)
    reasons = []
    if "falling back" in err and "Mock" in err:
        reasons.append("risk model missing -> silently fell back to MockRiskClassifier (agent_factory stderr note)")
    for f in missing_required:
        reasons.append(f"required file absent: {f}")
    if data.get("regulator") in (None, "None") and "models/regulator_classifier.pkl" in missing_required:
        reasons.append("no regulator_candidates (regulator_classifier.pkl absent)")
    return SmokeResult("DEGRADED" if reasons else "PASS", reasons, data)


# ---------------------------------------------------------------- git fresh-checkout simulation
def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True)


def git_ls_files(root: Path) -> List[str]:
    p = _git(root, "ls-files", "-z")
    if p.returncode != 0:
        raise RuntimeError("git ls-files failed: " + p.stderr.decode(errors="replace").strip())
    return [x for x in p.stdout.decode("utf-8").split("\0") if x]


def git_modified_tracked(root: Path) -> List[str]:
    p = _git(root, "status", "--porcelain", "--untracked-files=no")
    return [l[3:] for l in p.stdout.decode(errors="replace").splitlines()] if p.returncode == 0 else []


def git_is_ignored(root: Path, rel: str) -> bool:
    return _git(root, "check-ignore", "-q", rel).returncode == 0


def copy_tracked(root: Path, dest: Path, files: Sequence[str]) -> List[str]:
    """Copy ONLY the listed files (working-tree content). Returns tracked files that no longer exist on disk."""
    gone = []
    for rel in files:
        src = Path(root) / rel
        if not src.is_file():
            gone.append(rel)
            continue
        out = Path(dest) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    return gone


def untracked_reasons(root: Path, tracked: Sequence[str], items: Sequence[Item] = tuple(ITEMS)) -> List[dict]:
    """Demo files present locally but NOT in git (absent from a fresh clone), with why."""
    ts = set(tracked)
    out = []
    for it in items:
        if it.file in ts:
            continue
        local = (Path(root) / it.file).is_file()
        why = ("gitignored" if git_is_ignored(root, it.file) else "untracked (never `git add`ed)") if local else "not in repo and not on disk"
        out.append({"file": it.file, "required": it.required, "local_copy": local, "why": why, "create": it.produced_by})
    return out


def fresh_checkout_smoke(root: Path, runner: Callable[[Path], dict] = run_smoke_subprocess, keep: bool = False) -> dict:
    root = Path(root)
    tracked = git_ls_files(root)
    tmp = Path(tempfile.mkdtemp(prefix="reglingua_fresh_"))
    try:
        gone = copy_tracked(root, tmp, tracked)
        absent = untracked_reasons(root, tracked)
        missing_required = [a["file"] for a in absent if a["required"]]
        res = evaluate_smoke(runner(tmp), tmp, missing_required)
        return {"result": res, "n_files": len(tracked) - len(gone), "tracked_but_deleted": gone, "absent": absent,
                "modified_tracked": git_modified_tracked(root), "path": str(tmp) if keep else None}
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- read-only wiring re-check (M3/M4 files, text only)
def retriever_wiring(root: Path) -> dict:
    root = Path(root)
    src = (root / "agents" / "agent_factory.py")
    text = src.read_text(encoding="utf-8") if src.is_file() else ""
    hyb = (root / "services" / "retrieval" / "hybrid_retriever.py").is_file()
    anc = _git(root, "merge-base", "--is-ancestor", "d72dfb2", "HEAD").returncode == 0
    code = re.sub(r"#.*", "", text)
    injects_det = bool(re.search(r"DeterministicRetriever\(", code))
    uses_hyb = "HybridRetriever" in code
    return {"agent_factory_injects_keyword_only_retriever": injects_det and not uses_hyb,
            "hybrid_retriever_file_present": hyb, "fix_commit_d72dfb2_in_history": anc}


def day7_gate_rerun(root: Path, timeout: int = 600) -> Optional[int]:
    """Re-run scripts/check_day7_gate.py (writes to a temp CSV, not the committed one); COMPLETE count or None."""
    out = Path(tempfile.mkdtemp()) / "gate.csv"
    env = {k: v for k, v in os.environ.items()}
    env.update(OFFLINE_ENV)
    try:
        p = subprocess.run([sys.executable, str(Path(root) / "scripts" / "check_day7_gate.py"), "--output", str(out)],
                           cwd=str(root), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    n_rows = sum(1 for l in p.stdout.splitlines() if ": COMPLETE" in l or ": INCOMPLETE" in l)
    return sum(1 for l in p.stdout.splitlines() if ": COMPLETE" in l) if n_rows else None   # exit code is non-zero when <10/10


def clean_machine_commands() -> List[str]:
    return [
        "git clone <repo-url> RegLingua && cd RegLingua",
        "python -m venv .venv && .venv\\Scripts\\activate          (POSIX: source .venv/bin/activate)",
        "pip install -r requirements.txt",
        "python -c \"from sentence_transformers import SentenceTransformer as S; S('%s')\"   # ONE online run to fill the HF cache" % ENCODER_NAME,
        "python scripts\\prepare_demo_artifacts.py                  # trains models\\regulator_classifier.pkl, verifies baseline, writes models\\MANIFEST_m2.json",
        "python scripts\\prepare_demo_artifacts.py --with-embedding # optional: embedding classifier (needs the cached encoder)",
        "set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1        (POSIX: export ...)",
        "python scripts\\check_demo_readiness.py",
    ]
