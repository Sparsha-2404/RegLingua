"""Static contract snapshot (Day 11, M2).

Reads contract classes straight from source with `ast` (no pydantic/langgraph/model
imports), so the snapshot is reproducible anywhere. Records, per class: kind, bases,
decorators, ordered fields (name, annotation, default marker) and enum members.
Default marker: "<required>", a literal repr, or "<expr>" (non-literal) -- so edits to
descriptions/factories don't cause noise, but literal default changes do.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / "data" / "fixtures" / "contract_snapshot_m2.json"

# path -> class names. OWNER marks who may change them.
TARGETS: dict[str, list[str]] = {
    "services/classifier/contracts.py": ["RiskLevel", "RiskPrediction", "AgentOutput"],
    "services/verification/independent_recheck.py": ["RecheckFlag", "RecheckResult"],
    "services/classifier/abstention.py": ["Action", "Reason", "AbstentionDecision"],
    "core/contracts.py": ["RiskLevel", "RiskPrediction", "AgentOutput"],
    "core/schemas.py": ["RiskLabel", "Regulator", "Language", "ResolutionStatus", "EventType",
                        "EvidenceItem", "AgentEvent", "CaseState"],
    "agents/research_contracts.py": ["RiskPrediction"],
    "apps/api/main.py": ["ScanRequest", "ScanResponse"],
}
M2_PATHS = {"services/classifier/contracts.py", "services/verification/independent_recheck.py",
            "services/classifier/abstention.py"}


def owner(path: str) -> str:
    return "M2" if path in M2_PATHS else "read-only (M3/M4)"


def _default_marker(value: ast.expr | None) -> str:
    if value is None:
        return "<required>"
    if isinstance(value, ast.Constant):
        return "<required>" if value.value is Ellipsis else repr(value.value)
    if isinstance(value, ast.Call) and ast.unparse(value.func).split(".")[-1] == "Field":
        if value.args:
            return _default_marker(value.args[0])
        for kw in value.keywords:
            if kw.arg == "default":
                return _default_marker(kw.value)
        return "<expr>"
    return "<expr>"


def _extract_class(node: ast.ClassDef) -> dict[str, Any]:
    bases = [ast.unparse(b) for b in node.bases]
    decorators = [ast.unparse(d) for d in node.decorator_list]
    is_enum = any(b.split(".")[-1] in ("Enum", "IntEnum", "StrEnum") for b in bases)
    if is_enum:
        kind = "enum"
    elif any(d.split("(")[0].split(".")[-1] == "dataclass" for d in decorators):
        kind = "dataclass"
    elif any(b.split(".")[-1] == "BaseModel" for b in bases):
        kind = "pydantic"
    else:
        kind = "class"
    out: dict[str, Any] = {"kind": kind, "bases": bases, "decorators": decorators}
    if is_enum:
        out["members"] = {
            s.targets[0].id: ast.unparse(s.value)
            for s in node.body
            if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
        }
    else:
        out["fields"] = [
            {"name": s.target.id, "type": ast.unparse(s.annotation), "default": _default_marker(s.value)}
            for s in node.body
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            and not ast.unparse(s.annotation).startswith("ClassVar")
        ]
    return out


def build_snapshot(root: Path = ROOT) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for rel, names in TARGETS.items():
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        found = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        for name in names:
            key = f"{rel}::{name}"
            if name not in found:
                classes[key] = {"kind": "MISSING"}
                continue
            entry = _extract_class(found[name])
            entry["owner"] = owner(rel)
            classes[key] = entry
    return {
        "_meta": {
            "purpose": "Day 11 contract freeze (M2). Regenerate only via scripts/update_contract_snapshot.py --update.",
            "method": "static ast extraction; field names, annotations, default markers, enum members",
        },
        "classes": classes,
    }


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(snapshot: dict[str, Any], path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Human-readable drift lines; empty list means identical."""
    msgs: list[str] = []
    oc, nc = old["classes"], new["classes"]
    for key in sorted(set(oc) | set(nc)):
        who = (nc.get(key) or oc.get(key) or {}).get("owner", "?")
        if key not in nc:
            msgs.append(f"[{who}] {key}: class no longer tracked")
            continue
        if key not in oc:
            msgs.append(f"[{who}] {key}: new class not in snapshot")
            continue
        o, n = oc[key], nc[key]
        if n.get("kind") == "MISSING" or o.get("kind") == "MISSING":
            if o.get("kind") != n.get("kind"):
                msgs.append(f"[{who}] {key}: class {'REMOVED/RENAMED' if n.get('kind') == 'MISSING' else 'reappeared'}")
            continue
        for attr in ("kind", "bases", "decorators"):
            if o.get(attr) != n.get(attr):
                msgs.append(f"[{who}] {key}: {attr} {o.get(attr)!r} -> {n.get(attr)!r}")
        if "members" in o or "members" in n:
            om, nm = o.get("members", {}), n.get("members", {})
            for m in sorted(set(om) | set(nm)):
                if m not in nm:
                    msgs.append(f"[{who}] {key}: enum member {m} removed")
                elif m not in om:
                    msgs.append(f"[{who}] {key}: enum member {m} added ({nm[m]})")
                elif om[m] != nm[m]:
                    msgs.append(f"[{who}] {key}: enum member {m} value {om[m]} -> {nm[m]}")
        if "fields" in o or "fields" in n:
            of = {f["name"]: f for f in o.get("fields", [])}
            nf = {f["name"]: f for f in n.get("fields", [])}
            for f in of:
                if f not in nf:
                    msgs.append(f"[{who}] {key}.{f}: field removed (was {of[f]['type']})")
                else:
                    for a in ("type", "default"):
                        if of[f][a] != nf[f][a]:
                            msgs.append(f"[{who}] {key}.{f}: {a} {of[f][a]!r} -> {nf[f][a]!r}")
            for f in nf:
                if f not in of:
                    msgs.append(f"[{who}] {key}.{f}: field added ({nf[f]['type']})")
            if [f for f in of if f in nf] != [f for f in nf if f in of]:
                msgs.append(f"[{who}] {key}: field ORDER changed")
    return msgs


# ---------------------------------------------------------------------------
# Architecture-plan response vs live API model (report only; never edits main.py)
# ---------------------------------------------------------------------------
PLAN_RESPONSE = {
    "case_id": "str",
    "status": "str (e.g. 'completed')",
    "risk": {"label": "HIGH|LOW", "confidence": "float", "factors": "list[str]"},
    "regulators": "list[str]",
    "evidence": [{"source_id": "str", "section": "str", "text": "str", "score": "float"}],
    "explanation": "str",
    "rewrite": "str",
    "verification": {"verdict": "PASS|FAIL|REVIEW", "before_risk": "str", "after_risk": "str"},
    "agent_trace": [{"agent": "str", "status": "str"}],
}


def api_routes(root: Path = ROOT) -> list[str]:
    tree = ast.parse((root / "apps/api/main.py").read_text(encoding="utf-8"))
    routes = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.args \
                        and isinstance(d.args[0], ast.Constant) and d.func.attr in ("get", "post", "put", "delete"):
                    routes.append(f"{d.func.attr.upper()} {d.args[0].value}")
    return routes


def api_vs_plan(snapshot: dict[str, Any]) -> list[str]:
    """Mismatches between PLAN_RESPONSE and the live ScanResponse/CaseState, as text for M4."""
    cls = snapshot["classes"]
    resp = {f["name"]: f["type"] for f in cls["apps/api/main.py::ScanResponse"]["fields"]}
    state = {f["name"]: f["type"] for f in cls["core/schemas.py::CaseState"]["fields"]}
    ev = [f["name"] for f in cls["core/schemas.py::EvidenceItem"]["fields"]]
    lab = list(cls["core/schemas.py::RiskLabel"]["members"].values())
    res = list(cls["core/schemas.py::ResolutionStatus"]["members"].values())
    routes = api_routes()
    m: list[str] = []
    if "POST /api/analyze" not in routes:
        m.append(f"endpoint: plan POST /api/analyze; live routes {routes}")
    req = [f["name"] for f in cls["apps/api/main.py::ScanRequest"]["fields"]]
    m.append(f"request: plan {{text, mode, language}}; live {req}")
    if "case_id" in resp:
        m.append(f"case_id: OK ({resp['case_id']})")
    m.append(f"status: plan 'status' ('completed'); live 'resolution_status' with values {res}")
    m.append(f"risk: plan nested {{label, confidence, factors}}; live flat 'risk_label' ({resp.get('risk_label')}) "
             f"with labels {lab} (plan uses HIGH/LOW); confidence NOT exposed "
             f"(CaseState.classifier_confidence exists); factors NOT exposed (CaseState.risk_phrase only)")
    m.append(f"regulators: plan list; live single 'regulator' ({resp.get('regulator')})")
    m.append(f"evidence: plan list of {{source_id, section, text, score}}; live MISSING from response "
             f"(CaseState.evidence exists; EvidenceItem fields {ev})")
    m.append(f"explanation: plan 'explanation'; live 'reason' + 'clause_cited'")
    m.append(f"rewrite: plan 'rewrite'; live 'rewrite_candidate'")
    m.append(f"verification: plan {{verdict PASS/FAIL/REVIEW, before_risk, after_risk}}; live only "
             f"'verified_compliant' ({resp.get('verified_compliant')}); no before/after risk, no verdict")
    trace = "trace" in state
    trace_route = [r for r in routes if "trace" in r]
    m.append(f"agent_trace: plan list in response; live MISSING from response "
             f"(CaseState.trace exists: {trace}; route {trace_route} is a 501 stub)")
    extra = [k for k in resp if k not in ("case_id", "risk_label", "regulator", "reason", "rewrite_candidate",
                                          "verified_compliant", "resolution_status")]
    m.append(f"live-only fields not in plan: {extra}")
    return m
