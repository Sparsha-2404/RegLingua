"""Day 8 (M2) gate: "Rewriter cannot declare its own success; Verifier decides."

(a) RewriteResponse carries no compliance/success claim field.
(b) Static AST scan: where `verified_compliant` is written in non-test code.
(c) A rewriter that CLAIMS "fully compliant" but contains a banned phrase still FAILs.
(d) Can a no-op rewrite PASS the current VerifierAgent? Run, don't assume: it can, so
    the desired behaviour is an expected failure (strict xfail) until M4/M1 close it.

Cases below are SYNTHETIC; classifiers are FAKE (always LOW, i.e. the worst case where
the classifier agrees with the rewriter's claim).
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from agents.risk_agent_adapter import RiskAgentAdapter
from agents.rewriter_agent import RewriterAgent
from agents.verifier_agent import VerifierAgent
from core.schemas import CaseState, EventType, Regulator, RewriteRequest, RewriteResponse
from services.classifier.contracts import RiskPrediction
from services.verification.independent_recheck import RecheckFlag, recheck

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM_WORDS = re.compile(r"compliant|compliance|success|passed|pass_|verified|verdict|approved|safe|valid|ok$", re.I)
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "site-packages", "__pycache__", ".pytest_cache", "tests"}

# Files allowed to assign `verified_compliant`. The real verifier is the only one that
# should be; the two mock verifiers are listed explicitly so they are visible, not hidden.
REAL_VERIFIER = "agents/verifier_agent.py"
MOCK_VERIFIERS = {"agents/mock_agents.py", "reglingua_supervisor/agents/mock_agents.py"}


class FakeAlwaysLowClassifier:
    """FAKE: calls everything LOW/Compliant with high confidence."""

    def predict(self, text):
        return RiskPrediction(risk_level="LOW", confidence=0.99)


def make_verifier():
    return VerifierAgent(risk_agent=RiskAgentAdapter(classifier=FakeAlwaysLowClassifier()))


def verify(original, rewrite, regulator=Regulator.SEBI):
    state = CaseState(input_text=original, regulator=regulator, rewrite_candidate=rewrite)
    state = make_verifier().verify(state)
    run = [e for e in state.trace if e.event_type == EventType.VERIFICATION_RUN][-1]
    return state.verified_compliant, run.payload["failure_reasons"]


# ------------------------------------------------------------------ (a)
def test_rewrite_response_has_no_compliance_or_success_field():
    fields = set(RewriteResponse.model_fields)
    offenders = {f for f in fields if CLAIM_WORDS.search(f)}
    assert not offenders, f"RewriteResponse exposes claim-like fields: {offenders}"


def test_rewrite_response_field_set_tripwire():
    # Tripwire, not a spec: if M3 adds a field, re-review it for claim semantics, then update.
    assert set(RewriteResponse.model_fields) == {"case_id", "rewritten_text", "attempt_number", "rationale"}


def test_rewriter_returns_rewrite_response():
    ann = inspect.signature(RewriterAgent.rewrite).return_annotation
    assert ann is RewriteResponse or ann == "RewriteResponse"


# ------------------------------------------------------------------ (b)
def _source_files():
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if SKIP_DIRS & set(rel.parts) or rel.name.startswith("test_") or rel.name == "conftest.py":
            continue
        yield rel.as_posix(), path


def _writes_verified_compliant(tree: ast.AST) -> list[int]:
    """Line numbers where code assigns/sets `verified_compliant` on an object.

    Detects: `x.verified_compliant = ...` (also AnnAssign/AugAssign/tuple targets),
    `setattr(x, "verified_compliant", ...)`, and `CaseState(verified_compliant=...)`.
    Not detected: dynamic names (setattr with a variable), `model_copy(update=...)`.
    Class-body field declarations and read-only copies into API models are not writes.
    """
    hits = []

    def attr_targets(t):
        if isinstance(t, ast.Attribute):
            yield t
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                yield from attr_targets(e)

    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for t in targets:
            hits += [a.lineno for a in attr_targets(t) if a.attr == "verified_compliant"]
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name == "setattr" and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value == "verified_compliant":
                hits.append(node.lineno)
            if name == "CaseState" and any(k.arg == "verified_compliant" for k in node.keywords):
                hits.append(node.lineno)
    return hits


def _writers() -> dict[str, list[int]]:
    found = {}
    for rel, path in _source_files():
        hits = _writes_verified_compliant(ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel))
        if hits:
            found[rel] = sorted(hits)
    return found


def test_scanner_detects_writes_and_ignores_reads_and_declarations():
    # SYNTHETIC snippets for the scanner itself.
    assert _writes_verified_compliant(ast.parse("state.verified_compliant = True"))
    assert _writes_verified_compliant(ast.parse("a.verified_compliant, b = 1, 2"))
    assert _writes_verified_compliant(ast.parse("setattr(s, 'verified_compliant', True)"))
    assert _writes_verified_compliant(ast.parse("CaseState(input_text='x', verified_compliant=True)"))
    assert not _writes_verified_compliant(ast.parse("x = state.verified_compliant"))
    assert not _writes_verified_compliant(ast.parse("class M:\n    verified_compliant: bool = None"))
    assert not _writes_verified_compliant(ast.parse("Resp(verified_compliant=result.verified_compliant)"))


def test_real_verifier_is_a_writer_and_no_other_non_mock_module_writes():
    writers = set(_writers())
    assert REAL_VERIFIER in writers, "scan failed to find the real verifier's assignment"
    unexpected = writers - {REAL_VERIFIER} - MOCK_VERIFIERS
    assert not unexpected, f"verified_compliant written outside verifier modules: {sorted(unexpected)}"


@pytest.mark.xfail(strict=True, reason="Mock verifiers in agents/mock_agents.py and "
                   "reglingua_supervisor/agents/mock_agents.py also assign verified_compliant "
                   "(derived from re-classification, not rewriter claims). Strict 'only "
                   "agents/verifier_agent.py' fails until M4 removes/renames them.")
def test_strict_only_real_verifier_assigns_verified_compliant():
    assert set(_writers()) == {REAL_VERIFIER}


# ------------------------------------------------------------------ (c)
class FakeLyingRewriter:
    """FAKE rewriter: asserts compliance in text and rationale while keeping banned phrases."""

    def rewrite(self, request):
        return RewriteResponse(
            case_id=request.case_id,
            rewritten_text="Fully compliant and SEBI approved. We guarantee 20% returns, risk-free.",
            attempt_number=request.attempt_number,
            rationale="Fully compliant; verified safe; no further review needed.",
        )


def test_rewriter_claiming_compliance_with_banned_phrase_fails():
    original = "We guarantee 20% returns."
    resp = FakeLyingRewriter().rewrite(RewriteRequest(
        case_id="G1", flagged_text=original, clause_cited="SEBI_EVAL", reason="synthetic", attempt_number=1))
    passed, reasons = verify(original, resp.rewritten_text)
    assert passed is False
    assert "banned_phrase_present" in reasons  # FAIL comes from policy even though classifier says LOW


def test_verdict_independent_of_rewriter_rationale_and_claims():
    bad = "We guarantee 20% returns, risk-free."
    base = verify("x", bad)
    claim = verify("x", "Fully compliant. " + bad)
    assert base[0] is False and claim[0] is False
    assert "banned_phrase_present" in base[1] and "banned_phrase_present" in claim[1]


# ------------------------------------------------------------------ (d)
NOOP_ORIGINAL = "Big wins are coming your way, subject to market risk."  # SYNTHETIC


def test_report_noop_behaviour_and_independent_detector():
    passed, reasons = verify(NOOP_ORIGINAL, NOOP_ORIGINAL)  # rewrite == original
    # Observed (not assumed): printed for the record; the xfail test below pins the policy gap.
    print(f"no-op rewrite -> verified_compliant={passed}, reasons={reasons}")
    r = recheck(NOOP_ORIGINAL, NOOP_ORIGINAL, FakeAlwaysLowClassifier(), "SEBI")
    assert RecheckFlag.NOOP_REWRITE in r.flags  # independent recheck does see the no-op


@pytest.mark.xfail(strict=True, reason="Current verify() never reads state.input_text, so a no-op "
                   "rewrite of an original that already satisfies the policy PASSES. Desired: "
                   "no-op must not PASS. Remove this marker once M4/M1 use recheck().")
def test_noop_rewrite_must_not_pass_verifier():
    passed, _ = verify(NOOP_ORIGINAL, NOOP_ORIGINAL)
    assert passed is not True
