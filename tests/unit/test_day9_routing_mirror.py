"""Day 9 (M2): drift guard - the routing mirror must equal the real supervisor routing.

Compares services/evaluation/routing_sim.mirror_route_after_classification with the real
agents.supervisor.route_after_classification over a grid of (label, confidence) pairs,
the floor constants, and the classify_risk edge map. Grid values are SYNTHETIC.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agents import supervisor as sup
from core.schemas import CaseState, RiskLabel
from services.classifier.abstention import AbstentionConfig, decide
from services.evaluation import routing_sim as sim
from services.evaluation.routing_demo import patched_constants

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = list(RiskLabel)
EDGE_CONFS = [0.0, 0.1, 0.3499, 0.35, 0.3501, 0.5, 0.6999, 0.7, 0.7001, 0.99, 1.0]
SWEEP = [round(i * 0.005, 3) for i in range(0, 201)]


def real_route(label, conf):
    return sup.route_after_classification(CaseState(input_text="x", risk_label=label, classifier_confidence=conf))


def test_mirror_constants_equal_supervisor_constants():
    assert sim.SUPERVISOR_COMPLIANT_CONFIDENCE_FLOOR == sup._COMPLIANT_CONFIDENCE_FLOOR
    assert sim.SUPERVISOR_ABSTENTION_CONFIDENCE_FLOOR == sup._ABSTENTION_CONFIDENCE_FLOOR


@pytest.mark.parametrize("label", LABELS)
def test_mirror_route_equals_real_route_over_grid(label):
    for conf in EDGE_CONFS + SWEEP:
        assert sim.mirror_route_after_classification(label.value, conf) == real_route(label, conf), (label, conf)


def test_mirror_handles_missing_confidence_like_real():
    for label in LABELS:
        assert sim.mirror_route_after_classification(label.value, None) == real_route(label, None) == "abstain"


def test_route_to_node_map_matches_supervisor_graph_edges():
    tree = ast.parse((REPO_ROOT / "agents" / "supervisor.py").read_text(encoding="utf-8-sig"))
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_conditional_edges" and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "classify_risk"):
            found = ast.literal_eval(node.args[2])
    assert found == sim.SUPERVISOR_ROUTE_TO_NODE


def test_label_mapping_covers_every_classifier_level_and_risk_label():
    assert set(sim.LEVEL_TO_RISK_LABEL) == {"HIGH", "MEDIUM", "LOW", "NEEDS_REVIEW"}
    assert set(sim.LEVEL_TO_RISK_LABEL.values()) <= {l.value for l in RiskLabel}


def _vec(label, conf):
    rest = (1.0 - conf) / 2
    return {k: (conf if k == label else rest) for k in ("LOW", "MEDIUM", "HIGH")}


@pytest.mark.parametrize("label", ["LOW", "MEDIUM", "HIGH"])
def test_default_abstention_config_routes_like_supervisor_on_clean_text(label):
    cfg = AbstentionConfig()
    for conf in [0.34, 0.3499, 0.35, 0.36, 0.5, 0.6999, 0.7, 0.7001, 0.9, 1.0]:
        probs = _vec(label, conf) if conf < 1.0 else {label: 1.0}
        want = real_route(RiskLabel(sim.LEVEL_TO_RISK_LABEL[label]), max(probs.values()))
        got = decide(probs, "Investments are subject to market risks.", None, None, cfg).route_hint
        assert got == want == sim.route_current(probs), (label, conf)


def test_route_current_uses_argmax_label_and_confidence():
    assert sim.route_current({"LOW": 0.8, "MEDIUM": 0.1, "HIGH": 0.1}) == "compliant"
    assert sim.route_current({"LOW": 0.6, "MEDIUM": 0.2, "HIGH": 0.2}) == "needs_review"
    assert sim.route_current({"LOW": 0.34, "MEDIUM": 0.33, "HIGH": 0.33}) == "abstain"
    assert sim.route_current({"HIGH": 0.9, "MEDIUM": 0.05, "LOW": 0.05}) == "needs_review"


@pytest.mark.parametrize("label", LABELS)
def test_mirror_with_patched_floors_equals_real_route_under_patched_constants(label):
    for a, c in [(0.42, 0.59), (0.35, 0.6), (0.5, 0.5), (0.0, 1.0)]:
        with patched_constants(sup, a, c):
            for conf in EDGE_CONFS + [a, c, round(a - 1e-4, 6), round(c - 1e-4, 6)] + SWEEP[::5]:
                assert sim.mirror_route_after_classification(label.value, conf, a, c) == real_route(label, conf), (label, a, c, conf)
    assert sup._ABSTENTION_CONFIDENCE_FLOOR == 0.35 and sup._COMPLIANT_CONFIDENCE_FLOOR == 0.7  # restored
