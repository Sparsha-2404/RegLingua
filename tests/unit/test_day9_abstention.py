"""Day 9 (M2): tests for services/classifier/abstention.py.

All probability vectors and texts are SYNTHETIC (hand-written); no model is loaded.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
from enum import Enum
from pathlib import Path

import pytest

from services.classifier.abstention import (
    AbstentionConfig, Action, Reason, decide, entropy, top2_margin,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = "Investments are subject to market risks."
BANNED_GLOBAL = "This plan is risk-free for everyone."   # global banned phrase
BANNED_SEBI = "Our fund has sure-shot profit."             # SEBI-only banned phrase

LOW_HIGH_CONF = {"LOW": 0.8, "MEDIUM": 0.15, "HIGH": 0.05}
HIGH_CLEAR = {"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1}


def run(probs, text=CLEAN, **kw):
    return decide(probs, text, **kw)


# ------------------------------------------------------------------ every reason
def test_accept_low_when_all_checks_pass():
    d = run(LOW_HIGH_CONF)
    assert d.action is Action.ACCEPT and d.reasons == [Reason.ACCEPTED] and d.route_hint == "compliant"


def test_accept_non_low_label_routes_to_needs_review_like_supervisor():
    d = run(HIGH_CLEAR)
    assert d.action is Action.ACCEPT and d.signals["label"] == "HIGH" and d.route_hint == "needs_review"


def test_reason_below_abstain_floor():
    d = run({"LOW": 0.34, "MEDIUM": 0.33, "HIGH": 0.33})
    assert d.action is Action.ABSTAIN and d.reasons == [Reason.BELOW_ABSTAIN_FLOOR] and d.route_hint == "abstain"


def test_reason_conflict_banned_phrase_low_never_accepts():
    d = run({"LOW": 0.9, "MEDIUM": 0.07, "HIGH": 0.03}, BANNED_GLOBAL)
    assert d.action is Action.REVIEW and d.reasons == [Reason.CONFLICT_BANNED_PHRASE_LOW]
    assert d.signals["conflict"] is True and d.signals["banned_phrase"] == "risk-free"
    assert d.route_hint == "needs_review"


def test_reason_low_below_compliant_floor():
    d = run({"LOW": 0.6, "MEDIUM": 0.3, "HIGH": 0.1})
    assert d.action is Action.REVIEW and d.reasons == [Reason.LOW_BELOW_COMPLIANT_FLOOR]


def test_reason_low_margin_and_high_entropy_when_enabled():
    cfg = AbstentionConfig(margin_floor=0.5, entropy_ceiling=0.5)
    d = run({"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2}, config=cfg)
    assert d.action is Action.REVIEW and d.reasons == [Reason.LOW_MARGIN, Reason.HIGH_ENTROPY]


def test_margin_and_entropy_rules_are_off_by_default():
    d = run({"HIGH": 0.4, "MEDIUM": 0.35, "LOW": 0.25})   # tiny margin, high entropy
    assert d.action is Action.ACCEPT


def test_reason_classifier_needs_review():
    d = run({"NEEDS_REVIEW": 0.6, "LOW": 0.2, "HIGH": 0.2})
    assert d.action is Action.REVIEW and d.reasons == [Reason.CLASSIFIER_NEEDS_REVIEW]


def test_every_reason_is_exercised():
    seen = set()
    scenarios = [
        run(LOW_HIGH_CONF),
        run({"LOW": 0.34, "MEDIUM": 0.33, "HIGH": 0.33}),
        run({"LOW": 0.9, "MEDIUM": 0.07, "HIGH": 0.03}, BANNED_GLOBAL),
        run({"LOW": 0.6, "MEDIUM": 0.3, "HIGH": 0.1}),
        run({"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2}, config=AbstentionConfig(margin_floor=0.5, entropy_ceiling=0.5)),
        run({"NEEDS_REVIEW": 0.6, "LOW": 0.2, "HIGH": 0.2}),
    ]
    for d in scenarios:
        seen.update(d.reasons)
    assert seen == set(Reason)


# ------------------------------------------------------------------ boundaries
def test_abstain_floor_boundary_is_strict_less_than():
    at = run({"HIGH": 0.35, "MEDIUM": 0.35, "LOW": 0.3})
    below = run({"HIGH": 0.3499, "MEDIUM": 0.3499, "LOW": 0.3002})
    assert at.action is Action.ACCEPT and at.signals["max_prob"] == 0.35
    assert below.action is Action.ABSTAIN


def test_compliant_floor_boundary_is_inclusive():
    at = run({"LOW": 0.7, "MEDIUM": 0.2, "HIGH": 0.1})
    below = run({"LOW": 0.6999, "MEDIUM": 0.2, "HIGH": 0.1001})
    assert at.action is Action.ACCEPT
    assert below.action is Action.REVIEW and below.reasons == [Reason.LOW_BELOW_COMPLIANT_FLOOR]


def test_compliant_floor_does_not_apply_to_non_low_labels():
    d = run({"HIGH": 0.36, "MEDIUM": 0.34, "LOW": 0.30})
    assert d.action is Action.ACCEPT


def test_margin_boundary_strict_less_than():
    p = {"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1}
    m = top2_margin(p)
    assert run(p, config=AbstentionConfig(margin_floor=m)).action is Action.ACCEPT
    d = run(p, config=AbstentionConfig(margin_floor=m + 1e-9))
    assert d.action is Action.REVIEW and d.reasons == [Reason.LOW_MARGIN]


def test_entropy_boundary_strict_greater_than():
    p = {"HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1}
    h = entropy(p)
    assert run(p, config=AbstentionConfig(entropy_ceiling=h)).action is Action.ACCEPT
    d = run(p, config=AbstentionConfig(entropy_ceiling=h - 1e-9))
    assert d.action is Action.REVIEW and d.reasons == [Reason.HIGH_ENTROPY]


def test_config_boundary_values_zero_and_one_are_valid():
    AbstentionConfig(abstain_floor=0.0, compliant_floor=0.0)
    AbstentionConfig(abstain_floor=1.0, compliant_floor=1.0, margin_floor=1.0, entropy_ceiling=0.0)
    assert run({"LOW": 1.0}, config=AbstentionConfig(abstain_floor=1.0, compliant_floor=1.0)).action is Action.ACCEPT


# ------------------------------------------------------------------ conflict rule
def test_conflict_needs_low_label_and_banned_phrase():
    assert run({"HIGH": 0.9, "MEDIUM": 0.07, "LOW": 0.03}, BANNED_GLOBAL).signals["conflict"] is False
    assert run(LOW_HIGH_CONF, CLEAN).signals["conflict"] is False


def test_conflict_uses_regulator_specific_phrases():
    p = {"LOW": 0.9, "MEDIUM": 0.07, "HIGH": 0.03}
    assert run(p, BANNED_SEBI, regulator="SEBI").action is Action.REVIEW
    assert run(p, BANNED_SEBI, regulator=None).action is Action.ACCEPT


def test_conflict_at_low_confidence_abstains_and_records_conflict():
    d = run({"LOW": 0.34, "MEDIUM": 0.33, "HIGH": 0.33}, BANNED_GLOBAL)
    assert d.action is Action.ABSTAIN
    assert d.reasons == [Reason.BELOW_ABSTAIN_FLOOR, Reason.CONFLICT_BANNED_PHRASE_LOW]


def test_banned_phrase_with_low_label_never_accepts_at_any_confidence():
    cfg_loose = AbstentionConfig(abstain_floor=0.0, compliant_floor=0.0)
    for conf in [0.34 + 0.01 * i for i in range(0, 66)]:
        rest = (1.0 - conf) / 2
        p = {"LOW": conf, "MEDIUM": rest, "HIGH": rest}
        for cfg in (None, cfg_loose):
            assert run(p, BANNED_GLOBAL, config=cfg).action is not Action.ACCEPT


# ------------------------------------------------------------------ language, ties, keys
def test_language_floor_overrides_abstain_floor_for_that_language_only():
    cfg = AbstentionConfig(language_abstain_floors={"hi": 0.5})
    p = {"HIGH": 0.45, "MEDIUM": 0.35, "LOW": 0.2}
    assert run(p, language="hi", config=cfg).action is Action.ABSTAIN
    assert run(p, language="en", config=cfg).action is Action.ACCEPT
    assert run(p, language=None, config=cfg).action is Action.ACCEPT
    at = run({"HIGH": 0.5, "MEDIUM": 0.3, "LOW": 0.2}, language="hi", config=cfg)
    assert at.action is Action.ACCEPT and at.signals["abstain_floor_used"] == 0.5


def test_tie_breaks_toward_riskier_label():
    d = run({"LOW": 0.5, "HIGH": 0.5})
    assert d.signals["label"] == "HIGH"
    assert run({"LOW": 0.5, "MEDIUM": 0.5}).signals["label"] == "MEDIUM"


def test_enum_and_lowercase_keys_accepted():
    class Lvl(Enum):
        LOW = "LOW"
        HIGH = "HIGH"

    assert run({Lvl.LOW: 0.8, Lvl.HIGH: 0.2}).signals["label"] == "LOW"
    assert run({"low": 0.8, "high": 0.2}).signals["label"] == "LOW"


# ------------------------------------------------------------------ input validation
@pytest.mark.parametrize("bad", [
    {},
    {"LOW": -0.1, "HIGH": 1.1},
    {"LOW": 0.5, "HIGH": 0.6},
    {"LOW": 0.3, "HIGH": 0.3},
    {"FOO": 1.0},
    {"LOW": float("nan"), "HIGH": 0.5},
    {"LOW": "0.5", "HIGH": 0.5},
    {"LOW": True, "HIGH": 0.0},
])
def test_invalid_probabilities_raise(bad):
    with pytest.raises(ValueError):
        decide(bad, CLEAN)


def test_duplicate_label_via_case_raises():
    with pytest.raises(ValueError):
        decide({"LOW": 0.5, "low": 0.5}, CLEAN)


# ------------------------------------------------------------------ purity
def test_decide_is_pure_and_deterministic():
    p = {"LOW": 0.8, "MEDIUM": 0.15, "HIGH": 0.05}
    snapshot = dict(p)
    a, b = run(p, BANNED_SEBI, regulator="SEBI", language="en"), run(p, BANNED_SEBI, regulator="SEBI", language="en")
    assert p == snapshot and a == b


def test_module_has_no_agents_import():
    src = (REPO_ROOT / "services" / "classifier" / "abstention.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("agents")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("agents") for a in node.names)


# ------------------------------------------------------------------ config
def test_defaults_match_supervisor_constants_tripwire():
    tree = ast.parse((REPO_ROOT / "agents" / "supervisor.py").read_text(encoding="utf-8-sig"))
    vals = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
    cfg = AbstentionConfig()
    assert cfg.abstain_floor == vals["_ABSTENTION_CONFIDENCE_FLOOR"]
    assert cfg.compliant_floor == vals["_COMPLIANT_CONFIDENCE_FLOOR"]


def test_config_round_trip_dict_and_json_string():
    cfg = AbstentionConfig(abstain_floor=0.4, compliant_floor=0.8, margin_floor=0.1, entropy_ceiling=1.0,
                           language_abstain_floors={"hi": 0.5, "hi-en": 0.45})
    assert AbstentionConfig.from_dict(cfg.to_dict()) == cfg
    assert AbstentionConfig.from_json(cfg.to_json()) == cfg
    assert AbstentionConfig.from_json(AbstentionConfig().to_json()) == AbstentionConfig()


def test_config_round_trip_json_file():
    cfg = AbstentionConfig(margin_floor=0.2, language_abstain_floors={"hi": 0.5})
    fd, name = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cfg.to_json(name)
        assert json.loads(Path(name).read_text(encoding="utf-8"))["margin_floor"] == 0.2
        assert AbstentionConfig.from_json(name) == cfg
        assert AbstentionConfig.from_json(Path(name)) == cfg
    finally:
        os.remove(name)


@pytest.mark.parametrize("kwargs", [
    {"abstain_floor": 0.8, "compliant_floor": 0.7},
    {"abstain_floor": -0.1},
    {"compliant_floor": 1.1},
    {"margin_floor": 1.5},
    {"entropy_ceiling": -1.0},
    {"language_abstain_floors": {"hi": 2.0}},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        AbstentionConfig(**kwargs)


def test_unknown_json_key_raises():
    with pytest.raises(ValueError):
        AbstentionConfig.from_json('{"abstain_flor": 0.3}')


def test_decision_to_dict_is_json_serialisable():
    d = run(LOW_HIGH_CONF, BANNED_SEBI, regulator="SEBI")
    out = json.loads(json.dumps(d.to_dict()))
    assert out["action"] == "REVIEW" and out["reasons"] == ["CONFLICT_BANNED_PHRASE_LOW"]
    assert out["route_hint"] == "needs_review"
