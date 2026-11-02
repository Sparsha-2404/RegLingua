"""Day 11 contract freeze (M2). Fails with a readable drift list; see scripts/update_contract_snapshot.py."""
import dataclasses

import pytest

from services.evaluation.contract_snapshot import (
    SNAPSHOT_PATH, api_vs_plan, build_snapshot, diff_snapshots, load_snapshot,
)

HINT = ("\nIf intentional: notify the class owner (read-only classes belong to M3/M4), then run\n"
        "  python scripts\\update_contract_snapshot.py --update")


def test_snapshot_file_exists():
    assert SNAPSHOT_PATH.exists(), "Missing snapshot. Run: python scripts\\update_contract_snapshot.py --update"


def test_no_contract_drift():
    drift = diff_snapshots(load_snapshot(), build_snapshot())
    assert not drift, "CONTRACT DRIFT:\n  " + "\n  ".join(drift) + HINT


def test_all_tracked_classes_exist():
    missing = [k for k, v in build_snapshot()["classes"].items() if v["kind"] == "MISSING"]
    assert not missing, f"Tracked contract classes not found in source: {missing}"


def test_m2_and_core_contracts_are_identical():
    c = build_snapshot()["classes"]
    for name in ("RiskLevel", "RiskPrediction", "AgentOutput"):
        a, b = c[f"services/classifier/contracts.py::{name}"], c[f"core/contracts.py::{name}"]
        strip = lambda d: {k: v for k, v in d.items() if k != "owner"}  # noqa: E731
        assert strip(a) == strip(b), f"{name}: services/classifier/contracts.py differs from core/contracts.py"


def test_runtime_matches_snapshot_for_m2_contracts():
    from services.classifier.contracts import AgentOutput, RiskLevel, RiskPrediction

    snap = load_snapshot()["classes"]
    base = "services/classifier/contracts.py::"
    assert [m.name for m in RiskLevel] == list(snap[base + "RiskLevel"]["members"]), "RiskLevel members differ at runtime"
    for cls in (RiskPrediction, AgentOutput):
        runtime = [f.name for f in dataclasses.fields(cls)]
        assert runtime == [f["name"] for f in snap[base + cls.__name__]["fields"]], f"{cls.__name__} runtime fields differ"


def test_pydantic_runtime_matches_snapshot_if_available():
    pytest.importorskip("pydantic")
    from core.schemas import CaseState

    snap = load_snapshot()["classes"]["core/schemas.py::CaseState"]["fields"]
    assert list(CaseState.model_fields) == [f["name"] for f in snap], "CaseState runtime fields differ from snapshot"


def test_api_gap_report_runs():
    assert api_vs_plan(load_snapshot()), "api_vs_plan returned nothing"
