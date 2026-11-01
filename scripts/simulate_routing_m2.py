"""Day 9 (M2): simulate supervisor routing on held-out rows under the current constants vs M2's
three abstention operating points. Read-only: agents/supervisor.py is NOT imported or edited
(services/evaluation/routing_sim.py mirrors route_after_classification; a unit test guards drift).

Paths: route "compliant" -> mark_compliant (compliant fast path); "needs_review" -> gather_evidence;
"abstain" -> needs_review node (human review). False-compliant fast path = routed to the compliant
fast path while true_risk_level != LOW. Configs come from abstention_thresholds_m2.json (chosen on
non-held-out rows); nothing here is tuned. Bootstrap 95% CI, seed 42, 2000 resamples.

Usage (repo root): python scripts/simulate_routing_m2.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.classifier.abstention import AbstentionConfig  # noqa: E402
from services.evaluation.confidence_signals import SEED, bootstrap, percentile_ci  # noqa: E402
from services.evaluation.routing_sim import (  # noqa: E402
    ROUTE_TO_PATH, SUPERVISOR_ABSTENTION_CONFIDENCE_FLOOR as A, SUPERVISOR_COMPLIANT_CONFIDENCE_FLOOR as C,
    route_current, route_with_config,
)
from services.classifier.abstention import decide  # noqa: E402

DATA = REPO_ROOT / "data" / "evaluation"
SIGNALS, EVAL, THRESH = DATA / "confidence_signals_m2.csv", DATA / "risk_evidence_eval_m2.csv", DATA / "abstention_thresholds_m2.json"
OUT = DATA / "routing_simulation_m2.csv"
pc = lambda x: "n/a" if x is None else f"{x*100:.1f}%"  # noqa: E731


def load_held_out():
    ev = {r["query_id"]: r for r in csv.DictReader(open(EVAL, newline="", encoding="utf-8-sig"))}
    rows = []
    for r in csv.DictReader(open(SIGNALS, newline="", encoding="utf-8-sig")):
        if r["split"] != "held_out":
            continue
        e = ev[r["query_id"]]
        rows.append({"text": e["text"], "language": e["language"], "regulator": e["true_regulator"],
                     "true_risk": e["true_risk_level"],
                     "probs": {k[2:]: float(v) for k, v in r.items() if k in ("p_HIGH", "p_LOW", "p_MEDIUM")}})
    return rows


def main() -> None:
    held = load_held_out()
    pts = json.loads(THRESH.read_text(encoding="utf-8"))["operating_points"]
    scenarios = {f"current_constants (a={A}, c={C})": (lambda r: route_current(r["probs"]), None)}
    for name in ("conservative", "balanced", "permissive"):
        cfg = AbstentionConfig.from_dict(pts[name]["config"])
        scenarios[name] = ((lambda r, cfg=cfg: route_with_config(r["probs"], r["text"], r["regulator"], r["language"], cfg)), cfg)

    # sanity: the new logic with default floors vs the mirrored supervisor routing on the same rows
    dflt = AbstentionConfig()
    diff = [r for r in held if route_with_config(r["probs"], r["text"], r["regulator"], r["language"], dflt) != route_current(r["probs"])]
    print(f"held-out n={len(held)} | decide(default config) vs supervisor mirror: {len(held)-len(diff)}/{len(held)} same route"
          + (f" (differences are the conflict rule: {sum(decide(r['probs'], r['text'], r['regulator'], r['language'], dflt).signals['conflict'] for r in diff)}/{len(diff)})" if diff else ""))

    true_low = np.array([r["true_risk"] == "LOW" for r in held])
    cols = ["scenario", "abstain_floor", "compliant_floor", "n", "compliant_fast_path", "gather_evidence", "abstain",
            "fast_path_share", "fast_lo", "fast_hi", "gather_share", "gather_lo", "gather_hi", "abstain_share", "abstain_lo",
            "abstain_hi", "false_compliant_fast_path", "fc_lo", "fc_hi", "false_compliant_high", "true_low_total", "true_low_on_fast_path"]
    out, summary = [], {}
    for name, (fn, cfg) in scenarios.items():
        routes = np.array([fn(r) for r in held])
        fast, gather, ab = (routes == k for k in ("compliant", "needs_review", "abstain"))
        fc = fast & ~true_low
        fch = fast & np.array([r["true_risk"] == "HIGH" for r in held])
        b = bootstrap({"fast": lambda i: fast[i].mean(), "gather": lambda i: gather[i].mean(), "ab": lambda i: ab[i].mean(),
                       "fc": lambda i: float(fc[i].sum())}, len(held))
        ci = {k: percentile_ci(v)[:2] for k, v in b.items()}
        a_f, c_f = (cfg.abstain_floor, cfg.compliant_floor) if cfg else (A, C)
        out.append([name, a_f, c_f, len(held), int(fast.sum()), int(gather.sum()), int(ab.sum()),
                    f"{fast.mean():.4f}", f"{ci['fast'][0]:.4f}", f"{ci['fast'][1]:.4f}", f"{gather.mean():.4f}", f"{ci['gather'][0]:.4f}",
                    f"{ci['gather'][1]:.4f}", f"{ab.mean():.4f}", f"{ci['ab'][0]:.4f}", f"{ci['ab'][1]:.4f}", int(fc.sum()),
                    f"{ci['fc'][0]:.0f}", f"{ci['fc'][1]:.0f}", int(fch.sum()), int(true_low.sum()), int((fast & true_low).sum())])
        summary[name] = (int(fast.sum()), int(gather.sum()), int(ab.sum()), int(fc.sum()), ci["fc"], int((fast & true_low).sum()))
        print(f"\n[{name}] fast_path={fast.sum()} ({pc(fast.mean())}) gather_evidence={gather.sum()} ({pc(gather.mean())}) "
              f"abstain->human_review={ab.sum()} ({pc(ab.mean())}) | false_compliant_fast_path={fc.sum()} [{ci['fc'][0]:.0f}-{ci['fc'][1]:.0f}] "
              f"(HIGH: {fch.sum()}) | true-LOW rows on fast path: {(fast & true_low).sum()}/{true_low.sum()}")
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(out)
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
