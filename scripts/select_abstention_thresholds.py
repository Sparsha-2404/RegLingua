"""Day 9 (M2): choose abstention operating points on NON-held-out rows, report on held-out.

Inputs (from real runs): data/evaluation/confidence_signals_m2.csv (scripts/analyze_confidence_signals.py:
out-of-fold probabilities for non-held-out rows, committed-model probabilities for held-out rows) and
data/evaluation/risk_evidence_eval_m2.csv (text, language, labels).
Selection sees only non-held-out rows (services/evaluation/abstention_eval refuses held-out rows).
Outputs: abstention_thresholds_m2.json, abstention_operating_points_m2.csv, abstention_risk_coverage_m2.csv.

Declared before looking at held-out (constants below):
  accept = decide(...).action == ACCEPT; false-compliant = accepted LOW whose true label is not LOW.
  Every point maximises coverage s.t. false-compliant <= budget and accuracy-on-accepted >= MIN_ACCURACY.
  conservative: budget 0 AND, if it accepts any LOW, exact 95% upper bound on its false-compliant rate
  <= MAX_FC_UCB (else compliant acceptance is switched off: compliant_floor = 1.0).
  balanced / permissive: budget 1% / 3% of non-held-out rows, no safety rule (flagged unsafe if they fail it).

Usage (repo root): python scripts/analyze_confidence_signals.py && python scripts/select_abstention_thresholds.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.classifier.abstention import AbstentionConfig, Action, decide  # noqa: E402
from services.evaluation.abstention_eval import (  # noqa: E402
    clopper_pearson_upper, decide_rows, frange, grid_metrics, per_row_flags, select_point, summarize,
)
from services.evaluation.confidence_signals import SEED, bootstrap, percentile_ci  # noqa: E402

DATA = REPO_ROOT / "data" / "evaluation"
SIGNALS, EVAL = DATA / "confidence_signals_m2.csv", DATA / "risk_evidence_eval_m2.csv"
OUT_JSON, OUT_CSV, OUT_CURVE = (DATA / "abstention_thresholds_m2.json", DATA / "abstention_operating_points_m2.csv",
                                DATA / "abstention_risk_coverage_m2.csv")
MIN_ACCURACY, MAX_FC_UCB = 0.70, 0.10
FC_FRACTION = {"conservative": 0.0, "balanced": 0.01, "permissive": 0.03}
pc = lambda x: "n/a" if x is None else f"{x*100:.1f}%"  # noqa: E731


def load_rows():
    ev = {r["query_id"]: r for r in csv.DictReader(open(EVAL, newline="", encoding="utf-8-sig"))}
    rows = []
    for r in csv.DictReader(open(SIGNALS, newline="", encoding="utf-8-sig")):
        e = ev[r["query_id"]]
        rows.append({"query_id": r["query_id"], "held_out": r["split"] == "held_out", "text": e["text"],
                     "language": e["language"], "regulator": e["true_regulator"], "true_risk": e["true_risk_level"],
                     "probs": {k[2:]: float(v) for k, v in r.items() if k in ("p_HIGH", "p_LOW", "p_MEDIUM")}})
    return rows


def ci_fmt(draws, kind="pct"):
    lo, hi, _ = percentile_ci(draws)
    return "n/a" if lo is None else (f"[{pc(lo)}-{pc(hi)}]" if kind == "pct" else f"[{lo:.0f}-{hi:.0f}]")


def main() -> None:
    rows = load_rows()
    train = [r for r in rows if not r["held_out"]]            # selection data
    n_tr = len(train)

    # ---- selection on non-held-out only
    a_vals, c_vals = frange(0.35, 0.70), frange(0.35, 1.00)
    table = grid_metrics(train, a_vals, c_vals)
    curve = [(cfg.abstain_floor, m) for cfg, m in table if cfg.abstain_floor == cfg.compliant_floor]
    chosen, notes = {}, {}
    for name, frac in FC_FRACTION.items():
        budget = int(frac * n_tr)
        pick = select_point(table, budget, MIN_ACCURACY, MAX_FC_UCB if name == "conservative" else None)
        chosen[name] = (pick, budget)
    zero_fc = [(m["low_accepted"], m["fc_upper95_among_low_accepted"]) for _, m in table
               if m["false_compliant"] == 0 and m["low_accepted"] > 0]
    kmax = max(zero_fc)[0] if zero_fc else 0
    ucb_kmax = clopper_pearson_upper(0, kmax) if kmax else None
    k_needed = next(k for k in range(1, 1000) if clopper_pearson_upper(0, k) <= MAX_FC_UCB)

    # ---- held-out report (configs are frozen above)
    held = [r for r in rows if r["held_out"]]
    out_rows, json_points = [], {}
    for name, (pick, budget) in chosen.items():
        if pick is None:
            json_points[name] = {"config": None, "infeasible": True, "false_compliant_budget": budget}
            continue
        cfg, m_tr = pick
        safe = m_tr["low_accepted"] > 0 and m_tr["fc_upper95_among_low_accepted"] <= MAX_FC_UCB
        json_points[name] = {
            "config": cfg.to_dict(), "false_compliant_budget_non_held_out": budget,
            "compliant_acceptance": "enabled" if m_tr["low_accepted"] > 0 else "disabled",
            "safe_for_compliant_acceptance": bool(safe), "selected_on_non_held_out": m_tr}
    references = {"supervisor_default (reference, not tuned)": AbstentionConfig()}
    evals = {n: AbstentionConfig.from_dict(j["config"]) for n, j in json_points.items() if j.get("config")}
    evals.update(references)

    print(f"non-held-out n={n_tr} (selection) | held-out n={len(held)} (report only) | seed={SEED}")
    print(f"\nRisk-coverage curve, non-held-out, threshold t on max_prob (a=c=t):")
    print("  t     coverage  acc_accepted  false_compliant  LOW_accepted")
    csv_curve = []
    for t, m in curve:
        csv_curve.append([t, m["accepted"], f"{m['coverage']:.4f}", "" if m["accuracy_on_accepted"] is None else f"{m['accuracy_on_accepted']:.4f}",
                          m["false_compliant"], m["low_accepted"]])
        if round(t * 100) % 5 == 0:
            print(f"  {t:.2f}  {pc(m['coverage']):>8}  {pc(m['accuracy_on_accepted']):>12}  {m['false_compliant']:>15}  {m['low_accepted']:>12}")
    with open(OUT_CURVE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["threshold", "accepted", "coverage", "accuracy_on_accepted", "false_compliant", "low_accepted"])
        w.writerows(csv_curve)

    print("\nChosen on non-held-out -> reported on held-out (95% bootstrap CI, seed 42)")
    cols = ["operating_point", "group_type", "group", "n", "accept", "review", "abstain", "coverage", "coverage_lo",
            "coverage_hi", "accuracy_on_accepted", "acc_lo", "acc_hi", "false_compliant", "fc_lo", "fc_hi", "low_accepted",
            "abstain_floor", "compliant_floor"]
    for name, cfg in evals.items():
        ref = name in references
        d = decide_rows(held, cfg)
        fl = per_row_flags(held, d)
        s = summarize(fl)
        arr = {k: np.array(v) for k, v in fl.items()}
        boot = bootstrap({
            "cov": lambda i: arr["accept"][i].mean(),
            "acc": lambda i: (arr["correct"][i][arr["accept"][i] == 1].mean() if arr["accept"][i].sum() else float("nan")),
            "fc": lambda i: float(arr["fc"][i].sum())}, len(held))
        c_lo, c_hi, _ = percentile_ci(boot["cov"]); a_lo, a_hi, _ = percentile_ci(boot["acc"]); f_lo, f_hi, _ = percentile_ci(boot["fc"])
        tr = json_points.get(name, {}).get("selected_on_non_held_out")
        print(f"\n[{name}] a={cfg.abstain_floor:.2f} c={cfg.compliant_floor:.2f}"
              + (f" | non-held-out: cov={pc(tr['coverage'])} acc={pc(tr['accuracy_on_accepted'])} FC={tr['false_compliant']} LOWacc={tr['low_accepted']}" if tr else ""))
        print(f"  held-out: coverage={pc(s['coverage'])} {ci_fmt(boot['cov'])} | accuracy_on_accepted={pc(s['accuracy_on_accepted'])} {ci_fmt(boot['acc'])}"
              f" | false_compliant={s['false_compliant']} {ci_fmt(boot['fc'], 'n')} (of {s['low_accepted']} LOW accepted) | review={s['review']} abstain={s['abstain']}")
        out_rows.append([name, "all", "all", s["n"], s["accepted"], s["review"], s["abstain"], f"{s['coverage']:.4f}", f"{c_lo:.4f}", f"{c_hi:.4f}",
                         "" if s["accuracy_on_accepted"] is None else f"{s['accuracy_on_accepted']:.4f}", "" if a_lo is None else f"{a_lo:.4f}",
                         "" if a_hi is None else f"{a_hi:.4f}", s["false_compliant"], f"{f_lo:.0f}", f"{f_hi:.0f}", s["low_accepted"],
                         cfg.abstain_floor, cfg.compliant_floor])
        for gtype, key in (("language", "language"), ("regulator", "regulator")):
            groups = defaultdict(list)
            for r, dec in zip(held, d):
                groups[r[key]].append(dec.action)
            line = []
            for g, acts in sorted(groups.items()):
                n = len(acts)
                na, nr, nb = (sum(a is x for a in acts) for x in (Action.ACCEPT, Action.REVIEW, Action.ABSTAIN))
                out_rows.append([name, gtype, g, n, na, nr, nb, f"{na/n:.4f}"] + [""] * 8 + [cfg.abstain_floor, cfg.compliant_floor])
                line.append(f"{g}: A{na}/R{nr}/B{nb} of {n}")
            print(f"  by {gtype}: " + "; ".join(line))
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(out_rows)

    any_safe = any(j.get("safe_for_compliant_acceptance") for j in json_points.values())
    verdict = (f"No operating point can safely auto-accept Compliant predictions. On non-held-out rows the most LOW predictions "
               f"that can be accepted with zero false-compliant is {kmax} (exact 95% upper bound on the false-compliant rate "
               f"{pc(ucb_kmax)}); the safety rule (upper bound <= {MAX_FC_UCB:.0%}) needs at least {k_needed} such acceptances with zero errors. "
               f"The conservative point therefore switches compliant acceptance off (compliant_floor = 1.0)."
               if not any_safe else "At least one operating point passes the compliant-acceptance safety rule.")
    print("\nVERDICT:", verdict)
    meta = {"selection_data": "non-held-out rows only (out-of-fold probabilities)", "n_non_held_out": n_tr, "n_held_out": len(held),
            "grid": {"abstain_floor": [a_vals[0], a_vals[-1], 0.01], "compliant_floor": [c_vals[0], c_vals[-1], 0.01]},
            "min_accuracy_on_accepted": MIN_ACCURACY, "max_fc_upper95_for_compliant_acceptance": MAX_FC_UCB,
            "false_compliant_budget_fraction": FC_FRACTION, "bootstrap_seed": SEED,
            "false_compliant_definition": "accepted as LOW (compliant) while true_risk_level != LOW"}
    OUT_JSON.write_text(json.dumps({"meta": meta, "verdict": verdict, "operating_points": json_points}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSON.name}, {OUT_CSV.name}, {OUT_CURVE.name}")


if __name__ == "__main__":
    main()
