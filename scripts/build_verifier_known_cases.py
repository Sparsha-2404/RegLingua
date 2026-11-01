"""Day 8 (M2): build data/evaluation/verifier_known_cases_m2.csv (seeded, reproducible).

Real rows come from data/evaluation/rewrite_eval_results_m2.csv (Day 7 run);
rows with any hand-written text are source=synthetic. `expected_verdict` is the
*desired* verdict (annotation), not what verify() currently returns; the script
also prints how today's policy + real classifier agree with it (needs sklearn
and models/risk_classifier_tfidf_logreg.pkl; skipped if unavailable).

Usage (repo root): python scripts/build_verifier_known_cases.py [--seed 8]
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.verification_policy import evaluate, find_banned_phrase, has_required_disclaimer  # noqa: E402

RESULTS = REPO_ROOT / "data" / "evaluation" / "rewrite_eval_results_m2.csv"
OUT = REPO_ROOT / "data" / "evaluation" / "verifier_known_cases_m2.csv"
MODEL = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
COLS = ["case_id", "original", "rewrite", "regulator", "language", "category",
        "expected_verdict", "rationale", "source", "origin_row_id"]
CATS = {"a": "a_noop_risky", "b": "b_fixed_with_disclaimer", "c": "c_no_disclaimer",
        "d": "d_reintroduced_phrase", "e": "e_garbled", "f": "f_hindi_noop", "g": "g_compliant_benign"}

# Real Day 7 rewrites, annotated (RewriterAgent output, unedited).
REAL_B = ["M1_RW001", "M1_RW009", "EVALQ0007"]
REAL_C = ["M1_RW003", "M1_RW004", "M1_RW005", "M1_RW006", "M1_RW008", "M1_RW010", "EVALQ0006", "EVALQ0141"]
REAL_E = ["M1_RW002", "M1_RW007"]

# Synthetic templates: (regulator, language, risky original, safe rewrite, rewrite with phrase softened but no disclaimer)
T = [
    ("SEBI", "en", "We guarantee 15% annual returns on this fund.",
     "This fund aims for competitive annual returns; investments are subject to market risks.",
     "This fund aims for competitive annual returns."),
    ("SEBI", "en", "Double your money in one year with our sure-shot profit scheme.",
     "Our scheme seeks long-term growth; investments are subject to market risks.",
     "Our scheme seeks long-term growth."),
    ("SEBI", "hi-en", "Guaranteed 25% return milega, bilkul risk-free hai.",
     "Hum 25% tak return ka target rakhte hain; investments are subject to market risks.",
     "Hum long-term growth ka target rakhte hain."),
    ("SEBI", "en", "Our fund delivers the highest returns in the market with zero risk.",
     "Our fund aims for competitive returns; past performance does not indicate future results.",
     "Our fund aims for competitive returns."),
    ("RBI", "en", "Get instant approval with zero risk on your personal loan.",
     "Apply for a personal loan; approval is subject to credit approval.",
     "Apply for a personal loan with a quick online process."),
    ("RBI", "en", "No questions asked loan, hidden charges waived forever!",
     "Loans available; terms and conditions apply and all charges are disclosed upfront.",
     "Loans available with all charges disclosed upfront."),
    ("RBI", "hi-en", "Instant approval with zero risk, bas ek click mein loan lo.",
     "Ek click mein apply karo; loan is subject to eligibility.", "Ek click mein apply karo."),
    ("DPDP", "en", "We can sell your data to any partner, no consent required.",
     "We share your data with partners only with your consent.",
     "We handle your data carefully and share it only with partners."),
    ("DPDP", "en", "We share your data with anyone who pays us.",
     "We process your data only as permitted under applicable law.",
     "We process your data for business purposes."),
    ("DPDP", "hi-en", "Aapka data hum kisi ko bhi bech denge, no consent required.",
     "Aapka data hum sirf with your consent share karenge.", "Aapka data hum carefully handle karenge."),
]
G = [  # already-compliant original + benign rewrite (both carry the disclaimer)
    ("SEBI", "en", "Mutual fund investments are subject to market risks. Read all scheme documents carefully.",
     "Mutual fund investments are subject to market risks; please read all scheme documents carefully before investing."),
    ("RBI", "en", "Personal loans are subject to credit approval and eligibility.",
     "Personal loan offers are subject to credit approval; eligibility criteria apply."),
    ("DPDP", "en", "We collect your data with your consent and use it only for the stated purpose.",
     "With your consent, we collect your data and use it only for the stated purpose."),
    ("SEBI", "hi-en", "Investments market-linked hain; past performance is not indicative of future results.",
     "Investments market-linked hain, aur past performance is not indicative of future results."),
    ("RBI", "en", "Loan terms and conditions apply. Apply online today.",
     "Terms and conditions apply to every loan. Apply online today."),
    ("DPDP", "en", "Your data is processed as permitted under applicable law.",
     "We process your data as permitted under applicable law."),
]
REINTRO = {"SEBI": ["highest returns", "sure-shot profit"], "RBI": ["zero risk", "no questions asked loan"],
           "DPDP": ["no consent required", "sell your data"]}
GLOBAL = ["risk-free", "guaranteed", "100% safe", "no risk"]


def garble(text: str, rng: random.Random) -> tuple[str, str]:
    mode = rng.choice(["shuffle", "truncate", "stutter", "noise"])
    words = text.split()
    if mode == "shuffle":
        w = words[:]
        while w == words and len(set(w)) > 1:
            rng.shuffle(w)
        return " ".join(w), "words shuffled"
    if mode == "truncate":
        return text[: max(8, int(len(text) * 0.45))], "truncated mid-sentence"
    if mode == "stutter":
        i = rng.randrange(len(words))
        return " ".join(words[:i] + [words[i]] * 4 + words[i + 1:]), "token repeated"
    return "".join("?" if (c in "aeiou" and rng.random() < 0.6) else c for c in text), "character noise"


def load_real() -> dict[str, dict]:
    with open(RESULTS, newline="", encoding="utf-8-sig") as f:
        return {r["row_id"]: r for r in csv.DictReader(f)}


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    real = load_real()
    rows: list[dict] = []

    def add(cat, orig, rew, reg, lang, verdict, why, src, origin=""):
        rows.append(dict(original=orig, rewrite=rew, regulator=reg, language=lang, category=CATS[cat],
                         expected_verdict=verdict, rationale=why, source=src, origin_row_id=origin))

    def add_real(cat, rid, verdict, why):
        r = real[rid]
        add(cat, r["original_text"], r["rewritten_text"], r["regulator"], r["language"], verdict, why, "real", rid)

    def pick(pool, quota):  # stratified by regulator, seeded
        out = []
        for reg, n in quota.items():
            cands = sorted(r["row_id"] for r in pool if r["regulator"] == reg)
            out += rng.sample(cands, n)
        return out

    unchanged_high = [r for r in real.values() if r["text_unchanged"] == "True" and r["before_label"] == "HIGH"]
    for rid in pick([r for r in unchanged_high if r["language"] != "hi"], {"SEBI": 4, "RBI": 3, "DPDP": 3}):
        add_real("a", rid, "FAIL", "Rewrite identical to risky original (Day 7 real no-op); nothing was fixed.")
    for rid in pick([r for r in unchanged_high if r["language"] == "hi"], {"SEBI": 3, "RBI": 3, "DPDP": 4}):
        add_real("f", rid, "FAIL", "Hindi rewrite identical to risky original (Day 7 real no-op); policy phrase list is English-only.")
    for rid in REAL_B:
        add_real("b", rid, "PASS", "Real rewrite: banned phrase removed and required disclaimer phrase present.")
    for rid in REAL_C:
        add_real("c", rid, "FAIL", "Real rewrite: banned phrase removed but no required regulator disclaimer.")
    for rid in REAL_E:
        add_real("e", rid, "FAIL", "Real rewrite: banned phrase gone but wording incoherent / meaning broken.")

    order = list(range(len(T)))
    rng.shuffle(order)
    for i in order[:7]:
        reg, lang, o, safe, _ = T[i]
        add("b", o, safe, reg, lang, "PASS", "Synthetic: phrase replaced and required disclaimer added.", "synthetic")
    rng.shuffle(order)
    for i in order[:3]:
        reg, lang, o, _, soft = T[i]
        add("c", o, soft, reg, lang, "FAIL", "Synthetic: phrase softened but required disclaimer missing.", "synthetic")
    rng.shuffle(order)
    for i in order[:8]:
        reg, lang, o, safe, _ = T[i]
        bad = rng.choice(REINTRO[reg] + GLOBAL)
        add("d", o, f"{safe} This is {bad}." if " " not in bad else f"{safe} Enjoy {bad}.", reg, lang, "FAIL",
            f"Synthetic: otherwise-good rewrite re-introduces banned phrase '{bad}'.", "synthetic")
    rng.shuffle(order)
    for i in order[:6]:
        reg, lang, o, safe, _ = T[i]
        g, how = garble(safe, rng)
        add("e", o, g, reg, lang, "FAIL", f"Synthetic: garbled rewrite ({how}).", "synthetic")
    for reg, lang, o, rw in G:
        add("g", o, rw, reg, lang, "PASS", "Synthetic: original already compliant; benign rewording keeps disclaimer.", "synthetic")

    rows.sort(key=lambda r: (r["category"], r["source"], r["regulator"], r["origin_row_id"], r["original"]))
    for n, r in enumerate(rows, 1):
        r["case_id"] = f"VK{n:03d}"
    return rows


def validate(rows: list[dict]) -> None:
    """Structural invariants per category, from core.verification_policy (fails loudly)."""
    for r in rows:
        c, reg, o, w = r["category"][0], r["regulator"], r["original"], r["rewrite"]
        banned, disc = find_banned_phrase(w, reg), has_required_disclaimer(w, reg)
        ok = {"a": w == o,
              "f": w == o,
              "b": banned is None and disc,
              "c": banned is None and not disc,
              "d": banned is not None,
              "e": True,
              "g": banned is None and disc and find_banned_phrase(o, reg) is None and has_required_disclaimer(o, reg)}[c]
        if not ok:
            raise AssertionError(f"{r['case_id']} violates category invariant {r['category']}: {w!r}")


def policy_agreement(rows):
    try:
        from services.classifier.baseline import TfidfLogisticClassifier
        from services.classifier.preprocessing import normalize_for_classifier as norm
        clf = TfidfLogisticClassifier().load(str(MODEL))
    except Exception as exc:  # sklearn / model missing
        print(f"\n[agreement skipped: {type(exc).__name__}]")
        return
    mp = {"LOW": "Compliant", "MEDIUM": "Ambiguous", "HIGH": "Risky", "NEEDS_REVIEW": "Ambiguous"}
    agree, tot = collections.Counter(), collections.Counter()
    for r in rows:
        p = clf.predict(norm(r["rewrite"])["normalized_text"])
        v = evaluate(r["rewrite"], r["regulator"], mp[p.risk_level.value], p.confidence)
        tot[r["category"]] += 1
        agree[r["category"]] += ("PASS" if v.passed else "FAIL") == r["expected_verdict"]
    print("\nToday's policy.evaluate + real classifier vs expected_verdict (agree/total):")
    for k in sorted(tot):
        print(f"  {k:26} {agree[k]}/{tot[k]}")
    print(f"  {'ALL':26} {sum(agree.values())}/{sum(tot.values())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    rows = build(a.seed)
    validate(rows)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(rows)} rows (seed={a.seed})")
    for key in ("category", "source", "regulator", "expected_verdict", "language"):
        c = collections.Counter(r[key] for r in rows)
        print(f"by {key}: " + ", ".join(f"{k}={c[k]}" for k in sorted(c)))
    policy_agreement(rows)


if __name__ == "__main__":
    main()
