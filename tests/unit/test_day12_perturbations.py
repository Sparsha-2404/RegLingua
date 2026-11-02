import math

from core.verification_policy import GLOBAL_BANNED_PHRASES, REGULATOR_BANNED_PHRASES, find_banned_phrase
from services.evaluation import perturbations as P

EN = "We guarantee 20% returns on your investment safely"
HI = "हम आपके निवेश पर रिटर्न की गारंटी देते हैं।"
MIX = "हम guaranteed returns देते हैं।"


def test_seed_is_stable_across_processes_and_key_sensitive():
    assert P.derive_seed(42, "EVALQ0001", "typo", 0) == 7986341498846978579
    assert P.derive_seed(42, "a") != P.derive_seed(42, "b") != P.derive_seed(43, "b")
    assert P.rng_for(42, "k").random() == P.rng_for(42, "k").random()


def test_typos_deterministic_ascii_only_and_inner_letters_kept():
    a, b = P.add_typos(EN, P.rng_for(42, "t")), P.add_typos(EN, P.rng_for(42, "t"))
    assert a == b != EN
    assert len(a.split()) == len(EN.split())
    for w0, w1 in zip(EN.split(), a.split()):  # at most one edit per word; first/last letters kept
        assert abs(len(w1) - len(w0)) <= 1 and w1[0] == w0[0] and w1[-1] == w0[-1]
    assert P.add_typos(HI, P.rng_for(42, "t")) == HI  # no ASCII word -> untouched, never corrupts Devanagari
    out = P.add_typos(MIX, P.rng_for(42, "t"))
    assert out != MIX
    assert "".join(c for c in out if ord(c) > 0x900) == "".join(c for c in MIX if ord(c) > 0x900)


def test_typo_always_changes_eligible_word_even_with_double_letters():
    for seed in range(30):
        assert P.add_typos("all will pool", P.rng_for(seed, "d")) != "all will pool"


def test_case_modes():
    r = P.rng_for(42, "c")
    assert P.change_case(EN, r, "upper") == EN.upper()
    assert P.change_case("we don't guarantee", r, "title") == "We Don't Guarantee"
    assert P.change_case(HI, r, "upper") == HI and P.change_case(HI, r, "random") == HI
    x = P.change_case(EN, P.rng_for(42, "c"), "random")
    assert x.lower() == EN.lower() and x == P.change_case(EN, P.rng_for(42, "c"), "random")
    try:
        P.change_case(EN, r, "nope")
        assert False
    except ValueError:
        pass


def test_whitespace_keeps_tokens_and_changes_text():
    for seed in range(20):
        out = P.extra_whitespace(EN, P.rng_for(seed, "w"))
        assert out != EN and out.split() == EN.split()
    assert P.extra_whitespace("one", P.rng_for(1, "w")) != "one"


def test_punctuation_and_emoji_keep_original_characters():
    for seed in range(20):
        p = P.extra_punctuation(EN, P.rng_for(seed, "p"))
        assert p != EN and EN in p
        e = P.add_emoji(EN, P.rng_for(seed, "e"))
        assert e != EN and "".join(c for c in e if c not in "".join(P.EMOJIS)).replace("  ", " ").split() == EN.split()


def test_append_and_inject():
    assert P.append_phrase("Hello. ", "X") == "Hello. X" and P.append_phrase("   ", "X") == "X"
    assert P.inject("T", 1, "suffix").endswith(P.INJECTIONS[1]) and P.inject("T", 2, "prefix").startswith(P.INJECTIONS[2])
    assert P.inject("T", 4) == P.inject("T", 1)  # variant wraps
    assert set(P.GUARANTEE_PHRASES) == {"en", "hi", "hi-en"}


def test_controls_mirror_injections_and_are_length_matched():
    assert P.control("T", 0, "suffix") == "T " + P.NEUTRAL_CONTROLS[0] and P.control("T", 5, "prefix").startswith(P.NEUTRAL_CONTROLS[2])
    for inj, ctl in zip(P.INJECTIONS, P.NEUTRAL_CONTROLS):
        assert 0.7 <= len(ctl) / len(inj) <= 1.5
        assert (any("\u0900" <= c <= "\u097f" for c in inj)) == (any("\u0900" <= c <= "\u097f" for c in ctl))
    assert not any(w in " ".join(P.NEUTRAL_CONTROLS).lower() for w in ("ignore", "instruction", "label", "compliant"))


def test_replace_banned_longest_first_case_insensitive_and_counts():
    out, n = P.replace_banned("Guaranteed returns, we GUARANTEE it. Zero risk!")
    assert (out, n) == ("estimated returns, we estimate it. market risk!", 3)
    assert P.replace_banned("nothing risky here") == ("nothing risky here", 0)


def test_neutral_replacements_cover_policy_and_are_clean():
    phrases = set(GLOBAL_BANNED_PHRASES) | {p for v in REGULATOR_BANNED_PHRASES.values() for p in v}
    assert phrases <= set(P.NEUTRAL_REPLACEMENTS), phrases - set(P.NEUTRAL_REPLACEMENTS)
    for new in P.NEUTRAL_REPLACEMENTS.values():
        for reg in (None, *REGULATOR_BANNED_PHRASES):
            assert find_banned_phrase(new, reg) is None, new


def test_rate_ci():
    assert P.rate_ci([True] * 5, list("abcde")) == (1.0, 1.0, 1.0)
    assert P.rate_ci([False] * 5, list("abcde")) == (0.0, 0.0, 0.0)
    r = P.rate_ci([True, False, False, False], ["a", "a", "b", "b"])
    assert r == (0.25, 0.0, 0.5) and r == P.rate_ci([True, False, False, False], ["a", "a", "b", "b"])
    assert all(math.isnan(x) for x in P.rate_ci([], []))
    try:
        P.rate_ci([True], ["a", "b"])
        assert False
    except ValueError:
        pass
