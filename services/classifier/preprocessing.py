"""
DAY 2 DELIVERABLE (M2): Preprocessing.

"Preprocessing + train/test split." (Day 2 plan)

Text cleaning for English and Hindi/code-mixed fintech product copy.
Deliberately conservative: regulatory phrases can hinge on exact wording
(e.g. "guaranteed returns" vs "expected returns"), so this does NOT stem,
remove stopwords, or aggressively normalize punctuation the way a generic
NLP pipeline would — doing so risks destroying the exact phrase a downstream
classifier or human reviewer needs to see.
"""
from __future__ import annotations

import re
import unicodedata

# Unicode block covering Devanagari script (used by Hindi).
_DEVANAGARI_RANGE = (0x0900, 0x097F)

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_PUNCT_RE = re.compile(r"([!?.])\1{1,}")


def contains_devanagari(text: str) -> bool:
    """True if any character in `text` falls in the Devanagari Unicode block."""
    return any(_DEVANAGARI_RANGE[0] <= ord(ch) <= _DEVANAGARI_RANGE[1] for ch in text)


def detect_language(text: str) -> str:
    """
    Cheap rule-based language tag: 'hi' if any Devanagari script is present
    (covers both pure Hindi and Hindi-English code-mixed text), else 'en'.

    This matches the architecture doc's Version 1 layer for language
    detection ("Lightweight library/rules", Section 8) — Day 10 is where
    this gets upgraded to a proper multilingual classifier if code-mixing
    accuracy turns out to need it.
    """
    return "hi" if contains_devanagari(text) else "en"


def clean_text(text: str, *, lowercase_english: bool = True) -> str:
    """
    Normalize raw product copy while preserving regulatory-relevant wording.

    Steps:
      1. Unicode-normalize (NFC) so visually identical characters compare equal.
      2. Strip URLs — never legally meaningful for this classification task.
      3. Collapse repeated punctuation ("!!!" -> "!") without deleting it —
         marketing tone/emphasis can matter to a human reviewer.
      4. Collapse whitespace/newlines.
      5. Lowercase ONLY when the text has no Devanagari, since Hindi has no
         case and lowercasing must never be applied to it.

    Does NOT: remove stopwords, strip all punctuation, or transliterate —
    those would risk destroying exact regulatory trigger phrases.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text)}")

    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub("", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if lowercase_english and not contains_devanagari(text):
        text = text.lower()

    return text


def normalize_for_classifier(text: str) -> dict:
    """
    Combined Day-2 preprocessing step used by the Intake stage: returns the
    cleaned text plus its detected language, matching the CaseState fields
    `normalized_text` and `language` from Section 5.
    """
    language = detect_language(text)
    normalized = clean_text(text, lowercase_english=(language == "en"))
    return {"normalized_text": normalized, "language": language}


if __name__ == "__main__":
    samples = [
        "  We GUARANTEE 20%!!! returns   on your investment.  See http://x.co ",
        "हम आपको 20% गारंटीशुदा रिटर्न देते हैं।",
        "Yeh product aapko guaranteed returns deta hai — 100% sach!",
    ]
    for s in samples:
        print(normalize_for_classifier(s))
