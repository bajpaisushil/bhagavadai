"""IAST normalisation and deterministic IAST <-> Devanagari transliteration.

Conventions (applied by :func:`normalize_iast`):

* Unicode NFC (precomposed) - NFD text tokenises differently, so this matters for training;
* anusvara as ``ṃ`` (dot below, standard IAST), not ``ṁ`` (dot above, common in Gaudiya editions);
* ISO 15919 vocalic ``r̥ r̥̄ l̥ l̥̄`` mapped to IAST ``ṛ ṝ ḷ ḹ``;
* typographic apostrophes for avagraha mapped to ``'``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Final

_IAST_FIXES: Final = (
    ("r̥̄", "ṝ"),
    ("R̥̄", "Ṝ"),
    ("l̥̄", "ḹ"),
    ("L̥̄", "Ḹ"),
    ("r̥", "ṛ"),
    ("R̥", "Ṛ"),
    ("l̥", "ḷ"),
    ("L̥", "Ḷ"),
    ("ṁ", "ṃ"),
    ("Ṁ", "Ṃ"),
    ("’", "'"),
    ("‘", "'"),
    ("ʼ", "'"),
)

IAST_LETTERS: Final = frozenset("aāiīuūṛṝḷḹeokgṅcjñṭḍṇtdnpbmyrlvśṣshṃḥ")


def normalize_iast(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = unicodedata.normalize("NFC", text)
    for src, dst in _IAST_FIXES:
        text = unicodedata.normalize("NFC", text.replace(unicodedata.normalize("NFC", src), dst))
    return text


def unexpected_iast_chars(text: str, allowed_extra: str = " \n'-/|.,;:!?()[]0123456789&$%*") -> Counter[str]:
    """Characters in ``text`` that are neither IAST letters nor expected punctuation."""
    counts: Counter[str] = Counter()
    for char in text:
        if char.lower() not in IAST_LETTERS and char not in allowed_extra:
            counts[char] += 1
    return counts


# --------------------------------------------------------------------------- transliteration

_VOWELS: Final = {  # iast: (independent, matra)
    "a": ("अ", ""),
    "ā": ("आ", "ा"),
    "i": ("इ", "ि"),
    "ī": ("ई", "ी"),
    "u": ("उ", "ु"),
    "ū": ("ऊ", "ू"),
    "ṛ": ("ऋ", "ृ"),
    "ṝ": ("ॠ", "ॄ"),
    "ḷ": ("ऌ", "ॢ"),
    "ḹ": ("ॡ", "ॣ"),
    "e": ("ए", "े"),
    "ai": ("ऐ", "ै"),
    "o": ("ओ", "ो"),
    "au": ("औ", "ौ"),
}
_CONSONANTS: Final = {
    "k": "क",
    "kh": "ख",
    "g": "ग",
    "gh": "घ",
    "ṅ": "ङ",
    "c": "च",
    "ch": "छ",
    "j": "ज",
    "jh": "झ",
    "ñ": "ञ",
    "ṭ": "ट",
    "ṭh": "ठ",
    "ḍ": "ड",
    "ḍh": "ढ",
    "ṇ": "ण",
    "t": "त",
    "th": "थ",
    "d": "द",
    "dh": "ध",
    "n": "न",
    "p": "प",
    "ph": "फ",
    "b": "ब",
    "bh": "भ",
    "m": "म",
    "y": "य",
    "r": "र",
    "l": "ल",
    "v": "व",
    "ś": "श",
    "ṣ": "ष",
    "s": "स",
    "h": "ह",
    "ḻ": "ळ",
}
_MARKS: Final = {"ṃ": "ं", "ḥ": "ः", "m̐": "ँ", "'": "ऽ"}
_PUNCT: Final = {"//": "॥", "/": "।", "||": "॥", "|": "।"}
_VIRAMA: Final = "्"
_DIGITS: Final = dict(zip("0123456789", "०१२३४५६७८९", strict=True))

_TOKENS: Final = sorted([*_VOWELS, *_CONSONANTS, *_MARKS, *_PUNCT], key=len, reverse=True)
_TOKEN_RE: Final = re.compile("|".join(re.escape(t) for t in _TOKENS) + r"|.", re.DOTALL)


def iast_to_devanagari(text: str) -> str:
    """Transliterate IAST to Devanagari. Hyphens (compound separators) are removed first."""
    source = normalize_iast(text).lower().replace("-", "")
    out: list[str] = []
    pending_consonant = False
    for token in _TOKEN_RE.findall(source):
        if token in _CONSONANTS:
            if pending_consonant:
                out.append(_VIRAMA)
            out.append(_CONSONANTS[token])
            pending_consonant = True
        elif token in _VOWELS:
            independent, matra = _VOWELS[token]
            out.append(matra if pending_consonant else independent)
            pending_consonant = False
        else:
            if pending_consonant:
                out.append(_VIRAMA)
                pending_consonant = False
            if token in _MARKS:
                out.append(_MARKS[token])
            elif token in _PUNCT:
                out.append(_PUNCT[token])
            else:
                out.append(_DIGITS.get(token, token))
    if pending_consonant:
        out.append(_VIRAMA)
    return "".join(out)


_D_VOWELS: Final = {ind: iast for iast, (ind, _m) in _VOWELS.items()}
_D_MATRAS: Final = {m: iast for iast, (_i, m) in _VOWELS.items() if m}
_D_CONSONANTS: Final = {dev: iast for iast, dev in _CONSONANTS.items()}
_D_OTHER: Final = {"ं": "ṃ", "ः": "ḥ", "ँ": "m̐", "ऽ": "'", "ॐ": "oṃ", "।": "/", "॥": "//", "़": ""}
_D_DIGITS: Final = {d: a for a, d in _DIGITS.items()}


def devanagari_to_iast(text: str) -> str:
    """Transliterate Devanagari to IAST (NFC)."""
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in _D_CONSONANTS:
            out.append(_D_CONSONANTS[char])
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == "़":  # nukta: drop and look one further
                i += 1
                nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _D_MATRAS:
                out.append(_D_MATRAS[nxt])
                i += 2
                continue
            out.append("a")
        elif char in _D_VOWELS:
            out.append(_D_VOWELS[char])
        elif char in _D_OTHER:
            out.append(_D_OTHER[char])
        elif char in _D_DIGITS:
            out.append(_D_DIGITS[char])
        elif char in {"‌", "‍"}:
            pass
        else:
            out.append(char)
        i += 1
    return unicodedata.normalize("NFC", "".join(out))
