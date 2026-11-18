"""Unicode and typography normalisation for English and IAST text."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

LIGATURES: Final = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
QUOTES: Final = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"'}
# Invisible characters that never carry meaning in Latin-script text. ZWJ/ZWNJ are kept for
# Devanagari, where they control conjunct rendering.
_INVISIBLE: Final = re.compile("[­\u200b⁠﻿]")
_LATIN_JOINERS: Final = re.compile("[‌‍]")
_CONTROL: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SPACES: Final = re.compile(r"[ \t  -   　]+")
# UTF-8 text decoded as Latin-1/CP1252 ("â€™", "Ã¡"...).
MOJIBAKE: Final = re.compile(r"(?:Ã[\x80-\xbfa-zA-Z]|â€|Â[\x80-\xbf ])")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str, *, keep_newlines: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not keep_newlines:
        return _SPACES.sub(" ", text.replace("\n", " ")).strip()
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_latin_text(text: str, *, straighten_quotes: bool = True) -> str:
    """NFC, ligatures, invisible/control characters, quotes and whitespace for Latin-script text."""
    text = nfc(text)
    text = _INVISIBLE.sub("", text)
    text = _LATIN_JOINERS.sub("", text)
    text = _CONTROL.sub(" ", text)
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    if straighten_quotes:
        for src, dst in QUOTES.items():
            text = text.replace(src, dst)
    return collapse_whitespace(text)


def has_mojibake(text: str) -> bool:
    return MOJIBAKE.search(text) is not None
