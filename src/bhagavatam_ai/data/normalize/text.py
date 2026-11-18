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
# C0 controls only. C1 (U+0080-U+009F) is left in place: it signals mis-decoded UTF-8, which
# validation must catch (CONTROL_CHARS), so it is never silently replaced here.
_CONTROL: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
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


# A lead byte of a multi-byte UTF-8 sequence decoded as cp1252/latin-1 ("á", "Å", "Ä"...) followed by
# continuation bytes (U+0080-U+00BF or their cp1252 renderings).
_SUSPECT: Final = re.compile(
    "[\u00c0-\u00ff][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc"
    "\u2013\u2014\u2018-\u201e\u2020-\u2022\u2026\u2030\u2039\u203a\u20ac\u2122]+"
)


def has_mojibake(text: str) -> bool:
    """True when the text contains UTF-8 that was decoded as cp1252 or latin-1 (e.g. "Ká¹›á¹£á¹‡a")."""
    if "\ufffd" in text or MOJIBAKE.search(text):
        return True
    for match in _SUSPECT.finditer(text):
        segment = match.group(0)
        for encoding in ("cp1252", "latin-1"):
            try:
                if segment.encode(encoding).decode("utf-8") != segment:
                    return True
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return False
