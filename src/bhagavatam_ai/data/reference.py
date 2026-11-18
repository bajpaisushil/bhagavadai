"""Canonical Srimad-Bhagavatam verse references (``SB canto.chapter.verse``).

The 12 cantos contain 335 chapters (90 in Canto 10). Verse counts per chapter differ slightly
between editions, so verse-level bounds are validated against an edition's own structure table,
not hard-coded here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

CHAPTERS_PER_CANTO: Final[dict[int, int]] = {
    1: 19,
    2: 10,
    3: 33,
    4: 31,
    5: 26,
    6: 19,
    7: 15,
    8: 24,
    9: 24,
    10: 90,
    11: 31,
    12: 13,
}
TOTAL_CHAPTERS: Final = sum(CHAPTERS_PER_CANTO.values())  # 335

_REF_RE: Final = re.compile(
    r"^\s*(?:(?:SB|Ś?B|Bh[āa]g(?:avatam)?|BhP|Śrīmad-Bhāgavatam)\.?\s*)?"
    r"(?P<c>\d{1,2})\s*[.:]\s*(?P<ch>\d{1,2})\s*[.:]\s*(?P<v>\d{1,3})"
    r"(?:\s*[-–]\s*(?P<end>\d{1,3}))?\s*$",
    re.IGNORECASE,
)
# In free text, require an explicit "SB"-style prefix to avoid matching dates or section numbers.
_FIND_RE: Final = re.compile(
    r"\b(?:SB|Bhāg\.|Bhag\.|BhP|Bhāgavatam|Bhagavatam)\s*(?P<c>\d{1,2})\.(?P<ch>\d{1,2})\.(?P<v>\d{1,3})"
    r"(?:\s*[-–]\s*(?P<end>\d{1,3}))?",
)


class InvalidReferenceError(ValueError):
    """A verse reference is malformed or outside the 12-canto / 335-chapter structure."""


@dataclass(frozen=True, order=True)
class ChapterRef:
    canto: int
    chapter: int

    def __post_init__(self) -> None:
        if self.canto not in CHAPTERS_PER_CANTO:
            raise InvalidReferenceError(f"canto {self.canto} is not in 1..12")
        if not 1 <= self.chapter <= CHAPTERS_PER_CANTO[self.canto]:
            raise InvalidReferenceError(
                f"canto {self.canto} has chapters 1..{CHAPTERS_PER_CANTO[self.canto]}, got {self.chapter}"
            )

    def __str__(self) -> str:
        return f"{self.canto}.{self.chapter}"

    @classmethod
    def parse(cls, text: str) -> ChapterRef:
        match = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\s*", text)
        if not match:
            raise InvalidReferenceError(f"not a chapter reference: {text!r}")
        return cls(int(match[1]), int(match[2]))


@dataclass(frozen=True, order=True)
class VerseRef:
    canto: int
    chapter: int
    verse: int

    def __post_init__(self) -> None:
        ChapterRef(self.canto, self.chapter)  # validates canto/chapter
        if self.verse < 1:
            raise InvalidReferenceError(f"verse numbers start at 1, got {self.verse}")

    @property
    def chapter_ref(self) -> ChapterRef:
        return ChapterRef(self.canto, self.chapter)

    def __str__(self) -> str:
        return f"SB {self.canto}.{self.chapter}.{self.verse}"

    @property
    def key(self) -> str:
        return f"{self.canto}.{self.chapter}.{self.verse}"


@dataclass(frozen=True)
class VerseSpan:
    """A reference to one verse or a contiguous range within a chapter (``SB 1.1.17-18``)."""

    start: VerseRef
    end_verse: int

    def __post_init__(self) -> None:
        if self.end_verse < self.start.verse:
            raise InvalidReferenceError(f"range end {self.end_verse} precedes start {self.start.verse}")

    @property
    def verses(self) -> list[VerseRef]:
        s = self.start
        return [VerseRef(s.canto, s.chapter, v) for v in range(s.verse, self.end_verse + 1)]

    def __str__(self) -> str:
        return str(self.start) if self.end_verse == self.start.verse else f"{self.start}-{self.end_verse}"


def parse_reference(text: str) -> VerseSpan:
    """Parse ``"SB 1.2.6"``, ``"1.2.6"``, ``"Bhāg. 10.14.58"`` or ranges like ``"SB 1.1.17-18"``."""
    match = _REF_RE.match(text)
    if not match:
        raise InvalidReferenceError(f"not a verse reference: {text!r}")
    start = VerseRef(int(match["c"]), int(match["ch"]), int(match["v"]))
    return VerseSpan(start, int(match["end"]) if match["end"] else start.verse)


def find_references(text: str) -> list[tuple[str, VerseSpan | None]]:
    """Find ``SB x.y.z``-style citations in free text.

    Returns ``(matched_text, span)`` pairs; ``span`` is ``None`` when the citation is structurally
    impossible (e.g. ``SB 13.1.1``) - useful for detecting hallucinated references.
    """
    found: list[tuple[str, VerseSpan | None]] = []
    for match in _FIND_RE.finditer(text):
        try:
            start = VerseRef(int(match["c"]), int(match["ch"]), int(match["v"]))
            span: VerseSpan | None = VerseSpan(start, int(match["end"]) if match["end"] else start.verse)
        except InvalidReferenceError:
            span = None
        found.append((match.group(0), span))
    return found


def all_chapters() -> list[ChapterRef]:
    return [ChapterRef(c, ch) for c, n in CHAPTERS_PER_CANTO.items() for ch in range(1, n + 1)]
