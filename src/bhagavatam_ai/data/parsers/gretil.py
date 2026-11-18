"""Parser for GRETIL's TEI edition of the Bhāgavata Purāṇa (``sa_bhAgavatapurANa.xml``).

Structure (verified against the 2020-09-10 file, sha256 5b4271b0...):

* the verse ID comes *after* the verse: ``... āsata // BhP_01.01.004 //``; all lines since the
  previous ID belong to it. ``_P`` marks split verses, ``*`` long metres, a trailing ``_*`` prose;
* speakers are separate lines ``BhP_01.01.006/0 ṛṣaya ūcuḥ`` preceding the verse they introduce;
* long metres mark pāda breaks with ``$`` / ``%`` and the half-verse with ``&``.

Known defects of this file, repaired here and flagged on the affected records:

* 584 long-metre verses repeat their first line (2020 TEI conversion bug) -> consecutive duplicate
  lines are dropped (``duplicate_line_removed``);
* 258 lines are double-escaped (``&amp;amp;``) -> unescaped twice;
* 30 records carry a wrong chapter label (3.31.22-48 labelled 3.32.x; single records in 7.10, 8.7,
  9.11) -> relabelled from the ID sequence (``label_repaired``): a chapter can only start at verse 1;
* one bare speaker ID (5.15.1) has the speaker text on the next line -> joined;
* stray ``#`` and ``\\`` characters -> removed (``stray_char_removed``).
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from bhagavatam_ai.data.parsers.base import ParsedDocument, ParsedVerse, ParserError, register

PARSER_NAME: Final = "gretil_tei_bhp"
PARSER_VERSION: Final = "1.3.0"

_ID: Final = r"BhP_(?P<c>\d{2})\.(?P<ch>\d{2})\.(?P<v>\d{3})(?:_(?P<part>\d))?"
_VERSE_END: Final = re.compile(
    rf"^(?P<txt>.*?)[\s/]*(?:/\d/)?[\s/]*{_ID}(?P<star>\*)?\s*/{{1,2}}(?:\s*/{{1,2}})?\s*(?P<prose>_\*)?\s*$"
)
_SPEAKER: Final = re.compile(rf"^{_ID}/0(?:\s+(?P<spk>.+?))?\s*$")
_TAG_BREAK: Final = re.compile(r"</?(?:l|lg|p|div|head|ab)\b[^>]*>")
_ANY_TAG: Final = re.compile(r"<[^>]+>")
_SENTINEL: Final = "\x00"
_STRAY: Final = re.compile(r"[#\\]")
_LICENCE: Final = re.compile(r'<licence\s+target="([^"]+)"')


def _body_lines(raw: str) -> list[tuple[int, str]]:
    """Extract (line_number, text) pairs from the TEI ``<body>``, one entry per logical line."""
    start, end = raw.find("<body"), raw.find("</body>")
    if start < 0 or end < 0:
        raise ParserError("no <body> element found")
    first_line = raw.count("\n", 0, start) + 1
    lines: list[tuple[int, str]] = []
    # Split on the file's real newlines first so line numbers stay true to the raw file; element
    # boundaries inside a physical line become a sentinel that separates logical lines.
    for offset, physical in enumerate(raw[start:end].split("\n")):
        text = _TAG_BREAK.sub(_SENTINEL, physical)
        text = html.unescape(html.unescape(_ANY_TAG.sub("", text)))
        for logical in text.split(_SENTINEL):
            stripped = logical.strip()
            if stripped:
                lines.append((first_line + offset, stripped))
    # A bare speaker ID with the speaker on the following line (5.15.1): join them.
    joined: list[tuple[int, str]] = []
    for number, line in lines:
        if joined and re.fullmatch(rf"{_ID}/0", joined[-1][1]):
            joined[-1] = (joined[-1][0], f"{joined[-1][1]} {line}")
        else:
            joined.append((number, line))
    return joined


def _clean_line(line: str, flags: set[str]) -> list[str]:
    if _STRAY.search(line):
        line = _STRAY.sub("", line)
        flags.add("stray_char_removed")
    if any(mark in line for mark in "$%&"):
        flags.add("metre_markup_converted")
        line = re.sub(r"\s*&\s*", " /", line)
        parts = re.split(r"\s*[$%]\s*", line)
        return [p.strip() for p in parts if p.strip()]
    return [line.strip()]


def _repair_labels(verses: list[ParsedVerse]) -> int:
    """Relabel records whose chapter label is wrong, using the ID sequence.

    A record is relabelled to the previous record's chapter when its label moves to another chapter of
    the same canto while its verse number simply continues the previous numbering (n, n+1 or n+2), and
    either (a) the next record is back in the previous chapter (an isolated mislabel such as 7.9.47
    between 7.10.46 and 7.10.48), or (b) the labelled chapter's verse 1 appears later in the file (a
    run such as 3.31.22-48 labelled 3.32.x before the real 3.32.1). A genuine chapter start that merely
    lacks verse 1 satisfies neither condition and is left alone.
    """
    first_verse_one: dict[tuple[int, int], int] = {}
    for index, verse in enumerate(verses):
        if verse.verse == 1:
            first_verse_one.setdefault((verse.canto, verse.chapter), index)
    repaired = 0
    for i in range(1, len(verses)):
        prev, cur = verses[i - 1], verses[i]
        if cur.canto != prev.canto or cur.chapter == prev.chapter or cur.verse <= 1:
            continue
        if cur.verse not in {prev.verse, prev.verse + 1, prev.verse + 2}:
            continue
        nxt = verses[i + 1] if i + 1 < len(verses) else None
        sandwiched = nxt is not None and (nxt.canto, nxt.chapter) == (prev.canto, prev.chapter)
        starts_later = first_verse_one.get((cur.canto, cur.chapter), -1) > i
        if sandwiched or starts_later:
            cur.flags.append(f"label_repaired:{cur.canto}.{cur.chapter}.{cur.verse}")
            cur.chapter = prev.chapter
            repaired += 1
    return repaired


def _assign_variants(verses: list[ParsedVerse]) -> int:
    """Number repeated (canto, chapter, verse, part) keys so record ids stay unique."""
    seen: Counter[tuple[int, int, int, int | None]] = Counter()
    duplicates = 0
    for verse in verses:
        key = (verse.canto, verse.chapter, verse.verse, verse.part)
        if seen[key]:
            verse.variant = seen[key]
            verse.flags.append("duplicate_ref")
            duplicates += 1
        seen[key] += 1
    return duplicates


def _mark_uncertain_before(verses: list[ParsedVerse], verse: ParsedVerse, label: str) -> None:
    """A speaker line labelled for an earlier verse of the same chapter: the carried-forward speakers
    of the verses it should have introduced are probably wrong."""
    canto, chapter, labelled = (int(x) for x in label.split("."))
    if (canto, chapter) != (verse.canto, verse.chapter) or labelled >= verse.verse:
        return
    for other in verses:
        if (
            (other.canto, other.chapter) == (canto, chapter)
            and labelled <= other.verse < verse.verse
            and "speaker_carried_forward" in other.flags
            and "speaker_uncertain" not in other.flags
        ):
            other.flags.append("speaker_uncertain")


def _carry_speakers(verses: list[ParsedVerse]) -> None:
    """Propagate the most recent explicit speaker forward within each chapter.

    Carried-forward attributions are inferences (flagged ``speaker_carried_forward``): dialogue that
    lacks explicit speaker lines makes them wrong at times (e.g. 10.62.15-17). Nothing is carried forward
    from a speaker line whose verse label disagrees with its position (``speaker_label_mismatch``).
    """
    current: str | None = None
    chapter: tuple[int, int] | None = None
    for verse in verses:
        if (verse.canto, verse.chapter) != chapter:
            chapter, current = (verse.canto, verse.chapter), None
        if verse.speaker_explicit:
            label = next(
                (f.split(":", 1)[1] for f in verse.flags if f.startswith("speaker_label_mismatch")), None
            )
            if label is not None:
                _mark_uncertain_before(verses, verse, label)
            current = None if label is not None else verse.speaker_raw
        elif current is not None:
            verse.speaker_raw = current
            verse.flags.append("speaker_carried_forward")


@dataclass
class _Reader:
    """Line-by-line state machine: verse text accumulates until its trailing verse ID."""

    verses: list[ParsedVerse] = field(default_factory=list)
    buffer: list[str] = field(default_factory=list)
    buffer_start: int | None = None
    flags: set[str] = field(default_factory=set)
    pending_speaker: tuple[str, tuple[int, int, int], int] | None = None
    duplicate_lines: int = 0
    speaker_mismatches: list[str] = field(default_factory=list)

    def feed(self, number: int, line: str) -> None:
        if speaker := _SPEAKER.match(line):
            if self.buffer:
                raise ParserError(f"line {number}: speaker line inside an unterminated verse")
            ident = (int(speaker["c"]), int(speaker["ch"]), int(speaker["v"]))
            self.pending_speaker = ((speaker["spk"] or "").strip(), ident, number)
            return
        if self.buffer_start is None:
            self.buffer_start = number
        end = _VERSE_END.match(line)
        if end is None:
            self.buffer.append(line)
            return
        if end["txt"].strip():
            self.buffer.append(end["txt"].strip())
        self._finish(number, end)

    def _finish(self, number: int, end: re.Match[str]) -> None:
        # De-duplicate raw lines *before* splitting pādas: the conversion bug repeats whole lines.
        raw_lines = [x for i, x in enumerate(self.buffer) if i == 0 or x != self.buffer[i - 1]]
        if len(raw_lines) < len(self.buffer):
            self.flags.add("duplicate_line_removed")
            self.duplicate_lines += len(self.buffer) - len(raw_lines)
        lines = [part for raw_line in raw_lines for part in _clean_line(raw_line, self.flags)]
        if not lines:
            raise ParserError(f"line {number}: verse ID without verse text")
        verse = ParsedVerse(
            canto=int(end["c"]),
            chapter=int(end["ch"]),
            verse=int(end["v"]),
            part=int(end["part"]) if end["part"] else None,
            text="\n".join(lines),
            locator=f"lines {self.buffer_start}-{number}",
            flags=sorted(self.flags) + ["long_metre"] * bool(end["star"]) + ["prose"] * bool(end["prose"]),
        )
        self._attach_speaker(verse)
        self.verses.append(verse)
        self.buffer, self.buffer_start, self.flags = [], None, set()

    def _attach_speaker(self, verse: ParsedVerse) -> None:
        if self.pending_speaker is None:
            return
        name, ident, line = self.pending_speaker
        if ident != (verse.canto, verse.chapter, verse.verse):
            self.speaker_mismatches.append(
                f"speaker at line {line} labelled {ident} precedes {verse.locator}"
            )
            verse.flags.append("speaker_label_mismatch:{}.{}.{}".format(*ident))
        if name:
            verse.speaker_raw, verse.speaker_explicit = name, True
        self.pending_speaker = None


@register(PARSER_NAME)
def parse_gretil_tei(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    licence = _LICENCE.search(raw)
    reader = _Reader()
    for number, line in _body_lines(raw):
        reader.feed(number, line)
    if reader.buffer:
        raise ParserError(f"{len(reader.buffer)} trailing lines without a verse ID")
    verses = reader.verses
    if not verses:
        raise ParserError("no verses found")
    repaired = _repair_labels(verses)
    duplicates = _assign_variants(verses)
    _carry_speakers(verses)
    per_canto = Counter(v.canto for v in verses)
    return ParsedDocument(
        parser=PARSER_NAME,
        parser_version=PARSER_VERSION,
        verses=verses,
        header={"licence_url": licence.group(1) if licence else None},
        stats={
            "records": len(verses),
            "chapters": len({(v.canto, v.chapter) for v in verses}),
            "records_per_canto": dict(sorted(per_canto.items())),
            "duplicate_lines_removed": reader.duplicate_lines,
            "labels_repaired": repaired,
            "duplicate_refs": duplicates,
            "explicit_speakers": sum(v.speaker_explicit for v in verses),
            "speaker_label_mismatches": reader.speaker_mismatches,
        },
    )
