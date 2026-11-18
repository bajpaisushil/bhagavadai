"""Clean-up for OCR'd scans of printed translations (running heads, page numbers, hyphenation).

All functions are conservative: when a decision is ambiguous the text is kept and the case is
counted, so that it can be surfaced for human review rather than silently "fixed".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

_PAGE_NUMBER: Final = re.compile(r"^\s*[\[(]?(?:\d{1,4}|[ivxlcdm]{1,7})[\])]?\s*$", re.IGNORECASE)
_WORD: Final = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_TRAILING_HYPHEN: Final = re.compile(r"([^\W\d_]+)-$", re.UNICODE)
_LEADING_WORD: Final = re.compile(r"^([^\W\d_]+)(.*)$", re.UNICODE | re.DOTALL)
_DIACRITICS: Final = frozenset("āīūṛṝḷḹṃṁḥṅñṭḍṇśṣĀĪŪṚṜḶḸṂṀḤṄÑṬḌṆŚṢ")


def is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER.match(line))


def _furniture_key(line: str) -> str:
    """Normalise a header/footer line so 'CANTO I 23' and 'CANTO I 24' compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def _edges(page: Sequence[str], edge_lines: int) -> list[int]:
    """Indices of the first/last ``edge_lines`` non-empty lines; empty for pages too short to have a body."""
    content = [i for i, line in enumerate(page) if line.strip()]
    if len(content) < 2 * edge_lines + 1:
        return []
    return content[:edge_lines] + content[-edge_lines:]


def find_running_heads(
    pages: Sequence[Sequence[str]], *, edge_lines: int = 1, min_fraction: float = 0.2, min_pages: int = 5
) -> set[str]:
    """Keys of lines that recur at the top or bottom of many pages (running heads/footers).

    Only the outermost ``edge_lines`` lines of a page are considered, so a phrase that merely recurs
    in the text (e.g. a speaker line such as "Sūta said:") is not mistaken for page furniture.
    """
    if len(pages) < min_pages:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        keys = {_furniture_key(page[i]) for i in _edges(page, edge_lines)}
        counts.update(key for key in keys if key and len(key) <= 80)
    threshold = max(min_pages, int(len(pages) * min_fraction))
    return {key for key, n in counts.items() if n >= threshold and key not in {"#", ""}}


def strip_page_furniture(
    pages: Sequence[Sequence[str]], heads: set[str], *, edge_lines: int = 1
) -> list[list[str]]:
    """Remove page numbers and known running heads from the outermost ``edge_lines`` lines of each page."""
    cleaned: list[list[str]] = []
    for page in pages:
        edge = set(_edges(page, edge_lines))
        cleaned.append(
            [
                line
                for i, line in enumerate(page)
                if not (i in edge and (is_page_number(line) or _furniture_key(line) in heads))
            ]
        )
    return cleaned


def build_vocabulary(texts: Iterable[str]) -> Counter[str]:
    """Lower-cased word frequencies, excluding both halves of words broken across a line end."""
    vocab: Counter[str] = Counter()
    for text in texts:
        previous_broken = False
        for line in text.splitlines():
            words = _WORD.findall(line)
            broken = line.rstrip().endswith("-") and bool(words)
            if broken:
                words = words[:-1]
            if previous_broken and words and line.lstrip()[:1].isalpha():
                words = words[1:]
            vocab.update(w.lower() for w in words)
            previous_broken = broken
    return vocab


@dataclass
class DehyphenationStats:
    joined: int = 0
    kept_hyphen: int = 0
    ambiguous: list[str] = field(default_factory=list)


def dehyphenate(
    lines: Sequence[str], vocabulary: Counter[str], stats: DehyphenationStats | None = None
) -> list[str]:
    """Join words broken across line ends.

    ``"Bhaga-" + "vatam..."`` becomes ``"Bhagavatam..."`` when the joined form is attested elsewhere in
    the corpus; a real compound (``"Śrīmad-" + "Bhāgavatam"``) keeps its hyphen when the hyphenated
    form is attested or either half is capitalised/diacritic-bearing. Unattested cases keep the
    hyphen and are recorded in ``stats.ambiguous`` for review.
    """
    stats = stats if stats is not None else DehyphenationStats()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        match = _TRAILING_HYPHEN.search(line)
        nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
        lead = _LEADING_WORD.match(nxt) if match and nxt else None
        if not match or not lead:
            out.append(lines[i])
            i += 1
            continue
        left, right, rest = match.group(1), lead.group(1), lead.group(2)
        joined, hyphenated = (left + right).lower(), f"{left}-{right}".lower()
        if vocabulary.get(joined, 0) > vocabulary.get(hyphenated, 0):
            merged, stats.joined = left + right, stats.joined + 1
        elif vocabulary.get(hyphenated, 0) > 0 or right[:1].isupper() or (set(left + right) & _DIACRITICS):
            merged, stats.kept_hyphen = f"{left}-{right}", stats.kept_hyphen + 1
        else:
            merged = f"{left}-{right}"
            stats.kept_hyphen += 1
            stats.ambiguous.append(f"{left}-{right}")
        out.append(line[: match.start(1)] + merged + rest)
        i += 2
    return out


def unwrap_paragraphs(lines: Sequence[str]) -> list[str]:
    """Join hard-wrapped lines into paragraphs; blank lines separate paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs
