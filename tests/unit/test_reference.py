from __future__ import annotations

import pytest

from bhagavatam_ai.data.reference import (
    CHAPTERS_PER_CANTO,
    TOTAL_CHAPTERS,
    ChapterRef,
    InvalidReferenceError,
    VerseRef,
    all_chapters,
    find_references,
    parse_reference,
)


def test_structure_constants() -> None:
    assert TOTAL_CHAPTERS == 335
    assert CHAPTERS_PER_CANTO[10] == 90
    assert len(all_chapters()) == 335


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SB 1.2.6", "SB 1.2.6"),
        ("1.2.6", "SB 1.2.6"),
        ("  SB 10.14.58 ", "SB 10.14.58"),
        ("Bhāg. 3.25.21", "SB 3.25.21"),
        ("BhP 12.13.23", "SB 12.13.23"),
        ("SB 1.1.17-18", "SB 1.1.17-18"),
        ("SB 1.1.17–19", "SB 1.1.17-19"),
    ],
)
def test_parse_reference(text: str, expected: str) -> None:
    assert str(parse_reference(text)) == expected


def test_span_verses() -> None:
    span = parse_reference("SB 1.1.17-19")
    assert [v.verse for v in span.verses] == [17, 18, 19]


@pytest.mark.parametrize(
    "bad", ["SB 13.1.1", "SB 1.20.1", "SB 1.1.0", "SB 0.1.1", "chapter one", "SB 1.1", "SB 1.1.5-3"]
)
def test_invalid_references(bad: str) -> None:
    with pytest.raises(InvalidReferenceError):
        parse_reference(bad)


def test_ordering_and_keys() -> None:
    assert VerseRef(1, 2, 3) < VerseRef(1, 10, 1) < VerseRef(2, 1, 1)
    assert VerseRef(3, 25, 21).key == "3.25.21"
    assert str(VerseRef(3, 25, 21).chapter_ref) == "3.25"
    assert ChapterRef.parse("10.90") == ChapterRef(10, 90)


def test_find_references_flags_impossible_citations() -> None:
    text = "As stated in SB 1.2.6 and Bhāgavatam 10.14.58, but not in SB 13.4.2. Dated 2026.11.18."
    found = find_references(text)
    assert [m for m, _ in found] == ["SB 1.2.6", "Bhāgavatam 10.14.58", "SB 13.4.2"]
    assert found[0][1] is not None and found[2][1] is None  # canto 13 does not exist
