from __future__ import annotations

from pathlib import Path

import pytest

from bhagavatam_ai.data.parsers import ParserError, get_parser
from bhagavatam_ai.data.parsers.gretil import parse_gretil_tei

REAL_FILE = Path(__file__).resolve().parents[2] / "data/raw/gretil-bhagavata-purana/sa_bhAgavatapurANa.xml"


@pytest.fixture
def doc(fixtures_dir: Path):  # type: ignore[no-untyped-def]
    return parse_gretil_tei(fixtures_dir / "mini_gretil.xml")


def _by_key(doc):  # type: ignore[no-untyped-def]
    return {(v.canto, v.chapter, v.verse, v.part, v.variant): v for v in doc.verses}


def test_registry() -> None:
    assert get_parser("gretil_tei_bhp") is parse_gretil_tei
    with pytest.raises(ParserError):
        get_parser("nope")


def test_header_and_counts(doc) -> None:  # type: ignore[no-untyped-def]
    assert doc.header["licence_url"] == "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    assert doc.stats["records"] == 10
    assert doc.stats["chapters"] == 2
    assert doc.stats["labels_repaired"] == 1
    assert doc.stats["duplicate_lines_removed"] == 1
    assert doc.stats["duplicate_refs"] == 1
    assert doc.stats["explicit_speakers"] == 2


def test_plain_verse(doc) -> None:  # type: ignore[no-untyped-def]
    v = _by_key(doc)[(1, 1, 1, None, 0)]
    assert v.text == "ka kha ga /\ngha ṅa ca"
    assert v.speaker_raw is None and v.locator.startswith("lines ")


def test_long_metre_duplicate_line_and_double_escape(doc) -> None:  # type: ignore[no-untyped-def]
    v = _by_key(doc)[(1, 1, 2, None, 0)]
    assert v.text == "ta tha\nda dha /\nna pa\npha ba"  # duplicate removed, $ % split, &amp;amp; -> " /"
    assert {"long_metre", "duplicate_line_removed", "metre_markup_converted"} <= set(v.flags)
    assert v.speaker_raw == "śrī-śuka uvāca" and v.speaker_explicit


def test_speaker_carried_forward_within_chapter(doc) -> None:  # type: ignore[no-untyped-def]
    v = _by_key(doc)[(1, 1, 3, None, 0)]
    assert v.speaker_raw == "śrī-śuka uvāca" and "speaker_carried_forward" in v.flags
    first_of_next_chapter = _by_key(doc)[(1, 2, 1, None, 0)]
    assert first_of_next_chapter.speaker_raw is None  # not carried across chapters


def test_prose(doc) -> None:  # type: ignore[no-untyped-def]
    v = _by_key(doc)[(1, 1, 4, None, 0)]
    assert "prose" in v.flags and v.text.startswith("la va")


def test_label_repair(doc) -> None:  # type: ignore[no-untyped-def]
    v = _by_key(doc)[(1, 1, 5, None, 0)]  # labelled 1.2.5 in the file
    assert "label_repaired:1.2.5" in v.flags


def test_bare_speaker_id_joined(doc) -> None:  # type: ignore[no-untyped-def]
    assert _by_key(doc)[(1, 2, 2, None, 0)].speaker_raw == "sūta uvāca"


def test_duplicate_ref_variant_and_parts(doc) -> None:  # type: ignore[no-untyped-def]
    keys = _by_key(doc)
    assert "duplicate_ref" in keys[(1, 2, 2, None, 1)].flags
    assert keys[(1, 2, 3, 1, 0)].text == "ku ke"
    part2 = keys[(1, 2, 3, 2, 0)]
    assert part2.text == "ko kau" and "stray_char_removed" in part2.flags


def test_genuine_chapter_start_without_verse_one_is_not_relabelled(tmp_path: Path) -> None:
    xml = tmp_path / "t.xml"
    xml.write_text(
        "<TEI><text><body>\n"
        "<lg><l>a // BhP_01.01.001 //</l></lg>\n"
        "<lg><l>b // BhP_01.02.002 //</l></lg>\n"  # chapter 2 lacks verse 1 in this edition
        "<lg><l>c // BhP_01.02.003 //</l></lg>\n"
        "</body></text></TEI>\n",
        encoding="utf-8",
    )
    doc = parse_gretil_tei(xml)
    assert [(v.chapter, v.verse) for v in doc.verses] == [(1, 1), (2, 2), (2, 3)]
    assert doc.stats["labels_repaired"] == 0


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<lg><l>dangling text without id</l></lg>", "trailing lines"),
        ("", "no verses"),
        ("<p>BhP_01.01.001/0 sūta uvāca\n<lg><l>x /</l>\n<p>BhP_01.01.002/0 y uvāca", "speaker line inside"),
    ],
)
def test_malformed_inputs(tmp_path: Path, body: str, message: str) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(f"<TEI><text><body>\n{body}\n</body></text></TEI>", encoding="utf-8")
    with pytest.raises(ParserError, match=message):
        parse_gretil_tei(xml)


def test_missing_body(tmp_path: Path) -> None:
    xml = tmp_path / "nobody.xml"
    xml.write_text("<TEI></TEI>", encoding="utf-8")
    with pytest.raises(ParserError, match="no <body>"):
        parse_gretil_tei(xml)


@pytest.mark.skipif(
    not REAL_FILE.exists(), reason="real GRETIL file not downloaded (bhagavatam data download)"
)
def test_real_gretil_file_matches_known_structure() -> None:
    doc = parse_gretil_tei(REAL_FILE)
    assert doc.stats["records"] == 14099
    assert doc.stats["chapters"] == 335
    assert doc.stats["labels_repaired"] == 30
    assert doc.stats["duplicate_lines_removed"] == 584
    assert doc.stats["explicit_speakers"] == 1332
    assert doc.stats["records_per_canto"] == {
        1: 813,
        2: 391,
        3: 1413,
        4: 1449,
        5: 663,
        6: 851,
        7: 750,
        8: 933,
        9: 965,
        10: 3940,
        11: 1366,
        12: 565,
    }
