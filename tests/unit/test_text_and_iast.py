from __future__ import annotations

import unicodedata

import pytest

from bhagavatam_ai.data.normalize.iast import (
    devanagari_to_iast,
    iast_to_devanagari,
    normalize_iast,
    unexpected_iast_chars,
)
from bhagavatam_ai.data.normalize.text import clean_latin_text, collapse_whitespace, has_mojibake


def test_normalize_iast_forms() -> None:
    nfd = unicodedata.normalize("NFD", "kṛṣṇa")
    assert normalize_iast(nfd) == "kṛṣṇa"
    assert normalize_iast("saṁsāra") == "saṃsāra"  # dot-above anusvara -> dot-below
    assert normalize_iast("r̥ṣi") == "ṛṣi"  # ISO 15919 vocalic r
    assert normalize_iast("l̥̄") == "ḹ"
    assert normalize_iast("yato’nvayād") == "yato'nvayād"


@pytest.mark.parametrize(
    ("iast", "deva"),
    [
        ("kṛṣṇa", "कृष्ण"),
        ("jñāna", "ज्ञान"),
        ("kṣatriya", "क्षत्रिय"),
        ("vaiṣṇava", "वैष्णव"),
        ("gauḍīya", "गौडीय"),
        ("śrīmad-bhāgavatam", "श्रीमद्भागवतम्"),
        ("tat tvam", "तत् त्वम्"),
        ("yato 'nvayād", "यतो ऽन्वयाद्"),
        ("saṃsāraḥ", "संसारः"),
        ("goṃ", "गों"),  # 'oṃ' inside a word must not become ॐ
        ("mana-indriya", "मनइन्द्रिय"),  # hiatus across a compound hyphen, not the diphthong 'ai'
        ("tama-udbhūte", "तमउद्भूते"),  # ... nor 'au'
        ("tat-tvam", "तत्त्वम्"),  # consonants across a hyphen still form a conjunct
        ("ṛṣi ṝ ḷ", "ऋषि ॠ ऌ"),
        ("ka / kha //", "क । ख ॥"),
        ("12", "१२"),
    ],
)
def test_iast_to_devanagari(iast: str, deva: str) -> None:
    assert iast_to_devanagari(iast) == deva


@pytest.mark.parametrize(
    "word",
    [
        "kṛṣṇa",
        "dharmaḥ",
        "jñāna",
        "vaiṣṇava",
        "bhagavān",
        "prahlāda",
        "uddhava",
        "śukadeva",
        "pṛthu",
        "āditya",
    ],
)
def test_round_trip(word: str) -> None:
    assert devanagari_to_iast(iast_to_devanagari(word)) == word


def test_devanagari_specials() -> None:
    assert devanagari_to_iast("ॐ") == "oṃ"
    assert devanagari_to_iast("कँ") == "kam̐"
    assert devanagari_to_iast("क़") == "ka"  # nukta dropped
    assert devanagari_to_iast("धर्म॥१॥") == "dharma//1//"


def test_unexpected_iast_chars() -> None:
    assert not unexpected_iast_chars("dharmaḥ projjhita-kaitavo 'tra /")
    assert unexpected_iast_chars("dharma ç x") == {"ç": 1, "x": 1}


def test_clean_latin_text() -> None:
    raw = "The ﬁrst­ verse “quoted”\u200b  here\x07\r\nnext line"
    assert clean_latin_text(raw) == 'The first verse "quoted" here\nnext line'
    assert clean_latin_text("“x”", straighten_quotes=False) == "“x”"


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("a  b\n\n\n\nc") == "a b\n\nc"
    assert collapse_whitespace("a\nb", keep_newlines=False) == "a b"


@pytest.mark.parametrize("name", ["Kṛṣṇa", "Śukadeva Gosvāmī", "Parīkṣit", "Bhāgavatam", "Nārada", "saṃsāra"])
@pytest.mark.parametrize("encoding", ["cp1252", "latin-1"])
def test_mojibake_of_iast_names_is_detected(name: str, encoding: str) -> None:
    damaged = name.encode("utf-8").decode(encoding, errors="replace")
    assert has_mojibake(damaged), damaged
    assert not has_mojibake(name)


def test_mojibake() -> None:
    assert has_mojibake("KrÃ¡Â¹Â£Ã¡Â¹â€¡a")
    assert has_mojibake("donâ€™t")
    assert has_mojibake("broken \ufffd char")
    assert not has_mojibake("Kṛṣṇa don't café «quote»")


def test_clean_latin_text_keeps_c1_controls_for_validation() -> None:
    damaged = "Lord Kṛṣṇa".encode().decode("latin-1")  # contains C1 bytes such as \x9b
    assert any("\x80" <= ch <= "\x9f" for ch in clean_latin_text(damaged))
