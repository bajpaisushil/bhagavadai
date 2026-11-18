from __future__ import annotations

from collections import Counter

from bhagavatam_ai.data.normalize.ocr import (
    DehyphenationStats,
    build_vocabulary,
    dehyphenate,
    find_running_heads,
    is_page_number,
    strip_page_furniture,
    unwrap_paragraphs,
)


def test_page_numbers() -> None:
    for line in ["12", " 345 ", "[23]", "(7)", "xiv", "IX"]:
        assert is_page_number(line)
    for line in ["12 cows", "Canto 1", "", "chapter"]:
        assert not is_page_number(line)


def _page(n: int) -> list[str]:
    head = f"SRIMAD BHAGAVATAM {n}" if n % 2 else f"CANTO I, CHAPTER {n // 10 + 1}"
    body = [f"Body line {n}.{i} with distinct words {n * 7 + i}." for i in range(5)]
    speaker = ["Sūta said:"] if n % 3 == 0 else []
    return [head, *speaker, *body, str(n)]


def test_running_heads_detected_and_removed() -> None:
    pages = [_page(n) for n in range(1, 31)]
    heads = find_running_heads(pages)
    assert "srimad bhagavatam #" in heads and "canto i, chapter #" in heads
    assert "sūta said:" not in heads  # recurring text line, not page furniture
    cleaned = strip_page_furniture(pages, heads)
    for original, page in zip(pages, cleaned, strict=True):
        assert len(page) == len(original) - 2  # head + page number removed
        assert all(line.startswith(("Body", "Sūta")) for line in page)


def test_short_pages_are_left_alone() -> None:
    pages = [["HEAD", "x"]] * 10
    assert strip_page_furniture(pages, {"head"}) == [list(p) for p in pages]


def test_running_heads_need_enough_pages() -> None:
    assert find_running_heads([["HEAD", "body"]] * 3) == set()


def test_dehyphenate_uses_corpus_evidence() -> None:
    vocab = Counter({"bhagavatam": 5, "srimad-bhagavatam": 3, "devotion": 4})
    stats = DehyphenationStats()
    lines = [
        "the Bhaga-",
        "vatam is read",
        "Srimad-",
        "Bhagavatam again",
        "with devo-",
        "tion.",
        "an unknown-",
        "word",
    ]
    out = dehyphenate(lines, vocab, stats)
    assert out == ["the Bhagavatam is read", "Srimad-Bhagavatam again", "with devotion.", "an unknown-word"]
    assert stats.joined == 2 and stats.kept_hyphen == 2 and stats.ambiguous == ["unknown-word"]


def test_dehyphenate_keeps_diacritic_compounds() -> None:
    assert dehyphenate(["Śrīmad-", "Bhāgavatam"], Counter()) == ["Śrīmad-Bhāgavatam"]


def test_build_vocabulary_ignores_line_end_fragments() -> None:
    vocab = build_vocabulary(["the Bhaga-\nvatam and the bhagavatam"])
    assert vocab["bhagavatam"] == 1 and "bhaga" not in vocab and vocab["the"] == 2


def test_unwrap_paragraphs() -> None:
    assert unwrap_paragraphs(["a b", "c", "", "d", "", "", "e"]) == ["a b c", "d", "e"]
