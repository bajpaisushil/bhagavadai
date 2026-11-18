from __future__ import annotations

from typing import Any

import pytest

from bhagavatam_ai.config import DedupConfig, SplitConfig, ValidationConfig
from bhagavatam_ai.data.dedup import exact_duplicates, near_duplicates, shingles
from bhagavatam_ai.data.schema import VerseRecord
from bhagavatam_ai.data.split import leakage_report, plan_split
from bhagavatam_ai.data.validate import Severity, validate_records

CFG = ValidationConfig()


def rec(canto: int, chapter: int, verse: int, text: str | None = None, **kw: Any) -> VerseRecord:
    data: dict[str, Any] = {
        "record_id": f"s:{canto}.{chapter}.{verse}"
        + (f"#{kw.get('variant', 0)}" if kw.get("variant") else ""),
        "source_id": "s",
        "canto": canto,
        "chapter": chapter,
        "verse": verse,
        "role": "sanskrit_text",
        "language": "sa",
        "script": "latn-iast",
        "text": text or f"dharma {canto} {chapter} {verse} satyaṃ paraṃ dhīmahi",
        "provenance": {
            "source_id": "s",
            "raw_file": "r",
            "raw_sha256": "0",
            "locator": "l",
            "parser": "p",
            "parser_version": "1",
        },
    }
    data.update(kw)
    return VerseRecord.model_validate(data)


def codes(report: Any, severity: Severity | None = None) -> set[str]:
    return {i.code for i in report.issues if severity is None or i.severity == severity}


def test_clean_chapter_has_no_errors() -> None:
    report = validate_records("s", [rec(1, 1, v) for v in range(1, 6)], CFG)
    assert report.ok and report.stats["chapters"] == 1


def test_gaps_missing_chapters_and_counts() -> None:
    records = [rec(1, 1, 1), rec(1, 1, 2), rec(1, 1, 4), rec(1, 3, 1)]
    report = validate_records("s", records, CFG, expected={"1.1": 5, "1.3": 1})
    assert {"VERSE_GAP", "MISSING_CHAPTERS", "VERSE_COUNT_DIFFERS"} <= codes(report, Severity.WARNING)
    assert report.ok  # warnings only


def test_errors_are_detected() -> None:
    records = [
        rec(1, 1, 1),
        rec(1, 1, 1),  # duplicate key
        rec(1, 1, 2, text="kārma"),  # not NFC
        rec(1, 1, 3, text="KrÃ¡Â¹Â£ mojibake"),
        rec(1, 1, 4, text="ok\x07bell"),
        rec(1, 20, 1),  # canto 1 has 19 chapters
    ]
    report = validate_records("s", records, CFG)
    assert {"DUPLICATE_RECORD_KEY", "NOT_NFC", "MOJIBAKE", "CONTROL_CHARS", "INVALID_CHAPTER"} <= codes(
        report, Severity.ERROR
    )
    assert not report.ok


def test_warnings_for_content() -> None:
    records = [
        rec(1, 1, 1, text="ab"),
        rec(1, 1, 2, text="dharma ç"),
        rec(1, 1, 3, variant=0),
        rec(1, 1, 3, variant=1, text="anyaḥ pāṭhaḥ tasyaiva saṅkhyāyāḥ"),
    ]
    report = validate_records("s", records, CFG)
    assert {"VERY_SHORT", "UNEXPECTED_CHARS", "REPEATED_VERSE_NUMBER"} <= codes(report, Severity.WARNING)
    assert report.stats["unexpected_chars"] == {"ç": 1}


def test_repeated_text_and_order() -> None:
    same = "eko dharmaḥ sanātanaḥ iti"
    report = validate_records("s", [rec(1, 2, 1, text=same), rec(1, 1, 1, text=same)], CFG)
    assert "REPEATED_TEXT" in codes(report, Severity.INFO) and "ORDER" in codes(report, Severity.WARNING)


def test_empty_and_report_dict() -> None:
    report = validate_records("s", [], CFG)
    assert not report.ok and codes(report) == {"EMPTY"}
    as_dict = validate_records("s", [rec(1, 1, 1), rec(1, 1, 3)], CFG).to_dict(max_examples=1)
    assert as_dict["issues"]["VERSE_GAP"]["count"] == 1
    assert as_dict["counts"]["warning"] == 2  # the gap, plus chapters 2-19 of canto 1 missing
    assert len(as_dict["issues"]["MISSING_CHAPTERS"]["examples"]) == 1


def test_speaker_stats() -> None:
    records = [
        rec(1, 1, 1, speaker_raw="sūta uvāca", speaker="Sūta Gosvāmī"),
        rec(1, 1, 2, speaker_raw="sūta uvāca", speaker="Sūta Gosvāmī", flags=("speaker_carried_forward",)),
        rec(1, 1, 3, speaker_raw="rājovāca", flags=("speaker_ambiguous",)),
        rec(1, 1, 4, speaker_raw="xyz uvāca", flags=("speaker_unmapped:xyz",)),
    ]
    stats = validate_records("s", records, CFG).stats["speakers"]
    assert stats == {
        "explicit_attributions": 3,
        "records_with_speaker_label": 4,
        "records_with_canonical_speaker": 2,
        "ambiguous": 1,
        "unmapped_keys": {"xyz": 1},
    }


# --------------------------------------------------------------------------- dedup

DCFG = DedupConfig(shingle_size=5, num_perm=64, bands=16, near_duplicate_threshold=0.8)
LONG = (
    "the sages assembled in the forest of naimisa asked suta about the absolute truth and the duties of man"
)


def test_shingles_normalise() -> None:
    assert shingles("A-B", 2) == shingles("a b", 2)
    assert shingles("", 3) == set()


def test_exact_duplicates() -> None:
    groups = exact_duplicates({"a": "Same text!", "b": "same   text", "c": "other"})
    assert groups == [["a", "b"]]


def test_near_duplicates_found_and_scored() -> None:
    texts = {"a": LONG, "b": LONG.replace("forest", "woods"), "c": "completely unrelated words about rivers"}
    pairs = near_duplicates(texts, DCFG)
    assert [(p.a, p.b) for p in pairs] == [("a", "b")] and 0.8 <= pairs[0].jaccard < 1


def test_near_duplicates_cross_mode_only_reports_cross_pairs() -> None:
    pairs = near_duplicates({"t1": LONG, "t2": LONG}, DCFG, cross={"v1": LONG + "!"})
    assert {(p.a, p.b) for p in pairs} == {("t1", "v1"), ("t2", "v1")}


def test_bands_must_divide_perms() -> None:
    with pytest.raises(ValueError, match="divisible"):
        near_duplicates({"a": "x"}, DedupConfig(num_perm=10, bands=3))


# --------------------------------------------------------------------------- split


def test_split_is_deterministic_stratified_and_seeded() -> None:
    cfg = SplitConfig(seed=7, val_fraction=0.05)
    plan = plan_split(cfg)
    assert plan == plan_split(cfg)
    assert plan != plan_split(SplitConfig(seed=8, val_fraction=0.05))
    cantos = {int(c.split(".")[0]) for c in plan.val_chapters}
    assert cantos == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}  # every canto with >= 10 chapters contributes
    assert 12 <= len(plan.val_chapters) <= 25


def test_split_assignment_and_forced() -> None:
    plan = plan_split(SplitConfig(seed=1, val_fraction=0.0, forced_val_chapters=("3.25", "11.7")))
    assert plan.val_chapters == ("3.25", "11.7")
    assert plan.split_of(3, 25) == "val" and plan.split_of(3, 24) == "train"
    assert plan.to_dict()["val_chapters"] == ["3.25", "11.7"]


def test_split_rejects_bad_forced_chapter() -> None:
    with pytest.raises(ValueError):
        plan_split(SplitConfig(forced_val_chapters=("1.99",)))


def test_leakage_report() -> None:
    report = leakage_report({"t1": LONG, "t2": "train only text here"}, {"v1": LONG, "v2": "val only"}, DCFG)
    assert report["exact_cross_split_groups"] == 1 and report["near_duplicate_pairs"] == 1


def test_equivalence_groups_are_transitive() -> None:
    from bhagavatam_ai.data.split import equivalence_groups

    texts = {
        "a": LONG,
        "b": LONG.replace("forest", "woods"),
        "c": LONG.replace("forest", "woods") + " indeed",
        "d": "unrelated text about something else entirely",
    }
    groups = equivalence_groups(texts, DCFG)
    assert groups == [["a", "b", "c"]]
