"""Dataset validation: numbering, completeness, duplicates, character hygiene and speakers.

Errors mean the data must not be used as-is; warnings are expected edition differences or known
source defects that are reported for review; info items are statistics worth knowing.
"""

from __future__ import annotations

import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from bhagavatam_ai.config import ValidationConfig
from bhagavatam_ai.data.normalize.iast import unexpected_iast_chars
from bhagavatam_ai.data.normalize.text import has_mojibake
from bhagavatam_ai.data.reference import CHAPTERS_PER_CANTO
from bhagavatam_ai.data.schema import VerseRecord


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    ref: str | None = None


@dataclass
class ValidationReport:
    source_id: str
    records: int
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: Severity, code: str, message: str, ref: str | None = None) -> None:
        self.issues.append(Issue(severity, code, message, ref))

    def count(self, severity: Severity) -> int:
        return sum(1 for i in self.issues if i.severity == severity)

    @property
    def ok(self) -> bool:
        return self.count(Severity.ERROR) == 0

    def to_dict(self, max_examples: int = 25) -> dict[str, Any]:
        by_code: dict[str, list[Issue]] = defaultdict(list)
        for issue in self.issues:
            by_code[issue.code].append(issue)
        return {
            "source_id": self.source_id,
            "records": self.records,
            "ok": self.ok,
            "counts": {s.value: self.count(s) for s in Severity},
            "issues": {
                code: {
                    "severity": items[0].severity.value,
                    "count": len(items),
                    "examples": [asdict(i) | {"severity": i.severity.value} for i in items[:max_examples]],
                }
                for code, items in sorted(by_code.items())
            },
            "stats": self.stats,
        }


def _check_keys_and_order(records: Sequence[VerseRecord], report: ValidationReport) -> None:
    seen: Counter[tuple[int, int, int, int | None, int]] = Counter()
    previous: tuple[int, int, int] | None = None
    order_breaks = 0
    for record in records:
        if record.chapter > CHAPTERS_PER_CANTO.get(record.canto, 0):
            report.add(
                Severity.ERROR,
                "INVALID_CHAPTER",
                f"canto {record.canto} has no chapter {record.chapter}",
                record.ref_key,
            )
        seen[(record.canto, record.chapter, record.verse, record.part, record.variant)] += 1
        position = (record.canto, record.chapter, record.verse)
        order_breaks += previous is not None and position < previous
        previous = position
        if record.variant > 0:
            report.add(
                Severity.WARNING, "REPEATED_VERSE_NUMBER", "edition repeats this verse number", record.ref_key
            )
    for ident, n in seen.items():
        if n > 1:
            report.add(
                Severity.ERROR,
                "DUPLICATE_RECORD_KEY",
                f"{n} records share key {ident}",
                f"{ident[0]}.{ident[1]}.{ident[2]}",
            )
    if order_breaks:
        report.add(Severity.WARNING, "ORDER", f"{order_breaks} records are out of canonical order")


def _check_coverage(
    records: Sequence[VerseRecord], expected: Mapping[str, int] | None, report: ValidationReport
) -> None:
    verses_by_chapter: dict[tuple[int, int], set[int]] = defaultdict(set)
    for record in records:
        verses_by_chapter[(record.canto, record.chapter)].add(record.verse)
    cantos = sorted({c for c, _ in verses_by_chapter})
    for canto in cantos:
        present = {ch for c, ch in verses_by_chapter if c == canto}
        missing = sorted(set(range(1, CHAPTERS_PER_CANTO[canto] + 1)) - present)
        if missing:
            report.add(
                Severity.WARNING, "MISSING_CHAPTERS", f"canto {canto} lacks chapters {missing}", str(canto)
            )
    for (canto, chapter), verses in sorted(verses_by_chapter.items()):
        key = f"{canto}.{chapter}"
        gaps = sorted(set(range(1, max(verses) + 1)) - verses)
        if gaps:
            report.add(Severity.WARNING, "VERSE_GAP", f"missing verse numbers {gaps}", key)
        want = expected.get(key) if expected is not None else None
        if want is not None and want != max(verses):
            report.add(
                Severity.WARNING, "VERSE_COUNT_DIFFERS", f"last verse {max(verses)} vs reference {want}", key
            )
    report.stats["chapters"] = len(verses_by_chapter)
    report.stats["cantos"] = cantos


def _record_issues(record: VerseRecord, cfg: ValidationConfig, mean: float, stdev: float) -> list[Issue]:
    text, ref = record.text, record.ref_key
    found: list[Issue] = []
    if unicodedata.normalize("NFC", text) != text:
        found.append(Issue(Severity.ERROR, "NOT_NFC", "text is not Unicode NFC", ref))
    if has_mojibake(text):
        found.append(Issue(Severity.ERROR, "MOJIBAKE", f"encoding damage: {text[:60]!r}", ref))
    if any(unicodedata.category(ch) == "Cc" and ch != "\n" for ch in text):
        found.append(Issue(Severity.ERROR, "CONTROL_CHARS", "control characters in text", ref))
    if len(text) < cfg.min_chars_per_verse:
        found.append(Issue(Severity.WARNING, "VERY_SHORT", f"only {len(text)} characters: {text!r}", ref))
    if stdev and abs(len(text) - mean) > 5 * stdev:
        found.append(Issue(Severity.INFO, "LENGTH_OUTLIER", f"{len(text)} chars (mean {mean:.0f})", ref))
    return found


def _check_content(records: Sequence[VerseRecord], cfg: ValidationConfig, report: ValidationReport) -> None:
    lengths = [len(r.text) for r in records]
    mean = statistics.fmean(lengths) if lengths else 0.0
    stdev = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    odd_chars: Counter[str] = Counter()
    for record in records:
        report.issues.extend(_record_issues(record, cfg, mean, stdev))
        if record.script == "latn-iast" and (odd := unexpected_iast_chars(record.text)):
            odd_chars.update(odd)
            report.add(
                Severity.WARNING, "UNEXPECTED_CHARS", f"non-IAST characters {dict(odd)}", record.ref_key
            )
    report.stats["chars"] = {"total": sum(lengths), "mean": round(mean, 1), "stdev": round(stdev, 1)}
    if odd_chars:
        report.stats["unexpected_chars"] = dict(odd_chars.most_common(30))


def _check_duplicates(records: Sequence[VerseRecord], report: ValidationReport) -> None:
    by_text: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_text[" ".join(record.text.lower().split())].append(record.ref_key)
    repeated = {text: refs for text, refs in by_text.items() if len(refs) > 1}
    for refs in repeated.values():
        report.add(Severity.INFO, "REPEATED_TEXT", f"identical text at {refs}", refs[0])
    report.stats["identical_text_groups"] = len(repeated)


def _speaker_stats(records: Sequence[VerseRecord], report: ValidationReport) -> None:
    explicit = sum(1 for r in records if r.speaker_raw and "speaker_carried_forward" not in r.flags)
    resolved = sum(1 for r in records if r.speaker)
    unmapped = Counter(f for r in records for f in r.flags if f.startswith("speaker_unmapped:"))
    report.stats["speakers"] = {
        "explicit_attributions": explicit,
        "records_with_speaker_label": sum(1 for r in records if r.speaker_raw),
        "records_with_canonical_speaker": resolved,
        "ambiguous": sum(1 for r in records if "speaker_ambiguous" in r.flags),
        "unmapped_keys": {k.split(":", 1)[1]: n for k, n in unmapped.most_common(30)},
    }
    for key, n in unmapped.items():
        report.add(Severity.INFO, "SPEAKER_UNMAPPED", f"{n} records", key.split(":", 1)[1])


def validate_records(
    source_id: str,
    records: Sequence[VerseRecord],
    cfg: ValidationConfig,
    expected: Mapping[str, int] | None = None,
) -> ValidationReport:
    report = ValidationReport(source_id=source_id, records=len(records))
    if not records:
        report.add(Severity.ERROR, "EMPTY", "no records")
        return report
    _check_keys_and_order(records, report)
    _check_coverage(records, expected, report)
    _check_content(records, cfg, report)
    _check_duplicates(records, report)
    _speaker_stats(records, report)
    report.stats["flags"] = dict(Counter(f.split(":")[0] for r in records for f in r.flags).most_common())
    return report
