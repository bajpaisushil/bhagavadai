"""Pipeline orchestration: raw files -> parsed, normalised records -> validation -> corpus splits.

Stages and outputs (all paths relative to the project root):

1. ``download``  data/raw/<source>/<file> + PROVENANCE.json            (licence gate: download)
2. ``parse``     data/interim/<source>/verses.jsonl + parse_stats.json  (gate: metadata/training_text)
3. ``validate``  reports/data_validation/<run>.json
4. ``corpus``    data/processed/cpt/{train,val}.jsonl + dataset_card.json (licence gate: training_text)

Only sources whose manifest entry grants ``training_text`` ever reach ``data/processed``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bhagavatam_ai.config import DataConfig
from bhagavatam_ai.data.manifest import (
    LicenseGateError,
    get_source,
    load_manifest,
    require_usage,
    sources_for,
)
from bhagavatam_ai.data.normalize.iast import iast_to_devanagari, normalize_iast
from bhagavatam_ai.data.normalize.speakers import SpeakerTable
from bhagavatam_ai.data.normalize.text import clean_latin_text
from bhagavatam_ai.data.parsers import ParsedDocument, get_parser
from bhagavatam_ai.data.schema import (
    ContentRole,
    Provenance,
    Script,
    SourceEntry,
    SourceManifest,
    VerseRecord,
)
from bhagavatam_ai.data.split import leakage_report, plan_split
from bhagavatam_ai.data.validate import Severity, ValidationReport, validate_records
from bhagavatam_ai.paths import ProjectPaths
from bhagavatam_ai.utils.hashing import sha256_file
from bhagavatam_ai.utils.io import iter_jsonl, read_json, write_json, write_jsonl

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineContext:
    paths: ProjectPaths
    config: DataConfig
    manifest: SourceManifest
    speakers: SpeakerTable

    @classmethod
    def load(cls, paths: ProjectPaths, config: DataConfig) -> PipelineContext:
        return cls(
            paths=paths,
            config=config,
            manifest=load_manifest(paths.resolve(config.manifest)),
            speakers=SpeakerTable.load(paths.reference / "speakers.yaml"),
        )

    def interim_file(self, source_id: str) -> Path:
        return self.paths.interim / source_id / "verses.jsonl"

    def show(self, path: Path) -> str:
        """Project-relative path for messages (absolute paths can reveal the user name)."""
        return str(path.relative_to(self.paths.root)) if path.is_relative_to(self.paths.root) else path.name


def _script_for(entry: SourceEntry) -> Script:
    if ContentRole.SANSKRIT_TEXT in entry.content_roles:
        return "latn-iast"
    return "latn"


def to_records(
    entry: SourceEntry, doc: ParsedDocument, raw_rel: str, raw_sha: str, speakers: SpeakerTable
) -> list[VerseRecord]:
    script = _script_for(entry)
    role = entry.content_roles[0]
    records = []
    for v in doc.verses:
        flags = list(v.flags)
        if script == "latn-iast":
            text = normalize_iast(v.text)
            alt = {"deva": iast_to_devanagari(text)}
        else:
            text = clean_latin_text(v.text)
            alt = {}
        speaker = None
        if v.speaker_raw:
            speaker, ambiguous, key = speakers.resolve(v.speaker_raw)
            if ambiguous:
                flags.append("speaker_ambiguous")
            elif speaker is None:
                flags.append(f"speaker_unmapped:{key}")
        suffix = (f"_{v.part}" if v.part else "") + (f"#{v.variant}" if v.variant else "")
        records.append(
            VerseRecord(
                record_id=f"{entry.id}:{v.canto}.{v.chapter}.{v.verse}{suffix}",
                source_id=entry.id,
                canto=v.canto,
                chapter=v.chapter,
                verse=v.verse,
                part=v.part,
                variant=v.variant,
                role=role,
                language=entry.language,
                script=script,
                text=text,
                alt_scripts=alt,
                speaker_raw=normalize_iast(v.speaker_raw) if v.speaker_raw else None,
                speaker=speaker,
                flags=tuple(flags),
                provenance=Provenance(
                    source_id=entry.id,
                    raw_file=raw_rel,
                    raw_sha256=raw_sha,
                    locator=v.locator,
                    parser=doc.parser,
                    parser_version=doc.parser_version,
                ),
            )
        )
    return records


def parse_source(ctx: PipelineContext, source_id: str) -> dict[str, Any]:
    entry = get_source(ctx.manifest, source_id)
    if not (entry.usage.metadata or entry.usage.training_text):
        require_usage(entry, "metadata")  # raises with an explanatory message
    if entry.parser is None or not entry.downloads:
        raise PipelineError(f"source {source_id} has no parser/download configured")
    spec = entry.downloads[0]
    raw = ctx.paths.raw / entry.id / spec.filename
    if not raw.exists():
        raise PipelineError(
            f"{ctx.show(raw)} not found; run `bhagavatam data download --source {source_id}` first"
        )
    raw_sha = sha256_file(raw)
    if spec.sha256 and raw_sha != spec.sha256:
        raise PipelineError(f"{raw.name}: sha256 {raw_sha} differs from the manifest's pinned {spec.sha256}")
    doc = get_parser(entry.parser)(raw)
    declared = doc.header.get("licence_url")
    if declared and entry.license_url and declared.rstrip("/") != entry.license_url.rstrip("/"):
        raise LicenseGateError(
            f"{source_id}: file declares licence {declared}, manifest records {entry.license_url}. "
            "The upstream licence changed - review before use."
        )
    raw_rel = str(raw.relative_to(ctx.paths.root))
    records = to_records(entry, doc, raw_rel, raw_sha, ctx.speakers)
    out = ctx.interim_file(source_id)
    write_jsonl(out, (r.model_dump(mode="json") for r in records))
    stats = {
        "source_id": source_id,
        "parser": doc.parser,
        "parser_version": doc.parser_version,
        "raw_file": raw_rel,
        "raw_sha256": raw_sha,
        **doc.stats,
    }
    write_json(out.parent / "parse_stats.json", stats)
    logger.info("Parsed %s: %d records -> %s", source_id, len(records), out.relative_to(ctx.paths.root))
    return stats


def load_records(ctx: PipelineContext, source_id: str) -> list[VerseRecord]:
    path = ctx.interim_file(source_id)
    if not path.exists():
        raise PipelineError(f"{ctx.show(path)} not found; run the parse stage for {source_id} first")
    return [VerseRecord.model_validate(row) for row in iter_jsonl(path)]


def structure_of(records: list[VerseRecord]) -> dict[str, int]:
    """Chapter key -> highest verse number (the edition's verse count)."""
    last: dict[str, int] = {}
    for r in records:
        last[r.chapter_key] = max(last.get(r.chapter_key, 0), r.verse)
    return dict(sorted(last.items(), key=lambda kv: tuple(int(x) for x in kv[0].split("."))))


def write_reference_structure(ctx: PipelineContext, source_id: str) -> Path:
    """Persist verse counts per chapter (non-expressive facts) from a metadata source."""
    entry = get_source(ctx.manifest, source_id)
    require_usage(entry, "metadata")
    records = load_records(ctx, source_id)
    chapters = structure_of(records)
    path = ctx.paths.resolve(ctx.config.expected_structure)
    write_json(
        path,
        {
            "description": "Highest verse number per chapter (canto.chapter -> verses) in the reference.",
            "derived_from": source_id,
            "edition": entry.edition,
            "note": "Verse numbering differs slightly between editions; treat mismatches as warnings.",
            "chapters": chapters,
            "totals": {
                "chapters": len(chapters),
                "verses_by_numbering": sum(chapters.values()),
                "records": len(records),
            },
        },
    )
    return path


def load_reference_structure(ctx: PipelineContext) -> dict[str, int] | None:
    path = ctx.paths.resolve(ctx.config.expected_structure)
    if not path.exists():
        return None
    chapters: dict[str, int] = read_json(path)["chapters"]
    return chapters


def validate_source(ctx: PipelineContext, source_id: str, *, use_reference: bool = True) -> ValidationReport:
    records = load_records(ctx, source_id)
    expected = load_reference_structure(ctx) if use_reference else None
    return validate_records(source_id, records, ctx.config.validation, expected)


def build_corpus(ctx: PipelineContext) -> dict[str, Any]:
    """Assemble the continued-pretraining corpus from training-eligible sources only."""
    plan = plan_split(ctx.config.split)
    out_dir = ctx.paths.processed / "cpt"
    write_json(ctx.paths.processed / "split_plan.json", plan.to_dict())
    eligible = sources_for(ctx.manifest, "training_text")
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    per_source: dict[str, Counter[str]] = {}
    for entry in eligible:
        require_usage(entry, "training_text")
        records = load_records(ctx, entry.id)
        report = validate_records(entry.id, records, ctx.config.validation, load_reference_structure(ctx))
        if not report.ok and ctx.config.validation.fail_on_error:
            raise PipelineError(f"{entry.id} failed validation ({report.count(Severity.ERROR)} errors)")
        counts: Counter[str] = Counter()
        for r in records:
            split = plan.split_of(r.canto, r.chapter)
            counts[split] += 1
            rows[split].append(
                {
                    "record_id": r.record_id,
                    "source_id": r.source_id,
                    "ref": f"SB {r.ref_key}",
                    "chapter": r.chapter_key,
                    "role": r.role.value,
                    "language": r.language,
                    "text": r.text,
                    "speaker": r.speaker,
                }
            )
        per_source[entry.id] = counts
    for split_name, split_rows in rows.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", split_rows)
    leakage = (
        leakage_report(
            {r["record_id"]: r["text"] for r in rows["train"]},
            {r["record_id"]: r["text"] for r in rows["val"]},
            ctx.config.dedup,
        )
        if rows["train"] and rows["val"]
        else {"note": "not computed: a split is empty"}
    )
    card = {
        "name": "bhagavatam-cpt",
        "split_plan": plan.to_dict(),
        "sources": [
            {
                "id": e.id,
                "title": e.title,
                "author_translator": e.author_translator,
                "license": e.license,
                "attribution": e.attribution_requirements,
                "counts": dict(per_source.get(e.id, {})),
            }
            for e in eligible
        ],
        "records": {split: len(split_rows) for split, split_rows in rows.items()},
        "characters": {split: sum(len(r["text"]) for r in split_rows) for split, split_rows in rows.items()},
        "leakage": leakage,
        "note": None
        if eligible
        else (
            "No source is cleared for training_text yet: the corpus is intentionally empty until a "
            "public-domain or properly licensed English translation is verified and added to the manifest."
        ),
    }
    write_json(out_dir / "dataset_card.json", card)
    return card
