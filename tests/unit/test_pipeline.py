from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from bhagavatam_ai.config import load_data_config
from bhagavatam_ai.data.manifest import LicenseGateError
from bhagavatam_ai.data.parsers.base import _REGISTRY, ParsedDocument, ParsedVerse
from bhagavatam_ai.data.pipeline import (
    PipelineContext,
    PipelineError,
    build_corpus,
    load_records,
    parse_source,
    validate_source,
    write_reference_structure,
)
from bhagavatam_ai.paths import ProjectPaths
from bhagavatam_ai.utils.hashing import sha256_file
from bhagavatam_ai.utils.io import iter_jsonl, read_json


def _source(sid: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": sid,
        "title": sid,
        "author_translator": "A",
        "language": "sa",
        "content_roles": ["sanskrit_text"],
        "edition": "E",
        "publication_years": "2020",
        "copyright_holder": "x",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution_requirements": "credit",
        "commercial_use": "not_allowed",
        "ml_training": "unclear",
        "permission_status": "under_review",
        "usage": {"download": True, "metadata": True},
        "evidence": [{"url": "https://example.org", "quote": "q", "retrieved_on": "2026-11-18"}],
        "downloads": [{"url": "https://example.org/mini.xml", "filename": "mini.xml", "format": "tei-xml"}],
        "parser": "gretil_tei_bhp",
        "reviewed_on": "2026-11-18",
    }
    entry.update(overrides)
    return entry


def _write_manifest(root: Path, sources: list[dict[str, Any]]) -> None:
    doc = {"version": 1, "updated": "2026-11-18", "policy": "test", "sources": sources}
    (root / "data/manifests/sources.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8"
    )


def _context(root: Path) -> PipelineContext:
    return PipelineContext.load(ProjectPaths(root), load_data_config(root / "configs/data.yaml"))


@pytest.fixture
def ctx(project: Path, fixtures_dir: Path) -> PipelineContext:
    raw = project / "data/raw/mini-sanskrit"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "mini_gretil.xml", raw / "mini.xml")
    _write_manifest(project, [_source("mini-sanskrit", reference_structure=True)])
    return _context(project)


def test_parse_validate_reference(ctx: PipelineContext) -> None:
    stats = parse_source(ctx, "mini-sanskrit")
    assert stats["records"] == 10 and stats["raw_file"] == "data/raw/mini-sanskrit/mini.xml"
    records = load_records(ctx, "mini-sanskrit")
    first = records[0]
    assert first.record_id == "mini-sanskrit:1.1.1" and first.alt_scripts["deva"].startswith("क ख ग")
    assert first.provenance.raw_sha256 == sha256_file(ctx.paths.raw / "mini-sanskrit/mini.xml")
    shuka = next(r for r in records if r.ref_key == "1.1.2")
    assert shuka.speaker == "Śukadeva Gosvāmī"
    assert any(r.record_id.endswith("#1") for r in records)
    path = write_reference_structure(ctx, "mini-sanskrit")
    assert read_json(path)["chapters"] == {"1.1": 5, "1.2": 3}
    report = validate_source(ctx, "mini-sanskrit")
    assert report.ok, report.to_dict()


def test_parse_requires_raw_file(ctx: PipelineContext) -> None:
    (ctx.paths.raw / "mini-sanskrit/mini.xml").unlink()
    with pytest.raises(PipelineError, match="data download"):
        parse_source(ctx, "mini-sanskrit")


def test_pinned_checksum_enforced(project: Path, fixtures_dir: Path) -> None:
    raw = project / "data/raw/mini-sanskrit"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "mini_gretil.xml", raw / "mini.xml")
    src = _source("mini-sanskrit")
    src["downloads"][0]["sha256"] = "0" * 64
    _write_manifest(project, [src])
    with pytest.raises(PipelineError, match="differs from the manifest"):
        parse_source(_context(project), "mini-sanskrit")


def test_upstream_licence_change_is_detected(project: Path, fixtures_dir: Path) -> None:
    raw = project / "data/raw/mini-sanskrit"
    raw.mkdir(parents=True)
    shutil.copy(fixtures_dir / "mini_gretil.xml", raw / "mini.xml")
    _write_manifest(
        project, [_source("mini-sanskrit", license_url="https://creativecommons.org/licenses/by/4.0/")]
    )
    with pytest.raises(LicenseGateError, match="licence changed"):
        parse_source(_context(project), "mini-sanskrit")


def test_excluded_source_cannot_be_parsed(project: Path) -> None:
    _write_manifest(project, [_source("blocked", permission_status="excluded", usage={}, downloads=[])])
    with pytest.raises(LicenseGateError):
        parse_source(_context(project), "blocked")


def test_corpus_empty_without_trainable_sources(ctx: PipelineContext) -> None:
    parse_source(ctx, "mini-sanskrit")
    card = build_corpus(ctx)
    assert card["records"] == {"train": 0, "val": 0} and "No source is cleared" in card["note"]
    assert (ctx.paths.processed / "split_plan.json").exists()


@pytest.fixture
def fake_english_parser() -> Any:
    name = "test_fake_english"

    def parse(path: Path) -> ParsedDocument:
        verses = [
            ParsedVerse(
                canto=c,
                chapter=ch,
                verse=v,
                text=f"English rendering {c}.{ch}.{v} of the story.",
                locator=f"line {i}",
            )
            for i, (c, ch, v) in enumerate([(1, 1, 1), (1, 1, 2), (3, 25, 1), (3, 25, 2)])
        ]
        return ParsedDocument(parser=name, parser_version="0", verses=verses)

    _REGISTRY[name] = parse
    yield name
    _REGISTRY.pop(name)


def test_corpus_only_contains_trainable_sources_split_by_chapter(
    project: Path, fixtures_dir: Path, fake_english_parser: str
) -> None:
    for sid in ("mini-sanskrit", "pd-english"):
        (project / f"data/raw/{sid}").mkdir(parents=True)
    shutil.copy(fixtures_dir / "mini_gretil.xml", project / "data/raw/mini-sanskrit/mini.xml")
    (project / "data/raw/pd-english/en.txt").write_text("fixture", encoding="utf-8")
    english = _source(
        "pd-english",
        language="en",
        content_roles=["english_translation"],
        license="Public domain",
        license_url=None,
        commercial_use="allowed",
        ml_training="allowed",
        permission_status="public_domain",
        usage={"download": True, "metadata": True, "training_text": True},
        parser=fake_english_parser,
        downloads=[{"url": "https://example.org/en.txt", "filename": "en.txt", "format": "txt"}],
    )
    _write_manifest(project, [_source("mini-sanskrit"), english])
    data_cfg = project / "configs/data.yaml"
    cfg = yaml.safe_load(data_cfg.read_text())
    cfg["data"]["split"]["forced_val_chapters"] = ["3.25"]
    cfg["data"]["split"]["val_fraction"] = 0.0
    data_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ctx = _context(project)
    parse_source(ctx, "mini-sanskrit")
    parse_source(ctx, "pd-english")
    card = build_corpus(ctx)
    assert card["records"] == {"train": 2, "val": 2}
    assert [s["id"] for s in card["sources"]] == ["pd-english"]  # the NC-SA Sanskrit never enters the corpus
    val = list(iter_jsonl(project / "data/processed/cpt/val.jsonl"))
    assert {r["chapter"] for r in val} == {"3.25"} and all(r["source_id"] == "pd-english" for r in val)
    assert card["leakage"]["exact_cross_split_groups"] == 0
