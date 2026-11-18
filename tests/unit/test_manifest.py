from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from bhagavatam_ai.config import load_data_config
from bhagavatam_ai.data.manifest import (
    LicenseGateError,
    ManifestError,
    get_source,
    host_is_blocked,
    load_manifest,
    require_url_allowed,
    require_usage,
    sources_for,
)
from bhagavatam_ai.data.schema import SourceEntry, SourceManifest


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "test-source",
        "title": "T",
        "author_translator": "A",
        "language": "en",
        "content_roles": ["english_translation"],
        "edition": "E",
        "publication_years": "1900",
        "copyright_holder": "none",
        "license": "Public domain",
        "attribution_requirements": "none",
        "commercial_use": "allowed",
        "ml_training": "allowed",
        "permission_status": "public_domain",
        "usage": {"download": True, "metadata": True, "training_text": True},
        "evidence": [{"url": "https://example.org", "quote": "q", "retrieved_on": "2026-11-18"}],
        "downloads": [{"url": "https://example.org/a.txt", "filename": "a.txt", "format": "txt"}],
        "reviewed_on": "2026-11-18",
    }
    base.update(overrides)
    return base


def test_real_manifest_is_valid(repo_root: Path) -> None:
    manifest = load_manifest(repo_root / "data" / "manifests" / "sources.yaml")
    ids = [s.id for s in manifest.sources]
    assert "gretil-bhagavata-purana" in ids
    for source in manifest.sources:
        assert source.evidence, source.id
        if source.usage.training_text:
            assert source.permission_status.value in {"public_domain", "open_license", "permission_granted"}


def test_real_manifest_never_downloads_from_blocked_domains(repo_root: Path) -> None:
    manifest = load_manifest(repo_root / "data" / "manifests" / "sources.yaml")
    blocked = load_data_config(repo_root / "configs" / "data.yaml").blocked_domains
    assert "vedabase.io" in blocked
    for source in manifest.sources:
        for spec in source.downloads:
            assert not host_is_blocked(spec.url, blocked), spec.url


def test_valid_entry() -> None:
    entry = SourceEntry.model_validate(_entry())
    assert entry.usage.training_text


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"permission_status": "under_review"}, "training_text requires status"),
        ({"ml_training": "not_allowed"}, "conflicts with ml_training"),
        ({"permission_status": "excluded", "usage": {}}, "must not list downloads"),
        (
            {"permission_status": "excluded", "usage": {"metadata": True}, "downloads": []},
            "cannot grant usage",
        ),
        ({"usage": {"metadata": True}}, "usage.download is false"),
        ({"usage": {"download": True}, "downloads": []}, "no downloads are listed"),
        ({"id": "Bad_ID"}, "lowercase slug"),
        ({"evidence": []}, "at least 1"),
        ({"downloads": [{"url": "http://insecure.org/a", "filename": "a", "format": "t"}]}, "https"),
        ({"reference_structure": True, "usage": {"download": True}}, "reference_structure requires"),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_schema_rules(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SourceEntry.model_validate(_entry(**overrides))


def test_duplicate_ids_rejected() -> None:
    entry = _entry()
    with pytest.raises(ValidationError, match="duplicate source ids"):
        SourceManifest.model_validate(
            {"version": 1, "updated": "2026-11-18", "policy": "p", "sources": [entry, copy.deepcopy(entry)]}
        )


def test_load_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="Invalid manifest"):
        load_manifest(bad)


def test_usage_gate() -> None:
    entry = SourceEntry.model_validate(
        _entry(permission_status="under_review", usage={"download": True, "metadata": True})
    )
    require_usage(entry, "metadata")
    with pytest.raises(LicenseGateError, match="does not permit 'training_text'"):
        require_usage(entry, "training_text")


def test_get_source_and_sources_for() -> None:
    manifest = SourceManifest.model_validate(
        {"version": 1, "updated": "2026-11-18", "policy": "p", "sources": [_entry()]}
    )
    assert get_source(manifest, "test-source").id == "test-source"
    assert [s.id for s in sources_for(manifest, "training_text")] == ["test-source"]
    assert sources_for(manifest, "evaluation_reference") == []
    with pytest.raises(ManifestError, match="Unknown source"):
        get_source(manifest, "nope")


@pytest.mark.parametrize(
    ("url", "blocked"),
    [
        ("https://vedabase.io/en/library/sb/1/1/1/", True),
        ("https://www.vedabase.io/x", True),
        ("https://VEDABASE.IO./x", True),
        ("https://notvedabase.io/x", False),
        ("https://vedabase.io.example.com/x", False),
        ("https://gretil.sub.uni-goettingen.de/x", False),
    ],
)
def test_domain_blocklist(url: str, blocked: bool) -> None:
    assert host_is_blocked(url, ("vedabase.io",)) is blocked


def test_require_url_allowed() -> None:
    with pytest.raises(LicenseGateError):
        require_url_allowed("https://bbt.org/x", ("bbt.org",))
