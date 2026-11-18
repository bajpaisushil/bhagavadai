"""Pydantic schemas for the source/licensing manifest and for verse-level records."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# =========================================================================== manifest


class PermissionStatus(StrEnum):
    PUBLIC_DOMAIN = "public_domain"
    OPEN_LICENSE = "open_license"
    PERMISSION_GRANTED = "permission_granted"
    UNDER_REVIEW = "under_review"
    PERMISSION_REQUIRED = "permission_required"
    EXCLUDED = "excluded"


TRAINABLE_STATUSES = frozenset(
    {PermissionStatus.PUBLIC_DOMAIN, PermissionStatus.OPEN_LICENSE, PermissionStatus.PERMISSION_GRANTED}
)


class ContentRole(StrEnum):
    SANSKRIT_TEXT = "sanskrit_text"
    ENGLISH_TRANSLATION = "english_translation"
    COMMENTARY = "commentary"
    METADATA = "metadata"


Tristate = Literal["allowed", "not_allowed", "unclear"]
Script = Literal["latn-iast", "deva", "latn"]


class Evidence(Strict):
    url: str
    quote: str = Field(min_length=1)
    retrieved_on: date


class DownloadSpec(Strict):
    url: str = Field(pattern=r"^https://")
    filename: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    format: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    notes: str | None = None


class UsageRights(Strict):
    """What the project may do with a source. Enforced in code (see ``manifest.require_usage``)."""

    download: bool = False
    metadata: bool = False  # derive non-expressive facts (structure, verse counts, speaker labels)
    training_text: bool = False  # include the text itself in a training corpus
    evaluation_reference: bool = False  # use the text privately to check model answers


class SourceEntry(Strict):
    id: str
    title: str
    author_translator: str
    language: str
    content_roles: tuple[ContentRole, ...] = Field(min_length=1)
    edition: str
    publication_years: str
    copyright_holder: str
    license: str
    license_url: str | None = None
    attribution_requirements: str
    commercial_use: Tristate
    ml_training: Tristate
    permission_status: PermissionStatus
    usage: UsageRights
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    original_source_urls: tuple[str, ...] = ()
    downloads: tuple[DownloadSpec, ...] = ()
    parser: str | None = None
    # The source whose verse numbering serves as the reference structure for validating others.
    reference_structure: bool = False
    coverage: str = ""
    risks: tuple[str, ...] = ()
    notes: str = ""
    reviewed_on: date
    review_status: Literal["researched", "adversarially_verified", "legal_reviewed"] = "researched"

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG.match(value):
            raise ValueError("id must be a lowercase slug (a-z, 0-9, '-')")
        return value

    @model_validator(mode="after")
    def _usage_consistent_with_status(self) -> Self:
        if self.usage.training_text:
            if self.permission_status not in TRAINABLE_STATUSES:
                raise ValueError(
                    f"{self.id}: training_text requires status in {sorted(TRAINABLE_STATUSES)}, "
                    f"got {self.permission_status}"
                )
            if self.ml_training == "not_allowed":
                raise ValueError(f"{self.id}: training_text conflicts with ml_training=not_allowed")
        if self.permission_status in {PermissionStatus.EXCLUDED, PermissionStatus.PERMISSION_REQUIRED}:
            granted = [name for name, value in self.usage.model_dump().items() if value]
            if granted:
                raise ValueError(f"{self.id}: status {self.permission_status} cannot grant usage {granted}")
            if self.downloads:
                raise ValueError(f"{self.id}: status {self.permission_status} must not list downloads")
        if self.downloads and not self.usage.download:
            raise ValueError(f"{self.id}: downloads listed but usage.download is false")
        if self.usage.download and not self.downloads:
            raise ValueError(f"{self.id}: usage.download is true but no downloads are listed")
        if self.reference_structure and not self.usage.metadata:
            raise ValueError(f"{self.id}: reference_structure requires usage.metadata")
        return self


class SourceManifest(Strict):
    version: int = Field(ge=1)
    updated: date
    policy: str
    sources: tuple[SourceEntry, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [s.id for s in self.sources]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate source ids: {duplicates}")
        return self


# =========================================================================== records


class Provenance(Strict):
    source_id: str
    raw_file: str  # path relative to the project root
    raw_sha256: str
    locator: str  # where in the raw file, e.g. "line 1234"
    parser: str
    parser_version: str


class VerseRecord(Strict):
    """One verse (or verse part) from one source, after parsing and normalisation."""

    record_id: str
    source_id: str
    canto: int = Field(ge=1, le=12)
    chapter: int = Field(ge=1, le=90)
    verse: int = Field(ge=1)
    part: int | None = None  # split verses (e.g. GRETIL "_1/_2")
    variant: int = 0  # >0 when an edition repeats a verse number for extra verses
    role: ContentRole
    language: str
    script: Script
    text: str = Field(min_length=1)
    # The same text in other scripts, e.g. {"deva": "..."} for IAST Sanskrit records.
    alt_scripts: dict[str, str] = Field(default_factory=dict)
    speaker_raw: str | None = None
    speaker: str | None = None  # canonical name when unambiguous
    flags: tuple[str, ...] = ()
    provenance: Provenance

    @property
    def ref_key(self) -> str:
        return f"{self.canto}.{self.chapter}.{self.verse}"

    @property
    def chapter_key(self) -> str:
        return f"{self.canto}.{self.chapter}"
