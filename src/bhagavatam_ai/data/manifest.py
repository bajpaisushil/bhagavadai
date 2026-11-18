"""Loading the source manifest and enforcing its licence decisions in code.

Every stage that touches source text calls :func:`require_usage` first, so a source marked
``excluded`` or ``permission_required`` (e.g. BBT / Prabhupada material) cannot be downloaded,
parsed into a corpus or trained on by accident - even if someone adds a URL for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from bhagavatam_ai.data.schema import SourceEntry, SourceManifest

Purpose = Literal["download", "metadata", "training_text", "evaluation_reference"]


class ManifestError(RuntimeError):
    """The manifest file is missing or invalid."""


class LicenseGateError(PermissionError):
    """An operation is not permitted by a source's recorded licence/permission status."""


def load_manifest(path: Path) -> SourceManifest:
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        return SourceManifest.model_validate(document)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ManifestError(f"Invalid manifest {path}:\n{exc}") from exc


def get_source(manifest: SourceManifest, source_id: str) -> SourceEntry:
    for entry in manifest.sources:
        if entry.id == source_id:
            return entry
    raise ManifestError(f"Unknown source id {source_id!r}; known: {[s.id for s in manifest.sources]}")


def require_usage(entry: SourceEntry, purpose: Purpose) -> None:
    """Raise :class:`LicenseGateError` unless ``entry`` permits ``purpose``."""
    if not getattr(entry.usage, purpose):
        raise LicenseGateError(
            f"Source '{entry.id}' ({entry.permission_status}) does not permit '{purpose}'. "
            f"Licence: {entry.license}. Update the manifest only with documented evidence."
        )


def host_is_blocked(url: str, blocked_domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in (b.lower().lstrip(".") for b in blocked_domains))


def require_url_allowed(url: str, blocked_domains: tuple[str, ...]) -> None:
    if host_is_blocked(url, blocked_domains):
        raise LicenseGateError(f"Refusing to fetch {url}: domain is on the project's blocklist")


def sources_for(manifest: SourceManifest, purpose: Purpose) -> list[SourceEntry]:
    return [s for s in manifest.sources if getattr(s.usage, purpose)]
