"""Loading the source manifest and enforcing its licence decisions in code.

Every stage that touches source text calls :func:`require_usage` first, so a source marked
``excluded`` or ``permission_required`` (e.g. BBT / Prabhupada material) cannot be downloaded,
parsed into a corpus or trained on by accident - even if someone adds a URL for it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

import httpx
import yaml
from pydantic import ValidationError

from bhagavatam_ai.data.schema import SourceEntry, SourceManifest

Purpose = Literal["download", "metadata", "training_text", "evaluation_reference"]

# Policy invariant: always blocked, whatever a config file says (BBT / Prabhupada copyrighted text).
ALWAYS_BLOCKED: Final = (
    "vedabase.io",
    "vedabase.com",
    "bbt.org",
    "bbt.info",
    "bbtmedia.com",
    "prabhupadabooks.com",
)


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


def normalize_domain(domain: str) -> str:
    return domain.strip().lower().strip(".")


def connect_host(url: str | httpx.URL) -> str:
    """The host httpx will actually connect to: IDNA-normalised (so U+3002 etc. become '.'), lower-case,
    without userinfo, port or trailing dot. Checking any other parse of the URL can be bypassed."""
    return normalize_domain(httpx.URL(url).host if isinstance(url, str) else url.host)


def host_is_blocked(url: str | httpx.URL, blocked_domains: Iterable[str]) -> bool:
    host = connect_host(url)
    domains = {normalize_domain(d) for d in (*ALWAYS_BLOCKED, *blocked_domains) if normalize_domain(d)}
    return any(host == d or host.endswith("." + d) for d in domains)


def require_url_allowed(url: str | httpx.URL, blocked_domains: Iterable[str]) -> None:
    if host_is_blocked(url, blocked_domains):
        raise LicenseGateError(f"Refusing to fetch {url}: domain is on the project's blocklist")


def sources_for(manifest: SourceManifest, purpose: Purpose) -> list[SourceEntry]:
    return [s for s in manifest.sources if getattr(s.usage, purpose)]
