"""Licence-gated, reproducible downloads of raw source files.

For each manifest download entry: check the licence gate, honour robots.txt (RFC 9309), fetch with
retries and back-off (respecting ``Retry-After``), enforce a size limit, write atomically, verify the
pinned SHA-256 when the manifest has one, and record provenance (including the final URL after any
redirects) next to the file.

The HTTPS-only rule and the domain blocklist are enforced by an httpx request hook on *every* request
the client sends - robots.txt, each redirect hop and the final fetch - so a permitted URL that redirects
to a blocked domain is refused before any byte is requested from it.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx

from bhagavatam_ai.config import DownloadConfig
from bhagavatam_ai.data.manifest import LicenseGateError, connect_host, require_url_allowed, require_usage
from bhagavatam_ai.data.schema import DownloadSpec, SourceEntry
from bhagavatam_ai.utils.hashing import sha256_file
from bhagavatam_ai.utils.io import atomic_write_bytes, read_json, write_json

logger = logging.getLogger(__name__)
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    url: str
    final_url: str | None  # after redirects; None for cached files
    file: str
    status: str  # downloaded | cached
    sha256: str
    bytes: int
    http_status: int | None
    content_type: str | None
    last_modified: str | None
    retrieved_utc: str


class RequestGuard:
    """httpx request hook: refuse any request that is not HTTPS or targets a blocked domain."""

    def __init__(self, blocked_domains: tuple[str, ...]) -> None:
        self.blocked_domains = blocked_domains

    def __call__(self, request: httpx.Request) -> None:
        if request.url.scheme != "https":
            raise LicenseGateError(f"Refusing non-HTTPS request to {request.url}")
        require_url_allowed(request.url, self.blocked_domains)


def guarded(client: httpx.Client, guard: RequestGuard) -> httpx.Client:
    """Install ``guard`` as a request hook on ``client`` (idempotent)."""
    hooks = dict(client.event_hooks)
    requests = [h for h in hooks.get("request", []) if not isinstance(h, RequestGuard)]
    hooks["request"] = [guard, *requests]
    client.event_hooks = hooks
    return client


@dataclass(frozen=True)
class _RobotsRules:
    """Parsed robots.txt, or a blanket verdict when the file is absent (allow) or forbidden (deny)."""

    parser: RobotFileParser | None = None
    allow_everything: bool = False

    def can_fetch(self, user_agent: str, url: str) -> bool:
        return self.allow_everything if self.parser is None else self.parser.can_fetch(user_agent, url)


class RobotsCache:
    """robots.txt per origin, following RFC 9309 section 2.3.1: 2xx -> parse; 401/403 -> disallow all;
    other 4xx -> allow all; 5xx, 429 or network errors -> unreachable, so nothing may be fetched."""

    def __init__(self, client: httpx.Client, user_agent: str) -> None:
        self._client = client
        self._user_agent = user_agent
        self._rules: dict[str, _RobotsRules] = {}

    def _fetch_rules(self, origin: str) -> _RobotsRules:
        try:
            response = self._client.get(f"{origin}/robots.txt")
        except httpx.HTTPError as exc:
            raise DownloadError(
                f"robots.txt at {origin} unreachable ({type(exc).__name__}); retry later"
            ) from exc
        status = response.status_code
        if 200 <= status < 300:
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            return _RobotsRules(parser=parser)
        if status in {401, 403}:
            return _RobotsRules(allow_everything=False)
        if 400 <= status < 500 and status != 429:
            return _RobotsRules(allow_everything=True)
        raise DownloadError(f"robots.txt at {origin} returned HTTP {status}; retry later")

    def allowed(self, url: str) -> bool:
        parsed = httpx.URL(url)
        origin = f"{parsed.scheme}://{connect_host(parsed)}" + (f":{parsed.port}" if parsed.port else "")
        if origin not in self._rules:
            self._rules[origin] = self._fetch_rules(origin)
        return self._rules[origin].can_fetch(self._user_agent, url)


def _retry_delay(attempt: int, cfg: DownloadConfig, retry_after: str | None) -> float:
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after.strip()), 300.0)
    return float(cfg.backoff_s * 2.0**attempt + random.uniform(0, cfg.backoff_s))


def _fetch(
    client: httpx.Client,
    spec: DownloadSpec,
    cfg: DownloadConfig,
    sleep: Callable[[float], None] | None = None,
) -> tuple[bytes, httpx.Response]:
    """GET ``spec.url`` with retries on transient failures; raises :class:`DownloadError`."""
    last_error: str = "no attempt made"
    for attempt in range(cfg.max_retries + 1):
        retry_after: str | None = None
        try:
            with client.stream("GET", spec.url) as response:
                if response.status_code in _RETRY_STATUS:
                    retry_after = response.headers.get("retry-after")
                    last_error = f"HTTP {response.status_code}"
                elif response.status_code >= 400:
                    raise DownloadError(f"{spec.url}: HTTP {response.status_code}")
                else:
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > cfg.max_bytes:
                            raise DownloadError(f"{spec.url} exceeds the {cfg.max_bytes} byte limit")
                        chunks.append(chunk)
                    return b"".join(chunks), response
        except httpx.TransportError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < cfg.max_retries:
            delay = _retry_delay(attempt, cfg, retry_after)
            logger.warning("Retrying %s in %.1fs (%s)", spec.url, delay, last_error)
            (sleep or time.sleep)(delay)
    raise DownloadError(f"{spec.url}: giving up after {cfg.max_retries + 1} attempts ({last_error})")


def download_source(
    entry: SourceEntry,
    raw_root: Path,
    cfg: DownloadConfig,
    blocked_domains: tuple[str, ...],
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> list[DownloadResult]:
    """Download every file of ``entry`` into ``raw_root/<source id>/``."""
    require_usage(entry, "download")
    for spec in entry.downloads:
        require_url_allowed(spec.url, blocked_domains)
    target_dir = raw_root / entry.id
    provenance_path = target_dir / "PROVENANCE.json"
    previous = {r["file"]: r for r in read_json(provenance_path)["files"]} if provenance_path.exists() else {}
    owns_client = client is None
    http = guarded(
        client
        or httpx.Client(headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout_s, follow_redirects=True),
        RequestGuard(blocked_domains),
    )
    robots = RobotsCache(http, cfg.user_agent)
    results: list[DownloadResult] = []
    try:
        for index, spec in enumerate(entry.downloads):
            path = target_dir / spec.filename
            if path.exists() and not force:
                digest = sha256_file(path)
                if spec.sha256 is None or digest == spec.sha256:
                    old = previous.get(spec.filename, {})
                    results.append(
                        DownloadResult(
                            source_id=entry.id,
                            url=spec.url,
                            final_url=old.get("final_url"),
                            file=spec.filename,
                            status="cached",
                            sha256=digest,
                            bytes=path.stat().st_size,
                            http_status=old.get("http_status"),
                            content_type=old.get("content_type"),
                            last_modified=old.get("last_modified"),
                            retrieved_utc=old.get("retrieved_utc", ""),
                        )
                    )
                    continue
            if cfg.respect_robots_txt and not robots.allowed(spec.url):
                raise DownloadError(f"robots.txt disallows {spec.url} for our user agent")
            if index:
                time.sleep(cfg.polite_delay_s)
            logger.info("Downloading %s", spec.url)
            data, response = _fetch(http, spec, cfg)
            atomic_write_bytes(path, data)
            digest = sha256_file(path)
            if spec.sha256 is not None and digest != spec.sha256:
                path.unlink(missing_ok=True)
                raise DownloadError(f"{spec.filename}: sha256 {digest} does not match pinned {spec.sha256}")
            results.append(
                DownloadResult(
                    source_id=entry.id,
                    url=spec.url,
                    final_url=str(response.url),
                    file=spec.filename,
                    status="downloaded",
                    sha256=digest,
                    bytes=len(data),
                    http_status=response.status_code,
                    content_type=response.headers.get("content-type"),
                    last_modified=response.headers.get("last-modified"),
                    retrieved_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                )
            )
    finally:
        if owns_client:
            http.close()
    write_json(
        provenance_path,
        {
            "source_id": entry.id,
            "license": entry.license,
            "permission_status": entry.permission_status.value,
            "files": [asdict(r) for r in results],
        },
    )
    return results
