from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from bhagavatam_ai.config import DownloadConfig
from bhagavatam_ai.data import download as dl
from bhagavatam_ai.data.download import DownloadError, download_source
from bhagavatam_ai.data.manifest import LicenseGateError
from bhagavatam_ai.data.schema import SourceEntry
from bhagavatam_ai.utils.io import read_json

PAYLOAD = b"<TEI>fixture</TEI>"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
CFG = DownloadConfig(max_retries=2, backoff_s=0.0, polite_delay_s=0.0, max_bytes=1024)


def _entry(url: str = "https://example.org/a.xml", sha: str | None = SHA, **overrides: Any) -> SourceEntry:
    data: dict[str, Any] = {
        "id": "src",
        "title": "T",
        "author_translator": "A",
        "language": "sa",
        "content_roles": ["sanskrit_text"],
        "edition": "E",
        "publication_years": "2020",
        "copyright_holder": "x",
        "license": "CC0",
        "attribution_requirements": "none",
        "commercial_use": "allowed",
        "ml_training": "allowed",
        "permission_status": "open_license",
        "usage": {"download": True, "metadata": True},
        "evidence": [{"url": "https://example.org", "quote": "q", "retrieved_on": "2026-11-18"}],
        "downloads": [{"url": url, "filename": "a.xml", "format": "xml", "sha256": sha}],
        "reviewed_on": "2026-11-18",
    }
    data.update(overrides)
    return SourceEntry.model_validate(data)


class Server:
    """Scripted responses for httpx.MockTransport; records requested paths."""

    def __init__(self, responses: list[httpx.Response], robots: str = "") -> None:
        self.responses = responses
        self.robots = robots
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=self.robots)
        return self.responses.pop(0)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), headers={"User-Agent": CFG.user_agent})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    return sleeps


def test_download_writes_file_and_provenance(tmp_path: Path) -> None:
    server = Server([httpx.Response(200, content=PAYLOAD, headers={"last-modified": "Thu, 10 Sep 2020"})])
    results = download_source(_entry(), tmp_path, CFG, (), client=server.client())
    assert results[0].status == "downloaded" and results[0].sha256 == SHA
    assert (tmp_path / "src" / "a.xml").read_bytes() == PAYLOAD
    provenance = read_json(tmp_path / "src" / "PROVENANCE.json")
    assert provenance["files"][0]["last_modified"] == "Thu, 10 Sep 2020"
    assert server.paths == ["/robots.txt", "/a.xml"]


def test_cached_file_is_not_refetched(tmp_path: Path) -> None:
    download_source(
        _entry(), tmp_path, CFG, (), client=Server([httpx.Response(200, content=PAYLOAD)]).client()
    )
    server = Server([])
    results = download_source(_entry(), tmp_path, CFG, (), client=server.client())
    assert results[0].status == "cached" and server.paths == []
    assert read_json(tmp_path / "src" / "PROVENANCE.json")["files"][0]["status"] == "cached"


def test_checksum_mismatch_removes_file(tmp_path: Path) -> None:
    server = Server([httpx.Response(200, content=b"tampered")])
    with pytest.raises(DownloadError, match="does not match pinned"):
        download_source(_entry(), tmp_path, CFG, (), client=server.client())
    assert not (tmp_path / "src" / "a.xml").exists()


def test_retries_transient_errors_honouring_retry_after(tmp_path: Path, no_sleep: list[float]) -> None:
    server = Server(
        [
            httpx.Response(503),
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, content=PAYLOAD),
        ]
    )
    assert download_source(_entry(), tmp_path, CFG, (), client=server.client())[0].status == "downloaded"
    assert no_sleep[-1] == 7.0 and len(no_sleep) == 2


def test_gives_up_after_retries(tmp_path: Path) -> None:
    server = Server([httpx.Response(500)] * 3)
    with pytest.raises(DownloadError, match="giving up after 3 attempts"):
        download_source(_entry(), tmp_path, CFG, (), client=server.client())


def test_client_errors_fail_fast(tmp_path: Path) -> None:
    server = Server([httpx.Response(404)])
    with pytest.raises(DownloadError, match="HTTP 404"):
        download_source(_entry(), tmp_path, CFG, (), client=server.client())


def test_size_limit(tmp_path: Path) -> None:
    server = Server([httpx.Response(200, content=b"x" * 2048)])
    with pytest.raises(DownloadError, match="byte limit"):
        download_source(_entry(sha=None), tmp_path, CFG, (), client=server.client())


def test_robots_txt_disallow(tmp_path: Path) -> None:
    server = Server([], robots="User-agent: *\nDisallow: /")
    with pytest.raises(DownloadError, match=r"robots\.txt disallows"):
        download_source(_entry(), tmp_path, CFG, (), client=server.client())


def test_blocked_domain_and_licence_gate(tmp_path: Path) -> None:
    with pytest.raises(LicenseGateError, match="blocklist"):
        download_source(
            _entry(url="https://vedabase.io/a.xml"),
            tmp_path,
            CFG,
            ("vedabase.io",),
            client=Server([]).client(),
        )
    no_download = _entry(usage={"metadata": True}, downloads=[])
    with pytest.raises(LicenseGateError, match=r"does not permit 'download'"):
        download_source(no_download, tmp_path, CFG, (), client=Server([]).client())


def test_transport_errors_are_retried(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, content=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert download_source(_entry(), tmp_path, CFG, (), client=client)[0].status == "downloaded"
    assert calls["n"] == 2
