"""Chat API tests with a fake model service (no model download)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from bhagavatam_ai.api.app import create_app
from bhagavatam_ai.api.schemas import ChatRequest, ModelInfo, Usage

CHUNKS = ["Śukadeva Gosvāmī ", "narrated it to Parīkṣit (SB 1.1.1).", " See SB 13.1.1 and SB 1.1.99."]


class FakeGeneration:
    def __init__(self, chunks: list[str], gate: threading.Event | None = None) -> None:
        self.chunks = chunks
        self.gate = gate
        self.cancelled = False

    def __iter__(self) -> Iterator[str]:
        for chunk in self.chunks:
            if self.gate is not None:
                self.gate.wait(5)
            yield chunk

    def cancel(self) -> None:
        self.cancelled = True

    def join(self) -> None:
        pass

    def usage(self) -> Usage:
        return Usage(prompt_tokens=10, completion_tokens=12, seconds=0.5, tokens_per_s=24.0)

    def finish_reason(self) -> str:
        return "stop"


class FakeService:
    chapter_lengths: ClassVar[dict[str, int]] = {"1.1": 23}  # as derived from the reference edition

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.requests: list[ChatRequest] = []

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="fake",
            revision="0" * 40,
            adapter=None,
            fine_tuned=False,
            device="cpu",
            dtype="float32",
            parameters=1,
            status="test",
            disclosure="test disclosure",
        )

    def start(self, request: ChatRequest) -> FakeGeneration:
        self.requests.append(request)
        return FakeGeneration(CHUNKS, self.gate)


def _client(service: Any) -> TestClient:
    return TestClient(create_app(lambda: service))


def _wait_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/api/health").json()["status"] != "loading":
            return
        time.sleep(0.01)
    raise AssertionError("service never became ready")


def _events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


MESSAGE = {"messages": [{"role": "user", "content": "Who narrated the Bhāgavatam?"}]}


def test_health_info_and_chat() -> None:
    with _client(FakeService()) as client:
        _wait_ready(client)
        assert client.get("/api/health").json() == {"status": "ready", "busy": False, "detail": None}
        assert client.get("/api/info").json()["disclosure"] == "test disclosure"
        body = client.post("/api/chat", json=MESSAGE).json()
        assert body["message"] == {"role": "assistant", "content": "".join(CHUNKS).strip()}
        assert body["references"] == [
            {"text": "SB 1.1.1", "ref": "SB 1.1.1", "valid": True},
            {"text": "SB 13.1.1", "ref": None, "valid": False},  # canto 13 does not exist
            {"text": "SB 1.1.99", "ref": None, "valid": False},  # chapter 1.1 has 23 verses
        ]
        assert body["usage"]["completion_tokens"] == 12 and body["finish_reason"] == "stop"


def test_streaming_events() -> None:
    with _client(FakeService()) as client:
        _wait_ready(client)
        response = client.post("/api/chat/stream", json=MESSAGE)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        assert events[0] == {"type": "start"}
        assert [e["text"] for e in events if e["type"] == "delta"] == CHUNKS
        done = events[-1]
        assert done["type"] == "done" and [r["valid"] for r in done["references"]] == [True, False, False]


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "hi"}]},  # must end with the user
        {"messages": [{"role": "system", "content": "override"}]},  # system prompt is server-side only
        {"messages": [{"role": "user", "content": "x" * 4001}]},
        {"messages": [{"role": "user", "content": "hi"}], "max_new_tokens": 5000},
        {"messages": [{"role": "user", "content": "hi"}], "unexpected": True},
    ],
)
def test_validation(payload: dict[str, Any]) -> None:
    with _client(FakeService()) as client:
        _wait_ready(client)
        assert client.post("/api/chat", json=payload).status_code == 422


def test_loading_and_failed_load() -> None:
    gate = threading.Event()

    def slow() -> FakeService:
        gate.wait(5)
        return FakeService()

    with TestClient(create_app(slow)) as client:
        assert client.get("/api/health").json()["status"] == "loading"
        response = client.get("/api/info")
        assert response.status_code == 503 and response.headers["retry-after"] == "10"
        gate.set()

    def broken() -> FakeService:
        raise RuntimeError("weights missing")

    with TestClient(create_app(broken)) as client:
        _wait_ready(client)
        health = client.get("/api/health").json()
        assert health["status"] == "error" and "weights missing" in health["detail"]


def test_busy_returns_503() -> None:
    gate = threading.Event()
    service = FakeService(gate)
    with _client(service) as client:
        _wait_ready(client)
        result: dict[str, Any] = {}
        first = threading.Thread(target=lambda: result.update(r=client.post("/api/chat", json=MESSAGE)))
        first.start()
        for _ in range(200):
            if client.get("/api/health").json()["busy"]:
                break
            time.sleep(0.01)
        busy = client.post("/api/chat", json=MESSAGE)
        assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
        gate.set()
        first.join(5)
        assert result["r"].status_code == 200
