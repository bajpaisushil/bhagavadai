"""FastAPI application: health/info endpoints and chat (JSON or server-sent events).

The model loads in a background thread at start-up, so ``/api/health`` answers immediately with
``loading``. One generation runs at a time; a concurrent request gets HTTP 503 with ``Retry-After``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from bhagavatam_ai.api.schemas import ChatMessage, ChatRequest, ChatResponse, Health, ModelInfo
from bhagavatam_ai.api.service import ChatService, Generation, extract_references

logger = logging.getLogger(__name__)
_DONE = object()


@dataclass
class ApiSettings:
    model_config: Path = Path("configs/model.yaml")
    device: str = "auto"
    adapter: Path | None = None
    cors_origins: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> ApiSettings:
        adapter = os.environ.get("BHAGAVATAM_ADAPTER")
        origins = os.environ.get("BHAGAVATAM_CORS_ORIGINS", "")
        return cls(
            model_config=Path(os.environ.get("BHAGAVATAM_MODEL_CONFIG", "configs/model.yaml")),
            device=os.environ.get("BHAGAVATAM_DEVICE", "auto"),
            adapter=Path(adapter) if adapter else None,
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )


class _State:
    def __init__(self) -> None:
        self.service: ChatService | None = None
        self.error: str | None = None
        self.lock = asyncio.Lock()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ready(state: _State) -> ChatService:
    if state.service is None:
        raise HTTPException(
            503, detail=state.error or "The model is still loading.", headers={"Retry-After": "10"}
        )
    return state.service


def _ensure_idle(state: _State) -> None:
    if state.lock.locked():
        raise HTTPException(
            503,
            detail="The model is answering another question. Please retry shortly.",
            headers={"Retry-After": "5"},
        )


def _register_status_routes(app: FastAPI, state: _State) -> None:
    @app.get("/api/health", response_model=Health)
    async def health() -> Health:
        if state.error:
            return Health(status="error", busy=False, detail=state.error)
        if state.service is None:
            return Health(status="loading", busy=False)
        return Health(status="ready", busy=state.lock.locked())

    @app.get("/api/info", response_model=ModelInfo)
    async def info() -> ModelInfo:
        return _ready(state).info()


async def _finish(generation: Generation) -> None:
    generation.cancel()
    await run_in_threadpool(generation.join)


def _register_chat_routes(app: FastAPI, state: _State) -> None:
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        service = _ready(state)
        _ensure_idle(state)
        async with state.lock:
            generation = await run_in_threadpool(service.start, body)
            try:
                text = "".join(await run_in_threadpool(list, generation)).strip()
            finally:
                await _finish(generation)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=text or "…"),
            references=extract_references(text),
            usage=generation.usage(),
            finish_reason=generation.finish_reason(),
        )

    @app.post("/api/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        service = _ready(state)
        _ensure_idle(state)

        async def events() -> AsyncIterator[str]:
            async with state.lock:
                generation = await run_in_threadpool(service.start, body)
                parts: list[str] = []
                iterator = iter(generation)
                try:
                    yield _sse({"type": "start"})
                    while (chunk := await run_in_threadpool(next, iterator, _DONE)) is not _DONE:
                        parts.append(str(chunk))
                        yield _sse({"type": "delta", "text": chunk})
                        if await request.is_disconnected():
                            break
                    text = "".join(parts).strip()
                    yield _sse(
                        {
                            "type": "done",
                            "references": [r.model_dump() for r in extract_references(text)],
                            "usage": generation.usage().model_dump(),
                            "finish_reason": generation.finish_reason(),
                        }
                    )
                except Exception as exc:  # reported in-band: the HTTP status is already 200
                    logger.exception("streaming failed")
                    yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    await _finish(generation)

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


def create_app(service_factory: Callable[[], ChatService], settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    state = _State()

    def load() -> None:
        try:
            state.service = service_factory()
            logger.info("model ready")
        except Exception as exc:  # reported through /api/health
            logger.exception("model failed to load")
            state.error = f"{type(exc).__name__}: {exc}"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        threading.Thread(target=load, name="model-loader", daemon=True).start()
        yield

    app = FastAPI(
        title="Bhagavatam AI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )
    _register_status_routes(app, state)
    _register_chat_routes(app, state)
    app.state.chat_state = state  # exposed for tests and diagnostics
    return app


def create_default_app() -> FastAPI:
    """Entry point for ``uvicorn bhagavatam_ai.api.app:create_default_app --factory``."""
    from bhagavatam_ai.api.service import TransformersChatService
    from bhagavatam_ai.paths import ProjectPaths

    settings = ApiSettings.from_env()
    paths = ProjectPaths.discover()
    model_config = paths.resolve(settings.model_config)
    adapter = paths.resolve(settings.adapter) if settings.adapter else None
    return create_app(
        lambda: TransformersChatService(model_config, device=settings.device, adapter=adapter), settings
    )
