"""Request and response models of the chat API."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_MESSAGE_CHARS = 4000
MAX_MESSAGES = 40


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(ApiModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_new_tokens: int = Field(default=320, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ends_with_user(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must come from the user")
        return self


class Reference(ApiModel):
    text: str  # as written in the answer, e.g. "SB 1.2.6"
    ref: str | None  # canonical form when structurally valid
    valid: bool  # False for impossible citations such as "SB 13.1.1"


class Usage(ApiModel):
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    tokens_per_s: float | None


class ChatResponse(ApiModel):
    message: ChatMessage
    references: list[Reference]
    usage: Usage
    finish_reason: Literal["stop", "length", "cancelled"]


class ModelInfo(ApiModel):
    name: str
    revision: str
    adapter: str | None
    fine_tuned: bool
    device: str
    dtype: str
    parameters: int
    status: str
    disclosure: str


class Health(ApiModel):
    status: Literal["loading", "ready", "error"]
    busy: bool
    detail: str | None = None
