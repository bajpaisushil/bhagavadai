"""Generation helpers shared by smoke tests, evaluation and the API."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from bhagavatam_ai.hardware.device import synchronize

CHAT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05

    def to_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return kwargs


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    new_tokens: int
    seconds: float
    tokens_per_s: float | None
    stopped_on_stop_token: bool


def stop_token_ids(tokenizer: Any, stop_tokens: Sequence[str] = CHAT_STOP_TOKENS) -> list[int]:
    ids = []
    for token in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            ids.append(token_id)
    return ids


def render_chat(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool = False
) -> str:
    """Render ``messages`` with the model's chat template, ending in an assistant prompt."""
    rendered: str = tokenizer.apply_chat_template(
        [dict(m) for m in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return rendered


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    params: GenerationParams,
    *,
    stop_tokens: Sequence[str] = CHAT_STOP_TOKENS,
) -> GenerationResult:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    stops = stop_token_ids(tokenizer, stop_tokens)
    with torch.inference_mode():
        start = time.perf_counter()
        output = model.generate(
            **inputs, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id, **params.to_generate_kwargs()
        )
        synchronize(device)
        seconds = time.perf_counter() - start
    new_ids = output[0, inputs.input_ids.shape[1] :].tolist()
    stopped = bool(new_ids) and new_ids[-1] in stops
    return GenerationResult(
        text=tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        prompt_tokens=int(inputs.input_ids.shape[1]),
        new_tokens=len(new_ids),
        seconds=round(seconds, 3),
        tokens_per_s=round(len(new_ids) / seconds, 2) if seconds > 0 else None,
        stopped_on_stop_token=stopped,
    )
