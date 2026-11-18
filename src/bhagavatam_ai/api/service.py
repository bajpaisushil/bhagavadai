"""Model service: loads the model once and runs one generation at a time.

Generation runs in a background thread and is streamed through ``TextIteratorStreamer``. A
``threading.Event`` stops it early (via a stopping criterion) when the HTTP client disconnects, so an
abandoned request releases the model promptly instead of generating to ``max_new_tokens``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from bhagavatam_ai.api.schemas import ChatRequest, ModelInfo, Reference, Usage
from bhagavatam_ai.data.reference import find_references
from bhagavatam_ai.inference.generate import render_chat, stop_token_ids
from bhagavatam_ai.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DISCLOSURE = (
    "Bhagavatam AI is an experimental AI model, not a sage, teacher or religious authority, and it is not "
    "affiliated with any religious organisation. It can be wrong: verify anything important against a "
    "trusted translation of the Śrīmad-Bhāgavatam. Conversations are processed only to generate replies "
    "and are not stored by the server."
)


class Generation(Protocol):
    def __iter__(self) -> Iterator[str]: ...
    def cancel(self) -> None: ...
    def join(self) -> None: ...
    def usage(self) -> Usage: ...
    def finish_reason(self) -> str: ...


class ChatService(Protocol):
    def info(self) -> ModelInfo: ...
    def start(self, request: ChatRequest) -> Generation: ...


def extract_references(text: str) -> list[Reference]:
    return [
        Reference(text=matched, ref=str(span) if span else None, valid=span is not None)
        for matched, span in find_references(text)
    ]


def build_messages(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(m.model_dump() for m in request.messages)]


@dataclass
class _TransformersGeneration:
    streamer: Any
    thread: threading.Thread
    cancelled: threading.Event
    prompt_tokens: int
    max_new_tokens: int
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None

    def __iter__(self) -> Iterator[str]:
        for chunk in self.streamer:
            if chunk:
                yield chunk
        self.finished = time.perf_counter()

    def cancel(self) -> None:
        self.cancelled.set()

    def join(self) -> None:
        self.thread.join()
        if self.finished is None:
            self.finished = time.perf_counter()

    def usage(self) -> Usage:
        seconds = (self.finished or time.perf_counter()) - self.started
        generated = int(getattr(self.streamer, "generated_tokens", 0))
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=generated,
            seconds=round(seconds, 3),
            tokens_per_s=round(generated / seconds, 2) if seconds > 0 and generated else None,
        )

    def finish_reason(self) -> str:
        if self.cancelled.is_set():
            return "cancelled"
        generated = int(getattr(self.streamer, "generated_tokens", 0))
        return "length" if generated >= self.max_new_tokens else "stop"


class TransformersChatService:
    """Serves the configured base model (plus an optional PEFT adapter from a later phase)."""

    def __init__(self, model_config: Path, *, device: str = "auto", adapter: Path | None = None) -> None:
        from bhagavatam_ai.config import load_model_config
        from bhagavatam_ai.hardware.device import dtype_for_device, select_device
        from bhagavatam_ai.modeling.loader import load_model, load_tokenizer, snapshot_dir

        self.cfg = load_model_config(model_config)
        self.device = select_device(device)  # type: ignore[arg-type]
        self.dtype = dtype_for_device(self.device, self.cfg.dtype)
        snapshot = snapshot_dir(self.cfg)
        self.tokenizer = load_tokenizer(self.cfg, snapshot=snapshot)
        model, report = load_model(self.cfg, self.device, self.dtype, snapshot=snapshot)
        self.parameters = report.num_parameters
        self.adapter = adapter
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        self.model = model.eval()
        self.stops = stop_token_ids(self.tokenizer)

    def info(self) -> ModelInfo:
        fine_tuned = self.adapter is not None
        return ModelInfo(
            name=self.cfg.name_or_path,
            revision=self.cfg.revision,
            adapter=self.adapter.name if self.adapter else None,
            fine_tuned=fine_tuned,
            device=self.device.type,
            dtype=str(self.dtype).removeprefix("torch."),
            parameters=self.parameters,
            status=(
                "Fine-tuned on the Śrīmad-Bhāgavatam."
                if fine_tuned
                else "Untrained base model (pre-training baseline): expect confident mistakes."
            ),
            disclosure=DISCLOSURE,
        )

    def start(self, request: ChatRequest) -> Generation:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class CountingStreamer(TextIteratorStreamer):  # type: ignore[misc]
            generated_tokens = 0

            def put(self, value: torch.Tensor) -> None:
                if not (self.skip_prompt and self.next_tokens_are_prompt):
                    self.generated_tokens += int(value.numel())
                super().put(value)

        class Cancelled(StoppingCriteria):  # type: ignore[misc]
            def __init__(self, event: threading.Event) -> None:
                self.event = event

            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                return torch.full(
                    (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
                )

        prompt = render_chat(self.tokenizer, build_messages(request))
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = CountingStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancelled = threading.Event()
        sampling = request.temperature > 0
        kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": sampling,
            "repetition_penalty": 1.05,
            "eos_token_id": self.stops,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([Cancelled(cancelled)]),
        }
        if sampling:
            kwargs.update(temperature=request.temperature, top_p=request.top_p)

        def run() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception:  # surface in logs; the streamer is ended so the client is released
                logger.exception("generation failed")
                streamer.end()

        thread = threading.Thread(target=run, name="generation", daemon=True)
        thread.start()
        return _TransformersGeneration(
            streamer=streamer,
            thread=thread,
            cancelled=cancelled,
            prompt_tokens=int(inputs.input_ids.shape[1]),
            max_new_tokens=request.max_new_tokens,
        )
