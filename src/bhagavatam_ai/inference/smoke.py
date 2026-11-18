"""Small inference smoke test of the *base* model.

This is not an evaluation: it proves the inference path works end to end (tokenise, template,
generate, stop, decode) and records a pre-training baseline. Outputs are expected to be imperfect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Final

from bhagavatam_ai.inference.generate import GenerationParams, generate, render_chat
from bhagavatam_ai.prompts import SYSTEM_PROMPT

SMOKE_PROMPTS: Final[tuple[dict[str, Any], ...]] = (
    {"id": "completion-en", "mode": "completion", "prompt": "The Śrīmad-Bhāgavatam is"},
    {
        "id": "completion-iast",
        "mode": "completion",
        "prompt": "In the Bhāgavata Purāṇa, King Parīkṣit asked Śukadeva",
    },
    {
        "id": "chat-narrator",
        "mode": "chat",
        "question": "Who narrated the Śrīmad-Bhāgavatam to King Parīkṣit?",
    },
    {
        "id": "chat-philosophy",
        "mode": "chat",
        "question": "Why does a person suffer even when they try to do good?",
    },
)


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """Share of word n-grams that are repeats; high values indicate degenerate looping output."""
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return round(sum(c - 1 for c in counts.values()) / len(grams), 3)


def run_smoke(model: Any, tokenizer: Any, *, max_new_tokens: int = 64) -> dict[str, Any]:
    params = GenerationParams(max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    results = []
    for case in SMOKE_PROMPTS:
        if case["mode"] == "chat":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case["question"]},
            ]
            prompt = render_chat(tokenizer, messages)
        else:
            prompt = case["prompt"]
        result = generate(model, tokenizer, prompt, params)
        results.append(
            {
                "id": case["id"],
                "mode": case["mode"],
                "input": case.get("question") or case.get("prompt"),
                **asdict(result),
                "repeated_trigram_ratio": repeated_ngram_ratio(result.text),
                "ok": result.new_tokens > 0 and bool(result.text),
            }
        )
    return {"params": asdict(params), "results": results, "all_ok": all(r["ok"] for r in results)}
