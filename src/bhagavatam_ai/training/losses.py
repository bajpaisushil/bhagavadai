"""Memory-efficient causal language-modelling loss for very large vocabularies.

Qwen3.5 has a 248,320-entry vocabulary. Materialising float32 logits for one 2,048-token sequence
costs 2048 x 248320 x 4 B ~= 1.9 GiB, and the backward pass needs the same again. Computing the
cross-entropy chunk by chunk under activation checkpointing keeps only one chunk of logits alive at a
time, which is what makes longer sequences fit in 16 GB of unified memory.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _chunk_ce_sum(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="sum")


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 1024,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Mean next-token cross-entropy, numerically equivalent to Hugging Face's causal-LM loss.

    Args:
        hidden_states: final decoder states ``[batch, seq, hidden]``.
        lm_head_weight: output projection ``[vocab, hidden]`` (tied to the embeddings for Qwen3.5).
        labels: token ids ``[batch, seq]``; positions equal to ``ignore_index`` are excluded. Labels are
            shifted internally: position ``t`` predicts ``labels[t + 1]``.
        chunk_size: number of token positions per logits chunk.
    """
    if hidden_states.dim() != 3 or labels.dim() != 2:
        raise ValueError("expected hidden_states [B, T, H] and labels [B, T]")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            f"shape mismatch: hidden {tuple(hidden_states.shape)} vs labels {tuple(labels.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
    targets = labels[:, 1:].reshape(-1).to(hidden.device)
    num_valid = (targets != ignore_index).sum()

    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    needs_grad = torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad)
    for start in range(0, hidden.size(0), chunk_size):
        h = hidden[start : start + chunk_size]
        t = targets[start : start + chunk_size]
        if needs_grad:
            part = checkpoint(_chunk_ce_sum, h, lm_head_weight, t, ignore_index, use_reentrant=False)
        else:
            part = _chunk_ce_sum(h, lm_head_weight, t, ignore_index)
        total = total + part
    return total / num_valid.clamp(min=1).to(total.dtype)


def fused_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Same loss via ``torch.nn.functional.linear_cross_entropy`` (torch >= 2.13), which chunks
    internally and never materialises the full logits tensor."""
    options_cls = getattr(torch.nn, "LinearCrossEntropyOptions", None)
    if options_cls is None or not hasattr(F, "linear_cross_entropy"):
        raise RuntimeError("torch.nn.functional.linear_cross_entropy needs torch >= 2.13")
    hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
    targets = labels[:, 1:].reshape(-1).to(hidden.device)
    # Per-row losses reduced in float32: with bf16 inputs the op's own mean accumulates in bf16 and
    # was measured 1.5 nats (15 %) off at ~2k rows; acc_dtype=float32 did not change that.
    rows = F.linear_cross_entropy(
        hidden, lm_head_weight, targets, ignore_index=ignore_index, reduction="none", options=options_cls()
    )
    num_valid = (targets != ignore_index).sum().clamp(min=1)
    return rows.float().sum() / num_valid.to(torch.float32)


def causal_lm_loss(
    model: Any,
    input_ids: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    mode: str = "chunked",
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Next-token loss for a (possibly PEFT-wrapped) ``*ForCausalLM`` model.

    ``mode="standard"`` uses the model's own loss (materialises full fp32 logits). ``"chunked"`` and
    ``"fused"`` run the decoder, then compute the loss against the (tied) output projection without
    materialising all logits at once.
    """
    targets = input_ids if labels is None else labels
    if mode == "standard":
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets, use_cache=False)
        loss: torch.Tensor = output.loss
        return loss
    if mode not in {"chunked", "fused"}:
        raise ValueError(f"unknown loss mode {mode!r}")
    base: Any = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    if mode == "fused":
        return fused_causal_lm_loss(hidden, base.lm_head.weight, targets)
    return chunked_causal_lm_loss(hidden, base.lm_head.weight, targets, chunk_size=chunk_size)
