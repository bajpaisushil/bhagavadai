"""LoRA adapter wiring shared by the hardware benchmark and the training phases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel

    from bhagavatam_ai.config import LoraSettings


def enable_activation_checkpointing(model: Any, *, every_n_layers: int = 1) -> None:
    """Enable non-reentrant activation checkpointing on a (possibly PEFT-wrapped) model."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    kwargs = {"use_reentrant": False}
    if hasattr(base, "activation_checkpointing_enable"):  # transformers >= 5.x
        base.activation_checkpointing_enable(
            activation_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers
        )
    else:  # pragma: no cover - older transformers
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    base.config.use_cache = False


def build_lora_model(
    model: PreTrainedModel,
    settings: LoraSettings,
    target_modules: Sequence[str],
    *,
    gradient_checkpointing: bool,
) -> Any:
    """Wrap ``model`` with LoRA adapters (kept in float32 by PEFT for stability) and set train mode."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    if gradient_checkpointing:
        enable_activation_checkpointing(peft_model)
    peft_model.train()
    return peft_model


def trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(
    params: Sequence[torch.nn.Parameter], lr: float, device: torch.device
) -> torch.optim.Optimizer:
    """AdamW, fused on accelerators when the backend supports it (falls back to foreach)."""
    import torch

    if device.type in {"mps", "cuda"}:
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, fused=True)
        except (RuntimeError, ValueError):
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=True)
