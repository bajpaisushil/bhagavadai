"""Model-level hardware check: load, numerics vs a CPU reference, generation speed, LoRA training.

Run this on the training machine (``bhagavatam hardware check``). It answers, with measurements
rather than assumptions:

* do the ops Qwen3.5 needs work on this backend, in which dtypes?
* does the model load intact and produce the same predictions as a float32 CPU reference?
* how fast is generation?
* can LoRA training run (forward + backward + optimizer step) and does the loss actually decrease?
* how much memory does training take at each sequence length, and where does it hit memory pressure?

The report ends with a verdict; continued pretraining should start only once ``ready_for_phase2``
is true on the target machine.
"""

from __future__ import annotations

import gc
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from bhagavatam_ai.config import HardwareCheckConfig, ModelConfig
from bhagavatam_ai.hardware.device import (
    collect_environment,
    dtype_for_device,
    dtype_from_name,
    empty_cache,
    is_out_of_memory,
    select_device,
    synchronize,
    thermal_state,
)
from bhagavatam_ai.hardware.memory import GiB, MemoryMonitor
from bhagavatam_ai.hardware.ops_check import descriptions, run_op_checks
from bhagavatam_ai.modeling.loader import load_model, load_tokenizer, snapshot_dir
from bhagavatam_ai.modeling.lora import build_lora_model, make_optimizer, trainable_parameters
from bhagavatam_ai.training.losses import causal_lm_loss
from bhagavatam_ai.utils.seed import seed_everything

logger = logging.getLogger(__name__)

# Written for this project (not scripture): English with IAST names, like the training data.
BENCHMARK_TEXT = (
    "The Bhāgavata Purāṇa is a Sanskrit text arranged in twelve cantos. Its frame story places the sage "
    "Śukadeva before King Parīkṣit on the bank of the Gaṅgā, while Sūta later retells the conversation to "
    "the sages assembled in the Naimiṣa forest. Readers meet figures such as Nārada, Dhruva, Prahlāda and "
    "Kṛṣṇa, and encounter recurring ideas: dharma, bhakti, the three guṇas, and the difference between the "
    "temporary body and the self. "
)
GENERATION_PROMPT = "The Bhāgavata Purāṇa is a Sanskrit text that"


def _token_block(tokenizer: Any, min_tokens: int) -> torch.Tensor:
    ids: list[int] = tokenizer(BENCHMARK_TEXT, add_special_tokens=False).input_ids
    repeats = math.ceil(min_tokens / len(ids))
    return torch.tensor([ids * repeats], dtype=torch.long)[:, :min_tokens]


def _next_token_stats(model: Any, ids: torch.Tensor, device: torch.device) -> dict[str, Any]:
    with torch.inference_mode():
        logits = model(input_ids=ids.to(device), use_cache=False).logits.float()
    target = ids[0, 1:].to(device)
    return {
        "loss": float(F.cross_entropy(logits[0, :-1], target)),
        "finite": bool(torch.isfinite(logits).all()),
        "top1": logits[0, :-1].argmax(-1).cpu(),
    }


def _cpu_reference(model_cfg: ModelConfig, ids: torch.Tensor) -> dict[str, Any]:
    """Float32 CPU predictions for the numerics comparison (loaded, used, then freed)."""
    model, _ = load_model(model_cfg, torch.device("cpu"), torch.float32, verify=False)
    try:
        return _next_token_stats(model, ids, torch.device("cpu"))
    finally:
        del model
        gc.collect()


def _numerics(
    model: Any,
    ids: torch.Tensor,
    device: torch.device,
    reference: dict[str, Any] | None,
    cfg: HardwareCheckConfig,
) -> dict[str, Any]:
    stats = _next_token_stats(model, ids, device)
    result: dict[str, Any] = {
        "tokens": int(ids.shape[1]),
        "loss": round(stats["loss"], 4),
        "finite": stats["finite"],
    }
    ok = stats["finite"] and math.isfinite(stats["loss"])
    if reference is not None:
        diff = abs(stats["loss"] - reference["loss"])
        agreement = float((stats["top1"] == reference["top1"]).float().mean())
        result.update(
            cpu_fp32_loss=round(reference["loss"], 4),
            abs_loss_diff=round(diff, 4),
            top1_agreement=round(agreement, 4),
        )
        ok = ok and diff <= cfg.numerics.max_abs_loss_diff and agreement >= cfg.numerics.min_top1_agreement
    result["ok"] = ok
    return result


def _generation(model: Any, tokenizer: Any, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    inputs = tokenizer(GENERATION_PROMPT, return_tensors="pt").to(device)
    common = {"do_sample": False, "pad_token_id": tokenizer.pad_token_id}
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=2, **common)  # warm-up (kernel compilation, caches)
        synchronize(device)
        start = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, **common
        )
        synchronize(device)
    seconds = time.perf_counter() - start
    new_tokens = output[0, inputs.input_ids.shape[1] :].tolist()
    return {
        "prompt": GENERATION_PROMPT,
        "new_tokens": len(new_tokens),
        "seconds": round(seconds, 3),
        "tokens_per_s": round(len(new_tokens) / seconds, 2) if seconds > 0 else None,
        "distinct_token_ratio": round(len(set(new_tokens)) / max(1, len(new_tokens)), 3),
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "ok": len(new_tokens) == max_new_tokens,
    }


@dataclass
class _Trainer:
    """A LoRA model plus its optimizer on one device: runs and times optimizer steps."""

    model: Any
    device: torch.device
    cfg: HardwareCheckConfig
    learning_rate: float
    params: list[torch.nn.Parameter] = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self) -> None:
        self.params = trainable_parameters(self.model)
        self.optimizer = make_optimizer(self.params, self.learning_rate, self.device)

    def step(self, ids: torch.Tensor, mode: str | None = None) -> float:
        tcfg = self.cfg.training
        loss = causal_lm_loss(self.model, ids, mode=mode or tcfg.loss, chunk_size=tcfg.loss_chunk_size)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    def measure(self, ids: torch.Tensor, *, mode: str) -> dict[str, Any]:
        """Time warm-up + measured steps on one batch; returns status, speed and memory."""
        tcfg = self.cfg.training
        entry: dict[str, Any] = {
            "seq_len": int(ids.shape[1]),
            "batch_size": int(ids.shape[0]),
            "loss_mode": mode,
        }
        try:
            with MemoryMonitor(self.device) as monitor:
                times: list[float] = []
                losses: list[float] = []
                for step in range(tcfg.warmup_steps + tcfg.measured_steps):
                    start = time.perf_counter()
                    loss = self.step(ids, mode)
                    synchronize(self.device)
                    monitor.sample_device()
                    if step >= tcfg.warmup_steps:
                        times.append(time.perf_counter() - start)
                        losses.append(loss)
            memory = monitor.summary()
            median = statistics.median(times)
            tokens_per_s = ids.numel() / median
            entry.update(
                status="ok",
                step_seconds=[round(t, 3) for t in times],
                median_step_seconds=round(median, 3),
                tokens_per_s=round(tokens_per_s, 1),
                hours_per_million_tokens=round(1e6 / tokens_per_s / 3600, 2),
                losses=[round(x, 4) for x in losses],
                memory=memory,
                memory_pressure=_under_pressure(memory, self.cfg),
            )
        except Exception as exc:
            status = "oom" if is_out_of_memory(exc) else "error"
            entry.update(status=status, detail=f"{type(exc).__name__}: {exc}"[:600])
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            empty_cache(self.device)
        return entry

    def sustained(self, ids: torch.Tensor) -> dict[str, Any]:
        """Train continuously for ``sustained_minutes``; report throughput per minute (throttling)."""
        deadline = time.perf_counter() + self.cfg.training.sustained_minutes * 60
        per_minute: list[float] = []
        window_start, window_tokens = time.perf_counter(), 0
        thermal_start = thermal_state()
        while time.perf_counter() < deadline:
            self.step(ids)
            synchronize(self.device)
            window_tokens += ids.numel()
            if time.perf_counter() - window_start >= 60:
                per_minute.append(round(window_tokens / (time.perf_counter() - window_start), 1))
                window_start, window_tokens = time.perf_counter(), 0
        first, last = (per_minute[0], per_minute[-1]) if per_minute else (None, None)
        return {
            "seq_len": int(ids.shape[1]),
            "minutes": self.cfg.training.sustained_minutes,
            "tokens_per_s_per_minute": per_minute,
            "throttling_drop_pct": round(100 * (1 - last / first), 1) if first and last else None,
            "thermal_start": thermal_start,
            "thermal_end": thermal_state(),
        }


def _under_pressure(memory: dict[str, Any], cfg: HardwareCheckConfig) -> bool:
    swap = memory.get("swap_growth_gb") or 0.0
    available = memory.get("min_system_available_gb")
    return swap > cfg.max_swap_growth_gb or (
        available is not None and available < cfg.min_system_available_gb
    )


def _overfit_sanity(
    model: Any, tokenizer: Any, device: torch.device, cfg: HardwareCheckConfig
) -> dict[str, Any]:
    """A few steps on one fixed batch: the loss must fall, proving gradients reach the adapters."""
    steps = cfg.training.sanity_overfit_steps
    if steps == 0:
        return {"skipped": True, "ok": True}
    trainer = _Trainer(model, device, cfg, learning_rate=1e-3)
    ids = _token_block(tokenizer, 128).to(device)
    losses = [trainer.step(ids) for _ in range(steps)]
    synchronize(device)
    finite = all(math.isfinite(x) for x in losses)
    decreased = finite and losses[-1] < losses[0] - 0.05
    return {
        "losses": [round(x, 4) for x in losses],
        "finite": finite,
        "decreased": decreased,
        "ok": decreased,
    }


def _training_benchmark(
    model: Any, tokenizer: Any, device: torch.device, cfg: HardwareCheckConfig
) -> dict[str, Any]:
    tcfg = cfg.training
    trainer = _Trainer(model, device, cfg, learning_rate=tcfg.learning_rate)
    block = _token_block(tokenizer, max(tcfg.seq_lens))
    sweep: list[dict[str, Any]] = []
    for seq_len in sorted(tcfg.seq_lens):
        entry = trainer.measure(block[:, :seq_len].repeat(tcfg.batch_size, 1).to(device), mode=tcfg.loss)
        sweep.append(entry)
        logger.info("train seq_len=%d -> %s", seq_len, {k: entry.get(k) for k in ("status", "tokens_per_s")})
        if entry["status"] != "ok" or entry["memory_pressure"]:
            break  # longer sequences will only be worse
    result: dict[str, Any] = {"sweep": sweep}
    fitting = [e for e in sweep if e["status"] == "ok" and not e["memory_pressure"]]
    if not fitting:
        return result
    best_len = max(e["seq_len"] for e in fitting)
    ids = block[:, :best_len].repeat(tcfg.batch_size, 1).to(device)
    comparisons = []
    for mode in tcfg.compare_loss_modes:
        if mode != tcfg.loss:
            entry = trainer.measure(ids, mode=mode)
            comparisons.append(entry)
            logger.info(
                "loss mode %s @%d -> %s",
                mode,
                best_len,
                {k: entry.get(k) for k in ("status", "tokens_per_s")},
            )
    result["loss_mode_comparison"] = comparisons
    if tcfg.sustained_minutes > 0:
        result["sustained"] = trainer.sustained(ids)
    return result


def theoretical_memory(
    num_params: int, num_embedding_params: int, lora_params: int, weight_bytes: int
) -> dict[str, float]:
    """Static memory (weights + grads + AdamW states), excluding activations, in GiB."""
    adam_fp32 = 16  # fp32 master weight + grad + two moments
    non_embedding = num_params - num_embedding_params
    return {
        "lora_static_gb": round((num_params * weight_bytes + lora_params * adam_fp32) / GiB, 2),
        "full_ft_all_params_gb": round(num_params * adam_fp32 / GiB, 2),
        "full_ft_frozen_embeddings_gb": round(
            (non_embedding * adam_fp32 + num_embedding_params * weight_bytes) / GiB, 2
        ),
    }


def _verdict(report: dict[str, Any], cfg: HardwareCheckConfig) -> dict[str, Any]:
    ops = report.get("op_checks", [])
    dtype = report["dtype"]
    relevant = [o for o in ops if o["dtype"] in {dtype, "float32"}]
    ops_ok = all(o["status"] == "pass" for o in relevant)
    train = report.get("training", {}).get("sweep", [])
    ok_runs = [t for t in train if t.get("status") == "ok" and not t.get("memory_pressure")]
    best = max(ok_runs, key=lambda t: t["seq_len"]) if ok_runs else None
    sanity_ok = bool(report.get("training_sanity", {}).get("ok"))
    verdict: dict[str, Any] = {
        "device": report["device"],
        "dtype": dtype,
        "op_checks_ok": ops_ok,
        "failed_ops": sorted(
            {f"{o['name']}[{o['dtype']}]:{o['status']}" for o in relevant if o["status"] != "pass"}
        ),
        "model_load_ok": report.get("load", {}).get("ok", False),
        "numerics_ok": report.get("numerics", {}).get("ok", False),
        "generation_ok": report.get("generation", {}).get("ok", False),
        "lora_training_ok": sanity_ok and best is not None,
        "max_seq_len_without_memory_pressure": best["seq_len"] if best else None,
        "tokens_per_s_at_max_seq_len": best["tokens_per_s"] if best else None,
        "hours_per_million_tokens_at_max_seq_len": best["hours_per_million_tokens"] if best else None,
    }
    comparisons = [
        c for c in report.get("training", {}).get("loss_mode_comparison", []) if c.get("status") == "ok"
    ]
    candidates = ([best] if best else []) + [c for c in comparisons if not c.get("memory_pressure")]
    if candidates:
        fastest = max(candidates, key=lambda c: c["tokens_per_s"])
        verdict["fastest_loss_mode"] = fastest["loss_mode"]
        verdict["fastest_tokens_per_s"] = fastest["tokens_per_s"]
    keys = ("op_checks_ok", "model_load_ok", "numerics_ok", "generation_ok", "lora_training_ok")
    verdict["ready_for_phase2"] = all(verdict[k] for k in keys) and report["device"] != "cpu"
    if report["device"] == "cpu":
        verdict["note"] = (
            "Ran on CPU: this validates the pipeline only. Run on the Apple Silicon Mac for the real verdict."
        )
    return verdict


def run_hardware_check(
    model_cfg: ModelConfig, cfg: HardwareCheckConfig, *, skip_training: bool = False
) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    dtype = dtype_for_device(device, model_cfg.dtype)
    if device.type == "mps" and cfg.mps_memory_fraction is not None:
        torch.mps.set_per_process_memory_fraction(cfg.mps_memory_fraction)
    report: dict[str, Any] = {
        "environment": collect_environment(),
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "config": cfg.model_dump(mode="json"),
        "thermal_before": thermal_state(),
    }
    logger.info("Hardware check on %s (%s)", device, report["dtype"])

    if cfg.op_checks.enabled:
        dtypes = [dtype_from_name(name) for name in cfg.op_checks.dtypes]
        report["op_checks"] = [vars(r) for r in run_op_checks(device, dtypes, seed=cfg.seed)]
        report["op_check_descriptions"] = descriptions()

    tokenizer = load_tokenizer(model_cfg, snapshot=snapshot_dir(model_cfg))
    ids = _token_block(tokenizer, 96)
    reference = None
    if device.type != "cpu" and cfg.numerics.compare_with_cpu_reference:
        logger.info("Computing float32 CPU reference predictions")
        reference = _cpu_reference(model_cfg, ids)

    with MemoryMonitor(device) as monitor:
        model, load_report = load_model(model_cfg, device, dtype)
        synchronize(device)
        monitor.sample_device()
    report["load"] = {"ok": True, **load_report.to_dict(), "memory": monitor.summary()}
    report["numerics"] = _numerics(model, ids, device, reference, cfg)

    with MemoryMonitor(device) as monitor:
        report["generation"] = _generation(model, tokenizer, device, cfg.generation.max_new_tokens)
        monitor.sample_device()
    report["generation"]["memory"] = monitor.summary()

    if cfg.training.enabled and not skip_training:
        peft_model = build_lora_model(
            model,
            cfg.training.lora,
            model_cfg.lora_target_modules,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
        )
        lora_params = sum(p.numel() for p in trainable_parameters(peft_model))
        report["lora"] = {
            "trainable_parameters": lora_params,
            "target_modules": list(model_cfg.lora_target_modules),
        }
        report["training_sanity"] = _overfit_sanity(peft_model, tokenizer, device, cfg)
        report["training"] = _training_benchmark(peft_model, tokenizer, device, cfg)
        report["thermal_after"] = thermal_state()
        report["theoretical_memory"] = theoretical_memory(
            load_report.num_parameters,
            load_report.num_embedding_parameters,
            lora_params,
            torch.finfo(dtype).bits // 8,
        )
    report["verdict"] = _verdict(report, cfg)
    return report
