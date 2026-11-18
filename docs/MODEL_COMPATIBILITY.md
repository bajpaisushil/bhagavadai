# Qwen3.5-0.8B-Base: compatibility for MPS training

Status as of 2026-11-18. Each claim is tagged with where its evidence comes from:

- **[measured]**: run in this repository (reports in `reports/phase1/codespace-cpu/`, tests in `tests/`);
- **[source]**: read in the installed library source (transformers 5.23.0, peft 0.21.3, torch 2.15.0);
- **[researched]**: from documentation, issue trackers or benchmarks (links below);
- **[to verify on Mac]**: cannot be measured without the M5. The hardware check produces it.

## 1. What the checkpoint is

| Property | Value |
|---|---|
| Repo / revision | `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (pinned) |
| Checkpoint class | `Qwen3_5ForConditionalGeneration`, a vision-language model (12-block vision tower, 1 MTP layer) |
| Licence | Apache-2.0 |
| Text decoder | 24 layers = 6 × (3 × Gated DeltaNet + 1 × gated full attention); hidden 1024; FFN 3584 |
| Attention layers | 8 query / 2 KV heads, head_dim 256, partial RoPE (64 dims), output gate |
| DeltaNet layers | 16 heads × 128, depthwise causal conv (k=4), recurrent state kept in fp32 |
| Vocabulary | 248,320 (padded), tied input/output embeddings |
| Parameters (text) | 752.4 M = 254.3 M embedding + 498.1 M other **[measured]** |
| Weights on disk | 1.648 GB BF16, single safetensors file (SHA-256 verified) **[measured]** |

## 2. Loading only the text model **[measured] [source]**

`AutoModelForCausalLM.from_pretrained(...)` builds `Qwen3_5ForCausalLM` with `Qwen3_5TextConfig`.
`conversion_mapping.py` renames `model.language_model.*` to `model.*`. `_keys_to_ignore_on_load_unexpected`
silently drops `model.visual.*` and `mtp.*`.

Because a silent mapping failure would leave layers randomly initialised, `modeling/loader.py` makes
loading strict:

- missing, shape-mismatched or non-allow-listed unexpected keys raise `ModelIntegrityError`;
- six tensors (embeddings, a DeltaNet projection, `A_log`, an attention projection, the last MLP, the
  final norm) are compared **byte for byte** with the checkpoint file;
- the architecture (class, layers, vocabulary, tied embeddings) is checked against `configs/model.yaml`.

All checks pass: 0 missing, 0 unexpected, 6/6 spot checks equal, `lm_head` tied.

## 3. Kernels and the MPS path

| Component | CUDA fast path | Apple Silicon |
|---|---|---|
| Chunked gated delta rule (training/prefill) | `flash-linear-attention` (Triton) | `torch_chunk_gated_delta_rule` reference PyTorch **[source]** |
| Recurrent delta rule (decode) | `fla` fused kernel | `torch_recurrent_gated_delta_rule` **[source]** |
| Causal conv1d | `causal-conv1d` (CUDA) | `F.conv1d(groups=C)` **[source]** |
| Attention | FlashAttention-2 / SDPA | SDPA with `enable_gqa=True` (torch ≥ 2.13 on MPS) **[source]** |

The reference path is correct and differentiable: LoRA training through it reduced the loss from
2.44 to 0.11 in 4 sanity steps **[measured, CPU]**. It is slower than the CUDA kernels. A maintainer
calls the torch paths "already >10x slower than the kernels" **[researched]**.

`PYTORCH_ENABLE_MPS_FALLBACK` should not be needed. Every op on the text path has an MPS kernel in
torch 2.15 **[researched]**. The hardware check runs without it, so a missing op surfaces as
`unsupported` instead of silently running on the CPU.

### Op-level checks (`bhagavatam hardware ops`)

The op checks run the exact functions the model uses: the transformers delta-rule functions themselves,
depthwise causal conv + SiLU, SDPA 8q/2kv head_dim 256 with `enable_gqa` (causal and masked), gated
RMSNorm, cross-entropy and embedding backward over the full 248,320 vocabulary, fused AdamW, autocast
and sampling. Each runs forward and backward and is compared with a float32 CPU reference.

- On CPU, every check passes in fp32 and bf16 **[measured]**.
- **fp16 fails the chunked delta-rule backward** (relative error 2.2e-2 > 2e-2) because small
  gradients underflow; bf16 passes (7.5e-3) **[measured]**.

## 4. Precision

- Use **bf16** weights on MPS (supported on macOS ≥ 14), with LoRA adapters in fp32. PEFT's default
  `autocast_adapter_dtype=True` does this. The DeltaNet reference path casts to fp32 internally anyway.
- **Never fp16** (see above; the model code itself warns that `A` can become `-inf` in fp16).
- fp32 on CPU: CPUs without AVX512-BF16 emulate bf16 very slowly (1.4 tok/s bf16 generation on the
  codespace vs ~4 tok/s fp32) **[measured]**.

## 5. Version pins and known bugs **[researched]**

| Pin | Why |
|---|---|
| `torch>=2.15` | M5-specific corruption of bf16/fp16 `F.linear` for >2-D inputs without bias, which is every Qwen3.5 projection (pytorch#180776, fixed ≥ 2.13). Wrong bf16 `lerp_` in AdamW on 2.14.x (pytorch#196067, fixed 2.15.0). Fast MPS `solve_triangular` used by the delta rule (pytorch PR #196900, 2.15). |
| `transformers>=5.23,<5.24` | Qwen3.5 arrived in 5.2. The DeltaNet fallback was rewritten with `solve_triangular` in 5.17. `dtype=` was ignored on the Auto path before 5.12. The next release renames cache kwargs (breaking). |
| `peft>=0.21.3` | Tested version. PEFT has no default LoRA targets for `qwen3_5`, so `target_modules` must be explicit. |

Also do not use `torch.compile` on MPS: it fails on `Qwen3_5GatedDeltaNet` with "number of constant
buffers exceeds maximum supported (31)" (pytorch#201864, open).

## 6. LoRA targets **[measured] [source]**

`nn.Linear` modules in the text decoder:

- DeltaNet: `in_proj_qkv` (1024→6144), `in_proj_z` (1024→2048), `in_proj_a`/`in_proj_b` (1024→16), `out_proj` (2048→1024)
- Attention: `q_proj` (1024→4096, query + output gate), `k_proj`/`v_proj` (1024→512), `o_proj` (2048→1024)
- MLP: `gate_proj`, `up_proj` (1024→3584), `down_proj` (3584→1024)

Configured targets (`configs/model.yaml`) are all of these except the 16-wide `in_proj_a/b`. That gives
10.22 M trainable parameters at r=16 (1.34 %). A unit test asserts every configured name exists in the
real architecture.

## 7. Memory on a 16 GB Mac

| Item | Size |
|---|---|
| Base weights, bf16 | ~1.5 GB |
| LoRA r=16 (fp32 weights + grads + AdamW) | ~0.16 GB |
| fp32 logits + gradient at 2,048 tokens if materialised | ~3.8 GB, **avoided** by chunked/fused CE |
| DeltaNet fallback activations without checkpointing | ~100–200 MiB per layer per 1–2K tokens [researched]; activation checkpointing is on |
| Full fine-tune, all params, fp32 AdamW | 11.2 GB static, infeasible |
| Full fine-tune, frozen embeddings | 8.4 GB static, borderline at best |

MPS budget: Metal's `recommendedMaxWorkingSetSize` is about 75 % of RAM (11.5–12.7 GB reported on
16 GB Macs). PyTorch's default high watermark is 1.7× that, which is more than physical RAM, so a
too-large job swaps rather than failing. The hardware check therefore sets
`torch.mps.set_per_process_memory_fraction(1.0)` and also flags swap growth and low free memory as
*memory pressure*. **LoRA is the practical training method on this machine.**

## 8. Throughput expectations

- No published benchmark covers Qwen3.5 on PyTorch MPS on an M5. The closest is Qwen3-0.6B full
  fine-tuning on an M5 Air (32 GB): 673 tok/s bf16 on PyTorch MPS and 1,655 tok/s on MLX
  (khooeee/training-mlx-vs-mps) **[researched]**.
- Expected for this project: **~300–600 tok/s sustained** (LoRA, bf16, checkpointing, DeltaNet torch
  fallback, after throttling). The fanless Air loses 13–23 % under sustained load **[researched]**.
  That is roughly 0.5–1 h per million training tokens. **[to verify on Mac]** with
  `bhagavatam hardware check --sustained-minutes 20`.
- On the codespace CPU (2 vCPU): ~10 tok/s, so CPU training is not viable **[measured]**.

## 9. Batching, padding and packing **[source] [researched]**

- **No packed sequences with document boundaries.** Without `fla`/`causal-conv1d`, transformers 5.23
  raises if given `cu_seq_lens`. `position_ids`-only packing leaks recurrent and conv state across
  documents in the 18 DeltaNet layers.
- Continued pretraining concatenates documents with `<|endoftext|>` and cuts fixed-length blocks.
  Cross-document continuity is acceptable for a single-domain corpus.
- SFT uses one conversation per row, right-padded to a multiple of 64. Fixed shapes avoid MPS
  graph-cache growth, and 64 matches the DeltaNet chunk size.
- `pad_token == eos_token` (`<|endoftext|>`). A standard collator would mask EOS out of the loss and the
  model would never learn to stop, so Phase 3 uses its own collator.

## 10. Tokenizer and chat template **[measured]**

- `Qwen3_5Tokenizer`, 248,077 entries. `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<think>` and
  `</think>` are single tokens.
- Their embeddings carry learned structure: cos(`<|im_end|>`,`<|endoftext|>`) = 0.52 and
  cos(`<think>`,`</think>`) = 0.33, versus ~0 ± 0.03 for random rows. `<|fim_pad|>` shares its
  embedding with padding rows, i.e. it is untrained.
- English costs ~1.2 tokens/word. IAST Sanskrit costs ~1.4 characters/token, about 3× more per
  character than English.
- The tokenizer applies NFC itself, but `ṁ`/`ṃ` tokenise differently, so the corpus normalises
  anusvara to `ṃ`.
- The chat template is ChatML with no default system prompt. The generation prompt ends with an
  **empty think block**: `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Earlier assistant turns are
  rendered without it.
- Therefore SFT renders each assistant turn exactly as at inference: one training sample per assistant
  turn, loss only on `{answer}<|im_end|>`.
- The base repo has no `generation_config.json`. Inference stops on both `<|im_end|>` and
  `<|endoftext|>`.

## 11. MLX as an alternative **[researched]**

- `mlx-lm` supports Qwen3.5 (≥ 0.30.7) with LoRA and full fine-tuning. However, its training path runs
  the gated delta rule as a per-token Python loop: there is no kernel VJP in stock mlx-lm (mlx-lm
  #1956, #2050). One report shows 1,628 ms/step vs 853 ms on torch-MPS for this model.
- MLX core gained a differentiable `gated_delta_update` in 0.32.4. It is used by the community
  `mlx-lm-lora` but not yet measured here. There is no official MLX → PEFT exporter.
- **Decision:** PyTorch MPS is the primary path. MLX is a possible later optimisation once its
  DeltaNet training path matures.

## 12. Open risks

1. MPS numerics for the DeltaNet path on M5 are unverified until the hardware check runs on the Mac.
   The check compares the loss and top-1 agreement with a float32 CPU reference.
2. Sustained throughput on the fanless Air, and whether 2,048-token sequences fit without swap.
3. Generation instability reports on MPS (DynamicCache NaN, transformers#47396). Keep in-training
   evaluation short and greedy, and cross-check on CPU.
4. Publishing format: a text-only save is `qwen3_5_text`. transformers, vLLM and llama.cpp load it;
   mlx-lm 0.32 does not.

### Key references

- transformers v5.23.0 `models/qwen3_5/modeling_qwen3_5.py`, `conversion_mapping.py`,
  `integrations/hub_kernels.py`, `docs/source/en/{apple_silicon,padding_free}.md`
- pytorch/pytorch #180776, #196067, PR #196900, #201864; transformers #47625, #49440, #49472, #47396
- peft #3718, #4130; fla-org/flash-linear-attention #113; ml-explore/mlx-lm #1956, #2050
- Apple, MacBook Air (M5) technical specifications (support.apple.com/126320); Notebookcheck M5 Air
  reviews
