# Runbook: Phase 1 checks on the MacBook Air M5

These checks decide whether continued pretraining (Phase 2) can start. They must run on the training
machine itself: the development codespace has no Apple GPU.

## Before you start

- macOS 26.x on Apple Silicon. Check with `uname -m`, which must print `arm64`.
- About 10 GB of free disk: model 1.7 GB, environment ~2 GB, plus reports and caches.
- **Plug in power, and turn off Low Power Mode.** Close memory-heavy apps (browsers, IDEs with
  indexers, Docker). macOS and WindowServer already use 4–5 GB of the 16 GB.
- Install uv: `brew install uv` (or see https://docs.astral.sh/uv/).

## Run

```bash
git clone https://github.com/bajpaisushil/bhagavadai && cd bhagavadai
make setup                             # Python 3.12 + locked deps (torch 2.15 with MPS)
scripts/mac_phase1.sh                  # ~15-30 minutes
```

Optional thermal measurement (adds 20 minutes of continuous training):

```bash
scripts/mac_phase1.sh --sustained-minutes 20
```

The script refuses to run under Rosetta, confirms that PyTorch sees MPS, and makes sure
`PYTORCH_ENABLE_MPS_FALLBACK` is unset. With it unset, unsupported ops fail visibly instead of silently
running on the CPU. It wraps the hardware check in `caffeinate` so the Mac neither sleeps nor
down-clocks the GPU through App Nap when the terminal is in the background.

## What runs, and what "good" looks like

| Step | Command | Good result |
|---|---|---|
| Environment | `bhagavatam env` | `torch.mps_available: true`, `python_is_arm64_native: true`, torch 2.15.x |
| Model download | `bhagavatam model download` | 12 files, all checksums verified |
| Inspection | `bhagavatam model inspect` | control tokens single; template suffix with empty think block |
| Hardware check | `bhagavatam hardware check --strict` | `verdict.ready_for_phase2: true` |
| Smoke test | `bhagavatam model smoke` | 4 generations, readable text (factually unreliable: that is the baseline) |
| Data | `bhagavatam data download` and `data build` | GRETIL: 14,099 records, `ok=True` |

### Reading the hardware verdict (`reports/hardware_check/latest.json` → `verdict`)

| Field | Meaning |
|---|---|
| `op_checks_ok`, `failed_ops` | every fp32/bf16 op matched the CPU reference (fp16 failures are expected and ignored) |
| `model_load_ok` | integrity checks passed on MPS |
| `numerics_ok` | loss within 0.08 of the CPU fp32 reference and ≥ 85 % top-1 agreement |
| `generation_ok` | all decoding logits were finite, and the cached decode agreed (≥ 90 %) with a no-cache forward pass over the same tokens |
| `lora_training_ok` | the loss fell during the sanity steps, at least one sequence length trained without memory pressure, and **no step produced NaN/inf** (a non-finite step is refused, never applied) |
| `checks_ok` | all functional checks above passed on this device. The command exits 2 when this is false. |
| `failed_stages` | stages that raised (e.g. out of memory) or were skipped. The rest of the report is still written. |
| `max_seq_len_without_memory_pressure` | longest of 256/512/1024/2048 that trained without swap growth or low free memory |
| `tokens_per_s_at_max_seq_len`, `hours_per_million_tokens_at_max_seq_len` | training speed, used to plan epochs |
| `fastest_loss_mode` | `chunked`, `fused` or `standard` cross-entropy, whichever was fastest without memory pressure |

**Please share `reports/hardware_check/latest.json`** (and `reports/model_smoke/latest.json`). The
reports contain no hostname, user name or file paths.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `MPS requested but unavailable` / `mps_available: false` | Python is x86_64 under Rosetta. Use the uv-managed Python from `make setup`, not a Rosetta Homebrew install. |
| `unsupported` in `failed_ops` | An op has no MPS kernel in this torch version. Report the op name. Do **not** paper over it with `PYTORCH_ENABLE_MPS_FALLBACK=1`, because that moves the op to the CPU on every step. |
| `numeric_mismatch` for bf16 | A possible MPS numerical bug. Confirm torch is 2.15.x (earlier versions corrupt bf16 `F.linear` on M5). |
| `status: oom` or `memory_pressure: true` at small lengths | Other apps are using memory. Close them and rerun. The check caps MPS at Metal's recommended working set (`mps_memory_fraction: 1.0`) so it fails fast instead of swapping. |
| Very slow steps, or a large `training.sustained.throttling_drop_pct` | Thermal throttling. Expected on a fanless Air under sustained load. Keep the Mac on a hard surface. Budget for the sustained figure. (The `pmset` thermal fields in the report are informational only: on Apple Silicon they do not describe GPU throttling.) |
| Download errors | Re-run; downloads resume from cache. A Hugging Face token (`HF_TOKEN`) raises rate limits but is not required. |

## After the check

If `ready_for_phase2` is true, Phase 2 is configured from the measured numbers: sequence length,
loss mode and the tokens/s budget per epoch. If it is false, the `failed_ops` and memory fields say
whether the fix is a configuration change (shorter sequences, fewer LoRA targets) or a platform issue.
