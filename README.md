# Bhagavatam AI

An English conversational language model specialised in the Śrīmad-Bhāgavatam, built by **training
model weights** (continued pretraining, then supervised fine-tuning) on properly licensed text, not by
retrieval. Base model: [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base),
pinned to revision `dc7cdfe2`. Target training machine: Apple Silicon MacBook Air M5 (16 GB) via
PyTorch MPS.

> **Status: Phase 1 (foundation) complete, plus an early web demo.** Model loading, integrity checks,
> tokenizer inspection, hardware and MPS compatibility checks, inference smoke test, the source/licence
> manifest, the licence-gated data pipeline, and a chat API with a web UI are implemented and tested.
> **Training has not started.** It is gated on (a) the hardware check passing on the Mac and (b) the
> English corpus being built from M. N. Dutt's 1895–96 translation. That translation is now verified
> as public domain everywhere; its parser is Phase 1b.

The model will identify itself as *Bhagavatam AI*. It teaches in a calm, respectful, sage-like style,
but never claims to be Śukadeva Gosvāmī, Sūta Gosvāmī, Prabhupāda or any other person.

## Quick start (Apple Silicon Mac)

```bash
brew install uv                  # Python package/environment manager
git clone https://github.com/bajpaisushil/bhagavadai && cd bhagavadai
make setup                       # locked deps (Python 3.12, torch 2.15 with MPS) + git hooks
scripts/mac_phase1.sh            # all Phase 1 checks on the Mac (~15-30 min)
```

`scripts/mac_phase1.sh` verifies native arm64 Python and MPS availability, downloads and checksums the
model, and runs the hardware check under `caffeinate` (so App Nap cannot down-clock the GPU). It then
runs the inference smoke test and the data pipeline. The decisive output is
`reports/hardware_check/latest.json` → `verdict.ready_for_phase2`. Add `--sustained-minutes 20` to
measure thermal throttling on the fanless Air.

Without `make`/`uv`: `python3 -m venv .venv && .venv/bin/pip install -e ".[ml,dev]"` (Python ≥ 3.11).

## Web demo

A FastAPI backend (streaming over server-sent events) and a Next.js + TypeScript + Tailwind chat UI. It
currently serves the **untrained base model**, which is labelled as such in the UI, so expect fluent but
unreliable answers. After training, it serves the fine-tuned model via `bhagavatam serve --adapter <dir>`.

```bash
make setup          # Python deps, once
make demo           # API on :8000 + web UI on :3000 (installs the UI's npm deps on first run)
```

Open http://localhost:3000. In **GitHub Codespaces**, open the *Ports* tab and use *Open in Browser* on
port 3000; the UI proxies `/api` to the backend, so only that one port is needed. The model loads in
the background (~20 s); the badge shows "Loading model…" until it is ready. Generation is ~3–4
tokens/s on a 2-vCPU codespace and much faster on Apple Silicon.

Run the pieces separately with `make api` and `make web`. Interactive API docs are at
http://localhost:8000/api/docs. Endpoints: `GET /api/health`, `GET /api/info`, `POST /api/chat`, and
`POST /api/chat/stream`. The server runs one generation at a time (a concurrent request gets a 503 with
`Retry-After`) and stops generating if the client disconnects. Conversations are stored only in the
browser.

## Commands

| Command | What it does |
|---|---|
| `bhagavatam env` | OS, chip, memory, library versions (no hostname, user name or paths) |
| `bhagavatam model download` | Download the pinned snapshot; verify every file's SHA-256 against the Hub |
| `bhagavatam model inspect` | Tokenizer, chat template, architecture, control-token embedding diagnostics |
| `bhagavatam model smoke` | Small end-to-end generation test of the base model (pre-training baseline) |
| `bhagavatam hardware ops` | Fast op-level checks of the exact kernels Qwen3.5 uses, against a CPU reference |
| `bhagavatam hardware check` | Ops, load + integrity, numerics vs CPU, generation speed, LoRA training memory/speed sweep, verdict |
| `bhagavatam data manifest` | Validate and list the source/licence manifest |
| `bhagavatam data download` | Download raw files, only for sources whose licence permits it |
| `bhagavatam data build` | Parse, normalise, validate, then build the licence-gated training corpus |
| `bhagavatam serve` | Chat API used by the web demo (`--adapter` to serve a fine-tuned model later) |

Every command writes a JSON report to `reports/<command>/<run_id>.json` (and `latest.json`) with the git
commit and package version. Exit codes: `0` success, `1` error, `2` checks ran but failed.

## What Phase 1 established

Measured in this repository, not assumed:

- **Text-only loading is exact.** `AutoModelForCausalLM` builds `Qwen3_5ForCausalLM` (752.4 M parameters:
  254.3 M embedding + 498.1 M other). It has 0 missing and 0 unexpected keys; the vision tower and MTP
  head are deliberately skipped. Sampled tensors match the checkpoint byte for byte, and loading fails
  hard if any of this changes.
- **Architecture:** 18 Gated DeltaNet + 6 gated-attention layers. The DeltaNet CUDA kernels
  (`flash-linear-attention`, `causal-conv1d`) do not exist for Apple Silicon. transformers falls back to
  a reference PyTorch implementation, which is correct and trainable but slower. LoRA through that
  path works: a sanity run's loss fell from 2.44 to 0.11 in 4 steps.
- **Use bf16, never fp16.** In the op checks the DeltaNet backward pass exceeds tolerance in fp16
  (gradient underflow) and passes in bf16.
- **Vocabulary of 248,320 tokens.** Full fp32 logits cost about 1.9 GiB per 2,048-token sequence, so
  training uses a chunked or fused cross-entropy that never materialises them. Both match the standard
  loss to about 1e-8.
- **Tokenizer:** the chat control tokens are single tokens with trained embeddings. IAST Sanskrit
  costs about 1.4 characters per token, roughly 3x English. The tokenizer applies NFC itself, but
  `ṁ` and `ṃ` tokenise differently, so the corpus is normalised to `ṃ`.
- **Baseline behaviour:** the base model is fluent but hallucinates. For example, it says the
  Bhāgavatam was "narrated by Lord Caitanya to King Parikshit. This is explicitly stated in the first
  canto", which is wrong and fabricates a scriptural claim. This is the behaviour training and
  evaluation must fix (`reports/model_smoke/`).

## Data and licensing policy

The policy is enforced in code, not only documented:

- `data/manifests/sources.yaml` records, for every source, the title, translator, copyright holder,
  licence, attribution and commercial-use terms, ML-training permission, permission status, and
  evidence (URL + exact quote + date). Pydantic validation rejects inconsistent entries. For example,
  `training_text: true` is only allowed for `public_domain`, `open_license` or `permission_granted`.
- Every stage calls a licence gate first. Excluded or permission-required sources cannot be
  downloaded, parsed or trained on. A domain blocklist (BBT / vedabase and sites that prohibit AI
  ingestion) also blocks such URLs even if one is added to the manifest by mistake.
- **Prabhupāda / BBT translations and purports are excluded.** They are copyrighted, and we have no
  permission covering ML training, weight distribution and a public demo.
- Raw and processed texts are never committed (`data/raw`, `data/interim`, `data/processed` are
  git-ignored). They are reproducible from the manifest, and checksums are pinned.
- The split is leakage-free by construction: whole chapters are assigned from a fixed seed, identically
  for every source. A chapter cannot reach training through one translation and validation through
  another. Near-duplicate detection (MinHash) checks for cross-split leakage.

**Sanskrit (GRETIL), current status:** the complete text is parsed: 14,099 verse records, 335 chapters
and 1,332 speaker attributions. Four file defects are repaired and flagged: duplicated half-verses,
mislabelled IDs, double escaping and stray characters. Its licence is CC BY-NC-SA 4.0, and research
found evidence that about two-thirds of it derives from the BBT romanised edition. It is therefore
used **only for non-expressive facts** (chapter and verse structure, speaker labels) until provenance
is resolved.

## Project layout

```
configs/               model.yaml (pinned model, dtype policy, integrity checks, LoRA targets),
                       hardware.yaml (Mac), hardware.cpu-smoke.yaml (CI), data.yaml
data/manifests/        sources.yaml - source + licence manifest (evidence required)
data/reference/        speakers.yaml (reviewed speaker table), expected_structure.json (verse counts)
src/bhagavatam_ai/
  cli.py               `bhagavatam` command
  config.py            typed, strict (extra="forbid") YAML configuration
  data/                manifest gate, download, parsers/, normalize/ (IAST<->Devanagari, OCR, speakers),
                       validate, dedup (MinHash), split (chapter-level), pipeline
  hardware/            device + environment, memory (unified-memory aware), op checks, benchmark
  modeling/            integrity-verified loader, LoRA wiring, tokenizer inspection
  training/            memory-efficient losses (Phase 2+: continued pretraining, SFT)
  inference/           generation helpers, smoke test
  api/                 FastAPI app: health/info, chat (JSON + SSE streaming), cancellation on disconnect
frontend/              Next.js 16 + React 19 + Tailwind 4 chat UI (proxies /api to the backend)
tests/unit/            fast tests (run in CI); tiny random Qwen3.5 for training-path tests
tests/integration/     real-model tests (opt-in: --run-model)
scripts/mac_phase1.sh  one-command Phase 1 run on the Mac
```

## Development

```bash
make check        # ruff lint + format check, mypy --strict, pytest with coverage (what CI runs)
make test-model   # integration tests against the real model (~1.7 GB download, ~5 GB RAM)
make help         # all targets
```

CI (GitHub Actions) runs lint, types and manifest validation, plus tests on Python 3.11, 3.12 and 3.13.
The real-model integration job runs weekly and on manual dispatch. Dependencies are locked in `uv.lock`
for both Apple Silicon (MPS-enabled torch from PyPI) and Linux (CPU torch wheels).

## Roadmap

1. **Phase 1 - foundation** (this release): model, tokenizer, hardware checks, manifest, data pipeline.
2. **Phase 2 - continued pretraining** on verified English translation text (LoRA, bf16, checkpointed and
   resumable). Starts only after `ready_for_phase2` is true on the Mac and a training source is cleared.
3. **Phase 3 - SFT:** question-answer conversations generated from the source, validated against it,
   and human-reviewed before inclusion (factual, philosophical, character, multi-turn, abstention).
4. **Phase 4 - evaluation:** a held-out question set, base vs fine-tuned, measuring accuracy, verse
   references, hallucination and abstention.
5. **Phase 5 - demo:** FastAPI backend with streaming, plus a Next.js/Tailwind front end, on free-tier
   hosting.

## Licences

The project's code licence has not been chosen yet. Model weights: Qwen3.5 is Apache-2.0. Source texts
keep their own terms, recorded per source in `data/manifests/sources.yaml`.
