# Bhagavatam AI: project plan

*Last updated 2026-11-18, end of Phase 1. Companion documents:
[MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md), [DATA_LICENSING.md](DATA_LICENSING.md),
[RUNBOOK_MAC.md](RUNBOOK_MAC.md).*

## 1. Summary

- **Goal:** adapt `Qwen/Qwen3.5-0.8B-Base` *by training its weights* so it explains the
  Śrīmad-Bhāgavatam accurately in natural English. Retrieval is deliberately not used.
- **Phase 1 is complete in code and tested.** Model, tokenizer, hardware/MPS checks, inference smoke
  test, source manifest and licence-gated data pipeline are all done. 178 unit tests and 5 real-model
  integration tests pass; ruff and mypy --strict are clean on Python 3.11, 3.12 and 3.14.
- **Two gates before training (Phase 2):**
  1. `bhagavatam hardware check` must report `ready_for_phase2: true` **on the M5**. All checks pass on
     the CPU codespace, but MPS behaviour can only be measured on the Mac.
  2. A **verifiably public-domain English translation** must be cleared in the manifest. No English
     source is cleared yet. M. N. Dutt's 1895 translation is the leading candidate (section 4).
- **The base model hallucinates about the Bhāgavatam.** For example, it says the text was "narrated by
  Lord Caitanya to King Parikshit. This is explicitly stated in the first canto". This baseline is
  recorded (`reports/phase1/codespace-cpu/model_smoke.json`) and the fine-tuned model will be measured
  against it.

## 2. Environment

| | Development (this repo's CI/codespace) | Training (your machine) |
|---|---|---|
| Hardware | Linux x86_64, 2 vCPU, 7.8 GB RAM, no GPU | MacBook Air M5, 16 GB unified memory, 8/10-core GPU |
| Role | code, tests, data pipeline, CPU validation of every code path | MPS checks, all training runs |
| Status | all Phase 1 checks pass on CPU | **pending:** run `scripts/mac_phase1.sh` |

No cloud GPU or paid API is used or required.

## 3. Model findings (details in MODEL_COMPATIBILITY.md)

- `Qwen3_5ForCausalLM` text-only load is verified byte-exact: 752.4 M parameters, 18 Gated DeltaNet + 6
  attention layers, vision tower and MTP head skipped.
- On Apple Silicon the DeltaNet layers run a reference PyTorch path (the CUDA kernels don't exist
  there). It is correct and trainable but slower.
- **bf16 on MPS; never fp16** (the DeltaNet backward underflows in fp16, measured).
- **torch ≥ 2.15** is required on M5: earlier versions corrupt bf16 linear layers on this chip and
  have an AdamW bf16 bug.
- **LoRA, not full fine-tuning.** Full fine-tuning needs 8–11 GB of static memory before activations.
- A chunked/fused cross-entropy keeps the 248k-vocabulary logits off the memory budget.
- No packed sequences (DeltaNet state would leak across documents). The empty `<think>` block in the
  chat template determines how SFT samples are built.

## 4. Data sources (details and evidence in DATA_LICENSING.md and `data/manifests/sources.yaml`)

| Source | What | Status | Use |
|---|---|---|---|
| **M. N. Dutt**, *A Prose English Translation of Srimadbhagabatam*, Calcutta 1895–96 | English, chapter-structured prose | Public domain by date: published 1895–96, translator died 1912. Verification and completeness check pending. | **Primary candidate** for training text |
| J. M. Sanyal, *The Srimad-Bhagbatam*, Calcutta 1929–36 (26 parts) | English prose, with verse-range markers per paragraph | **Blocked:** translator unidentified (no death date in any authority file), so India/EU status is unknown. In the US only the 1929–30 parts are clearly PD. A 1954 review calls it "a very free paraphrase". | not used until the translator is identified |
| P. N. Sinha, *A Study of the Bhāgavata Purāṇa* (1901) | English study/summary with interpretation | likely PD; to verify | possible *interpretation* source, always labelled as such |
| GRETIL Bhāgavatapurāṇa | Sanskrit, complete (14,099 verses) | CC BY-NC-SA 4.0. Evidence that ~2/3 derives from the BBT romanised edition. | **metadata only** (verse structure, speaker labels) |
| Sanskrit Wikisource | Sanskrit, complete | CC BY-SA 4.0; source edition undocumented | candidate Sanskrit text |
| BBT / Prabhupāda translation and purports | English | © BBT; no ML permission | **excluded** (permission could be requested) |
| Modern translations (Gita Press 1971, Tagare 1976, Tapasyananda, Debroy 2018) | English | in copyright | excluded |
| sanskritdocuments.org | Sanskrit | terms prohibit AI ingestion | excluded |

Implication: the first trainable corpus is a single 19th-century English translation. Two
consequences follow.

1. **Style.** CPT will teach facts and vocabulary in Victorian English. Modern conversational style
   comes from SFT answers written in modern English and grounded in that text.
2. **Interpretation.** Dutt is one translator's reading. The model must attribute it ("in M. N. Dutt's
   1895 translation…") where wording or interpretation matters.

## 5. Training strategy

### Stage A: continued pretraining (CPT)

- **Data format:** one document per chapter, with structural headers and verse-range markers, e.g.
  `Śrīmad-Bhāgavatam, Canto 1, Chapter 2 (SB 1.2) … [SB 1.2.6] …`. The model then learns which content
  belongs to which reference, which is the basis for citing verses correctly later. The structure table
  derived from GRETIL validates chapter and verse numbering.
- **Method:** LoRA on all attention, DeltaNet and MLP projections. Use a higher rank than for SFT
  (r = 64, α = 64–128, rsLoRA), because knowledge injection benefits from capacity. Embeddings stay
  frozen.
- **Hyper-parameters (initial):** bf16, sequence length = the largest the hardware check passes
  without memory pressure (expected 1,024–2,048), batch 1 × gradient accumulation to ~16k tokens per
  step, LR 1e-4 cosine with 3 % warm-up, 2–4 epochs, grad-clip 1.0, activation checkpointing,
  chunked/fused CE (whichever the check finds fastest).
- **Overfitting and forgetting:** validate on held-out chapters (5 %, stratified by canto, identical
  for every source) and stop early on validation loss. Track general-English perplexity on a small
  public-domain English sample to catch forgetting.
- **Checkpointing:** every N steps, save adapter + optimizer + scheduler + RNG states + data position,
  so training resumes after interruption. Keep the last 3 checkpoints plus the best.

### Stage B: supervised fine-tuning (SFT)

- **Format:** the official chat template (non-thinking), one sample per assistant turn, loss only on
  `{answer}<|im_end|>`. Pad to multiples of 64. Use a custom collator, because pad == eos would
  otherwise mask the end-of-turn signal. Use the shared system prompt (`src/bhagavatam_ai/prompts.py`).
- **Example types:** factual, philosophical, character/relationship, story summaries, teachings of
  specific speakers (using the verse-level speaker labels), cross-chapter comparisons, multi-turn
  dialogues, and **abstention**: questions the Bhāgavatam doesn't address, where the answer says so.
- **Generation pipeline (no fabrication):**
  1. Deterministic, template-based questions from structure: who speaks in a chapter, where an episode
     occurs, which canto covers a topic.
  2. Passage-grounded drafting of question/answer pairs from specific verse ranges. The default drafter
     is a local model (free). An external LLM is optional and only with your explicit opt-in.
  3. Automatic validation: every cited reference must exist, quoted spans must occur verbatim in the
     source, named entities must occur in the cited passage, and the answer must not cite outside its
     passage.
  4. Human review queue (JSONL/CSV with approve/edit/reject). **Only approved examples enter
     training.**
- **Hyper-parameters (initial):** LoRA r = 16–32 on top of the merged CPT weights, LR 1e-4, 2–3 epochs,
  validation on held-out conversations.

### Stage C: evaluation (built before SFT, separate from training data)

A fixed question set covers facts, references, characters and relationships, philosophy, multi-turn
consistency, source vs interpretation, and out-of-scope abstention. Every question is tied to
held-out chapters or to explicit facts. It runs on the base model and on every checkpoint, measuring:

- reference accuracy (via `find_references`, which also detects impossible citations);
- entity and relationship accuracy;
- hallucination rate;
- abstention precision and recall;
- validation perplexity;
- a human-rated sample for fluency and faithfulness.

## 6. Estimates

| Item | Estimate | Basis |
|---|---|---|
| English CPT corpus | ~0.8–1.3 M tokens per translation | ~14,100 verses; 0.6–1.0 M English words at 1.2 tokens/word (measured) |
| SFT set | 5–15 k reviewed examples, ~2–5 M tokens/epoch | ~350 tokens/example |
| Training speed (M5 Air, LoRA, bf16) | ~300–600 tok/s sustained | researched estimate; **to be replaced by the hardware check** |
| CPT time | ~1.5–4 h for 3 epochs | 1.3 M tokens × 3 at 300–600 tok/s |
| SFT time | ~2–9 h for 2 epochs | |
| Storage | < 15 GB total | model 1.7 GB, environment ~3 GB, scans for re-OCR ≤ 4 GB, checkpoints ~150 MB each, merged export 1.5 GB |

These numbers fit overnight runs on the Mac. CPU training is not viable: about 28 h per million tokens
on the codespace (measured).

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No clearly public-domain English translation, or OCR too poor | Verify Dutt (Phase 1b). Re-OCR the scans locally with Apple's Vision framework (free, on the Mac) or Tesseract. Human-review samples. Validate structure against the GRETIL verse table. |
| A single 19th-century translation yields archaic style and one interpretation | Modern-English SFT answers; attribution to the translation; abstain where the source is silent. |
| MPS numerical or operator problems on M5 | Op-level and model-level checks against CPU fp32 before training; torch ≥ 2.15 pinned. |
| Memory pressure or swap on 16 GB | LoRA, chunked CE, activation checkpointing, MPS memory cap, sequence length chosen from measurements. |
| Thermal throttling (fanless Air) | Sustained-run measurement; plan with throttled throughput; checkpoint and resume. |
| 0.8 B model capacity limits recall of fine detail | Measure on the evaluation set. Prefer abstention to guessing. Consider a larger Qwen3.5 later if justified. |
| Hallucinated verse references | References in CPT data; reference validation in SFT data; reference accuracy metric; abstention training. |
| Licence or provenance errors | Evidence-backed manifest enforced in code; adversarial verification of every training source; no raw text committed. |
| Free hosting too slow for a 0.8 B model | Hugging Face Spaces CPU (2 vCPU / 16 GB) with a quantised llama.cpp (GGUF) backend, or local-only demo. |

## 8. Phases and exit criteria

| Phase | Work | Exit criterion |
|---|---|---|
| **1. Foundation** | done (this branch) | **Mac:** `ready_for_phase2: true` |
| **1b. English source** | verify Dutt (licence + completeness); fetch scans; re-OCR; cleaning; chapter and verse-range alignment; parser + tests; human spot-check | Dutt cleared in the manifest as `public_domain`; validation report with 0 errors; corpus built |
| **2. Continued pretraining** | training loop with resumable checkpoints; validation; forgetting check | val perplexity on held-out chapters improves over base; no fluency regression |
| **3. SFT** | QA generation + validation + human review; SFT training | reviewed set ≥ 5k; held-out conversation loss improves |
| **4. Evaluation** | evaluation set + harness; base vs CPT vs SFT comparison report | measurable gains on reference accuracy and hallucination rate |
| **5. Demo** | FastAPI (streaming) + Next.js/Tailwind UI; model disclosure page; free hosting | public demo running the trained model |

## 9. Decisions needed from you

1. Accept **M. N. Dutt (1895)** as the primary English source once verified, knowing its style is
   19th-century and the SFT layer supplies modern English?
2. Should we **ask the BBT for permission**? It is a long shot, and any permission would need to cover
   training, weights and a public demo.
3. SFT drafting: **local model only** (free, slower, more review effort) or **optional external LLM**
   drafting with mandatory validation and human review?
4. A **licence for the code** (e.g. Apache-2.0 or MIT).
