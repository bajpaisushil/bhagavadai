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
  2. A **verifiably public-domain English translation** must be cleared in the manifest. M. N. Dutt's
     1895–96 translation is now adversarially verified as public domain everywhere. It becomes training
     text when the Phase 1b parser exists and the missing volume (from Google Books) has been added.
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

| Source | What | Status (adversarially verified 2026-11-18) | Use |
|---|---|---|---|
| **M. N. Dutt**, *A Prose English Translation of Srimadbhagavatam*, Calcutta 1895–96 | English prose, all 12 books, chapters + speaker tags, **no verse numbers** | **Public domain everywhere**, with conditions: drop a bound-in unrelated pamphlet from one scan; fetch Books X.6–XII from Google Books by hand | **Primary training text** (after Phase 1b) |
| S. Subba Rau, Tirupati 1928 | English, **verse-numbered**, Cantos 1–7 | Public domain **in the US only** (translator's death year unknown) | your decision (section 9) |
| M. N. Chatterjee, Calcutta 1895 | English, **verse-numbered**, Canto 1 (+2 at HathiTrust), Devanagari, notes | Public domain **in the US only** | your decision (section 9) |
| J. M. Sanyal, Calcutta 1929–36 | English prose with verse-range markers | **Blocked:** translator unidentified; a "very free paraphrase" | not used |
| GRETIL Bhāgavatapurāṇa | Sanskrit, complete (14,099 verses) | CC BY-NC-SA 4.0; ~2/3 appears to derive from the BBT romanised edition | **metadata only** (structure, speaker labels, verse equivalences) |
| DCS; Sanskrit Wikisource | Sanskrit (partial / complete) | CC BY 4.0 annotations / CC BY-SA 4.0, editions undocumented | under review |
| BBT / Prabhupāda; bhagavata.org (= srimadbhagavatam.org, BBT-derived); ~20 vedabase-scraped datasets | English | © BBT, or derived from it | **excluded** |
| Gita Press, Tagare, Tapasyananda, Debroy, Prabhavananda | English | in copyright | excluded |

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

1. Accept **M. N. Dutt (1895–96)** as the primary English source? It is verified public domain, but its
   19th-century prose means the modern voice must come from the SFT layer.
2. **Download one PDF by hand:** Dutt Books VIII–XII from Google Books (`RQQ-AQAAMAAJ`, full view). Google's
   guidelines forbid scripted downloads, and only that scan contains Book X ch. 6–90, XI and XII.
3. Use **verse-numbered US-only public-domain** sources (Subba Rau 1928, Chatterjee 1895) to enable
   verse-level citations? The trade-off: the published weights would rest on US status for that text.
   Without them, citations stay at chapter level ("SB 10.29").
4. Should we **ask the BBT for permission**? Any permission would need to cover training, weights and a
   public demo.
5. SFT drafting: **local model only** (free) or **optional external LLM** drafting, with mandatory
   validation and human review?
6. A **licence for the code** (e.g. Apache-2.0 or MIT).

## 10. Status beyond Phase 1

- **Web demo:** a FastAPI backend with streaming plus a Next.js chat UI. It currently serves the base
  model, labelled as untrained (`make demo`).
- **Independent code review:** 4 reviewers, then adversarial verifiers. 25 findings were reported:
  - 19 were confirmed;
  - 5 were real but already fixed by the time the verifiers looked;
  - 1 was refuted (and documented anyway).

  All real findings are fixed with regression tests. They include a licence-gate redirect bypass, NaN
  training steps reported as success, and a bf16 fused-loss error of 15 %.
