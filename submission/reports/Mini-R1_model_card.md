# Model Card — Mini-R1 (42.1M)

**Version**: 1.0 (final, frozen 2026-08-23)
**Checkpoint**: `artifacts/sft/task23_75_25/best.pt`
**SHA256**: `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c`
**Internal label**: Model H — Task 23, 75/25 replay, all parameters trainable

Mini-R1 is a small research language model built from scratch in pure PyTorch on a single 6 GB consumer GPU. It was built to study whether verified arithmetic supervision transfers to natural-language mathematical reasoning at small scale. **It is a research artifact, not a usable math assistant.**

---

## Architecture

| | |
| :--- | :--- |
| Type | Decoder-only causal Transformer (pre-LayerNorm) |
| Parameters | **42,132,480** |
| Layers | 12 |
| Model dim | 512 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Context window | 512 tokens |
| Position encoding | Learned absolute (`pos_emb`, 512×512) |
| Linear bias | None |
| Weight tying | **Yes** — `lm_head.weight` *is* `embeddings.tok_emb.weight` |
| Dropout | 0.0 |
| Precision | FP16 AMP (training), FP16 autocast (inference) |

## Tokenizer

SentencePiece BPE, **8,000** vocab, with atomic special tokens `<think>`, `</think>`, `<answer>`, `</answer>` (each a single token id — `</answer>` is id 7).
SHA256 `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`.

## Training data

**Pretraining** — 500M tokens (SmolLM 60/40 mixture; `python-edu` excluded for schema incompatibility), 10M held-out validation tokens, contiguous EOS-packed at 100% token density. 8.02 hours, peak LR 6e-4, best checkpoint at step 18,000. Validation PPL **17.68**.

**Supervised fine-tuning** — 650 optimizer steps, effective batch 32 (micro 8 × accum 4), AdamW, peak LR 1e-4 cosine → 1e-5, warmup 50, weight decay 0.01, grad clip 1.0, seed 42, completion-only loss (prompt tokens masked to −100). Initialized from the Task 20 arithmetic checkpoint (`fdcecade71…`).

Mixture (deterministic global-index schedule, `is_replay = idx % 4 == 0`):

| Source | Share | Examples seen |
| :--- | ---: | ---: |
| Compact verified GSM8K reasoning (6,531 traces) | **75.000%** | 15,600 |
| Operator-matched replay arithmetic (8,000 traces) | **12.506%** | 2,600 |
| Task 20 progressive subtraction (12,000 traces) | **12.494%** | 2,600 |

Total 920,031 active completion tokens. Every arithmetic equation in both replay corpora is deterministically verified by Python evaluation.

## Replay strategy

The central finding of the project. GSM8K SFT alone destroys the arithmetic substrate (2D retention 100% → ~4%). Interleaving a **25% replay block that contains the target arithmetic distribution** preserves it (2D → 94.80%). Replaying arithmetic that does *not* match the retention regime does not help (50% off-distribution replay → 3.80%). Composition matters at least as much as dose.

## Prompt format

```
Question: {problem text}\n
```
The model completes:
```
<think>
{equation}
{equation}
</think>
<answer>{number}</answer>
```

## Benchmarks

Greedy decoding unless noted. Full data: `reports/task24_final_benchmarks.json`.

| Benchmark | Result |
| :--- | ---: |
| GSM8K test (1,319) | **2.43%** strict |
| GSM8K step-level equation validity | **31.64%** |
| SVAMP test (300) | 1.67% |
| AQuA test (254) | **0.00%** |
| E1 IID 2-digit subtraction | **89.60%** |
| E2 OOD 2-digit | 20.00% |
| E3 IID 3-digit | **84.60%** |
| E4 OOD 3-digit | 16.60% |
| E5 IID 4-digit | 38.40% |
| E6 OOD 4-digit | 4.60% |
| E7 5-digit acid test (never trained) | **0.60%** |
| Retention probe 2D / 3D / 4D | **94.80% / 78.20% / 32.40%** |
| General-language PPL (10M held-out) | **50.74** |

Recommended decoding: **greedy** for arithmetic (100% EOS, 0% repetition on E1–E6). For GSM8K, T=0.6 / top_p=0.95 / rep_pen=1.2 gives far healthier generations (16.4% vs 77.3% repetition) at lower strict accuracy (1.21% vs 2.43%).

## Limitations

1. **Not a reasoning model.** 2.43% on GSM8K. Word-problem accuracy never exceeded ~2.4% in any configuration across the whole project.
2. **Arithmetic is regime-local, not general.** Strong inside trained operand ranges (89.6% IID 2-digit), weak just outside them (20.0% OOD 2-digit), and effectively zero at 5 digits (0.60%).
3. **Substantial language regression.** PPL 50.74 vs the 17.68 base — 2.9× worse. Fine-tuning cost most of the general language modelling quality.
4. **No MCQ capability.** AQuA 0.00%, with 0.0% full-format compliance.
5. **Repetition under greedy on word problems** — 77.3% 4-gram repetition on GSM8K.
6. **512-token context.**
7. **Single seed.** All results are one run at seed 42; no variance estimates.
8. **Benchmark numbers are conservative.** The evaluation harness left-pads without an attention mask while the model uses absolute position embeddings; a controlled test showed this depresses measured equation validity by roughly 3–14 points.

## Known failure modes

- **Semantic grounding failure** — maps word problems to the wrong operations (e.g. "3 sprints 3 times a week, 60 m each" → `3 * 60` instead of `3 * 3` then `* 60`).
- **Arithmetic hallucination outside trained regimes** — emits confident, well-formatted, wrong equations (`16 * 2 = 12`).
- **Digit-copy loops** on long operands; repetition climbs to 18.6% on 5-digit inputs.
- **Format collapse on unfamiliar prompt shapes** — MCQ prompts produce no `<answer>` block.
- **Compounding multi-step error** — each additional step multiplies failure probability; direct-answer SFT scored higher on GSM8K than any multi-step variant.

## Ethical and use limitations

- **Research use only.** Do not use for education, tutoring, finance, health, or any setting where a wrong number carries cost.
- **Do not trust its arithmetic.** It produces fluent, correctly formatted, confidently wrong equations. This is more hazardous than visible failure.
- Pretrained on a public web/education mixture (SmolLM); it inherits that corpus's biases and was not audited for them.
- No safety tuning, no alignment, no content filtering.
- No PII filtering was applied beyond whatever the upstream corpus performed.

## Inference requirements

| | |
| :--- | :--- |
| Minimum VRAM | ~1 GB (FP16 weights ≈ 84 MB; checkpoint file 509 MB includes optimizer state) |
| Reference hardware | NVIDIA RTX 4050 Laptop (6 GB) |
| Latency | ~34 ms/token greedy (base-model measurement, RTX 4050) |
| Dependencies | PyTorch, SentencePiece, NumPy. No external services. |
| CPU inference | Supported; substantially slower |

See `submission/INFERENCE.md` for runnable code.
