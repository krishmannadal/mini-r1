# Mini-R1 — Final Research Summary

**Project**: Mini-R1 — a 42.1M-parameter decoder-only Transformer built from scratch in pure PyTorch on a single 6 GB RTX 4050.
**Period**: Tasks 1–24. **Status**: research closed, final model frozen 2026-08-23.
**Final model**: `artifacts/sft/task23_75_25/best.pt` (`d630daac…e49c`)

---

## 1. Research question

Can a 42.1M-parameter language model, given **deterministically verified arithmetic supervision**, acquire arithmetic competence that **transfers** to natural-language mathematical reasoning (GSM8K)?

Secondary: if it does not transfer, *why* not — and can the failure be localised to a specific, measurable mechanism?

## 2. Motivation

Chain-of-thought reasoning is usually studied at ≥1B parameters, where arithmetic ability is already latent. At 42M, arithmetic must be *taught*, which makes the transfer question cleanly separable: teach arithmetic in isolation, measure it, then fine-tune on word problems and measure what survives. Every arithmetic claim can be checked by Python rather than by another model.

## 3. Architecture

Decoder-only pre-LN Transformer: 12 layers, d_model 512, 8 heads, d_ff 2048, context 512, no linear bias, tied input/output embeddings, **42,132,480 parameters** — the largest configuration fitting 6 GB VRAM at batch 32 × seq 512. Tokenizer: SentencePiece BPE, 8,000 vocab, with `<think>`/`<answer>` as atomic single tokens.

## 4. 500M-token pretraining

500M training / 10M validation tokens from a qualified SmolLM 60/40 mixture (`python-edu` excluded — git schema incompatibility). Contiguous EOS-packing achieved 100% token density; naive truncation had wasted 58.4%. An LR sweep selected 6e-4. Training ran **8.02 hours**; loss plateaued at step 18,000, locked as `best.pt`.

**Result**: validation loss 2.8726, **PPL 17.68**, matching pretraining parity to 1.95e-04 nats.

## 5. Initial reasoning failure

Zero-shot base model on GSM8K: **0.76%** greedy (10/1319), 0.91% sampled. Format compliance 0.0%, average output 125.7 tokens. The base model had no reasoning capability to elicit.

## 6. CoT failure

Task 15C, a controlled 650-step paired comparison on identical GSM8K data:

| Condition | GSM8K strict | Repetition |
| :--- | ---: | ---: |
| Direct-answer SFT | **2.73%** | 0.00% |
| Chain-of-thought SFT | **1.21%** | 82.64% |

**CoT actively hurt** (p = 0.0051). Verbose reasoning traces induced autoregressive degeneration rather than reasoning.

## 7. Decoding intervention

Task 16B tested whether this was an inference artifact, sweeping five decoding configurations on the frozen CoT checkpoint. Repetition fell **82.64% → 11.45%** under T=0.6 / top_p=0.95 / rep_pen=1.2, and accuracy reached 2.20% — still below Direct SFT's 2.73%.

**Conclusion**: degeneration was real and fixable; the missing arithmetic was not. Decoding is not the bottleneck.

## 8. Verified arithmetic supervision

Task 18A built a 6-level deterministic arithmetic corpus (13,200 problems, 26.4k verified traces). Task 18B trained on it. This established that the model *can* learn arithmetic to high accuracy when supervision is verified — but raised the question of whether it was learning or memorising.

## 9. Leakage discovery

Tasks 19A–19B audited train/eval overlap and numeric representation. The audit found train/eval contamination inflating apparent generalization, forcing regeneration of clean, disjoint splits (Task 19C). **This is the reason all subsequent arithmetic claims are restricted to explicitly held-out IID and OOD suites.**

## 10. Generalization experiments

Tasks 19C → 19D → 20 progressively widened numeric-regime exposure (2D only → mixed 2D+3D → progressive 2D+3D+4D), yielding the Task 20 arithmetic model:

| Suite | Task 20 result |
| :--- | ---: |
| E1 IID 2-digit `[50,99]` | **100.00%** |
| E2 OOD 2-digit | 18.20% |
| E3 IID 3-digit `[101,199]` | **99.80%** |
| E4 OOD 3-digit | 26.20% |
| E5 IID 4-digit `[1001,1999]` | **67.20%** |
| E6 OOD 4-digit | 8.60% |
| E7 5-digit acid test | **0.00%** |

**Finding**: arithmetic competence is strong but **regime-local**. Exposure to more numeric regimes improves generalization within them; cross-regime extrapolation is 0.00%.

## 11. Arithmetic-to-GSM8K transfer

**Task 21A** (verbose CoT targets): GSM8K 1.44%, retention collapsed to 3.6/0.6/0.4, PPL 28.87, 86.1% repetition.

**Task 21B** (compact verified equation traces) fixed the *mechanics* — format compliance rose to 97.8%, completion tokens fell 85.3% — but not the transfer:

| | Model A (arith-init) | Model B (base-init) | Δ |
| :--- | ---: | ---: | ---: |
| GSM8K strict | 0.91% | 1.21% | **−0.30%** |

Retention collapsed to 4.4/5.2/0.6 and PPL rose to 43.88. **Conclusion C: no measurable transfer.**

Diagnosis of *why*: the Task 20 substrate was **subtraction only** over `[50,99]∪[101,199]∪[1001,1999]`, while measured GSM8K compact traces are `*` 37.40%, `+` 28.97%, `/` 17.13%, `-` 16.50%, with 40.85% of operands in 1–9. The substrate never covered the target distribution.

## 12. Parameter-freezing forensic experiment

**Task 22** acted on that diagnosis: build an operator-matched replay corpus (matched to TVD 0.0063 on operators, 0.0212 on operand magnitude; 25,762 equations, 0 verification failures) and freeze the lower half of the network to protect the substrate.

It **failed, badly**: equation validity **6.45%** vs Model A's 20.45% (−14.00 pts). Isolation held exactly (max |Δ| = 0 on all 62 frozen tensors), and the result survived denominator, padding and harness-fidelity controls.

**Task 22B** ran the missing control — same data, **all** parameters trainable:

| | Model E (frozen) | Model F (trainable) | Δ |
| :--- | ---: | ---: | ---: |
| Equation validity (sampled) | 6.45% | **31.77%** | **+25.32** |
| Equation validity (greedy) | 12.19% | **42.56%** | **+30.38** |
| Equations/generation (greedy) | 5.12 | 2.74 | — |
| p90 length | **256** (cap) | 59 | — |

**Freezing was the failure mechanism**, not the corpus. Because `tie_word_embeddings=True`, freezing `tok_emb` also froze `lm_head`; unable to move its output projection, the model could not learn the stop token and degenerated into `12 + 12 + 12 + …` loops. This also **retracted** Task 22's own conclusion that the corpus design caused the runaway chains.

But retention still collapsed (3.8/1.8/0.0) and PPL worsened to 60.02.

## 13. Replay-ratio experiment

**Task 23** completed the trainable curve at 10% and 25% replay, with the replay block split 50/50 between GSM8K-matched arithmetic and the Task 20 subtraction distribution:

| Replay % | 2D | 3D | 4D | Eq. validity | GSM8K | PPL | RCS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% (Model A) | 4.40% | 5.20% | 0.60% | 23.67% | 0.91% | **43.88** | 0.1661 |
| 10% (Model G) | 65.60% | 54.80% | 13.40% | 31.25% | 2.12% | 47.90 | 0.4110 |
| **25% (Model H)** | **94.80%** | **78.20%** | **32.40%** | 31.64% | **2.43%** | 50.74 | **0.4569** |
| 50% (Model F) | 3.80% | 1.80% | 0.00% | **42.56%** | 2.27% | 60.02 | 0.1520 |

**Arithmetic retention was recovered for the first time in the project** — 2D from 4.40% to 94.80%. PPL rose monotonically with replay share and **no arm passed the pre-registered PPL ≤ 35 gate**, including the 0% baseline at 43.88.

Model F's collapse at the *highest* replay share is a **composition** effect, not dose: its replay contained essentially none of the subtraction regime being probed. Ratio and composition are therefore confounded in this series.

## 14. Final model

**Model H** (Task 23, 75/25 replay, all parameters trainable), selected by the pre-registered fallback rule (highest RCS when no arm passes all gates).

GSM8K 2.43% · equation validity 31.64% · SVAMP 1.67% · AQuA 0.00% · E1 89.60% · E3 84.60% · E5 38.40% · E7 0.60% · retention 94.80/78.20/32.40 · PPL 50.74.

## 15. Limitations

- GSM8K accuracy (2.43%) is statistically indistinguishable from Direct SFT (2.35%) — a one-problem difference. **No configuration in this project produced real word-problem reasoning.**
- Arithmetic is regime-local: 89.60% IID 2-digit vs 20.00% OOD 2-digit vs 0.60% at 5 digits.
- PPL 50.74 is 2.9× the 17.68 base. Fine-tuning cost most of the general language quality.
- Single seed (42), one run per arm, no variance estimates.
- The 0/10/25/50 curve confounds replay ratio with replay composition.
- The evaluation harness left-pads without an attention mask while the model uses absolute position embeddings; a bs=1 control (N=200) showed this depresses measured equation validity by ~3–14 points. All arms measured identically, so comparisons hold, but absolute numbers are conservative.
- No RL was ever run. No scaling comparison was ever run.

## 16. Conclusions

**Established:**
1. Verified arithmetic SFT produces strong **IID** arithmetic at 42M parameters (E1 100%, E3 99.8%).
2. Arithmetic generalization improves with exposure to more numeric regimes, but cross-regime extrapolation remains ~0%.
3. GSM8K SFT causes **catastrophic forgetting** of arithmetic (100% → ~4%) across every unprotected configuration tested.
4. **Parameter freezing is not a viable protection mechanism** here — with tied embeddings it froze the output head and caused severe generation degeneration (−30 pts equation validity, non-terminating output).
5. **Target-distribution replay preserves arithmetic**: 25% replay containing the probed regime lifted 2D retention from 4.40% to **94.80%**.
6. **Replay composition matters at least as much as replay ratio**: 50% off-distribution replay preserved nothing (3.80%); 25% on-distribution preserved almost everything.

**Not established:** general symbolic reasoning; arbitrary arithmetic extrapolation; any RL effect; any scaling effect; an optimal replay ratio.

**Overall**: the project answered its question in the negative for transfer and in the positive for retention. Verified arithmetic competence at 42M is real but regime-bound, and it does **not** convert into natural-language reasoning. It *can*, however, be **protected** through GSM8K fine-tuning by replaying the target distribution — which is the project's transferable result.

## 17. Future work

1. **Deconfound ratio from composition** — rerun 50% replay with the mixed replay block. One run; the only clean way to read the ratio curve.
2. **Recover general PPL** — add a language-replay stream from the pretraining shards, which no arm used. All arms exceeded the PPL ≤ 35 gate including the 0% baseline, so this is the largest open deficit.
3. **Fix the harness padding artifact** — pass the attention mask that `batch_generate` already builds and discards; re-measure all arms. Cheap and raises every absolute number.
4. **Attack semantic grounding directly** — the dominant GSM8K failure is mapping words to operations, not computing. Arithmetic supervision cannot fix it; intermediate supervision on the mapping step might.
5. **Scale check** — one 70–150M run would test whether transfer is capacity-bound or method-bound.
6. **RL with a verifier** — deterministic equation checking makes a clean reward signal; untested here.

---

### Artifact index

| Report | Contents |
| :--- | :--- |
| `task13_base_pretraining_results.md` | 500M-token pretraining |
| `task14_base_model_evaluation.md` | base zero-shot baseline |
| `task15c_direct_vs_cot.md` | Direct vs CoT (CoT hurts) |
| `task16b_decoding_ablation.md` | decoding intervention |
| `task19b_pre2_overlap_audit.md` | leakage discovery |
| `task20_progressive_numeric_curriculum.md` | arithmetic generalization (E1–E7) |
| `task21b_compact_reasoning_transfer.md` | transfer null result |
| `task22_replay_sft.md` | replay + freezing (disconfirmed) |
| `task22b_replay_only.md` | freezing isolated as the cause |
| `task23_replay_ratio.md` | replay-ratio curve, final selection |
| `task24_final_integrity.md` | final integrity verification |
| `task24_final_benchmarks.md` | final benchmarks + claim audit |
| `Mini-R1_model_card.md` | model card |
