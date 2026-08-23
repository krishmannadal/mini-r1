# Mini-R1: Pretraining Readiness Analysis

**Role**: Senior Research Advisor  
**Date**: 2026-08-16  
**Model**: 42.1M decoder-only Transformer  
**Hardware**: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)

---

## 1. Learning-Rate Decision

### Raw Evidence

$\eta = 6\times 10^{-4}$ leads $\eta = 3\times 10^{-4}$ at every validation checkpoint from step 100 through step 1000, without a single crossing. The gap in validation loss:

| Step | $\Delta_{\text{val}} = L_{3\text{e-4}} - L_{6\text{e-4}}$ | Gap Trend |
|---|---|---|
| 100 | +0.1386 | — |
| 200 | +0.0838 | Narrowing |
| 300 | +0.0700 | Narrowing |
| 400 | +0.0829 | Widening slightly |
| 500 | +0.0958 | Widening |
| 600 | +0.1054 | Widening |
| 700 | +0.1186 | Widening |
| 800 | +0.1286 | Widening |
| 900 | +0.1344 | Widening |
| 1000 | +0.1342 | Stable |

### Analysis

1. **Consistency**: The lead is unbroken across 10 validation checkpoints. This is not noise.

2. **Gap dynamics**: The gap initially narrowed (steps 100→300), which is what you would expect if $6\times 10^{-4}$ simply "got there faster" and $3\times 10^{-4}$ was catching up. But from step 400 onward, the gap **widened monotonically** and then plateaued at ~0.134 nats. This means $6\times 10^{-4}$ is not merely faster — it is reaching a better loss basin within this training horizon.

3. **Saturation**: Neither curve has plateaued. At step 1000:
   - $\eta = 3\times 10^{-4}$: loss dropped by 0.019 between steps 900→1000 (still descending)
   - $\eta = 6\times 10^{-4}$: loss dropped by 0.019 between steps 900→1000 (still descending at the same rate)
   
   Both curves are still learning at comparable rates.

4. **Instability indicators**: Zero. GradScaler remained at 65,536 for both runs across all 2,000 combined optimizer steps. Zero skipped updates. Gradient norms stabilized to 0.85–0.90 with no late-stage spikes. There is no evidence of $6\times 10^{-4}$ approaching a divergence boundary.

5. **Cosine schedule fairness**: Both runs used the same warmup duration (50 steps) and the same schedule shape (cosine decay to $\eta_{\text{min}} = \eta_{\text{peak}} / 10$). At step 1000, Run A has decayed to $3\times 10^{-5}$ and Run B to $6\times 10^{-5}$. Since we are comparing peak LR and both schedules are structurally identical, this is a fair comparison. Run B's minimum LR ($6\times 10^{-5}$) is still above Run A's peak ($3\times 10^{-4}$) for only the first ~15% of training, which is expected and does not invalidate the comparison.

6. **Hidden 3e-4 advantage**: If $3\times 10^{-4}$ had a late-stage convergence advantage (flatter minimum, better generalization), you would expect the gap to narrow as both curves approach their asymptotes. The gap is widening or stable, not narrowing. There is no evidence of a hidden advantage.

### Verdict on LR

**$6\times 10^{-4}$ is the preferred peak learning rate.** The evidence is consistent, the gap is widening, and there are zero stability concerns.

---

## 2. Should We Run More LR Qualification?

**Choice: A. Commit to $6\times 10^{-4}$ for long training.**

Justification:
- We have tested 3 candidates across 300 steps and 2 finalists across 1,000 steps.
- $6\times 10^{-4}$ dominates at every checkpoint with a widening gap and zero instability signals.
- Further LR sweeps would consume GPU hours without resolving any remaining ambiguity — there is no ambiguity.
- The scientific weakness of this project is **not** the learning rate. It is the dataset (see §3).

Running more LR experiments would be procrastination disguised as rigor.

---

## 3. Dataset Adequacy

> [!CAUTION]
> **This is the critical finding of this analysis. The dataset situation is not adequate for language model pretraining.**

### Measured Facts

| Metric | Value |
|---|---|
| **Total available training samples** | 104,788 |
| **Total unique training tokens (untruncated)** | **21,161,629 (~21.2M)** |
| **Average tokens per sample** | 201.9 |
| **Samples used in qualification (limit=3000/dataset)** | 6,630 |
| **Tokens in qualification subset** | 1,185,768 (~1.19M) |
| **Model parameters** | 42,100,000 (42.1M) |

### Token-to-Parameter Ratio

The Chinchilla scaling law (Hoffmann et al., 2022) prescribes approximately **20 tokens per parameter** for compute-optimal training of a language model.

$$D_{\text{optimal}} \approx 20 \times N = 20 \times 42.1\text{M} = 842\text{M tokens}$$

| Scenario | Unique Tokens | Token/Param Ratio | Relative to Chinchilla |
|---|---|---|---|
| **Qualification run (limit=3000)** | 1.19M | **0.028** | **0.14%** of optimal |
| **Full dataset (no limit)** | 21.2M | **0.50** | **2.5%** of optimal |
| **Chinchilla optimal** | 842M | 20.0 | 100% |

> [!WARNING]
> Even using the **entire** GSM8K + SVAMP + AQuA training set without any limit, the token-to-parameter ratio is **0.50** — roughly **40× below** the Chinchilla-optimal budget. The qualification run used a ratio of **0.028**, which is **714× below** optimal.

### What This Means

The 1,000-step qualification saw each sample approximately **5 times** (the logs confirm reaching epoch 5). At 16,384 tokens per optimizer step × 1,000 steps = **16.4M token positions processed**, but drawn from only **1.19M unique tokens**. The model has seen the same data approximately **14× over** in terms of raw token positions.

Even with the full 21.2M unique tokens and no sample limit, a single pass through the data would require only:

$$\text{Steps for 1 epoch} = \frac{21.2\text{M}}{16{,}384} \approx 1{,}294 \text{ steps}$$

To reach Chinchilla-optimal training, you would need:

$$\text{Steps} = \frac{842\text{M}}{16{,}384} \approx 51{,}400 \text{ steps}$$

This is physically impossible with 21.2M unique tokens — you cannot create 842M tokens of learning signal from 21.2M unique tokens by repeating them 40 times. Repeated passes over the same data yield diminishing returns and eventual memorization, not generalization.

### Diagnosis

**The current training is not language model pretraining. It is supervised fine-tuning (SFT) on a narrow, domain-specific reasoning benchmark dataset that happens to use a next-token-prediction loss.**

The model is learning the surface patterns of GSM8K/SVAMP/AQuA formatted math reasoning — question templates, arithmetic vocabulary, the `<think>`/`</think>`/`<answer>` delimiters — not general language understanding.

---

## 4. Dataset Quality

### What the datasets actually are

| Dataset | Purpose in Literature | Samples | Character |
|---|---|---|---|
| **GSM8K** | Grade-school math **evaluation benchmark** | 6,726 train / 1,319 test | Short word problems with chain-of-thought |
| **SVAMP** | Arithmetic **evaluation benchmark** | 630 train / 300 test | Minimal reasoning (one equation) |
| **AQuA-RAT** | Multiple-choice math **training/evaluation** | 97,432 train / 254 test | Longer rationales, MCQ format |

### Critical Assessment

1. **GSM8K and SVAMP are evaluation benchmarks.** They were designed to *test* mathematical reasoning, not to *teach* language modeling. Training on them and then evaluating on their test splits is a legitimate SFT experiment, but it is not pretraining.

2. **AQuA-RAT is the bulk of the data** (97K of 105K samples, or 93%). AQuA is reasonable as reasoning supervision data, but it is still narrow — it covers only multiple-choice quantitative reasoning with rationales.

3. **There is no general language pretraining corpus.** The original project plan mentioned `HuggingFaceTB/smollm-corpus` as a general pretraining source, but it was never integrated into the training pipeline. All training to date has been exclusively on formatted math reasoning data.

4. **SVAMP reasoning is synthetic and trivial.** The SVAMP "reasoning" is a single template: *"To solve this problem, we evaluate the arithmetic expression: {equation} = {answer}."* This is not genuine chain-of-thought — it is a fixed template the model will memorize in a few passes.

### Correct Data Roles

| Data Source | Correct Role | Current Role | Assessment |
|---|---|---|---|
| General text corpus (e.g., smollm-corpus) | **Base LM pretraining** | Not used | ❌ Missing |
| GSM8K train | **SFT / CoT fine-tuning** | Base training | ⚠️ Misused |
| SVAMP train | **SFT / CoT fine-tuning** | Base training | ⚠️ Misused |
| AQuA-RAT train | **SFT / CoT fine-tuning** | Base training | ⚠️ Misused |
| GSM8K test | **Evaluation only** | Isolated | ✅ Correct |
| SVAMP test | **Evaluation only** | Isolated | ✅ Correct |
| AQuA-RAT test | **Evaluation only** | Isolated | ✅ Correct |

### Leakage Risk

The test splits are correctly isolated (verified by unit tests). However, there is a subtler risk: with AQuA dominating 93% of training, the model will become highly specialized on AQuA-style MCQ patterns. Validation loss on a mixture that includes GSM8K and SVAMP validation may not reflect true generalization — it reflects generalization within the narrow math-reasoning domain.

---

## 5. Loss Interpretation

### The Numbers

$$\text{PPL} = e^{L} = e^{2.4995} = 12.18$$

A perplexity of 12.18 means the model's predicted distribution is, on average, as uncertain as a uniform distribution over ~12 tokens at each position.

### What PPL=12.18 Does Mean

- The model has learned the formatting structure (`Question:`, `<think>`, `</think>`, `<answer>`, `</answer>`) almost perfectly.
- It can predict common arithmetic tokens, number patterns, and reasoning template phrases with reasonable confidence.
- It has memorized substantial portions of the AQuA-RAT rationale patterns.
- With a vocabulary of 8,000 tokens, PPL=12.18 is a reduction factor of 657× from uniform random chance (PPL=8000). This is strong on the training distribution.

### What PPL=12.18 Does NOT Mean

- **It does not mean the model can reason.** Low perplexity on formatted math data means the model can predict the *next token in a math rationale*, not that it can *solve novel math problems*.
- **It does not mean the model has general language understanding.** If you tested this model on Wikipedia text, news articles, or conversational English, the perplexity would likely be catastrophically high.
- **It does not mean the model is well-pretrained.** A model that has seen 1.19M unique tokens 14 times will naturally achieve low perplexity on those tokens through partial memorization. This is not evidence of generalization — it is evidence of overfitting to a narrow domain.

### Comparison Point

Well-pretrained small language models (e.g., Phi-1.5 at 1.3B, or SmolLM at 135M) achieve perplexities of 10–20 on *diverse* held-out text after seeing 50B+ tokens. Our model achieves PPL=12.18 on *its own training distribution* after seeing 1.19M unique tokens. These numbers are not comparable.

---

## 6. Training Curve Interpretation

### Gap Analysis

The validation loss gap ($L_{3\text{e-4}} - L_{6\text{e-4}}$) evolved as follows:
- Steps 100→300: Gap narrowed from 0.139 to 0.070 (the "catch-up" phase)
- Steps 300→900: Gap widened monotonically from 0.070 to 0.134
- Steps 900→1000: Gap stabilized at ~0.134

This pattern — initial narrowing followed by widening — indicates that $6\times 10^{-4}$ found a qualitatively better optimization trajectory, not merely a speed advantage.

### Is the Model Undertrained?

**On its current data**: Borderline. The model has seen each sample ~5 times (5 epochs). With the limited dataset (6,630 samples), there is still marginal loss improvement, but diminishing returns from data repetition are likely setting in. The loss is still descending at step 1000, but the descent rate has slowed.

**As a language model**: Profoundly undertrained. The model has seen 0.14% of the tokens a model of this size requires for competent language modeling.

### Is the LR Still Doing Useful Work at Step 1000?

Yes. Both curves dropped ~0.019 nats between steps 900 and 1000. The cosine schedule has decayed to near-minimum LR at step 1000 ($6\times 10^{-5}$), so the remaining descent is slow but non-zero. There would be marginal benefit to extending training on this data, but the returns are sharply diminishing because the dataset is exhausted, not because the learning rate is too low.

---

## 7. Gradients and Clipping

### Evidence

- `max_grad_norm` threshold: 1.0
- Maximum observed gradient norm: ~2.6 (during warmup)
- Gradient norms at step 1000: ~0.85–0.90

### Analysis

1. **Early clipping**: During warmup (steps 1–50), gradient norms exceeded 1.0 regularly, reaching 2.4–2.6. Clipping was actively engaged and was performing its intended function — preventing large early gradients from destabilizing the randomly initialized model. This is **correct and expected behavior**.

2. **Late training**: By step 200+, gradient norms stabilized below 1.0 (0.80–0.95 range). Clipping is rarely or never active in the latter 80% of training. The optimizer is operating unconstrained.

3. **Is clipping dominating optimization?** No. Clipping was dominant only during warmup, which is exactly its purpose. The model is not "fighting" the clipping threshold during steady-state training.

4. **Should `max_grad_norm=1.0` be changed?** No. The threshold is appropriate. It is loose enough to allow normal gradient flow during stable training but tight enough to protect against early-training instability. The gradient norms naturally settled below the threshold without being forced there.

---

## 8. Hardware

### VRAM Budget

| Component | Measured |
|---|---|
| Peak allocated tensors | ~3,290 MB (3.21 GB) |
| Reserved caching pool | ~3,810 MB (3.72 GB) |
| Physical VRAM | 6,144 MB (6.00 GB) |
| **Headroom (physical − reserved)** | **~2,334 MB (2.28 GB)** |

### Assessment

1. **Safe for unattended training**: Yes. The 2.28 GB headroom (38% of total VRAM) provides substantial margin for temporary allocation spikes, CUDA caching allocator fragmentation, and OS/driver overhead. The RTX 4050 Laptop GPU typically reserves ~500–700 MB for display and driver, leaving ~1.6 GB of genuine headroom. This is sufficient.

2. **Temporary memory spikes**: The CUDA caching allocator can temporarily over-allocate during backward passes. The observed gap between `peak_allocated` (3,290 MB) and `reserved` (3,810 MB) of ~520 MB indicates normal fragmentation. No OOM risk observed across 2,000+ steps.

3. **Should we increase batch or context?**
   - **Larger micro_batch** (e.g., 12): Would increase activation memory proportionally. Estimated peak would rise to ~4.5 GB allocated, leaving ~1.0 GB headroom. Possible but risky for unattended training.
   - **Longer context** (e.g., 1024): Would approximately double activation memory. Not safe on 6 GB VRAM with the current model.
   - **Recommendation**: Keep micro_batch=8, context=512, accumulation=4. The current configuration is well-optimized for this hardware.

4. **Architecture change**: Not warranted. The 42.1M architecture fits comfortably and trains at ~11,500 tok/s. Changing architecture would invalidate all qualification work without clear benefit.

---

## 9. Research Decision

> [!CAUTION]
> ### Verdict: NOT READY FOR LONG PRETRAINING

### Reason

The training infrastructure is fully qualified and reliable. The learning rate is selected. The hardware is safe. **But the fundamental training paradigm is wrong.**

You are not doing language model pretraining. You are doing supervised next-token prediction on ~1.19M unique tokens of formatted math reasoning data from evaluation benchmarks. Calling this "pretraining" and extending it to more steps will produce a model that has memorized GSM8K/SVAMP/AQuA patterns, not a model that understands language or can reason about novel problems.

### Conditions Required Before Long Training

1. **Decide what you are actually building.** There are two valid paths:

   **Path A: True Pretraining → SFT Pipeline**
   - Integrate a general text corpus (e.g., `HuggingFaceTB/smollm-corpus`, FineWeb-Edu, or similar) for base language model pretraining.
   - Target at minimum 200M–500M tokens of diverse text (compromising below Chinchilla-optimal but still meaningful).
   - Then fine-tune on GSM8K/SVAMP/AQuA as a separate SFT stage.
   - This is the scientifically rigorous approach but requires substantially more training time.

   **Path B: Direct Reasoning SFT (Acknowledge the Scope)**
   - Accept that you are training a narrow math-reasoning model, not a general language model.
   - Use the full 104K training samples (21.2M tokens) without the `limit_per_dataset=3000` cap.
   - Train for 2–4 epochs (2,500–5,000 steps) to extract maximum signal from the limited data without excessive memorization.
   - Evaluate rigorously on held-out test splits.
   - This is honest and achievable on your hardware, but the resulting model will only handle math-style reasoning in the exact format it was trained on.

2. **Remove the `limit_per_dataset` cap.** The qualification runs used only 6,630 of 104,788 available samples. This means 93.7% of the available data was discarded. At minimum, use all available training data.

3. **Establish a proper train/val split for the dominant dataset.** AQuA-RAT comprises 93% of the data. The current validation set uses only 100 samples per dataset (300 total). This is too small for reliable validation loss estimates and may produce noisy validation curves at longer training horizons.

---

## 10. Recommended Configuration (Conditional on Path Selection)

The following assumes **Path B** (Direct Reasoning SFT on full dataset):

| Parameter | Value | Rationale |
|---|---|---|
| **A. Peak LR** | $6\times 10^{-4}$ | Empirically dominant across 1,300 combined qualification steps |
| **B. Min LR** | $6\times 10^{-5}$ | 10:1 peak-to-min ratio (standard) |
| **C. Warmup strategy** | Linear warmup, 100 steps | ~3–4% of total steps; sufficient for initialization stabilization |
| **D. Effective batch size** | 32 (micro=8, accum=4) | Qualified safe on RTX 4050 |
| **E. Tokens per optimizer step** | 16,384 | 32 × 512 |
| **F. Total token budget** | ~21.2M unique × 3 epochs ≈ 63.5M token positions | 3 epochs is a reasonable ceiling before memorization dominates |
| **G. Approximate optimizer steps** | ~3,900 steps (3 epochs over 104K samples at batch 32) | $\lceil 104{,}788 / 32 \rceil \times 3 \approx 3{,}900 \times 3$ |
| **H. Checkpoint interval** | Every 500 steps | ~8 checkpoints across training |
| **I. Validation interval** | Every 250 steps | ~16 validation evaluations |

### J. Metrics to Monitor

| Metric | Purpose |
|---|---|
| `train_loss` | Convergence tracking |
| `val_loss` | Generalization / overfitting detection |
| `val_perplexity` | Interpretable generalization metric |
| `train_loss − val_loss` gap | **Overfitting detector**: if gap exceeds ~0.3 and is widening, the model is memorizing |
| `grad_norm` | Stability indicator; should remain < 1.0 in steady state |
| `scaler_scale` | AMP health; any decrease signals numerical instability |
| `skipped_optimizer_steps` | AMP health; any non-zero value signals gradient overflow |
| `tokens_per_sec_step` | Hardware health; sudden drops indicate thermal throttling |
| `gpu_peak_allocated_mb` | OOM risk monitoring |

### K. Failure Conditions (Stop Training If)

1. `scaler_scale` drops below 32,768 (two consecutive halvings) — indicates systematic gradient overflow
2. `skipped_optimizer_steps` exceeds 5% of total steps — optimizer is not making progress
3. `val_loss` increases for 3 consecutive validation checkpoints (750 steps) — overfitting
4. `train_loss − val_loss` gap exceeds 0.5 and is monotonically widening — memorization
5. `grad_norm` consistently exceeds 2.0 after warmup — optimization instability
6. `gpu_peak_allocated_mb` exceeds 5,000 MB — approaching OOM boundary
7. `tokens_per_sec_step` drops below 5,000 sustained — hardware thermal issue

### L. What to Collect from the Final Long Run

1. Complete JSONL structured log (already implemented)
2. Validation loss at every checkpoint
3. Train-val loss gap trajectory
4. Final `best.pt` checkpoint
5. Test-set evaluation on GSM8K test, SVAMP test, AQuA test (held-out, never seen during training)
6. Sample generations from the trained model (greedy decode 5–10 held-out questions) to qualitatively assess whether the model produces coherent reasoning vs memorized templates

---

## Final Question

> *"Are we now doing meaningful language-model pretraining, or are we still primarily validating the training infrastructure?"*

**Neither, exactly.** The training infrastructure is validated — that work is genuinely complete and well-executed. But what we have been calling "pretraining" is actually **narrow-domain supervised fine-tuning on 1.19M tokens of formatted math benchmark data**. The model has learned to predict tokens in GSM8K/SVAMP/AQuA-formatted text. It has not learned language.

The qualification experiments successfully answered the engineering questions they were designed to answer: the LR is selected, the hardware is safe, the checkpointing works, the gradients are stable. These are real achievements.

But committing the GPU to a "long pretraining run" on the current data would produce a model that has memorized math reasoning templates, not one that can reason. The bottleneck is not training infrastructure or learning rate — **it is the absence of a pretraining corpus**.

The honest next step is to decide: do you want a general-purpose small language model (which requires a pretraining corpus), or a narrow math-reasoning model (which is what the current pipeline will produce, and that is a legitimate goal if acknowledged as such)?
