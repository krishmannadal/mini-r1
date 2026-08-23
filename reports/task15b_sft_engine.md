# Mini-R1 — Task 15B: Masked SFT Engine & Smoke Test Qualification Report

**Date**: 2026-08-18 20:18:45 UTC
**Status**: QUALIFIED FOR SUPERVISED FINE-TUNING EXPERIMENTS
**Target Architecture**: Mini-R1 Decoder Transformer (42.1M parameters, d_model=512, n_layers=12, n_heads=8, vocab=8000, context=512)
**Base Pretrained Checkpoint**: `best.pt` (step 18,000, val loss 2.8726, SHA256: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`)

---

## 1. Executive Summary

Task 15B successfully designed, implemented, and verified the **Masked Supervised Fine-Tuning (SFT) Engine** for Mini-R1 across two controlled experimental conditions:
1. **Direct-Answer SFT (Control Condition)**: `Question: {q}\n<answer>{a}</answer>`
2. **Chain-of-Thought SFT (Treatment Condition)**: `Question: {q}\n<think>\n{r}\n</think>\n<answer>{a}</answer>`

All requirements specified in the research protocol were validated through rigorous unit tests and 10-optimizer-step smoke tests on both branches.

---

## 2. Masking Mechanics & Causal Loss Formulation

### Mathematical Formulation
Given a tokenized sequence $x = [x_0, x_1, \dots, x_{T-1}]$ with prompt boundary $L_{\text{prompt}}$:
- **Target Labels**:
  $$\text{labels}[i] = \begin{cases} -100 & \text{if } 0 \le i < L_{\text{prompt}} \quad (\text{Prompt tokens ignored}) \\ x_i & \text{if } L_{\text{prompt}} \le i < T \quad (\text{Completion tokens active}) \end{cases}$$
- **Causal LM Shift**:
  $$\text{shift\_logits} = \text{logits}[:, :-1, :] \quad (\text{positions } 0, \dots, T-2)$$
  $$\text{shift\_targets} = \text{labels}[:, 1:] \quad (\text{positions } 1, \dots, T-1)$$
- **Next-Token Target Mapping**:
  At position $t = L_{\text{prompt}} - 1$ (the final prompt token, e.g. `\n`), the target is $\text{shift\_targets}[L_{\text{prompt}} - 1] = \text{labels}[L_{\text{prompt}}] = x_{L_{\text{prompt}}}$ (the first completion token, e.g. `<think>` or `<answer>`).
  Thus, the prompt provides full conditioning context, while SFT loss backpropagates solely into completion generation.

### Visual Masking Demonstration
```

=======================================================
 Toy Direct-Answer SFT Example (Total Tokens: 26)
 Prompt: 22 toks (MASKED) | Completion: 4 toks (ACTIVE)
=======================================================
Idx   | Tok ID   | Decoded Piece             | Label    | Role
------+----------+---------------------------+----------+-----------------------
0     | 404      | Question                  | -100     | PROMPT (Ignored, Loss=0)
1     | 7825     | :                         | -100     | PROMPT (Ignored, Loss=0)
2     | 4841     |  Janet                    | -100     | PROMPT (Ignored, Loss=0)
3     | 398      |  has                      | -100     | PROMPT (Ignored, Loss=0)
4     | 7789     |                           | -100     | PROMPT (Ignored, Loss=0)
5     | 7823     | 3                         | -100     | PROMPT (Ignored, Loss=0)
6     | 1215     |  apples                   | -100     | PROMPT (Ignored, Loss=0)
7     | 7810     | .                         | -100     | PROMPT (Ignored, Loss=0)
8     | 834      |  She                      | -100     | PROMPT (Ignored, Loss=0)
9     | 1465     |  buys                     | -100     | PROMPT (Ignored, Loss=0)
10    | 7789     |                           | -100     | PROMPT (Ignored, Loss=0)
11    | 7816     | 2                         | -100     | PROMPT (Ignored, Loss=0)
12    | 512      |  more                     | -100     | PROMPT (Ignored, Loss=0)
13    | 7810     | .                         | -100     | PROMPT (Ignored, Loss=0)
14    | 459      |  How                      | -100     | PROMPT (Ignored, Loss=0)
15    | 418      |  many                     | -100     | PROMPT (Ignored, Loss=0)
16    | 1215     |  apples                   | -100     | PROMPT (Ignored, Loss=0)
17    | 636      |  does                     | -100     | PROMPT (Ignored, Loss=0)
18    | 441      |  she                      | -100     | PROMPT (Ignored, Loss=0)
19    | 496      |  have                     | -100     | PROMPT (Ignored, Loss=0)
20    | 7834     | ?                         | -100     | PROMPT (Ignored, Loss=0)
21    | 18       | <0x0A>                    | -100     | PROMPT (Ignored, Loss=0)
22    | 6        | <answer>                  | 6        | COMPLETION (Active Loss)
23    | 7820     | 5                         | 7820     | COMPLETION (Active Loss)
24    | 7        | </answer>                 | 7        | COMPLETION (Active Loss)
25    | 2        | </s>                      | 2        | COMPLETION (Active Loss)
=======================================================

```

```

=======================================================
 Toy CoT SFT Example (Total Tokens: 56)
 Prompt: 22 toks (MASKED) | Completion: 34 toks (ACTIVE)
=======================================================
Idx   | Tok ID   | Decoded Piece             | Label    | Role
------+----------+---------------------------+----------+-----------------------
0     | 404      | Question                  | -100     | PROMPT (Ignored, Loss=0)
1     | 7825     | :                         | -100     | PROMPT (Ignored, Loss=0)
2     | 4841     |  Janet                    | -100     | PROMPT (Ignored, Loss=0)
3     | 398      |  has                      | -100     | PROMPT (Ignored, Loss=0)
4     | 7789     |                           | -100     | PROMPT (Ignored, Loss=0)
5     | 7823     | 3                         | -100     | PROMPT (Ignored, Loss=0)
6     | 1215     |  apples                   | -100     | PROMPT (Ignored, Loss=0)
7     | 7810     | .                         | -100     | PROMPT (Ignored, Loss=0)
8     | 834      |  She                      | -100     | PROMPT (Ignored, Loss=0)
9     | 1465     |  buys                     | -100     | PROMPT (Ignored, Loss=0)
10    | 7789     |                           | -100     | PROMPT (Ignored, Loss=0)
11    | 7816     | 2                         | -100     | PROMPT (Ignored, Loss=0)
12    | 512      |  more                     | -100     | PROMPT (Ignored, Loss=0)
13    | 7810     | .                         | -100     | PROMPT (Ignored, Loss=0)
14    | 459      |  How                      | -100     | PROMPT (Ignored, Loss=0)
15    | 418      |  many                     | -100     | PROMPT (Ignored, Loss=0)
16    | 1215     |  apples                   | -100     | PROMPT (Ignored, Loss=0)
17    | 636      |  does                     | -100     | PROMPT (Ignored, Loss=0)
18    | 441      |  she                      | -100     | PROMPT (Ignored, Loss=0)
19    | 496      |  have                     | -100     | PROMPT (Ignored, Loss=0)
20    | 7834     | ?                         | -100     | PROMPT (Ignored, Loss=0)
21    | 18       | <0x0A>                    | -100     | PROMPT (Ignored, Loss=0)
22    | 4        | <think>                   | 4        | COMPLETION (Active Loss)
23    | 18       | <0x0A>                    | 18       | COMPLETION (Active Loss)
24    | 7716     | Jan                       | 7716     | COMPLETION (Active Loss)
25    | 310      | et                        | 310      | COMPLETION (Active Loss)
26    | 3466     |  starts                   | 3466     | COMPLETION (Active Loss)
27    | 389      |  with                     | 389      | COMPLETION (Active Loss)
28    | 7789     |                           | 7789     | COMPLETION (Active Loss)
29    | 7823     | 3                         | 7823     | COMPLETION (Active Loss)
30    | 1215     |  apples                   | 1215     | COMPLETION (Active Loss)
31    | 7810     | .                         | 7810     | COMPLETION (Active Loss)
32    | 834      |  She                      | 834      | COMPLETION (Active Loss)
33    | 1465     |  buys                     | 1465     | COMPLETION (Active Loss)
34    | 7789     |                           | 7789     | COMPLETION (Active Loss)
35    | 7816     | 2                         | 7816     | COMPLETION (Active Loss)
36    | 512      |  more                     | 512      | COMPLETION (Active Loss)
37    | 7811     | ,                         | 7811     | COMPLETION (Active Loss)
38    | 431      |  so                       | 431      | COMPLETION (Active Loss)
39    | 7789     |                           | 7789     | COMPLETION (Active Loss)
40    | 7823     | 3                         | 7823     | COMPLETION (Active Loss)
41    | 447      |  +                        | 447      | COMPLETION (Active Loss)
42    | 7789     |                           | 7789     | COMPLETION (Active Loss)
43    | 7816     | 2                         | 7816     | COMPLETION (Active Loss)
44    | 316      |  =                        | 316      | COMPLETION (Active Loss)
45    | 7789     |                           | 7789     | COMPLETION (Active Loss)
46    | 7820     | 5                         | 7820     | COMPLETION (Active Loss)
47    | 1215     |  apples                   | 1215     | COMPLETION (Active Loss)
48    | 7810     | .                         | 7810     | COMPLETION (Active Loss)
49    | 18       | <0x0A>                    | 18       | COMPLETION (Active Loss)
50    | 5        | </think>                  | 5        | COMPLETION (Active Loss)
51    | 18       | <0x0A>                    | 18       | COMPLETION (Active Loss)
52    | 6        | <answer>                  | 6        | COMPLETION (Active Loss)
53    | 7820     | 5                         | 7820     | COMPLETION (Active Loss)
54    | 7        | </answer>                 | 7        | COMPLETION (Active Loss)
55    | 2        | </s>                      | 2        | COMPLETION (Active Loss)
=======================================================

```

---

## 3. Controlled Experimental Configurations

| Hyperparameter | Direct-Answer SFT (Control) | Chain-of-Thought SFT (Treatment) | Scientific Justification |
| :--- | :---: | :---: | :--- |
| **Base Checkpoint** | `best.pt` (step 18k) | `best.pt` (step 18k) | Identical starting weights |
| **Peak Learning Rate** | `1.0e-4` | `1.0e-4` | Conservative LR prevents catastrophic forgetting on 42M model |
| **Minimum Learning Rate** | `1.0e-5` | `1.0e-5` | 10x cosine decay floor |
| **Warmup Steps** | 50 | 50 | Smooth linear warmup |
| **Micro Batch Size** | 8 | 8 | Fits comfortably in 6GB VRAM |
| **Gradient Accumulation** | 4 | 4 | Effective batch size = 32 |
| **Effective Batch Size** | 32 | 32 | Matched optimization dynamics |
| **Weight Decay** | 0.01 | 0.01 | Decoupled 2D AdamW decay |
| **Max Grad Norm** | 1.0 | 1.0 | Gradient clipping threshold |
| **Max Sequence Length** | 512 | 512 | Full context capacity |
| **Random Seed** | 42 | 42 | Deterministic batch sequencing |

---

## 4. Token Accounting & Multiplier Analysis

| Metric | Direct SFT (10 Steps) | CoT SFT (10 Steps) | CoT / Direct Ratio |
| :--- | :---: | :---: | :---: |
| **Total Tokens Processed** | 35,480 | 85,824 | 2.42x |
| **Active Completion Tokens (Loss)** | 1,718 | 32,486 | **18.91x** |
| **Ignored Prompt Tokens (Masked)** | 21,282 | 21,282 | 1.00x |

---

## 5. 10-Step Smoke Test Trajectory & Metrics

### Step-by-Step Training Progression

| Step | Direct Loss | Direct Grad Norm | Direct LR | CoT Loss | CoT Grad Norm | CoT LR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 10.3437 | 42.2175 | 1.18e-05 | 3.5052 | 4.7606 | 1.18e-05 |
| 02 | 10.2118 | 41.4876 | 1.36e-05 | 3.6059 | 4.5723 | 1.36e-05 |
| 03 | 9.1898 | 30.9705 | 1.54e-05 | 3.4057 | 4.0958 | 1.54e-05 |
| 04 | 9.0614 | 29.4111 | 1.72e-05 | 3.2328 | 3.8652 | 1.72e-05 |
| 05 | 7.8379 | 25.1528 | 1.90e-05 | 3.1382 | 3.3810 | 1.90e-05 |
| 06 | 7.7148 | 22.2024 | 2.08e-05 | 2.9800 | 3.0792 | 2.08e-05 |
| 07 | 7.6379 | 26.3014 | 2.26e-05 | 3.2768 | 3.3068 | 2.26e-05 |
| 08 | 7.1540 | 22.9641 | 2.44e-05 | 3.0647 | 3.0316 | 2.44e-05 |
| 09 | 6.5399 | 19.2867 | 2.62e-05 | 2.7981 | 2.6466 | 2.62e-05 |
| 10 | 5.7842 | 16.6378 | 2.80e-05 | 2.7960 | 2.4552 | 2.80e-05 |

### Dev-Set Evaluation (Before vs After 10 Steps)

| Condition | Metric | Step 0 (Pre-SFT) | Step 10 (Post-SFT) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Direct SFT** | Dev Loss | 10.6142 | 5.6597 | -4.9545 |
| **Direct SFT** | Completion PPL | 40708.92 | 287.06 | -40421.86 |
| **Direct SFT** | Format Compliance | 0.0% | 0.0% | +0.0% |
| **CoT SFT** | Dev Loss | 3.6065 | 2.9054 | -0.7010 |
| **CoT SFT** | Completion PPL | 36.84 | 18.27 | -18.56 |
| **CoT SFT** | Format Compliance | 0.0% | 0.0% | +0.0% |

---

## 6. Engineering & Checkpoint Qualification

1. **Finite Gradients**: 100% of gradient norms were strictly finite (no NaN, no Inf, no skipped steps).
2. **Weight Movement**: Both models verified non-zero parameter movement from `best.pt` (Direct norm $\Delta = 0.2448$, CoT norm $\Delta = 0.2493$).
3. **Base Checkpoint Immutability**: Base checkpoint `best.pt` verified 100% unchanged (SHA256: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`).
4. **Checkpoint Resumability**: Resumed training from Step 10 checkpoint and verified exact state restoration and monotonic continuation to Step 12 for both conditions.
5. **Unit Test Suite**: 8/8 unit tests passed in `tests/test_sft_engine.py`.

---

## 7. Artifact Manifest

- `src/sft_dataset.py`: SFT data formatting, token masking, dynamic batch collator
- `src/sft_trainer.py`: Causal masked SFT trainer, dev evaluator, atomic checkpointing
- `configs/sft_direct.yaml`: Direct-Answer SFT configuration
- `configs/sft_cot.yaml`: CoT SFT configuration
- `tests/test_sft_engine.py`: 8 unit tests for SFT masking, shifting, and checkpointing
- Direct Checkpoint: `runs\sft_direct\run_20260818_201652_6fb499\checkpoints\step_10.pt`
- CoT Checkpoint: `runs\sft_cot\run_20260818_201750_8d41a9\checkpoints\step_10.pt`

**Verdict**: SFT Engine is verified and ready for full training upon user authorization.