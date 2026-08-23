# Mini-R1: Task 16A — General-Language Perplexity Evaluation Audit Report

**Date**: 2026-08-19  
**Role**: Senior AI Research Scientist / Evaluation Lead  
**Audit Purpose**: Forensic resolution of conflicting general-language validation perplexity figures reported in Task 15C across Base, Direct SFT, and CoT SFT checkpoints.  
**Evaluation Target**: 10,000,000-token general-language pretraining validation suite (`data/pretrain/val_manifest.json`).

---

## 1. Executive Summary

Task 16A independently audited and reproduced the general-language evaluation across all three model checkpoints using the exact pretraining validation dataset and evaluation harness:

1. **Base Pretrained Checkpoint (`best.pt`, step 18,000)**:
   - Raw Cross-Entropy Loss: **2.872419**
   - Perplexity: **17.6797** ($\approx \mathbf{17.68}$)
   - Tokens Evaluated: **9,980,852 tokens** across **19,532 sequences**
   - **Parity with Pretraining Run**: Matches within $1.9539 \times 10^{-4}\text{ nats}$, confirming exact model weight immutability and test repeatability.

2. **Direct-Answer SFT Checkpoint (`best.pt`, step 650)**:
   - Raw Cross-Entropy Loss: **3.083407**
   - Perplexity: **21.8327** ($\approx \mathbf{21.83}$)
   - Tokens Evaluated: **9,980,852 tokens** across **19,532 sequences**
   - Relative Shift vs Base: $\mathbf{+23.49\%}$ (+0.2110 nats).

3. **Chain-of-Thought SFT Checkpoint (`best.pt`, step 650)**:
   - Raw Cross-Entropy Loss: **3.160299**
   - Perplexity: **23.5776** ($\approx \mathbf{23.58}$)
   - Tokens Evaluated: **9,980,852 tokens** across **19,532 sequences**
   - Relative Shift vs Base: $\mathbf{+33.36\%}$ (+0.2879 nats).

---

## 2. Checkpoint & Artifact Cryptographic Integrity

All checkpoints and data manifests were verified via SHA-256 cryptographic hashing prior to evaluation:

| Artifact Name | Local File Path | File Size | SHA-256 Cryptographic Hash | Checkpoint Recorded `val_loss` |
| :--- | :--- | :---: | :--- | :---: |
| **Base Checkpoint** | `checkpoints/mini_r1_500m_pretrain/best.pt` | 508.9 MB | `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4` | 2.872614 (General Val) |
| **Direct SFT Checkpoint** | `artifacts/sft/direct/best.pt` | 508.9 MB | `79323e228f222da6f55ada9bccc84ea5c49d5bb1b0c0a1d20a3746b1b431dacd` | 0.951430 (GSM8K Dev) |
| **CoT SFT Checkpoint** | `artifacts/sft/cot/best.pt` | 508.9 MB | `0df4b0d86730e72c15e1cf8c3aaeebf58a834fc36232b804520760adb0311d93` | 1.343169 (GSM8K Dev) |
| **Validation Manifest** | `data/pretrain/val_manifest.json` | 556 B | `0011c0cb1a848a76ac0c6331a69e6c7d57bb16b7ee9d9721d2bc34152a59640c` | 10M tokens (2 shards) |
| **Tokenizer Model** | `artifacts/tokenizer/tokenizer.model` | 129.5 KB | `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044` | Vocab 8,000 |

---

## 3. Quantitative General Language Validation Results

All three evaluations were executed sequentially on the **NVIDIA GeForce RTX 4050 Laptop GPU** under FP16 Automatic Mixed Precision (AMP), batch size 32, using the exact same teacher-forcing sequence shifting logic in `src/evaluation.py::evaluate_general_language`.

| Evaluation Metric | Pretrained Base (`best.pt`) | Direct-Answer SFT (`best.pt`) | Chain-of-Thought SFT (`best.pt`) |
| :--- | :---: | :---: | :---: |
| **Global Training Step** | 18,000 | 650 (from step 18k base) | 650 (from step 18k base) |
| **Evaluated Sequences** | 19,532 | 19,532 | 19,532 |
| **Evaluated Target Tokens** | 9,980,852 | 9,980,852 | 9,980,852 |
| **Sequence Length ($T$)** | 512 tokens | 512 tokens | 512 tokens |
| **Target Tokens / Sequence** | 511 tokens ($T - 1$) | 511 tokens ($T - 1$) | 511 tokens ($T - 1$) |
| **Raw Cross-Entropy Loss** | **2.872419** | **3.083407** | **3.160299** |
| **Cross-Entropy Loss Delta ($\Delta L$)** | Baseline ($0.000000$) | $+0.210988\text{ nats}$ | $+0.287880\text{ nats}$ |
| **Exact Perplexity ($\exp(\bar{L})$)** | **17.679727** | **21.832661** | **23.577642** |
| **Relative Perplexity Shift ($\Delta \text{PPL}$)** | **Baseline (0.00%)** | $\mathbf{+23.49\%}$ | $\mathbf{+33.36\%}$ |
| **Validation Throughput** | $60{,}453.7\text{ tok/s}$ | $62{,}322.1\text{ tok/s}$ | $62{,}399.2\text{ tok/s}$ |
| **Validation Elapsed Time** | 165.10 s | 160.15 s | 159.95 s |

---

## 4. Root-Cause Forensic Discrepancy Analysis

### 4.1 The Conflicting Claims in the Task 15C Report
In `reports/task15c_direct_vs_cot.md`, two sections reported contradictory numbers:
- **Section 7 (Table)**: Reported Base PPL = **17.68**, Direct SFT PPL = **21.83**, CoT SFT PPL = **23.58**.
- **Section 10 (Discussion Text)**: Reported Direct SFT PPL $\approx \mathbf{17.72}$, CoT SFT PPL $\approx \mathbf{17.84}$ (and referenced mock GSM8K values 3.87% / 2.05%).

### 4.2 Code Path Forensic Tracing
Inspection of `scripts/evaluate_sft_experiment.py` revealed the exact mechanism:
1. **Section 7 Table Generation (Lines 706–712)**:
   ```python
   # Dynamically formatted from live evaluation result dataclass
   f"| **Direct-Answer SFT (`best.pt`)** | {direct_gen_eval.avg_loss:.4f} | **{direct_gen_eval.perplexity:.2f}** | {(direct_gen_eval.perplexity - base_gen_eval.perplexity)/base_gen_eval.perplexity*100:+.2f}% |",
   f"| **Chain-of-Thought SFT (`best.pt`)** | {cot_gen_eval.avg_loss:.4f} | **{cot_gen_eval.perplexity:.2f}** | {(cot_gen_eval.perplexity - base_gen_eval.perplexity)/base_gen_eval.perplexity*100:+.2f}% |"
   ```
   *Verdict*: Section 7 was **100% empirically calculated from the live model execution**.

2. **Section 10 Text Generation (Lines 770–773)**:
   ```python
   "### Mechanistic Observations:",
   "1. **Markup Acquisition**: Pretrained SLMs at 42.1M parameters rapidly learn structural markup tags (`<think>` and `<answer>`) within ~1-2 epochs of masked SFT without deteriorating general syntax.",
   "2. **Arithmetic Bottleneck vs Reasoning Overhead**: In Direct SFT, the model outputs short, direct answers that match simple numerical outputs (3.87% on GSM8K). In CoT SFT, while the model generates extensive plausible multi-step explanations, error propagation across sequential arithmetic steps degrades exact match accuracy to 2.05%.",
   "3. **General Forgetting**: Neither SFT condition caused catastrophic forgetting (General validation perplexity changed from base 17.68 to 17.72 for Direct and 17.84 for CoT), confirming that the conservative $1.0 \\times 10^{-4}$ peak LR preserved baseline representations."
   ```
   *Verdict*: Section 10 contained **hardcoded drafting template text** authored prior to the completion of the empirical benchmark run. The script writer drafted the markdown template with illustrative mock numbers (`17.72`, `17.84`, `3.87%`, `2.05%`) and failed to replace them with dynamic format strings before publishing `reports/task15c_direct_vs_cot.md`.

### 4.3 Provenance & Truth Determination

| Section | Provenance | Status | Scientific Validity |
| :--- | :--- | :---: | :--- |
| **Section 7 Table** | Live runtime output from `evaluate_general_language()` | **AUTHENTIC EMPIRICAL DATA** | **100% Verified & Reproducible** |
| **Section 10 Text** | Hardcoded authoring template string | **DRAFTING ARTIFACT** | **Invalid / Rejected** |

---

## 5. Scientific Implications of Verified Perplexity Shifts

Now that the genuine empirical figures are authenticated:

1. **Both SFT Conditions Incur Genuine Perplexity Degradation**:
   - Direct-Answer SFT increases general-language perplexity by **+23.49%** ($17.68 \to 21.83$).
   - Chain-of-Thought SFT increases general-language perplexity by **+33.36%** ($17.68 \to 23.58$).
   - CoT SFT exhibits **+9.87% greater perplexity degradation** than Direct SFT on the general language corpus.

2. **Why CoT Causes Greater Representation Shift**:
   - In Task 15A/15B accounting, CoT SFT exposed the 42.1M model to **19.05x more active completion tokens** ($2{,}106{,}652$ vs $110{,}594$) over the same 650 steps at learning rate $1.0 \times 10^{-4}$.
   - The model experienced 19x more gradient updates concentrated on reasoning syntax, which applied greater optimization pressure to shift internal transformer weights away from general educational prose geometry.

3. **Classification of Forgetting**:
   - A shift from PPL 17.68 to PPL 23.58 is moderate domain specialization shift, not catastrophic collapse (a collapsed model produces PPL > 1,000). However, it confirms that SFT at LR $1.0 \times 10^{-4}$ does alter general language representation geometry.

---

## 6. Final Authoritative Values for Repository Records

| Model Condition | Checkpoint Step | General Loss (10M Tokens) | General Perplexity | PPL Shift vs Base | GSM8K Strict Accuracy | GSM8K Fallback Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Pretrained** | 18,000 | **2.8724** | **17.68** | Baseline (0.00%) | 0.76% | 0.76% |
| **Direct-Answer SFT** | 650 | **3.0834** | **21.83** | **+23.49%** | 2.73% | 2.73% |
| **Chain-of-Thought SFT** | 650 | **3.1603** | **23.58** | **+33.36%** | 1.21% | 1.97% |

---

GENERAL-LM PPL AUDIT STATUS: RESOLVED
