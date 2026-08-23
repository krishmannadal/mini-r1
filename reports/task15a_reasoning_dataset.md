# Mini-R1: Task 15A — Reasoning SFT Dataset Construction & Validation Report

**Role**: Senior Reasoning-Model Researcher / Data Architect  
**Date**: 2026-08-19  
**Source Dataset**: `openai/gsm8k (main config)`  
**Split Seed**: `42`  
**Base Checkpoint**: `checkpoints/mini_r1_500m_pretrain/best.pt` (Step 18,000 | SHA256: `f058a558edc2020b...`)  
**Tokenizer**: `artifacts/tokenizer/tokenizer.model` (Vocab Size: 8000 | MD5: `e7ef63605f920bf411f0c897c44e31e3`)  

---

## 1. Executive Summary

This report establishes the verified, leak-free **paired Reasoning SFT datasets** for the 42.1M parameter Mini-R1 model:
1. **Direct-Answer SFT Branch**: Problem $\to$ `<answer>{answer}</answer>` (Experimental Control)
2. **Chain-of-Thought SFT Branch**: Problem $\to$ `<think>{reasoning}</think><answer>{answer}</answer>` (Treatment)

Both datasets share the **exact same 6,973 training problems and 500 development problems**, maintaining rigorous isolation against all held-out test splits.

---

## 2. Dataset Partition & Provenance (Part 1 & 2)

| Split | Number of Examples | Source Split | Status | Delimiters Validated |
|---|---|---|---|---|
| **SFT Train** | **6,973** | `openai/gsm8k:train` | 100% Parsed & Cleaned | Yes (`####` parsed) |
| **SFT Dev** | **500** | `openai/gsm8k:train` | 100% Parsed & Cleaned | Yes (`####` parsed) |
| **Total Processed** | **7,473** | `openai/gsm8k:train` | 0 Invalid / 0 Discarded | 100.0% Valid |

> [!NOTE]
> All 7,473 GSM8K training examples contained canonical `####` delimiters and were losslessly parsed without dropping any examples. Calculator annotations (`<<...>>`) were extracted to derive exact operation counts and then stripped to produce clean natural-language reasoning traces.

---

## 3. Two Experimental Training Conditions (Part 3)

### Condition A: Direct-Answer SFT (Control)
Format specification:
```text
Question: {question}
<answer>{answer}</answer>
```

### Condition B: Chain-of-Thought (CoT) SFT (Treatment)
Format specification:
```text
Question: {question}
<think>
{validated reasoning}
</think>
<answer>{answer}</answer>
```

---

## 4. Token Count & Asymmetry Analysis (Part 5)

Tokenized using the frozen 8,000-vocabulary SentencePiece tokenizer:

| Dataset / Condition | Total Tokens | Mean Tok/Ex | Median | p90 | Max |
|---|---|---|---|---|---|
| **Direct-Answer Train** | **508,894** | 73.0 | 68.0 | 106.0 | 255 |
| **CoT Train** | **1,178,275** | 169.0 | 157.0 | 253.0 | 497 |
| **Direct-Answer Dev** | **35,507** | 71.0 | 66.0 | 100.1 | 166 |
| **CoT Dev** | **81,533** | 163.1 | 154.0 | 235.1 | 379 |

### Token Multiplier Confound Metric:
$$\text{CoT Token Multiplier} = \frac{\text{CoT Tokens}}{\text{Direct Tokens}} = \frac{1,178,275}{508,894} = \mathbf{2.32\times}$$

> [!WARNING]
> **Token Multiplier Confound**: The CoT dataset exposes the model to **2.32x** more training tokens than the Direct-Answer dataset across the exact same number of problem instances. When executing Task 15B training, optimizer steps and learning rate schedules must be strictly monitored to account for this asymmetry.

---

## 5. Problem Difficulty & Operation Complexity Breakdown (Part 6)

Arithmetic complexity was derived from GSM8K calculator tags (`<<expr=result>>`):

| Operation Complexity Bucket | Number of Problems | Percentage of Train Set | Description |
|---|---|---|---|
| **1-Step (1 op)** | 464 | 6.65% | Single elementary calculation ($a \pm b$) |
| **2-Step (2 ops)** | 2,022 | 29.0% | Two chained calculations |
| **3-Step (3 ops)** | 1,989 | 28.52% | Three chained calculations |
| **4+ Step ($\ge 4$ ops)** | 2,498 | 35.82% | Multi-step reasoning ($4$ to 9 ops) |

- **Mean Arithmetic Operations**: 3.18 ops/problem (Max: 9)
- **Median Answer Magnitude**: 45.0 (p90: 1123.2)
- **Mean Question Length**: 234.7 characters

---

## 6. Sample Quality Audit (Part 8)

A deterministic random sample of **100 training examples** was audited:
- **Questions Preserved & Non-Empty**: 100/100
- **Reasoning Coherent & Complete**: 100/100
- **Final Numerical Answer Valid**: 100/100
- **Tag Balance & Well-Formedness**: 100/100
- **Pass Rate**: **100.0%**

---

## 7. Strict Benchmark Isolation & Leakage Verification (Part 9)

Exact-string and SHA-256 hash comparison across all held-out evaluation splits:

| Held-Out Benchmark Split | Evaluation Benchmark Size | Exact Match Overlap | Leakage Status |
|---|---|---|---|
| **GSM8K Test** | 1,319 problems | **0** | **100% Clean / Isolated** |
| **SVAMP Test** | 300 problems | **0** | **100% Clean / Isolated** |
| **AQuA-RAT Test** | 254 problems | **0** | **100% Clean / Isolated** |

> [!IMPORTANT]
> **Scope of Verification**: This check guarantees zero exact string or normalized hash collision between SFT train/dev data and the evaluation test benchmarks. It does not perform fuzzy semantic deduplication.

---

## 8. Artifact File Manifest (Part 10)

| File Path | Records | Size | SHA256 Hash |
|---|---|---|---|
| [`data/reasoning/gsm8k_sft_train.jsonl`](data/reasoning/gsm8k_sft_train.jsonl) | 6,973 | ~3.8 MB | `f040b181642281edd4354c2cd67f6ec0af1b922658d8ec3a28136b5ae5e1b6d6` |
| [`data/reasoning/gsm8k_sft_dev.jsonl`](data/reasoning/gsm8k_sft_dev.jsonl) | 500 | ~275 KB | `8bd18a20db0a126b87f26ac18c0627a6eb72a222af7ba0016ac09c39fbfbdc7b` |
| [`data/reasoning/gsm8k_direct_train.jsonl`](data/reasoning/gsm8k_direct_train.jsonl) | 6,973 | ~1.6 MB | `8f5fcbfb48b0ff9a59596147cd90b6924f486d1757ee9e9534425fde5d88d765` |
| [`data/reasoning/gsm8k_direct_dev.jsonl`](data/reasoning/gsm8k_direct_dev.jsonl) | 500 | ~115 KB | `d0ed361828ebffece4b5d42b964f74754aa4a815d0c15c20068ba8b6d8c3ef84` |
| [`data/reasoning/dataset_manifest.json`](data/reasoning/dataset_manifest.json) | Full Spec | ~4 KB | Provenance Manifest |

---

These datasets are constructed and qualified for Controlled Reasoning SFT experimentation.
