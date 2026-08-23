# Mini-R1: Task 18A — Verified Computational Training Corpus Report

**Date**: 2026-08-20 03:44:09  
**Dataset Location**: `data/arithmetic_sft/`  
**Total Volume**: 13,200 Unique Problems (12,000 Train / 1,200 Dev)  
**Verification Engine**: Python Deterministic Integer Verifier (0% LLM Labels, 100% Verified)  

---

## 1. Executive Summary

To address the complete absence of arithmetic computation capability uncovered in Task 17A (<2% accuracy), Task 18A establishes a 6-level verified computational curriculum. Every training and validation example is generated deterministically using exact Python integer arithmetic with machine-verifiable step metadata.

---

## 2. Dataset File Overview & Verification Results

| Split File | Description | Total Rows | Valid Rows | Error Count | Pass Rate | Mean Prompt Toks | Mean Comp Toks |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `compact_train.jsonl` | Compact Trace | 12000 | 12000 | 0 | **100.00%** | 24.2 | 34.7 |
| `compact_dev.jsonl` | Compact Trace | 1200 | 1200 | 0 | **100.00%** | 24.0 | 34.4 |
| `verbose_train.jsonl` | Verbose Trace | 12000 | 12000 | 0 | **100.00%** | 24.2 | 48.2 |
| `verbose_dev.jsonl` | Verbose Trace | 1200 | 1200 | 0 | **100.00%** | 24.0 | 47.8 |

---

## 3. Curriculum Level Distribution (Train Set)

| Level | Level Name | Operation Depth ($K$) | Examples (Train) | Examples (Dev) | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Level 1** | Single Operations | 1 | 2,000 | 200 | Direct addition, subtraction, multiplication, exact integer division |
| **Level 2** | Two Operations | 2 | 2,000 | 200 | Binary compositions with explicit parenthesis notation |
| **Level 3** | Three Operations | 3 | 2,000 | 200 | Multi-step expressions (e.g. ((a * b) + c) - d) |
| **Level 4** | Four Operations | 4 | 2,000 | 200 | Four-step nested compositions |
| **Level 5** | Sequential State | 2–4 | 2,000 | 200 | State tracking transitions ('Start with X. Add Y. Multiply by Z...') |
| **Level 6** | Word-Wrapped | 1–3 | 2,000 | 200 | Natural language word problems wrapping Levels 1–3 arithmetic |

---

## 4. Operation Type Breakdown (Train Set Total Operations)

| Operator | Symbol | Total Step Operations (Train) | Distribution % |
| :--- | :---: | :---: | :---: |
| **Multiplication** | `*` | 10,603 | 35.3% |
| **Addition** | `+` | 9,610 | 32.0% |
| **Subtraction** | `-` | 7,989 | 26.6% |
| **Division** | `/` | 1,876 | 6.2% |

---

## 5. Token Length & Multiplier Analysis

| Metric | Compact Trace | Verbose Trace | Ratio (Verbose / Compact) |
| :--- | :---: | :---: | :---: |
| **Mean Completion Tokens** | 34.7 | 48.2 | **1.39x** |
| **Median Completion Tokens** | 34 | 50 | 1.47x |
| **p90 Completion Tokens** | 51 | 74 | 1.45x |
| **Mean Prompt Tokens** | 24.2 | 24.2 | 1.00x |

---

## 6. Number Range Control & Generalization Protocol

- **In-Distribution Training Domain**: Operands sampled strictly from defined bounds ($[2, 99]$ for $+/-$, $[2, 25]$ for $\times$, $[2, 12]$ for $\div$).
- **Pre-Registered Holdout Domain (Out-of-Distribution)**: Future generalization benchmark will evaluate operands strictly in $[101, 250]$ (for $+/-$), $[26, 60]$ (for $\times$), and $[13, 30]$ (for $\div$).
- **Leakage Audit**: 0% overlap of problem tuples between train/dev sets and external test sets (GSM8K/SVAMP/AQuA).

---

## 7. Representative Examples

### Compact Trace Example (Level 3):

```xml
Question: Calculate: ((8 * 7) + 15) - 9

<think>
8 * 7 = 56
56 + 15 = 71
71 - 9 = 62
</think>
<answer>62</answer>
```

### Verbose Trace Example (Level 3):

```xml
Question: Calculate: ((8 * 7) + 15) - 9

<think>
First, 8 multiplied by 7 gives 56.
Then, 56 plus 15 equals 71.
Finally, 71 minus 9 leaves 62.
</think>
<answer>62</answer>
```

---

## 8. Quality Verification Verdict

### **Status: 100% VERIFIED AND AUDITED**

All 26,400 dataset rows across compact and verbose train/dev splits were fully audited by the deterministic arithmetic verification engine. Zero mathematical errors, zero division remainder errors, and zero delimiter syntax faults were detected.
