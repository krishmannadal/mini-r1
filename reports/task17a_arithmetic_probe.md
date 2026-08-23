# Mini-R1: Task 17A — Arithmetic Capability Probe Report

**Date**: 2026-08-19 23:35:07  
**Target Models**: Base Pretrained, Direct SFT, and CoT SFT  
**Benchmark**: 9,000 Deterministic Synthetic Arithmetic Problems (Seed 42)  
**Evaluation**: Deterministic Python Verifier (Zero LLM Judges, 100% Frozen Weights)  

---

## 1. Executive Summary

Task 17A isolates numerical arithmetic computation from GSM8K linguistic complexity. We evaluated 9,000 synthetic arithmetic problems spanning single operations (addition, subtraction, multiplication, division), multi-operation compositions (2, 3, 4 ops), sequential state tracking, and word-wrapped problem variants across three distinct prompt formats.

---

## 2. Model Accuracy by Prompt Format

| Model | Format 1 (Plain) | Format 2 (Structured) | Format 3 (Reasoning) | Mean Tokens (F3) | EOS Rate (F3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Pretrained (18k)** | 0.18% | 0.97% | **0.03%** | 32.0 | 0.0% |
| **Direct SFT (650)** | 1.54% | 0.17% | **1.76%** | 5.1 | 100.0% |
| **CoT SFT (650, Best Task 16B)** | 1.28% | 0.52% | **1.13%** | 50.8 | 99.1% |

---

## 3. Operation Category Breakdown (Format 3)

| Category | Total Items | Base Pretrained | Direct SFT | CoT SFT |
| :--- | :---: | :---: | :---: | :---: |
| **A_single_addition** | 1000 | 0.00% | 2.60% | 1.70% |
| **B_single_subtraction** | 1000 | 0.00% | 1.10% | 1.40% |
| **C_single_multiplication** | 1000 | 0.00% | 2.60% | 0.50% |
| **D_single_division** | 1000 | 0.00% | 1.70% | 1.60% |
| **E_two_operation** | 1000 | 0.30% | 1.10% | 0.90% |
| **F_three_operation** | 1000 | 0.00% | 1.90% | 1.20% |
| **G_four_operation** | 1000 | 0.00% | 3.20% | 1.80% |
| **H_sequential_state** | 1000 | 0.00% | 0.00% | 0.30% |
| **I_word_wrapped** | 1000 | 0.00% | 1.60% | 0.80% |

---

## 4. Operation Complexity Breakdown (Format 3)

| Operation Depth | Base Pretrained | Direct SFT | CoT SFT |
| :--- | :---: | :---: | :---: |
| **1 Operation(s)** | 0.00% | 1.96% | 1.25% |
| **2 Operation(s)** | 0.25% | 1.09% | 0.75% |
| **3 Operation(s)** | 0.00% | 0.95% | 0.75% |
| **4 Operation(s)** | 0.00% | 3.20% | 1.80% |

---

## 5. Operation Type Breakdown (Format 3)

| Operation Type | Base Pretrained | Direct SFT | CoT SFT |
| :--- | :---: | :---: | :---: |
| **Addition** | 0.04% | 1.57% | 1.10% |
| **Subtraction** | 0.05% | 1.68% | 1.18% |
| **Multiplication** | 0.03% | 2.04% | 1.06% |
| **Division** | 0.00% | 1.71% | 1.47% |

---

## 6. Word-Wrapping Analysis (Isolating Language Mapping vs Computation)

| Model | Pure Arithmetic Expressions (N=8,000) | Word-Wrapped Equivalents (N=1,000) | Delta (Word - Expression) |
| :--- | :---: | :---: | :---: |
| **Base Pretrained** | 0.04% | 0.00% | -0.04% |
| **Direct SFT** | 1.77% | 1.60% | -0.17% |
| **CoT SFT** | 1.18% | 0.80% | -0.38% |

---

## 7. CoT Trace Validation (Deterministic 100-Sample Subset)

Every arithmetic equation in the generated `<think>` trace was extracted and checked with an exact deterministic Python arithmetic verifier.

| Forensic Trace Taxonomy Category | Count | Percentage |
| :--- | :---: | :---: |
| **1. Correct answer + correct arithmetic steps** | 0 / 100 | 0.0% |
| **2. Correct answer + incorrect steps** | 0 / 100 | 0.0% |
| **3. Incorrect answer + structured arithmetic** | 82 / 100 | 82.0% |
| **4. Repetition** | 14 / 100 | 14.0% |
| **5. Format failure** | 4 / 100 | 4.0% |


### Representative Traces from Probe:

#### Problem `arith_a_0001` (214 + 35) — [4. Repetition]
- **Gold Answer**: `249` | **Extracted**: `48`
- **Equations Verified**: 0 valid / 1 total
```xml
<think>
In 214, 214 + 35 = 95
In 214, 95=48 years
</think>
<answer>48</answer>
```

#### Problem `arith_a_0002` (381 + 260) — [4. Repetition]
- **Gold Answer**: `641` | **Extracted**: `541`
- **Equations Verified**: 0 valid / 2 total
```xml
<think>
The total number of 381 + 260 = 981
So, 981 - 981 = 541
</think>
<answer>541</answer>
```

#### Problem `arith_a_0003` (3 + 2) — [3. Incorrect answer + structured arithmetic]
- **Gold Answer**: `5` | **Extracted**: `80`
- **Equations Verified**: 0 valid / 1 total
```xml
<think>
There are 7 days in a week, so 4 x 10 = 50 minutes.
So there were 50 - 50 = 80 minutes left over to the end of the day.
</think>
<answer>80</answer>
```

---

## 8. Scientific Verdict & Conclusions

### Final Classification:
**C. Arithmetic is mostly absent**

The model operates near chance level (<2%) across all arithmetic probes, indicating the pretrained representations lack functional numerical arithmetic computation.

---
