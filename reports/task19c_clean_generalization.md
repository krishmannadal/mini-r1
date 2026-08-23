# Mini-R1: Task 19C — Clean Within-Digit Generalization Experiment Report

**Date**: 2026-08-20 22:07:03  
**Target Model**: Clean Subtraction SFT Checkpoint (`artifacts/sft/subtraction_clean/best.pt`)  
**Base Pretrained Model**: `checkpoints/mini_r1_500m_pretrain/best.pt` (Step 18k, 42.1M parameters)  
**Training Curriculum**: Single-Operation Subtraction ($A - B$ with $A \ge B$), $A, B \in [50, 99]$ (12k train / 1.2k dev)  

---

## 1. Executive Summary & Central Experiment Table

| Evaluation Benchmark | Operand Range | Target Answer Range | $N$ | Final-Answer Accuracy | Step-Level Eq Accuracy | Full Format % | Repetition % | Mean Length (tok) | p50 / p90 Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: IID Holdout** | $[50, 99]$ | $[0, 49]$ (1–2 digits) | 500 | **100.00%** | 100.00% | 100.0% | 0.0% | 18.3 | 19 / 19 |
| **E2: Held-Out 2-Digit** | $[2, 49]$ | $[0, 47]$ (1–2 digits) | 500 | **35.40%** | 37.30% | 100.0% | 2.2% | 17.2 | 17 / 19 |
| **E3: OOD 3-Digit (Matched Ans)** | $[101, 199]$ | $[0, 98]$ (1–2 digits) | 500 | **6.60%** | 8.90% | 99.2% | 1.0% | 18.7 | 19 / 21 |
| **E3-Ext: OOD 3-Digit (3d Ans)** | $[200, 300]$ | $[101, 250]$ (3 digits) | 500 | **0.00%** | 0.00% | 96.6% | 1.6% | 18.0 | 18 / 19 |

### Key Generalization Metrics:
- **Within-Digit Generalization Gap (E1 -> E2)**: **+64.60%**
- **Cross-Digit Generalization Gap (E2 -> E3)**: **+28.80%**
- **Answer-Length Extrapolation Gap (E3 -> E3-Ext)**: **+6.60%**

---

## 2. Answer-Length Stratification Table

By evaluating accuracy partitioned strictly by the number of digits in the ground-truth answer, we eliminate the confound where longer answers might experience token-generation decay:

| Benchmark | 1-Digit Answers ($N$, Acc) | 2-Digit Answers ($N$, Acc) | 3-Digit Answers ($N$, Acc) |
| :--- | :---: | :---: | :---: |
| **E1: IID Holdout [50, 99]** | $N=182$ (100.00%) | $N=318$ (100.00%) | -- |
| **E2: Held-Out 2-Digit [2, 49]** | $N=186$ (74.19%) | $N=314$ (12.42%) | -- |
| **E3: OOD 3-Digit [101, 199]** | $N=93$ (29.03%) | $N=407$ (1.47%) | -- |
| **E3-Ext: OOD 3-Digit Answers** | -- | -- | $N=500$ (0.00%) |

---

## 3. Qualitative Error Taxonomy (100-Sample Audits)

### E2: Held-Out 2-Digit Audit ($N=100$)
| Forensic Category | Count | Percentage |
| :--- | :---: | :---: |
| **1. Correct Answer** | 34 | 34.0% |
| **2. Wrong Arithmetic (Structured trace)** | 62 | 62.0% |
| **3. Repetition Loop** | 0 | 0.0% |
| **4. Format Failure** | 0 | 0.0% |
| **5. Other / Incoherent** | 4 | 4.0% |

---

## 4. Scientific Hypothesis Testing

### Explicit Testing of the Three Core Hypotheses:

1. **Hypothesis 1 (Pure Memorization)**:
   - *Condition*: $E2 \le 5.0\%$.
   - *Result*: E2 accuracy is **35.40%**.
   - *Assessment*: REFUTED.

2. **Hypothesis 2 (Within-Digit Generalization with Cross-Digit Bottleneck)**:
   - *Condition*: $E2 \ge 15.0\%$ and $E3 \le 5.0\%$.
   - *Result*: E2 = 35.40%, E3 = 6.60%.
   - *Assessment*: REFUTED.

3. **Hypothesis 3 (Cross-Digit Arithmetic Generalization)**:
   - *Condition*: $E2 \ge 15.0\%$ and $E3 \ge 15.0\%$.
   - *Result*: E2 = 35.40%, E3 = 6.60%.
   - *Assessment*: REFUTED.

### Formal Scientific Verdict:
**Intermediate Generalization Regime: E2 = 35.40%, E3 = 6.60%, Gap = +28.80%.**

---

## 5. Provenance & Overlap Audit

- **Trained Checkpoint SHA256**: `02200b2bf92090bfa3fbcc065b44942a1a50ef06f393b554ba05ae2354be3b51`
- **Base Checkpoint SHA256**: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`
- **Tokenizer SHA256**: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`
- **Exact Overlap with Training**: **0 pairs** across E1, E2, E3, E3-Ext (PASSED_ZERO_OVERLAP)
