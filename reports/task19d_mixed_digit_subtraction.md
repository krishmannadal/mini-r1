# Mini-R1: Task 19D — Mixed 2-Digit + 3-Digit Subtraction Generalization Report

**Date**: 2026-08-20 22:40:36  
**Target Model**: Mixed Subtraction SFT (`artifacts/sft/mixed_subtraction/best.pt`)  
**Base Pretrained Model**: `checkpoints/mini_r1_500m_pretrain/best.pt` (Step 18,000, 42.1M parameters)  
**Training Curriculum**: Single-Operation Subtraction ($A - B$ with $A \ge B$), 12,000 items (6,000 from $[50, 99]$ + 6,000 from $[101, 199]$)  
**Training Convergence**: Best Step 1125 | Dev Loss: 0.0003 | Dev PPL: 1.00  

---

## 1. Central Factorial Results Table

| Evaluation Benchmark | Operand Range | Target Answer Range | $N$ | Final-Answer Accuracy | Step-Level Eq Accuracy | Full Format % | Repetition % | Mean Length (tok) | p50 / p90 Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1a: IID 2-Digit** | $[50, 99]$ | $[0, 49]$ (1–2 digits) | 500 | **100.00%** | 100.00% | 100.0% | 0.0% | 18.3 | 19 / 19 |
| **E1b: IID 3-Digit** | $[101, 199]$ | $[0, 98]$ (1–2 digits) | 500 | **100.00%** | 100.00% | 100.0% | 0.0% | 20.7 | 21 / 21 |
| **E2: OOD 2-Digit** | $[2, 49]$ | $[0, 47]$ (1–2 digits) | 500 | **28.40%** | 33.99% | 98.4% | 0.2% | 17.4 | 17 / 19 |
| **E3: OOD 3-Digit** | $[200, 299]$ | $[0, 98]$ (1–2 digits) | 500 | **34.40%** | 34.60% | 100.0% | 0.0% | 20.5 | 21 / 21 |
| **E4: Acid Test 4-Digit** | $[1001, 1999]$ | $[0, 998]$ (1–3 digits) | 500 | **0.40%** | 2.02% | 97.6% | 3.4% | 24.4 | 22 / 23 |

---

## 2. Generalization Ratios & Longitudinal Comparisons with Task 19C

| Metric | Task 19C (2-Digit Only SFT) | Task 19D (Mixed 2+3-Digit SFT) | Absolute $\Delta$ |
| :--- | :---: | :---: | :---: |
| **IID 2-Digit (E1 / E1a)** | 100.00% | **100.00%** | +0.00% |
| **IID 3-Digit (E1b)** | — (Untrained) | **100.00%** | New Capability |
| **OOD 2-Digit Transfer (E2)** | 35.40% | **28.40%** | **-7.00%** |
| **OOD 3-Digit Transfer (E3)** | 6.60% | **34.40%** | **+27.80%** |
| **Acid Test 4-Digit (E4)** | 0.00% | **0.40%** | **+0.40%** |
| **2-Digit Generalization Ratio (G_2 = E2 / E1a)** | 0.354 | **0.284** | — |
| **3-Digit Generalization Ratio (G_3 = E3 / E1b)** | — | **0.344** | — |

---

## 3. Answer-Length Stratification Table

| Benchmark | 1-Digit Answers ($N$, Acc) | 2-Digit Answers ($N$, Acc) | 3-Digit Answers ($N$, Acc) |
| :--- | :---: | :---: | :---: |
| **E1a: IID 2-Digit [50, 99]** | $N=174$ (100.00%) | $N=326$ (100.00%) | — |
| **E1b: IID 3-Digit [101, 199]** | $N=84$ (100.00%) | $N=416$ (100.00%) | — |
| **E2: OOD 2-Digit [2, 49]** | $N=194$ (53.61%) | $N=306$ (12.42%) | — |
| **E3: OOD 3-Digit [200, 299]** | $N=108$ (93.52%) | $N=392$ (18.11%) | — |
| **E4: Acid Test 4-Digit [1001, 1999]** | $N=9$ (0.00%) | $N=81$ (2.47%) | $N=410$ (0.00%) |

---

## 4. Qualitative Failure Taxonomy ($N=200$ Samples from E3 and E4)

### A. Classification of Generations
| Category | E3 (OOD 3-Digit) Count (%) | E4 (Acid Test 4-Digit) Count (%) |
| :--- | :---: | :---: |
| **1. Correct Answer + Subtraction** | 53 (26.5%) | 2 (1.0%) |
| **2. Structured Incorrect Subtraction** | 147 (73.5%) | 191 (95.5%) |
| **3. Repetition Loop** | 0 (0.0%) | 0 (0.0%) |
| **4. Format Failure** | 0 (0.0%) | 7 (3.5%) |
| **5. Other / Incoherent** | 0 (0.0%) | 0 (0.0%) |

### B. Micro-Error Breakdown on Incorrect Outputs
| Error Mechanism | E3 (OOD 3-Digit) | E4 (Acid Test 4-Digit) |
| :--- | :---: | :---: |
| **Correct Units Digit ($A \% 10 - B \% 10$)** | 92.3% | 22.4% |
| **Correct Magnitude (within $\pm 25\%$)** | 79.0% | 4.2% |
| **Borrow Failure (Direct $|A_i - B_i|$)** | 0.0% | 0.0% |
| **Collapsed to Seen Training Range ($[0, 98]$)** | 100.0% | 99.0% |

---

## 5. Scientific Interpretation & Verdict

### Classification:
**A: Conclusion A: 3-Digit exposure enables meaningful transfer to unseen 3-digit subtraction ranges.**

### Detailed Scientific Analysis:
1. **Capacity vs Bounded Representation**:
   - Splitting the 12,000 dataset 50/50 between 2-digit and 3-digit subtraction yielded near-perfect memorization on both IID holdout sets (**100.00%** on E1a, **100.00%** on E1b). The 42.1M parameter model possesses ample capacity to represent two disjoint arithmetic domains simultaneously.
2. **Transfer Across Operand Ranges**:
   - On the unseen 3-digit range $[200, 299]$ (E3), accuracy reached **34.40%** (compared to $6.60\%$ in Task 19C, a $\Delta = +27.80\%$ improvement).
   - On the unseen 2-digit range $[2, 49]$ (E2), accuracy is **28.40%** (compared to $35.40\%$ in Task 19C).
3. **Cross-Digit Extrapolation (The Acid Test)**:
   - On completely untrained 4-digit subtraction $[1001, 1999]$ (E4), accuracy is **0.40%**.
   - The qualitative audit reveals that on 4-digit problems, 99.0% of model predictions collapse into the 1–2 digit range $[0, 98]$ seen during training, confirming that positional token length and output token bounding are dominant structural barriers.

---

## 6. Provenance & Overlap Audit

- **Trained Checkpoint SHA256**: `7ee14e348813aaf6931aad73ca75b3dfadd0cf9d95a43d16a553a94a99d3b505`
- **Base Pretrained Checkpoint SHA256**: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`
- **Tokenizer SHA256**: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`
- **Train/Dev/Eval Overlap**: **0 pairs** across all 5 evaluation benchmarks (PASSED_ZERO_OVERLAP)
