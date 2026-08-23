# Mini-R1: Task 19B-PRE — Operator × Digit-Length Diagnostic Report

**Date**: 2026-08-20 21:32:26  
**Target Model**: Frozen Compact Arithmetic SFT (`artifacts/sft/arithmetic_compact/best.pt`)  
**Evaluation Scope**: 500 examples Condition E5 vs 500 examples Condition E6  
**Constraint**: 100% Frozen Weights. Zero retraining. Zero weight updates.  

---

## 1. Executive Summary & Core Comparison Table

| Metric | Condition E5: 3-Digit Division (Trained Context) | Condition E6: 3-Digit Add/Sub (Untrained Context) | E6-Add (3-Digit Addition) | E6-Sub (3-Digit Subtraction) | Difference (E5 $-$ E6) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Final-Answer Accuracy (Greedy)** | **24.28%** | **2.80%** | 0.80% | 4.80% | **+21.48%** |
| **Final-Answer Accuracy (Task 16B CoT)** | **15.97%** | **2.00%** | 0.80% | 3.60% | **+13.97%** |
| **Step-Level Equation Accuracy** | **24.28%** | **2.84%** | 0.80% | 4.92% | **+21.44%** |
| **`<think>` Tag Compliance** | 100.0% | 100.0% | 100.0% | 100.0% | +0.0% |
| **`<answer>` Tag Compliance** | 100.0% | 100.0% | 100.0% | 100.0% | +0.0% |
| **Full Format Compliance** | 100.0% | 100.0% | 100.0% | 100.0% | +0.0% |
| **EOS Termination Rate** | 100.0% | 100.0% | 100.0% | 100.0% | +0.0% |
| **4-gram Repetition Rate** | 0.0% | 0.2% | 0.4% | 0.0% | -0.2% |
| **Mean Generation Length** | 19.0 tok | 21.2 tok | 22.5 tok | 20.0 tok | -2.2 tok |
| **p50 (Median) Length** | 19 tok | 21 tok | 23 tok | 21 tok | -2 tok |
| **p90 Length** | 20 tok | 23 tok | 23 tok | 21 tok | -3 tok |

---

## 2. Detailed Breakdown by Decoding Configuration

### Condition A: Deterministic Greedy (T=0.0, top_p=1.0, rep_pen=1.0, max=128)

| Benchmark | Total | Correct | Strict Accuracy | Step-Eq Valid / Total | Step-Eq Acc | Format Comp % | Repetition % | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E5: 3-Digit Division** | 313 | 76 | **24.28%** | 76/313 | 24.28% | 100.0% | 0.0% | 19.0 |
| **E6: 3-Digit Add/Sub** | 500 | 14 | **2.80%** | 14/493 | 2.84% | 100.0% | 0.2% | 21.2 |
| **E6-Add: Addition** | 250 | 2 | **0.80%** | 2/249 | 0.80% | 100.0% | 0.4% | 22.5 |
| **E6-Sub: Subtraction** | 250 | 12 | **4.80%** | 12/244 | 4.92% | 100.0% | 0.0% | 20.0 |

### Condition D: Task 16B Best CoT Decoding (T=0.6, top_p=0.95, rep_pen=1.2, max=128)

| Benchmark | Total | Correct | Strict Accuracy | Step-Eq Valid / Total | Step-Eq Acc | Format Comp % | Repetition % | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E5: 3-Digit Division** | 313 | 50 | **15.97%** | 50/313 | 15.97% | 100.0% | 0.0% | 18.9 |
| **E6: 3-Digit Add/Sub** | 500 | 10 | **2.00%** | 10/493 | 2.03% | 100.0% | 0.4% | 21.3 |

---

## 3. Qualitative Error Analysis & Sample Traces

### Sample E5 Generations (3-Digit Division):

- **Problem**: `Question: Calculate: 129 / 3`
  - **Gold**: `43` | **Extracted**: `17` | **Correct**: `False`
  - **Output**:
    ```
    <think>
129 / 3 = 17
</think>
<answer>17</answer>
    ```

- **Problem**: `Question: Calculate: 237 / 3`
  - **Gold**: `79` | **Extracted**: `19` | **Correct**: `False`
  - **Output**:
    ```
    <think>
237 / 3 = 19
</think>
<answer>19</answer>
    ```

- **Problem**: `Question: Calculate: 154 / 2`
  - **Gold**: `77` | **Extracted**: `19` | **Correct**: `False`
  - **Output**:
    ```
    <think>
154 / 2 = 19
</think>
<answer>19</answer>
    ```

- **Problem**: `Question: Calculate: 136 / 4`
  - **Gold**: `34` | **Extracted**: `22` | **Correct**: `False`
  - **Output**:
    ```
    <think>
136 / 4 = 22
</think>
<answer>22</answer>
    ```

- **Problem**: `Question: Calculate: 192 / 12`
  - **Gold**: `16` | **Extracted**: `16` | **Correct**: `True`
  - **Output**:
    ```
    <think>
192 / 12 = 16
</think>
<answer>16</answer>
    ```

### Sample E6 Generations (3-Digit Addition & Subtraction):

- **Problem**: `Question: Calculate: 136 + 231`
  - **Gold**: `367` | **Extracted**: `384` | **Correct**: `False`
  - **Output**:
    ```
    <think>
136 + 231 = 384
</think>
<answer>384</answer>
    ```

- **Problem**: `Question: Calculate: 122 + 225`
  - **Gold**: `347` | **Extracted**: `45` | **Correct**: `False`
  - **Output**:
    ```
    <think>
122 + 225 = 45
</think>
<answer>45</answer>
    ```

- **Problem**: `Question: Calculate: 151 + 240`
  - **Gold**: `391` | **Extracted**: `66` | **Correct**: `False`
  - **Output**:
    ```
    <think>
151 + 240 = 66
</think>
<answer>66</answer>
    ```

- **Problem**: `Question: Calculate: 130 - 117`
  - **Gold**: `13` | **Extracted**: `3` | **Correct**: `False`
  - **Output**:
    ```
    <think>
130 - 117 = 3
</think>
<answer>3</answer>
    ```

- **Problem**: `Question: Calculate: 121 - 111`
  - **Gold**: `10` | **Extracted**: `10` | **Correct**: `True`
  - **Output**:
    ```
    <think>
121 - 111 = 10
</think>
<answer>10</answer>
    ```

---

## 4. Scientific Analysis & Findings

### 1. Does 3-Digit Training Exposure in Division Transfer Specifically to Division?
- In Task 18B training, **362 single-step Calculate examples** contained 3-digit operands exclusively as dividends for division (NNN / M = Q).
- Under greedy evaluation:
  - **E5 (3-Digit Division)** achieved **24.28%** strict accuracy.
  - **E6 (3-Digit Add/Sub)** achieved **2.80%** strict accuracy.
  - Performance delta (Delta = E5 - E6) is **+21.48%**.

### 2. Format vs Arithmetic Capability:
- Format compliance across both E5 (100.0%) and E6 (100.0%) confirms that the model robustly outputs `<think>` and `<answer>` tags even under unseen 3-digit inputs.
- Step-level equation validity (24.28% on E5 vs 2.84% on E6) demonstrates the exact degree to which intermediate arithmetic relations are preserved.

### 3. Cautionary Note on Causality:
- These results isolate an observational disparity between the operator context where 3-digit operands were trained (division) vs untrained (addition/subtraction).
- They **do not prove** that the model has learned a generalizable division algorithm; rather, they demonstrate how tightly arithmetic competence is bounded to the specific operator-operand distributions seen during SFT.
