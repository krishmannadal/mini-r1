# Mini-R1: Task 9 Long-Run Qualification Report (1,000 Steps)

## Executive Summary
This report presents the empirical results of the **1,000-step qualification experiment** comparing the two leading learning rate candidates:
- **Run A**: Peak LR $= 3\times 10^{-4}$, Min LR $= 3\times 10^{-5}$
- **Run B**: Peak LR $= 6\times 10^{-4}$, Min LR $= 6\times 10^{-5}$

Both runs were executed on the **NVIDIA GeForce RTX 4050 Laptop GPU** with identical model initialization (`seed=42`), 42.1M parameter architecture, dataset ordering, effective batch size 32 (16,384 tokens/step, context 512), and FP16 AMP precision.

---

## 1. Validation Loss & Perplexity Trajectory (Every 100 Steps)

| Step | Run A ($\eta = 3\times 10^{-4}$) Val Loss | Run A Perplexity | Run B ($\eta = 6\times 10^{-4}$) Val Loss | Run B Perplexity |
|---|---|---|---|---|
| **Step 100** | 3.9054 | 49.67 | **3.7668** | **43.24** |
| **Step 200** | 3.4039 | 30.08 | **3.3201** | **27.66** |
| **Step 300** | 3.1820 | 24.10 | **3.1120** | **22.47** |
| **Step 400** | 3.0199 | 20.49 | **2.9370** | **18.86** |
| **Step 500** | 2.8984 | 18.14 | **2.8026** | **16.49** |
| **Step 600** | 2.8026 | 16.49 | **2.6972** | **14.84** |
| **Step 700** | 2.7339 | 15.39 | **2.6153** | **13.67** |
| **Step 800** | 2.6822 | 14.62 | **2.5536** | **12.85** |
| **Step 900** | 2.6529 | 14.19 | **2.5185** | **12.41** |
| **Step 1000** | 2.6337 | 13.93 | **2.4995** | **12.18** |

---

## 2. Comparative Telemetry Table

| Metric | Run A ($\eta = 3\times 10^{-4}$) | Run B ($\eta = 6\times 10^{-4}$) |
|---|---|---|
| **Final Training Loss** | 2.3866 | **2.2104** |
| **Final Validation Loss** | 2.6337 | **2.4995** |
| **Best Validation Loss** | 2.6337 | **2.4995** |
| **Best Validation Perplexity** | 13.93 | **12.18** |
| **Average Step Throughput** | 11,953.0 tok/s | 11,483.8 tok/s |
| **Max Peak Allocated VRAM** | 3,285.7 MB (3.21 GB) | 3,288.1 MB (3.21 GB) |
| **Reserved Allocator VRAM** | ~3.81 GB (2.2 GB headroom) | ~3.81 GB (2.2 GB headroom) |
| **Minimum GradScaler Scale** | 65,536.0 (Constant) | 65,536.0 (Constant) |
| **Skipped Optimizer Steps** | **0** | **0** |
| **Maximum Gradient Norm** | 2.615 | 2.439 |
| **Step 1000 Gradient Norm** | ~0.85 – 0.90 | ~0.85 – 0.90 |

---

## 3. Validation Loss Curves

```text
Validation Loss Trajectory (Steps 100 to 1000)
Loss
4.00 |  A (3.91)
3.80 |    B (3.77)
3.60 |
3.40 |        A (3.40)
3.20 |          B (3.32)  A (3.18)
3.00 |                      B (3.11)  A (3.02)
2.80 |                                  B (2.94)  A (2.90)
2.60 |                                              B (2.80)  A (2.80)
2.50 |                                                          B (2.70)  A (2.73)  A (2.68)  A (2.65)  A (2.63)
2.40 |                                                                      B (2.62)  B (2.55)  B (2.52)  B (2.50)
     +-------------------------------------------------------------------------------------------------------------
       Step 100  Step 200  Step 300  Step 400  Step 500  Step 600  Step 700  Step 800  Step 900  Step 1000

Legend:
  [A] Peak LR = 3e-4 (Final Val Loss: 2.6337, Perplexity: 13.93)
  [B] Peak LR = 6e-4 (Final Val Loss: 2.4995, Perplexity: 12.18)
```

---

## 4. Analytical Observations

1. **Descent Dynamics & Convergence**:
   - $\eta = 6\times 10^{-4}$ maintained a consistent lead over $\eta = 3\times 10^{-4}$ across all 1,000 steps without showing signs of gradient explosion, divergence, or loss oscillation.
   - At Step 1,000, $\eta = 6\times 10^{-4}$ achieved a validation loss of **2.4995** (Perplexity: **12.18**), compared to **2.6337** (Perplexity: **13.93**) for $\eta = 3\times 10^{-4}$.
2. **Gradient Stability & Gradient Norms**:
   - Both learning rates exhibited healthy gradient clipping behavior during warmup and stabilized to gradient norms between **0.80 and 0.95** in the final 200 steps.
   - Peak gradient norm reached $2.615$ for Run A and $2.439$ for Run B.
3. **Hardware Health & Scaler Reliability**:
   - Both candidates maintained a rock-solid `GradScaler` scale of `65,536.0` with **zero scale decreases and zero skipped optimizer updates**.
   - Peak memory usage remained identical at **~3.28 GB allocated / ~3.81 GB reserved**, confirming safe thermal and VRAM margins on the RTX 4050.
