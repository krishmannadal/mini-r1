# Mini-R1: Task 24 Parts 2–4 — Final Benchmarks, Comparison, Claim Audit

**TASK 24 PARTS 2–4 STATUS: COMPLETED**

**Model**: `artifacts/sft/task23_75_25/best.pt` (SHA256 `d630daac…e49c`)
**Raw data**: `reports/task24_final_benchmarks.json`

Decoding: **greedy** for all suites below (matches the Task 20/21B/22/23 measurement contract). GSM8K is additionally reported under the Task 16B best sampled configuration (T=0.6, top_p=0.95, rep_pen=1.2). GSM8K, the 2D/3D/4D retention probes and general-language PPL are carried forward unmodified from `reports/task23_replay_ratio.json` — the checkpoint SHA was asserted equal before reuse, so these are the identical numbers the model was selected on.

---

## Part 2 — Final benchmark results

### 2.1 Reasoning benchmarks

| Benchmark | N | Strict acc | Equation validity | `<think>` | `<answer>` | Full format | EOS | Repetition | Mean len | p90 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **GSM8K test** (greedy) | 1,319 | **2.43%** | **31.64%** | 98.2% | 98.2% | 98.2% | 98.2% | 77.3% | 44.7 | 61 |
| **GSM8K test** (sampled T=0.6) | 1,319 | 1.21% | 24.21% | 100.0% | 99.9% | 99.9% | 100.0% | 16.4% | 27.0 | 37 |
| **SVAMP test** (greedy) | 300 | **1.67%** | 22.09% | 100.0% | 100.0% | 100.0% | 100.0% | 49.7% | 32.4 | 43 |
| **AQuA test** (greedy) | 254 | **0.00%** | 14.75% | 97.2% | 26.0% | **0.0%** | 97.2% | 15.4% | 16.0 | 30 |

**AQuA note**: full-format compliance is **0.0%** — on multiple-choice prompts the model opens `<think>` but almost never emits a well-formed `<answer>` block (26.0% answer compliance), producing only 0.24 equations per generation. It does not transfer to the MCQ format at all.

### 2.2 Arithmetic suites (E1–E7), against the Task 20 source model

| Suite | Regime | N | **Final model** | Task 20 model | Δ | Eq. validity | Repetition | Mean len |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | IID 2D `[50,99]` | 500 | **89.60%** | 100.00% | −10.40 | 89.60% (448/500) | 0.0% | 18.3 |
| E2 | OOD 2D | 500 | **20.00%** | 18.20% | **+1.80** | 19.80% (99/500) | 0.0% | 17.1 |
| E3 | IID 3D `[101,199]` | 500 | **84.60%** | 99.80% | −15.20 | 84.60% (423/500) | 0.0% | 20.7 |
| E4 | OOD 3D | 500 | **16.60%** | 26.20% | −9.60 | 16.60% (83/500) | 0.0% | 20.4 |
| E5 | IID 4D `[1001,1999]` | 500 | **38.40%** | 67.20% | −28.80 | 38.40% (192/500) | 0.6% | 24.6 |
| E6 | OOD 4D | 500 | **4.60%** | 8.60% | −4.00 | 4.60% (23/500) | 1.8% | 24.5 |
| E7 | **ACID 5D** (never trained) | 500 | **0.60%** | 0.00% | +0.60 | 0.60% (3/500) | 18.6% | 27.1 |

All E1–E7 suites: **100.0% EOS, 100.0% `<think>`, 100.0% `<answer>`, 100.0% full-format, exactly 1.00 equation per generation**. Equation validity tracks strict accuracy exactly on six of seven suites (single-step problems); E2 differs by one item (19.80% vs 20.00%) where the answer tag was right but the emitted equation was not.

Repetition rises with digit count (0.0% → 18.6% at 5 digits), a legible signature of the model falling back on digit-copying loops once the regime exceeds what it was trained on.

### 2.3 Retention probe (generated, seed 42) and general language

| Metric | Final model | Task 20 source |
| :--- | ---: | ---: |
| 2D subtraction retention | **94.80%** | 100.00% |
| 3D subtraction retention | **78.20%** | 99.80% |
| 4D subtraction retention | **32.40%** | 67.20% |
| General-language PPL (10M held-out tokens) | **50.74** | 27.63 |

The generated retention probe (94.80/78.20/32.40) and the fixed E1/E3/E5 files (89.60/84.60/38.40) measure the same three regimes with different item samples; both are reported and neither is preferred.

---

## Part 3 — Final model comparison

All numbers from stored JSON artifacts. Negative results retained.

| Model | GSM8K | SVAMP | AQuA | Eq. validity (greedy) | 2D ret | 3D ret | 4D ret | 5D acid | General PPL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base pretrained (42.1M, 500M tok) | 0.76% | — | — | — | — | — | — | — | **17.68** |
| Direct GSM8K SFT | **2.35%** | 2.00% | 0.00% | — | 4.40% | 1.60% | 0.00% | — | 21.83 |
| Verbose CoT SFT | 1.06% | 0.33% | 0.00% | — | 1.40% | 0.00% | 0.00% | — | 23.58 |
| Task 20 arithmetic model | — | — | — | 100.00%¹ | **100.00%** | **99.80%** | **67.20%** | 0.00% | 27.63 |
| Task 21A arithmetic→verbose GSM8K | 1.44% | 2.00% | 0.00% | — | 3.60% | 0.60% | 0.40% | — | 28.87 |
| Task 21B arithmetic→compact (Model A) | 0.91% | 3.67% | 0.00% | 23.67% | 4.40% | 5.20% | 0.60% | — | 43.88 |
| Task 22 replay + frozen (Model E) | 1.52% | — | — | 12.19% | 5.60% | 4.20% | 0.00% | — | 30.33 |
| Task 22B replay 50% trainable (Model F) | 2.27% | — | — | **42.56%** | 3.80% | 1.80% | 0.00% | — | 60.02 |
| Task 23 replay 10% (Model G) | 2.12% | — | — | 31.25% | 65.60% | 54.80% | 13.40% | — | 47.90 |
| **▶ Task 23 replay 25% (Model H) — FINAL** | **2.43%** | **1.67%** | **0.00%** | **31.64%** | **94.80%** | **78.20%** | **32.40%** | **0.60%** | 50.74 |

¹ Task 20 E1 step-level equation accuracy on single-step arithmetic.
GSM8K figures are greedy for Task 21B onward; Direct/CoT/21A rows are from the Task 21B evaluation suite. "—" = not measured for that arm.

**Honest reading.** The final model leads on arithmetic retention by a very wide margin and is at the top of a narrow GSM8K band (2.43% vs Direct SFT's 2.35% — a 1-problem difference, not a real gap). It is **not** best on equation validity (Model F, 42.56%), and it has the **second-worst general PPL** of any model in the project (50.74; only Model F is worse at 60.02). AQuA is 0.00%, unchanged from every other arm.

---

## Part 4 — Scientific claim audit

Every claim below was checked against the JSON artifacts named beside it.

### PROVEN

| Claim | Evidence | Source |
| :--- | :--- | :--- |
| Verified arithmetic SFT creates strong IID arithmetic | E1 100.00%, E3 99.80%, E5 67.20% | `task20_progressive_numeric_curriculum.json` |
| Arithmetic generalization improves with exposure to numeric regimes | 2D-only SFT → mixed 2D+3D → progressive 2D+3D+4D raised IID 4D from untrained to 67.20%; OOD 3D 26.20% | `task19c/19d/20*.json` |
| GSM8K SFT causes major arithmetic forgetting | 100.0/99.8/67.2 → 3.6/0.6/0.4 (21A), 4.4/5.2/0.6 (21B), 3.8/1.8/0.0 (22B) | `task21a/21b/task22b_*.json` |
| Parameter freezing caused severe generation degeneration in Task 22 | Frozen: 72.8% full-format, 99.9% repetition, p90 = 256 (cap). Same data unfrozen: 98.6%, 84.5%, p90 = 59. Equation validity +30.38 pts | `task22_replay_sft.json`, `task22b_replay_only.json` |
| Target-distribution replay substantially preserves arithmetic | 0% replay → 2D 4.40%; 10% → 65.60%; 25% → **94.80%** | `task23_replay_ratio.json` |
| Replay composition matters | 50% replay *without* target-regime data (Model F) → 2D 3.80%; 25% replay *with* it → 94.80% | `task22b_*.json`, `task23_*.json` |

### NOT PROVEN

| Claim | Why not |
| :--- | :--- |
| General symbolic reasoning | GSM8K 2.43%, SVAMP 1.67%, **AQuA 0.00%**. No benchmark shows competence above the noise floor. |
| Arbitrary arithmetic extrapolation | E7 5-digit acid test **0.60%** (3/500). OOD 4D 4.60%. Ability is regime-local, not general. |
| RL improvement | No RL was ever run in this project. Zero evidence either way. |
| Architectural scaling benefit | One architecture (42.1M) only. No scaling comparison exists. |
| Optimal replay ratio | Confounded: the 0/10/25/50 series varies replay *composition* as well as share (Model F replays no target-regime data). Ratio and composition cannot be separated. 25% is the best **tested** point, not a demonstrated optimum. |

### Additional limitation carried forward from Task 22

The evaluation harness left-pads prompts and does not pass an attention mask, while the model uses learned absolute position embeddings. A batch-size-1 control (N=200) measured this as worth roughly +3 to +14 points of equation validity depending on the model. All arms were measured identically so comparisons are valid, but **absolute** benchmark numbers are depressed by this artifact. Source: `task22_padding_control.json`.
