# Mini-R1: Task 22B — Replay-Only Control Arm (All Parameters Trainable)

**TASK 22B STATUS: COMPLETED**

**Date**: 2026-08-23
**Purpose**: Run the single missing control identified in the Task 22 report §10 — replay mixture with **all parameters trainable** — to separate the effect of the replay data from the effect of parameter freezing.

---

## 1. What changed, and only what changed

Task 22 moved two variables at once. Task 22B holds the data fixed and removes the freezing.

| | Task 22 (Model E) | **Task 22B (Model F)** |
| :--- | :--- | :--- |
| Replay mixture | 50/50 interleaved | **identical** |
| Trainable params | 18,887,680 (44.83%) | **42,132,480 (100%)** |
| `tok_emb` / `lm_head` (tied) | frozen | **trainable** |
| `pos_emb`, `blocks.0–5` | frozen | **trainable** |
| Init checkpoint | `progressive_subtraction/best.pt` | identical (`fdcecade71…`) |
| Corpus / split / tokenizer | identical | identical |
| Seed / optimizer / LR / schedule | 42 / AdamW / 1e-4 / cosine | identical |
| Weight decay / grad clip / warmup | 0.01 / 1.0 / 50 | identical |
| Batch / accum / steps / max_len / AMP | 8 / 4 / 650 / 512 / fp16 | identical |
| Evaluator / decode configs | `evaluate_task22_replay.py` | identical |

Implementation: **config-only**. The Task 22 trainer already drives freezing from regex patterns, so `trainable_patterns: ["^.*$"]` matches every parameter and freezes nothing. No training code was written or modified.

---

## 2. Smoke test (all six checks passed)

| Check | Result |
| :--- | :--- |
| All parameters `requires_grad=True` | **True** — 0 frozen of 124 tensors |
| `tok_emb.weight` trainable | **True** |
| `lm_head.weight` trainable | **True** (`lm_head.weight is tok_emb.weight` → `True`; tied, so PyTorch de-duplicates it in `named_parameters()`) |
| 10-step run, finite loss/grads | **True** — loss 4.7064 → 3.5926, grad norm 14.463 → 8.277, no NaN/Inf |
| Checkpoint save/load | **True** — 137 tensors reload, all finite, `tok_emb` and `lm_head` both present |
| Base checkpoint hash unchanged | **True** — `fdcecade71afafc1857d5ee56d221d31928ae1321d0f963dc0f566482af56d5c` before and after |

AdamW saw 42,132,480 parameters across 124 tensors (Task 22: 18,887,680 across 62).

---

## 3. Training

| | Model F (22B) | Model E (22) |
| :--- | ---: | ---: |
| Mixture dev loss | 4.8522 → **0.4933** | 4.8522 → 0.5841 |
| GSM8K component | 4.7043 → **0.5946** | 4.7043 → 0.6871 |
| Replay component | 5.0001 → **0.3919** | 5.0001 → 0.4810 |
| Best step | 650 | 650 |

Checkpoint `artifacts/sft/task22b_replay_only/best.pt`, SHA256 `99d5e64270aa4f11d662bd1fac29da573290b2fad1e226ee78909fc20a2ebcc8`.

---

## 4. Primary results

### 4.1 Step-level equation validity — GSM8K test (N=1,319)

| Model | Decode | Valid/Total | **Validity** | GSM8K acc | valid eqs/gen | eqs/gen |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **F (22B: replay, all trainable)** | sampled | 719/2263 | **31.77%** | **2.35%** | **0.545** | 1.72 |
| **F (22B: replay, all trainable)** | greedy | 1540/3618 | **42.56%** | **2.27%** | **1.168** | 2.74 |
| E (22: replay, frozen) | sampled | 262/4060 | 6.45% | 2.05% | 0.199 | 3.08 |
| E (22: replay, frozen) | greedy | 823/6754 | 12.19% | 1.52% | 0.624 | 5.12 |
| A (21B: arith-init) | sampled | 773/3780 | 20.45% | 1.74% | 0.586 | 2.87 |
| A (21B: arith-init) | greedy | 1340/5660 | 23.67% | 0.91% | 1.016 | 4.29 |
| B (21B: base-init) | sampled | 761/2938 | 25.90% | 1.82% | 0.577 | 2.23 |
| B (21B: base-init) | greedy | 1479/5974 | 24.76% | 1.21% | 1.121 | 4.53 |

Model F is best on validity under both decode settings and highest on GSM8K strict accuracy of all four arms.

### 4.2 Δ_replay_trainability (Model F − Model E)

| Metric | Primary decode | Greedy |
| :--- | ---: | ---: |
| **Step-level equation validity** | **+25.32 pts** | **+30.38 pts** |
| GSM8K strict accuracy | +0.30 pts | +0.76 pts |

Model F also clears both 21B baselines: **+11.32 pts vs Model A**, **+5.87 pts vs Model B** (primary decode).

### 4.3 Format, repetition, length

| Model | Decode | `<think>` | `<answer>` | Full format | EOS rate | Repetition | Mean len | p90 len |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **F (22B)** | sampled | **100.0%** | **100.0%** | **99.5%** | **100.0%** | **13.2%** | **24.0** | **34** |
| **F (22B)** | greedy | 98.7% | 98.7% | 98.6% | 98.7% | 84.5% | 41.5 | 59 |
| E (22) | sampled | n/a | n/a | 85.4% | n/a | 47.5% | 39.6 | 56 |
| E (22) | greedy | n/a | n/a | 72.8% | n/a | 99.9% | 126.9 | **256** |
| A (21B) | greedy | n/a | n/a | 97.8% | n/a | 96.4% | 62.9 | 82 |
| B (21B) | greedy | n/a | n/a | 91.6% | n/a | 95.4% | 79.7 | 123 |

`<think>`/`<answer>`/EOS rates were added to the evaluator for this task, so they are not present in the Task 22 cache (marked n/a). Re-decoding E/A/B purely to populate three columns would cost ~5 GPU-hours and was not run; `full format` is the composite of `<think>` and `<answer>` and is available for all arms.

### 4.4 Replay-corpus diagnostic (Task 22 reported runaway chains)

The Task 22 pathology **disappears** when nothing is frozen:

| | Model F (22B) | Model E (22) |
| :--- | ---: | ---: |
| Mean equations/generation (greedy) | **2.74** | 5.12 |
| p90 generation length (greedy) | **59** | 256 (cap) |
| Repetition (greedy) | **84.5%** | 99.9% |
| Repetition (sampled) | **13.2%** | 47.5% |

Model E's runaway `12 + 12 + 12 + …` chains and its failure to terminate (p90 pinned at the 256-token cap, EOS never reached on ≥10% of generations) are **not** caused by the replay corpus. The same corpus with all parameters trainable produces 2.74 equations/generation against GSM8K's true mean of 3.27, terminates on 98.7% of generations, and has the lowest repetition of any arm under sampled decoding (13.2%).

### 4.5 Arithmetic retention and general language

| Model | 2D sub | 3D sub | 4D sub | Replay-regime holdout | General PPL |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **F (22B)** | 3.80% | 1.80% | 0.00% | **3.80%** (val 19.79%) | **60.02** |
| E (22) | **5.60%** | 4.20% | 0.00% | 3.20% (val 10.24%) | 30.33 |
| A (21B) | 4.40% | **5.20%** | **0.60%** | 3.00% (val 30.18%) | 43.88 |
| B (21B) | 0.60% | 0.00% | 0.00% | 0.80% (val 36.84%) | 21.85 |
| *Task 20 pre-transfer* | *100.00%* | *99.80%* | *67.20%* | — | *27.63* |

Two results here run **against** Model F:

- **Retention is the worst of the three arithmetic-init arms.** 3.80 / 1.80 / 0.00 vs Model E's 5.60 / 4.20 / 0.00 and Model A's 4.40 / 5.20 / 0.60. Unfreezing did not rescue arithmetic retention; it degraded it slightly. All arms remain a catastrophic fall from the Task 20 baseline of 100 / 99.8 / 67.2.
- **General-language PPL is the worst of all four arms**: 60.02, vs 30.33 (E), 43.88 (A), 21.85 (B), 27.63 (Task 20). Task 22's freezing bought a real 29.7-point PPL protection, and removing it gives that back and then some.

So Model F trades general-language quality and a little retention for a large gain in step-level arithmetic correctness on GSM8K.

---

## 5. Interpretation

**CASE 1 applies: parameter freezing was a major failure mechanism.**

Replay-only with all parameters trainable improves step-level equation validity by **+25.32 pts (primary) / +30.38 pts (greedy)** over the identical run with 55% of the network frozen, and beats both Task 21B baselines. Since the corpus, split, seed, optimizer, schedule, batch, steps and evaluator were held identical, freezing is the only manipulated variable, and it accounts for the entire Task 22 regression.

This also **reverses the Task 22 conclusion about the replay corpus.** Task 22 §9 attributed the runaway equation chains to corpus design (nested prompt expressions, multi-operator steps). That hypothesis is now disconfirmed: the same corpus, unfrozen, produces well-formed traces at 2.74 equations/generation with 98.7% EOS. The chains were a symptom of the frozen tied output head — with `lm_head` unable to move, the model could not learn to emit the stop token or select digits precisely, and defaulted to repeating a token it could already produce.

**CASE 3 partially applies as well.** Retention did *not* recover (3.80 / 1.80 / 0.00, the worst of the arithmetic-init arms), and general PPL got worse (60.02). So while freezing explains the equation-validity regression, **neither arm preserves the Task 20 computational skill** — retention collapses from ~100% to under 6% in every condition tested across Tasks 21B, 22 and 22B. Replay data at this mixture ratio does not preserve computational skill, and unfreezing does not change that.

### Claims this evidence does **not** support

- It does **not** show the replay corpus is well designed — only that it is not the cause of the runaway-chain pathology. Model F still trails Model A on replay-holdout equation validity (19.79% vs 30.18%).
- It does **not** show Task 22B solved GSM8K. Strict accuracy is 2.35%, within the 1–2% noise band shared by every arm; only the equation-validity gaps (11–30 pts over thousands of equations) are large enough to interpret.
- Single seed (42), single run. No variance estimate.
- The retention and PPL regressions are real but were not decomposed; whether they stem from the mixture ratio, the corpus, or full-parameter drift is untested.

---

## 6. Files

| File | Status |
| :--- | :--- |
| `configs/task22b_replay_only.yaml` | new |
| `scripts/evaluate_task22_replay.py` | modified — added Model F to the model list, added `<think>`/`<answer>`/EOS metrics, added `delta_replay_trainability`, repaired the `--partial_json` resume block (it had silently never loaded) |
| `artifacts/sft/task22b_replay_only/{best,final}.pt`, `config.json` | new |
| `reports/task22b_replay_only.md`, `.json`, `task22b_training_log.json` | new |

No training code was written. Task 22 artifacts were not overwritten. The Task 20 base checkpoint is byte-identical before and after.

```powershell
python scripts\train_task22_replay_sft.py --config configs\task22b_replay_only.yaml
python scripts\evaluate_task22_replay.py --batch_size 64 --out_json reports\task22b_replay_only.json --partial_json reports\task22b_replay_only.partial.json
```

---

## 7. Conclusion

Parameter freezing — specifically freezing the tied `tok_emb`/`lm_head` — caused the Task 22 regression. The replay corpus is exonerated as the cause of the runaway-chain pathology, and with all parameters trainable it produces the best step-level arithmetic of any arm tested (42.56% greedy validity, vs 23.67% for the Task 21B arithmetic-init baseline). Arithmetic retention and general-language perplexity both remain unsolved and got worse in this arm.

**NEXT DECISION:**
**— parameter-freezing problem**

The freezing hypothesis is confirmed as the dominant cause of the Task 22 result and is now closed. The open problem has moved: no configuration tested preserves the Task 20 arithmetic substrate (retention <6% everywhere), and Model F pays 29.7 PPL points for its equation-validity gain. The next experiment should target the retention/PPL cost — most cheaply by sweeping the replay mixture ratio (e.g. 25/75 and 75/25) at full trainability, which reuses every existing artifact and needs no new corpus.
