# Mini-R1: Task 22 — Operator-Matched Replay SFT with Parameter Isolation

**Date**: 2026-08-23
**Task**: Repair the arithmetic substrate's operator/range coverage, then re-run compact reasoning SFT with the lower half of the network frozen.
**Status**: **COMPLETED**
**Verdict**: **Conclusion D — step-level equation validity regressed by 14.00 points. The success criterion was NOT met.**

---

## 1. Executive summary

Task 21B measured no transfer from the Task 20 arithmetic checkpoint to GSM8K. The established diagnosis was distributional: Task 20 trained **subtraction only** over `[50,99] ∪ [101,199] ∪ [1001,1999]`, while GSM8K's compact traces are dominated by `*` and `+` over small operands, and cross-regime extrapolation measured 0.00%.

Task 22 acted on that diagnosis directly: build a verified arithmetic replay corpus whose operator and magnitude distribution *matches* GSM8K, interleave it 50/50 with compact GSM8K, and freeze the bottom half of the network so the substrate cannot be overwritten.

**The intervention worked mechanically and failed scientifically.**

Everything that was supposed to happen, happened:
- The replay corpus matches the GSM8K trace distribution to within TVD 0.0063 on operators and 0.0212 on operand magnitude.
- Parameter isolation held exactly: max |Δ| on all 62 frozen tensors = **0.000e+00**.
- The model learned the mixture: replay dev loss 5.0001 → 0.4810 (PPL 1.62).
- General-language perplexity was substantially protected: **30.33** vs Task 21B Model A's 43.88.

But the primary metric moved the wrong way:

| Model | Step-level equation validity (primary decode) |
| :--- | ---: |
| **Model E (Task 22)** | **6.45%** (262/4060) |
| Model A (21B: Arithmetic → Compact) | 20.45% (773/3780) |
| Model B (21B: Base → Compact) | 25.90% (761/2938) |

**Δ(E − A) = −14.00 pts. Δ(E − B) = −19.45 pts.** The regression reproduces under greedy decoding (−11.49 / −12.57) and survives a no-padding control (§8), where the gap *widens* to −22.16.

GSM8K strict accuracy was explicitly not the target, and it was flat-to-marginally-positive (+0.30 pts vs Model A). That part of the outcome is the acceptable, reportable result the brief anticipated. The equation-validity regression is not.

---

## 2. Step 1 — The diagnosis, quantified

Measured from `data/gsm8k_compact_reasoning/train.jsonl`: **6,531 traces / 21,377 equations / 24,505 operator occurrences / 45,857 operands**.

| Operator | Count | % | | Operand bucket | Operands | % | Results | % |
| :--- | ---: | ---: | :-- | :--- | ---: | ---: | ---: | ---: |
| `*` | 9,166 | **37.40%** | | `<1` | 307 | 0.67% | 107 | 0.50% |
| `+` | 7,099 | **28.97%** | | `1–9` | 18,733 | **40.85%** | 4,171 | 19.51% |
| `/` | 4,197 | **17.13%** | | `10–99` | 18,161 | **39.60%** | 10,540 | **49.31%** |
| `-` | 4,043 | **16.50%** | | `100–999` | 6,412 | 13.98% | 4,677 | 21.88% |
| | | | | `1000+` | 2,244 | 4.89% | 1,882 | 8.80% |

Structure: 85.14% of equations are single-operator; 11.52% have 2 operators; mean 3.27 equations per trace; **76.75% of non-first steps reuse the previous result** as an operand (53% left / 47% right). Negatives are negligible (0.03%); 96.68% of divisions are exact-integer.

**The coverage gap this exposes.** Task 20's substrate covered `-` only (16.50% of GSM8K's operator mass) with exactly two operands, over ranges that *exclude the entire `1–9` bucket* — 40.85% of all GSM8K operands. `*` and `/` together are 54.53% of GSM8K's operator mass and were never trained at all.

Machine-readable: `reports/task22_compact_distribution.json`.

### Loss masking and decoding audit

- **Masking confirmed.** `scripts/train_task21b_compact_sft.py:99-101` sets `labels[i] = -100` over the first `len(prompt_toks)-1` target positions; `collate_sft_batch` pads labels with `-100`; loss is `CrossEntropyLoss(ignore_index=-100)`. Completion-only supervision is correct.
- **Task 21B was pure greedy.** `evaluate_task21b_transfer.py:132` uses `torch.argmax` with no temperature, top-p or repetition penalty. **Every accuracy in the Task 21B report is greedy-only**; the Task 16B best decoding configuration was never applied to it. This is why Task 22 re-decodes Models A and B from scratch rather than citing 21B's numbers.

---

## 3. Step 2 — The replay corpus

`scripts/build_replay_arithmetic.py` → `data/replay_arithmetic/` (8,000 train / 500 dev, **25,762 equations**).

The generator measures its targets from the GSM8K traces at runtime (no hard-coded constants), builds traces as expression **trees** (a pool of ≤2 open sub-results collapsed to one root, so each trace has one well-defined prompt expression and one verifiable answer), and fits its free sampling knobs — fresh-operator weights, operand-bucket weights, and per-result-bucket acceptance probabilities — by iterative proportional fitting against those targets. Fresh operand *values* are drawn from GSM8K's empirical within-bucket histogram, which reproduces round-number bias and the true frequency of small operands.

**Realized vs target (train split):**

| Dimension | TVD | Notes |
| :--- | ---: | :--- |
| Operator frequency | **0.0063** | `*` 37.25% vs 37.40%, `+` 28.49% vs 28.97% |
| Operand magnitude | **0.0212** | `1–9` 41.83% vs 40.85% |
| Result magnitude | **0.0234** | `10–99` 47.33% vs 49.31% |
| Equations per trace | **0.0159** | mean 3.22 vs 3.27 |
| Ops per equation | 0.0454 | single-op 91.29% vs 86.76% — the loosest fit; the result-magnitude filter rejects multi-op steps slightly more often |

**Verification.** Every equation and every prompt expression is checked by deterministic Python evaluation, twice: once inside the generator and again in an independent second pass. An independent third check written separately for this report re-evaluated all 25,762 train equations, all 1,592 dev equations, every prompt expression against its stated answer, and every `completion` string against its components: **0 failures**.

```
Question: Calculate: 6 * 48 * 3 / 432        Question: Calculate: 10 + 6 * 2 * (2 + 7)
<think>                                      <think>
6 * 48 = 288                                 6 * 2 = 12
288 * 3 = 864                                2 + 7 = 9
864 / 432 = 2                                12 * 9 = 108
</think>                                     10 + 108 = 118
<answer>2</answer>                            </think><answer>118</answer>
```

Note the carried operands render as **numeric values** (`288 * 3`, not `(6 * 48) * 3`), matching GSM8K trace style.

---

## 4. Step 3 — Training and parameter isolation

`configs/task22_replay_sft.yaml` + `scripts/train_task22_replay_sft.py`. The trainer is a thin driver that **imports** `CompactReasoningDataset`, `collate_sft_batch`, `get_cosine_lr` and `evaluate_dev_loss` from the Task 21B trainer — no existing file was modified.

Step budget (650), effective batch (32), seed (42), warmup (50), weight decay (0.01) and grad clip (1.0) are identical to Task 21B.

**Correction to the brief:** the specified LR of 1e-4 is described as "down from prior SFT LR", but Task 21B already ran at `peak_lr = 0.0001` (`artifacts/sft/compact_transfer/arithmetic_init/config.json`). 1e-4 is the *same* LR, not a reduction. That is better for comparability, so it was used as specified.

**Mixture.** Strict 1:1 alternation, not a 50/50 expectation: index 2k draws GSM8K, 2k+1 draws replay, consumed with `shuffle=False`, so every micro-batch of 8 is exactly 4+4. Each source is independently permuted per epoch. Epoch length 13,062.

**Isolation.** Trainable: `blocks.6–11.*` and `ln_f.*` — **18,887,680 params (44.83%)**. Frozen: `embeddings.tok_emb`, `embeddings.pos_emb`, `blocks.0–5.*` — **23,244,800 params (55.17%)**. AdamW receives only the trainable tensors.

> **Material constraint discovered during implementation.** This checkpoint has `tie_word_embeddings: True`, and `lm_head.weight` was verified to be the *same tensor* as `embeddings.tok_emb.weight`. Freezing `tok_emb` as specified therefore **also freezes the output projection**. `ln_f` is the only remaining degree of freedom before the frozen head — the model cannot re-tune how hidden states map onto digit tokens. This is implemented as specified, but it is a real and probably significant limit on what this configuration could achieve (see §9).

**Post-training audit (printed, not assumed):**

| Check | Result |
| :--- | :--- |
| Frozen tensors checked | 62 |
| Max \|Δ\| on any frozen parameter | **0.000e+00** |
| Isolation held | **True** |
| Task 20 base checkpoint SHA256 unchanged | **True** (`fdcecade71afafc…`) |

**Learning curve:** mixture dev loss 4.8522 → **0.5841**; GSM8K component 4.7043 → 0.6871 (PPL 1.99); replay component 5.0001 → 0.4810 (PPL 1.62). Best at step 650. 982,943 active completion tokens.

---

## 5. Step 4 — Results

### 5.1 Primary metric: step-level equation validity

Computed over the `<think>` blocks of GSM8K test generations (N=1,319). A line counts as an equation if it has the form `<arithmetic expression> = <number>`; it verifies true when deterministic Python evaluation of the LHS equals the stated RHS within 1e-4. Restatements without an operator (`5 = 5`) are not counted as equations.

| Model | Decode | Valid | Total | **Validity** | GSM8K acc | eqs/gen | **valid eqs/gen** | final-step consistency |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Model E (Task 22)** | sampled T=0.6/p=.95/rp=1.2 | 262 | 4060 | **6.45%** | 2.05% | 3.08 | **0.199** | 90.9% |
| **Model E (Task 22)** | greedy | 823 | 6754 | **12.19%** | 1.52% | 5.12 | **0.624** | 94.8% |
| Model A (21B) | sampled T=0.6/p=.95/rp=1.2 | 773 | 3780 | 20.45% | 1.74% | 2.87 | 0.586 | 97.6% |
| Model A (21B) | greedy | 1340 | 5660 | 23.67% | 0.91% | 4.29 | 1.016 | 99.3% |
| Model B (21B) | sampled T=0.6/p=.95/rp=1.2 | 761 | 2938 | 25.90% | 1.82% | 2.23 | 0.577 | 97.5% |
| Model B (21B) | greedy | 1479 | 5974 | 24.76% | 1.21% | 4.53 | 1.121 | 98.3% |

`valid eqs/gen` is reported because the headline metric is a **rate**, and a rate is denominator-sensitive: a model that emits almost no equations can post a high validity score on the handful it does emit. Model B on the replay holdout is exactly that case (0.11 eqs/gen at 36.84% validity). On GSM8K the absolute measure tells the same story as the rate — Model E produces **3× fewer verified-correct equations per generation** than Model A under primary decoding — so the regression is not an artifact of the denominator.

### 5.2 Deltas

| Comparison | Equation validity | GSM8K strict acc |
| :--- | ---: | ---: |
| E − A (primary decode) | **−14.00 pts** | +0.30 pts |
| E − B (primary decode) | **−19.45 pts** | +0.23 pts |
| E − A (greedy) | **−11.49 pts** | +0.61 pts |
| E − B (greedy) | **−12.57 pts** | +0.30 pts |

### 5.3 Retention, replay-regime holdout, and general language

| Model | 2D sub | 3D sub | 4D sub | Replay-regime holdout | General PPL |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Model E (Task 22)** | **5.60%** (28/500) | 4.20% (21/500) | 0.00% (0/500) | 3.20% (16/500) | **30.33** |
| Model A (21B) | 4.40% (22/500) | **5.20%** (26/500) | **0.60%** (3/500) | 3.00% (15/500) | 43.88 |
| Model B (21B) | 0.60% (3/500) | 0.00% | 0.00% | 0.80% (4/500) | **21.85** |
| *Task 20 pre-transfer baseline* | *100.00%* | *99.80%* | *67.20%* | — | *27.63* |

Two things stand out.

**Parameter isolation did not preserve arithmetic.** Retention against the Task 20 baseline collapsed just as completely as in Task 21B — 100% → 5.60%, 99.8% → 4.20%, 67.2% → 0.00%. Freezing 55% of the network, including all of blocks 0–5 and both embedding matrices, bought essentially nothing on the retention probes (+1.2 pts on 2D, −1.0 on 3D, −0.6 on 4D vs Model A). The arithmetic capability does not live in the frozen parameters.

**Parameter isolation did protect general language.** PPL 30.33 vs Model A's 43.88 — a 13.55-point recovery, landing near the Task 20 pre-transfer level of 27.63. This is the one unambiguous win of the isolation, and it is consistent with the frozen embeddings and lower blocks carrying the language model.

**Model E does not beat Model A on its own training distribution.** On the held-out replay-regime arithmetic — the exact distribution Model E trained 8,000 examples on — it scores 3.20% vs Model A's 3.00%. Statistically indistinguishable (16 vs 15 of 500). Teacher-forced replay dev PPL of 1.62 did not convert into free-running arithmetic competence.

### 5.4 Generation health and step-complexity

| Model | Decode | Full format % | Repetition % | Mean len | p90 len |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Model E** | sampled | 85.4% | 47.5% | 39.6 | 56 |
| **Model E** | greedy | **72.8%** | **99.9%** | **126.9** | **256** |
| Model A | sampled | 99.6% | 37.1% | 36.6 | 49 |
| Model A | greedy | 97.8% | 96.4% | 62.9 | 82 |
| Model B | greedy | 91.6% | 95.4% | 79.7 | 123 |

Model E is the *least* format-compliant and *most* degenerate of the three under greedy decoding. Its p90 generation length is 256 — the cap — meaning at least 10% of generations never terminate at all. (Lengths and repetition here are measured up to and including `</answer>`; see §7.)

| Step complexity | Model E | Model A | Model B |
| :--- | ---: | ---: | ---: |
| 2-step (N=315) | 2.22% (7) | 1.27% (4) | 0.63% (2) |
| 3-step (N=360) | 1.11% (4) | 1.39% (5) | 2.78% (10) |
| 4+-step (N=644) | 2.48% (16) | 2.17% (14) | 1.86% (12) |

---

## 6. Harness validation

The Task 22 evaluator reuses the Task 21B harness for answer extraction, repetition detection, benchmark loading and PPL, and preserves its generation contract verbatim (BOS-prefixed prompt, left padding, no attention mask passed to the model). Under greedy decoding it reproduces the Task 21B published numbers **exactly**:

| Quantity | Task 21B report | Task 22 recomputation | Match |
| :--- | ---: | ---: | :---: |
| Model A GSM8K strict acc | 0.91% | 0.91% | ✅ |
| Model A 2D / 3D / 4D retention | 4.4 / 5.2 / 0.6% | 4.4 / 5.2 / 0.6% | ✅ |
| Model A mean generation length | 62.9 | 62.9 | ✅ |
| Model A general PPL | 43.88 | 43.88 | ✅ |
| Model B GSM8K strict acc | 1.21% | 1.21% | ✅ |
| Model B 2D / 3D / 4D retention | 0.6 / 0.0 / 0.0% | 0.6 / 0.0 / 0.0% | ✅ |
| Model B general PPL | 21.85 | 21.85 | ✅ |

Seven independent exact reproductions. The comparison in §5 is genuinely apples-to-apples.

---

## 7. Two harness properties worth recording

**(a) The Task 21B generator ignores its own attention mask.** `evaluate_task21b_transfer.py` builds `attn_mask` at lines 116–121 and never passes it to the model. Combined with left padding and *learned absolute* position embeddings (`embeddings.pos_emb.weight`, 512×512), the same question begins at a different position index depending on which prompts share its batch — while training right-pads, so real content always started at position 0. This is baked into every Task 21B number. It was preserved unchanged for comparability and measured rather than assumed (§8).

**(b) Early stopping on `</answer>`.** `</answer>` is a single dedicated token (id 7). Generation now halts there as well as on EOS. Equation validity, strict accuracy and format compliance are unaffected — all three read only the first `<think>…</think>` and first `<answer>…</answer>` — and this was verified empirically: Model E's sampled pass scored 262/4069 = 6.44% before the change and 262/4060 = 6.45% after, i.e. 9 stray equations had appeared after the answer tag. Reported **length and repetition** figures are therefore measured up to `</answer>` and are *not* comparable to Task 21B's, which included post-answer continuation.

---

## 8. Control: is the padding artifact responsible?

Batch size 1 has zero padding and always starts real content at position 0 — the condition the models were trained under. Comparing bs=1 against bs=64 on identical prompts isolates the artifact. Greedy, N=200.

| Model | Suite | Validity @ bs=64 | Validity @ bs=1 | Δ |
| :--- | :--- | ---: | ---: | ---: |
| **Model E** | GSM8K | 12.42% | **15.42%** | +3.00 |
| **Model A** | GSM8K | 23.80% | **37.58%** | **+13.78** |
| Model E | replay holdout | 8.38% | 10.40% | +2.02 |
| Model A | replay holdout | 26.72% | 15.35% | −11.37 |

The padding artifact is real and materially depresses measured arithmetic. But it does **not** explain away the result — it was *flattering* Model E. Removing padding helps Model A more than four times as much, and under the clean condition the GSM8K gap **widens from −11.49 to −22.16 points** (15.42% vs 37.58%). N=200 makes these estimates noisy, but the direction is unambiguous and it is the opposite of a rescue.

Raw: `reports/task22_padding_control.json`.

---

## 9. Mechanistic analysis: why it regressed

The failure is legible in the generations themselves. Model E, greedy, `gsm8k_test_0`:

```
<think>
16 / 2 = 3
3 * 2 = 6
3 * 3 = 12
16 + 3 + 12 + 6 = 12
16 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + 12 + ...   (to the 256-token cap)
```

Three mechanisms, in order of confidence:

**1. The replay corpus taught runaway additive chains.** GSM8K traces are 85.14% single-operator. My corpus reproduced that marginal (91.29%) but the *prompt expressions* are nested multi-operator strings like `((50 + 2) * 5 - 5) * 3`, and 8.71% of steps carry 2–3 operators. Model E emits **5.12 equations per generation** against GSM8K's true mean of 3.27 and Model A's 4.29, and its p90 length hits the 256-token cap. Training on multi-operand chained expressions appears to have installed a "keep appending `+ n`" attractor that dominates under greedy decoding. This is a defect in my corpus *design* — I matched the equation-level statistics the brief specified while introducing a prompt format (deeply nested arithmetic expressions) that has no analogue in GSM8K.

**2. The tied output head could not adapt.** Because `lm_head` is `tok_emb` and both were frozen, the only path from the trainable blocks to digit selection is through `ln_f` into a fixed projection. Arithmetic output demands precise digit-token discrimination, and this configuration forbids retuning it. Model A, which trained all 42.1M parameters, kept a 14-point advantage. This is a plausible major contributor but is **not isolated by this experiment** — Task 22 changed the data mixture and the trainable parameter set simultaneously, so their contributions are confounded.

**3. Half the gradient signal went to a prompt format GSM8K never uses.** 50% of every batch was `Question: Calculate: <expr>` rather than a word problem. With only the top six blocks free to adapt, that may have been too large a share of limited capacity.

**What the result does establish:** the Task 21B null was *not* caused solely by operator/range coverage. Closing the coverage gap — verifiably, to TVD 0.0063 — while protecting the substrate by freezing 55% of the network made step-level arithmetic worse, not better. Distributional coverage of the arithmetic substrate is not the binding constraint.

---

## 10. Honest scope and limitations

- **Confounded manipulation.** Replay mixture and parameter isolation were changed together, as specified. Their individual effects cannot be separated from this run. A replay-only arm (all parameters trainable) is the missing cell and is the single most informative next experiment.
- **One seed.** Seed 42 only, matching Task 21B. No variance estimate.
- **The padding control is N=200**, and its replay-holdout row for Model A moves the opposite way to its GSM8K row — a sign of genuine noise at that sample size. Only the GSM8K direction (large, consistent across both models) should be leaned on.
- **Retention probes are greedy-only**, so they line up with the Task 20 baseline and the Task 21B report.
- **Absolute accuracies are all in the 1–2% band**, where differences of a few tenths of a point are not meaningful. The equation-validity gaps (11–22 points, over thousands of equations) are the only differences here large enough to interpret.
- SVAMP and AQuA were not run; they were not requested for this task.

---

## 11. Artifacts and reproduction

```powershell
python scripts\build_replay_arithmetic.py
python scripts\train_task22_replay_sft.py --config configs\task22_replay_sft.yaml
python scripts\evaluate_task22_replay.py --batch_size 64
```

| Artifact | Path | SHA256 |
| :--- | :--- | :--- |
| Replay train corpus | `data/replay_arithmetic/train.jsonl` | `4bd131c4e31ce89aa9423b8a879fd1afe463a50267aed386e4b7676c5aba7d14` |
| Replay dev corpus | `data/replay_arithmetic/dev.jsonl` | `75dba1f5272b109263a6f6bb39967f9e3b0990027d9e2027cfa399f0b5f312b8` |
| Model E checkpoint | `artifacts/sft/task22_replay_isolated/best.pt` | `f339f5432de85fd28757f9ca5395704636e694a6b6b821537eb8d66ded96a300` |
| Task 20 init (unmodified) | `artifacts/sft/progressive_subtraction/best.pt` | `fdcecade71afafc1857d5ee56d221d31928ae1321d0f963dc0f566482af56d5c` |

Reports: `task22_replay_sft.json` (full results), `task22_training_log.json` (curve + isolation audit), `task22_compact_distribution.json` (Step 1), `task22_padding_control.json` (§8).

---

## 12. Verdict

$$\textbf{Conclusion D: step-level equation validity regressed } (-14.00 \text{ pts vs Task 21B Model A})$$

The success criterion — *step-level equation validity rises materially above the Task 21B baseline* — was **not met**. GSM8K strict accuracy was flat (+0.30 pts), which the brief anticipated as acceptable; the equation-validity regression is the finding.

The operator-coverage hypothesis is disconfirmed as a sufficient explanation for Task 21B's null result. Two by-products are worth carrying forward: parameter isolation recovered 13.55 points of general-language perplexity (43.88 → 30.33), and it demonstrably does *not* protect arithmetic retention, which collapsed identically to Task 21B. Whatever encodes arithmetic competence in this model does not reside in the embeddings or the lower six blocks.

**TASK 22 STATUS: COMPLETED — hypothesis disconfirmed.**
