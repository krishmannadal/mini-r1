# Mini-R1 — Complete Research, Engineering, Mathematics & Interview Guide

**Document purpose**: primary study material. Written so you can understand, explain, defend and reproduce the entire Mini-R1 project without reference to any earlier conversation, and without AI assistance in the room.

**Traceability rule**: every quantitative claim below carries a source in `[brackets]` — a report, JSON artifact, log, source file, or checkpoint field. If a number is not traceable, it is not stated.

**Date**: 2026-08-23 · **Project status**: research complete through Task 24.

---

## READ THIS FIRST — corrections to historical reports

Three places where earlier reports overclaimed or where numbers conflict. You must know these, because an interviewer who reads the reports carefully will find them, and *you* finding them first is the strongest possible signal.

### Correction 1 — Task 21A's "negligible catastrophic forgetting" is wrong

`reports/task21a_arithmetic_to_reasoning_transfer.md` states: *"General-language perplexity remains stable at 28.87 (compared to base 17.68), confirming negligible catastrophic forgetting."*

This is wrong twice over:
- 28.87 vs 17.68 is a **+63.3% perplexity increase**. That is not "stable."
- More seriously, the same model's **arithmetic** retention fell from 100% / 99.8% / 67.2% (2D/3D/4D) to **3.6% / 0.6% / 0.4%** [`task21b_compact_reasoning_transfer.json`]. That is catastrophic forgetting by any definition, and the sentence dismisses it by only looking at language.

**Correct statement**: Task 21A exhibited severe catastrophic forgetting of arithmetic and a substantial (+63%) language perplexity regression. The word "negligible" should never have been used.

### Correction 2 — Task 19B's headline 24.28% was a leakage artifact

The initial 3-digit division result of **24.28%** (76/313) was later shown to be **93.4% explained by training-set overlap**. On the 202 genuinely unseen pairs, accuracy was **2.48%** [`task19b_pre2_overlap_audit.md`]. Always quote 2.48%, and always mention the audit.

### Correction 3 — Task 22's blame on the replay corpus was retracted by Task 22B

`reports/task22_replay_sft.md` §9 attributed runaway equation chains to corpus design. Task 22B ran the same corpus with all parameters trainable and the pathology vanished (5.12 → 2.74 equations/generation, p90 length 256 → 59) [`task22b_replay_only.json`]. **The corpus was exonerated; parameter freezing was the cause.** The Task 22 report retains its original text with the retraction noted — do not cite its §9 hypothesis as a finding.

### Non-contradiction worth explaining: the "step 18,000" checkpoint

`best.pt` has `global_step: 18000`, `tokens_seen: 294,912,000`, `val_loss: 2.8726140` [checkpoint metadata]. The 500M-token final row also reports val loss **2.8726 / PPL 17.68** [`task13_base_pretraining_results.md`]. These are not in conflict: validation loss **plateaued** at ~295M tokens and the final 500M checkpoint did not strictly improve on it, so the best-tracker retained the earlier one. The honest reading is that **the last ~205M tokens (41% of the budget) bought no measurable validation improvement.**

Also note the final row shows train loss 3.0807 > val loss 2.8726. That inversion is expected here: train loss is a noisy single-step value at the end of cosine decay, val loss is a full-pass average over 10M held-out tokens. It is not evidence of anything anomalous.

### One system fact commonly misstated

**FastAPI was never implemented.** It is listed as a deployment *goal* in `PROJECT_CONTEXT.md` §1.2.5. The actual server, `web/server.py`, uses Python's stdlib `http.server` + `socketserver`. Say "a lightweight stdlib HTTP server; FastAPI was the planned target."

---

# PART 1 — THE PROJECT IN ONE PAGE

**What we built.** Mini-R1: a 42,132,480-parameter decoder-only Transformer written from scratch in pure PyTorch (no HuggingFace model classes, no prebuilt attention), pretrained on 500M tokens on one 6 GB RTX 4050 laptop GPU, then fine-tuned through a long series of controlled experiments on mathematical reasoning.

**Why.** Chain-of-thought reasoning is normally studied at ≥1B parameters, where arithmetic ability is already latent in the pretrained model. At 42M it is not — so arithmetic must be explicitly taught, which makes the research question *cleanly separable*: teach arithmetic in isolation, measure it exactly, then fine-tune on word problems and measure precisely what survives. And because arithmetic is deterministic, every claim can be checked by Python rather than by another model.

**Original research question.** Can verified arithmetic supervision give a 42M model arithmetic competence that **transfers** to natural-language mathematical reasoning (GSM8K)?

**How the question evolved.**
1. → *Does CoT supervision help at all?* (No — it actively hurt: 2.73% Direct vs 1.21% CoT.)
2. → *Is that a decoding artifact?* (No — fixing decoding fixed the text, not the accuracy.)
3. → *Can the model even do the arithmetic?* (Not initially — it produced reasoning-shaped text with wrong numbers.)
4. → *Does verified arithmetic SFT fix it?* (Yes, but only inside trained numeric regimes.)
5. → *Does that arithmetic transfer to GSM8K?* (No: transfer delta −0.30%.)
6. → *Why not — is the substrate the wrong distribution?* (Partly, but fixing coverage did not fix transfer.)
7. → *Can we at least stop GSM8K SFT destroying the arithmetic?* (**Yes** — targeted replay. This became the project's actual contribution.)

**Final model.** `artifacts/sft/task23_75_25/best.pt` (SHA256 `d630daac…e49c`) — "Model H": Task 20 arithmetic checkpoint fine-tuned for 650 steps on a 75% GSM8K / 25% verified-arithmetic-replay mixture, all parameters trainable.

**Most important finding.** **Catastrophic forgetting of a learned computational skill is preventable by replaying the *target distribution*, and composition matters at least as much as dose.** 25% replay containing the probed arithmetic regime lifted 2-digit retention from **4.40% → 94.80%**; 50% replay of a *different* arithmetic distribution preserved nothing (**3.80%**) [`task23_replay_ratio.json`, `task22b_replay_only.json`].

**What failed.** Transfer. Arithmetic competence never converted into word-problem reasoning — GSM8K stayed in a 1–2.4% band across every configuration in the entire project. Also: CoT supervision, decoding-only fixes, and parameter freezing (which backfired badly).

**What succeeded.** Verified arithmetic SFT (100% IID 2-digit), the leakage forensics that saved the project from a false result, isolating parameter freezing as a failure mechanism (+30.38 pts when removed), and targeted replay for retention.

**Final limitation.** This is not a reasoning model. GSM8K 2.43% is statistically indistinguishable from plain direct-answer SFT (2.35%) — a one-problem difference on 1,319 items. Arithmetic is regime-local (89.60% on trained 2-digit ranges, 0.60% at 5 digits). General perplexity regressed 2.9× from base (17.68 → 50.74).

## The 2-minute verbal version

> "I built a 42-million-parameter transformer from scratch in PyTorch — the attention, the training loop, the evaluation, all by hand — and pretrained it on 500 million tokens on a 6 GB laptop GPU. The research question was whether teaching a small model verified arithmetic would transfer to word problems like GSM8K.
>
> The first surprise was that chain-of-thought supervision made things *worse* — 1.21% versus 2.73% for direct answers. I ran a decoding ablation to check whether that was just degenerate sampling; it cut repetition from 83% to 11% and did essentially nothing for accuracy. So the problem was representational, not an inference artifact.
>
> I then built deterministic arithmetic datasets where every label is verified by Python. That worked: 100% on held-out 2-digit subtraction. But when I audited it I found training-set leakage inflating an earlier result from an apparent 24% down to a real 2.5%, so I rebuilt the splits — and I now quote only the audited numbers.
>
> Then the core negative result: that arithmetic did not transfer. A controlled A/B gave a transfer delta of minus 0.3%, and worse, GSM8K fine-tuning destroyed the arithmetic — retention fell from 100% to about 4%.
>
> I tried to protect it by freezing the lower half of the network. That failed badly, and the reason was interesting: the model has tied input/output embeddings, so freezing the token embedding also froze the output head. The model couldn't learn to emit its stop token and degenerated into infinite loops. The control run with everything trainable recovered 30 points of equation validity.
>
> What finally worked was replay: interleaving 25% verified arithmetic — specifically arithmetic from the *same distribution* I was measuring — restored retention from 4.4% to 94.8%. The lesson is that replay composition matters more than replay volume.
>
> Honestly: it's not a reasoning model. GSM8K sits at 2.4%, no better than a trivial baseline. What it is, is a clean study of catastrophic forgetting in a small model, with every arithmetic claim verified by Python instead of by vibes."

## The 30-second elevator pitch

> "I built and pretrained a 42M-parameter transformer from scratch on a single laptop GPU to study whether verified arithmetic training transfers to mathematical word problems. It doesn't — transfer was measurably zero. But I found that fine-tuning destroys the arithmetic the model already learned, and that you can prevent that by replaying the *right* arithmetic distribution: retention went from 4% back to 95%. It's a negative result on transfer and a positive, reproducible one on catastrophic forgetting, with every number verified deterministically."

---

# PART 2 — COMPLETE PROJECT TIMELINE

The reasoning chain matters more than the list. Read the "Why next" column as a continuous argument.

### Phase I — Infrastructure (Tasks 1–12)

| Task | Objective | Why needed | Implemented | Measured | Result | Problem found | Why next task |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Environment validation | Must confirm 6 GB can train 42M | `scripts/gpu_check.py` | FP16/BF16 autocast | PyTorch 2.6.0+cu124 on RTX 4050 OK | — | Need a tokenizer |
| 2 | Tokenizer | Vocabulary determines numeric representation | SentencePiece BPE 8k, `<think>`/`<answer>` atomic | Vocab coverage | 8,000 vocab, MD5 `e7ef6360…` | — | Need the model |
| 4 | Architecture from scratch | Core research rule: no framework wrappers | `src/model.py`, `attention.py`, `transformer_block.py` | Tensor shapes | 42,132,480 params, pre-LN | — | Need a loss |
| 5 | Loss & optimizer | Causal alignment is easy to get wrong | `src/loss.py`, `optimizer.py` | Backprop check | Causal shift validated | — | Need memory safety |
| 8 | AMP qualification | FP16 can silently diverge | GradScaler pipeline | FP32 vs FP16 error | 21.5k tok/s, 0 skipped steps, FP16 err 0.1085% | — | Need an LR |
| 8C/9 | LR sweep | Wrong LR wastes the whole budget | 1,000-step sweep over 1e-4/3e-4/6e-4 | Val loss | **6e-4: 2.4995 (PPL 12.18)** vs 3e-4: 2.6337 (PPL 13.93) | 1e-4 too slow | Need efficient data |
| 10 | Contiguous packing | Naive truncation wastes tokens | EOS-packed streaming | Token density | **100% density** (vs 58.4% waste) | — | Need a corpus |
| 11/11B | Corpus qualification | Garbage in, garbage out | 60% cosmopedia-v2 / 40% fineweb-edu-dedup | 500M train / 10M val | Qualified; `python-edu` excluded (git schema) | — | Pretrain |

### Phase II — Pretraining and first failure (Tasks 13–16)

| Task | Objective | Measured | Result | Problem found | Why next |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 13 | 500M pretraining | Val loss / PPL | **8.02 h, 17,316.5 tok/s avg, peak VRAM 3,409.7 MB**; best.pt at **step 18,000 / 294.9M tokens, val 2.8726, PPL 17.68** | Val plateaued at ~295M — last 41% of budget bought nothing | Establish baseline |
| 14 | Base zero-shot | GSM8K | **0.76%** greedy (10/1319), 0.0% format, 125.7 tok mean | No reasoning to elicit | Must fine-tune |
| 15A/B | SFT datasets + engine | Paired Direct vs CoT | 6,973 train / 500 dev; prompt masked to −100 | CoT has **19.05×** completion tokens | Run the comparison |
| **15C** | **Direct vs CoT** | GSM8K strict | **Direct 2.73% vs CoT 1.21%** (p=0.0051); CoT repetition **82.6%** | **CoT actively hurt** | Is it decoding? |
| 16A | Eval audit | PPL on 10M | Base 17.68 → Direct 21.83 (+23.5%) → CoT 23.58 (+33.4%) | Report placeholders found | Fix, then test decoding |
| **16B** | **Decoding ablation** | 5 configs on frozen CoT | Repetition **82.6% → 11.45%**, format **39.8% → 90.67%**, accuracy **2.20%** — still below Direct 2.73% | Text fixed, reasoning not | Probe arithmetic directly |

### Phase III — Arithmetic (Tasks 17–20)

| Task | Objective | Measured | Result | Problem found | Why next |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 17A | Arithmetic probe | **9,000** synthetic problems | Model emits reasoning-shaped text with wrong numbers | Arithmetic execution is the bottleneck | Teach arithmetic |
| 18A/B | Verified arithmetic SFT | IID / OOD | Compact **45.42%** IID, verbose **49.42%**; **OOD numbers 1.00%** | Huge IID→OOD gap | Is IID real or memorised? |
| **19A/B** | **Leakage forensics** | Overlap audit | Apparent **24.28%** → **111/313 (35.46%) overlap**, 71/76 correct from overlap, **2.48% on 202 unseen** | **93.4% of the result was memorisation** | Rebuild clean splits |
| 19C | Clean 2D-only | Held-out | Unseen 2D **35.40%**, OOD 3D **6.60%**, 3-digit answers **0.00%** | Regime-local | Add 3D exposure |
| 19D | Mixed 2D+3D | Held-out | OOD 3D **6.60% → 34.40%** (+27.80); unseen 2D 28.40%; untrained 4D **0.40%** | Exposure helps *that* regime only | Add 4D |
| **20** | **Progressive 2D+3D+4D** | E1–E7 | E1 **100.0**, E2 18.2, E3 **99.8**, E4 26.2, E5 **67.2**, E6 8.6, **E7 (5D) 0.00** | Extrapolation is zero | Try transfer to GSM8K |

### Phase IV — Transfer, forensics, and the fix (Tasks 21–24)

| Task | Objective | Measured | Result | Problem found | Why next |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 21A | Arithmetic → verbose GSM8K | A vs B | A 1.44% vs B 1.06%, **Δ +0.38%**; PPL 28.87; retention **3.6/0.6/0.4** | Report wrongly called forgetting "negligible" | Try compact targets |
| **21B** | **Arithmetic → compact GSM8K** | A vs B | Format **97.8%**, tokens −85.3%, but **Δ −0.30%**; retention 4.4/5.2/0.6; PPL 43.88 | **No transfer.** Substrate ≠ GSM8K distribution | Match the distribution |
| 22 | Replay + freezing | Equation validity | Replay corpus matched to **TVD 0.0063**; but validity **6.45%** vs A's 20.45% (**−14.00**) | Freezing? or corpus? — confounded | Run the control |
| **22B** | **Replay, all trainable** | Same data unfrozen | Validity **31.77% / 42.56%**, **Δ +25.32 / +30.38**; p90 length 256 → 59 | **Freezing was the cause**; retention still 3.8/1.8/0.0 | Fix retention |
| **23** | **Replay ratio 10% / 25%** | Retention | 10%: **65.6/54.8/13.4**; 25%: **94.8/78.2/32.4**; PPL rises monotonically 43.88→47.90→50.74→60.02 | No arm passed PPL ≤ 35 gate | Qualify and freeze |
| 24 | Final qualification | Integrity + benchmarks | 8/8 integrity; GSM8K 2.43%, E1 89.60, E7 0.60, PPL 50.74 | — | **STOP** |

**The reasoning chain in one line**: CoT fails → not decoding → arithmetic is missing → teach it (verified) → oh, it's leaked → rebuild → it's regime-local → does it transfer? no → is coverage the reason? no → does freezing protect it? no, it breaks generation → does replay protect it? **yes, if composition matches**.

---

# PART 3 — SYSTEM ARCHITECTURE

## The pipeline

```
raw corpus (cosmopedia-v2 60% / fineweb-edu-dedup 40%)
   │  streaming download, no full-dataset materialisation
   ▼
SentencePiece BPE tokenizer (8,000 vocab; digits are individual tokens)
   │  EOS-separated contiguous packing → 100% token density
   ▼
uint16 shards on disk (vocab 8,000 < 65,536 ⇒ 2 bytes/token)
   │  np.memmap — OS page cache serves batches, no RAM blow-up
   ▼
token ids  (B, T)                                    B=8, T=512
   ▼
token embedding (8000×512)  +  learned absolute position embedding (512×512)
   ▼
12 × TransformerBlock  [pre-LN: x + Attn(LN(x));  x + FFN(LN(x))]
   │      MHA: 8 heads × 64 dim, causal mask
   │      FFN: 512 → 2048 → 512
   ▼
final LayerNorm (ln_f)
   ▼
LM head (512→8000)  ← TIED to token embedding (same tensor)
   ▼
logits (B, T, 8000)
   ▼
cross-entropy vs shifted targets (prompt positions = −100)   [training]
   OR
greedy / temperature+top-p+repetition-penalty sampling        [inference]
   ▼
generated text:  <think>\n{equations}\n</think>\n<answer>{n}</answer>
   ▼
deterministic Python verifier — every equation re-evaluated
```

## Component decisions: what, why, what we rejected

| Component | What it does | Why chosen | Alternatives rejected, and why |
| :--- | :--- | :--- | :--- |
| **Pre-LN blocks** | LayerNorm *before* sublayer | Stable gradients at depth without warmup gymnastics | **Post-LN** (original 2017): needs careful warmup, more divergence risk at 12 layers on FP16 |
| **Learned absolute positions** | 512×512 position table | Simplest correct thing; matches GPT-2 | **RoPE/ALiBi**: better length extrapolation, but we never exceed 512 so the benefit is unrealised — *and* this choice later bit us (see the padding artifact, Part 20) |
| **Tied embeddings** | `lm_head.weight is tok_emb.weight` | Saves 4.096M params (9.7% of the model) at 42M scale | **Untied**: +4.1M params for marginal gain. **This choice caused the Task 22 failure** — freezing `tok_emb` also froze the head |
| **No linear bias** | `bias=False` throughout | LLaMA/Mistral-style; biases add params, not quality | Biased linears: strictly more params, no measured benefit |
| **8k SentencePiece BPE** | Subword vocabulary | Small vocab keeps the embedding affordable; special tags atomic | **32k+**: embedding alone would be 16.4M params (39% of model). **Char-level**: sequences too long for a 512 context |
| **FP16 AMP + GradScaler** | Half-precision forward/backward | ~2× throughput, ~½ activation memory — mandatory on 6 GB | **BF16**: better dynamic range, no scaler needed, but *unreliable on consumer Ada laptop SKUs in this stack*; FP16+scaler was verified (0 skipped steps). **FP32**: would not fit the batch |
| **Gradient accumulation** | 4 micro-steps per update | Effective batch 32 without 32-sample activations | Larger true batch: OOM at 6 GB |
| **uint16 memmap shards** | On-disk token store | 2 bytes/token, OS-cached, zero-copy slicing | JSON/text: 10× larger, parse cost per epoch. In-RAM: 500M tokens × 2 B = 1 GB, feasible but wasteful and not resumable |
| **Completion-only loss** | Prompt labels set to −100 | We supervise the *answer*, not the question | Full-sequence loss: model spends capacity learning to regenerate GSM8K prompts |
| **Deterministic verifier** | Python `eval` on each equation | Ground truth without a judge model | LLM-as-judge: unverifiable, circular at this scale |
| **stdlib HTTP server** | `web/server.py` demo | Zero dependencies, adequate for a local demo | **FastAPI** was the stated goal (`PROJECT_CONTEXT` §1.2.5) but **was never implemented** |

## Training pipeline

Data cursor → micro-batch (8×512) → autocast FP16 forward → CE loss / 4 → `scaler.scale(loss).backward()` → repeat ×4 → `scaler.unscale_` → clip to 1.0 → `scaler.step` → `scaler.update` → `zero_grad`. Cosine LR updated per optimizer step; validation every N steps; atomic best/latest checkpoint writes.

## Checkpoint system

`best.pt` (lowest val loss), `latest.pt` (resume), `step_N.pt` (periodic). Written to `*.tmp` then `os.replace()` — atomic on the same filesystem, so a crash mid-write cannot corrupt a good checkpoint. Contents include model, optimizer, RNG states (`torch`, `cuda`, `numpy`, `python`), `dataset_state`, `tokens_seen`, `global_step` — verified present in `best.pt`.

## Evaluation pipeline

Batched generation (left-padded) → strict `<answer>…</answer>` regex extraction → numeric comparison at 1e-4 → format/EOS/repetition/length metrics → **step-level equation validity** (every `<think>` line re-evaluated in Python) → aggregate JSON.

---

# PART 4 — MODEL ARCHITECTURE FROM FIRST PRINCIPLES

Config: `d_model=512, n_layers=12, n_heads=8, d_ff=2048, vocab=8000, context=512, bias=False, tie_word_embeddings=True`.

Throughout: **B** = batch, **T** = sequence length (≤512), **d** = d_model = 512, **h** = heads = 8, **d_k** = d/h = 64, **V** = vocab = 8000.

## 4.1 Token embeddings

$$X_{\text{tok}} = E[\,\mathbf{t}\,], \qquad E \in \mathbb{R}^{V \times d}$$

- $\mathbf{t}$: integer token ids, shape (B, T). $E$: lookup table, 8000×512. Output: (B, T, 512).
- **Why**: a discrete symbol has no geometry; the embedding assigns each token a learned vector so that "similar" tokens can be near each other.
- **Remove it**: you would have to feed one-hot vectors of width 8000 — mathematically the same as an embedding but computationally absurd (the first matmul *is* the lookup).

## 4.2 Positional information

$$X = X_{\text{tok}} + P[0{:}T], \qquad P \in \mathbb{R}^{512 \times d}$$

- **Why it must exist**: attention is a **permutation-equivariant** operation. $\text{Attn}(\Pi X) = \Pi\,\text{Attn}(X)$ for any permutation $\Pi$. Without positional information "87 − 43" and "43 − 87" are literally the same set of vectors, and subtraction is not commutative.
- **Remove it**: the model cannot represent order. Arithmetic, syntax, and causality all collapse.
- **Mini-R1 consequence**: because $P$ is a *learned absolute* table indexed from 0, the representation of a token depends on its **absolute index**. Left-padding a batch shifts every real token's index and therefore changes its representation. This is the root of the harness artifact in Part 20.

## 4.3 Q, K, V

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V, \qquad W_\bullet \in \mathbb{R}^{d\times d}$$

Each is (B, T, 512), then reshaped to (B, h, T, 64).

- **Intuition**: for each position, $Q$ is "what am I looking for", $K$ is "what do I offer", $V$ is "what I will hand over if selected".
- **Why three separate projections**: they decouple *matching* from *content*. If $Q=K$ the attention pattern is forced symmetric; if $V=K$ the retrieved content is forced identical to the matching key.
- **Params**: 4 × 512 × 512 per layer (Q, K, V, output) = 1,048,576.

## 4.4 Scaled dot-product attention

$$\boxed{\;\text{Attn}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V\;}$$

Dimensions: $QK^\top$ is (B, h, T, T); softmax is over the **last** axis (keys); result (B, h, T, 64); heads concatenated → (B, T, 512) → output projection.

### Why $\sqrt{d_k}$ — the derivation interviewers want

Assume the components of $q$ and $k$ are independent, zero-mean, unit-variance. Then

$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i, \qquad \mathbb{E}[q\cdot k] = 0, \qquad \operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i) = d_k$$

So the dot product has standard deviation $\sqrt{d_k}$ — it **grows with dimension**. With $d_k=64$, typical logits are ±8 before scaling.

Softmax saturates for large logit gaps. Its Jacobian is $\operatorname{diag}(p) - pp^\top$; as $p \to$ one-hot, this $\to 0$, so **gradients vanish**. Dividing by $\sqrt{d_k}$ renormalises the variance to 1, keeping the softmax in its responsive region.

**Remove it**: at $d_k=64$ attention becomes near-argmax at initialisation, gradients through the attention weights vanish, and training stalls or diverges.

### Causal masking, mathematically

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Added *before* softmax, so $\exp(-\infty)=0$: position $i$ assigns exactly zero weight to every future position.

- **Why**: the training objective is $p(y_t \mid y_{<t})$. If position $t$ could see $y_t$, the task is trivially solved by copying and the model learns nothing. Masking is what makes **teacher forcing** valid — it lets us compute the loss at all T positions in one forward pass while each position only ever sees its legitimate past.
- **Implementation**: a `(1,1,512,512)` lower-triangular buffer registered per block (visible in the state dict as `blocks.N.attn.causal_mask`). In practice `-1e9` or `torch.finfo.min` is used rather than literal `-inf` to stay FP16-safe.
- **Remove it**: training loss collapses toward zero, generation is garbage. This is the classic "my loss is suspiciously low" bug.

### Multi-head

$$\text{MHA}(X) = \big[\text{head}_1;\dots;\text{head}_8\big]W_O,\qquad \text{head}_i = \text{Attn}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})$$

- **Why**: one 512-dim softmax produces **one** weighting per position. Eight 64-dim heads produce eight, so the layer can simultaneously attend to (say) the operator token, the left operand, and the sentence boundary.
- Note the parameter count is **unchanged** by head count — you are partitioning $d$, not adding to it. Trade-off: more heads = more distinct patterns but lower per-head resolution.

## 4.5 Residual connections

$$x \leftarrow x + \text{Sublayer}(\text{LN}(x))$$

- **Why**: the gradient path. $\frac{\partial}{\partial x}\big(x + F(x)\big) = I + \frac{\partial F}{\partial x}$ — the identity term guarantees gradient flows to earlier layers even when $\partial F/\partial x$ is small. Across 12 layers, without residuals the product of Jacobians shrinks geometrically.
- Also: each block learns a **delta** on the running representation rather than re-deriving it, which is a far easier optimisation problem.
- **Remove it**: 12-layer training becomes very hard; expect stalled loss.

## 4.6 LayerNorm (pre-LN placement)

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,\qquad \mu,\sigma^2 \text{ over the 512 features of each token}$$

- $\gamma,\beta \in \mathbb{R}^{512}$ learned; $\epsilon$ prevents division by zero.
- **Pre-LN** means normalise *then* sublayer, keeping the residual stream un-normalised — this is what makes deep transformers trainable without aggressive warmup.
- **Why per-token, not per-batch**: BatchNorm statistics depend on other sequences in the batch, which is wrong for variable-length autoregressive text and breaks at batch size 1.
- **Remove it**: activations drift in scale across depth; FP16 overflows.

## 4.7 Feed-forward network

$$\text{FFN}(x) = W_2\,\phi(W_1 x), \qquad W_1\in\mathbb{R}^{2048\times512},\ W_2\in\mathbb{R}^{512\times2048}$$

- **Why the 4× expansion**: attention *mixes* information across positions but is linear in $V$. The FFN is where per-position **nonlinear computation** happens. The widening gives it room; $d_{ff}=4d$ is the standard ratio.
- $\phi$ is the activation (GELU-family in modern stacks; smooth, non-monotonic near zero, better-behaved gradients than ReLU's hard kink).
- **Params**: 2 × 512 × 2048 = 2,097,152 per layer — **two-thirds of each block**.
- **Remove it**: stacked attention is (nearly) a sequence of linear maps; expressive power collapses.

## 4.8 LM head and softmax

$$z = \text{LN}_f(x)\,E^{\top} \in \mathbb{R}^{V}, \qquad p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

- The head is $E^\top$ **because embeddings are tied** — the same 8000×512 matrix used for input lookup scores the output.
- Softmax converts arbitrary reals to a probability simplex. Implementations subtract $\max_j z_j$ first (invariant, prevents `exp` overflow).
- **Remove the softmax**: no valid likelihood, no cross-entropy.

## 4.9 Autoregressive generation

$$p(y_{1:T}) = \prod_{t=1}^{T} p(y_t \mid y_{<t})$$

At inference: feed the prompt, take logits at the last position, choose a token, append, repeat. Mini-R1's harness stops on EOS **or** the `</answer>` token (a single token, id 7 — verified).

**The key asymmetry to state in an interview**: training is *teacher-forced* (the model always conditions on the ground-truth prefix), but generation conditions on **its own output**. One wrong token at step 1 poisons every subsequent step. This is **exposure bias**, and it is the single best explanation for why Mini-R1's multi-step arithmetic traces fail more than its single-step ones — and why Direct SFT (2.73%) beat CoT SFT (1.21%).

---

# PART 5 — PARAMETER COUNT, DERIVED

Do this on a whiteboard without notes.

### Per transformer block

| Component | Formula | Count |
| :--- | :--- | ---: |
| Attention Q,K,V,O | $4 d^2 = 4(512)^2$ | 1,048,576 |
| FFN $W_1$ | $d \cdot d_{ff} = 512 \cdot 2048$ | 1,048,576 |
| FFN $W_2$ | $d_{ff} \cdot d$ | 1,048,576 |
| LN1 $\gamma,\beta$ | $2d$ | 1,024 |
| LN2 $\gamma,\beta$ | $2d$ | 1,024 |
| **Block total** | $12d^2 + 4d$ | **3,147,776** |

### Whole model

| Component | Formula | Count |
| :--- | :--- | ---: |
| Token embedding | $V \cdot d = 8000 \cdot 512$ | 4,096,000 |
| Position embedding | $512 \cdot 512$ | 262,144 |
| 12 blocks | $12 \times 3{,}147{,}776$ | 37,773,312 |
| Final LN | $2d$ | 1,024 |
| LM head | **0 — tied to token embedding** | 0 |
| **TOTAL** | | **42,132,480** ✅ |

Matches the measured `sum(p.numel())` exactly [`task24_final_benchmarks.json`].

**Sanity ratios worth quoting**: FFN is 2/3 of every block; the 12 blocks are 89.7% of the model; embeddings are 10.3%. A useful rule of thumb is $N \approx 12\,L\,d^2$ = 37.75M, i.e. the blocks dominate.

### Tied vs untied embeddings

**Tied** means one matrix $E$ serves both roles: input lookup $E[t]$ and output scoring $x E^\top$.

- **Saves** $V \times d$ = 4,096,000 params = **9.7% of the model**.
- **Justification**: a token's input representation and the direction that should score it highly at output are plausibly related.
- **Untied** would give the head independent freedom at the cost of 4.1M params — a poor trade at this scale.

### The consequence that mattered — Task 22

`tie_word_embeddings=True` means `lm_head.weight` **is the same tensor object** as `embeddings.tok_emb.weight` (verified: `model.lm_head.weight is model.embeddings.tok_emb.weight` → `True`).

In Task 22 the freeze spec was "freeze `tok_emb` and blocks 0–5, train blocks 6–11 + `ln_f`". Because of tying, that **also froze the output projection**. The model retained only `ln_f` as a degree of freedom before a fixed head, and it:
- could not learn to emit the `</answer>` / EOS token in new contexts → **p90 generation length pinned at the 256-token cap**;
- degenerated into repeating tokens it could already produce → `12 + 12 + 12 + …`;
- lost **30.38 points** of greedy equation validity versus the identical run unfrozen [`task22b_replay_only.json`].

A subtle detail worth knowing: PyTorch's `named_parameters()` **de-duplicates** shared tensors, so `lm_head.weight` does not appear as a separate entry. Auditing "which parameters are frozen" by iterating `named_parameters()` will silently miss the tying. We verified it by object identity instead.

---

# PART 6 — TOKENIZER, AND WHY DIGITS MATTER

**SentencePiece BPE, vocab 8,000**, trained on `data/clean/tokenizer_corpus.txt` (math problems, reasoning steps, prose, code, delimiter tags). SHA256 `c1143f6e…7044`.

Special tokens are **atomic**: `<think>`, `</think>`, `<answer>`, `</answer>` each occupy exactly one id (`</answer>` = 7; `</think>` = 5; BOS 1, EOS 2, PAD 3). This is why the evaluator can early-stop on a single token id and why format compliance is cleanly measurable.

### The empirical finding: digits are individual tokens

Verified directly against the tokenizer:

| String | Tokens | Decoded |
| :--- | ---: | :--- |
| `1234` | 4 | `1`,`2`,`3`,`4` |
| `100` | 3 | `1`,`0`,`0` |
| `1738 - 1538 = 200` | 15 | `1`,`7`,`3`,`8`,` -`,` `,`1`,`5`,`3`,`8`,` =`,` `,`2`,`0`,`0` |

Every digit 0–9 is its own token; no multi-digit numeral is ever a single token.

### Why this matters — and the distinction to draw carefully

Digit-level tokenization is generally considered *helpful* for arithmetic: it prevents the vocabulary from arbitrarily fragmenting numbers (`1738` becoming `17`+`38` in one context and `173`+`8` in another), so each digit position is consistently addressable.

**But — and this is the interview point — it only removes an obstacle. It does not supply an algorithm.**

Three distinct things:

1. **Tokenization** — can the model *address* the digits? Yes, cleanly.
2. **Representation** — does the model encode "this token is the hundreds digit of the left operand"? Only insofar as attention + learned absolute positions let it infer that. Never directly verified in this project.
3. **Algorithmic computation** — can it execute carry/borrow across positions in the right order? **This is what it must learn, and it is what failed outside trained regimes.**

The evidence: with digit-level tokens throughout, the model reached **100% on 2-digit subtraction in `[50,99]`** but **0.00% on 5-digit subtraction** [`task20_progressive_numeric_curriculum.json`]. The tokenization was identical in both cases. So tokenization was never the binding constraint — the missing piece is a *positionally general* subtraction procedure, and the model instead learned something regime-bound.

**Say this**: "having digit tokens is necessary-ish but nowhere near sufficient; our 5-digit result is 0% with exactly the same tokenizer that gives 100% at 2 digits."

---

# PART 7 — PRETRAINING

### Corpus and the 60:40 mixture

- **60% `cosmopedia-v2`** — synthetic textbooks/course units generated by Mixtral-8x22B. Clean, pedagogical, densely explanatory prose.
- **40% `fineweb-edu-dedup`** — deduplicated high-quality educational web pages.
- **`python-edu` excluded** — git schema incompatibilities during qualification [`task11b_final_qualification.md`].

**Why this mixture**: the downstream target is mathematical reasoning, so we bias toward explanatory educational text rather than general web crawl. Cosmopedia is cleaner and more instructional but synthetic (risk: narrow stylistic distribution); fineweb-edu is real and more varied but noisier. 60:40 leans on the cleaner source while retaining genuine text diversity. Ratios are enforced **by token count**, not document count.

**Honest caveat**: this ratio was a reasoned choice, not an ablated one. We never ran 50:50 or 70:30. Do not claim it is optimal.

### Streaming, packing, shards, memmap

- **Streaming**: datasets are pulled with HF streaming rather than materialised — 500M tokens of raw text would not comfortably sit on the disk budget.
- **EOS-separated contiguous packing**: documents are concatenated with EOS between them and cut into fixed 512-token windows. Achieved **100% token density**; naive per-document truncation had wasted **58.4%** of tokens [`task10`].
- **uint16 shards**: vocab 8,000 < 65,536, so each token fits in 2 bytes. 500M tokens = ~1 GB on disk.
- **`np.memmap`**: shards are memory-mapped, so the OS page cache serves batches. RAM usage stays flat regardless of corpus size, and slicing is zero-copy.
- **Offline / network isolation**: once shards are built, training touches no network. Reproducible, and immune to a dataset being updated or rate-limited mid-run.

### The 500M budget, arithmetically

$$\text{tokens/optimizer step} = B_{\text{micro}} \times \text{accum} \times T = 8 \times 4 \times 512 = 16{,}384$$

$$\text{steps} = \frac{500{,}000{,}000}{16{,}384} \approx 30{,}518$$

which matches the logged final step **30,518** exactly [`task13_base_pretraining_results.md`]. Schedule: peak LR 6e-4, min 6e-5, warmup 100 steps, cosine over 30,518 steps.

Run: **8.02 hours**, average **17,316.5 tok/s**, peak VRAM **3,409.7 MB** of 6,144 MB.

### Why checkpointing was necessary

An 8-hour unattended run on a laptop GPU. A driver reset, thermal event, OS update, or power loss at hour 7 without resumable checkpoints costs the entire run. Checkpoints also enable **best-vs-latest** separation — which is exactly why `best.pt` is at step 18,000 rather than the final step.

### Why the final 500M checkpoint was *not* selected

`best.pt`: `global_step 18000`, `tokens_seen 294,912,000`, `val_loss 2.8726140` [checkpoint metadata].
Final step 30,518: `val_loss 2.8726`, PPL 17.68 [report].

They are equal to four decimals. Validation loss **plateaued at ~295M tokens**; the remaining 205M tokens (41% of budget) produced no measurable validation improvement, so the best-tracker never replaced the step-18,000 checkpoint.

**What this tells you**: for this architecture and corpus, the model was effectively **data-saturated at ~295M tokens**. A Chinchilla-style heuristic (~20 tokens/param) suggests ~840M tokens for 42M params, so we were arguably *under*-trained by that rule and still plateaued — which points at capacity or corpus diversity as the binding constraint rather than token count. **State that as a hypothesis, not a finding — we did not run the ablation.**

---

# PART 8 — OPTIMIZATION MATHEMATICS

## 8.1 The objective

$$\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \log p_\theta\!\left(y_t \mid y_{<t}, x\right)$$

- $y_t$: the correct next token; $p_\theta(\cdot)$: model softmax; $N$: number of **supervised** positions.
- For a single position with true class $c$: $\mathcal{L}_t = -\log p_c = -z_c + \log\sum_j e^{z_j}$.
- **Gradient w.r.t. logits** — memorise this, it is the most-asked derivation:

$$\frac{\partial \mathcal{L}_t}{\partial z_i} = p_i - \mathbb{1}[i=c]$$

Beautifully simple: *predicted minus actual*. Push down every wrong class in proportion to how much probability it got; push up the true class by $1-p_c$.

- In SFT, prompt positions carry label $-100$ and are **excluded from $N$** entirely (`ignore_index=-100`), so they contribute neither loss nor gradient.

## 8.2 Teacher forcing

During training, position $t$ conditions on the **ground-truth** prefix $y_{<t}$, not on what the model would have generated. Combined with causal masking this lets us compute all T losses in **one** forward pass — an enormous efficiency win.

The cost is **exposure bias**: at inference the model conditions on its own (possibly wrong) prefix, a distribution it never trained on. This is the mechanism behind Mini-R1's compounding multi-step failures and is a direct, honest explanation for Direct SFT (2.73%) beating CoT SFT (1.21%).

## 8.3 AdamW

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\boxed{\;\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_{t-1}\right)}$$

- $m$: first moment (momentum). $v$: second moment (per-parameter scale). Bias correction fixes the zero-initialisation of $m,v$, which otherwise makes early steps far too small.
- **Per-parameter adaptive step**: dividing by $\sqrt{\hat v}$ means rarely-updated parameters (rare token embeddings) still move. Critical for language models with heavy-tailed token frequencies.
- **The "W"**: weight decay is applied **directly to the parameters**, not added into the gradient. In plain Adam, L2-in-the-gradient gets divided by $\sqrt{\hat v}$, so parameters with large gradients get *less* regularisation — the opposite of the intent. AdamW decouples them.
- Mini-R1: $\beta=(0.9, 0.95)$, $\epsilon$=1e-8, $\lambda$=0.01. $\beta_2=0.95$ rather than 0.999 is standard for LM pretraining — faster adaptation to changing curvature.
- **Memory cost**: $m$ and $v$ are each the size of the parameters. Adam therefore costs **2× params** in optimizer state — see Part 9.

## 8.4 Warmup + cosine schedule

$$\eta_t = \begin{cases}
\eta_{\min} + (\eta_{\text{peak}}-\eta_{\min})\dfrac{t}{T_w} & t < T_w \\[2ex]
\eta_{\min} + (\eta_{\text{peak}}-\eta_{\min})\cdot\dfrac{1}{2}\left(1+\cos\dfrac{\pi(t-T_w)}{T-T_w}\right) & t \ge T_w
\end{cases}$$

- **Why warmup**: at step 0, Adam's $\hat v$ is estimated from almost no data and is unreliable; a full-size step on a bad curvature estimate can wreck initialisation. Warmup lets the moments stabilise.
- **Why cosine**: smooth decay to a small final LR lets the model settle into a minimum instead of bouncing. No discontinuities (unlike step decay).
- Pretraining: peak 6e-4, min 6e-5, warmup 100, over 30,518 steps. SFT: peak 1e-4, min 1e-5, warmup 50, over 650 steps.
- **Why SFT LR is 6× lower**: we are adapting a trained model, not training from scratch. High LR would destroy pretrained representations — which, notably, is *exactly what happened to the arithmetic anyway*, even at 1e-4.

## 8.5 Gradient clipping

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert_2}\right), \qquad c = 1.0$$

Rescales the **whole gradient vector** (preserving direction) when its global L2 norm exceeds $c$. Guards against loss spikes from rare pathological batches. Must be applied **after** `scaler.unscale_()` — clipping scaled gradients would clip at the wrong threshold.

## 8.6 Gradient accumulation — the mathematics

The claim is that micro-batch 8 × accumulation 4 ≈ true batch 32. Why:

$$\nabla \mathcal{L}_{\text{batch32}} = \nabla\left(\frac{1}{32}\sum_{i=1}^{32}\ell_i\right) = \frac{1}{32}\sum_{i=1}^{32}\nabla\ell_i$$

Split into 4 micro-batches of 8. If each micro-step computes $\frac{1}{8}\sum_{i \in \text{micro}}\ell_i$ and we divide by `accum`=4 before `backward()`, the accumulated gradient buffer holds

$$\sum_{k=1}^{4}\frac{1}{4}\cdot\frac{1}{8}\sum_{i\in \text{micro}_k}\nabla\ell_i = \frac{1}{32}\sum_{i=1}^{32}\nabla\ell_i$$

— identical. Gradients accumulate additively in `.grad` because PyTorch does not zero them automatically; that default, usually a footgun, is the feature here.

**Why "equivalent" is approximate in practice:**
1. **Floating-point non-associativity** — $(a+b)+c \ne a+(b+c)$ in finite precision. Four sequential accumulations sum in a different order than one fused reduction.
2. **FP16 accumulation error** — gradients are accumulated in reduced precision; small contributions can be absorbed.
3. **Batch-dependent layers** — none here (LayerNorm is per-token), but BatchNorm would break the equivalence outright.
4. **Padding effects** — if micro-batches pad to different lengths, per-token normalisation can differ slightly.

**Say**: "mathematically exact in real arithmetic, numerically approximate in floating point, and the difference is far below the noise of a single seed."

## 8.7 AMP and GradScaler

FP16 has ~10 bits of mantissa and a minimum normal magnitude around 6e-5. Small gradients **underflow to zero**.

GradScaler:
1. multiply loss by a large scale $S$ (dynamic, typically $2^{16}$-ish);
2. backward — all gradients scaled by $S$, lifted out of the underflow region;
3. `unscale_()` — divide gradients by $S$ before clipping/stepping;
4. inspect for inf/NaN. If found, **skip the step** and halve $S$. If clean for N steps, double $S$.

Master weights are kept in FP32; only the forward/backward compute is FP16. Mini-R1 logged **0 skipped steps** with FP16 numerical error of 0.1085% vs FP32 [`task8`].

**Why not BF16**: BF16 has FP32's exponent range so no scaler is needed — strictly nicer. It was not used because reliable BF16 support on this consumer Ada laptop SKU/stack was not established, whereas FP16+GradScaler was empirically verified. That is an engineering decision, not a claim BF16 is worse.

---

# PART 9 — HARDWARE AND VRAM

**Constraint**: NVIDIA RTX 4050 Laptop, **6 GB** VRAM. Every architectural decision traces back to this.

## Where training memory goes

For $N$ = 42.13M parameters:

| Item | Formula | Size |
| :--- | :--- | ---: |
| FP32 master weights | $4N$ | 168.5 MB |
| Gradients | $4N$ | 168.5 MB |
| Adam $m$ | $4N$ | 168.5 MB |
| Adam $v$ | $4N$ | 168.5 MB |
| **Static subtotal** | $16N$ | **~674 MB** |
| FP16 activations | $\propto B \cdot T \cdot d \cdot L$ | batch-dependent |
| Attention matrices | $B \cdot h \cdot T^2$ per layer | the dangerous term |

**Measured peak: 3,409.7 MB** — 2,734 MB headroom [`task13`].

### The quadratic term is what actually bites

Attention scores are $(B, h, T, T)$. At $B{=}8, h{=}8, T{=}512$: $8 \times 8 \times 512^2 = 16.8$M values = **33.6 MB in FP16 per layer**, and with 12 layers that is ~400 MB if all are retained for backward. Doubling $T$ to 1024 **quadruples** this. This is the direct reason context is capped at 512.

### Training vs inference memory

Inference needs only weights + one activation slice: FP16 weights are $2N \approx$ **84 MB**. Training needs ~**8× more** static state (FP32 master + gradients + two Adam moments) *plus* all retained activations for backpropagation. That is why a model that trains at the edge of 6 GB serves comfortably in under 1 GB.

### How the project stayed under 6 GB

1. 42M params (not 100M+) — sized deliberately to the card.
2. Micro-batch 8 with accumulation 4 — activations for 8, gradient statistics of 32.
3. Context 512 — caps the $T^2$ term.
4. FP16 AMP — halves activation memory.
5. Tied embeddings — 4.1M fewer params in all four static buffers.

### Reserved vs allocated

- **Allocated**: bytes actively held by live tensors (`torch.cuda.memory_allocated()`).
- **Reserved**: bytes PyTorch has taken from the driver into its caching allocator (`torch.cuda.memory_reserved()`).

PyTorch caches freed blocks rather than returning them to CUDA, because `cudaMalloc`/`cudaFree` are expensive and synchronising. So **reserved ≥ allocated**, often substantially, and reserved memory does not shrink when tensors are freed. Fragmentation can also cause OOM while `reserved − allocated` looks large — the free space exists but not as one contiguous block. `torch.cuda.empty_cache()` returns cached blocks (used in the evaluator between models).

**Always specify which number you mean.** Mini-R1's 3,409.7 MB is *allocated*.

### Throughput

Pretraining averaged **17,316.5 tok/s** (~21,700 peak). Sanity check: $30{,}518 \text{ steps} \times 16{,}384 \text{ tok} / (8.02 \times 3600 \text{ s}) \approx 17{,}300$ tok/s. ✅ Consistent.

---

# PART 10 — CHECKPOINTING AND REPRODUCIBILITY

## Atomic writes

```python
torch.save(state, path + ".tmp")
os.replace(path + ".tmp", path)      # atomic on the same filesystem
```

`os.replace` is a single filesystem operation: the name points either at the old file or the new one, never a half-written file. Without it, a crash during a multi-hundred-MB `torch.save` leaves a corrupt checkpoint — and if that checkpoint was `best.pt`, the run is lost. Used for every best-checkpoint write in the SFT trainers.

`best.pt` (lowest validation loss), `latest.pt` (most recent, for resume), `step_N.pt` (periodic snapshots).

## What a complete checkpoint must contain

Verified present in `checkpoints/mini_r1_500m_pretrain/best.pt`: `model_state_dict`, `optimizer_state_dict`, `global_step`, `tokens_seen`, `epoch`, `val_loss`, `config`, `dataset_state`, `torch_rng_state`, `cuda_rng_state`, `numpy_rng_state`, `python_rng_state`, `training_metrics`.

| Item | Why resume breaks without it |
| :--- | :--- |
| Optimizer state ($m$, $v$, step count) | Adam restarts with zero moments → bias correction misfires → a large, wrong first step |
| Scheduler position | LR jumps back to warmup or to the wrong point on the cosine |
| GradScaler scale | Scale re-searches from scratch; possible skipped steps or overflow |
| RNG states (torch/cuda/numpy/python) | Dropout masks, shuffles, and sampling diverge from the original trajectory |
| Dataset cursor: shard index, sequence index, `tokens_seen` | You re-read data already seen, or skip data — the token budget becomes a lie |
| Manifest hash | Silently different data would produce a "resumed" run that is actually a different experiment |

## Why scientific reproducibility ≠ "the loss looked similar"

"Similar loss" is consistent with: a different data order, a different subset of the corpus, a different LR trajectory, or a different RNG stream. Any of those makes the resumed run **a different experiment**, so any comparison against a pre-resume baseline is confounded.

The stronger standard is **bitwise parity**: run N steps, checkpoint, restart from the checkpoint, run M more; separately run N+M steps uninterrupted; assert the resulting parameters (and loss trajectory) match exactly. If they do, resume is a no-op with respect to the science and any interrupted run remains comparable to any uninterrupted one.

**Honest scope note**: the repository contains a gradient-accumulation equivalence test (`tests/test_gradient_accumulation.py`) and a checkpoint test (`tests/test_checkpoint.py`), and the pretraining checkpoint stores all the state listed above. Describe the bitwise-parity criterion as **the standard the checkpoint design targets**; do not assert a specific parity experiment result unless you re-run and verify it.

Caveats to mention: exact bitwise reproducibility also requires deterministic cuDNN kernels (`torch.use_deterministic_algorithms(True)`), fixed batch shapes, and identical hardware — non-deterministic reduction order in some CUDA kernels can break parity even with identical seeds.
---

# PART 11 — PRETRAINING RESULTS

## The LR sweep (Task 8C/9)

1,000-step qualification over {1e-4, 3e-4, 6e-4} [`long_run_qualification_report.md`]:

| Peak LR | Val loss @ step 1000 | PPL |
| :--- | ---: | ---: |
| 3e-4 | 2.6337 | 13.93 |
| **6e-4** | **2.4995** | **12.18** |
| 1e-4 | — | converged too slowly for the budget |

6e-4 locked in. **Why the higher LR won**: with only 30,518 total steps, a conservative LR spends too much of the budget in the slow early phase. Cosine decay plus warmup plus gradient clipping make the aggressive peak safe.

**Honest limitation**: this was decided at **step 1,000 of 30,518** (3.3% of training). Early-training loss is a decent but not guaranteed proxy for final loss — a higher LR can win early and lose late. We never re-ran the full budget at 3e-4 to check. Say so if asked.

## The training curve

| Milestone | Step | Tokens | Train loss | Val loss | Val PPL | LR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50M | 3,052 | 50,004,160 | 3.0124 | 3.5182 | 33.72 | 5.89e-4 |
| 100M | 6,104 | 100,008,320 | 2.9412 | 3.3210 | 27.69 | 5.56e-4 |
| 200M | 12,208 | 200,016,640 | 2.8941 | 3.1405 | 23.12 | 4.47e-4 |
| **best.pt** | **18,000** | **294,912,000** | — | **2.8726** | **17.68** | — |
| 300M | 18,311 | 300,007,424 | 2.8250 | 3.0118 | 20.32 | 3.09e-4 |
| 400M | 24,415 | 400,015,680 | 2.7831 | 2.9402 | 18.92 | 1.63e-4 |
| 500M final | 30,518 | 500,000,256 | 3.0807 | 2.8726 | 17.68 | 6.00e-5 |

**Read the anomaly**: `best.pt` at step 18,000 records val loss 2.8726, but the *milestone* row at step 18,311 records 3.0118. These come from different evaluation events (the best-tracker's own validation pass vs. the milestone logging pass) and should not be treated as one series. The defensible statement is the checkpoint metadata itself: **best.pt = step 18,000, 294.9M tokens, val loss 2.8726140, PPL 17.6832**, and the final 500M checkpoint reached the same 2.8726 without beating it. Flag this as a logging inconsistency in the report rather than papering over it.

## Train loss vs validation loss

- **Train loss**: measured on data the model is currently fitting, typically a noisy per-step or short-window value.
- **Validation loss**: full pass over 10M **held-out** tokens (9,980,852 evaluated), never trained on.

The final row shows train 3.0807 > val 2.8726. This inversion is expected here: train loss is one noisy step near the end of cosine decay; val loss is an average over 10M tokens. **Averaging reduces variance.** It is *not* evidence of underfitting or of a broken split.

The classic overfitting signature — train loss falling while val loss rises — **never appeared**. Val loss decreased monotonically and then flattened. With 500M tokens against 42M params, we are in the data-rich regime; memorisation of the corpus was never the risk.

## Perplexity

$$\text{PPL} = \exp(\mathcal{L}), \qquad \mathcal{L} = -\frac{1}{N}\sum \log p(y_t \mid y_{<t})$$

Check: $\exp(2.8726140) = 17.683$. ✅

**Interpretation**: PPL is the *effective branching factor* — the model is as uncertain as if choosing uniformly among ~17.7 options at each token. Uniform over the 8,000 vocab would be PPL 8,000; PPL 17.68 means substantial structure learned.

**What PPL tells you**: general language-modelling quality; comparable **only** across identical tokenizer + identical evaluation set. It is a sensitive, cheap forgetting detector — which is exactly how it is used from Task 16A onward.

**What PPL does NOT tell you** — the interview point:
- **Nothing about reasoning.** Our base model had PPL 17.68 and GSM8K **0.76%**.
- **Nothing about arithmetic correctness.** A model can be confidently fluent and numerically wrong.
- **It is not comparable across tokenizers.** Different vocab ⇒ different tokens per sentence ⇒ incomparable per-token loss. Comparing Mini-R1's 17.68 to a published GPT-2 number would be meaningless.
- It rewards hedging on high-entropy positions; task metrics don't.

**Mini-R1's PPL trajectory as a forgetting log**: base **17.68** → Direct SFT 21.83 → CoT SFT 23.58 → Task 20 arithmetic 27.63 → Task 21A 28.87 → Task 21B 43.88 → Task 22 (frozen) 30.33 → Task 22B 60.02 → Task 23 10% 47.90 → **final 50.74**. Every fine-tune cost language quality; the final model is 2.9× base.

---

# PART 12 — ZERO-SHOT REASONING FAILURE (Task 14)

The base model, greedy, on GSM8K test (1,319):

| Metric | Value |
| :--- | ---: |
| Strict accuracy | **0.76%** (10/1319) |
| Sampled accuracy | 0.91% (12/1319) |
| Extraction rate | 48.1% (634/1319) |
| **Format compliance** | **0.0%** |
| Mean output length | 125.7 tokens |
| Latency | 4.288 s/sample |

### Behaviours observed

- **Textbook continuation** — the base model was pretrained on educational prose, so given a word problem it continues the *document*: adds exercises, writes a new section heading, drifts into an unrelated lesson. It is doing exactly what it was trained to do; it has no notion that a question expects an answer.
- **Repetition/degeneration** under greedy decoding.
- **No format compliance whatsoever** — 0.0%. It has never seen `<think>`/`<answer>`; those tokens exist in the vocabulary but carry no learned function.
- **Weak arithmetic** — numbers appear plausible in magnitude and are wrong.
- **AQuA** — multiple-choice remained 0.00% for essentially every model in the project including the final one, and format compliance on AQuA is **0.0%** even for the final model [`task24_final_benchmarks.json`]. The MCQ prompt shape is simply outside everything these models were trained on.

### Strict vs fallback extraction — why "apparent accuracy" misleads

- **Strict extraction**: parse only `<answer>…</answer>`; if absent, the item is wrong.
- **Fallback extraction**: hunt for any number in the output — last number, largest number, first number after "=", etc.

Fallback inflates scores for a model that emits many numbers. A base model that writes 125 tokens of textbook prose mentioning a dozen numbers gets many free chances for one to match the gold answer. **That is measuring luck, not reasoning.**

Note the base model's 48.1% extraction rate against 0.76% accuracy: it produced a parseable number in nearly half of cases and was right in under 1% of them.

**Mini-R1 uses strict extraction everywhere from Task 16A onward.** This makes our numbers *lower* and *comparable*. If an interviewer says "2.43% seems low," part of the honest answer is "it's strict-extraction, format-required, single-answer accuracy — a fallback parser would report a higher and less meaningful number."

---

# PART 13 — DIRECT SFT vs CoT SFT (Task 15A–15C)

## Hypothesis

Explicit chain-of-thought supervision teaches intermediate reasoning steps and should outperform direct-answer supervision — the finding that made CoT famous at large scale.

**Result: the opposite, decisively.**

## Data formats

**Direct** — prompt masked, only the answer supervised:
```
Question: Natalia sold clips to 48 friends in April, and then she sold half as many in May. How many clips did Natalia sell altogether?
<answer>72</answer>
```

**CoT** — prompt masked, full reasoning trace supervised:
```
Question: Natalia sold clips to 48 friends in April, and then she sold half as many in May. How many clips did Natalia sell altogether?
<think>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether.
</think>
<answer>72</answer>
```

## Completion-only masking

```python
labels = target_ids.copy()
for i in range(len(prompt_toks) - 1):
    labels[i] = -100                       # prompt positions excluded
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
```

**Why**: we want $p(\text{answer} \mid \text{question})$, not $p(\text{question})$. Supervising the prompt spends capacity learning to *regenerate GSM8K questions*, which is not the task and, at 42M, is capacity we cannot spare. The `-100` positions contribute neither loss nor gradient.

The off-by-one (`len(prompt_toks) - 1`) is the causal shift: with `input = seq[:-1]`, `target = seq[1:]`, the target at index $i$ is `seq[i+1]`, so the last prompt position already predicts the first completion token and must stay supervised.

## Results (Task 15C)

| Metric | Direct SFT | CoT SFT | Δ |
| :--- | ---: | ---: | ---: |
| **GSM8K strict** | **2.73%** | **1.21%** | **−1.52%** (p = 0.0051) |
| 3-step problems | 2.50% (9/360) | 1.39% (5/360) | −1.11% |
| 4+-step problems | 2.17% (14/644) | 0.78% (5/644) | −1.40% |
| Repetition | 0.00% | **82.6%** | +82.6 |
| Mean length | 4.6 | 158.6 | 34× |
| **Active completion tokens** | **110,594** | **2,106,652** | **19.05×** |
| Val PPL | 21.83 (+23.5%) | 23.58 (+33.4%) | — |

Note the deficit **grows with problem depth** (−1.11% at 3 steps, −1.40% at 4+). That is the compounding-error signature.

## Why teacher-forced CoT need not produce free-running reasoning

Under teacher forcing, predicting step $k$ conditions on the **ground-truth** steps $1..k-1$. The model can learn the *local* transition "given these correct previous steps, the next line looks like this" without ever learning to produce a correct chain unaided.

At inference it conditions on **its own** step $k-1$. If a 42M model's per-step arithmetic accuracy is $q$, an $n$-step chain succeeds with roughly $q^n$ — so CoT converts one hard problem into $n$ chances to fail. Direct answering has no intermediate surface to corrupt.

Combined with 82.6% repetition, CoT produced long degenerate traces that also *diluted* the answer signal.

## The budget confound — the sophisticated point

Both arms ran **650 optimizer steps** with identical batch shape. That equalises *optimizer steps*, not *supervision*:

- Direct: **110,594** supervised completion tokens
- CoT: **2,106,652** — **19.05×** more

So CoT received ~19× the token-level supervision and still lost by 1.52 points. Two consequences:

1. **The negative result is strengthened, not weakened.** CoT lost despite far more supervision.
2. But "matched steps" ≠ "matched budget", and one *could* argue the arms differ in effective epochs over their respective signal. The clean designs would be to match completion tokens (fewer CoT examples) or to match wall-clock. **We matched steps. Name the confound before the interviewer does.**

This is also why Task 21B's compact traces mattered: they cut completion tokens **85.3%** versus verbose CoT, shrinking exactly this asymmetry.

---

# PART 14 — DECODING FORENSICS (Task 16B)

## The question

Was CoT's failure an **inference artifact** (degenerate decoding) or a **representational absence** (no reasoning learned)? Critically, this was run on the **frozen** CoT checkpoint — no retraining — so decoding is the only variable.

## The methods, mathematically

**Greedy**: $y_t = \arg\max_i p_i$. Deterministic; maximises local probability; prone to loops because a high-probability token can re-enter a high-probability state.

**Temperature**:
$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$
- $T \to 0$: → argmax (greedy).
- $T = 1$: unmodified distribution.
- $T > 1$: divides logits, **shrinking gaps** between them, so the distribution flattens toward uniform → more diverse, more error-prone.
- $T < 1$: sharpens toward the mode.
- Why it flattens: as $T\to\infty$, $z_i/T \to 0$ for all $i$, and softmax of the all-zeros vector is exactly uniform.

**Top-p (nucleus)**: sort by probability descending, keep the smallest set with cumulative mass $\ge p$, renormalise. Adaptive: a confident distribution keeps 1–2 tokens, a flat one keeps many. Superior to top-k, which uses a fixed count regardless of confidence.

**Repetition penalty** (as implemented in this repo):
$$z_i \leftarrow \begin{cases} z_i / \rho & z_i > 0 \\ z_i \cdot \rho & z_i \le 0\end{cases} \quad \text{for every already-seen token } i,\ \rho = 1.2$$
The sign-dependence is essential: dividing a negative logit would *increase* it. This directly suppresses tokens already emitted, breaking loops.

## Results — five conditions on the frozen CoT checkpoint

| Condition | Repetition | Full format | Strict acc |
| :--- | ---: | ---: | ---: |
| A: greedy | **82.64%** | 39.80% | 1.21% |
| B: T=0.6, p=0.95 | 64.90% | 62.40% | — |
| C: rep_pen=1.2 | 29.64% | 81.50% | — |
| **D: T=0.6, p=0.95, rep=1.2** | **11.45%** | **90.67%** | **2.20%** |
| E: D + max 384 tokens | 10.92% | 92.19% | — |
| *(Direct SFT reference)* | *0.00%* | *0.00%* | *2.73%* |

## What this proved — and did not

**Proved**: the degeneration was genuinely a decoding pathology, and it is fixable at inference time with no retraining. Repetition fell **7.2×**; format compliance rose from 39.8% to 90.7%.

**Did NOT prove — and this is the whole point**: that reasoning was present but hidden. Accuracy went 1.21% → 2.20%, **still below Direct SFT's 2.73%**. If correct reasoning had been latent and merely obscured by loops, cleaning up the text would have revealed it. It did not.

**Conclusion**: the traces became *structurally clean and mathematically wrong*. This reframed the entire project — the bottleneck is **arithmetic execution**, not text quality — and motivated Task 17A's direct arithmetic probe.

A methodological note worth making: Task 16B is a clean **ablation on a frozen checkpoint**. Only decoding varied; weights were untouched. That is why the causal claim ("decoding caused the repetition") is legitimate here, whereas most of the project's other comparisons required more care about confounds.

---

# PART 15 — SYNTHETIC ARITHMETIC PROBE (Task 17A)

## Why it was needed

GSM8K conflates at least four abilities: reading comprehension, semantic→operation mapping, arithmetic execution, and output formatting. A single 2% number cannot tell you which one failed. **You cannot debug a pipeline with an end-to-end metric.**

## Design

**9,000 deterministic synthetic problems, seed 42** [`task17a_arithmetic_probe.md`], spanning:
- single operations (+, −, ×, ÷)
- 2-op, 3-op, 4-op compositions
- sequential state tracking
- word-wrapped variants
- across three prompt formats

Everything verified in Python. No linguistic difficulty — if the model fails here, it is arithmetic, full stop.

## The critical finding

The model produced **reasoning-shaped text with wrong numbers**. Representative forensic traces recorded in the report show equation-verification counts of `0 valid / 1 total`, `0 valid / 2 total`, and outputs such as `The total number of 381 + 260 = 981` (the true value is 641).

The canonical illustration used throughout the project is of the form `6 + 8 = 375`: syntactically perfect, semantically confident, arithmetically absurd.

## Why "6 + 8 = 375" is not reasoning

It demonstrates the model learned the **surface form** of a calculation — operand, operator, operand, equals, number — as a *sequence pattern*, without the **function** that maps operands to results.

Formally, it learned $p(\text{token} \mid \text{context})$ well enough to place a plausible-looking numeral in the slot where a numeral belongs. It did not learn $f(a,b) = a+b$.

This distinction is the intellectual core of the project, and it motivates the **step-level equation validity** metric that becomes primary from Task 22 onward: rather than only asking "was the final answer right", extract every emitted equation and re-evaluate it in Python. That metric can distinguish "wrong because it set up the wrong computation" from "wrong because it computed incorrectly" — the two failure modes that a single accuracy number hides.

---

# PART 16 — VERIFIED ARITHMETIC SUPERVISION (Tasks 18A/18B)

## Why deterministic labels

Every label is generated by executing the arithmetic in Python, not by scraping or by another model. Consequences: zero label noise, exact difficulty control (operand ranges, operator mix, number of steps), unlimited data, and — crucially — **the evaluation metric is the same function as the label generator**, so "correct" is unambiguous.

Task 18A produced a 6-level corpus: 13,200 unique problems, 26.4k verified traces.

## Compact vs verbose

**Verbose** — natural-language reasoning:
```
<think>
First, we need to subtract 143 from 164.
164 minus 143 equals 21.
</think>
<answer>21</answer>
```

**Compact** — pure computation:
```
<think>
164 - 143 = 21
</think>
<answer>21</answer>
```

Compact traces are far shorter (85.3% fewer completion tokens than verbose GSM8K CoT at Task 21B), contain **no linguistic filler to degenerate into**, and every line is machine-checkable. The trade-off is that they carry no explanatory scaffolding.

## Results (Task 18B)

| Condition | IID dev | OOD numbers | OOD composition | Word-wrapped |
| :--- | ---: | ---: | ---: | ---: |
| Compact arithmetic SFT | **45.42%** | **1.00%** | — | — |
| Verbose arithmetic SFT | **49.42%** | ~1% | — | — |

The IID → OOD-numbers collapse from ~45–49% to **1.00%** is the headline. It "establishes the exact generalization boundary of the 42.1M parameter model" [`task18b_verified_arithmetic_sft.md`].

## Interpolation vs generalization vs extrapolation

- **Interpolation** — test points inside the convex hull of training data. E.g. trained on 2-digit operands in [50,99], tested on *unseen pairs* in [50,99]. Mini-R1 does this well (Task 20 E1: **100%**).
- **Generalization** — performing on a genuinely held-out sample from the *same* distribution. Also fine when the regime matches.
- **Extrapolation** — test points outside the training support. 5-digit operands after training only on 2–4 digits. Mini-R1: **0.00%** (Task 20 E7).

**The precise claim to make**: Mini-R1 learned a **regime-local** arithmetic competence — reliable interpolation within trained numeric ranges, essentially zero extrapolation beyond them.

**The claim NOT to make**: that this reveals a specific mechanism (positional encoding limits, digit-position circuits, induction heads). Those are plausible hypotheses. **We ran no interpretability experiment**, so we cannot attribute the failure to any named internal mechanism. Say "we observed regime-local behaviour; we did not localise its cause."

---

# PART 17 — LEAKAGE FORENSICS (Tasks 19A/19B) — the most important methodological section

## The false result

Task 19B initially reported **24.28%** accuracy on 3-digit division (76/313 correct) and read this as meaningful generalization.

## The audit

Every evaluation pair $(D, d)$ was checked against the full training corpus — including numbers appearing *inside* `<think>` traces of multi-step problems, not just as top-level prompts.

| Subset | Definition | N | Correct | Accuracy | Share of the 76 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Total** | complete benchmark | 313 | 76 | **24.28%** | 100% |
| **A: exact training matches** | pair appeared in training | **111** | 71 | **63.96%** | **93.4%** |
| ↳ A1: single-step prompt match | exact `Calculate: D / d` seen | 84 | 69 | 82.14% | 93.4% |
| ↳ A2: match only inside a `<think>` trace | | 27 | 2 | 7.41% | 0.0% |
| **B: never in training** | truly unseen | **202** | **5** | **2.48%** | **6.6%** |

**Overlap rate: 35.46% (111/313).** The real generalization number is **2.48%**, not 24.28% — a **9.8× overstatement**.

## Why the original conclusion was invalid

The benchmark was not a test of generalization; it was **~35% a memorisation test**. The model scored 82.14% on prompts it had literally seen and 2.48% on ones it had not. The aggregate 24.28% is a weighted average of *retrieval* and *computation* reported as if it were computation.

Note the beautiful internal evidence: subset A1 (seen as a standalone prompt) → 82.14%, but A2 (the same numbers seen only *inside* a longer trace) → 7.41%. Even memorisation was **format-specific**. The model was retrieving a prompt→completion mapping, not a numeric fact.

## Why this is good research practice

- The audit was run **against our own favourable result**. The incentive is always to accept a good number and move on.
- It was **quantitative and reproducible**, not a hand-wave — every subset counted and reported.
- It **changed downstream methodology permanently**: Tasks 19C/19D/20 rebuilt clean disjoint splits, and every later arithmetic claim is restricted to explicitly held-out suites.
- The original number was **preserved in the report** rather than deleted.

**In an interview this is an asset, not an embarrassment.** "I found leakage in my own experiment and corrected it" demonstrates exactly the skepticism research requires. Volunteer it.

## The general lessons

**Data leakage** — evaluation information present in training. Forms: exact duplication, near-duplication, distributional overlap, and (the one that caught us) **substring/embedded occurrence** inside longer training examples.

**How to detect it**: hash or key every eval item and search the *entire* training corpus, including nested content. Then **stratify the result** by seen/unseen and report both. If overlap is non-zero, the aggregate number is not a generalization measure.

**Matched controls** — the deeper fix. In Tasks 21A/21B/22B/23, every comparison uses two models differing in exactly one variable, trained with identical data, steps, seed, batch, and schedule. That is what licenses causal language.

---

# PART 18 — NUMERIC GENERALIZATION (Tasks 19C → 19D → 20)

## The conceptual progression

```
Task 19C:  train 2D only          →  unseen 2D: 35.40%   unseen 3D:  6.60%
              │                       (interpolates)      (does not transfer)
              ▼  add 3-digit exposure
Task 19D:  train 2D + 3D          →  unseen 2D: 28.40%   unseen 3D: 34.40%  untrained 4D: 0.40%
              │                                          (+27.80 vs 19C)
              ▼  add 4-digit exposure
Task 20:   train 2D + 3D + 4D     →  E1 IID 2D: 100.0%   E3 IID 3D: 99.8%   E5 IID 4D: 67.2%
                                      E2 OOD 2D:  18.2%   E4 OOD 3D: 26.2%   E6 OOD 4D:  8.6%
                                                                             E7 ACID 5D: 0.00%
```

## The numbers, precisely

| Measurement | Value | Source |
| :--- | ---: | :--- |
| 19C: unseen 2D `[2,49]` | **35.40%** | `task19c_clean_generalization.md` |
| 19C: OOD 3D `[101,199]`, matched 1–2 digit answers | **6.60%** | same |
| 19C: OOD 3D with 3-digit answers | **0.00%** | same |
| 19D: OOD 3D `[200,299]` after 3D exposure | **34.40%** (+27.80) | `task19d_mixed_digit_subtraction.md` |
| 19D: unseen 2D | 28.40% (−7.00 vs 19C) | same |
| 19D: untrained 4D `[1001,1999]` | **0.40%** | same |
| 20: E5 IID 4D after 4D exposure | **67.20%** | `task20_*.json` |
| 20: E6 OOD 4D | **8.60%** | same |
| 20: E7 acid 5D | **0.00%** | same |

## What this suggests — separating evidence from inference from speculation

**EVIDENCE (directly measured, reproducible):**
1. Training on a numeric regime dramatically improves accuracy *within* that regime (3D OOD: 6.60% → 34.40% after adding 3D data; 4D: 0.40% → 67.20% IID after adding 4D data).
2. Improvement does **not** propagate to untrained regimes (4D stayed at 0.40% while 2D and 3D were being learned).
3. Extrapolation one digit beyond the widest trained regime is **0.00%** (E7).
4. Adding regimes can **cost** performance on earlier ones: unseen 2D fell 35.40% → 28.40% between 19C and 19D. There is interference even within arithmetic.
5. Within a regime, answer length matters independently: 19C's OOD 3D scored 6.60% with 1–2 digit answers but **0.00%** with 3-digit answers.

**REASONABLE INFERENCE (consistent with the data, not proven):**
- The model is learning something closer to a **per-regime lookup/pattern-completion** strategy than a single positional subtraction algorithm. If it had learned general column-wise subtraction with borrow, 5-digit inputs would not be 0.00% — the algorithm is digit-count agnostic.
- Observation 5 hints that generating longer answers is itself a difficulty axis, separate from reading longer inputs.

**SPECULATION (do not assert):**
- Any claim about *which* internal mechanism is responsible — absolute positional encoding limits, absent carry-propagation circuits, induction-head deficits. These are testable hypotheses requiring interpretability work (activation patching, probing digit-position representations, attention-pattern analysis) that **this project did not do.**

**The line to use**: "We proved the behaviour is regime-local. We did not prove *why*, and I'd want activation-level experiments before claiming a mechanism."

---

# PART 19 — ARITHMETIC → GSM8K TRANSFER (Tasks 21A, 21B)

## Task 21A: verbose CoT targets

Matched controls: identical data, steps, seed, batch. Only the **initialization** differs.

| | Model A: Arithmetic-init | Model B: Base-init | Δ |
| :--- | ---: | ---: | ---: |
| GSM8K | **1.44%** | 1.06% | **+0.38%** |
| SVAMP | 2.00% | 0.33% | — |
| AQuA | 0.00% | 0.00% | — |
| Format | 14.6% | 30.4% | — |
| Repetition | 86.1% | 96.3% | — |

## Why +0.38% is not evidence of useful transfer

1. **Absolute scale**: +0.38% of 1,319 problems = **5 problems**. Five.
2. **No confidence interval / significance test** was reported for this delta. At these rates a 95% CI on a proportion near 1.2% with n=1319 is roughly ±0.6%, so the interval comfortably contains zero.
3. **The report's own verdict** was "Conclusion C: No measurable transfer."
4. **It did not replicate.** Task 21B, the better-designed follow-up with compact traces, measured **−0.30%** — the opposite sign. Two runs, opposite signs, both tiny: that is noise.

**The lesson to articulate**: a positive point estimate is not a positive result. Always ask "how many items is that, and what is the interval?"

## Task 21B: compact traces

Compact verified equation traces fixed the *mechanics* completely — format compliance 14.6% → **97.8%**, completion tokens **−85.3%**, mean length 159.6 → 62.9 — and changed nothing about transfer:

| | Model A | Model B | Δ |
| :--- | ---: | ---: | ---: |
| GSM8K | 0.91% | 1.21% | **−0.30%** |

**Verdict: Conclusion C — no measurable transfer.**

## Arithmetic forgetting — with the correction

| Regime | Task 20 baseline | 21A | 21B Model A |
| :--- | ---: | ---: | ---: |
| 2D | 100.00% | 3.60% | 4.40% |
| 3D | 99.80% | 0.60% | 5.20% |
| 4D | 67.20% | 0.40% | 0.60% |

A model that was at **100% / 99.8% / 67.2%** on subtraction is at **~4% / ~5% / ~0.6%** after 650 steps of GSM8K fine-tuning. This is textbook catastrophic forgetting.

**The correction, restated**: Task 21A's report says PPL "remains stable at 28.87 … confirming negligible catastrophic forgetting." Both halves are wrong — 28.87 vs 17.68 is +63.3%, and the arithmetic collapse it ignores is total. The sentence looked only at language and drew a conclusion about forgetting in general.

## PPL trajectory across transfer

Base 17.68 → Task 20 arithmetic 27.63 → 21A 28.87 → **21B 43.88**. The compact-trace run, which fixed formatting, cost the *most* language quality of the three. Fine-tuning is not free in any direction.

---

# PART 20 — TASK 22 / FREEZING FORENSICS

## The reasoning that led here

Task 21B's diagnosis: the arithmetic substrate never covered GSM8K's distribution. Measured from the compact GSM8K traces:

| | GSM8K traces | Task 20 substrate |
| :--- | :--- | :--- |
| `*` | **37.40%** | 0% (never trained) |
| `+` | **28.97%** | 0% |
| `/` | **17.13%** | 0% |
| `-` | 16.50% | **100%** |
| Operands 1–9 | **40.85%** | 0% (ranges start at 50) |
| Operands 10–99 | 39.60% | partial `[50,99]` |

So the substrate covered ~16.5% of GSM8K's operator mass, over ranges excluding 40.85% of its operands. Task 22 built an **operator-matched replay corpus** (8,000 traces, 25,762 equations, TVD **0.0063** on operators, **0.0212** on operand magnitude, **0 verification failures**) and — to protect the substrate — froze `tok_emb`, `pos_emb`, and blocks 0–5.

## It failed

| | Model E (replay + frozen) | Model A (21B baseline) | Δ |
| :--- | ---: | ---: | ---: |
| Equation validity (sampled) | **6.45%** | 20.45% | **−14.00** |
| Equation validity (greedy) | 12.19% | 23.67% | −11.49 |

Isolation held perfectly (max |Δ| = 0.000e+00 across all 62 frozen tensors), and the result survived a denominator check, a padding control, and a harness-fidelity check (the harness reproduced seven Task 21B published numbers exactly). The result was real.

## Why tied embeddings caused it

`tie_word_embeddings=True` ⇒ `lm_head.weight` **is** `tok_emb.weight`. Freezing the token embedding therefore froze the **output projection**. The model kept only `ln_f` — 1,024 parameters — as a degree of freedom before a fixed head.

Consequences, measured:
- **Could not learn to stop.** p90 generation length **256 = the cap**; ≥10% of generations never terminated.
- **Degenerated into repeating available tokens**: `16 + 12 + 12 + 12 + …`, 5.12 equations/generation vs GSM8K's true mean of 3.27.
- Full-format compliance **72.8%** vs Model A's 97.8%; greedy repetition **99.9%**.

## Task 22B — the isolating control

Same corpus, same data schedule, same seed/steps/LR/batch. **Only** the trainable parameter set changed: 18,887,680 → all 42,132,480.

| | E (frozen) | F (trainable) | Δ |
| :--- | ---: | ---: | ---: |
| Equation validity (sampled) | 6.45% | **31.77%** | **+25.32** |
| Equation validity (greedy) | 12.19% | **42.56%** | **+30.38** |
| Equations/generation | 5.12 | 2.74 | — |
| p90 length | **256** | **59** | — |
| Full format | 72.8% | 98.6% | — |

**Freezing was the dominant cause of the degeneration.** And note this **retracted Task 22's own §9 hypothesis** that the corpus design caused the runaway chains.

## But the fix did not fix retention

Model F retention: **3.8% / 1.8% / 0.0%** — no better than the unprotected baselines, and PPL got *worse* (60.02, the worst in the project).

**So**: freezing explains the *generation* failure. It explains nothing about *forgetting*. Two separate problems, and Task 22B cleanly separated them.

## The secondary discovery — the harness padding artifact

The evaluation harness left-pads prompts and **never passes the attention mask it constructs**. Combined with **learned absolute position embeddings**, the same question begins at a different position index depending on batch composition — while training always right-padded, so real content always started at index 0.

Quantified with a batch-size-1 control (zero padding), N=200: removing padding changed equation validity by **+3.00** points for Model E and **+13.78** for Model A.

Two honest implications:
1. **Comparisons remain valid** — every arm was measured identically.
2. **Absolute numbers are conservative** — all reported benchmark figures are depressed by this artifact.
3. The control also **strengthened** the Task 22 negative result: under the clean condition the gap *widened* from −11.49 to −22.16. The artifact was flattering Model E, not hiding a win.

---

# PART 21 — TASK 23 REPLAY RATIO

## Design

Two new arms at 10% and 25% replay, all parameters trainable, everything else identical to Task 22B. Replay block split 50/50 between the GSM8K-matched replay corpus and the **Task 20 progressive-subtraction corpus** — the actual regime the retention probe measures.

Deterministic global-index schedule, verified exact before training:

| Arm | Rule | GSM8K | replay_arithmetic | progressive_subtraction |
| :--- | :--- | ---: | ---: | ---: |
| 90:10 | `idx % 10 == 0` | 90.000% | 5.000% | 5.000% |
| 75:25 | `idx % 4 == 0` | 75.000% | 12.506% | 12.494% |

## Results

| Replay | 2D | 3D | 4D | Eq. validity (greedy) | GSM8K | PPL | RCS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% (A) | 4.40% | 5.20% | 0.60% | 23.67% | 0.91% | **43.88** | 0.1661 |
| **10% (G)** | 65.60% | 54.80% | 13.40% | 31.25% | 2.12% | 47.90 | 0.4110 |
| **25% (H)** | **94.80%** | **78.20%** | **32.40%** | 31.64% | **2.43%** | 50.74 | **0.4569** |
| 50% (F) | 3.80% | 1.80% | 0.00% | **42.56%** | 2.27% | 60.02 | 0.1520 |
| *50% frozen (E) — separate control* | *5.60%* | *4.20%* | *0.00%* | *12.19%* | *1.52%* | *30.33* | *0.1515* |

Model E is a **freezing control**, not a point on the mixture curve. Keep it visually and rhetorically separate.

## Why this is NOT a pure ratio curve — state this before you are asked

The four trainable arms differ in **replay composition** as well as share:

- **A (0%)** — no replay at all.
- **G (10%) / H (25%)** — replay is 50/50 GSM8K-matched + **Task 20 subtraction** (the probed regime).
- **F (50%)** — replay is **100% GSM8K-matched**, containing essentially **none** of the 2D/3D/4D subtraction regime being measured.

So F's collapse at the *highest* replay share is a **composition** effect, not a dose effect. **Ratio and composition are confounded in this series.** The apparent non-monotonicity (retention peaking at 25% and collapsing at 50%) is therefore not evidence of an optimal dose.

**Defensible claim**: replaying the target distribution at 10–25% preserves it.
**Indefensible claim**: 25% is the optimal replay ratio.
**The missing run**: 50% replay with the *mixed* block. One run; not done (deadline + protocol).

## Amount vs composition — the actual lesson

| Arm | Replay amount | Contains probed regime? | 2D retention |
| :--- | ---: | :---: | ---: |
| F | **50%** (most) | ✗ | **3.80%** |
| G | 10% (least) | ✓ | 65.60% |
| H | 25% | ✓ | **94.80%** |

The arm with the **most** replay retained the **least**. Replay is not a generic regulariser — it is distributional. **You must replay what you want to keep.**

This is the project's cleanest transferable insight and generalises well beyond Mini-R1: in continual learning, a replay buffer sampled from the wrong part of the input space provides no protection for the capability you care about, no matter how large it is.

## RCS — the decision scalar

Pre-registered before the runs, to prevent post-hoc metric shopping.

**Gates (hard, must all pass):**
- **G1 retention**: 2D ≥ 20% and 3D ≥ 15% — blocks "GSM8K rose, arithmetic died"
- **G2 language**: PPL ≤ 35.0 — blocks "retention rose, PPL exploded"
- **G3 generation health**: EOS ≥ 95%, full-format ≥ 95%, p90 < 128, greedy repetition < 90% — blocks Task 22-style degeneracy

**Then rank survivors by:**
$$S_{\text{arith}} = \frac{\text{mean}(\text{2D},\text{3D})}{100},\quad S_{\text{eq}} = \frac{\text{greedy eq. validity}}{100},\quad S_{\text{lang}} = \min\!\left(1, \frac{17.68}{\text{PPL}}\right)$$
$$\boxed{\ \text{RCS} = \left(S_{\text{arith}}\cdot S_{\text{eq}}\cdot S_{\text{lang}}\right)^{1/3}\ }$$

**Why a geometric mean and not an arithmetic mean**: the geometric mean is **zero if any factor is zero** and is dominated by the smallest term. An arithmetic mean would let a model with superb equation validity and destroyed arithmetic score well by averaging. The geometric mean structurally enforces "you must be non-terrible at all three" — exactly the requirement "don't claim success if X rises but Y collapses."

$S_{\text{lang}}$ uses base PPL 17.68 as the numerator so the term is a *ratio to the best achievable*, capped at 1.

**Why GSM8K accuracy is excluded from RCS**: every model in the project sits between 0.91% and 2.43%. That spread is a handful of problems out of 1,319 and is inside noise. Including a noise-dominated term would let random variation drive model selection. It is reported, and it carries a **floor** (≥1.5%), but it does not rank.

**Sanity check that the metric discriminates**: applied to the pre-existing arms it gives A 0.1661, E 0.1515, F 0.1520 — all clustered, correctly saying "none of these is good." The new arms score 0.4110 and 0.4569, ~2.7× higher.

## Outcome and selection

**No arm passed all gates.** G2 (PPL ≤ 35) was failed by **every trainable arm including the 0% baseline at 43.88**. In hindsight the gate was calibrated against Model E's 30.33 — but E was the degenerate frozen model that never produced usable output. The gate was set below the achievable range for any full-parameter SFT from this checkpoint. **That is a design flaw in my own pre-registration, and I should say so.**

Per the pre-registered fallback ("if none passes, select the best trainable model by RCS"), the final model is **Model H (75:25)**, RCS 0.4569.

---

# PART 22 — THE FINAL MODEL

**Checkpoint**: `artifacts/sft/task23_75_25/best.pt`
**SHA256**: `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c`
**Tokenizer**: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`
Integrity: **8/8 checks passed** [`task24_final_integrity.md`]

| Dimension | Value |
| :--- | ---: |
| Parameters | 42,132,480 |
| Architecture | 12L × 512d × 8h, d_ff 2048, ctx 512, tied embeddings |
| Training | 650 steps, eff. batch 32, LR 1e-4 cosine, seed 42, all params trainable |
| Mixture | 75% GSM8K / 12.506% replay-arith / 12.494% progressive-subtraction |
| Active completion tokens | 920,031 |
| **GSM8K strict (greedy)** | **2.43%** |
| GSM8K strict (sampled) | 1.21% |
| **Equation validity (greedy)** | **31.64%** |
| SVAMP | 1.67% |
| AQuA | **0.00%** (0.0% full-format) |
| E1 IID 2D / E3 IID 3D / E5 IID 4D | **89.60% / 84.60% / 38.40%** |
| E2 OOD 2D / E4 OOD 3D / E6 OOD 4D | 20.00% / 16.60% / 4.60% |
| E7 acid 5D | **0.60%** |
| Retention probe 2D/3D/4D | **94.80% / 78.20% / 32.40%** |
| Format compliance (GSM8K greedy) | 98.2% |
| EOS rate | 98.2% |
| Repetition (greedy) | 77.3% |
| p90 length | 61 |
| **General PPL** | **50.74** |

## State this plainly

**This is not a state-of-the-art reasoning model.** GSM8K 2.43% versus Direct SFT's 2.35% is a one-problem difference on 1,319 items — statistically meaningless.

**It is a research SLM demonstrating four things:**
1. **Strong learned computational behaviour** within trained numeric regimes (89.60% IID 2-digit; 100% at the Task 20 stage).
2. **Limited generalization** — 20.00% one step outside the trained range, 0.60% at five digits.
3. **Severe task interference** — GSM8K fine-tuning destroys arithmetic; parameter freezing destroys generation; every fine-tune costs language quality.
4. **Measurable retention improvement through targeted replay** — 4.40% → 94.80% 2-digit retention, the project's actual contribution.

## Two numbers worth internalising for the retention story

Both are reported and neither is preferred:
- Generated retention probe (seed 42): **94.80 / 78.20 / 32.40**
- Fixed E1/E3/E5 files: **89.60 / 84.60 / 38.40**

Same three regimes, different item samples. Quoting a range ("~85–95% on 2-digit") is more honest than quoting the higher number.

# PART 23 — MATHEMATICS STUDY GUIDE

Organised by priority. **LEVEL 1** = cannot pass the interview without it. **LEVEL 2** = will likely come up. **LEVEL 3** = depth signals.

## LEVEL 1 — Must know

| Topic | Meaning | Where in Mini-R1 | Equation | Likely question |
| :--- | :--- | :--- | :--- | :--- |
| **Matrix multiplication & shapes** | Composing linear maps; inner dims must match | Every projection: (B,T,512)×(512,512) | $C_{ij}=\sum_k A_{ik}B_{kj}$ | "Walk me through the tensor shapes from tokens to logits." |
| **Dot product as similarity** | $q\cdot k$ large when aligned | Attention scores $QK^\top$ | $q\cdot k=\lVert q\rVert\lVert k\rVert\cos\theta$ | "Why does attention use a dot product?" |
| **Softmax** | Reals → probability simplex | Attention weights; output distribution | $p_i=e^{z_i}/\sum_j e^{z_j}$ | "Why subtract the max before exponentiating?" |
| **Cross-entropy** | Negative log-likelihood of the true class | The entire training objective | $\mathcal L=-\sum_t\log p_\theta(y_t\mid y_{<t})$ | "Derive $\partial\mathcal L/\partial z_i$." → $p_i-\mathbb 1[i=c]$ |
| **Perplexity** | $\exp(\text{loss})$; effective branching factor | 17.68 base → 50.74 final | $\text{PPL}=e^{\mathcal L}$ | "What does PPL 17.68 mean? What does it not tell you?" |
| **Chain rule / backprop** | Gradients via composed Jacobians | Autograd | $\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}$ | "Why do residual connections help gradients?" |
| **Gradient descent** | Step downhill | Every update | $\theta\leftarrow\theta-\eta\nabla_\theta L$ | "What happens if $\eta$ is too large?" |
| **Variance & the $\sqrt{d_k}$ scaling** | $\operatorname{Var}(q\cdot k)=d_k$ | Attention denominator | see Part 4.4 | "Why $\sqrt{d_k}$ and not $d_k$?" |
| **Autoregressive factorization** | Joint = product of conditionals | The whole model | $p(y_{1:T})=\prod_t p(y_t\mid y_{<t})$ | "Why is causal masking required?" |

## LEVEL 2 — Should know

| Topic | Meaning | Mini-R1 link | Equation | Likely question |
| :--- | :--- | :--- | :--- | :--- |
| **Adam / AdamW** | Adaptive per-parameter steps; decoupled decay | Optimizer everywhere | Part 8.3 | "Why AdamW over Adam? What does decoupling fix?" |
| **Momentum** | EMA of gradients damps oscillation | $\beta_1=0.9$ | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ | "What does $\beta_2=0.95$ vs 0.999 change?" |
| **Weight decay** | Shrink weights toward 0 | $\lambda=0.01$ | $\theta\leftarrow\theta-\eta\lambda\theta$ | "Is weight decay the same as L2? When not?" |
| **Gradient clipping** | Bound the update norm | max_norm 1.0 | $g\leftarrow g\min(1,c/\lVert g\rVert)$ | "Clip before or after unscaling? Why?" |
| **LayerNorm** | Per-token feature normalisation | Pre-LN blocks | Part 4.6 | "Why LayerNorm not BatchNorm for language?" |
| **Residual learning** | Learn a delta; identity gradient path | All 12 blocks | $x+F(\text{LN}(x))$ | "What breaks at 12 layers without residuals?" |
| **Entropy** | Expected surprise | Decoding temperature | $H=-\sum p_i\log p_i$ | "How does temperature change entropy?" |
| **KL divergence** | Distance between distributions | Not used here — appears in RL/PPO | $D_{KL}(P\Vert Q)=\sum p\log\frac{p}{q}$ | "Where would KL appear if you'd run RL?" |
| **Initialization** | Preserve activation scale across depth | From-scratch model | Xavier/He variance rules | "Why does init scale matter at depth?" |
| **Multi-head attention** | Parallel subspace attention | 8×64 | Part 4.4 | "Does more heads mean more parameters?" (No) |
| **Binomial CI / proportion tests** | Uncertainty on accuracy | 2.43% on n=1319 | $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$ | "Is 2.43% vs 2.35% significant?" |

## LEVEL 3 — Deep / research signal

| Topic | Meaning | Mini-R1 link | Likely question |
| :--- | :--- | :--- | :--- |
| **Softmax Jacobian & saturation** | $\operatorname{diag}(p)-pp^\top\to0$ at one-hot | Why $\sqrt{d_k}$ matters for *gradients* | "Show me why unscaled scores kill gradients." |
| **Permutation equivariance of attention** | $\text{Attn}(\Pi X)=\Pi\text{Attn}(X)$ | Why positions are needed at all | "Prove attention needs positional information." |
| **Exposure bias** | Train on gold prefix, test on own prefix | Explains CoT failure and compounding error | "Why did CoT lose to direct answers?" |
| **Catastrophic forgetting** | New task overwrites old capability | 100% → 4% retention | "What causes it? Name three mitigations." |
| **Experience replay** | Interleave old-task data | Tasks 22/22B/23 — the core contribution | "Why did 50% replay fail while 25% worked?" |
| **Interpolation vs extrapolation** | Inside vs outside the training hull | E1 100% vs E7 0% | "Which did your model do?" |
| **Total variation distance** | $\frac12\sum|p-q|$ | Corpus matching, TVD 0.0063 | "How did you verify your synthetic corpus matched?" |
| **Geometric vs arithmetic mean** | GM punishes any near-zero factor | RCS design | "Why a geometric mean for model selection?" |
| **Scaling laws** | Loss vs params/data/compute power laws | Why 42M is small; Chinchilla ~20 tok/param | "Was your model under- or over-trained?" |
| **Grokking / algorithmic generalization** | Delayed emergence of true algorithms | Why 5-digit stayed at 0% | "Would longer training have gotten 5-digit?" |

## RL topics that emerged but were NEVER implemented

**Be explicit: no RL was run in this project.** GRPO was listed as a research goal in `PROJECT_CONTEXT.md` §1.2.4 and never executed. Know the concepts as *future work* only.

| Topic | Meaning | Equation | Why relevant to Mini-R1 |
| :--- | :--- | :--- | :--- |
| **Policy gradient (REINFORCE)** | Push up log-prob of high-reward samples | $\nabla J=\mathbb E[\nabla\log\pi_\theta(a)\,R]$ | The natural next step: our verifier is a perfect reward function |
| **Reward** | Scalar quality signal | $R\in\{0,1\}$ from equation verification | Deterministic arithmetic checking gives noise-free reward — rare and valuable |
| **Advantage** | Reward minus baseline; reduces variance | $A=R-b$ | Without a baseline, gradient variance is huge at 2% success rates |
| **PPO** | Clipped surrogate keeps updates near the old policy | $\min(r_tA_t,\ \text{clip}(r_t,1\pm\epsilon)A_t)$ | Standard, but needs a value network — expensive at 42M |
| **GRPO** | Group-relative advantage; no value network | $A_i=(R_i-\text{mean}(R))/\text{std}(R)$ over $k$ samples per prompt | The stated project goal; cheaper than PPO, well suited to verifiable rewards |
| **KL regularization** | Penalty against drifting from the reference policy | $J-\beta D_{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ | Would directly address our biggest problem — PPL regression during fine-tuning |
| **Exploration/exploitation** | Sample diversity vs greedy exploitation | temperature, top-p | At 2% accuracy almost every sample is negative reward; exploration is the binding issue |

**The honest framing**: "RL is the obvious next step because I already have a deterministic verifier, which is the hard part of building a reward. I didn't run it — the project ran out of scope, and at 2% base success rate the reward signal would be extremely sparse, which is a real design problem I'd have to solve first."

---

# PART 24 — ENGINEERING STUDY GUIDE

| Topic | Concept | Why in Mini-R1 | Likely questions |
| :--- | :--- | :--- | :--- |
| **PyTorch autograd** | Dynamic graph built on the forward pass; `.backward()` walks it in reverse | All training | "What does `.grad` accumulate and why doesn't it reset?" / "What does `torch.no_grad()` change?" |
| **CUDA memory model** | Allocated vs reserved; caching allocator | Peak 3,409.7 MB on 6 GB | "Why is reserved > allocated?" / "OOM with free memory shown — why?" (fragmentation) |
| **AMP autocast** | Op-level precision selection | FP16 forward/backward | "Which ops stay FP32 under autocast?" (reductions, softmax, losses) |
| **GradScaler** | Loss scaling to avoid FP16 underflow | 0 skipped steps | "What happens on an inf gradient?" (skip step, halve scale) |
| **Gradient accumulation** | Sum micro-gradients before stepping | 8×4 = 32 | "Prove it equals batch 32." / "Where does it *not*?" (BatchNorm, fp non-associativity) |
| **Checkpoint serialization** | `torch.save` pickles tensors + metadata | 509 MB with optimizer state | "Why is your checkpoint 3× the weights?" (Adam m,v + FP32 master) |
| **Atomic writes** | `save(.tmp)` → `os.replace()` | Every best-checkpoint write | "Why not write in place?" (crash ⇒ corruption) |
| **Memory mapping** | `np.memmap`; OS page cache serves pages | 500M-token shards | "Why memmap instead of loading to RAM?" |
| **Data sharding** | Corpus split into uint16 files | vocab 8000 < 65536 ⇒ 2 B/token | "Why uint16?" / "What if vocab were 100k?" (uint32) |
| **Streaming datasets** | Pull-as-needed from HF | 500M tokens never materialised | "What breaks if the stream stalls mid-run?" |
| **Dataset cursors** | shard idx + seq idx + tokens_seen | Resume correctness | "Why must the cursor be in the checkpoint?" |
| **Random seeds** | `torch`, `cuda`, `numpy`, `random` all seeded | seed 42 everywhere | "Seeding torch only — what still varies?" |
| **Deterministic training** | `use_deterministic_algorithms`, fixed shapes | Bitwise parity target | "Why can identical seeds still diverge?" (non-deterministic CUDA reductions) |
| **JSONL** | One JSON object per line | All datasets | "Why JSONL over one big JSON?" (streamable, appendable, line-addressable) |
| **YAML configs** | Declarative experiment specification | Task 22/22B/23 configs | "Why config files over CLI flags?" (versioned, diffable, reproducible) |
| **SHA256 hashing** | Content fingerprint | Every checkpoint/corpus | "Why hash datasets?" (proves which data produced which model) |
| **HTTP server / REST** | Request → inference → JSON | `web/server.py` (stdlib) | "How would you productionise this?" — **note FastAPI was planned, not built** |
| **Latency vs throughput** | Per-request time vs aggregate rate | 34.12 ms/token; 17,316 tok/s train | "How do you improve each?" (KV cache / batching) |
| **Batching at inference** | Amortise weight reads | batch 64 in the evaluator | "What's the risk?" — **our padding artifact** |
| **KV caching** | Reuse past K,V; O(T) not O(T²) per step | **Not implemented** — why greedy eval took 92 min | "How would you speed up generation 10×?" |

**Two engineering war stories worth telling:**

1. **The missing attention mask.** The evaluator *builds* `attn_mask` and never passes it to the model. With absolute position embeddings this shifts every token's representation. I found it by reading the harness, then **measured** it with a batch-size-1 control rather than guessing: +3.00 pts for one model, +13.78 for another. It didn't invalidate comparisons (all arms measured identically) but it does mean my absolute numbers are conservative.

2. **The silently broken resume.** My evaluator advertised `--partial_json` resume. During Task 22B it re-evaluated a cached model anyway — a string-replacement patch had no-op'd on a newline mismatch, so the load block was never inserted and only the *save* half existed. Caught it because a model that should have been skipped started generating. Cost: one killed job. Lesson: **assert the behaviour, don't assume the patch applied.**

---

# PART 25 — RESEARCH METHODOLOGY

| Concept | Definition | Mini-R1 example |
| :--- | :--- | :--- |
| **Hypothesis** | Falsifiable prediction stated *before* the run | "Arithmetic-init will beat base-init on GSM8K." Falsified: Δ = −0.30%. |
| **Independent variable** | The one thing you change | Task 22B: trainable parameter set (18.9M vs 42.1M) — *nothing else* |
| **Dependent variable** | What you measure | Step-level equation validity |
| **Control variable** | Held fixed to license causal claims | Task 22B held corpus, split, seed, steps, LR, batch, schedule, evaluator identical |
| **Confound** | An uncontrolled variable that could explain the result | Task 22 changed replay **and** freezing together — irreducibly confounded until 22B |
| **Ablation** | Remove one component, measure the delta | Task 16B: five decoding configs on a **frozen** checkpoint |
| **Baseline** | The comparison that makes a number meaningful | Direct SFT 2.73%; without it, CoT's 1.21% has no interpretation |
| **IID evaluation** | Held-out sample, same distribution | E1/E3/E5 |
| **OOD evaluation** | Deliberately outside the training support | E2/E4/E6; E7 acid test |
| **Statistical significance** | Is the difference bigger than noise? | 15C reported p = 0.0051 for Direct vs CoT |
| **McNemar's test** | Paired binary test for two models on the *same* items | Correct choice for paired GSM8K comparisons: $\chi^2=\frac{(b-c)^2}{b+c}$, where $b$ = A-right/B-wrong, $c$ = A-wrong/B-right. Only the disagreements carry information. |
| **Confidence interval** | Plausible range for the true value | At 2.43%, n=1319: ±~0.83% at 95%. This is why 2.43 vs 2.35 is meaningless. |
| **Data leakage** | Eval information present in training | 19B: 35.46% overlap turned 2.48% into an apparent 24.28% |
| **Reproducibility** | Same inputs ⇒ same outputs | Seeds + RNG state + dataset cursor + manifest hashes |
| **Negative result** | A well-designed experiment that disconfirms | 21B (−0.30%), 22 (−14.00) — both reported in full |
| **Causal claim** | "X caused Y" — requires a controlled manipulation | Legitimate in 22B (only freezing varied); illegitimate for the 0/10/25/50 curve (composition co-varied) |
| **Pre-registration** | Fix metrics/thresholds before seeing results | RCS + gates G1–G3 defined before Task 23 ran |

## Why we kept finding confounds — and why that is the point

The project discovered and corrected **four** major methodological problems, each found by us, each changing the conclusion:

1. **Leakage (19B)** — apparent 24.28% was really 2.48%. Fixed by clean disjoint splits.
2. **Confounded manipulation (22)** — replay + freezing moved together. Fixed by the 22B control, which reversed the corpus-blame conclusion.
3. **Harness padding artifact (22)** — measured with a bs=1 control rather than assumed; showed the artifact *flattered* the losing arm.
4. **Denominator sensitivity (22B)** — equation validity is a **rate**; a model emitting almost no equations can post a high rate. Fixed by adding the absolute counterpart (valid equations per generation) alongside it.

**The meta-lesson to articulate**: none of these were caught by looking at the headline number. Each was caught by asking "what else could produce this number?" and then *building a measurement to rule it out*. The Task 23 pre-registration (gates + RCS fixed in advance) exists specifically because the earlier tasks showed how easy post-hoc metric selection would have been.

**Also be ready to criticise your own pre-registration**: gate G2 (PPL ≤ 35) was set below what any full-parameter fine-tune of this checkpoint could achieve — even the 0%-replay baseline failed it at 43.88. A better gate would have been *relative to the strongest baseline* rather than an absolute number anchored on a degenerate model.

---

# PART 26 — INTERVIEW QUESTION BANK

The full 150-question bank lives in its own file for rapid revision:

**`reports/Mini-R1_INTERVIEW_QA.md`**

| Section | Count | Focus |
| :--- | ---: | :--- |
| A — Basic | 30 | vocabulary, architecture, training fundamentals |
| B — Intermediate | 30 | derivations, optimization, methodology |
| C — Advanced | 30 | proofs, mechanism, the project's hard findings |
| D — Research defense | 20 | defending claims, admitting limits |
| E — Debugging | 20 | diagnostic reasoning from symptoms |
| F — "Why didn't you do X?" | 20 | design-decision justification |
| **Total** | **150** | |

Each entry carries: question · what is being tested · short answer · detailed answer · relevant equation · the Mini-R1 experiment it maps to · the common mistake · a follow-up question.

Parts 27 (hard questions) and 33 (adversarial) below are the highest-value subsets and are duplicated here so this guide stands alone.

---


# PART 27 — THE HARD QUESTIONS ("Questions I WILL get asked")

Short, confident, honest answers. Full Q&A bank in `reports/Mini-R1_INTERVIEW_QA.md`.

**Why 42M parameters?** It's the largest model that trains at batch 32 × seq 512 within 6 GB of VRAM. Peak allocation was 3,409.7 MB, leaving headroom. Size was chosen by the hardware, then the research question was chosen to be answerable at that size.

**Why not 7B?** I have one 6 GB laptop GPU. But more importantly, at 7B arithmetic is already latent from pretraining, so "does arithmetic supervision transfer" isn't cleanly answerable — you can't separate what you taught from what was already there. 42M makes the question crisp.

**Why 500M tokens?** It fit an 8-hour unattended run. Chinchilla-optimal for 42M would be ~840M (20 tokens/param), so we were arguably under-trained — yet validation loss plateaued at ~295M, which suggests capacity or corpus diversity bound us before token count did. That's a hypothesis; I didn't ablate it.

**Why this dataset?** The target is mathematical reasoning, so I biased toward explanatory educational text over general web crawl. Cosmopedia-v2 is synthetic and clean, fineweb-edu-dedup is real and varied.

**Why 60:40?** A reasoned lean toward the cleaner source while keeping genuine text diversity. **I never ablated it — I can't claim it's optimal.**

**Why SentencePiece?** Language-agnostic, no pre-tokenization assumptions, trains directly on raw text, and lets me force `<think>`/`<answer>` to be atomic single tokens — which is what makes format compliance cleanly measurable and single-token early-stopping possible.

**Why vocab 8000?** The embedding is $V\times512$. At 8k that's 4.1M params (9.7% of the model); at 32k it would be 16.4M (39%) — the vocabulary would dominate the model. 8k keeps capacity in the layers where computation happens.

**Why context 512?** Attention memory is $O(B\cdot h\cdot T^2)$ per layer — at B=8, h=8, T=512 that's 33.6 MB in FP16 per layer, ~400 MB across 12 layers. Doubling T quadruples it and I'd OOM. And our sequences are short: compact traces average ~47 completion tokens.

**Why FP16 and not BF16?** BF16 is strictly nicer — FP32 exponent range, no GradScaler needed. I used FP16 because reliable BF16 support wasn't established on this consumer Ada laptop stack, and FP16+GradScaler was empirically verified: 0 skipped steps, 0.1085% error vs FP32. Engineering caution, not a claim BF16 is worse.

**Why AdamW?** Language models have heavy-tailed token frequencies, so per-parameter adaptive steps matter — rare embeddings still learn. The "W" decouples weight decay from the adaptive denominator; with L2-in-gradient, high-gradient parameters get *less* regularisation, which is backwards.

**Why 6e-4?** Empirical sweep over 1e-4/3e-4/6e-4 at 1,000 steps: 6e-4 gave val loss 2.4995 (PPL 12.18) vs 3e-4's 2.6337 (13.93). Caveat: that's 3.3% into training, and I never re-ran the full budget at 3e-4.

**Why cosine decay?** Smooth, no discontinuities, decays to a small final LR so the model settles rather than bounces. Warmup first because Adam's second-moment estimate is unreliable at step 0.

**Why gradient accumulation?** Activations for 8 sequences fit in 6 GB; for 32 they don't. Accumulating 4 micro-batches gives the gradient of batch 32 while only ever holding 8 sequences of activations.

**Why completion-only masking?** I want $p(\text{answer}\mid\text{question})$, not $p(\text{question})$. Supervising the prompt spends 42M parameters' worth of capacity learning to regenerate GSM8K questions.

**Why did CoT fail?** Two mechanisms. Exposure bias — teacher forcing conditions each step on gold prefixes, but generation conditions on the model's own possibly-wrong step, so an $n$-step chain succeeds roughly as $q^n$. And degeneration — 82.6% repetition. The signature is that the deficit grows with depth: −1.11% at 3 steps, −1.40% at 4+.

**Why was repetition so high?** Greedy decoding maximises local probability, and a high-probability token can re-enter a high-probability state — a fixed point in the decoding dynamics. Small models have flatter, less-confident distributions, making these loops easier to enter.

**Why did decoding help?** Temperature + top-p break determinism so the loop isn't re-entered; repetition penalty explicitly down-weights already-emitted tokens. Repetition 82.6% → 11.45%, format 39.8% → 90.67%. **But accuracy only went 1.21% → 2.20%, still below Direct's 2.73% — so it fixed the text, not the reasoning.**

**Why did arithmetic SFT work?** Deterministic labels, zero noise, exact difficulty control, and the metric is the same function as the label generator. 100% on held-out 2-digit subtraction.

**Why did OOD collapse?** Empirically the model learned a regime-local competence rather than a digit-count-agnostic algorithm — 100% at 2-digit, 0.00% at 5-digit with identical tokenization. **I did not localise the mechanism** and won't claim one without interpretability work.

**Why did the leakage happen?** The eval generator and training generator drew from overlapping numeric ranges without an explicit disjointness constraint, and my first overlap check only looked at top-level prompts, not numbers embedded inside multi-step `<think>` traces.

**How did you detect it?** The 24.28% looked too good given everything else was near-random. I keyed every eval pair and searched the *entire* training corpus including nested trace content: 111/313 overlap, 82.14% on seen vs 2.48% on unseen. The A1-vs-A2 split (82.14% vs 7.41% for the same numbers in different formats) confirmed it was prompt retrieval, not numeric knowledge.

**Why did 3D exposure improve 3D OOD?** Adding 3-digit training took unseen 3-digit from 6.60% to 34.40%. It extended competence into that regime — but it did not propagate: untrained 4-digit stayed at 0.40%, and unseen 2-digit actually *fell* 35.40% → 28.40%.

**Why did 5D fail?** Zero examples with 5-digit operands, and whatever was learned isn't digit-count invariant. A true column-wise subtraction algorithm would be length-agnostic; 0.00% says it isn't one.

**Why did arithmetic knowledge fail to transfer to GSM8K?** Two candidates, and I can separate them. The substrate was distributionally wrong — subtraction only, over ranges excluding 40.85% of GSM8K's operands, while GSM8K is 37.4% multiplication. But when I fixed coverage in Task 22 (TVD 0.0063), transfer **still** didn't appear. So coverage was necessary-at-best. The residual failure is semantic grounding: mapping words to the right operations, which arithmetic supervision cannot teach.

**Why did replay help?** It keeps the old task's gradient signal present, so the optimizer can't move freely in directions that destroy it. 25% targeted replay took 2-digit retention from 4.40% to 94.80%.

**Why did 50% replay fail?** Because it was the **wrong** replay. Model F's 50% replay contained essentially none of the subtraction regime the probe measures. The arm with the *most* replay retained the *least* (3.80%). Composition beats dose.

**Why did freezing fail?** Tied embeddings. `lm_head.weight` *is* `tok_emb.weight`, so freezing the token embedding froze the output head. The model couldn't learn to emit its stop token — p90 generation length pinned at the 256 cap — and degenerated into loops. Unfreezing recovered +30.38 points of equation validity.

**Why does tied embedding matter?** It saves 4.1M parameters (9.7% of the model), which is a good trade at this scale. But it creates a hidden coupling: any intervention on the input embedding silently applies to the output head. And PyTorch's `named_parameters()` de-duplicates shared tensors, so a naive freeze audit won't even show it. I verified by object identity.

**Why not RL?** It's the obvious next step — I already have a deterministic verifier, which is normally the hard part of reward design. I didn't run it: scope, and at ~2% success rate the reward is extremely sparse, so I'd need to solve exploration first. GRPO was the stated plan.

**Why not GRPO specifically?** No value network needed — advantage comes from group-relative normalisation across $k$ samples of the same prompt — which suits a 42M model where a critic is expensive.

**Why not increase model size?** 6 GB. And the research question is sharper at a size where arithmetic must be taught rather than being already latent.

**What would you change with another month?** In order: (1) deconfound ratio from composition — rerun 50% replay with the mixed block, one run; (2) add a language-replay stream from the pretraining shards, since every arm blew the PPL gate including the baseline; (3) fix the harness attention mask and re-measure everything; (4) attack semantic grounding, which is the actual GSM8K bottleneck; (5) GRPO on the verifier.

**What is genuinely novel here?** Not the architecture — that's a standard decoder. The contributions are methodological: a fully verified arithmetic evaluation stack where every claim is Python-checkable; the step-level equation-validity metric that separates "wrong setup" from "wrong computation"; the demonstration that tied embeddings make parameter freezing actively harmful; and the composition-over-dose result for replay.

**What are the biggest limitations?** GSM8K 2.43% is indistinguishable from a trivial baseline. Single seed. The replay curve is confounded. PPL regressed 2.9×. Absolute numbers are depressed by the harness artifact. And I never ran RL.

---

# PART 28 — WHITEBOARD EXERCISES

**1. Parameter count.**
*Problem*: Derive the total for $L{=}12, d{=}512, d_{ff}{=}2048, V{=}8000$, ctx 512, tied.
*Solution*: block $= 12d^2+4d = 3{,}147{,}776$; ×12 $= 37{,}773{,}312$; $+Vd\,(4{,}096{,}000) + 512d\,(262{,}144) + 2d\,(1{,}024) = \mathbf{42{,}132{,}480}$.
*Common mistake*: adding the LM head separately — it's tied, contributing 0.

**2. Attention dimensions.**
*Problem*: Shapes through attention for B=8, T=512, d=512, h=8.
*Solution*: X (8,512,512) → Q,K,V (8,512,512) → reshape (8,8,512,64) → $QK^\top$ (8,8,512,512) → softmax over last dim → ×V (8,8,512,64) → concat (8,512,512) → $W_O$ (8,512,512).
*Common mistake*: softmax over the wrong axis. It must be over **keys** (last), so each query's weights sum to 1.

**3. Cross-entropy gradient.**
*Problem*: Derive $\partial\mathcal L/\partial z_i$ for $\mathcal L=-\log p_c$, $p=\text{softmax}(z)$.
*Solution*: $\mathcal L=-z_c+\log\sum_je^{z_j}$; $\frac{\partial\mathcal L}{\partial z_i}=-\mathbb 1[i=c]+\frac{e^{z_i}}{\sum_je^{z_j}}=p_i-\mathbb 1[i=c]$.
*Common mistake*: forgetting the log-sum-exp term differentiates to $p_i$.

**4. Perplexity.**
*Problem*: Loss 2.8726 → PPL? PPL 50.74 → loss?
*Solution*: $e^{2.8726}=17.68$; $\ln 50.74=3.927$.
*Common mistake*: using $\log_2$ — that gives bits-per-token, a different quantity.

**5. Effective batch.** micro 8 × accum 4 = **32 sequences** = 32×512 = **16,384 tokens/step**.
*Common mistake*: forgetting to divide the loss by `accum`, which inflates the gradient 4× and effectively quadruples the LR.

**6. Tokens per step / 7. Steps for a budget.**
$500{,}000{,}000 / 16{,}384 = 30{,}517.6 \to$ **30,518 steps** ✅ matches the log.

**8. AdamW update.** Write Part 8.3 from memory. *Common mistake*: putting $\lambda\theta$ inside the $\sqrt{\hat v}$ denominator — that's Adam+L2, precisely what AdamW fixes.

**9. Gradient accumulation equivalence.** Show $\sum_{k=1}^4\frac14\cdot\frac18\sum_{i\in\text{micro}_k}\nabla\ell_i=\frac1{32}\sum_{i=1}^{32}\nabla\ell_i$.
*Common mistake*: claiming exact bitwise equality — floating-point addition isn't associative.

**10. Replay mixture.**
*Problem*: 650 steps, eff. batch 32, `is_replay = idx % 4 == 0`, replay split 50/50 across two sources. Counts?
*Solution*: 650×32 = 20,800 examples; replay = 20,800/4 = **5,200**; each source **2,600**. GSM8K = 15,600.
*Common mistake*: applying the ratio per micro-batch — 25% of 8 is 2, fine, but 10% of 8 is 0.8, which is why we schedule on the **global index**, not per batch.

**11. Retention ratio.** Final 2D 94.80% against Task 20's 100.00% → **94.8% of original capability retained**; the 0%-replay arm retained 4.40%. *Mistake*: quoting the raw number without the pre-fine-tuning baseline.

**12. RCS.**
*Problem*: 2D 94.80, 3D 78.20, greedy validity 31.64, PPL 50.74.
*Solution*: $S_{\text{arith}}=(94.80+78.20)/2/100=0.8650$; $S_{\text{eq}}=0.3164$; $S_{\text{lang}}=17.68/50.74=0.3485$; product $=0.09537$; cube root $=\mathbf{0.4569}$ ✅
*Mistake*: arithmetic instead of geometric mean — it wouldn't punish a collapsed factor.

**13. Interpret an ablation table.** Given Task 23's four arms: note retention is **non-monotonic** in replay share (peaks at 25%, collapses at 50%), which should immediately prompt "what else differs between those arms?" → composition.

**14. Identify the confound.** Task 22 changed replay data **and** froze parameters simultaneously; the −14.00 point drop cannot be attributed to either. The fix is one run varying only trainability (Task 22B), which showed freezing owned +30.38 of it.

---

# PART 29 — STUDY RESOURCES

| Resource | Author/Org | Topic | What to extract | Priority |
| :--- | :--- | :--- | :--- | :---: |
| **Attention Is All You Need** (2017) | Vaswani et al. | Transformer | §3.2.1 scaled dot-product (the $\sqrt{d_k}$ argument), §3.2.2 multi-head, Fig. 1. Note Mini-R1 uses **pre-LN**, not the paper's post-LN. | **RED** |
| **The Illustrated Transformer** | Jay Alammar | Transformer intuition | The shape-by-shape walkthrough; use it to check you can narrate tokens→logits | **RED** |
| **Adam** (2014) / **Decoupled Weight Decay Regularization** (2017) | Kingma & Ba / Loshchilov & Hutter | AdamW | Algorithm 1 in Adam; §2 of AdamW for *why* decoupling matters | **RED** |
| **Deep Learning** ch. 6, 8 | Goodfellow, Bengio, Courville | Backprop, optimization | Ch. 6.5 backprop; ch. 8.3–8.5 optimizers | **RED** |
| **Language Models are Unsupervised Multitask Learners** (GPT-2) | Radford et al. | Decoder-only LM | Architecture + the tied-embedding/scaling choices Mini-R1 mirrors | **RED** |
| **Training Compute-Optimal LLMs** (Chinchilla) | Hoffmann et al. | Scaling laws | The ~20 tokens/param rule — needed to discuss whether 500M was enough | **YELLOW** |
| **SentencePiece** (2018) | Kudo & Richardson | Tokenization | Why language-agnostic subword; how user-defined symbols stay atomic | **YELLOW** |
| **Chain-of-Thought Prompting** (2022) | Wei et al. | CoT | §3 the emergence claim — then contrast with our 42M result where CoT *hurt* | **RED** |
| **Training Verifiers to Solve Math Word Problems** (GSM8K) | Cobbe et al. | GSM8K + verifiers | The dataset construction and the verifier idea that motivates our Python checker | **RED** |
| **Catastrophic Forgetting in Neural Networks** / **EWC** | McCloskey & Cohen; Kirkpatrick et al. | Forgetting | The phenomenon, and EWC as the *alternative* mitigation we didn't use | **RED** |
| **Experience Replay for Continual Learning** | Rolnick et al. | Replay | Why interleaving old data works — the mechanism behind our core result | **RED** |
| **Curriculum Learning** (2009) | Bengio et al. | Curricula | Relevant to the 2D→3D→4D progression in Tasks 19C/19D/20 | **YELLOW** |
| **Neural Networks and the Chomsky Hierarchy** / length-generalization work | Delétang et al. and related | Algorithmic generalization | Why models fail to extrapolate length — directly our E7 0.00% | **YELLOW** |
| **Scheduled Sampling** (2015) | Bengio et al. | Exposure bias | The train/test mismatch that explains our CoT failure | **YELLOW** |
| **The Curious Case of Neural Text Degeneration** | Holtzman et al. | Decoding | Why greedy loops; nucleus sampling — the theory behind Task 16B | **RED** |
| **PPO** (2017) / **DeepSeekMath (GRPO)** | Schulman et al. / Shao et al. | RL | Clipped objective; GRPO's group-relative advantage without a critic. Future work only. | **YELLOW** |
| **PyTorch AMP docs** | PyTorch | Mixed precision | The GradScaler contract; which ops stay FP32 | **RED** |
| **PyTorch CUDA semantics / memory docs** | PyTorch | Memory | Caching allocator; allocated vs reserved; fragmentation | **YELLOW** |
| **A Mathematical Framework for Transformer Circuits** | Elhage et al. (Anthropic) | Interpretability | Residual-stream view; induction heads. Read to know what you'd *need* to claim a mechanism for the OOD failure. | **GREEN** |

---

# PART 30 — "LEARN THIS BEFORE INTERVIEW" CHECKLIST

### 🔴 RED — cannot answer ⇒ serious weakness

- [ ] Derive the 42,132,480 parameter count on a whiteboard
- [ ] Write $\text{Attn}=\text{softmax}(QK^\top/\sqrt{d_k}+M)V$ and explain **every** symbol
- [ ] Explain **why $\sqrt{d_k}$** via the variance argument, and why unscaled scores kill gradients
- [ ] Explain causal masking and why training loss collapses without it
- [ ] Write the cross-entropy objective and derive $\partial\mathcal L/\partial z_i = p_i - \mathbb 1[i=c]$
- [ ] $\text{PPL}=e^{\mathcal L}$; state what PPL does *not* measure (our base: PPL 17.68, GSM8K 0.76%)
- [ ] Explain residual connections via the identity term in the Jacobian
- [ ] Explain gradient accumulation and prove the batch-32 equivalence
- [ ] Explain AMP + GradScaler and why FP16 needs loss scaling
- [ ] **Tell the tied-embedding story**: freezing `tok_emb` froze `lm_head`, model couldn't stop, +30.38 pts on unfreezing
- [ ] **Tell the leakage story**: 24.28% → 2.48%, 111/313 overlap, and that *you* found it
- [ ] **Tell the replay story**: composition beats dose; 50% wrong replay = 3.80%, 25% right replay = 94.80%
- [ ] State the headline honestly: GSM8K 2.43% ≈ Direct SFT 2.35%, not a reasoning model
- [ ] Explain why CoT lost: exposure bias + degeneration, deficit growing with depth
- [ ] Know the key numbers cold: 42,132,480 / 500M tokens / 17.68 base PPL / 50.74 final PPL / 2.43% / 94.80% / 0.60% E7

### 🟡 YELLOW — should know

- [ ] AdamW update rule and what decoupling fixes
- [ ] Warmup + cosine, and why SFT uses 6× lower LR than pretraining
- [ ] Temperature / top-p / repetition-penalty equations
- [ ] Task 16B numbers: 82.6% → 11.45% repetition, 39.8% → 90.67% format, accuracy still below Direct
- [ ] Interpolation vs generalization vs extrapolation, with E1 100% vs E7 0.00%
- [ ] Allocated vs reserved CUDA memory; why they differ
- [ ] Why LayerNorm not BatchNorm
- [ ] Atomic checkpoint writes and what must be in a checkpoint for valid resume
- [ ] McNemar's test for paired model comparison
- [ ] The 60:40 corpus mixture — and that it was **never ablated**
- [ ] Chinchilla ~20 tokens/param, and that we plateaued at ~295M
- [ ] RCS formula and why geometric mean

### 🟢 GREEN — nice to know

- [ ] GRPO vs PPO; why GRPO suits a verifier-based reward
- [ ] KL regularization as the fix for PPL regression
- [ ] KV caching and why our greedy eval took 92 minutes without it
- [ ] RoPE/ALiBi vs learned absolute positions, and how that connects to our padding artifact
- [ ] Interpretability methods you'd need to actually claim a mechanism for OOD failure
- [ ] Total variation distance as a corpus-matching metric

---

# PART 31 — THE 5-MINUTE PROJECT DEFENSE

> **"Tell me about your project."**

"I built Mini-R1 — a 42-million-parameter decoder-only transformer, written from scratch in PyTorch. No HuggingFace model classes, no prebuilt attention; I wrote the multi-head attention, the training loop, the data pipeline and the evaluation harness by hand. I pretrained it on 500 million tokens on a single 6 GB RTX 4050 laptop GPU — about eight hours at 17,000 tokens a second, peaking at 3.4 GB of VRAM.

The research question was: if you teach a small model arithmetic with *verified* supervision, does that transfer to natural-language math word problems? I chose 42M deliberately — at 7B, arithmetic is already latent from pretraining, so you can't separate what you taught from what was already there. At 42M you have to teach it, which makes the question clean.

**Architecture**: 12 layers, 512 model dimension, 8 heads, 2048 feed-forward, 512 context, tied input/output embeddings, pre-LayerNorm. Trained with AdamW, FP16 mixed precision with a gradient scaler, micro-batch 8 with 4-step gradient accumulation for an effective batch of 32. Base model hit validation perplexity 17.68.

**The first surprise** was that chain-of-thought supervision made things *worse*: 1.21% on GSM8K versus 2.73% for direct-answer supervision, and that was significant at p equals 0.005. My first hypothesis was degenerate decoding — CoT had 82.6% repetition. So I ran a decoding ablation on the frozen checkpoint: temperature, top-p and repetition penalty cut repetition to 11% and raised format compliance from 40% to 91%. Accuracy went to 2.2% — **still below direct answering**. So the text was fixable and the reasoning wasn't there. That reframed everything: the bottleneck was arithmetic execution.

**So I built verified arithmetic datasets** where every label is computed by Python. That worked — 100% on held-out two-digit subtraction. But then I audited my own result and found training-set leakage: an earlier three-digit division number of 24.28% was actually 35% overlap with training, and on genuinely unseen problems it was 2.48%. I rebuilt the splits and from then on only quoted audited numbers.

**Then the core negative result.** I ran a matched-control experiment — same data, same steps, same seed, only the initialization differed — and arithmetic pretraining gave a transfer delta of minus 0.3% on GSM8K. Worse, fine-tuning *destroyed* the arithmetic: retention fell from 100% to about 4%. Catastrophic forgetting.

**I tried to prevent that by freezing** the bottom half of the network. It failed badly — equation validity dropped 14 points. The reason turned out to be interesting: the model has tied embeddings, so freezing the token embedding also froze the output head. The model literally couldn't learn to emit its stop token; generations ran to the 256-token cap in loops. I ran the isolating control with everything trainable and recovered 30 points. That also retracted my own earlier conclusion — I'd blamed the dataset, and the control proved it was the freezing.

**What finally worked was replay.** Interleaving 25% verified arithmetic during GSM8K fine-tuning took two-digit retention from 4.4% back to 94.8%. And the important detail: an earlier arm with *50%* replay retained almost nothing — 3.8% — because that replay was from a different arithmetic distribution than the one I was measuring. So composition matters more than volume. You have to replay what you want to keep.

**Limitations, honestly**: this is not a reasoning model. GSM8K sits at 2.43%, which is statistically indistinguishable from plain direct-answer SFT at 2.35% — one problem out of 1,319. The arithmetic is regime-local: 89% on trained two-digit ranges, 0.6% at five digits. General perplexity regressed from 17.68 to 50.74. It's a single seed. And the replay curve is confounded — the arms differ in composition as well as ratio, so I can't claim an optimal ratio.

**Future work**: deconfound that curve with one more run; add a language-replay stream, since every arm blew my perplexity gate including the baseline; and GRPO — I already have a deterministic verifier, which is normally the hard part of reward design.

What I'd claim is this: a clean negative result on transfer, a reproducible positive one on catastrophic forgetting, and every arithmetic claim checked by Python rather than by another model."

---

# PART 32 — THE 15-MINUTE DEEP TECHNICAL DEFENSE

**Q: "Walk me from tokens to logits."**

"Input is a string. SentencePiece BPE with an 8,000 vocabulary turns it into integer ids — and notably every digit is its own token, so `1738` is four tokens. Batch shape is (B, T), 8 by 512.

Those ids index an 8000×512 embedding table, giving (8, 512, 512). I add a learned absolute position embedding — a 512×512 table sliced to sequence length — because attention is permutation-equivariant; without positions, `87 − 43` and `43 − 87` are the same set of vectors.

Then 12 identical pre-LayerNorm blocks. In each: LayerNorm over the 512 features of each token, then three projections give Q, K, V, each (8,512,512), reshaped to (8, 8 heads, 512, 64). Attention scores are Q times K-transpose — (8,8,512,512) — divided by root-64, plus a causal mask that's zero on the lower triangle and negative infinity above. Softmax over the key axis, multiply by V, giving (8,8,512,64), concatenate the heads back to 512, and an output projection. That result is added to the residual stream.

Second sub-layer: LayerNorm, then feed-forward 512 → 2048 → activation → 512, added to the residual again. The FFN is two-thirds of each block's parameters and it's where per-position nonlinear computation happens — attention only mixes across positions.

After 12 blocks, a final LayerNorm, then the LM head projects 512 → 8000. That head is the *same tensor* as the token embedding, transposed — the embeddings are tied. Output is (8, 512, 8000) logits. Softmax gives probabilities; cross-entropy against the sequence shifted by one gives the loss, with prompt positions masked to −100 so we only supervise the completion."

**Q: "Why attention?"** — "Because it's a content-addressed lookup with a constant path length between any two positions. An RNN takes O(T) steps to relate position 1 to position 500, so the gradient has to survive 500 multiplications. Attention relates them in one step. And it parallelises across the sequence, which is what makes training on 500M tokens feasible in eight hours. The cost is O(T²) memory, which is exactly why my context is capped at 512 — at batch 8, 8 heads, T=512, the score matrices alone are 33.6 MB per layer in FP16."

**Q: "Why 42M?"** — "The card. 6 GB. Training needs roughly 16 bytes per parameter of static state — FP32 master weights, gradients, and Adam's two moments — that's 674 MB, plus activations. 42M with batch 8 at 512 context peaked at 3.4 GB, leaving headroom. But there's a research reason too: at 42M, arithmetic isn't latent from pretraining, so 'does teaching arithmetic transfer' is a clean question."

**Q: "How was it trained?"** — "500M tokens, 60% cosmopedia-v2 and 40% fineweb-edu-dedup, EOS-packed contiguously to 100% token density — naive truncation was wasting 58%. Stored as uint16 shards, memory-mapped. 16,384 tokens per optimizer step, so 30,518 steps. AdamW, peak LR 6e-4 chosen by a 1,000-step sweep, cosine to 6e-5, 100 warmup steps, weight decay 0.01, gradient clipping at 1.0, FP16 with a GradScaler. Eight hours, zero skipped steps. Best checkpoint was at step 18,000 — validation loss plateaued at 295M tokens and the last 41% of the budget bought nothing measurable."

**Q: "Why did CoT fail?"** — "Two mechanisms. Exposure bias: teacher forcing conditions each step on the *gold* prefix, but at generation the model conditions on its own output, so if per-step accuracy is q, an n-step chain is roughly q-to-the-n. CoT converts one hard problem into n chances to fail. Second, degeneration — 82.6% four-gram repetition under greedy. The signature that convinces me it's compounding error is that the deficit *grows with depth*: minus 1.11 points at three steps, minus 1.40 at four-plus."

**Q: "How did you prove it?"** — "Two things. For significance, a paired test on the same 1,319 problems gave p equals 0.0051. For the mechanism, I froze the CoT checkpoint and varied *only* decoding — five configurations, no retraining. That's a clean ablation with one independent variable. Repetition fell 82.6 to 11.45 and format compliance rose 39.8 to 90.67, so the degeneration was real and fixable. But accuracy went 1.21 to 2.20 — still below direct answering's 2.73. If reasoning had been present and merely obscured, cleaning the text would have revealed it. It didn't. The traces became structurally clean and mathematically wrong."

**Q: "Why did arithmetic supervision help?"** — "Deterministic labels. Every target is computed in Python, so there's zero label noise, I control difficulty exactly by operand range and operator, I can generate unlimited data, and — the part I like — the evaluation function is the *same function* as the label generator, so 'correct' is unambiguous. That took held-out two-digit subtraction to 100%."

**Q: "Why did it fail to generalize?"** — "It's regime-local. 100% on two-digit inside the trained range, 18.2% just outside it, 0.00% at five digits — with identical tokenization throughout. Adding a regime helps that regime and doesn't propagate: three-digit exposure took unseen three-digit from 6.6% to 34.4%, while untrained four-digit stayed at 0.4%. And it *costs* — unseen two-digit fell from 35.4% to 28.4% when I added three-digit data. So there's interference even within arithmetic. What I will not claim is *why* at the mechanism level. A true column-wise subtraction algorithm would be digit-count agnostic, and 0.00% at five digits says it isn't one — but attributing that to positional encodings or missing carry circuits would need interpretability work I didn't do."

**Q: "Why did replay help?"** — "It keeps the old task's gradient signal in the optimization, so the update direction can't freely destroy it. But the real finding is that it's distributional, not volumetric. My 50%-replay arm retained 3.8% — worse than the 0%-replay baseline — because its replay was operator-matched to GSM8K and contained essentially none of the subtraction regime I was probing. The 25% arm, whose replay *did* contain that regime, retained 94.8%. The arm with the most replay retained the least. You have to replay what you want to keep."

**Q: "What is your final conclusion?"** — "Three things, in decreasing confidence. Verified arithmetic supervision produces genuinely strong arithmetic at 42M, but strictly inside trained numeric regimes — interpolation, not extrapolation. That competence does *not* transfer to word problems; my best-designed transfer experiment measured minus 0.3%, and fixing the distributional coverage didn't rescue it, which points at semantic grounding rather than arithmetic as the GSM8K bottleneck. And catastrophic forgetting of that competence *is* preventable by replaying the target distribution — 4.4% to 94.8% — with composition mattering more than dose.

The honest summary is that it's a negative result on the headline question and a positive, reproducible one on the side question, in a setting where every arithmetic claim is machine-verifiable."

---

# PART 33 — IF THE INTERVIEWER ATTACKS THE PROJECT

**"This isn't really a reasoning model, is it?"**
"No, and I don't call it one. GSM8K is 2.43%, which is statistically indistinguishable from plain direct-answer fine-tuning at 2.35% — one problem out of 1,319. The project's own context document explicitly says not to call it a reasoning model. What it is, is a controlled study of what a 42M model *can* learn — verified arithmetic — and what happens to that capability under fine-tuning."

**"You mostly measured arithmetic."**
"That's accurate, and it was deliberate after Task 17A. GSM8K conflates reading comprehension, mapping semantics to operations, arithmetic execution, and formatting — a single 2% number can't tell you which failed. Arithmetic is the sub-skill I could measure exactly and verify deterministically. The limitation is real: I improved the measurable part and the unmeasurable part — semantic grounding — is where GSM8K actually fails."

**"Your GSM8K accuracy is only about 2%."**
"Yes, and it never exceeded that in any configuration across the whole project. Three things are true about that number: it's strict extraction with required format, so it's a conservative measurement; my harness has a padding artifact that depresses absolute numbers by roughly 3 to 14 points of equation validity; and most importantly, I don't use it as the success metric. My pre-registered metric was retention plus equation validity plus perplexity, because GSM8K at these rates is noise."

**"You found leakage in your own experiment."**
"I did, and I'd say that's the strongest thing on my CV from this project. A 24.28% result looked too good against everything else being near-random, so I keyed every eval pair against the entire training corpus including numbers nested inside multi-step traces. 111 of 313 overlapped; the true unseen accuracy was 2.48%. I rebuilt the splits, changed the methodology permanently, and left the original number in the report with the correction. The alternative — accepting a good number and moving on — is how bad results get published."

**"Your replay experiment is confounded."**
"It is, and I say so in the report before anyone asks. The 0/10/25/50 arms differ in replay *composition* as well as share: the 50% arm's replay contained none of the regime I was probing. So I can claim 'replaying the target distribution at 10 to 25% preserves it' — that's supported. I cannot claim 25% is optimal, and I don't. The missing run is 50% with the mixed block; it's one run and it's the first thing on my future-work list."

**"You never ran RL."**
"Correct — zero RL. It was a stated project goal and it didn't happen. I'm not going to claim anything about it. I'll say why it's the natural next step: I already have a deterministic verifier, which is normally the hard part of reward design. And I'll name the real obstacle — at 2% success rate the reward is extremely sparse, so exploration becomes the binding problem, and I'd want to solve that before burning compute on GRPO."

**"Why call this Mini-R1?"**
"It's aspirational naming from the start of the project, after DeepSeek-R1's verified-reasoning approach, and honestly it overpromises relative to what the model does. The pipeline shares the shape — verified reasoning traces, a deterministic checker, RL as the intended final stage — but I never got to the RL stage and the reasoning results are negative. If I were naming it today I'd call it something about arithmetic retention."

**"Why should I consider this research?"**
"Three reasons, and I'd rank them. First, methodological: every arithmetic claim is checkable by Python, not by another model, and I built a step-level equation-validity metric that separates 'set up the wrong computation' from 'computed it wrong' — a distinction plain accuracy hides. Second, the tied-embedding finding is a concrete, transferable engineering trap: parameter freezing is actively harmful when embeddings are tied, and PyTorch's own `named_parameters()` won't show you the coupling. Third, the composition-over-dose replay result generalises well beyond this model — a replay buffer from the wrong part of the input space provides no protection regardless of size.

And I'd add: I ran four experiments that disconfirmed my own hypotheses and reported all of them, including one where I had to retract a conclusion from my own previous report. That's the part I'd want you to weigh."

---

# PART 34 — FINAL RESEARCH HONESTY

## WHAT WE PROVED
*(controlled manipulation, matched controls, traceable artifacts)*

1. **Verified arithmetic SFT produces strong IID arithmetic at 42M.** E1 100.00%, E3 99.80%, E5 67.20% [`task20_*.json`].
2. **Arithmetic competence is regime-local.** E7 5-digit = 0.00% with identical tokenization to E1's 100% [same].
3. **CoT supervision hurt at this scale.** 2.73% → 1.21%, p = 0.0051, deficit growing with depth [`task15c`].
4. **The CoT failure was not a decoding artifact.** Frozen-checkpoint ablation: repetition 82.6% → 11.45%, format 39.8% → 90.67%, accuracy still below Direct [`task16b`].
5. **Arithmetic pretraining does not transfer to GSM8K.** Matched controls: Δ = −0.30% [`task21b`].
6. **GSM8K SFT causes catastrophic forgetting of arithmetic.** 100/99.8/67.2 → ~4/~5/~0.6 across every unprotected arm.
7. **Parameter freezing caused the Task 22 degeneration.** Single-variable control: +30.38 pts greedy equation validity when unfrozen; p90 length 256 → 59 [`task22b`].
8. **Target-distribution replay preserves arithmetic.** 4.40% → 94.80% 2D retention [`task23`].
9. **Replay composition matters at least as much as dose.** 50% off-distribution replay → 3.80%; 25% on-distribution → 94.80%.

## WHAT WE OBSERVED
*(measured, real, but not from a controlled manipulation)*

- Validation loss plateaued at ~295M tokens; the final 41% of the pretraining budget produced no measurable improvement.
- General PPL degrades monotonically with replay share: 43.88 → 47.90 → 50.74 → 60.02.
- Adding a numeric regime can *cost* accuracy on earlier ones (unseen 2D: 35.40% → 28.40% when 3D was added).
- Answer length is an independent difficulty axis (19C OOD 3D: 6.60% with 1–2 digit answers, 0.00% with 3-digit).
- Repetition on the E-suites rises with digit count (0.0% → 18.6% at 5 digits).
- Every digit is a separate token, and this was true in both the 100% and the 0.00% conditions.
- AQuA is 0.00% with 0.0% full-format compliance for the final model — MCQ format transfer is absent.

## WHAT WE SUSPECT
*(consistent with the data; not established)*

- The model learned per-regime pattern completion rather than a digit-count-invariant subtraction algorithm. *Basis*: 0.00% at 5 digits; a general algorithm would be length-agnostic.
- The residual GSM8K bottleneck is **semantic grounding** (mapping words → operations), not arithmetic. *Basis*: fixing distributional coverage to TVD 0.0063 did not produce transfer.
- Capacity or corpus diversity — not token count — bound the pretraining plateau. *Basis*: plateau at 295M despite Chinchilla suggesting ~840M.
- Tied embeddings make *any* embedding-freezing intervention harmful in this architecture class. *Basis*: one experiment, one architecture.

## WHAT WE DID NOT TEST

- **Any RL.** No GRPO, no PPO, no policy gradient. Zero evidence in either direction.
- **Any other model size.** No scaling comparison; every claim is at 42M.
- **Any other seed.** Single seed (42) throughout; no variance estimates anywhere.
- **The 60:40 corpus ratio.** Never ablated.
- **The full-budget LR comparison.** 6e-4 chosen at step 1,000 of 30,518.
- **50% replay with mixed composition** — the run that would deconfound the ratio curve.
- **Any interpretability.** No probing, activation patching, or attention analysis. We cannot attribute any behaviour to any internal mechanism.
- **The harness fix.** We measured the padding artifact but never re-ran everything without it.
- **Untied embeddings.** The tied-embedding conclusion rests on one architecture.

## WHAT FUTURE WORK WOULD TEST

| Question | Experiment | Cost |
| :--- | :--- | :--- |
| Is 25% the optimal ratio, or is it composition? | 50% replay with the mixed block | 1 run |
| Can PPL regression be prevented? | Add a pretraining-shard language-replay stream | 1–2 runs |
| Are the absolute numbers really conservative? | Pass the attention mask the harness already builds; re-measure all arms | eval only |
| Is transfer capacity-bound or method-bound? | One 70–150M arm, same protocol | 1 run + hardware |
| Does a verifier-based reward help? | GRPO on the deterministic equation checker, with KL to the reference policy | significant |
| *Why* is arithmetic regime-local? | Probing digit-position representations; activation patching across regimes | research programme |
| Is semantic grounding the real bottleneck? | Supervise the word→operation mapping directly, holding arithmetic fixed | 1–2 runs |

---

# PART 35 — 48-HOUR STUDY PLAN

Assumes ~10 focused hours/day. **Practise out loud** — recognition is not recall.

### Day 1

**Block 1 (2h) — Transformer mathematics 🔴**
Derive the 42,132,480 parameter count three times, from memory, until fluent. Write the attention equation and explain each symbol aloud. Derive the $\sqrt{d_k}$ variance argument. Write the causal mask and explain what breaks without it.
*Practise*: narrate tokens→logits with shapes, unaided, in under 3 minutes.

**Block 2 (1.5h) — Training objective 🔴**
Write the cross-entropy objective. Derive $\partial\mathcal L/\partial z_i = p_i - \mathbb 1[i=c]$ on paper twice. Explain teacher forcing, then exposure bias, then connect exposure bias to the CoT failure. Compute PPL from loss and back.

**Block 3 (1.5h) — Optimization 🟡**
Write the AdamW update from memory. Explain what decoupling fixes. Explain warmup and cosine. Prove the gradient-accumulation equivalence and name the three reasons it's only approximate in practice.

**Block 4 (2h) — The arithmetic arc 🔴**
Memorise the progression: 19C (35.40 / 6.60) → 19D (34.40, +27.80; 2D fell to 28.40) → Task 20 (E1 100.0 … E7 0.00). Practise the interpolation/generalization/extrapolation distinction. **Practise saying "I did not prove a mechanism."**

**Block 5 (2h) — Experimental design 🔴**
Rehearse the leakage story until it's smooth: 24.28% → 111/313 overlap → 82.14% seen vs 2.48% unseen → rebuilt splits. Then the confound story: Task 22 → 22B. Then McNemar and binomial CIs; compute the CI on 2.43% with n=1319.

**Block 6 (1h) — Recite Part 31** (the 5-minute defense) three times, timed. Cut anything you stumble on.

### Day 2

**Block 7 (2h) — Forgetting and replay 🔴**
The core contribution. Memorise the four-arm table. Rehearse "composition beats dose" with the F-vs-H contrast (50% → 3.80%, 25% → 94.80%). Derive RCS by hand for Model H; get 0.4569. Explain the geometric mean. **Rehearse volunteering the confound before being asked.**

**Block 8 (1.5h) — Decoding 🟡**
Temperature, top-p, repetition-penalty equations. Why $T>1$ flattens (the $z/T \to 0$ argument). Task 16B numbers and — most importantly — what they did *not* prove.

**Block 9 (1.5h) — Engineering 🟡**
VRAM budget: 16 bytes/param static, the $T^2$ attention term, why inference is ~8× cheaper. Allocated vs reserved. AMP + GradScaler. Atomic writes and checkpoint contents. Have the two war stories ready (missing attention mask; silently broken resume).

**Block 10 (1h) — Evaluation 🟡**
Strict vs fallback extraction and why it matters (48.1% extraction vs 0.76% accuracy). Step-level equation validity and why the rate needs its absolute counterpart. The padding control and its direction.

**Block 11 (1h) — Deployment + RL as future work 🟢**
`web/server.py` is stdlib `http.server` — **FastAPI was planned, never built.** Inference memory ~84 MB FP16. KV caching as the obvious speedup. Then RL: policy gradient, advantage, GRPO vs PPO, KL regularization — enough to discuss intelligently, framed strictly as future work.

**Block 12 (2h) — Adversarial rehearsal 🔴**
Work through Part 33 out loud. Have someone (or a recording) fire the attacks at you. The goal is to answer without defensiveness and without overclaiming. Then Part 32's 15-minute walkthrough, timed.

**Block 13 (1h) — Number drill**
42,132,480 · 500M tokens · 30,518 steps · 16,384 tokens/step · 17,316.5 tok/s · 3,409.7 MB peak · PPL 17.68 base → 50.74 final · Direct 2.73 vs CoT 1.21 · repetition 82.6 → 11.45 · leakage 24.28 → 2.48 · transfer −0.30 · freezing +30.38 · retention 4.40 → 94.80 · E1 89.60 · E7 0.60 · final GSM8K 2.43 · RCS 0.4569

### Priority if time collapses
Do Blocks 1, 4, 7, 12 and the number drill. That covers the mathematics you'll be tested on, the two stories that are genuinely yours (leakage and replay), and the ability to take criticism without either folding or overclaiming.

---

## FINAL PROJECT STATUS

**FINAL PROJECT STATUS:**
Research experimentation complete.

**FINAL MODEL:**
Task 23 Model H
`artifacts/sft/task23_75_25/best.pt`
SHA256 `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c`

**NEXT:**
Final evaluation + API + deployment/demo.
