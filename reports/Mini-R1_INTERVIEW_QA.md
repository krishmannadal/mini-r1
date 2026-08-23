# Mini-R1 — Interview Question Bank (Rapid Revision)

**150 questions.** Companion to `Mini-R1_MASTER_RESEARCH_AND_INTERVIEW_GUIDE.md`.
Format for each: **Q** · *Testing* · **Short** · Detailed · Equation · Mini-R1 link · ⚠️ Common mistake · → Follow-up

**Key numbers to have cold**: 42,132,480 params · 500M tokens · 30,518 steps · 16,384 tok/step · PPL 17.68 base → 50.74 final · Direct 2.73% vs CoT 1.21% · repetition 82.6% → 11.45% · leakage 24.28% → 2.48% · transfer −0.30% · freezing +30.38 · retention 4.40% → 94.80% · E7 0.60% · final GSM8K 2.43% · RCS 0.4569

---

# SECTION A — BASIC (30)

**A1. What is a transformer?**
*Testing*: foundational vocabulary.
**Short**: A neural architecture built on self-attention that processes a sequence in parallel, relating any two positions in one step.
*Detailed*: Stacked blocks, each with multi-head self-attention (mixes information across positions) and a feed-forward network (per-position nonlinear computation), joined by residual connections and LayerNorm. Mini-R1 is decoder-only: causally masked, autoregressive.
*Equation*: $\text{Attn}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d_k})V$
*Mini-R1*: 12 blocks, d=512, 8 heads, written from scratch in `src/model.py`.
⚠️ Saying "transformers use RNNs internally" — they explicitly replaced recurrence.
→ Why is parallelism across the sequence possible at training but not at generation?

**A2. Encoder-decoder vs decoder-only?**
*Testing*: architecture families.
**Short**: Decoder-only does causal next-token prediction; encoder-decoder maps an input sequence to a separate output sequence with cross-attention.
*Detailed*: Decoder-only (GPT family) is simpler and directly matches language modelling. Encoder-decoder (T5, original Transformer) suits translation where input and output are distinct sequences.
*Mini-R1*: Decoder-only — the task is next-token prediction and prompt+completion live in one sequence.
⚠️ Thinking decoder-only can't do "input→output" tasks; it can, by concatenation.
→ How do you stop the model training on the prompt tokens?

**A3. What is self-attention?**
**Short**: Each position builds a query, compares it to every position's key, and takes a weighted sum of their values.
*Detailed*: "Self" because Q, K, V all come from the same sequence. It's a differentiable, content-addressed lookup.
*Equation*: $Q=XW_Q,\ K=XW_K,\ V=XW_V$
*Mini-R1*: `src/attention.py`, 8 heads × 64 dims.
⚠️ Confusing the *attention weights* (T×T) with the *output* (T×d).
→ Why three separate projections instead of reusing X?

**A4. Why Q, K, and V separately?**
**Short**: They decouple *matching* from *content*.
*Detailed*: If Q=K the attention matrix is forced symmetric; if V=K the retrieved content is forced to equal the matching signal. Separating them lets "what I look for", "what I advertise" and "what I deliver" differ.
*Mini-R1*: 4×512×512 = 1,048,576 attention params per block (Q,K,V,O).
⚠️ Saying they're "just three copies."
→ What if you tied Q and K?

**A5. What does the softmax do?**
**Short**: Maps arbitrary reals to a probability distribution.
*Equation*: $p_i=e^{z_i}/\sum_j e^{z_j}$
*Detailed*: Positive, sums to 1, monotonic, differentiable. Used twice: attention weights and output token distribution. Implementations subtract $\max_j z_j$ first — invariant, prevents `exp` overflow.
*Mini-R1*: over 8,000 logits at the head; over T keys in attention.
⚠️ Forgetting the max-subtraction trick.
→ What's the softmax Jacobian?

**A6. What is cross-entropy loss?**
**Short**: Negative log-probability the model assigned to the correct token.
*Equation*: $\mathcal L=-\sum_t \log p_\theta(y_t\mid y_{<t})$
*Detailed*: Equivalently the KL from the one-hot target to the predicted distribution, up to a constant. Minimising it is maximum likelihood.
*Mini-R1*: the entire training objective, pretraining and SFT.
⚠️ Confusing it with MSE.
→ Derive its gradient w.r.t. logits.

**A7. What is perplexity?**
**Short**: $\exp(\text{loss})$ — the effective number of choices per token.
*Equation*: $\text{PPL}=e^{\mathcal L}$
*Detailed*: PPL 17.68 means as uncertain as choosing uniformly among ~17.7 options. Uniform over our 8,000 vocab would be 8,000.
*Mini-R1*: base 17.68 → final model 50.74.
⚠️ Comparing PPL across different tokenizers — meaningless.
→ Our base model had PPL 17.68 and 0.76% on GSM8K. What does that tell you?

**A8. What is a token?**
**Short**: The atomic unit the model consumes — here a BPE subword.
*Detailed*: Not a word and not a character. Mini-R1's vocab is 8,000; every digit 0–9 is its own token, and `<think>`/`<answer>` tags are single tokens.
*Mini-R1*: `1738` → `1`,`7`,`3`,`8` (4 tokens), verified.
⚠️ Assuming numbers are single tokens.
→ Why does digit-level tokenization matter for arithmetic?

**A9. What is an embedding?**
**Short**: A learned vector per token id.
*Equation*: $X=E[\mathbf t],\ E\in\mathbb R^{8000\times512}$
*Detailed*: Discrete symbols have no geometry; embeddings give them one so similarity is meaningful.
*Mini-R1*: 4,096,000 params — 9.7% of the model.
⚠️ Thinking it's a "one-hot times matrix" at runtime; it's a lookup.
→ What does tying embeddings mean?

**A10. Why do transformers need positional encoding?**
**Short**: Attention is permutation-equivariant; without positions, order is invisible.
*Equation*: $\text{Attn}(\Pi X)=\Pi\,\text{Attn}(X)$
*Detailed*: `87 − 43` and `43 − 87` would be the same set of vectors. Subtraction isn't commutative, so this matters enormously for us.
*Mini-R1*: learned absolute table, 512×512 = 262,144 params.
⚠️ Saying "the model learns order from data" — it structurally cannot.
→ What's the downside of *absolute* learned positions?

**A11. What is causal masking?**
**Short**: Blocking attention to future positions.
*Equation*: $M_{ij}=0$ if $j\le i$, else $-\infty$
*Detailed*: Added before softmax so $e^{-\infty}=0$. Makes teacher forcing valid — every position is trained on its legitimate past only, in one forward pass.
*Mini-R1*: `blocks.N.attn.causal_mask`, a (1,1,512,512) buffer.
⚠️ Applying it *after* softmax — weights then don't sum to 1.
→ What symptom would a missing causal mask produce?

**A12. What is a residual connection?**
**Short**: Adding a sublayer's input to its output.
*Equation*: $x \leftarrow x+F(\text{LN}(x))$; $\partial/\partial x = I + \partial F/\partial x$
*Detailed*: The identity term guarantees a gradient path to earlier layers.
*Mini-R1*: two per block, 24 total.
⚠️ Saying it "prevents overfitting" — it's about gradient flow and easier optimisation.
→ What breaks at 12 layers without them?

**A13. What is LayerNorm?**
**Short**: Normalise each token's features to zero mean, unit variance, then scale and shift.
*Equation*: $\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$
*Detailed*: Statistics computed over the 512 features of a single token — independent of batch and sequence position.
*Mini-R1*: pre-LN, plus a final `ln_f`.
⚠️ Saying it normalises across the batch — that's BatchNorm.
→ Why not BatchNorm for language?

**A14. Pre-LN vs post-LN?**
**Short**: Pre-LN normalises before the sublayer, leaving a clean residual highway; post-LN normalises after.
*Detailed*: Pre-LN trains stably at depth without delicate warmup; post-LN (the 2017 original) is more prone to divergence.
*Mini-R1*: pre-LN, chosen in Task 4.
⚠️ Assuming the original paper's arrangement is what modern models use.
→ Which does GPT-2 use?

**A15. What is the FFN for?**
**Short**: Per-position nonlinear computation.
*Equation*: $\text{FFN}(x)=W_2\phi(W_1x)$, 512→2048→512
*Detailed*: Attention mixes across positions but is linear in V. The FFN is where transformation happens. It's 2/3 of each block's parameters.
*Mini-R1*: 2,097,152 params per block.
⚠️ Thinking attention does the "thinking" and the FFN is filler.
→ Why 4× expansion?

**A16. Why 8 heads instead of 1?**
**Short**: Multiple attention patterns simultaneously, at no parameter cost.
*Detailed*: One 512-dim softmax gives one weighting per position. Eight 64-dim heads give eight — e.g. one tracking the operator, one the left operand.
*Mini-R1*: 8 × 64 = 512.
⚠️ Thinking more heads = more parameters. You're partitioning $d$.
→ What's the trade-off in raising head count?

**A17. What is autoregressive generation?**
**Short**: Predict one token, append it, repeat.
*Equation*: $p(y_{1:T})=\prod_t p(y_t\mid y_{<t})$
*Mini-R1*: stops on EOS or `</answer>` (single token, id 7).
⚠️ Forgetting that generation is sequential — it can't be parallelised the way training can.
→ What's exposure bias?

**A18. Greedy vs sampling?**
**Short**: Greedy takes the argmax deterministically; sampling draws from the distribution.
*Detailed*: Greedy is reproducible and prone to loops; sampling is diverse and more error-prone.
*Mini-R1*: greedy 2.43% GSM8K vs sampled 1.21% — but sampled has 16.4% repetition vs greedy's 77.3%.
⚠️ Assuming sampling is always better.
→ Why does greedy loop?

**A19. What is temperature?**
**Short**: A divisor on logits before softmax controlling distribution sharpness.
*Equation*: $p_i=e^{z_i/T}/\sum_j e^{z_j/T}$
*Detailed*: $T\to0$ → argmax; $T>1$ flattens toward uniform (as $T\to\infty$, $z_i/T\to0$ and softmax of zeros is uniform).
*Mini-R1*: T=0.6 in the Task 16B best configuration.
⚠️ Saying temperature "adds randomness" — it reshapes an already-stochastic distribution.
→ Why does T=0.6 (below 1) help here?

**A20. What is top-p sampling?**
**Short**: Keep the smallest set of tokens whose cumulative probability ≥ p, renormalise, sample.
*Detailed*: Adaptive — a confident distribution keeps 1–2 tokens, a flat one keeps many. Better than top-k, which is fixed regardless of confidence.
*Mini-R1*: top_p = 0.95.
⚠️ Confusing it with top-k.
→ When would top-k be preferable?

**A21. What is fine-tuning?**
**Short**: Continuing training a pretrained model on task data.
*Mini-R1*: 650 steps at LR 1e-4 (vs pretraining's 6e-4) from the Task 20 checkpoint.
⚠️ Using the pretraining LR — destroys learned representations.
→ Why 6× lower LR?

**A22. What is SFT?**
**Short**: Supervised fine-tuning on (prompt, completion) pairs with cross-entropy.
*Detailed*: Same objective as pretraining; what changes is the data and the masking.
*Mini-R1*: Direct vs CoT (15C), arithmetic (18B), transfer (21A/B), replay (22/23).
⚠️ Thinking SFT uses a different loss.
→ What's masked and why?

**A23. What is completion-only masking?**
**Short**: Set prompt labels to −100 so only the answer is supervised.
*Equation*: `CrossEntropyLoss(ignore_index=-100)`
*Detailed*: We want $p(\text{answer}\mid\text{question})$, not $p(\text{question})$.
*Mini-R1*: `train_task21b_compact_sft.py` lines 99–101.
⚠️ Off-by-one — the last prompt position predicts the first completion token and must stay supervised.
→ What would supervising the prompt cost you at 42M?

**A24. What is a validation set?**
**Short**: Held-out data used to measure generalization and select checkpoints.
*Mini-R1*: 10M held-out tokens (9,980,852 evaluated).
⚠️ Selecting on test data.
→ Why did your best checkpoint come at step 18,000 rather than the end?

**A25. What is overfitting?**
**Short**: Train loss falling while validation loss rises.
*Mini-R1*: **never observed** — 500M tokens against 42M params is data-rich; val loss decreased then flattened.
⚠️ Claiming any train/val gap is overfitting. Our final row has train 3.0807 > val 2.8726 (noisy step vs 10M-token average).
→ Why is train loss above val loss in your final row?

**A26. What is a learning rate?**
**Short**: Step size along the negative gradient.
*Equation*: $\theta\leftarrow\theta-\eta\nabla_\theta L$
*Mini-R1*: 6e-4 pretraining, 1e-4 SFT.
⚠️ Treating LR as independent of batch size and schedule.
→ What happens if it's too high?

**A27. What is a batch?**
**Short**: The set of examples whose gradients are averaged into one update.
*Mini-R1*: micro 8 × accum 4 = effective 32 = 16,384 tokens/step.
⚠️ Confusing micro-batch with effective batch.
→ Why not just use batch 32 directly?

**A28. What is an epoch?**
**Short**: One full pass over the training data.
*Mini-R1*: SFT ran 650 steps × 32 = 20,800 examples over a 6,531-example GSM8K set ≈ 3.2 epochs.
⚠️ Assuming pretraining runs whole epochs; ours is token-budgeted.
→ How many optimizer steps was your pretraining?

**A29. What is a checkpoint?**
**Short**: Serialised model + optimizer + RNG + data-cursor state.
*Mini-R1*: `best.pt` 509 MB — weights are only 169 MB; the rest is Adam moments and optimizer state.
⚠️ Saving only `model.state_dict()` — resume then silently differs.
→ What must be in it for a *scientifically valid* resume?

**A30. What is inference vs training?**
**Short**: Inference is forward-only; training adds backward, gradients and optimizer state.
*Detailed*: Inference needs ~2 bytes/param (FP16 = 84 MB). Training needs ~16 bytes/param static (674 MB) plus activations.
*Mini-R1*: peak training VRAM 3,409.7 MB; inference under 1 GB.
⚠️ Sizing hardware from parameter count alone.
→ Why is training roughly 8× heavier in static state?

---

# SECTION B — INTERMEDIATE (30)

**B1. Why divide attention scores by $\sqrt{d_k}$?**
*Testing*: the single most-asked derivation.
**Short**: Because $\operatorname{Var}(q\cdot k)=d_k$, so scores grow with dimension and saturate the softmax.
*Detailed*: With unit-variance independent components, $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$, i.e. std $\sqrt{d_k}$. At $d_k=64$ that's ±8 — enough to push softmax near one-hot, where its Jacobian $\operatorname{diag}(p)-pp^\top\to0$ and gradients vanish. Dividing restores unit variance.
*Equation*: $\operatorname{Var}(q\cdot k)=\sum_i\operatorname{Var}(q_ik_i)=d_k$
*Mini-R1*: $d_k=64$, so we divide by 8.
⚠️ Saying "to normalise" without the variance argument, or dividing by $d_k$.
→ What if you scaled by a learned parameter instead?

**B2. Derive the cross-entropy gradient w.r.t. logits.**
**Short**: $p_i-\mathbb 1[i=c]$ — predicted minus actual.
*Detailed*: $\mathcal L=-z_c+\log\sum_je^{z_j}$. Differentiating: the first term gives $-\mathbb 1[i=c]$, the log-sum-exp gives $e^{z_i}/\sum_je^{z_j}=p_i$.
*Mini-R1*: every gradient in the project starts here.
⚠️ Forgetting the log-sum-exp contributes $p_i$.
→ What does this imply about the gradient when the model is already correct and confident?

**B3. Explain gradient accumulation mathematically.**
**Short**: Divide each micro-loss by `accum` and let `.grad` sum; the buffer equals the large-batch gradient.
*Equation*: $\sum_{k=1}^{4}\frac14\cdot\frac18\sum_{i\in\text{micro}_k}\nabla\ell_i=\frac1{32}\sum_{i=1}^{32}\nabla\ell_i$
*Mini-R1*: 8×4=32 everywhere.
⚠️ Forgetting to divide by `accum` — the gradient is 4× too large, effectively 4× the LR.
→ Where is it *not* equivalent?

**B4. Where does gradient accumulation break equivalence?**
**Short**: Floating-point non-associativity, FP16 accumulation error, and batch-dependent layers.
*Detailed*: $(a+b)+c\ne a+(b+c)$ in finite precision; four sequential accumulations sum in a different order than one fused reduction. BatchNorm would break it outright (we use LayerNorm, so we're safe). Padding differences can also shift per-token normalisation.
*Mini-R1*: `tests/test_gradient_accumulation.py`.
⚠️ Claiming bitwise equality.
→ Would BatchNorm break it? Why?

**B5. Why AdamW over Adam?**
**Short**: It decouples weight decay from the adaptive denominator.
*Equation*: $\theta\leftarrow\theta-\eta(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta)$
*Detailed*: In Adam with L2-in-the-gradient, the penalty gets divided by $\sqrt{\hat v}$, so high-gradient parameters receive *less* regularisation — backwards from the intent.
*Mini-R1*: AdamW, $\lambda$=0.01, $\beta$=(0.9, 0.95).
⚠️ Saying they're equivalent.
→ Why $\beta_2$=0.95 rather than 0.999?

**B6. Why warmup?**
**Short**: Adam's second-moment estimate is unreliable at step 0.
*Detailed*: With almost no gradient history, $\hat v$ is noisy; a full-size step on a bad curvature estimate can wreck initialisation. Warmup lets the moments stabilise.
*Mini-R1*: 100 steps pretraining, 50 SFT.
⚠️ Thinking warmup exists "to avoid divergence" generically without the moment-estimate reason.
→ Do optimizers without second moments need warmup?

**B7. Why cosine decay?**
**Short**: Smooth monotone decay to a small final LR so the model settles.
*Equation*: $\eta_t=\eta_{\min}+(\eta_{\text{peak}}-\eta_{\min})\cdot\frac12(1+\cos\frac{\pi(t-T_w)}{T-T_w})$
*Mini-R1*: 6e-4 → 6e-5 over 30,518 steps.
⚠️ Not knowing the schedule must be resumed with the correct step.
→ What happens on resume if you don't restore the scheduler position?

**B8. What is gradient clipping and when do you apply it?**
**Short**: Rescale the gradient if its global L2 norm exceeds a threshold — **after** unscaling.
*Equation*: $g\leftarrow g\cdot\min(1, c/\lVert g\rVert_2)$, $c=1.0$
*Detailed*: Preserves direction, bounds magnitude. Under AMP you must call `scaler.unscale_()` first or you're clipping scaled gradients at the wrong threshold.
*Mini-R1*: max_norm 1.0 in every trainer.
⚠️ Clipping before unscaling.
→ Per-parameter clipping vs global-norm clipping?

**B9. How does AMP work?**
**Short**: FP16 forward/backward with FP32 master weights, plus loss scaling.
*Detailed*: `autocast` selects precision per op (matmuls FP16; reductions, softmax and losses stay FP32). ~2× throughput, ~½ activation memory.
*Mini-R1*: mandatory on 6 GB; verified 0.1085% error vs FP32.
⚠️ Thinking everything becomes FP16.
→ Why does FP16 need a GradScaler?

**B10. Why does FP16 need a GradScaler?**
**Short**: Small gradients underflow to zero below ~6e-5.
*Detailed*: Multiply the loss by S so gradients land in representable range; unscale before clipping/stepping; if inf/NaN appear, skip the step and halve S; after N clean steps, double it.
*Mini-R1*: 0 skipped steps across the 500M run.
⚠️ Forgetting to unscale before clipping.
→ Why doesn't BF16 need one?

**B11. FP16 vs BF16?**
**Short**: Same 16 bits; BF16 trades mantissa for FP32's exponent range, so no scaler is needed.
*Detailed*: FP16 has ~10 mantissa bits and a narrow range (underflow risk). BF16 has ~7 mantissa bits and FP32's range — less precise, far more robust.
*Mini-R1*: FP16 because reliable BF16 support wasn't established on this consumer Ada laptop stack; FP16+scaler was empirically verified.
⚠️ Claiming BF16 is universally better — it's less precise.
→ Would you use BF16 today if available?

**B12. Why LayerNorm and not BatchNorm?**
**Short**: BatchNorm statistics depend on other sequences in the batch.
*Detailed*: That's wrong for variable-length autoregressive text, breaks at batch size 1, and creates a train/inference discrepancy via running statistics. LayerNorm is per-token and batch-independent.
*Mini-R1*: LayerNorm throughout — which is also why gradient accumulation is exactly equivalent for us.
⚠️ Thinking it's an arbitrary preference.
→ How does this interact with gradient accumulation?

**B13. What is teacher forcing?**
**Short**: Conditioning each training position on the ground-truth prefix.
*Detailed*: Combined with causal masking, it lets all T losses be computed in one forward pass. The cost is exposure bias.
*Mini-R1*: all training; directly explains the CoT failure.
⚠️ Thinking the model generates during training.
→ What's the inference-time mismatch called?

**B14. What is exposure bias?**
**Short**: Train on gold prefixes, generate on own prefixes — a distribution the model never trained on.
*Detailed*: If per-step accuracy is $q$, an $n$-step chain succeeds roughly as $q^n$.
*Mini-R1*: the mechanism behind CoT losing to Direct, and the deficit *growing* with depth (−1.11% at 3 steps, −1.40% at 4+).
⚠️ Attributing CoT's failure purely to repetition — decoding fixes reduced repetition 7× and didn't fix accuracy.
→ What mitigations exist? (scheduled sampling, RL on own samples)

**B15. Why did CoT fail at 42M when it works at scale?**
**Short**: Exposure bias plus degeneration, and no arithmetic substrate to chain.
*Detailed*: CoT multiplies opportunities to fail. At 175B the per-step accuracy is high enough that $q^n$ stays useful; at 42M it isn't. Ours produced structurally clean, mathematically wrong traces.
*Mini-R1*: 2.73% Direct vs 1.21% CoT, p=0.0051.
⚠️ Saying "CoT doesn't work" — it works at scale; it failed *at this scale*.
→ How did you rule out decoding?

**B16. What is data leakage?**
**Short**: Evaluation information present in training data.
*Detailed*: Forms include exact duplication, near-duplication, and embedded occurrence inside longer training examples — the one that caught us.
*Mini-R1*: 111/313 (35.46%) overlap turned a real 2.48% into an apparent 24.28%.
⚠️ Checking only top-level prompts, not nested `<think>` content.
→ How did you detect it?

**B17. What is catastrophic forgetting?**
**Short**: Training on a new task overwrites capability on an old one.
*Detailed*: Gradients for the new objective move parameters freely in directions that were load-bearing for the old task.
*Mini-R1*: 100%/99.8%/67.2% → ~4%/~5%/~0.6% after GSM8K SFT.
⚠️ Measuring only language PPL and concluding forgetting is "negligible" — Task 21A's actual error.
→ Name three mitigations.

**B18. Name three mitigations for catastrophic forgetting.**
**Short**: Replay, regularization (EWC/L2-SP), and parameter isolation (adapters/freezing).
*Detailed*: Replay interleaves old data. EWC penalises movement in parameters important to the old task (via Fisher information). Isolation restricts which parameters can move.
*Mini-R1*: we tested **replay** (worked: 4.40% → 94.80%) and **freezing** (failed badly — tied embeddings). We did *not* test EWC or adapters.
⚠️ Claiming we compared all three.
→ Why did freezing fail for you?

**B19. What is experience replay?**
**Short**: Interleaving old-task data into new-task training.
*Detailed*: Keeps the old objective's gradient present so the optimizer can't move freely against it.
*Mini-R1*: 25% replay → 94.80% 2D retention.
⚠️ Assuming more replay is monotonically better — our 50% arm retained 3.80%.
→ Why did the 50% arm fail?

**B20. Interpolation vs extrapolation?**
**Short**: Inside vs outside the training support.
*Detailed*: Interpolation = unseen points within the trained hull (unseen pairs in [50,99]); extrapolation = beyond it (5-digit after training on 2–4).
*Mini-R1*: E1 100.00% (interpolation) vs E7 0.00% (extrapolation).
⚠️ Calling held-out-but-in-range performance "generalization to new problems" without qualifying the regime.
→ Which did your model actually do?

**B21. What is an ablation?**
**Short**: Remove or vary one component, hold everything else fixed, measure the delta.
*Mini-R1*: Task 16B varied only decoding on a **frozen** checkpoint — a textbook ablation.
⚠️ Changing two things and attributing the result to one (Task 22's original sin).
→ Give an example of a confounded comparison in your own work.

**B22. What is a confound?**
**Short**: An uncontrolled variable that could equally explain the result.
*Mini-R1*: Task 22 changed replay data **and** froze parameters; the −14.00 drop couldn't be attributed. Task 22B varied only trainability and showed freezing owned +30.38 of it.
⚠️ Not naming your own confounds before the interviewer does.
→ Is your replay-ratio curve confounded?

**B23. What is McNemar's test?**
**Short**: A paired test for two models on the same items, using only their disagreements.
*Equation*: $\chi^2=\frac{(b-c)^2}{b+c}$, $b$ = A-right/B-wrong, $c$ = A-wrong/B-right
*Detailed*: Items both get right or both get wrong carry no information about which is better.
*Mini-R1*: the right test for paired GSM8K comparisons; 15C reported p=0.0051.
⚠️ Using an unpaired two-proportion test on paired data — throws away the pairing and loses power.
→ Compute a 95% CI on 2.43% with n=1319.

**B24. Is 2.43% vs 2.35% significant?**
**Short**: No — that's one problem out of 1,319.
*Equation*: $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}\approx 2.43\%\pm0.83\%$
*Mini-R1*: exactly why GSM8K accuracy is excluded from the RCS selection metric.
⚠️ Reporting a point-estimate win without an interval.
→ What did you use for model selection instead?

**B25. What is strict vs fallback answer extraction?**
**Short**: Strict parses only `<answer>…</answer>`; fallback hunts for any number.
*Detailed*: Fallback inflates scores for verbose models — more numbers, more chances to match by luck.
*Mini-R1*: base model had **48.1% extraction rate** but **0.76% accuracy** — it emitted parseable numbers half the time and was almost never right.
⚠️ Comparing your strict number to someone else's fallback number.
→ Why does this make your numbers look worse?

**B26. What is step-level equation validity?**
**Short**: Fraction of generated equations that verify true in Python.
*Equation*: valid equations / total equations
*Detailed*: Separates "set up the wrong computation" from "computed it wrong" — a distinction final-answer accuracy hides.
*Mini-R1*: final model 31.64% greedy. Introduced in Task 22 as the primary metric.
⚠️ Reporting the rate alone — it's denominator-sensitive.
→ How did you handle the denominator problem?

**B27. Why report "valid equations per generation" alongside the rate?**
**Short**: Because a model emitting almost no equations can post a high rate on the few it does.
*Mini-R1*: Model B on the replay holdout emitted 0.11 equations/generation at 36.84% validity — a high rate on almost nothing.
⚠️ Ranking models on a rate without checking the denominator.
→ Did this change any of your conclusions? (No — the absolute measure agreed on GSM8K.)

**B28. What is total variation distance?**
**Short**: $\frac12\sum_i|p_i-q_i|$ — the maximum probability disagreement between two distributions.
*Mini-R1*: used to verify the synthetic replay corpus matched GSM8K's operator distribution to TVD 0.0063 and operand magnitude to 0.0212.
⚠️ Eyeballing histograms instead of computing a distance.
→ How did you *achieve* that match? (Iterative proportional fitting on sampling weights.)

**B29. Why is your context 512?**
**Short**: Attention memory is $O(B\cdot h\cdot T^2)$ per layer.
*Detailed*: At B=8, h=8, T=512: 16.8M values = 33.6 MB FP16 per layer, ~400 MB across 12 layers. Doubling T quadruples it. Our sequences are short anyway (compact traces ~47 completion tokens).
⚠️ Thinking context length costs scale linearly.
→ How would you support 2048 on the same card?

**B30. Why vocab 8000?**
**Short**: The embedding is $V\times d$; at 32k it would be 39% of the model.
*Detailed*: 8k → 4.1M params (9.7%). Keeps capacity in the computational layers. Trade-off: longer sequences per sentence.
*Mini-R1*: also enables uint16 storage (8000 < 65536).
⚠️ Assuming bigger vocab is always better.
→ What would change at vocab 32k?
---

# SECTION C — ADVANCED (30)

**C1. Prove attention needs positional information.**
**Short**: Attention is permutation-equivariant: $\text{Attn}(\Pi X)=\Pi\,\text{Attn}(X)$.
*Detailed*: Permuting rows of X permutes rows of Q, K, V. $ (\Pi Q)(\Pi K)^\top=\Pi QK^\top\Pi^\top$, softmax is row-wise so commutes, and multiplying by $\Pi V$ recovers $\Pi(\text{softmax}(QK^\top)V)$. So the *set* of outputs is unchanged, only reordered — position carries no information.
*Mini-R1*: matters acutely because subtraction isn't commutative.
⚠️ Hand-waving "it can't see order."
→ Does causal masking break equivariance? (Yes — the mask is position-dependent, which is why masked models still need explicit positions for *content* ordering.)

**C2. What is the softmax Jacobian, and why does it matter for $\sqrt{d_k}$?**
**Short**: $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$, i.e. $\operatorname{diag}(p)-pp^\top$.
*Detailed*: As $p\to$ one-hot, every entry $\to0$, so gradients through attention weights vanish. Unscaled scores at $d_k=64$ have std 8, producing near-one-hot distributions at initialisation.
⚠️ Knowing the scaling rule without the gradient consequence.
→ Would layer normalising Q and K achieve the same thing? (Roughly — QK-norm is a real modern alternative.)

**C3. Derive the 42,132,480 parameter count.**
**Short**: $12\times(12d^2+4d)+Vd+512d+2d$ with $d$=512, $V$=8000.
*Detailed*: block = 4·512² (attn) + 2·512·2048 (FFN) + 4·512 (2 LNs) = 3,147,776; ×12 = 37,773,312; + 4,096,000 (tok emb) + 262,144 (pos emb) + 1,024 (ln_f) + **0** (tied head) = **42,132,480**.
⚠️ Adding the LM head separately.
→ What fraction is embeddings? (10.3%)

**C4. What does tie_word_embeddings actually do, and what did it cost you?**
**Short**: `lm_head.weight` *is* `tok_emb.weight` — one tensor, two roles. It saved 4.1M params and broke Task 22.
*Detailed*: Freezing `tok_emb` also froze the output projection, leaving only `ln_f` (1,024 params) between the trainable blocks and a fixed head. The model couldn't learn to emit its stop token — p90 generation length pinned at the 256 cap — and looped. Unfreezing recovered +30.38 pts greedy equation validity.
*Mini-R1*: verified by object identity; PyTorch's `named_parameters()` **de-duplicates** shared tensors, so a naive freeze audit misses it entirely.
⚠️ Auditing frozen parameters by iterating `named_parameters()`.
→ How would you get parameter isolation *with* tied embeddings? (Untie first, or use adapters/LoRA which add new parameters instead of freezing old ones.)

**C5. Why did fixing distributional coverage not produce transfer?**
**Short**: Because coverage wasn't the binding constraint — semantic grounding is.
*Detailed*: Task 21B diagnosed that the substrate was subtraction-only over ranges excluding 40.85% of GSM8K's operands, while GSM8K is 37.40% multiplication. Task 22 built a corpus matched to TVD 0.0063 on operators. Transfer still didn't appear. The residual failure is mapping words to operations, which arithmetic supervision cannot teach.
⚠️ Presenting the coverage diagnosis as the *answer* — it was a hypothesis that we tested and that failed.
→ How would you test the semantic-grounding hypothesis directly?

**C6. Explain the RCS metric and why a geometric mean.**
**Short**: $(S_{\text{arith}}S_{\text{eq}}S_{\text{lang}})^{1/3}$ — the GM is zero if any factor is zero, so it structurally forbids trading one axis to zero.
*Detailed*: An arithmetic mean would let a model with superb equation validity and destroyed arithmetic score well. The GM is dominated by the smallest term, enforcing "non-terrible at all three."
*Mini-R1*: Model H = $(0.8650\times0.3164\times0.3485)^{1/3}=0.4569$.
⚠️ Using an arithmetic mean; or including GSM8K accuracy (noise-dominated) in the score.
→ Why exclude GSM8K accuracy?

**C7. Why exclude GSM8K accuracy from model selection?**
**Short**: Every arm sits between 0.91% and 2.43% — a handful of problems, inside noise.
*Detailed*: Including a noise-dominated term lets random variation drive selection. It's reported and carries a floor (≥1.5%) but doesn't rank.
⚠️ Optimising the headline metric because it's the headline metric.
→ Isn't that convenient given your GSM8K is low? (Fair challenge — answer: the metric was pre-registered *before* the runs, precisely to prevent post-hoc selection.)

**C8. Your replay curve is confounded. Explain how.**
**Short**: The arms differ in replay *composition*, not just share.
*Detailed*: 0% has no replay; 10%/25% replay 50/50 GSM8K-matched + Task 20 subtraction; **50% replays 100% GSM8K-matched arithmetic containing almost none of the probed subtraction regime**. So the 50% arm's collapse is composition, not dose. The apparent non-monotonicity is an artifact.
*Mini-R1*: stated in the Task 23 report before anyone asked.
⚠️ Presenting 0/10/25/50 as a clean dose-response curve.
→ What single run would deconfound it? (50% replay with the mixed block.)

**C9. Why does replaying the *wrong* distribution fail?**
**Short**: Replay protects only what it contains.
*Detailed*: Gradient interference is directional. Replaying arithmetic from distribution A constrains updates that would damage A; it says nothing about directions that damage B. Our 50% arm replayed GSM8K-matched arithmetic (small operands, mostly ×/+) while the probe measured subtraction over [50,99]/[101,199]/[1001,1999].
*Mini-R1*: 50% wrong → 3.80%; 25% right → 94.80%. The arm with the most replay retained the least.
⚠️ Treating replay as a generic regulariser.
→ What does this imply for replay-buffer design in continual learning generally?

**C10. What is exposure bias and how would you fix it?**
**Short**: Train/test conditioning mismatch; fixes include scheduled sampling and RL on the model's own samples.
*Detailed*: Scheduled sampling gradually replaces gold prefixes with model samples during training. RL (policy gradient) trains directly on self-generated sequences, eliminating the mismatch by construction — and our deterministic verifier would supply the reward.
*Mini-R1*: the mechanism behind CoT's failure; never mitigated.
⚠️ Claiming we tried scheduled sampling.
→ Why is RL's sparse reward a problem at 2% accuracy?

**C11. What would GRPO look like here, concretely?**
**Short**: Sample $k$ completions per prompt, verify each with Python, normalise rewards within the group to get advantages, do a clipped policy-gradient update with a KL penalty to the reference model.
*Equation*: $A_i=\frac{R_i-\text{mean}(R)}{\text{std}(R)}$; objective $\mathbb E[\min(r_tA_t,\text{clip}(r_t,1\pm\epsilon)A_t)]-\beta D_{KL}$
*Detailed*: No value network needed — that's GRPO's advantage over PPO and it matters at 42M where a critic is expensive.
*Mini-R1*: **never run.** The verifier exists; the RL does not.
⚠️ Implying any RL result.
→ What's the obstacle? (Sparse reward: at ~2% success almost every sample is negative.)

**C12. How would KL regularization help your biggest problem?**
**Short**: It would directly penalise drift from the pretrained policy, attacking the PPL regression.
*Detailed*: Every arm blew the PPL ≤ 35 gate, including the 0%-replay baseline at 43.88. A KL term to the reference model constrains how far fine-tuning can move the output distribution.
⚠️ Presenting this as something we did.
→ What's the trade-off? (Too strong a KL prevents learning the task at all.)

**C13. Why did validation loss plateau at 295M tokens?**
**Short**: Observed, not explained. Candidates: capacity limit, corpus diversity limit, or schedule effects.
*Detailed*: Chinchilla's ~20 tokens/param suggests ~840M for 42M params, so we were arguably *under*-trained by that heuristic and still plateaued — which points at capacity or corpus diversity rather than token count. **We ran no ablation, so this is a hypothesis.**
*Mini-R1*: best.pt at step 18,000 / 294.9M tokens; the final 41% of budget gave no measurable improvement.
⚠️ Asserting a cause.
→ How would you test it? (Train a 2× model on the same corpus; or the same model on a more diverse corpus.)

**C14. Your train loss is above your val loss. Explain.**
**Short**: Noisy single-step train value vs a 10M-token averaged validation pass.
*Detailed*: Averaging reduces variance. Final row: train 3.0807, val 2.8726. Not underfitting, not a broken split.
⚠️ Diagnosing it as a data leak in the *other* direction, or as underfitting.
→ What *would* overfitting look like? (Train falling while val rises — never observed here.)

**C15. Explain the harness padding artifact and its direction.**
**Short**: Left-padding without an attention mask, combined with learned absolute positions, shifts every token's position index.
*Detailed*: Training right-pads (real content starts at index 0); the evaluator left-pads and never passes the mask it builds. A bs=1 control (zero padding, N=200) measured +3.00 pts equation validity for Model E and +13.78 for Model A. So the artifact **flattered the losing arm** — under the clean condition the gap widened from −11.49 to −22.16.
⚠️ Assuming an artifact must weaken your conclusion. Measure the direction.
→ Why didn't you just fix it? (Would have invalidated comparability with seven exactly-reproduced Task 21B numbers mid-project; it's on the future-work list.)

**C16. How did you verify your evaluation harness was faithful?**
**Short**: It reproduced seven published Task 21B numbers exactly.
*Detailed*: Model A GSM8K 0.91%, retention 4.4/5.2/0.6, mean length 62.9, PPL 43.88; Model B 1.21%, retention 0.6/0.0/0.0, PPL 21.85. Exact matches across independent re-decoding.
⚠️ Trusting a new harness without a reproduction check.
→ What would you have concluded if they hadn't matched?

**C17. Why is your equation validity metric better than final-answer accuracy?**
**Short**: It separates two distinct failure modes and has ~3–5× the sample size.
*Detailed*: A wrong final answer could mean wrong setup or wrong computation. Extracting every emitted equation and verifying it in Python distinguishes them. And 1,319 problems yield ~3,770 equations — a much tighter measurement.
*Mini-R1*: introduced Task 22; primary metric thereafter.
⚠️ Forgetting it's a rate needing its absolute counterpart.
→ Can it be gamed? (Yes — by emitting fewer, easier equations. Hence valid-equations-per-generation.)

**C18. Why did adding 3-digit training *hurt* 2-digit accuracy?**
**Short**: Interference within arithmetic — unseen 2D fell 35.40% → 28.40%.
*Detailed*: Observed, not explained. Consistent with the model reallocating capacity, or with regime-specific patterns competing. We did no interpretability work to establish which.
⚠️ Claiming a capacity explanation as fact.
→ Does this bear on the replay result? (Yes — it shows interference is not unique to the GSM8K/arithmetic boundary.)

**C19. Answer length appears to be an independent difficulty axis. Evidence?**
**Short**: 19C OOD 3-digit scored 6.60% with 1–2 digit answers and **0.00%** with 3-digit answers.
*Detailed*: Same input regime, different output length, dramatically different accuracy. Suggests generating longer numerals is its own failure mode, separate from reading longer operands.
⚠️ Conflating input-length and output-length generalization.
→ How would you disentangle them? (Grid over input digits × answer digits.)

**C20. Why is E7 exactly 0.00% at Task 20 but 0.60% for the final model?**
**Short**: 3 of 500 vs 0 of 500 — both are floor-level; the difference is noise.
*Detailed*: Don't present +0.60 as improvement. A binomial CI on 3/500 comfortably includes 0.
⚠️ Cherry-picking a "gain" from three problems.
→ What sample size would you need to detect a real 1% effect?

**C21. What does digit-level tokenization buy you, and what doesn't it?**
**Short**: It removes an obstacle (inconsistent numeral fragmentation); it does not supply an algorithm.
*Detailed*: Three separable things — tokenization (can the model address digits? yes), representation (does it encode digit place? unverified), computation (can it execute borrow in order? evidently not outside trained regimes). Same tokenizer gives 100% at 2 digits and 0.00% at 5.
⚠️ Claiming digit tokenization "enables arithmetic."
→ How would you test the representation question? (Probe for digit-position information in the residual stream.)

**C22. Could your OOD failure be a positional-encoding limitation?**
**Short**: Plausible hypothesis; **we did not test it.**
*Detailed*: Learned absolute positions are trained only up to indices actually seen, and longer operands shift downstream token positions. That's a coherent story — and so are missing carry circuits, or regime-specific pattern memorisation. Distinguishing them needs activation patching or probing, which we didn't do.
⚠️ Asserting a mechanism. This is the single most tempting overclaim in the project.
→ What experiment would settle it?

**C23. Why is inference memory ~8× smaller than training?**
**Short**: Training holds FP32 master weights + gradients + two Adam moments (~16 bytes/param) plus retained activations; inference holds FP16 weights (~2 bytes/param) and one activation slice.
*Detailed*: 42.13M params → 674 MB static training state vs 84 MB FP16 weights.
*Mini-R1*: peak training 3,409.7 MB; inference under 1 GB.
⚠️ Sizing from parameter count alone.
→ What dominates the remaining training memory? (Activations, especially the $T^2$ attention matrices.)

**C24. Explain allocated vs reserved CUDA memory.**
**Short**: Allocated = live tensors; reserved = what PyTorch's caching allocator holds from the driver.
*Detailed*: PyTorch caches freed blocks because `cudaMalloc`/`cudaFree` are expensive and synchronising, so reserved ≥ allocated and doesn't shrink on free. You can OOM with large `reserved − allocated` if the free space is fragmented rather than contiguous.
*Mini-R1*: 3,409.7 MB is *allocated*.
⚠️ Quoting one number without saying which.
→ How do you reduce fragmentation? (Fixed batch shapes; `empty_cache()`; expandable segments.)

**C25. Why were atomic checkpoint writes necessary?**
**Short**: A crash mid-`torch.save` of a 509 MB file leaves a corrupt checkpoint — and if it's `best.pt`, the run is lost.
*Detailed*: `torch.save(state, path+".tmp")` then `os.replace(tmp, path)` — a single filesystem operation, so the name points at either the old or the new file, never a partial one.
⚠️ Believing `torch.save` is atomic.
→ What's the caveat? (`os.replace` is atomic only within one filesystem.)

**C26. What must a checkpoint contain for a scientifically valid resume?**
**Short**: Model, optimizer moments, scheduler position, GradScaler scale, all four RNG states, and the dataset cursor.
*Detailed*: Missing optimizer state → Adam restarts with zero moments and bias correction misfires. Missing RNG → different dropout/shuffle stream. Missing cursor → you re-read or skip data and the token budget becomes a lie.
*Mini-R1*: `best.pt` contains `torch/cuda/numpy/python` RNG states, `dataset_state`, `tokens_seen`, `global_step`.
⚠️ Saving only `model.state_dict()`.
→ Why isn't "the loss looked similar after resume" sufficient?

**C27. Why isn't "similar loss after resume" sufficient?**
**Short**: Similar loss is consistent with a different data order, a different subset, or a different LR trajectory — any of which makes it a different experiment.
*Detailed*: The stronger criterion is bitwise parity: N steps + checkpoint + M steps must equal N+M uninterrupted. Then resume is a no-op w.r.t. the science.
⚠️ Claiming we ran a specific parity experiment — describe it as the *design target*.
→ What else does bitwise parity require? (Deterministic cuDNN kernels, fixed shapes, same hardware.)

**C28. Why uint16 shards?**
**Short**: vocab 8,000 < 65,536, so 2 bytes/token — half of uint32.
*Detailed*: 500M tokens = 1 GB. Combined with `np.memmap`, the OS page cache serves batches and RAM stays flat.
⚠️ Using int64 by default (4× the storage and I/O).
→ What changes at vocab 100k? (uint32 required.)

**C29. Why EOS-separated contiguous packing?**
**Short**: 100% token density; naive per-document truncation wasted 58.4%.
*Detailed*: Concatenate documents with EOS between them, cut into fixed 512-token windows. EOS teaches document boundaries so the model doesn't learn spurious cross-document continuations.
⚠️ Packing without a separator — the model learns to run documents together.
→ Any downside? (Sequences can straddle documents; the model must learn EOS means "reset".)

**C30. If you had to defend one contribution, which?**
**Short**: Composition-over-dose in replay.
*Detailed*: It's counter-intuitive (the arm with the *most* replay retained the *least*), quantitatively large (3.80% vs 94.80%), mechanistically legible, and generalises beyond this model: a replay buffer sampled from the wrong region protects nothing regardless of size.
⚠️ Claiming the architecture or the model itself is the contribution.
→ What's the caveat? (Ratio and composition are confounded in my series; the claim is about the *presence* of the target distribution, not the optimal ratio.)

---

# SECTION D — RESEARCH DEFENSE (20)

**D1. Is this actually research or just engineering?**
**Short**: Both, but the contribution is research-methodological.
*Detailed*: Hypotheses stated in advance, matched controls, pre-registered metrics, four disconfirmed hypotheses reported in full, one self-retraction. The engineering (from-scratch transformer, 6 GB pretraining) is the substrate.
→ Name your most rigorous experiment. (Task 22B — one independent variable.)

**D2. What is your strongest result?**
**Short**: Targeted replay restoring retention 4.40% → 94.80%, with the composition finding.
→ What makes it strong? (Matched controls; large effect; direction contradicts the naive "more replay is better".)

**D3. What is your weakest result?**
**Short**: Everything GSM8K. 2.43% ≈ 2.35% baseline, never exceeded ~2.4% in any configuration.
→ Why report it at all? (It's the original research question; hiding a null is the failure mode I most wanted to avoid.)

**D4. Which of your conclusions would you bet against?**
**Short**: That 25% is a meaningful ratio. It's confounded with composition.
→ What would change your mind? (The 50%-mixed-block run.)

**D5. What did you get wrong and have to retract?**
**Short**: Task 22 blamed my replay corpus for runaway generation; Task 22B proved it was parameter freezing.
*Detailed*: I wrote a mechanistic story about nested prompt expressions teaching additive chains. The control run — same corpus, unfrozen — produced 2.74 equations/generation instead of 5.12 and terminated normally. The corpus was fine.
→ How did you handle it? (Left the original text with an explicit retraction rather than editing history.)

**D6. Your Task 21A report says forgetting was "negligible." Defend that.**
**Short**: I can't — it's wrong, and I flag it.
*Detailed*: PPL went 17.68 → 28.87 (+63%), and the arithmetic collapsed to 3.6/0.6/0.4 from 100/99.8/67.2. The sentence looked only at language and generalised to "forgetting." It's the clearest overclaim in the project.
→ Why did it happen? (Measuring one axis and concluding about a phenomenon that spans several.)

**D7. How do you know your arithmetic results aren't leakage too?**
**Short**: Because we audited and rebuilt. Post-19B suites are explicitly disjoint held-out sets.
*Detailed*: E1–E7 are fixed generated files with declared operand ranges and recorded SHA256s; the OOD suites are by construction outside the training support.
→ Could there be residual overlap? (Within-regime, unseen *pairs* can still resemble seen ones — which is precisely why I call E1 "interpolation," not "generalization.")

**D8. Why should I trust your numbers?**
**Short**: Every claim traces to a JSON artifact, and the 37 headline numbers were programmatically re-verified against those artifacts.
*Detailed*: Checkpoints, corpora and tokenizer all carry SHA256s; the evaluator reproduced seven prior published numbers exactly.
→ What can't you verify? (Anything about mechanism — no interpretability was done.)

**D9. Why is your PPL so much worse than base?**
**Short**: 17.68 → 50.74. Fine-tuning on narrow arithmetic-heavy data costs general language modelling, monotonically with replay share.
*Detailed*: 43.88 (0%) → 47.90 (10%) → 50.74 (25%) → 60.02 (50%). This is the biggest unsolved deficit; every arm failed my own PPL gate including the baseline.
→ How would you fix it? (Language replay from the pretraining shards; or a KL penalty to the reference policy.)

**D10. Your PPL gate failed for every model. Isn't the gate broken?**
**Short**: Yes, and that's my error.
*Detailed*: I set G2 at PPL ≤ 35 anchored on Model E's 30.33 — but E was the *degenerate frozen* model that never produced usable output. The gate was below the achievable range for any full-parameter fine-tune. A better design is a gate relative to the strongest baseline.
→ Does it invalidate the selection? (No — the fallback rule was also pre-registered, and RCS ranks sensibly.)

**D11. Single seed. How seriously should I take these numbers?**
**Short**: Take the large effects seriously and the small ones not at all.
*Detailed*: +30.38 (freezing) and +90.4 points (retention) are far beyond plausible seed variance. The 2.43% vs 2.35% GSM8K difference is not — and I explicitly excluded it from selection for that reason.
→ How many seeds would you want? (3–5 for the retention claims; GSM8K would need far more or a bigger effect.)

**D12. Why 650 steps for every SFT run?**
**Short**: To hold the training budget identical across every arm from Task 21B onward, so arms are comparable.
*Detailed*: Comparability across six models mattered more than per-arm optimality.
→ Did you check for under/over-training? (Best checkpoints landed at step 600–650, so arms were still improving — a longer budget might help, and I didn't test it.)

**D13. Your GSM8K prompt format is very specific. Isn't that brittle?**
**Short**: Yes. AQuA is 0.00% with 0.0% full-format compliance — the model doesn't transfer to a different prompt shape at all.
*Detailed*: A real limitation, not a measurement artifact.
→ How would you address it? (Format augmentation during SFT.)

**D14. How is this different from just training on arithmetic?**
**Short**: The question wasn't "can it do arithmetic" but "does arithmetic competence survive and transfer."
*Detailed*: The interesting results are all about interaction — forgetting, interference, replay composition — not about arithmetic accuracy per se.
→ What would a pure-arithmetic project have missed? (The entire forgetting/replay finding.)

**D15. Why did you use AI coding assistance, and does that undermine the work?**
**Short**: I used it for implementation velocity; the experimental design, the diagnoses and the corrections are the intellectual content and I can defend every one.
*Detailed*: Ask me to derive the parameter count, the $\sqrt{d_k}$ argument, or the gradient-accumulation equivalence — I'll do it on the board. The leakage audit and the tied-embedding diagnosis are the parts that mattered, and they came from reading numbers that looked wrong.
→ What would you have done differently unassisted? (Slower; probably fewer arms — and I'd likely have missed the resume bug I caught by asserting behaviour rather than trusting the patch.)

**D16. What's the one experiment you most regret not running?**
**Short**: 50% replay with the mixed composition block. One run, and it deconfounds the headline curve.
→ Second? (Language replay from pretraining shards — it targets the biggest deficit.)

**D17. Is a 42M model even worth studying?**
**Short**: For this question, yes — precisely because arithmetic isn't latent at this scale.
*Detailed*: At 7B you can't separate taught ability from pretrained ability. The forgetting and replay findings are also scale-general in kind, if not in magnitude.
→ Would your findings hold at 7B? (Unknown. I ran no scaling comparison and won't speculate about magnitudes.)

**D18. What would falsify your replay conclusion?**
**Short**: A 50% mixed-composition arm that also collapses, or a 25% arm that fails to replicate across seeds.
*Detailed*: The claim is "replaying the target distribution at 10–25% preserves it." If retention were driven by ratio rather than presence-of-regime, the mixed 50% arm would collapse too.
→ Good — that's a falsifiable claim, which is the point.

**D19. Why is your model called Mini-R1 if you never did RL?**
**Short**: Aspirational naming from project start, and it overpromises.
*Detailed*: The pipeline shares the shape — verified traces, deterministic checker, RL as intended final stage — but the RL stage never happened and the reasoning results are negative.
→ What would you call it now? (Something about arithmetic retention.)

**D20. Sell me on this project in one sentence.**
**Short**: "A from-scratch 42M transformer used to show that verified arithmetic competence doesn't transfer to word problems, that fine-tuning destroys it, and that you can prevent that destruction by replaying the right distribution — with every arithmetic claim checked by Python."
→ And the caveat? (GSM8K stayed at ~2%; this is a forgetting study, not a reasoning model.)

---

# SECTION E — DEBUGGING (20)

**E1. Your training loss is suspiciously near zero. What's your first hypothesis?**
**Short**: Broken causal mask — the model can see the answer.
*Detailed*: Also check for label/input misalignment (off-by-one in the shift) or supervising positions you meant to mask.
⚠️ Celebrating.
→ How would you confirm? (Generate — output will be incoherent despite low loss.)

**E2. Loss is NaN after a few hundred steps under AMP. Causes?**
**Short**: FP16 overflow, LR too high, or a missing/incorrect GradScaler.
*Detailed*: Check the scaler is skipping steps; verify `unscale_` precedes clipping; look for `exp` without max-subtraction; check for division by a near-zero denominator.
*Mini-R1*: 0 skipped steps across 500M tokens — the pipeline was verified in Task 8.
→ Why does the scaler halve on inf?

**E3. OOM at batch 8 but batch 4 works, and `nvidia-smi` shows free memory. Why?**
**Short**: Fragmentation — free memory isn't contiguous.
*Detailed*: PyTorch's caching allocator holds reserved blocks; a large contiguous request can fail while `reserved − allocated` is large. Fixed batch shapes, `empty_cache()`, or expandable segments help.
→ Which memory number would you check first? (Both allocated and reserved, plus the fragmentation delta.)

**E4. A resumed run diverges from the uninterrupted one. Where do you look?**
**Short**: Optimizer moments, scheduler position, RNG states, and the dataset cursor.
*Detailed*: In that order. Zero-initialised Adam moments after resume are the classic cause; second is a scheduler restarting at step 0.
*Mini-R1*: all of these are in the checkpoint.
→ What if all are restored and it still diverges? (Non-deterministic CUDA reductions; different hardware.)

**E5. Your model generates the same token forever. Diagnose.**
**Short**: Greedy decoding trapped in a fixed point of the decoding dynamics.
*Detailed*: Confirm by sampling with temperature — if it breaks, it's a decoding pathology; if not, it's a representation problem.
*Mini-R1*: exactly the Task 16B experiment. Repetition 82.6% → 11.45% with T=0.6/top_p=0.95/rep_pen=1.2 — **and accuracy stayed below baseline**, so it was both.
→ What did that prove and not prove?

**E6. Generation never emits EOS and always hits max tokens. Hypotheses?**
**Short**: The model can't produce the stop token — check whether the output head is trainable.
*Detailed*: Also check EOS appears in training targets (`add_eos=True`), and that the stop-token id matches what the sampler checks.
*Mini-R1*: **this is literally the Task 22 bug** — tied embeddings meant freezing `tok_emb` froze `lm_head`; p90 length pinned at 256.
→ How would you verify the head is frozen? (Object identity — `named_parameters()` de-duplicates.)

**E7. Your eval accuracy is much higher than expected. First move?**
**Short**: Audit for train/eval overlap before believing it.
*Detailed*: Key every eval item, search the full training corpus **including nested content**, stratify results into seen/unseen, report both.
*Mini-R1*: 24.28% → 111/313 overlap → 2.48% truly unseen. 82.14% on seen vs 7.41% on the same numbers seen only inside traces.
→ What does the seen/unseen split tell you beyond the aggregate?

**E8. Two models score identically on aggregate but you suspect a difference. What test?**
**Short**: McNemar on the paired disagreements.
*Equation*: $\chi^2=(b-c)^2/(b+c)$
*Detailed*: Items both get right or both wrong are uninformative; only $b$ and $c$ carry signal.
→ Why not a two-proportion z-test? (Discards the pairing, loses power.)

**E9. A metric looks great but you're suspicious. What's your check?**
**Short**: Look at the denominator.
*Detailed*: Rates are denominator-sensitive. A model emitting almost no equations can post a high validity rate.
*Mini-R1*: Model B — 0.11 equations/generation at 36.84% validity. Fixed by adding valid-equations-per-generation.
→ Did it change your GSM8K conclusion? (No — both measures agreed there.)

**E10. Your evaluator re-processes a model you told it to skip. Diagnose.**
**Short**: The resume code path isn't running — verify the flag actually loads state.
*Detailed*: In my case a string-replacement patch silently no-op'd on a newline mismatch, so only the *save* half of resume existed; `results` started empty every run.
*Mini-R1*: cost one killed job during Task 22B.
⚠️ Assuming a patch applied because it printed "patched".
→ Prevention? (Assert observable behaviour — e.g. that a "skipping" line appears.)

**E11. Padding is different between training and evaluation. Does it matter?**
**Short**: With learned absolute positions, enormously.
*Detailed*: Training right-pads (content starts at index 0); left-padding at eval shifts every position index. Measure it rather than assume — bs=1 control gave +3.00 and +13.78 points for two models.
→ Would RoPE avoid this? (Largely, being relative — but you still need the attention mask.)

**E12. Your synthetic dataset generator produces wrong labels. How do you catch it?**
**Short**: Independent second verification, written separately from the generator.
*Detailed*: Re-evaluate every equation and every prompt expression with a fresh checker.
*Mini-R1*: the replay corpus passed an independent third check — 25,762 train + 1,592 dev equations, **0 failures**.
→ Why not trust the generator? (It's the same code path that could be wrong.)

**E13. Throughput drops halfway through training. Causes?**
**Short**: Thermal throttling, growing sequence lengths, memory fragmentation, or eval passes inside the timer.
*Mini-R1*: sustained 17,316.5 tok/s over 8.02 h with no collapse.
→ How would you isolate? (Log per-step time and GPU clocks separately.)

**E14. Your model's format compliance is 0% on one benchmark and 98% on another. Explain.**
**Short**: Prompt-shape mismatch — it never trained on that shape.
*Mini-R1*: AQuA 0.0% full-format vs GSM8K 98.2%. On MCQ prompts it opens `<think>` (97.2%) but rarely closes an `<answer>` (26.0%).
→ Fix? (Format augmentation during SFT.)

**E15. A hyperparameter change improves your metric. How do you know it's real?**
**Short**: Effect size vs seed variance, and whether it replicates.
*Detailed*: With one seed, only large effects are interpretable.
*Mini-R1*: +0.38% transfer in 21A "improved" — then 21B measured −0.30% with a better design. Two runs, opposite signs, both tiny: noise.
→ What's the lesson? (A positive point estimate is not a positive result.)

**E16. Your checkpoint is 3× your model size. Why?**
**Short**: Adam's two moment buffers plus FP32 master weights.
*Detailed*: 42.13M params ≈ 169 MB FP32; ×3 for $m$, $v$ ≈ 509 MB.
*Mini-R1*: `best.pt` is 508,892,602 bytes; stripping optimizer state gives ~169 MB.
→ When would you keep the optimizer state? (Whenever you might resume.)

**E17. Two reports give different numbers for the same thing. How do you resolve it?**
**Short**: Go to the artifact — checkpoint metadata or JSON — not the prose.
*Mini-R1*: `best.pt` step-18,000 val loss (2.8726140, from checkpoint metadata) vs the step-18,311 milestone row (3.0118, from the logging pass). Different evaluation events; the checkpoint field is authoritative for the checkpoint.
→ What do you do with the discrepancy? (Report it as a logging inconsistency; don't silently pick one.)

**E18. Generation is far slower than you expected. Why?**
**Short**: No KV cache — every step recomputes attention over the full growing context.
*Detailed*: That's $O(T^2)$ per step and $O(T^3)$ for a full sequence. Our greedy GSM8K pass took 5,514s for 1,319 problems because the model rarely terminated early.
*Mini-R1*: KV caching was never implemented.
→ What speedup would a KV cache give? (Roughly an order of magnitude here.)

**E19. Your loss curve is flat from step 0. Checklist?**
**Short**: LR zero or scheduler misconfigured; parameters not in the optimizer; `requires_grad=False`; missing `.backward()`; gradients zeroed before stepping.
*Mini-R1*: the Task 22 config drove trainability from regex patterns — a wrong pattern would silently freeze everything. We printed trainable/frozen counts every run for exactly this reason.
→ First diagnostic? (Print `sum(p.numel() for p in model.parameters() if p.requires_grad)`.)

**E20. How do you sanity-check a from-scratch transformer before a long run?**
**Short**: Overfit a single batch to near-zero loss.
*Detailed*: If it can't memorise 8 sequences, something is structurally broken. Then check shapes, gradient flow to every parameter, causal-mask correctness (shuffle the future and confirm outputs don't change), and AMP parity against FP32.
*Mini-R1*: `scripts/one_batch_train.py`, plus a 26-file test suite.
→ What does a *failed* single-batch overfit most often indicate? (Broken gradient path or misaligned labels.)

---

# SECTION F — "WHY DIDN'T YOU DO X?" (20)

**F1. Why not use a pretrained model?** — The core rule was from-scratch: no pretrained causal weights, no framework attention. The point was to understand and control every component; using GPT-2 would make "what did arithmetic supervision add" unanswerable.

**F2. Why not a bigger model?** — 6 GB VRAM. And the question is sharper at a scale where arithmetic must be taught rather than being already latent.

**F3. Why not more pretraining tokens?** — Validation loss plateaued at ~295M of 500M; the last 41% gave no measurable improvement. More tokens looked unlikely to help without more capacity or a more diverse corpus.

**F4. Why not RoPE instead of learned absolute positions?** — Simplicity, and we never exceed the 512 context where RoPE's extrapolation benefit appears. In hindsight it would have mitigated the padding artifact, since relative positions are shift-invariant.

**F5. Why not BF16?** — Reliable support wasn't established on this consumer Ada laptop stack; FP16+GradScaler was empirically verified (0 skipped steps, 0.1085% error vs FP32). I'd prefer BF16 given confirmed support.

**F6. Why not untie the embeddings?** — Tying saves 4.1M params, 9.7% of the model — a good trade at 42M. It did cause the Task 22 failure, but the right fix there is not to freeze tied parameters, not to spend 10% of the model.

**F7. Why not LoRA or adapters instead of freezing?** — That's the correct alternative and I didn't run it. Adapters add *new* trainable parameters rather than freezing existing ones, so they'd sidestep the tied-embedding trap entirely. It's on the future-work list.

**F8. Why not EWC for forgetting?** — Replay was simpler and I had verified data to replay. EWC needs a Fisher-information estimate over the old task, which is more machinery. A direct EWC-vs-replay comparison is untested.

**F9. Why not run RL/GRPO?** — Scope, and a real design problem: at ~2% success rate the reward is extremely sparse, so exploration becomes binding. I have the verifier — the hard part of reward design — but not a solution to sparsity.

**F10. Why not more seeds?** — Deadline and single-GPU compute. I mitigated by excluding noise-dominated metrics from selection, but it's a genuine limitation.

**F11. Why not ablate the 60:40 corpus ratio?** — It would have cost a full pretraining run (8 hours) per arm, before the actual research question started. I can't claim it's optimal.

**F12. Why not re-run the LR sweep at full budget?** — 30,518 steps per arm. The sweep decided at step 1,000, which is a real limitation — a higher LR can win early and lose late.

**F13. Why not fix the attention-mask bug immediately?** — Fixing it mid-project would have broken comparability with seven exactly-reproduced Task 21B numbers. I measured its size and direction instead, showed it flattered the *losing* arm, and put the fix on the future-work list.

**F14. Why not implement KV caching?** — Pure inference optimisation with no effect on results. It would have saved hours of eval time; it wouldn't change a single number.

**F15. Why not FastAPI as planned?** — It never got built. `web/server.py` is stdlib `http.server`. FastAPI remains the deployment goal in the project context document.

**F16. Why not evaluate on more benchmarks?** — GSM8K, SVAMP and AQuA covered the reasoning axis and all three sat near the floor. Adding benchmarks would have added measurement, not insight, at ~2% accuracy.

**F17. Why not train longer than 650 SFT steps?** — Comparability. Every arm from Task 21B onward used 650 so the six models are directly comparable. Best checkpoints landed at 600–650, so a longer budget might help — untested.

**F18. Why not use an LLM judge for evaluation?** — Arithmetic is deterministic. A Python verifier is exact, free, reproducible and unfoolable; a judge would add noise and circularity.

**F19. Why not do interpretability to explain the OOD failure?** — It's the most interesting open question and it's a research programme, not an afternoon. It's why I say "regime-local behaviour" and refuse to name a mechanism.

**F20. Why not just prompt a large model instead?** — That answers a different question. GPT-4 solves GSM8K; it tells you nothing about whether *teaching* arithmetic to a small model transfers, or about what happens to a learned skill under fine-tuning. The small scale is the experimental apparatus, not a limitation to apologise for.
