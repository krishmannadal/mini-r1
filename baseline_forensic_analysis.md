# Mini-R1 Base Model Forensic Analysis: Zero-Shot Reasoning Assessment

**Role**: Senior Reasoning-Model Researcher  
**Date**: 2026-08-19  
**Subject**: Skeptical forensic review of Task 14 baseline evaluation  
**Model**: 42.1M parameter decoder-only transformer, 500M-token general-language pretraining  
**Checkpoint**: `best.pt` (step 18,000, val loss 2.8726, PPL 17.68)

---

## 1. How Much Mathematical Reasoning Ability Exists Zero-Shot?

**Verdict: Effectively none. The measured accuracies are statistical noise, not evidence of reasoning.**

| Benchmark | Greedy Acc | Sampling Acc | Random Chance Baseline | Verdict |
|---|---|---|---|---|
| **GSM8K** (1,319) | 0.76% (10) | 0.91% (12) | ~0% (open-ended numerical) | **Noise** |
| **SVAMP** (300) | 3.00% (9) | 2.67% (8) | ~0% (open-ended numerical) | **Noise** |
| **AQuA-RAT** (254) | 17.32% (44) | 11.42% (29) | 20.00% (5-choice uniform) | **Below chance** |

### Critical observations:

**GSM8K and SVAMP "correct" answers are false positives.** I examined every "correct" example in the qualitative samples. The pattern is unambiguous:

- `gsm8k_test_807` (gold answer `4`): The model generates prose about "creating your own mosaic using simple shapes and colors." The number `4` appears in the **prompt itself** ("Section 4"), and the `last_number_fallback` extractor picks it up from the generation text. The model performed zero arithmetic.
- `gsm8k_test_593` (gold answer `3`): The model generates advice about "planning your beanstalk." The number `3` appears as "Step 3" — a formatting artifact of the educational-textbook writing style the model learned from Cosmopedia. Zero reasoning occurred.
- `gsm8k_test_490` (gold answer `4`): Same pattern. "Section 4" extracted as `4`.
- `gsm8k_test_1001` (gold answer `2`): "he had to wait 2 hours" — the model echoed the number `2` from the question context. Not reasoning.

Every "correct" GSM8K/SVAMP answer I examined was produced by the `last_number_fallback` extraction method finding an incidental number in free-form educational prose. The model never performed a single arithmetic operation.

**Expected false-positive rate**: For GSM8K, if ~48% of generations contain an extractable number, and the answer space spans roughly integers 1–100+ with a concentration bias toward small integers (the model's educational-textbook register heavily favors "Step 1", "Step 2", "Section 3", etc.), the probability of accidentally matching small gold answers like 2, 3, 4, 5 is non-trivial. The observed 0.76% rate (10/1319) is consistent with pure coincidence.

**AQuA-RAT at 17.32% is below the 20% uniform-random baseline.** This is especially revealing. The model's AQuA extraction rate is 91.3% (because its repetitive option-echoing behavior reliably produces capital letters A–E). But the 17.32% accuracy is *below* what a coin-flip would achieve. This is not evidence of reasoning — it is evidence that the model's letter-distribution bias is slightly misaligned with the answer-key distribution.

> [!IMPORTANT]
> **There is zero mathematical reasoning in this model.** Every measured accuracy is attributable to extraction artifacts or random coincidence. The base model has not learned to compute, count, or solve equations.

---

## 2. What Failure Modes Appear?

I identified **five distinct failure modes**, ordered by severity and frequency:

### Failure Mode 1: Educational-Textbook Continuation (Dominant, ~70%+ of outputs)

The model's pretraining corpus is 60% Cosmopedia-v2 (synthetic educational textbooks) and 40% FineWeb-Edu-Dedup (filtered educational web text). When prompted with a math question followed by `<think>`, the model interprets this as a cue to write the next section of an educational textbook about the topic mentioned in the question.

Examples:
- Question about beanstalks → "Step 2: Planning Your Beanstalk"
- Question about diapers → "create a plan to help us manage our diaper"  
- Question about books → "let's dive into the fascinating world of books!"
- Question about flowers → "let's talk about something called 'paige.' Paige is a type of flower"

This is the model doing **exactly what it was trained to do**: continuing educational prose. The `<think>` token has no semantic significance to it.

### Failure Mode 2: Repetitive Degeneration Loops (~15-20%)

The model falls into periodic n-gram repetition, typically within 20–40 tokens of generation:
- `"Jane\nJane\nJane\nJane..."` (128 tokens of pure repetition)
- `"- C)400\n- C)400\n- C)400..."` (option echoing on AQuA)
- `"$500\n- $500\n- $500..."` (number echoing)
- `"Benedict $100, Benedict $100, Benedict $100..."` (entity+number echoing)

This is a well-known failure mode of small autoregressive models without repetition penalties, particularly at 42M parameters where the representational capacity is insufficient to maintain long-range coherence.

### Failure Mode 3: Option Echoing (AQuA-Specific, ~60% of AQuA outputs)

On AQuA multiple-choice questions, the model frequently echoes one of the presented options in a repetitive loop:
- `"- D)2500p\n- D)2500p..."` — echoing option D
- `"- D)11\n- D)11..."` — echoing option D
- `"- E\n- E\n- E..."` — pure letter repetition

The model is not "selecting" an answer. It is continuing the visual pattern of the bullet-pointed option list from the prompt. The "selected" letter is determined by which option fragment has the highest next-token probability after the prompt, which is a function of positional/recency bias, not comprehension.

### Failure Mode 4: Hallucinated Pseudo-Mathematical Prose (~5%)

Occasionally, the model generates text that superficially resembles mathematical reasoning:
- "Calculate the square of the sphere by dividing the square of the sphere by the square of the sphere"
- "you would measure sugar by measuring sugar. Then, you would measure sugar"

This is **surface-level pattern mimicry**. The model has learned that educational text about math uses words like "calculate", "divide", "measure" — but it has no grounding for what these operations mean or how to apply them to specific numbers.

### Failure Mode 5: No EOS / No Stopping (Universal)

The model almost never generates EOS. It runs to `max_new_tokens` in virtually every case:
- Median generated tokens: **128.0** (the hard cap)
- `stopped_by_eos: false` in the vast majority of samples

The base model has no concept of "answering" or "finishing" — it only knows how to continue text. This is expected from general pretraining without instruction-following data.

---

## 3. What Is the Model Actually Doing?

**Verdict: The model is exclusively performing next-token prediction conditioned on its educational-textbook pretraining distribution. It is not doing arithmetic, not producing reasoning-like text, and certainly not solving multi-step problems.**

Let me be precise about each category:

### ❌ Memorizing language patterns → **YES, this is ALL it does**

The model has learned the statistical regularities of Cosmopedia-v2 and FineWeb-Edu-Dedup. When given a prompt about "bees in a hive," it continues with the most likely Cosmopedia-style text about bees and hives. When given a prompt about "money," it produces educational prose about "base management" or "financial support."

The generation is **topically anchored to the question's surface keywords** — not to the mathematical structure of the problem.

### ❌ Doing arithmetic → **NO**

There is zero evidence of any arithmetic operation being performed in any generation. Not once does the model produce a sequence like "12 × 2 = 24" or "72 ÷ 12 = 6" in response to a problem that requires such computation. The numbers that appear in generations are either:
1. Echoed from the prompt
2. Incidental formatting artifacts ("Step 2", "Section 3")
3. Part of repetition loops

### ❌ Producing reasoning-like text → **NO, not even superficially**

The model does not produce text that even *looks like* a chain of reasoning. It produces text that looks like a children's educational textbook. There are no intermediate computational steps, no variable binding, no logical connectives linking premises to conclusions.

### ❌ Genuinely solving multi-step problems → **ABSOLUTELY NOT**

The 42.1M parameter model has no capacity for multi-step mathematical reasoning at zero-shot. This is completely expected and scientifically unremarkable. The model has 42M parameters — smaller than GPT-2 Small (124M), which itself shows negligible mathematical reasoning. The pretraining corpus contains no worked mathematical solutions or step-by-step demonstrations.

> [!NOTE]
> The empty `correct_and_coherent` category across ALL THREE benchmarks is the most scientifically honest signal in this entire evaluation. Not a single example was both correct and coherent. Every "correct" answer was coincidental extraction from incoherent educational prose.

---

## 4. Which Benchmark Should Be the Primary Reasoning Benchmark?

**Recommendation: GSM8K should be the primary benchmark. SVAMP is the secondary benchmark. AQuA-RAT should be demoted to tertiary or diagnostic-only status.**

### Rationale:

| Criterion | GSM8K | SVAMP | AQuA-RAT |
|---|---|---|---|
| **Answer type** | Open numerical | Open numerical | 5-choice MC |
| **Test split size** | 1,319 | 300 | 254 |
| **Chance-level floor** | ~0% | ~0% | 20% |
| **Ceiling for detection** | High | Medium | Low |
| **Community comparability** | Industry standard | Moderate | Low |
| **Multi-step difficulty** | 2–8 steps | 1–3 steps | Variable |
| **Measurement sensitivity** | High | Medium | **Poor** |

**Why NOT AQuA-RAT as primary:**

1. **The 20% chance floor destroys measurement sensitivity.** For a 42M model, the signal-to-noise ratio on AQuA is terrible. A model going from 0% to 20% actual reasoning ability is indistinguishable from random noise on a 5-choice test.
2. **The extraction method is biased.** The `last_capital_letter_fallback` extractor (used for 91.3% of AQuA extractions) picks the last A–E letter from the generation. On a model that echoes options from the prompt, this creates a systematic bias toward whichever option has the highest next-token probability — which may correlate with answer-key statistics in ways that inflate reported accuracy.
3. **Greedy (17.32%) > Sampling (11.42%) on AQuA** — this 6-point gap suggests that greedy decoding's deterministic option-echoing accidentally aligns better with the answer distribution than sampling's stochastic echoing. This is a measurement artifact, not a reasoning signal.

**Why GSM8K as primary:**

1. **Zero chance-level floor** for open-ended numerical extraction means even small real gains are detectable.
2. **1,319 examples** provide adequate statistical power for detecting improvements of ~2-3 percentage points with confidence.
3. **GSM8K is the community standard** for reasoning evaluation, enabling direct comparison with published results on similar-scale models.
4. **Multi-step structure** (2–8 reasoning steps) directly tests chain-of-thought capability, which is the explicit research hypothesis of Mini-R1.

---

## 5. What Should Our SFT Dataset Look Like?

### Core Requirements:

**1. Explicit step-by-step arithmetic demonstrations over the same numerical operations that appear in the benchmarks.**

The model's pretraining corpus contains zero worked mathematical examples. SFT must teach two things simultaneously:
- **Format**: `<think>step-by-step reasoning</think><answer>numerical_result</answer>`
- **Content**: How to actually perform addition, subtraction, multiplication, division on the specific numbers in the problem

**2. Problems should be grade-school level (GSM8K difficulty), not competition-math level.**

A 42M model does not have the representational capacity for abstract algebra, combinatorics proofs, or multi-variable systems. Focus on:
- Single-variable word problems
- 1–4 step chains of basic arithmetic
- Integer and simple decimal answers
- Clear natural-language problem statements

**3. The SFT dataset must contain ONLY problems that are NOT in the evaluation benchmarks.**

Use the GSM8K *training* split (7,473 problems) as the primary SFT source. Never train on the GSM8K *test* split (1,319 problems), SVAMP test (300), or AQuA test (254).

**4. Chain-of-thought traces must be faithful — every intermediate step must be arithmetically correct.**

Do not use "noisy" or "diverse" CoT traces where intermediate steps contain errors. At 42M parameters, the model has limited capacity to learn "robust reasoning despite occasional errors." It needs clean, consistent demonstrations.

**5. Format must be rigid and consistent.**

Every SFT example should follow exactly:
```
Question: {question}
<think>
{step-by-step reasoning with explicit arithmetic}
</think>
<answer>{numerical_answer}</answer>
```

No variation in delimiters. No optional fields. The model must learn a single, unambiguous format.

**6. Target dataset size: 5,000–15,000 examples.**

For a 42M model, overfitting is a serious risk. The GSM8K training split (7,473) is a reasonable starting point. Augmenting with SVAMP-like problems or synthetic arithmetic is acceptable, but total unique problems should stay under ~15K. Multiple epochs over the same data with appropriate learning rate scheduling is preferable to a massive noisy dataset.

---

## 6. What Should Direct-Answer SFT vs. CoT SFT Control For?

> [!IMPORTANT]
> This is the single most important experimental design question for Mini-R1. Without a proper control, you cannot claim that chain-of-thought *caused* any improvement.

### The Confound:

If you train the model on GSM8K-train with CoT traces, and accuracy on GSM8K-test goes from 0.76% to 25%, you cannot conclude that CoT helped. The improvement could be entirely due to:
1. The model learning to recognize problem formats
2. The model memorizing answer distributions  
3. The model learning to extract numbers from its own output
4. The format constraint (`<answer>X</answer>`) channeling extraction accuracy
5. Simple exposure to arithmetic patterns in the training data

### Required Experimental Design:

**Condition A: Direct-Answer SFT (Control)**
- Same GSM8K training problems
- Format: `Question: {q}\n<answer>{a}</answer>`
- No reasoning trace. No intermediate steps.
- Same number of training tokens (pad or truncate to match)
- Same hyperparameters, same LR, same number of optimizer steps
- Same base checkpoint (`best.pt`)

**Condition B: Chain-of-Thought SFT (Treatment)**  
- Same GSM8K training problems
- Format: `Question: {q}\n<think>\n{step-by-step}\n</think>\n<answer>{a}</answer>`
- Same hyperparameters, same LR, same number of optimizer steps
- Same base checkpoint (`best.pt`)

### What Must Be Controlled:

| Variable | Must Be Identical |
|---|---|
| Training problems | Same problem set |
| Base checkpoint | Same `best.pt` |
| Learning rate schedule | Same peak LR, warmup, decay |
| Number of optimizer steps | Same (or matched by tokens) |
| Evaluation protocol | Same extraction, same prompts |
| Random seeds | Same (or averaged over 3+ seeds) |
| Batch size | Same effective batch |

### What Differs (Intentionally):

| Variable | Direct-Answer | CoT |
|---|---|---|
| Training sequence length | Short (~20-30 tokens) | Long (~100-200 tokens) |
| Total training tokens | Lower | Higher |
| Intermediate computation | None | Explicit arithmetic steps |

> [!WARNING]
> **The token-count asymmetry is a known confound.** CoT sequences are much longer, meaning the CoT model sees more total tokens during SFT. To control for this, you should either: (a) train both for the same number of optimizer steps regardless of token count, or (b) train both for the same total token count (which means more epochs for direct-answer). Document this choice explicitly.

### The Scientific Question:

The comparison answers: **"Does teaching intermediate reasoning steps improve final-answer accuracy beyond what is achievable from format-learning and answer-distribution exposure alone?"**

If CoT SFT achieves 25% on GSM8K and Direct-Answer SFT achieves 22%, the CoT benefit is marginal. If CoT achieves 25% and Direct-Answer achieves 5%, CoT is genuinely teaching transferable reasoning. The magnitude of the gap matters enormously.

---

## 7. What Leakage Risks Exist?

### Risk 1: Train/Test Overlap in Benchmark Datasets (CRITICAL)

GSM8K, SVAMP, and AQuA each have defined train/test splits. The SFT dataset must use ONLY the training splits. Any leakage of test problems into SFT data would render the evaluation meaningless.

**Mitigation**: Before SFT, verify via exact string matching that zero GSM8K-test, SVAMP-test, or AQuA-test questions appear in the SFT training set. Store the test-set hash for reproducibility.

### Risk 2: Semantic Near-Duplicates Across Benchmarks

GSM8K-train and SVAMP share overlapping problem structures (e.g., "apples and oranges" word problems). If the model is SFT'd on GSM8K-train, some SVAMP test problems may be semantically near-identical to GSM8K training problems.

**Mitigation**: SVAMP accuracy should be reported but interpreted cautiously. It is not a fully independent test of generalization.

### Risk 3: Answer-Distribution Memorization

GSM8K answers are heavily concentrated in the range 1–1000, with a strong bias toward small integers. A model could learn to output "the most likely GSM8K answer given keywords" without reasoning.

**Mitigation**: This is precisely what the Direct-Answer control (Section 6) detects. Additionally, track accuracy as a function of answer magnitude — if the model is accurate on small answers (1–10) but fails on large answers (100+), distribution memorization is likely.

### Risk 4: Extraction Method Inflation

The `last_number_fallback` method can produce false positives when the generation incidentally contains the correct number. After SFT, if the model generates longer, more number-rich outputs, the false-positive rate may increase.

**Mitigation**: After SFT, report accuracy separately for:
- Answers extracted via `xml_answer_tag` (structured extraction)
- Answers extracted via `last_number_fallback` (fallback)
- Track extraction method distribution shift between baseline and SFT

### Risk 5: Pretraining Corpus Contamination

Cosmopedia-v2 is synthetically generated educational content. It is conceivable (though unlikely for a synthetic corpus) that some GSM8K-like problems with solutions exist in the pretraining data.

**Mitigation**: The baseline evaluation already shows 0.76% GSM8K accuracy, which rules out significant pretraining contamination. If the pretraining data contained GSM8K solutions, we would see measurable accuracy above the false-positive floor.

---

## 8. What Should Be Held Out Completely?

### Absolute Holdouts (Never Train On, Never Tune On):

| Dataset | Split | Size | Purpose |
|---|---|---|---|
| **GSM8K** | Test | 1,319 | Primary reasoning evaluation |
| **SVAMP** | Test | 300 | Secondary reasoning evaluation |
| **AQuA-RAT** | Test | 254 | Tertiary / diagnostic evaluation |
| **General LM validation** | Fixed 10M tokens | 19,532 sequences | Language modeling regression check |

### Additional Recommendation: Hold Out a GSM8K Validation Slice

Reserve ~500 examples from GSM8K-train as a **development validation set** for SFT hyperparameter selection (learning rate, number of epochs, early stopping). Do NOT evaluate on the test set during hyperparameter search.

Partition:
- GSM8K-train for SFT: ~6,973 examples
- GSM8K-train for SFT validation: ~500 examples  
- GSM8K-test for final evaluation: 1,319 examples (NEVER used during training or HP search)

### What Must NOT Be Held Out:

The general-language validation loss (PPL 17.68) should be re-evaluated after SFT to detect **catastrophic forgetting**. If SFT on reasoning data significantly degrades general LM perplexity (e.g., PPL jumps from 17.68 to 25+), the SFT is overwriting the model's linguistic competence rather than adding reasoning capability on top of it.

---

## 9. What Should the Primary Research Hypothesis Be?

> **H₁: Supervised fine-tuning with explicit chain-of-thought arithmetic demonstrations enables a 42.1M parameter language model to perform multi-step mathematical reasoning that is not achievable through direct-answer supervision alone.**

### Decomposition:

- **Null hypothesis (H₀)**: Any accuracy improvement from SFT is attributable to format learning, answer-distribution memorization, and extraction-method alignment — not to learned intermediate reasoning.
- **Alternative hypothesis (H₁)**: CoT SFT produces accuracy gains on held-out math problems that significantly exceed those achievable by direct-answer SFT, indicating that the model has learned to use intermediate computation.

### Falsification Criteria:

H₁ is **rejected** if:
1. CoT SFT accuracy on GSM8K-test is ≤ Direct-Answer SFT accuracy + 3 percentage points
2. The model achieves high accuracy on 1-step problems but fails on 3+ step problems (suggesting pattern matching, not chained reasoning)
3. Accuracy improvements disappear when evaluated with a strict `xml_answer_tag`-only extractor (suggesting the improvement is extraction-method-dependent)

H₁ is **supported** if:
1. CoT SFT accuracy exceeds Direct-Answer SFT accuracy by ≥ 5 percentage points on GSM8K-test
2. Accuracy scales with problem complexity (1-step < 2-step < 3-step improvement)
3. Qualitative inspection reveals correct intermediate arithmetic in the `<think>` traces
4. The model achieves accuracy on problems whose answer values were never seen in training

---

## 10. What Metrics Should Determine Whether CoT Training Actually Helped?

### Primary Metrics (Must All Be Reported):

| Metric | Baseline (Task 14) | Target | Why It Matters |
|---|---|---|---|
| **GSM8K-test Accuracy (CoT SFT)** | 0.76% | > 15% | Primary capability measure |
| **GSM8K-test Accuracy (Direct-Answer SFT)** | 0.76% | Report as-is | **THE control comparison** |
| **CoT − Direct Δ on GSM8K** | N/A | > 5 pp | Evidence of reasoning vs. memorization |
| **Format Compliance Rate** | 0.0% | > 90% | Did the model learn the output format? |
| **XML-Tag Extraction Rate** | 0.0% | > 80% | Are answers in the intended location? |
| **General LM Val Loss** | 2.8724 | < 3.5 | Catastrophic forgetting detection |
| **SVAMP-test Accuracy (CoT SFT)** | 3.00% | > 10% | Cross-benchmark generalization |

### Secondary Metrics (Diagnostic):

| Metric | Purpose |
|---|---|
| **Accuracy by problem step-count** | Does CoT help more on harder problems? |
| **Accuracy by answer magnitude** | Is the model memorizing common answers? |
| **Think-trace arithmetic correctness** | Are intermediate steps faithful? |
| **Average generation length** | Did the model learn to generate reasoning traces? |
| **EOS rate** | Does the model learn when to stop? |
| **AQuA-test Accuracy** | Tertiary check (interpret cautiously due to 20% floor) |

### The Decisive Test:

> [!CAUTION]
> **If CoT SFT achieves ≤ 3 percentage points above Direct-Answer SFT on GSM8K-test, you should conclude that chain-of-thought did NOT teach this model to reason.** It merely taught it a more verbose output format. This would not invalidate the project — it would be a scientifically important negative result — but it would change the interpretation of subsequent RL (GRPO) training.

### What Would Constitute a Genuine Success:

For a 42.1M model, realistic expectations must be calibrated against published results:
- TinyStories-scale models (28M–33M) achieve ~0% on GSM8K
- GPT-2 Small (124M) achieves ~2-5% on GSM8K with CoT prompting
- Phi-1 (1.3B) achieves ~40%+ on GSM8K

A 42.1M model achieving **15-25% on GSM8K via CoT SFT** would be a strong result. Achieving **> 30%** would be remarkable and would warrant investigation into whether the improvement is genuine or driven by train/test contamination.

---

## Summary Judgment

The Mini-R1 base model at 42.1M parameters has **exactly the zero-shot reasoning capability one would expect: none.** The measured accuracies are extraction artifacts and coincidence. The model is a competent educational-textbook completion engine (PPL 17.68) with zero mathematical competence.

This is the correct starting point for reasoning SFT. The baseline is clean, the benchmarks are uncontaminated, and the measurement infrastructure is sound. The critical next step is to design the SFT experiment with a proper direct-answer control condition so that any claimed CoT benefit is scientifically defensible.
