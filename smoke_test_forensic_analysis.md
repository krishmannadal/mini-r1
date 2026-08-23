# Forensic Analysis: Mini-R1 Smoke Test

## 1. Smoke Test Validity Assessment

### 1a. Forward Pass

**Verdict: PROVEN**

The model produces logits, cross-entropy loss is computed, and the loss value at step 1 (~9.06–9.15) is consistent with the theoretical uniform-distribution baseline for `V=8000`:

$$\ln(8000) \approx 8.987$$

The slightly-above-random initial loss (~9.06) is expected: Xavier-initialized weights do not produce a perfectly uniform distribution over the vocabulary. The loss is finite, non-NaN, and in the correct range.

No additional test needed.

---

### 1b. Backpropagation

**Verdict: PROVEN**

Gradient norms are reported and non-zero at every step (range: 2.3–8.3). Loss decreases monotonically in Phase 1 (9.06 → 7.73). Parameters are being updated—if backward were broken, loss would remain constant and gradient norms would be zero or NaN.

The unit test `test_loss_decreases_after_training_step` additionally proves this in isolation.

No additional test needed.

---

### 1c. AMP (Automatic Mixed Precision)

**Verdict: PARTIALLY PROVEN**

What is proven:
- `torch.amp.autocast` was engaged with `dtype=float16` (reported as "FLOAT16 (AMP Enabled: True)").
- `GradScaler` was created with `enabled=True`.
- Training did not produce NaN loss or NaN gradients, meaning the scaler was functional.
- VRAM usage is consistent with FP16 activations (~515 MB allocated for a 42.1M model at `d_model=512`, `seq_len=512`, batch 4).

What is **not** proven:
- **No comparison run exists.** You have not run the same smoke test with `amp_dtype="float32"` and compared VRAM and throughput. Without this, you cannot confirm AMP actually reduced memory or increased speed.
- **BF16 has never been tested.** The smoke test only exercises FP16.
- **The scaler `_scale` factor was never logged.** You don't know if GradScaler ever skipped a step due to overflow. This is important—a silently skipped step means the optimizer didn't update, and you wouldn't notice in an 8-step run.

> [!IMPORTANT]
> **Additional tests required:**
> 1. Run the same 8-step smoke test with `amp_dtype="float32"`. Compare VRAM and loss trajectory.
> 2. Log `scaler.get_scale()` at each optimizer step to verify no skipped updates.
> 3. Run with `amp_dtype="bfloat16"` to confirm BF16 path works (RTX 4050 Ada Lovelace SM 8.9 supports BF16).

---

### 1d. Gradient Accumulation

**Verdict: PARTIALLY PROVEN**

What is proven:
- `gradient_accumulation_steps=2` is configured.
- `scaled_loss = loss / grad_accum_steps` is applied ([trainer.py line 271](src/trainer.py#L271)).
- Optimizer steps occur every 2 micro-batches (smoke test: `micro_batch_size=4`, `accumulation=2`, `effective=8`).
- 8 micro-batches → 4 optimizer steps in Phase 1: arithmetic checks out.

What is **not** proven:
- **Gradient equivalence.** The definitive test would be: run a batch of size 8 without accumulation and compare the resulting gradient norms and parameter deltas against `micro_batch=4, accum=2`. If accumulation is correct, they should be identical (up to numerical precision). This test does not exist.
- **The partial-epoch flush logic** ([trainer.py lines 416–427](src/trainer.py#L416-L427)) has never been triggered in the smoke test, because the smoke test always uses a dataset large enough that accumulation boundaries align with epoch boundaries.

> [!IMPORTANT]
> **Additional test required:**
> Construct a dataset with exactly 3 samples, `micro_batch=2`, `accumulation=2`. This forces 1 full accumulation step + 1 remainder micro-batch. Verify the partial-flush path produces correct gradients and advances `global_step`.

---

### 1e. Checkpointing

**Verdict: PARTIALLY PROVEN — with a bug.**

What is proven:
- `latest.pt`, `best.pt`, `step_N.pt` files are created and contain valid tensors (resume succeeds).
- Atomic write uses `os.replace`.

What is **buggy**:
- **`latest_tmp_mvfbfpcs.pt` still exists on disk** (508 MB, timestamped 7:32 PM—from the first failed smoke run). This is a stale orphan from the earlier `PermissionError` crash. The old code used `tempfile.NamedTemporaryFile` which created the file with a different naming convention (`_tmp_` prefix). After the code was fixed to use `uuid`-based naming, the old orphan was never cleaned up.
- **`best_model.pt` and `best.pt` are duplicates.** Both are 508,906,934 bytes. `best_model.pt` is a "legacy alias for backward test compatibility" ([trainer.py line 382–393](src/trainer.py#L382-L393)). This wastes ~485 MB of disk. For a real run with many checkpoints, this adds up.
- **`checkpoint_epoch_N.pt` files are saved unconditionally at every epoch boundary** ([trainer.py lines 462–472](src/trainer.py#L462-L472)), even though `step_N.pt` and `latest.pt` already capture the same state. With `retained_checkpoints=3`, the step rotation correctly purges old step files, but epoch files are never purged.

More detail in Section 8.

---

### 1f. Resume

**Verdict: PROVEN (with caveats)**

What is proven:
- Phase 2 resumed at `global_step=4` and continued to `global_step=8`. The global step counter was correctly restored.
- Loss at step 5 after resume (7.62–7.71) is lower than the loss at step 4 (7.73–7.90), indicating that the model weights were not re-initialized.
- The epoch counter advanced correctly from epoch 1 to epoch 2.

Caveats:
- **You have not verified bitwise determinism.** If you ran steps 1–8 continuously (no resume) and compared against the resume run, the loss values should be *identical* if all state was restored correctly. You did not do this. The two runs you have (lines 1–12 vs. lines 13–22 of the JSONL) used different random seeds (different model initializations), so they cannot be compared.
- **The resume creates a new `model2` with fresh Xavier initialization and then overwrites it** ([train.py lines 117–118](scripts/train.py#L117-L118)). This is technically correct but wasteful—it allocates memory for random weights that are immediately discarded. More importantly, it means there is a brief moment where 2× model memory is consumed. On a 6 GB GPU during a real run, this could cause an OOM crash during resume.

> [!WARNING]
> **Critical test required:**
> Run steps 1–8 continuously. Then run steps 1–4, stop, resume, run 5–8. Compare the loss at every step. They must match within FP16 epsilon (~1e-3). If they diverge significantly, RNG or optimizer state restoration is broken.

---

### 1g. Scheduler State Restoration

**Verdict: PARTIALLY PROVEN**

What is proven:
- The LR at step 5 after resume (2.6046e-4) is the correct value from the cosine schedule, not the initial LR.
- The unit test `test_scheduler_state_dict_restore` passes.

What is **not** directly proven by the smoke test:
- The LR sequence after resume was not compared against the LR sequence from a continuous run. The LR values *look* correct, but they haven't been verified against an oracle.

The unit test is sufficient here. **Acceptable.**

---

### 1h. Optimizer State Restoration

**Verdict: PARTIALLY PROVEN**

What is proven:
- Training continues to decrease loss after resume, which would be unlikely if optimizer momentum buffers were reset to zero.

What is **not** proven:
- **No bitwise comparison of optimizer `state_dict` before save and after load.** AdamW has `exp_avg` (first moment) and `exp_avg_sq` (second moment) per parameter. These must be *exactly* restored. A test should save the optimizer state, reload it, and assert `torch.equal` on every buffer.
- The VRAM jump from ~515 MB (Phase 1 step 1) to ~998 MB (Phase 2 step 5) is actually explained by optimizer state initialization: AdamW allocates 2 FP32 buffers per parameter at the first `step()` call. At 42.1M params × 4 bytes × 2 buffers = ~337 MB. This is consistent with 515 + 337 ≈ 852 MB (close to the observed ~998 MB, with the remainder explained by activation cache differences). This *indirectly* confirms optimizer states were loaded, because if they weren't loaded, they would have been lazily re-allocated on the first step of Phase 2—which is exactly what happened in Phase 1. But in Phase 2, the VRAM jump happens *before* step 5, meaning the optimizer state was pre-allocated by `load_state_dict`.

> [!TIP]
> This is acceptable for now, but add a bitwise optimizer state test before long training.

---

### 1i. GradScaler State Restoration

**Verdict: NOT PROVEN**

- `scaler.get_scale()` was never logged.
- No test compares the scaler's internal `_scale` factor before and after resume.
- The scaler state *is* saved and loaded in the checkpoint payload (`scaler_state_dict`), and the code path exists, but there is no empirical evidence that the restored scale factor matches the pre-save value.
- The unit test `test_atomic_checkpoint_save_and_load` verifies that the key exists in the payload, but does not assert that `scaler.load_state_dict()` produces the correct internal state.

> [!WARNING]
> **Test required:** Save `scaler.get_scale()` before checkpoint. Resume. Assert `scaler.get_scale()` matches exactly. Also log the scale value at every step during real training to detect NaN/Inf overflow events.

---

### 1j. RNG Restoration

**Verdict: NOT PROVEN**

The checkpoint code captures `random.getstate()`, `np.random.get_state()`, `torch.get_rng_state()`, and `torch.cuda.get_rng_state()`. The restore code calls the corresponding setters.

However:
- **The smoke test does not verify deterministic replay.** The data loader uses `shuffle=True`, so after resume, the data order depends on the RNG state. If the RNG was not correctly restored, the model would see a different data order, but you would never notice in an 8-step run.
- **The DataLoader's worker RNG states are not captured.** If `num_workers > 0`, each worker has its own RNG that is not in `torch.get_rng_state()`. (Currently `num_workers=0`, so this is not an issue yet, but will be if you add workers later.)

> [!IMPORTANT]
> **Test required:** Set a fixed seed at step 1. Run to step 4, save, note the data batch at step 5. Resume from step 4. Verify step 5 sees the same data batch. This is the only way to confirm RNG restoration.

---

## 2. Loss Trajectory Analysis

### Observed loss sequence (latest run):

| Step | Train Loss | Δ from previous |
|------|-----------|-----------------|
| 1    | 9.061     | —               |
| 2    | 8.662     | −0.399          |
| 3    | 7.965     | −0.697          |
| 4    | 7.731     | −0.234          |
| 5    | 7.710     | −0.021          |
| 6    | 7.440     | −0.270          |
| 7    | 7.603     | +0.163 ⚠️       |
| 8    | 7.279     | −0.324          |

### Is the decrease meaningful?

**Partially, but you should not be confident.**

- The initial loss (9.06) is close to `ln(8000) ≈ 8.99`. After 8 optimizer steps, loss reaches 7.28. This is a reduction of 1.78 nats, corresponding to perplexity dropping from ~8000 to ~1390.
- **This is almost certainly memorization.** The smoke test uses `limit_per_dataset=50`, meaning only 150 total training samples. With `micro_batch=4` and `accumulation=2`, the effective batch is 8 samples. Over 8 optimizer steps, the model sees only 64 micro-batches × 4 = 256 sample presentations. But many of these are repeats from the 150-sample pool (the DataLoader shuffles and loops). The model is likely memorizing token-frequency statistics of this tiny dataset, not learning language.
- **Loss can decrease on random data too.** Cross-entropy loss on a 42.1M parameter model will decrease even on random token sequences for the first few hundred steps—the model simply learns the marginal token distribution (bias terms in the LM head capture unigram frequencies). This is not evidence of learning.

### Instability?

- **Step 7 shows a loss increase** (7.44 → 7.60). This is normal stochastic noise with a batch size of 8 and no momentum averaging. It is not a sign of instability at this scale.
- **The increase is small (2.2%)** relative to the loss magnitude. In a real training run with hundreds of steps, you should monitor the moving average, not individual step losses.

### Gradient norms

| Step | Grad Norm |
|------|-----------|
| 1    | 7.57      |
| 2    | 5.44      |
| 3    | 4.72      |
| 4    | 3.19      |
| 5    | 2.77      |
| 6    | 2.86      |
| 7    | 2.34      |
| 8    | 2.78      |

- **Initial grad norms are high (7.57)** because the model is far from any local minimum and gradients are large. They decrease and stabilize around 2.5–3.0. This is healthy behavior.
- **Gradient clipping at 1.0 was triggered at every step** (all norms > 1.0). At step 1, the gradient was clipped from 7.57 to 1.0—a 7.57× reduction. This is aggressive but correct for early training. As norms decrease toward 1.0, clipping will become less aggressive.
- **For a real run, you should monitor whether grad norms stabilize between 0.5–2.0 after warmup.** If they stay persistently above 3.0 after 100+ steps, the learning rate is too high.

### Could the improvement be memorization?

**Yes, almost certainly.** 150 samples with 8 optimizer steps = the model has barely begun to fit unigram statistics. The validation loss (7.63 → 7.24) also decreases, but the validation set is only 30 samples from the same distribution. This is not a meaningful generalization test.

**Do not draw any conclusions about model quality from this smoke test. Its purpose was to verify the training infrastructure, not the model.**

---

## 3. Learning Rate Schedule Analysis

### Observed LR trajectory:

| Step | LR         | Phase         |
|------|------------|---------------|
| 0    | 0          | Init          |
| 1    | 3.00e-5    | Warmup (1/2)  |
| 2    | 1.65e-4    | Warmup (2/2)  |
| 3    | 3.00e-4    | Peak          |
| 4    | 2.897e-4   | Cosine decay  |
| 5    | 2.605e-4   | Cosine decay  |
| 6    | 2.167e-4   | Cosine decay  |
| 7    | 1.65e-4    | Cosine decay  |
| 8    | 1.134e-4   | Cosine decay  |

### Warmup analysis

The scheduler is configured with `warmup_steps=2` and `max_steps=10` for the smoke test.

**Warmup in the smoke test reaches peak LR in 2 steps. This is irrelevant—it's a smoke test parameter.**

The real [`configs/train.yaml`](configs/train.yaml) specifies `warmup_steps=20` and `max_steps=100`. At 20 warmup steps out of 100 total, warmup occupies 20% of training. This is high—typical warmup ratios are 1–5% of total steps.

**However, 100 `max_steps` is not a real pretraining run.** If you increase to, say, 5000 steps, then 20 warmup steps becomes 0.4%, which is reasonable.

### Is 3e-4 plausible for this architecture?

Reasoning from first principles:

**Model size:** 42.1M parameters, `d_model=512`, 12 layers.

**Reference scaling laws:** The Chinchilla/GPT-3 convention for AdamW suggests:
- For models around 100M: peak LR ≈ 6e-4
- For models around 1B: peak LR ≈ 1.5e-4
- Scaling roughly as `LR ∝ 1/sqrt(d_model)` or empirically by model size

For `d_model=512`, the µP-style scaling gives: `base_lr * sqrt(256/512) ≈ base_lr * 0.707`. If we take GPT-2 small (`d_model=768`) at 6e-4 as reference: `6e-4 * sqrt(768/512) = 6e-4 * 1.22 ≈ 7.3e-4`. But this is aggressive.

**Empirical references:**
- TinyStories (28M params, `d_model=512`): used 5e-4
- SmolLM (135M): used 3e-4
- LLaMA-7B: used 3e-4

**3e-4 for 42.1M is within the plausible range but on the aggressive side for such a small effective batch size (32 in the real config).**

The concern is:
- Effective batch of 32 sequences × 512 context = ~16K tokens per update
- This is small. High LR + small batch = high gradient noise relative to the true gradient
- **Risk:** oscillating loss, failure to converge after warmup, or divergence mid-training

### Recommendation

Before committing to a long run, you should test **at least 3 LR candidates** over 200–500 steps:
1. `1e-4` (conservative)
2. `3e-4` (current)
3. `6e-4` (aggressive)

Run each for 300 steps, plot loss curves, select the one with the steepest stable descent. Do not guess.

### Should `min_lr` change?

`min_lr = 3e-5` (1/10 of peak) is standard. No change needed.

---

## 4. Perplexity Analysis

### Mathematical meaning

Perplexity = $e^{\text{cross-entropy loss}}$

| Measurement | Loss   | Perplexity | Interpretation |
|-------------|--------|------------|----------------|
| Step 4 val  | 7.634  | 2067       | Model assigns probability ~1/2067 to the correct next token on average |
| Step 8 val  | 7.237  | 1390       | Model assigns probability ~1/1390 to the correct next token on average |
| Random      | 8.987  | 8000       | Uniform distribution over vocabulary |

### What it tells us

The model has learned *something*—it assigns higher-than-random probability to correct tokens. But perplexity of 1390 means the model is still effectively choosing among ~1390 equally likely tokens at each position. For comparison:
- A decent language model achieves perplexity 20–50 on typical text
- A model that has learned unigram frequency statistics alone can reach perplexity ~200–500 on English text
- Perplexity 1390 means the model has not even fully learned unigram statistics

### What it does NOT tell us

1. **It does not tell you the model can reason.** Perplexity measures average next-token prediction quality. A model can have low perplexity by memorizing common bigrams without understanding math.
2. **It does not tell you about generation quality.** Perplexity is measured on teacher-forced evaluation. The model never generates autoregressively during this test.
3. **Perplexity on 30 validation samples is statistically meaningless.** The confidence interval is enormous. A single long sequence could swing the result by hundreds of points.
4. **The improvement 2067 → 1390 after 4 more optimizer steps on 150 training samples is memorization, not generalization.** The val set is from the same tiny pool.

---

## 5. VRAM Analysis

### Observed values

| Phase | Step | Allocated MB | Reserved MB |
|-------|------|-------------|-------------|
| 1     | 1    | 515         | 1062        |
| 1     | 2    | 516         | 1372        |
| 1     | 3    | 513         | 1372        |
| 1     | 4    | 519         | 1384        |
| 2     | 5    | 1018        | 2520        |
| 2     | 6    | 1010        | 2520        |
| 2     | 7    | 1011        | 2520        |
| 2     | 8    | 1003        | 2578        |

### Component breakdown

**Allocated memory** = actual tensor data currently held by PyTorch:

| Component | Estimated Size | Calculation |
|-----------|---------------|-------------|
| Model weights (FP32 master) | ~161 MB | 42.1M × 4 bytes (weight-tied, so embedding counted once) |
| Model weights (FP16 copy for forward) | ~80 MB | Created dynamically by autocast |
| Activations (batch 4, seq 512) | ~200–300 MB | 12 layers × attention matrices + FFN intermediates |
| **Total Phase 1 expected** | **~441–541 MB** | **Matches observed ~515 MB** ✅ |

**Phase 2 jump to ~1018 MB:**

| Additional component | Size | Calculation |
|---------------------|------|-------------|
| AdamW `exp_avg` (first moment) | ~161 MB | 42.1M × 4 bytes |
| AdamW `exp_avg_sq` (second moment) | ~161 MB | 42.1M × 4 bytes |
| Gradients (FP32) | ~161 MB | 42.1M × 4 bytes |
| **Total additional** | **~483 MB** | |
| **Expected Phase 2** | **~998 MB** | **Matches observed ~1018 MB** ✅ |

**Why is Phase 1 lower?** AdamW lazily allocates its momentum buffers. At step 1, the optimizer hasn't called `step()` yet, so `exp_avg` and `exp_avg_sq` don't exist. By step 2, they're allocated. But in Phase 1, the measurement at step 1 catches the model *before* full optimizer allocation.

**Wait—why would Phase 2 have higher allocated memory than Phase 1 step 4?** In Phase 1, the optimizer states *were* allocated (they had to be—4 optimizer steps were taken). The answer: **the measurement reports `torch.cuda.memory_allocated()` at the logging point, which is *after* `optimizer.zero_grad()`.** Activation tensors from the forward pass have been freed. But optimizer buffers persist.

The Phase 1 vs Phase 2 difference (~515 → ~1018 MB) is actually an artifact of **when `memory_allocated()` is called relative to the optimizer state lifecycle.** In Phase 1, the optimizer states are allocated incrementally. In Phase 2, `load_state_dict` pre-allocates all optimizer buffers *before* the first forward pass. This is expected and correct.

### Reserved vs Allocated

- **Reserved** (1062–2578 MB) is the CUDA memory allocator's pool. PyTorch pre-allocates chunks from the GPU driver to avoid frequent `cudaMalloc` calls. Reserved memory can be 1.5–3× allocated memory.
- **This is not a concern.** Reserved memory is recycled internally. The actual GPU memory visible to `nvidia-smi` will be `reserved + ~300 MB` (CUDA context, cuDNN, driver).
- **Estimated total GPU usage: ~2.5–2.9 GB** out of 6 GB. This leaves ~3.1–3.5 GB headroom. Comfortable.

### Concerning?

**No.** The numbers are consistent with expectations. But before real training:

> [!IMPORTANT]
> **Additional measurements to collect:**
> 1. `torch.cuda.max_memory_allocated()` — the peak allocated, not the instantaneous value. This catches transient spikes during backward pass.
> 2. `nvidia-smi` output during training — reports actual driver-level allocation.
> 3. Test with the real `micro_batch_size=8` (not smoke test's 4) and verify peak VRAM stays under 5.0 GB.

---

## 6. Tokens/sec Analysis

### Observed throughput

| Step | Tok/s  | Phase | Notes |
|------|--------|-------|-------|
| 1    | 1607   | 1     | First step (cold) |
| 2    | 2698   | 1     | Warm |
| 3    | 1220   | 1     | Warm |
| 4    | 1427   | 1     | Warm |
| 5    | 4894   | 2     | First step after resume |
| 6    | 5603   | 2     | Second step after resume |
| 7    | 1670   | 2     | Back to normal |
| 8    | 1960   | 2     | Normal |

### Explanation of the anomaly

The wild variation (1220 → 5603) has three likely causes:

**1. `MetricsTracker` timing artifact.** The tracker uses `time.perf_counter()` and accumulates tokens across all steps since its last `reset()`. Looking at the code:

[`trainer.py` line 304](src/trainer.py#L304):
```python
step_tracker.update(loss=avg_step_loss, num_tokens=accum_tokens)
```

[`trainer.py` line 312](src/trainer.py#L312):
```python
metrics = step_tracker.compute()
```

The `MetricsTracker` is created *once* at [line 244](src/trainer.py#L244) and **never reset.** So `tokens_per_sec` is the *cumulative average* since training started, not the per-step rate. After resume, `start_time` is reset ([line 224](src/trainer.py#L224): `self.start_time = time.time()`), but `step_tracker` is a *new* `MetricsTracker` created at line 244. This means:

- Steps 5–6 after resume: the tracker has accumulated very few tokens over a very short wall-clock time (0.23 sec at step 5). The apparent ~5000 tok/s is because the elapsed time in the denominator is tiny.
- Steps 7–8: more time has elapsed, and the cumulative average stabilizes.

**2. CUDA kernel warmup.** The first step after a fresh model instantiation triggers JIT compilation of cuDNN/cuBLAS kernels. Step 1 is always slow. After resume, the GPU caches are warm from Phase 1, so steps 5–6 are faster.

**3. Validation overhead.** Steps 3–4 in Phase 1 are slower than steps 1–2 because the validation at step 4 triggers `model.eval()` and a full pass over the val set, but the elapsed time in the `step_tracker` spans all steps including this overhead.

### Should you trust these numbers?

**No.** The cumulative-average `tokens_per_sec` metric is misleading for step-level analysis. For a real training run:

> [!WARNING]
> **You should compute per-step throughput** using wall-clock time between consecutive optimizer steps, not cumulative averages. The current implementation will report increasingly smoothed (and thus increasingly stale) throughput as training progresses.

---

## 7. Logging Hygiene

### The JSONL log contains interleaved runs

The file has 22 non-empty lines:
- Lines 1–2: First crashed run (Phase 1 only, 2 steps before PermissionError)
- Lines 3–12: Second run (Phase 1 steps 1–4 + val, Phase 2 steps 5–8 + val)
- Lines 13–22: Third run (latest successful smoke test)

**This is unacceptable for research.** Multiple experiments append to the same `train_log.jsonl` with no run identifier, no timestamp, no separator. You cannot parse this log programmatically to extract a single run's trajectory without manual inspection.

### Recommended logging scheme

You should implement:

**1. Run ID.** Generate a unique run identifier at training start:
```
run_id = f"{timestamp}_{short_uuid}"
# Example: "20260816_193530_a3f7"
```

**2. Per-run log directory:**
```
logs/
  run_20260816_193530_a3f7/
    train_log.jsonl
    config_snapshot.yaml
    git_hash.txt           # output of git rev-parse HEAD
    system_info.json       # GPU, PyTorch version, CUDA version
```

**3. Every JSONL record must include:**
```json
{
  "run_id": "20260816_193530_a3f7",
  "timestamp": "2026-08-16T19:35:30.123Z",
  "global_step": 1,
  "epoch": 1,
  "train_loss": 9.06,
  "...": "..."
}
```

**4. Separate smoke tests from real training:**
- Smoke tests → `artifacts/smoke_logs/`
- Real training → `logs/run_<id>/`
- Never share log files between runs.

**5. Config hash:** SHA256 of the YAML config, recorded in every run directory. This lets you associate a run with a specific configuration without manual comparison.

**6. Git hash:** If you're using version control, record the commit hash at training start. This connects a run to a specific code version.

---

## 8. Checkpoint Analysis

### Files on disk

| File | Size (MB) | Source | Expected? |
|------|-----------|--------|-----------|
| `best.pt` | 485 | `CheckpointManager.save(is_best=True)` | ✅ Yes |
| `best_model.pt` | 485 | `save_checkpoint(...)` at [trainer.py line 383](src/trainer.py#L383) | ❌ **Unnecessary duplicate** |
| `checkpoint_epoch_1.pt` | 485 | End-of-epoch save at [trainer.py line 462](src/trainer.py#L462) | ⚠️ Overlaps with `step_4.pt` |
| `checkpoint_epoch_2.pt` | 485 | End-of-epoch save at [trainer.py line 462](src/trainer.py#L462) | ⚠️ Overlaps with `step_8.pt` |
| `latest.pt` | 485 | `CheckpointManager.save` always saves | ✅ Yes |
| `latest_tmp_mvfbfpcs.pt` | 485 | **ORPHAN from crashed first run** | ❌ **BUG** |
| `step_4.pt` | 485 | Step checkpoint at interval 2 | ✅ Yes |
| `step_6.pt` | 485 | Step checkpoint at interval 2 | ✅ Yes |
| `step_8.pt` | 485 | Step checkpoint at interval 2 | ✅ Yes |

**Total disk usage: 9 × 485 MB ≈ 4.4 GB** for an 8-step smoke test.

### Issues

**1. `latest_tmp_mvfbfpcs.pt` — Stale orphan (CRITICAL)**

This file was created by the old `tempfile.NamedTemporaryFile` code that crashed with `PermissionError` before `os.replace` could succeed. The exception handler tried to `unlink()` the temp file but also got `PermissionError`.

**What should happen to a temp file:** It should either be successfully renamed to the target (on success) or deleted (on failure). A temp file that survives on disk is a bug. It means the atomic write protocol was violated.

The current code fix (uuid-based naming + retry loop) prevents this from happening *going forward*, but:
- **The orphan file from the previous crash is still on disk.**
- **No cleanup mechanism exists.** If training crashes mid-save, the `.tmp` file will remain forever. You should add startup cleanup: at `Trainer.__init__`, scan `checkpoint_dir` for `*.tmp` files and delete them.

**2. `best_model.pt` — Legacy duplicate**

[`trainer.py` lines 382–393](src/trainer.py#L382-L393) saves a duplicate `best_model.pt` "for backward test compatibility." This wastes 485 MB. The tests should be updated to use `best.pt` and this legacy alias should be removed.

**3. `checkpoint_epoch_N.pt` — Unmanaged epoch checkpoints**

[`trainer.py` lines 462–472](src/trainer.py#L462-L472) save an epoch checkpoint unconditionally. These files:
- Are never rotated or purged
- Overlap with step checkpoints (step_4 = end of epoch 1, step_8 = end of epoch 2 in this test)
- Will accumulate indefinitely during a long run

For a 50-epoch run, this creates 50 × 485 MB = **24 GB** of unmanaged checkpoint files.

**4. Each `CheckpointManager.save()` call writes up to 3 separate files**

When `is_best=True` and `is_step_checkpoint=True`, the manager writes:
1. `latest.pt`
2. `best.pt`
3. `step_N.pt`

Each is a full 485 MB `torch.save()`. That's 1.46 GB of I/O for a single logical checkpoint event. Plus the epoch-end call adds 2 more saves (epoch checkpoint + epoch-end best), totaling up to 5 saves per epoch boundary.

**5. Double-saving best checkpoint**

At the step validation checkpoint ([lines 368–393](src/trainer.py#L368-L393)), `CheckpointManager.save(is_best=True)` saves `latest.pt` + `best.pt`. Then immediately, `save_checkpoint(best_model.pt)` saves a third copy. At the epoch-end validation ([lines 436–459](src/trainer.py#L436-L459)), it does this *again*. So at an epoch boundary where a best model is found, the code saves: `latest.pt` + `best.pt` + `best_model.pt` from the step check, then `latest.pt` + `best.pt` + `best_model.pt` from the epoch check, then `checkpoint_epoch_N.pt`. That's **7 file writes** of ~485 MB each = **3.4 GB of I/O at a single epoch boundary.** This is wasteful.

---

## 9. Prioritized Fix List

### CRITICAL (must fix before ANY real training)

| # | Issue | Rationale |
|---|-------|-----------|
| C1 | **Delete orphan `latest_tmp_mvfbfpcs.pt`** and add startup cleanup for `*.tmp` files | Orphan temp files indicate the atomic protocol can fail silently. A cleanup sweep at trainer init prevents accumulation. |
| C2 | **Remove `best_model.pt` duplication** | 485 MB wasted per best-model event. Update tests to use `best.pt`. |
| C3 | **Remove or manage epoch checkpoint files** | Unbounded disk growth. Either stop saving them, or add them to the rotation pool. |
| C4 | **Fix `tokens_per_sec` to be per-step, not cumulative** | Current metric is misleading and will mask throughput degradation in long runs. |
| C5 | **Add run ID and per-run log directories** | Without this, log files from repeated runs are interleaved and unparseable. |
| C6 | **Clear log file at training start (or use per-run dirs)** | Current code appends to existing JSONL, mixing run data. |

### HIGH (strongly recommended before long training)

| # | Issue | Rationale |
|---|-------|-----------|
| H1 | **Log `scaler.get_scale()` at each step** | Detect FP16 overflow and skipped optimizer steps. Without this, training can silently stall. |
| H2 | **Add `torch.cuda.max_memory_allocated()` logging** | Peak memory, not instantaneous, determines OOM risk. |
| H3 | **Test with real batch size (micro_batch=8, accum=4)** before long training | Smoke test used micro_batch=4, accum=2. Real config uses 8×4=32. VRAM could be very different. |
| H4 | **Run LR sweep (3 candidates, 300 steps each)** | 3e-4 is plausible but unvalidated. A short sweep avoids wasting a long run on a suboptimal LR. |
| H5 | **Reduce checkpoint I/O at best+step boundaries** | Up to 7 saves (3.4 GB I/O) at one boundary is excessive. Consolidate. |

### MEDIUM (recommended)

| # | Issue | Rationale |
|---|-------|-----------|
| M1 | **Bitwise resume determinism test** | Run 1–8 continuous vs. 1–4 + resume + 5–8. Compare loss at every step. |
| M2 | **Gradient accumulation equivalence test** | batch=8 without accum vs. micro=4, accum=2. Compare gradient norms. |
| M3 | **Test BF16 path** | Never exercised. RTX 4050 supports it. Could be more stable than FP16. |
| M4 | **Log optimizer `step_count` to detect skipped steps** | GradScaler can skip steps silently. |

### SAFE TO DEFER

| # | Issue | Rationale |
|---|-------|-----------|
| D1 | DataLoader `num_workers > 0` RNG handling | Not currently used. Add when scaling throughput. |
| D2 | `nvidia-smi` integration for driver-level VRAM reporting | Nice to have, not critical. |
| D3 | Gradient histogram logging (per-layer) | Useful for debugging instability, not needed until instability occurs. |

---

## 10. Next Experiment Design

### Experiment: LR Sweep with Real Batch Size

**Purpose:** Validate that the training system is stable under real configuration and select the optimal learning rate before committing to a long run.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | 42.1M (current [`configs/model.yaml`](configs/model.yaml)) |
| Precision | FP16 AMP |
| Micro batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Context length | 512 |
| Dataset | Combined GSM8K + SVAMP + AQuA train (full training set) |
| Validation set | Combined validation (full) |
| Max steps | 300 per LR candidate |
| Warmup steps | 30 (10% of 300) |
| Gradient clipping | 1.0 |
| Weight decay | 0.01 |
| Validation interval | 50 steps |
| Checkpoint interval | 100 steps |
| Log interval | 1 step |

**LR Candidates:**

| Run | Peak LR | Min LR | Rationale |
|-----|---------|--------|-----------|
| A   | 1e-4    | 1e-5   | Conservative; baseline |
| B   | 3e-4    | 3e-5   | Current config |
| C   | 6e-4    | 6e-5   | Aggressive; test upper bound |

**Metrics to collect per step:**

```
global_step, epoch, train_loss, val_loss, lr, grad_norm, 
scaler_scale, tokens_per_sec_step, gpu_peak_allocated_mb, 
gpu_reserved_mb, elapsed_sec, run_id
```

**Evidence required before declaring "safe for long training":**

1. All 3 runs complete 300 steps without NaN loss
2. No runs show loss divergence (loss increasing for >20 consecutive steps)
3. Gradient norms stabilize below 2.0 after warmup
4. `scaler.get_scale()` does not drop below 1.0 (would indicate chronic overflow)
5. Peak VRAM (`max_memory_allocated`) stays below 5.0 GB with micro_batch=8
6. One LR candidate shows clearly steeper descent than others
7. Validation loss tracks training loss (no immediate overfitting divergence in 300 steps)
8. Resume test: stop run B at step 150, resume, verify loss continuity

**Estimated time per run:** 300 steps × ~2 sec/step ≈ 10 minutes. Total: ~30 minutes for all 3 candidates.

---

## 11. Final Verdict

### A. Safe to start long training?

**NO.**

### B. Exact reasons

1. **Stale temp file on disk** proves the atomic checkpoint mechanism can leave orphans. A long run that crashes mid-checkpoint could corrupt state.
2. **Checkpoint I/O is 3–7× redundant**, creating up to 3.4 GB of writes per epoch boundary. In a long run, this wastes hours of wall-clock time and fills disk.
3. **`tokens_per_sec` metric is broken** (cumulative average, not per-step). You will be unable to diagnose throughput degradation.
4. **No logging of `scaler.get_scale()`** means FP16 overflow is invisible. The scaler could silently skip every optimizer step and you wouldn't know.
5. **Log files are appended across runs** without run identifiers. You cannot analyze results from a long run if it shares a log file with previous experiments.
6. **LR has not been validated.** 3e-4 might be fine, might diverge at step 500. A 30-minute sweep eliminates this risk.
7. **Real batch size (8×4=32) has never been tested on GPU.** The VRAM estimate is ~2.5 GB, but peak activation memory during backward could exceed this.

### C. Exact fixes required before long training

1. Delete `latest_tmp_mvfbfpcs.pt`. Add `*.tmp` cleanup at trainer init.
2. Remove `best_model.pt` legacy save. Remove duplicate epoch-end best save.
3. Either stop saving `checkpoint_epoch_N.pt` or add rotation.
4. Compute per-step `tokens_per_sec` (wall-clock between consecutive optimizer steps).
5. Log `scaler.get_scale()` in every structured log entry.
6. Log `torch.cuda.max_memory_allocated()` alongside instantaneous.
7. Implement per-run log directories with run ID, config snapshot, and git hash.
8. Run the 3-candidate LR sweep (300 steps each, ~30 min total).
9. Run one test at real batch size (`micro_batch=8`, `accum=4`) for 50 steps to verify VRAM fits.

### D. Exact recommended configuration for the next run (after fixes)

```yaml
training:
  seed: 42
  device: "cuda"
  amp_dtype: "float16"
  micro_batch_size: 8
  gradient_accumulation_steps: 4    # effective batch = 32
  lr: 3e-4                          # PENDING: update after LR sweep
  min_lr: 3e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_steps: 30
  max_steps: 300                    # LR sweep: 300 steps per candidate
  epochs: 10                        # large enough to not be the bottleneck
  val_interval_steps: 50
  checkpoint_interval_steps: 100
  log_interval_steps: 1             # log every step during sweep
  retained_checkpoints: 3
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  max_train_samples: null           # use full dataset for sweep
  max_val_samples: null
```

### E. What to send after the next run

Send me:
1. **The full JSONL log** from each of the 3 LR candidates (with all new fields: `scaler_scale`, `peak_vram`, per-step `tok/s`).
2. **The val loss at step 50, 100, 150, 200, 250, 300** for each candidate.
3. **The `nvidia-smi` output** during training at micro_batch=8.
4. **The resume determinism test result**: loss at steps 145–155 continuous vs. stopped-at-150-and-resumed.
5. **The checkpoint directory listing** (to verify no orphan temp files or unbounded accumulation).

I will analyze these to select the final LR and declare readiness for long pretraining.
