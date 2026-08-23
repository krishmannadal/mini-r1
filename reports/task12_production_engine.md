# Mini-R1: Task 12 Production-Scale Pretraining Engine Qualification Report

**Role**: Senior Systems & Training Infrastructure Engineer  
**Date**: 2026-08-17  
**Model**: 42.1M parameter Decoder-Only Transformer (`d_model=512`, `n_layers=12`, `n_heads=8`, `d_ff=2048`, `vocab=8000`, `context=512`)  
**Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Corpus**: `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2` 60%, `fineweb-edu-dedup` 40%)  
**Storage Architecture**: Local Binary Token Shards (`uint16`, 2 bytes/token, 512-token dense packed sequences)

---

## 1. Executive Summary

Task 12 establishes the **Production-Scale Pretraining Engine** for Mini-R1, completely replacing the in-memory preloading prototype with an **offline, sharded, memory-mapped binary streaming pipeline** that operates with **100% network isolation**.

### Key Milestones Achieved:
1. **Zero-RAM Shard Engine**: Implemented `ShardedPretrainDataset` and `build_pretrain_shards.py` using compact raw `uint16` binary format. Memory consumption is fixed at a single active shard (~20 MB) regardless of corpus size.
2. **Exact Bitwise Resumability**: Validated 100-step continuous run vs 50+50 step resumed run. Final loss matches to **$0.00\text{e}+00$ difference** with **zero parameter divergence**.
3. **100% Air-Gapped Network Isolation**: Validated with simulated network failure/socket blocking; zero network requests during training loop.
4. **Hardware & Compute Efficiency**: Verified stable throughput between **$16{,}000$ and $21{,}700\text{ tokens/sec}$** with peak VRAM locked at **$3{,}246\text{ MB}$** (2.75 GB headroom on RTX 4050).
5. **Full Test Suite Health**: **107/107 pytest unit tests passing**.

---

## 2. Binary Shard Architecture & Storage Safety

### Storage Precision & Footprint Arithmetic:
- **Encoding**: SentencePiece vocabulary size is 8,000. Each token ID is encoded as an unsigned 16-bit integer (`np.uint16`, range 0 to 65,535).
- **Token Density**: 2 bytes per token position (1,024 bytes per 512-token sequence).
- **500M Token Production Footprint**:
  - 500,000,000 training tokens = 50 shards of 10M tokens $\approx \mathbf{1{,}000\text{ MB}}$ ($1.00\text{ GB}$).
  - 10,000,000 validation tokens = 1 shard of 10M tokens $\approx \mathbf{20\text{ MB}}$ ($0.02\text{ GB}$).
  - Total Disk Footprint: $\mathbf{1.02\text{ GB}}$ (vs >114 GB available on disk).

### Qualification Shard Build Metrics:
| Metric | Qualification Build Value | 500M Production Target |
|---|---|---|
| **Storage Format** | Raw `uint16` flat binary | Raw `uint16` flat binary |
| **Tokens per Shard** | 500,000 tokens | 10,000,000 tokens |
| **Shard File Size** | 1,000,000 bytes (0.95 MB) | 20,000,000 bytes (19.07 MB) |
| **Total Shard Count** | 6 train + 1 val shards | 50 train + 1 val shards |
| **Total Qualified Tokens** | 2,601,472 tokens | 510,000,000 tokens |
| **Generation Throughput** | ~20,000 tokens/sec | ~25,000 tokens/sec |
| **RAM Footprint during Load** | **< 2 MB** (Memmap/Lazy) | **< 20 MB** (Memmap/Lazy) |
| **Manifests** | `train_manifest.json`, `val_manifest.json` | `train_manifest.json`, `val_manifest.json` |

---

## 3. Resumability & Bitwise Parity Verification

A rigorous 100-step qualification benchmark was performed to prove exact dataset cursor tracking and deterministic optimization:

- **Run A (Continuous Baseline)**: Initialized with `seed=42`, trained continuously from Step 0 to Step 100 on local binary shards.
- **Run B1 (Pre-Interruption)**: Initialized with `seed=42`, trained from Step 0 to Step 50, saved checkpoint at Step 50 (`step_50.pt`), and terminated.
- **Run B2 (Post-Interruption Resumed)**: Initialized fresh process, restored model, optimizer, scheduler, GradScaler, RNG, and `dataset_state` (`shard_idx=1`, `seq_idx=623`, `tokens_seen=819,200`), and trained from Step 50 to Step 100.

### Step-by-Step Trajectory Comparison:

| Global Step | Continuous Run A Loss | Resumed Run B Loss | Learning Rate | Grad Norm | Scaler Scale | Loss $\Delta \|A - B\|$ |
|---|---|---|---|---|---|---|
| **10** | 7.7408 | 7.7408 | 5.46e-04 | 0.900 | 65536 | $0.00\text{e}+00$ |
| **25 (Val)** | 7.2526 (val) | 7.2526 (val) | 5.87e-04 | — | 65536 | $0.00\text{e}+00$ |
| **50** | **6.9010** | **6.9010** | 3.86e-04 | 0.540 | 65536 | $0.00\text{e}+00$ |
| *Checkpoint & Terminate* | *Continuous* | *Process Exit* | — | — | — | *step_50.pt saved* |
| *Process Restart* | *Continuous* | *Resume state* | — | — | — | *Shard 1 restored* |
| **60** | 6.8140 | 6.8140 | 2.92e-04 | 0.555 | 65536 | $0.00\text{e}+00$ |
| **75 (Val)** | 6.7641 (val) | 6.7641 (val) | 2.03e-04 | — | 65536 | $0.00\text{e}+00$ |
| **80** | 6.6193 | 6.6193 | 1.29e-04 | 0.481 | 65536 | $0.00\text{e}+00$ |
| **90** | 6.6906 | 6.6906 | 7.97e-05 | 0.647 | 65536 | $0.00\text{e}+00$ |
| **100 (Final)** | **6.746371** | **6.746371** | 6.02e-05 | 0.687 | 65536 | $\mathbf{0.00\text{e}+00}$ |

### Resumability Verdict:
- **Loss Difference at Step 100**: $\mathbf{0.000000}$ (Bitwise match).
- **Max Parameter Difference across all 42.1M Weights**: $\mathbf{0.000000}$.
- **Dataset Position Recovery**: `start_sequence_offset` correctly advanced data pointer without re-reading or skipping sequences.

---

## 4. Network Isolation & Security

- **Socket Blocking Test**: Replaced `socket.socket` with a failing assertion stub during local training.
- **Result**: **Zero network requests attempted**.
- **Guarantee**: After `build_pretrain_shards.py` completes, training runs 100% offline with zero network latency, DNS vulnerability, or Hugging Face Hub rate-limiting risks.

---

## 5. Hardware Performance & VRAM Profile

| Metric | Measured Value | Safe Threshold | Margin / Headroom |
|---|---|---|---|
| **Peak Tensor Memory** | **3,245.9 MB** | 6,144 MB | +2,898 MB (47.2% headroom) |
| **Total Reserved Memory** | **3,396.0 MB** | 6,144 MB | +2,748 MB (44.7% headroom) |
| **Training Throughput** | **16,500 – 21,780 tok/s** | > 15,000 tok/s | On target |
| **GradScaler Health** | **65,536.0 (Constant)** | > 1.0 | 0 scale drops, 0 skips |
| **DataLoader CPU Overhead** | **< 0.1 ms per micro-batch** | < 5.0 ms | Near-zero I/O overhead |

---

## 6. Checkpoint Infrastructure Compliance

The checkpoint manager adheres strictly to project rules:
- Saved files: `latest.pt`, `best.pt`, `step_<N>.pt`.
- Purged legacy duplicates: No `best_model.pt` or unmanaged `checkpoint_epoch_N.pt`.
- Payload includes:
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict`
  - `scaler_state_dict`
  - `global_step`
  - `tokens_seen`
  - `val_loss`, `current_loss`
  - `config`
  - `dataset_state` (`current_shard_idx`, `current_seq_in_shard`, `global_seq_idx`, `manifest_hash`)
  - `python_rng_state`, `numpy_rng_state`, `torch_rng_state`, `cuda_rng_state`

---

## 7. Windows Unattended Long-Run Operation Guide

To ensure unattended execution over the 30,517-step run (~6.5 hours on RTX 4050):

### 1. Power Management (Run in Administrator PowerShell prior to launch):
```powershell
# Disable screen and system sleep on AC power
powercfg -change -monitor-timeout-ac 0
powercfg -change -standby-timeout-ac 0
powercfg -change -hibernate-timeout-ac 0
```

### 2. Launching Production Pretraining:
```powershell
# Step 1: Generate full 500M binary shards (one-time offline prep)
python scripts/build_pretrain_shards.py --target_train_tokens 500000000 --target_val_tokens 10000000 --shard_tokens 10000000

# Step 2: Launch pretraining run
python scripts/train_pretrain.py --target_tokens 500000000 --run_id mini_r1_500m_pretrain
```

### 3. Resuming from Interruption:
```powershell
python scripts/train_pretrain.py --target_tokens 500000000 --resume_from checkpoints/mini_r1_500m_pretrain/latest.pt
```

---

## 8. Remaining Risks & Mitigations

1. **Hardware Thermals over 6.5 Hours**:
   - *Risk*: Laptop cooling limitations causing thermal throttling from ~21.5K tok/s down to ~16K tok/s.
   - *Mitigation*: Ensure laptop base is elevated for ventilation. Training loop logs per-step throughput and adapts time estimates.
2. **Initial Shard Generation Network Drops**:
   - *Risk*: Transient network drops while downloading smollm-corpus shards from HF Hub.
   - *Mitigation*: Exponential backoff retry wrapper (`load_streaming_dataset_with_retry`) integrated into `build_pretrain_shards.py`.

---

## 9. Final Qualification Verdict

# READY FOR 500M RUN

The pretraining engine is robust, memory-safe, air-gapped, bitwise reproducible upon resumption, and verified across all 107 repository unit tests.
