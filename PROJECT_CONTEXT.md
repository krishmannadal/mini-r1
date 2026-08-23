# MINI-R1: MASTER PROJECT CONTEXT & AUTHORITATIVE HANDOFF DOCUMENT

**Document Version**: 1.0.0  
**Date**: 2026-08-19  
**Repository**: `Mini-R1` **Document Purpose**: Authoritative, self-contained project handoff. This document contains the full technical context, architectural specifications, dataset provenance, experimental history, empirical results, known inconsistencies, and exact next steps for Mini-R1. A new AI assistant or researcher can continue the project immediately without historical session context.

---

## Table of Contents

1. [Project Identity & Research Philosophy](#1-project-identity--research-philosophy)
2. [Hardware & Execution Environment](#2-hardware--execution-environment)
3. [Software Stack & Dependencies](#3-software-stack--dependencies)
4. [Model Architecture Specification](#4-model-architecture-specification)
5. [Tokenizer Specification](#5-tokenizer-specification)
6. [Dataset History & Strategic Hierarchy](#6-dataset-history--strategic-hierarchy)
7. [Base Pretraining Infrastructure & Data Shards](#7-base-pretraining-infrastructure--data-shards)
8. [Chronological Task History (Tasks 1 – 15C)](#8-chronological-task-history-tasks-1--15c)
9. [Base Pretraining Results (500M Tokens)](#9-base-pretraining-results-500m-tokens)
10. [Base Model Zero-Shot Reasoning Baseline (Task 14)](#10-base-model-zero-shot-reasoning-baseline-task-14)
11. [Reasoning SFT Dataset Construction (Task 15A)](#11-reasoning-sft-dataset-construction-task-15a)
12. [Masked SFT Training Engine (Task 15B)](#12-masked-sft-training-engine-task-15b)
13. [Controlled Direct vs CoT Experiment Results (Task 15C)](#13-controlled-direct-vs-cot-experiment-results-task-15c)
14. [Known Inconsistencies & Contradictions (Scientific Data Audit)](#14-known-inconsistencies--contradictions-scientific-data-audit)
15. [Current Scientific Hypotheses: Why Did CoT SFT Hurt?](#15-current-scientific-hypotheses-why-did-cot-sft-hurt)
16. [Hyperparameter Decisions & Engineering Rationale](#16-hyperparameter-decisions--engineering-rationale)
17. [Artifacts, Checkpoints & Cryptographic Hashes](#17-artifacts-checkpoints--cryptographic-hashes)
18. [Repository File & Directory Map](#18-repository-file--directory-map)
19. [Common Operational Commands & Workflows](#19-common-operational-commands--workflows)
20. [AI Collaboration Policy & Role Specialization](#20-ai-collaboration-policy--role-specialization)
21. [Research Study Queue for Human Partner](#21-research-study-queue-for-human-partner)
22. [CURRENT STATE](#current-state)
23. [CURRENT SCIENTIFIC RESULT](#current-scientific-result)
24. [WHAT WE DO NOT KNOW](#what-we-do-not-know)
25. [NEXT TASK](#next-task)
26. [DO NOT DO YET](#do-not-do-yet)
27. [TRANSFER INSTRUCTIONS](#transfer-instructions)

---

## 1. Project Identity & Research Philosophy

### 1.1 Project Identity
- **Project Name**: Mini-R1
- **Project Type**: Small Language Model (SLM) mathematical reasoning research project.
- **Classification**: Base SLM undergoing reasoning specialization (do **not** refer to the model as a fully realized "reasoning model" yet; it is an experimental research prototype).

### 1.2 Core Research Goals
1. Implement a clean, modern decoder-only Transformer **entirely from scratch** (zero high-level framework wrappers, zero pretrained causal model weights, zero black-box attention abstractions).
2. Pretrain the 42.1M parameter model on a high-quality educational token corpus (500M tokens).
3. Investigate the empirical impact of **Supervised Fine-Tuning (SFT)**: Direct-Answer supervision vs explicit Chain-of-Thought (CoT) supervision.
4. Investigate **Outcome-Based Reinforcement Learning** (GRPO / relative-reward policy optimization) to explore whether verification/RL can overcome SFT capacity bottlenecks.
5. **Final Deployment Goal**: Package the fine-tuned/RL-trained model into a standalone FastAPI server running on the RTX 4050 GPU, demonstrating low-latency interactive inference requested from a separate networked client device.

### 1.3 Research Philosophy & Engineering Rules
- **No High-Level Framework Abstractions**: Core Transformer, multi-head attention, loss formulation, data collators, training loops, and evaluation metrics are written by hand in PyTorch.
- **Scientific Honesty & Negative Results**: Negative results are valid scientific data. Do not hide failures, smooth over contradictions, or claim genuine mathematical reasoning from superficial reasoning-like text.
- **Strict Base Model Immutability**: The best base pretraining checkpoint (`checkpoints/mini_r1_500m_pretrain/best.pt`) is **immutable** and must never be overwritten, modified, or degraded.
- **Mandatory Project Rules** (`.agents/rules/projectrule.md`):
  1. Never implement future tasks unless explicitly instructed.
  2. Before writing code: Explain the design, state mathematical/engineering assumptions, state which files will change, implement only the current task, provide verification commands, and stop immediately after the current task.
  3. Never hide important functionality behind a high-level abstraction when the goal is understanding the underlying mechanism.
  4. Prefer simple, readable implementations over clever implementations.

---

## 2. Hardware & Execution Environment

The entire training and evaluation pipeline is sized, optimized, and verified on a single dedicated consumer laptop GPU:

| Hardware / System Component | Exact Discovered Specification | Verification Method |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 4050 Laptop GPU | `torch.cuda.get_device_name(0)` |
| **VRAM Capacity** | **6.0 GB GDDR6** (6,144 MB physical) | `torch.cuda.get_device_properties()` |
| **GPU Architecture** | Ada Lovelace (SM 8.9, Compute Capability 8.9) | Hardware diagnostic |
| **Driver Version** | 581.57 | NVIDIA System Information |
| **Host Operating System** | Windows 11 Home / Pro (Build `Windows-10-10.0.26200-SP0`) | `platform.platform()` |
| **System Shell** | PowerShell (pwsh / powershell.exe) | Environment check |
| **Storage / Disk Capacity** | Total: **447 GB** \| Used: **364 GB** \| Free: **82 GB** | `shutil.disk_usage()` |
| **CUDA Runtime** | CUDA 12.4 (Driver CUDA support up to 13.0) | `torch.version.cuda` |
| **PyTorch Version** | `2.6.0+cu124` | `torch.__version__` |
| **Python Version** | Python 3.11.0 (64-bit AMD64, MSC v.1933) | `sys.version` |

---

## 3. Software Stack & Dependencies

All dependencies are pinned and verified in `requirements.txt`:

```text
torch==2.6.0+cu124
sentencepiece>=0.2.0
pyyaml>=6.0
pytest>=8.0.0
datasets>=2.18.0
numpy>=1.26.0
```

### Architecture Rule on External Libraries:
- **Hugging Face `datasets`**: Used solely for initial dataset downloading/streaming.
- **Hugging Face `transformers` / `accelerate` / `trl` / `deepspeed`**: **STRICTLY FORBIDDEN**. All modeling, attention, training, collation, optimization, and evaluation logic are implemented directly in `src/`.

---

## 4. Model Architecture Specification

The model architecture is defined in `src/model.py`, `src/embeddings.py`, `src/attention.py`, and `src/transformer_block.py`, configured via `configs/model.yaml`.

### 4.1 Exact Hyperparameters

| Hyperparameter | Config Key | Value | Engineering / Mathematical Rationale |
| :--- | :--- | :---: | :--- |
| **Architecture Type** | -- | **Decoder-Only Causal Transformer** | Standard autoregressive language modeling |
| **Total Parameters** | -- | **42,132,480 (~42.1M)** | Max capacity fitting 6 GB VRAM with batch=32, seq=512 |
| **Vocabulary Size** | `vocab_size` | **8,000** | SentencePiece BPE; compact embedding table ($4.1\text{M}$ params) |
| **Hidden Dimension ($d_{\text{model}}$)** | `d_model` | **512** | Embedding and residual stream width |
| **Decoder Layers** | `n_layers` | **12** | Deep representation hierarchy |
| **Attention Heads** | `n_heads` | **8** | Multi-head division; $d_k = d_{\text{model}} / n_{\text{heads}} = 64$ |
| **Head Dimension ($d_k$)** | `d_k` | **64** | Standard scaled dot-product head width |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | `d_ff` | **2,048** | Standard $4 \times d_{\text{model}}$ expansion |
| **Max Context Length** | `max_seq_len` | **512** | Captures complete multi-step reasoning traces |
| **Dropout Probability** | `dropout` | **0.0** | Standard 0.0 for pretraining/SFT efficiency |
| **Linear Bias Terms** | `bias` | **False** | Modern simplified parameterization (LLaMA/GPT-3 style) |
| **Weight Tying** | `tie_word_embeddings` | **True** | `lm_head.weight` is tied to `embeddings.tok_emb.weight` |

### 4.2 Parameter Count Derivation
$$\begin{aligned}
N_{\text{emb}} &= V \cdot d_{\text{model}} + T_{\text{max}} \cdot d_{\text{model}} = (8000 \cdot 512) + (512 \cdot 512) = 4{,}096{,}000 + 262{,}144 = 4{,}358{,}144 \\
N_{\text{attn\_per\_layer}} &= 4 \cdot d_{\text{model}}^2 = 4 \cdot (512 \cdot 512) = 1{,}048{,}576 \quad (W_q, W_k, W_v, W_o) \\
N_{\text{ffn\_per\_layer}} &= 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 2 \cdot (512 \cdot 2048) = 2{,}097{,}152 \quad (W_1, W_2) \\
N_{\text{ln\_per\_layer}} &= 2 \cdot 2 \cdot d_{\text{model}} = 2 \cdot (512 + 512) = 2{,}048 \quad (\text{LN}_1, \text{LN}_2 \text{ weights and biases}) \\
N_{\text{block\_per\_layer}} &= 1{,}048{,}576 + 2{,}097{,}152 + 2{,}048 = 3{,}147{,}776 \\
N_{\text{all\_12\_blocks}} &= 12 \cdot 3{,}147{,}776 = 37{,}773{,}312 \\
N_{\text{final\_ln}} &= 2 \cdot d_{\text{model}} = 1{,}024 \\
N_{\text{lm\_head}} &= 0 \quad (\text{tied with token embeddings}) \\
\mathbf{N_{\text{total}}} &= 4{,}358{,}144 + 37{,}773{,}312 + 1{,}024 = \mathbf{42{,}132{,}480}
\end{aligned}$$

### 4.3 Key Architectural Details
- **Normalization**: Pre-LayerNorm architecture ($x = x + \text{Attn}(\text{LN}(x))$, $x = x + \text{FFN}(\text{LN}(x))$) with final `ln_f` before `lm_head`.
- **Positional Encoding**: Learned 1D positional embeddings (`nn.Embedding(max_seq_len, d_model)` in `src/embeddings.py`), summed directly with token embeddings.
- **Activation**: Gaussian Error Linear Unit (`nn.GELU()`) in `FeedForwardNetwork`.
- **Causal Self-Attention**: Handcrafted in `src/attention.py`. Lower-triangular boolean mask buffer registered as non-persistent buffer (`torch.tril`). Queries, keys, and values explicitly projected, reshaped to `(B, n_heads, T, d_k)`, scaled by $1/\sqrt{d_k}$, masked with $-\infty$ for $j > i$, normalized via `F.softmax(dim=-1)`, aggregated with $V$, and projected through $W_o$.
- **Weight Initialization**: Gaussian distribution with $\mu = 0.0, \sigma = 0.02$. LayerNorm weights initialized to $1.0$, biases to $0.0$.

---

## 5. Tokenizer Specification

The tokenizer is implemented in `src/tokenizer.py` and trained via `scripts/train_tokenizer.py`.

- **Library**: Google `SentencePiece` (`sentencepiece.SentencePieceProcessor`)
- **Algorithm**: Byte-Pair Encoding (`bpe`)
- **Configuration**:
  - `vocab_size = 8000`
  - `character_coverage = 1.0` (preserves all mathematical symbols and digits)
  - `split_digits = True` (splits numbers into individual digit subwords e.g. `1`, `5`, `0`, preventing digit-combination vocab explosion)
  - `byte_fallback = True` (handles arbitrary UTF-8 characters without `<unk>` failure)
  - `normalization_rule_name = "identity"` (no destructive lowercase/NFKC stripping)
- **Special Token ID Mapping**:
  - `unk_id = 0` (`<unk>`)
  - `bos_id = 1` (`<s>`)
  - `eos_id = 2` (`</s>`)
  - `pad_id = 3` (`<pad>`)
  - `think_start_id = 4` (`<think>`) [User-defined atomic symbol]
  - `think_end_id = 5` (`</think>`) [User-defined atomic symbol]
  - `answer_start_id = 6` (`<answer>`) [User-defined atomic symbol]
  - `answer_end_id = 7` (`</answer>`) [User-defined atomic symbol]
- **Tokenizer Artifact**: `artifacts/tokenizer/tokenizer.model`
  - **MD5**: `e7ef63605f920bf411f0c897c44e31e3`
  - **SHA256**: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`
- **Training Corpus**: Trained on `data/clean/tokenizer_corpus.txt`, a representative mixture of mathematical problems, reasoning steps, natural language prose, code snippets, and structural delimiter tags.

---

## 6. Dataset History & Strategic Hierarchy

### 6.1 Phase 1 (Initial Setup) vs Phase 2 (Decoupled Pretraining)
- **Initial Setup (Tasks 2–6)**: The project originally explored training directly on math reasoning datasets (GSM8K, SVAMP, AQuA-RAT).
- **Strategic Pivot (Tasks 10–11)**: Research forensic analysis revealed that training a 42.1M parameter model from scratch on ~21M tokens of narrow math problem templates amounts to memorizing brittle templates without general language grounding.
- **Three-Tier Decoupled Hierarchy**:
  1. **Base Pretraining (500M Tokens)**: General educational text (`cosmopedia-v2` + `fineweb-edu-dedup`) to build foundational language syntax, grammar, and world representations.
  2. **Reasoning SFT (Task 15A–15C)**: GSM8K train split (6,973 examples) formatted for Direct vs CoT supervision.
  3. **Golden Evaluation Holdouts**: GSM8K test (1,319), SVAMP test (300), AQuA-RAT test (254), and 10M general validation tokens.

### 6.2 Base Pretraining Corpus Composition
- **Source Repository**: `HuggingFaceTB/smollm-corpus`
- **Mixture Ratios (Enforced strictly by token count)**:
  - **60% `cosmopedia-v2`**: Synthetic textbooks, course units, and tutorials generated by Mixtral-8x22B.
  - **40% `fineweb-edu-dedup`**: Deduplicated, high-quality educational web pages and articles.
  - **0% `python-edu` (EXCLUDED)**: Explicitly removed during Task 11B due to git repository blob schema incompatibilities.
- **Contiguous Token Packing**:
  - Documents are concatenated with `</s>` (EOS, id=2) delimiters into contiguous 512-token chunks.
  - Eliminates padding waste (100% token density) and prevents the 58.4% data loss associated with truncating long textbook chapters at the 512 boundary.

---

## 7. Base Pretraining Infrastructure & Data Shards

The pretraining engine was constructed in Task 12 (`src/pretrain_dataset.py`, `scripts/build_pretrain_shards.py`, `scripts/train_pretrain.py`) to replace in-memory preloading with high-performance, air-gapped binary streaming.

### 7.1 Binary Shard Architecture
- **Format**: Flat raw binary files storing token IDs as unsigned 16-bit integers (`np.uint16`, 2 bytes/token).
- **Shard Sizing**: 10,000,000 tokens per shard (20,000,000 bytes $\approx 19.07\text{ MB}$ per shard).
- **Total Storage**: 50 training shards ($500\text{M}$ tokens $\approx 1.00\text{ GB}$) + 1 validation shard ($10\text{M}$ tokens $\approx 20\text{ MB}$). Total disk footprint: **1.02 GB**.
- **RAM Footprint**: Under 20 MB during training via memory-mapped streaming (`np.memmap` / `ShardedPretrainDataset`).
- **Network Isolation**: 100% air-gapped. Training runs offline with zero network sockets.

### 7.2 Pretraining Hyperparameters & Configuration (`configs/pretrain_500m.yaml`)
- **Target Tokens**: $500{,}000{,}000\text{ tokens}$
- **Sequence Length**: 512 tokens
- **Micro-Batch Size**: 8 sequences ($4{,}096\text{ tokens}$)
- **Gradient Accumulation Steps**: 4 steps
- **Effective Batch Size**: 32 sequences ($16{,}384\text{ tokens per optimizer update}$)
- **Total Optimizer Steps**: $30{,}518\text{ steps}$
- **Precision**: FP16 Mixed Precision with `torch.amp.GradScaler`
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight\_decay} = 0.01, \epsilon = 1.0 \times 10^{-8}$)
- **Learning Rate Schedule**:
  - Peak Learning Rate: $6.0 \times 10^{-4}$
  - Minimum Learning Rate: $6.0 \times 10^{-5}$ (10x decay)
  - Warmup: 100 optimizer steps (linear warmup)
  - Decay: Cosine decay over $30{,}518$ steps
- **Gradient Clipping**: `max_grad_norm = 1.0`
- **Validation Frequency**: Every 500 optimizer steps on 10M held-out tokens
- **Checkpoint Schedule**: Periodic step checkpoints every 1,000 steps; continuous rotation of `latest.pt` and `best.pt`

---

## 8. Chronological Task History (Tasks 1 – 15C)

| Task ID | Status | Purpose & Scope | Key Files | Key Empirical Result | Important Scientific Conclusion |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Task 1** | ✅ Complete | Environment & Hardware Validation | `scripts/gpu_check.py`, `tests/test_environment.py` | PyTorch 2.6.0+cu124 on RTX 4050 (6GB VRAM); FP16/BF16 autocast verified | Hardware platform validated for 42M parameter training. |
| **Task 2** | ✅ Complete | Tokenizer Training & Special Tokens | `src/tokenizer.py`, `scripts/train_tokenizer.py`, `artifacts/tokenizer/` | 8,000-vocab SentencePiece BPE model with atomic `<think>` and `<answer>` tags | Atomic special tokens prevent subword fragmentation during reasoning. |
| **Task 3** | ✅ Complete | Multi-Dataset Reasoning Pipeline | `src/preprocessing.py`, `src/datasets.py`, `tests/test_dataset_pipeline.py` | Unified parsing & schema for GSM8K, SVAMP, and AQuA-RAT | Structured train/dev/test separation established with 0 leakage. |
| **Task 4** | ✅ Complete | Scratch Transformer Architecture | `src/embeddings.py`, `src/attention.py`, `src/transformer_block.py`, `src/model.py` | 42.1M-param Pre-LN Transformer with tied embeddings and causal mask | Architecture verified mathematically and tensor-shape tested. |
| **Task 5** | ✅ Complete | Loss, Teacher Forcing & Optimization | `src/loss.py`, `src/optimizer.py`, `src/training_step.py` | Causal shifting ($L_{\text{CE}}$), decoupled AdamW, single-batch loss drop | Backpropagation and gradient updates verified without NaN/overflow. |
| **Task 6** | ✅ Complete | Complete Training Infrastructure | `src/torch_dataset.py`, `src/dataloader.py`, `src/checkpoint.py`, `src/trainer.py` | Full multi-epoch trainer, dynamic collator, atomic checkpointing | Complete end-to-end training pipeline operational (**68/68 tests passing**). |
| **Task 7** | ✅ Complete | Architecture Scaling & Memory Profiling | `configs/model.yaml`, `src/memory.py`, `scripts/estimate_memory.py` | Scaled to 12 layers, $d=512$, $h=8$, $d_{\text{ff}}=2048$ (42.1M params); 3.25 GB peak VRAM | 42.1M model fits comfortably within 6 GB VRAM with 47% safety headroom. |
| **Task 8** | ✅ Complete | Scaled System Qualification | `tests/test_scaled_training.py`, `scripts/run_batch_qualification.py` | 50-step batch qualification with GradScaler and FP16 AMP | Stable throughput ~21,500 tok/s; zero skipped steps. |
| **Task 8C / 9** | ✅ Complete | Learning Rate Sweep & Selection | `scripts/run_lr_sweep.py`, `scripts/run_long_qualification.py` | Tested $1e-4, 3e-4, 6e-4$. $6e-4$ produced fastest monotonic descent to val loss 6.74 | $6.0 \times 10^{-4}$ selected as optimal pretraining peak learning rate. |
| **Task 10** | ✅ Complete | General Pretraining Streaming & Packing | `src/general_dataset.py`, `scripts/build_pretrain_corpus.py` | Contiguous token packing with EOS delimiters into 512-token chunks | Eliminates 58.4% data loss from naive document truncation. |
| **Task 11 / 11B** | ✅ Complete | Final Corpus Qualification | `src/corpus_qualifier.py`, `reports/task11b_final_qualification.md` | 60% `cosmopedia-v2` / 40% `fineweb-edu-dedup`; excluded `python-edu`; 10M val set | Corpus qualified for unattended 500M token pretraining. |
| **Task 12** | ✅ Complete | Production Sharded Pretraining Engine | `src/pretrain_dataset.py`, `scripts/build_pretrain_shards.py`, `scripts/train_pretrain.py` | Binary `uint16` sharding (<20MB RAM); 100% air-gapped; exact bitwise resume | Infrastructure qualified for 500M run (**107/107 tests passing**). |
| **Task 13** | ✅ Complete | 500M-Token Base Pretraining Run | `checkpoints/mini_r1_500m_pretrain/best.pt`, `reports/task13_base_pretraining_results.md` | $500{,}000{,}256\text{ tokens}$ in 8.02 hrs. Val loss $8.37 \to 2.8726$ (PPL **17.68**) | Base language pretraining successfully completed at 17,316 tok/s. |
| **Task 14** | ✅ Complete | Base Model Zero-Shot Evaluation | `scripts/evaluate_base_model.py`, `reports/task14_base_model_evaluation.md` | Zero-shot accuracy: GSM8K 0.76%, SVAMP 3.00%, AQuA 17.32%; 0.0% format compliance | Base model exhibits 0% native reasoning syntax; sets immutable baseline. |
| **Task 15A** | ✅ Complete | Reasoning SFT Dataset Construction | `scripts/build_reasoning_sft_dataset.py`, `data/reasoning/`, `reports/task15a_reasoning_dataset.md` | 6,973 Train / 500 Dev paired GSM8K splits for Direct vs CoT; 0 test leakage | Identical problem IDs created; revealed 2.32x token multiplier in CoT. |
| **Task 15B** | ✅ Complete | Masked SFT Engine Implementation | `src/sft_dataset.py`, `src/sft_trainer.py`, `scripts/run_sft_smoke_test.py` | Prompt tokens masked (`-100`); loss computed solely on completions | Masked causal LM SFT verified on both branches (**140/140 tests passing**). |
| **Task 15C** | ✅ Complete | Controlled Direct vs CoT Experiment | `scripts/train_sft_experiment.py`, `scripts/evaluate_sft_experiment.py`, `reports/task15c_direct_vs_cot.md` | Direct GSM8K **2.73%** vs CoT **1.21%** (Delta **-1.52%**, McNemar $p=0.0051$) | **`COT HURTS`**: CoT SFT degraded reasoning accuracy due to 82.6% repetition loops. |

---

## 9. Base Pretraining Results (500M Tokens)

Pretraining completed on 2026-08-18 with zero errors or unhandled exceptions:

- **Total Tokens Trained**: **$500{,}000{,}256\text{ tokens}$**
- **Total Optimizer Steps**: **$30{,}518\text{ steps}$**
- **Training Wall-Clock Time**: **8.02 hours** ($28{,}874.3\text{ seconds}$)
- **Average Throughput**: **$17{,}316.5\text{ tokens/sec}$** (Peak: ~$21{,}700\text{ tok/s}$)
- **Peak VRAM Allocated**: **$3{,}409.7\text{ MB}$** (44.6% safety headroom on 6GB card)
- **Numerical Stability**: `GradScaler` stable at $131{,}072.0$; only 11 skipped steps out of 30,518 ($0.036\%$).
- **Loss Progression**:
  - Initial Loss: Train $8.3418$ $\to$ Val $8.3674$ (PPL $4{,}304.45$)
  - Best Validation Loss: **$2.8726$** (PPL **$17.68$**) at Step 18,000
  - Final Loss (Step 30,518): Train $3.0807$ $\to$ Val $2.8726$ (PPL $17.68$)
- **Why `best.pt` (Step 18,000) is the Canonical Base Checkpoint**:
  Validation loss reached its minimum of $2.8726$ at step 18,000 and entered an asymptotic plateau through step 30,518. Checkpoint `best.pt` captures the optimal representation state prior to slight late-stage training loss oscillation.

---

## 10. Base Model Zero-Shot Reasoning Baseline (Task 14)

Evaluated on 2026-08-18 using the frozen `best.pt` checkpoint across all evaluation benchmarks:

### 10.1 Quantitative Results Summary

| Benchmark | Test Split Size | Decoding Strategy | Strict Accuracy | Fallback Accuracy | Extraction Success Rate | Format Compliance Rate | Mean Output Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K Test** | 1,319 | Greedy ($T=0.0$) | **0.76%** (10/1319) | 0.76% | 48.1% (634/1319) | **0.0%** (0/1319) | 125.7 |
| **GSM8K Test** | 1,319 | Sampling ($T=0.7$) | **0.91%** (12/1319) | 0.91% | 51.0% (673/1319) | **0.0%** (0/1319) | 124.7 |
| **SVAMP Test** | 300 | Greedy ($T=0.0$) | **3.00%** (9/300) | 3.00% | 50.7% (152/300) | **0.0%** (0/300) | 125.0 |
| **AQuA-RAT Test** | 254 | Greedy ($T=0.0$) | **17.32%** (44/254) | 17.32% | 91.3% (232/254) | **0.0%** (0/254) | 125.9 |

### 10.2 Forensic Observations on the Base Model
1. **Zero Delimiter Knowledge**: 0.0% compliance with `<think>` or `<answer>` markup. Base generations are standard textbook continuations.
2. **Spurious Accuracy Artifacts**:
   - The 0.76% GSM8K accuracy was entirely due to fallback regex extraction finding accidental matching numbers in unrelated textbook prose (e.g. "Section 4: Creating Your Mosaic" containing digit `4` matching gold answer `4`).
   - The 17.32% AQuA accuracy is below the random-guess baseline of 20.00% (5 multiple-choice options).
3. **General Language Modeling Match**: Evaluated on 10M validation tokens, achieving loss **2.8724** and PPL **17.68**, confirming exact weight immutability.

---

## 11. Reasoning SFT Dataset Construction (Task 15A)

Constructed in `data/reasoning/` using `openai/gsm8k` (train split, 7,473 total examples):

- **Partition (Deterministic Seed 42)**:
  - **SFT Train Split**: **6,973 examples**
  - **SFT Dev Split**: **500 examples**
- **Condition Formats**:
  - **Direct-Answer SFT (Control)**:
    ```text
    Question: {question}
    <answer>{final_answer}</answer>
    ```
  - **Chain-of-Thought SFT (Treatment)**:
    ```text
    Question: {question}
    <think>
    {natural_language_reasoning}
    </think>
    <answer>{final_answer}</answer>
    ```
- **Token Accounting & Confound Asymmetry**:
  - Direct Train Total Tokens: **508,894 tokens** (Mean: 73.0 tok/ex)
  - CoT Train Total Tokens: **1,178,275 tokens** (Mean: 169.0 tok/ex)
  - **Token Multiplier Confound**: CoT exposes the model to **2.32x** more total tokens and **19.05x** more completion tokens (2,106,652 vs 110,594) across the exact same 6,973 problem instances.
- **Leakage Verification**: SHA-256 hash and exact-string collision checks confirmed **0 collisions** against GSM8K test, SVAMP test, and AQuA test.

---

## 12. Masked SFT Training Engine (Task 15B)

Implemented in `src/sft_dataset.py` and `src/sft_trainer.py`:

- **Prompt Masking Formulation**:
  $$\text{labels}[i] = \begin{cases} -100 & \text{if } 0 \le i < L_{\text{prompt}} \quad (\text{Prompt context ignored in loss}) \\ x_i & \text{if } L_{\text{prompt}} \le i < T \quad (\text{Completion active in loss}) \end{cases}$$
- **Causal Shift**: Logits at position $t = L_{\text{prompt}} - 1$ predict target at $t = L_{\text{prompt}}$ (first completion token `<think>` or `<answer>`).
- **SFT Hyperparameters**:
  - Base Checkpoint: `checkpoints/mini_r1_500m_pretrain/best.pt`
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.01$)
  - Learning Rate: Peak $=1.0 \times 10^{-4}$ (reduced from pretraining $6e-4$ to prevent catastrophic forgetting), Min $=1.0 \times 10^{-5}$, 50 warmup steps, cosine decay.
  - Effective Batch Size: 32 (micro-batch 8, gradient accumulation 4)
  - Total Training Budget: Exactly 650 optimizer steps ($\sim 3.0$ epochs)
  - Validation: Every 50 optimizer steps on 500-example GSM8K dev split

---

## 13. Controlled Direct vs CoT Experiment Results (Task 15C)

Completed on 2026-08-19. Full report saved in `reports/task15c_direct_vs_cot.md`.

### 13.1 Primary & Secondary Reasoning Benchmarks

| Benchmark | Test Examples | Pretrained Base | Direct-Answer SFT (Strict) | Chain-of-Thought SFT (Strict) | Strict CoT Delta ($\Delta$) | Direct SFT (Fallback) | CoT SFT (Fallback) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K Test (Primary)** | 1,319 | 0.76% | **2.73%** (36/1319) | **1.21%** (16/1319) | **-1.52%** | 2.73% | 1.97% |
| **SVAMP Test (Secondary OOD)** | 300 | 3.00% | **1.67%** (5/300) | **0.67%** (2/300) | **-1.00%** | 1.67% | 0.67% |
| **AQuA-RAT Test (Tertiary MC)** | 254 | 17.32% | **0.00%** (0/254) | **0.00%** (0/254) | **+0.00%** | 8.66% | 5.12% |

### 13.2 Format Compliance & Generation Mechanics

| Condition | `<answer>` Compliance | `<think>` Compliance | Full Format Rate | Mean Gen Tokens | Median Gen Tokens | EOS Termination Rate | 4-gram Repetition Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct-Answer SFT** | **100.0%** | 0.0% | 0.0% | 4.6 | 4.0 | **100.0%** | **0.0%** |
| **Chain-of-Thought SFT** | 40.8% | 40.5% | 39.1% | 158.6 | 192.0 | 41.5% | **82.6%** |

### 13.3 Complexity Breakdown (GSM8K Test)

| Step Complexity Bucket | Total Problems | Direct SFT Accuracy | CoT SFT Accuracy | Strict Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **2-step Reasoning** | 315 | **4.13%** (13/315) | 1.90% (6/315) | **-2.22%** |
| **3-step Reasoning** | 360 | **2.50%** (9/360) | 1.39% (5/360) | **-1.11%** |
| **4+-step Reasoning** | 644 | **2.17%** (14/644) | 0.78% (5/644) | **-1.40%** |

### 13.4 Answer Magnitude Breakdown (GSM8K Test)

| Answer Magnitude Range | Problem Count | Direct SFT Accuracy | CoT SFT Accuracy | Strict Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **1 – 10** | 289 | 1.73% | 1.73% | **+0.00%** |
| **11 – 100** | 617 | **3.73%** | 1.46% | **-2.27%** |
| **101 – 1,000** | 286 | **2.80%** | 0.70% | **-2.10%** |
| **> 1,000** | 127 | 0.00% | 0.00% | **+0.00%** |

### 13.5 Statistical Significance (McNemar's Paired Discordance Test)
- Total paired samples: $N = 1,319$
- Both correct ($n_{11}$): 3
- Both incorrect ($n_{00}$): 1,270
- CoT correct / Direct incorrect ($n_{01}$): 13
- Direct correct / CoT incorrect ($n_{10}$): 33
- McNemar $\chi^2$ statistic: **7.8478**
- **Two-Tailed $p$-value**: **$5.0882 \times 10^{-3}$ ($p < 0.01$, statistically significant)**

### 13.6 Qualitative Forensic Taxonomy (CoT Traces)
1. **Correct Answer + Plausible Reasoning (0.2%, 2 traces)**: Genuine multi-step reasoning success.
2. **Correct Answer + Flawed Reasoning (0.0%, 0 traces)**: Zero fortuitous cancellations.
3. **Incorrect Answer + Plausible Reasoning (5.8%, 77 traces)**: Coherent narrative structure with intermediate arithmetic drift (e.g. $150 + 150 = 350$).
4. **Incorrect Answer + Incoherent Reasoning (0.1%, 1 trace)**: Severe hallucination.
5. **Format Failure (11.3%, 149 traces)**: Missing closing delimiters or unclosed `<think>` tags due to token limit exhaustion.
6. **Repetition / Degeneration (82.6%, 1,090 traces)**: Severe cyclic phrase loops during greedy decoding.

### 13.7 Authoritative Scientific Verdict
**FINAL VERDICT: `COT HURTS`**

*Scientifically conservative phrasing*:
> "Under the evaluated SFT configuration on a 42.1M-parameter model, explicit Chain-of-Thought supervision degraded GSM8K exact-match performance relative to Direct-Answer supervision (2.73% vs 1.21%, $p = 0.0051$)."

---

## 14. Known Inconsistencies & Contradictions (Scientific Data Audit)

The following inconsistencies exist across repository reports and must be explicitly recorded and audited before drawing definitive conclusions:

### 14.1 RESOLVED SCIENTIFIC DATA CONSISTENCY ISSUE: General Language Perplexity Shift (Audited in Task 16A)
- **Status**: **RESOLVED** via independent re-evaluation audit in Task 16A (`reports/task16a_evaluation_audit.md`).
- **Verified Ground Truth**:
  - Base Pretrained PPL (10M tokens): **17.68** (Loss 2.872419, verified match to pretraining baseline within $1.95 \times 10^{-4}$ nats)
  - Direct SFT PPL (10M tokens): **21.83** (Loss 3.083407, verified **+23.49%** shift)
  - CoT SFT PPL (10M tokens): **23.58** (Loss 3.160299, verified **+33.36%** shift)
- **Root Cause of Discrepancy**:
  The Section 7 table in Task 15C contained authentic empirical metrics. The Section 10 discussion text contained un-interpolated drafting template placeholder strings (`17.72` / `17.84`). The placeholder text has been corrected, and all data records are now 100% consistent.

### 14.2 Roadmap vs Completed Task Numbering
- **README.md (Lines 259–262)**: Lists Tasks 11, 12, 13 as "Pending" from an older roadmap draft.
- **Actual Repository State**: Tasks 11 (Corpus Qualification), 11B (Final Qualification), 12 (Production Engine), 13 (500M Pretraining), 14 (Base Evaluation), 15A (Dataset Construction), 15B (Masked SFT Engine), and 15C (Direct vs CoT Experiment) are **100% complete**.

---

## 15. Current Scientific Hypotheses: Why Did CoT SFT Hurt?

The central research question for Mini-R1 is now: **Why did explicit CoT supervision degrade performance at 42.1M parameters, and what mechanisms drove the 82.6% repetition rate?**

### Candidate Hypotheses to Disentangle in Task 16:
1. **Autoregressive Error Compounding & Working Memory Limits**:
   - In Direct SFT, the model makes a single token prediction for the answer (0.0% repetition, 100% format compliance).
   - In CoT SFT, the model must maintain coherent working memory across ~160 tokens. An arithmetic mistake at step 1 irrecoverably corrupts all downstream steps.
2. **Greedy Decoding Degeneration & Absence of Repetition Penalties**:
   - Deterministic greedy search ($T=0.0$) on small causal LMs frequently enters self-reinforcing cyclic attractors. 82.6% of CoT generations entered degenerate 4-gram repetition loops.
3. **Supervision Volume & Optimization Asymmetry**:
   - CoT SFT exposed the model to 19.05x more completion tokens than Direct SFT over the exact same 650 optimizer updates and learning rate. CoT may be undertrained per completion token or suffering from gradient dilution.
4. **Exposure Bias during SFT**:
   - During training, the model is teacher-forced on perfect gold reasoning tokens. At inference, its own slightly imperfect generated prefix pushes it out of distribution, triggering repetition.
5. **Model Capacity Emergence Threshold**:
   - 42.1M parameters may simply be below the capacity threshold required to simulate multi-step deduction, causing CoT to act as a distractor rather than a scaffold.

---

## 16. Hyperparameter Decisions & Engineering Rationale

| Phase | Parameter | Value | Rationale |
| :--- | :--- | :---: | :--- |
| **Model** | `vocab_size` | 8,000 | Keeps embedding table to 4.1M weights, leaving 38M weights for transformer blocks. |
| **Model** | `tie_word_embeddings` | True | Saves 4.1M parameters; stabilizes representation geometry between input and output. |
| **Model** | `bias` | False | Improves throughput, reduces parameter count by ~0.5%, mirrors modern LLaMA architectures. |
| **Pretraining** | Peak LR | $6.0 \times 10^{-4}$ | Selected via Task 8C/9 empirical sweep over $1e-4, 3e-4, 6e-4$. Achieved lowest validation loss. |
| **Pretraining** | Warmup Steps | 100 | Prevents early gradient shock with tied embeddings and random initialization. |
| **Pretraining** | Effective Batch Size | 32 (16,384 toks) | Optimal balance between gradient estimation stability and VRAM allocation. |
| **SFT** | Peak LR | $1.0 \times 10^{-4}$ | Lowered 6x from pretraining to prevent catastrophic destruction of base representations. |
| **SFT** | Warmup Steps | 50 | Allows gentle adaptation to masked SFT target distributions. |
| **SFT** | Steps | 650 | Matches ~3.0 full epochs over 6,973 GSM8K training examples. |

---

## 17. Artifacts, Checkpoints & Cryptographic Hashes

### 17.1 Checkpoints

| Checkpoint Path | Step | Val Loss | Size | SHA-256 Hash | Purpose / Status |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `checkpoints/mini_r1_500m_pretrain/best.pt` | 18,000 | 2.8726 | ~508 MB | `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4` | **IMMUTABLE BASE CHECKPOINT** |
| `checkpoints/mini_r1_500m_pretrain/latest.pt` | 30,518 | 2.8726 | ~508 MB | `6ceeb456637e1a3faef9a2cbbfeab1d3e8e19e7fa56a297a7e372559779df506` | Final pretraining step |
| `artifacts/sft/direct/best.pt` | 650 | 0.9514 | ~508 MB | `764c679a7852c03fb7d9d0ea46eb61bebaaaecce471d2b8b982bf7903823438a` | Best Direct SFT checkpoint |
| `artifacts/sft/cot/best.pt` | 650 | 1.3432 | ~508 MB | `bf4070a7b458b0f47e33dc781604a11aa750058b85fa1b88e14e10141fef67e2` | Best CoT SFT checkpoint |

### 17.2 Core Data & Tokenizer Artifacts

| Artifact Path | Size | MD5 Hash / SHA-256 Hash | Description |
| :--- | :---: | :--- | :--- |
| `artifacts/tokenizer/tokenizer.model` | 129 KB | MD5: `e7ef63605f920bf411f0c897c44e31e3`<br>SHA256: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044` | SentencePiece BPE 8k tokenizer |
| `data/manifest_500m.json` | 1.3 KB | Pretraining data manifest | 500M training / 10M val token layout |
| `data/reasoning/dataset_manifest.json` | 4.1 KB | Reasoning dataset manifest | GSM8K train/dev Direct/CoT provenance |
| `data/reasoning/gsm8k_sft_train.jsonl` | 9.4 MB | SHA256: `f040b181642281edd4354c2cd67f6ec0af1b922658d8ec3a28136b5ae5e1b6d6` | 6,973 CoT SFT train examples |
| `data/reasoning/gsm8k_sft_dev.jsonl` | 665 KB | SHA256: `8bd18a20db0a126b87f26ac18c0627a6eb72a222af7ba0016ac09c39fbfbdc7b` | 500 CoT SFT dev examples |
| `data/reasoning/gsm8k_direct_train.jsonl` | 5.9 MB | SHA256: `8f5fcbfb48b0ff9a59596147cd90b6924f486d1757ee9e9534425fde5d88d765` | 6,973 Direct SFT train examples |
| `data/reasoning/gsm8k_direct_dev.jsonl` | 418 KB | SHA256: `d0ed361828ebffece4b5d42b964f74754aa4a815d0c15c20068ba8b6d8c3ef84` | 500 Direct SFT dev examples |

---

## 18. Repository File & Directory Map

```text
Mini-R1/
├── README.md                      # Historical project overview
├── PROJECT_CONTEXT.md             # [THIS DOCUMENT] Authoritative master context document
├── requirements.txt               # Pinned Python dependencies
├── pytest.ini                     # Pytest configuration
├── .gitignore                     # Git ignore rules
│
├── configs/                       # Configuration YAML files
│   ├── model.yaml                 # 42.1M parameter model architecture configuration (STABLE)
│   ├── pretrain_500m.yaml         # Production 500M-token pretraining hyperparameters (FROZEN)
│   ├── sft_direct.yaml            # Direct-Answer SFT configuration (STABLE)
│   ├── sft_cot.yaml               # Chain-of-Thought SFT configuration (STABLE)
│   ├── base.yaml                  # Base fallback configuration (LEGACY)
│   └── train.yaml                 # Generic training template (LEGACY)
│
├── src/                           # Core implementation modules (scratch-built)
│   ├── model.py                   # MiniR1Transformer decoder model & configuration (FROZEN)
│   ├── embeddings.py              # Token & learned 1D positional embeddings (FROZEN)
│   ├── attention.py               # Multi-Head Causal Self-Attention from scratch (FROZEN)
│   ├── transformer_block.py       # Pre-LN Transformer decoder block with FFN (FROZEN)
│   ├── tokenizer.py               # SentencePiece wrapper & special token management (FROZEN)
│   ├── loss.py                    # Causal LM cross-entropy loss with shifting (FROZEN)
│   ├── optimizer.py               # AdamW creation with decoupled weight decay (FROZEN)
│   ├── scheduler.py               # Warmup cosine learning rate scheduler (FROZEN)
│   ├── training_step.py           # Atomic training step & gradient execution (FROZEN)
│   ├── pretrain_dataset.py        # Sharded binary uint16 dataset & memory-mapping (FROZEN)
│   ├── general_dataset.py         # Streaming dataset preprocessor & contiguous packing (FROZEN)
│   ├── sft_dataset.py             # Masked SFT dataset (-100 prompt masking collator) (ACTIVE)
│   ├── sft_trainer.py             # SFT training loop, dev evaluator, and checkpointing (ACTIVE)
│   ├── checkpoint.py              # Checkpoint serialization, rotation, and resumption (STABLE)
│   ├── evaluation.py              # Evaluation metrics, answer extraction & benchmark loaders (ACTIVE)
│   ├── metrics.py                 # Loss, perplexity, and throughput trackers (STABLE)
│   └── preprocessing.py           # Text cleaning, normalization, and tag formatting (STABLE)
│
├── scripts/                       # Executable CLI tools and experiment runners
│   ├── gpu_check.py               # Hardware & GPU diagnostic script
│   ├── train_tokenizer.py         # SentencePiece BPE model trainer
│   ├── build_pretrain_shards.py   # Offline 500M token binary shard packager
│   ├── train_pretrain.py          # 500M base pretraining engine CLI
│   ├── evaluate_base_model.py     # Base model benchmark evaluation CLI (Task 14)
│   ├── build_reasoning_sft_dataset.py # GSM8K SFT train/dev dataset builder (Task 15A)
│   ├── run_sft_smoke_test.py      # SFT engine verification smoke test (Task 15B)
│   ├── train_sft_experiment.py    # Direct vs CoT SFT training runner (Task 15C)
│   └── evaluate_sft_experiment.py # Complete SFT benchmark evaluation & report generator (Task 15C)
│
├── tests/                         # Full Pytest test suite (140 tests total, 77 offline unit tests)
│   ├── test_environment.py        # Hardware and PyTorch environment tests
│   ├── test_tokenizer.py          # Tokenizer encoding, decoding, and special tokens
│   ├── test_model.py              # Model forward pass, parameters, and shapes
│   ├── test_attention.py          # Attention shapes, masking, and scaled dot-product
│   ├── test_block.py              # Transformer block and FFN operations
│   ├── test_loss.py               # Cross-entropy loss and causal shifting
│   ├── test_training_step.py      # Backpropagation and gradient updates
│   ├── test_pretrain_engine.py    # Binary sharding and bitwise resumption
│   ├── test_sft_engine.py         # Masked SFT prompt masking and loss computation
│   ├── test_reasoning_dataset.py  # SFT dataset partitioning and format checks
│   └── test_evaluation.py         # Answer extraction and metric calculation
│
├── reports/                       # Formal research & qualification reports
│   ├── task11b_final_qualification.md # Pretraining data qualification report
│   ├── task12_production_engine.md    # Pretraining engine qualification report
│   ├── task13_base_pretraining_results.md # 500M base pretraining final report
│   ├── task14_base_model_evaluation.md # Base model zero-shot evaluation report
│   ├── task15a_reasoning_dataset.md    # Reasoning SFT dataset construction report
│   ├── task15b_sft_engine.md           # Masked SFT engine qualification report
│   └── task15c_direct_vs_cot.md        # Controlled Direct vs CoT experiment report
│
├── checkpoints/                   # Checkpoints
│   └── mini_r1_500m_pretrain/     # 500M pretraining run (`best.pt`, `latest.pt`, step checkpoints)
│
└── artifacts/                     # Generated artifacts
    ├── tokenizer/                 # Trained SentencePiece tokenizer models
    └── sft/                       # SFT experiment models (`direct/` and `cot/`)
```

---

## 19. Common Operational Commands & Workflows

### 19.1 Verification & Unit Tests
```powershell
# Run the fast offline unit test suite (77 tests, ~18s)
pytest tests/test_attention.py tests/test_block.py tests/test_checkpoint.py tests/test_loss.py tests/test_training_step.py tests/test_model.py tests/test_tokenizer.py tests/test_torch_dataset.py tests/test_trainer.py tests/test_pretrain_engine.py tests/test_sft_engine.py tests/test_evaluation.py tests/test_reasoning_dataset.py -v

# Run the complete test suite (includes online streaming tests)
pytest -v
```

### 19.2 Checking Base Checkpoint Immutability
```powershell
python -c "import hashlib; h = hashlib.sha256(open('checkpoints/mini_r1_500m_pretrain/best.pt', 'rb').read()).hexdigest(); print('best.pt sha256:', h); assert h == 'f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4', 'CORRUPTED BASE CHECKPOINT!'"
```

### 19.3 Running SFT Training
```powershell
# Direct-Answer SFT
python scripts/train_sft_experiment.py --config configs/sft_direct.yaml

# Chain-of-Thought SFT
python scripts/train_sft_experiment.py --config configs/sft_cot.yaml
```

### 19.4 Running SFT Benchmark Evaluations & Report Generation
```powershell
# Fast batched benchmark evaluation across GSM8K, SVAMP, AQuA, and 10M General Val
python scripts/evaluate_sft_experiment.py --batch_size 32
```

---

## 20. AI Collaboration Policy & Role Specialization

To maximize engineering efficiency, mathematical rigor, and research depth across AI accounts:

| Model Persona | Recommended Operational Scope | What NOT to use for |
| :--- | :--- | :--- |
| **Gemini** | - Writing implementation code, tests, and scripts<br>- Refactoring and bug fixing<br>- Command execution and environment diagnostics<br>- Data formatting and dataset processing | - Deep speculative theory without empirical evidence<br>- Uncontrolled hyperparameter proliferation |
| **Claude Opus** | - Forensic research analysis and failure mode dissection<br>- Experimental design and ablation planning<br>- Mathematical modeling and statistical significance analysis<br>- Designing GRPO / relative-reward RL algorithms | - Routine boilerplate coding<br>- Simple file renaming or repetitive test authoring |

---

## 21. Research Study Queue for Human Partner

While the AI handles code implementation, execution, and unit testing, the human researcher should study the following core topics to maintain deep technical understanding of the underlying mechanisms:

1. **Masked Causal Language Modeling**:
   - Why prompt tokens must be assigned label $-100$ in PyTorch cross-entropy loss.
   - Causal tensor shifting dynamics (`logits[:, :-1]` vs `targets[:, 1:]`).
2. **Autoregressive Degeneration & Repetition**:
   - The mathematical cause of self-reinforcing probability loops in greedy decoding ($T=0.0$).
   - Repetition penalties ($1/(p_k \cdot \theta)$ vs subtraction), frequency penalties, and presence penalties.
   - Nucleus sampling ($p=0.9$) and temperature scaling ($T=0.7$) vs greedy search.
3. **Exposure Bias in Autoregressive Sequence-to-Sequence Generation**:
   - The discrepancy between teacher-forced training (conditioning on ground truth) and autoregressive inference (conditioning on model's own prior tokens).
   - Scheduled sampling and student-forcing concepts.
4. **Statistical Significance Testing in NLP**:
   - McNemar's paired test for binary classification discordance.
   - Fisher's Exact test and binomial confidence intervals on small test splits.
5. **Outcome-Based Reinforcement Learning (GRPO)**:
   - Group Relative Policy Optimization (DeepSeek-R1 / Mini-R1 formulation).
   - Policy gradient loss with KL divergence penalty against frozen base model: $\mathcal{L}_{\text{GRPO}}(\theta)$.
   - Rule-based binary outcome verification ($r \in \{0, 1\}$) on final answers vs process supervision.

---

# CURRENT STATE

1. **Pretrained Base Model**: Fully trained on 500M general educational tokens (`cosmopedia-v2` 60% / `fineweb-edu-dedup` 40%). Validation loss = **2.8726**, validation perplexity = **17.68**. Saved immutably at `checkpoints/mini_r1_500m_pretrain/best.pt`.
2. **Base Zero-Shot Baseline (Task 14)**: 0.76% GSM8K, 3.00% SVAMP, 17.32% AQuA. 0.0% delimiter compliance.
3. **Reasoning SFT Datasets (Task 15A)**: 6,973 Train / 500 Dev GSM8K examples paired for Direct and CoT conditions with 0 test leakage.
4. **Masked SFT Engine (Task 15B)**: Full completion-only causal LM training engine operational and unit-tested.
5. **Controlled SFT Experiment (Task 15C)**: Completed 650-step runs on Direct and CoT branches. Evaluated on GSM8K (1,319), SVAMP (300), AQuA (254), and 10M general validation tokens.
6. **SFT Evaluation Audit (Task 16A)**: Re-evaluated exact 10M general validation tokens across all 3 checkpoints. Audited loss/PPL: Base = 2.8724 / 17.68; Direct SFT = 3.0834 / 21.83 (+23.49%); CoT SFT = 3.1603 / 23.58 (+33.36%). Audit status: **RESOLVED**.
7. **Decoding-Only CoT Ablation (Task 16B)**: Evaluated 5 decoding conditions on frozen `artifacts/sft/cot/best.pt` across all 1,319 GSM8K test examples. Condition D (Temp 0.6 + RepPen 1.2) reduced repetition from 82.6% to 11.4%, improved format compliance to 90.7%, and increased strict accuracy from 1.21% to 2.20% (vs Direct SFT 2.73%). Verdict: **B. Decoding helped but reasoning remains weak** (underlying arithmetic execution bottleneck).
8. **Arithmetic Capability Probe (Task 17A)**: Evaluated 9,000 synthetic arithmetic problems (seed 42) across Base, Direct SFT, and CoT SFT on 3 prompt formats. Strict accuracy on Reasoning format: Base = 0.03%, Direct SFT = 1.76%, CoT SFT = 1.13%. Word-wrapped accuracy matched pure expressions (-0.04% to -0.38% delta), proving language mapping is NOT the bottleneck. 100-sample CoT trace validation revealed 0.0% mathematically valid traces and 82.0% structured traces with invalid arithmetic operations. Verdict: **C. Arithmetic is mostly absent**.
9. **Verified Computational Training Corpus (Task 18A)**: Built 6-level deterministic arithmetic SFT curriculum (12,000 train / 1,200 dev) materialized in paired Compact (34.7 mean completion tokens) and Verbose (48.2 mean completion tokens, 1.39x multiplier) JSONL formats. Pre-registered non-overlapping holdout test ranges. Audited all 26,400 rows with 100% mathematical validity. Status: **100% VERIFIED**.
10. **Verified Arithmetic SFT Experiment (Task 18B)**: Trained Compact Arithmetic SFT and Verbose Arithmetic SFT (1,125 steps, peak LR 1e-4, FP16 AMP). Evaluated across IID Dev (1.2k), OOD Numbers (1.2k), OOD Composition (1.2k), and Word-Wrapped (1.2k). Compact reached **45.42%** IID accuracy (75.85% valid equations) and Verbose reached **49.42%** IID accuracy (78.62% valid equations) vs Base 0.00%. However, OOD number accuracy dropped to **1.00% - 2.17%** ($\Delta \approx -45\%$) and OOD composition to **8.58% - 10.33%** ($\Delta \approx -38\%$). Verdict: **B. Verified arithmetic supervision helps, but generalization remains weak**.
11. **Numeric Representation & Generalization Forensics (Task 19A)**: Audited tokenizer representation and training distribution bounds. Discovered tokenizer enforces **strict character/digit-by-digit tokenization** (0 multi-digit merges, 100% token piece overlap with OOD). Proved OOD failure is driven by **sequence-length / autoregressive position shift** (2-digit $\to$ 3-digit transitions) and **dense training bounding box interpolation** ($[2, 99]$ lookup table) rather than vocabulary token gaps. Verdict: **D. Combination of A/B/C**.

---

# CURRENT SCIENTIFIC RESULT

1. **Decoding Optimization Substantially Mitigates CoT Degeneration**: Applying Keskar repetition penalty ($\theta = 1.2$) and temperature scaling ($T=0.6, p=0.95$) reduces 4-gram repetition from 82.6% to 11.4% and lifts format compliance from 39.8% to 90.7% on frozen CoT SFT weights.
2. **Explicit Arithmetic Supervision Installs In-Distribution Calculation**: Verified SFT dramatically elevates IID arithmetic accuracy from 0.00% (Base) to **45.42%** (Compact) and **49.42%** (Verbose), with step-level equation truth rising from 3.03% to **78.62%**.
3. **Severe Generalization Collapse on OOD Numbers**: Despite high in-distribution mastery, performance drops to **1.00%** on unseen operands ($[101, 250]$) and **8.58%** on novel compositions, demonstrating the 42.1M model performs table-lookup interpolation rather than abstract symbolic rule execution.
4. **Compact vs Verbose Parity**: Compact computational traces achieve near-parity in accuracy (**45.42%** vs **49.42%**) while consuming 28% fewer completion tokens per step, offering higher computational sample efficiency.
5. **Token-Volume Catastrophic Forgetting**: CoT SFT suffered +33.36% general PPL shift (vs +23.49% for Direct) due to 19x higher completion token exposure under identical optimizer steps.

---

# WHAT WE DO NOT KNOW

1. **Token Step vs Optimization Steps**: Would CoT match or exceed Direct SFT if training steps were normalized by active completion tokens rather than batch instances (i.e. equal token volume)?
2. **Pretraining Arithmetic Density**: Can the 42.1M SLM compute multi-step arithmetic accurately if pretraining included synthetic arithmetic scaffolding or math-dense pretraining mixtures?
3. **RL / Verifier Correction**: Can rule-based reinforcement learning (GRPO / PPO / outcome reward modeling) steer the coherent syntactic reasoning traces toward arithmetically accurate solutions?

---

# DO NOT DO YET

- ❌ **DO NOT implement Reinforcement Learning (GRPO / PPO / DPO)** yet without explicit instruction.
- ❌ **DO NOT run blind multi-hyperparameter sweeps**.
- ❌ **DO NOT modify the Transformer architecture or tokenizer**.
- ❌ **DO NOT touch or overwrite `checkpoints/mini_r1_500m_pretrain/best.pt`**.
- ❌ **DO NOT build the FastAPI deployment server yet**.
- ❌ **DO NOT claim CoT "fundamentally fails for all SLMs"**; restrict conclusions strictly to the tested configuration.

---

# TRANSFER INSTRUCTIONS

> **To the next AI assistant**:  
> You are inheriting the Mini-R1 research codebase. Read this document (`PROJECT_CONTEXT.md`) and `.agents/rules/projectrule.md` thoroughly before taking any action.  
> - Treat the repository files, logs, and checkpoints as the source of truth.  
> - Never overwrite `checkpoints/mini_r1_500m_pretrain/best.pt` (SHA256: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`).  
> - Follow the mandatory project rules: explain your design, state assumptions, list changed files, implement only the assigned task, provide verification commands, and stop immediately after the current task.
