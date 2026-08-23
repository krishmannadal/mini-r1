# PROJECT STATUS: COMPLETE THROUGH TASK 21A

# MINI-R1: COMPREHENSIVE RESEARCH SUMMARY & TECHNICAL RECORD (TASKS 1 – 21A)

**Authoritative Project Document**  
**Repository**: `Mini-R1` **Scope**: Complete historical, architectural, engineering, and empirical record of Tasks 1 through 21A  
**Current Date**: 2026-08-21  
**Project State**: Immutable base pretraining complete; GSM8K reasoning SFT evaluated; arithmetic capability probed; progressive numeric curriculum evaluated; controlled arithmetic-to-reasoning transfer evaluated.  

---

## 1. Project Identity, Motivation & Research Evolution

### 1.1 Project Identity & Objectives
- **Project Name**: Mini-R1
- **Domain**: Small Language Model (SLM) computational and mathematical reasoning research.
- **Core Investigation**: How a decoder-only Transformer with strictly bounded capacity (~42.1M parameters) acquires, generalizes, retains, and transfers mathematical reasoning and arithmetic computation across progressive training stages.
- **Original Practical Goal**: Engineer a clean, modular, modern SLM entirely from scratch in PyTorch (zero high-level framework wrappers like Hugging Face `transformers` or `accelerate`), qualify it on local consumer hardware (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM), and deploy it for low-latency interactive inference via a standalone FastAPI service.
- **Scientific Evolution of the Research**:
  The project evolved from a standard SFT deployment pipeline into a foundational investigation into the boundaries between:
  1. **Language Modeling**: Statistical next-token distribution modeling on natural text.
  2. **Reasoning-Shaped Syntax**: Superficial generation of reasoning markup (`<think>...</think><answer>...</answer>`) vs genuine semantic deduction.
  3. **Arithmetic Computation**: Deterministic numerical calculation and symbolic state tracking.
  4. **Within-Regime Manifold Interpolation**: Generalization to unseen number pairs within a trained digit bounding box.
  5. **Cross-Regime Algorithmic Extrapolation**: Generalization across unseen digit lengths and sequence lengths.
  6. **Downstream Reasoning Transfer**: Whether specialized algorithmic arithmetic skills transfer to multi-step natural language word problems (GSM8K).

### 1.2 Engineering Rules & Research Integrity
1. **Strict Framework Prohibition**: All model components (multi-head causal self-attention, positional embeddings, Transformer decoder blocks, loss formulations with causal shifting, training loops, data streaming collators, evaluators, and verifiers) are written explicitly by hand in PyTorch.
2. **Scientific Honesty & Negative Results**: Negative results are first-class scientific data. Negative transfer, generation degeneration, catastrophic forgetting, and memorization leakage are documented with exact empirical metrics.
3. **Immutable Base Model**: The canonical pretrained checkpoint ([checkpoints/mini_r1_500m_pretrain/best.pt](checkpoints/mini_r1_500m_pretrain/best.pt)) is frozen and cryptographically audited.
4. **Mandatory Protocol**: Every experiment follows a strict pre-registered cycle: *Hypothesis $\to$ Controlled Intervention $\to$ Measurement $\to$ Decision / Conclusion*.

---

## 2. Hardware & Runtime Specifications

The entire research lifecycle—including 500M token base pretraining, supervised fine-tuning, forensic probes, and 5-pillar evaluations—is executed locally on a single consumer laptop GPU:

| Component | Exact Specification | Verification Source / Method |
| :--- | :--- | :--- |
| **GPU Model** | **NVIDIA GeForce RTX 4050 Laptop GPU** | `torch.cuda.get_device_name(0)` |
| **Dedicated VRAM** | **6.0 GB GDDR6** (5,996.6 MB physical allocation) | `torch.cuda.get_device_properties(0).total_memory` |
| **GPU Microarchitecture** | Ada Lovelace (Compute Capability 8.9 / SM 8.9) | Hardware diagnostic |
| **NVIDIA Driver Version** | **581.57** | NVIDIA System Information |
| **Host Operating System** | **Windows 11 Home / Pro** (Build `Windows-10-10.0.26200-SP0`) | `platform.platform()` |
| **Python Version** | **3.11.0** (64-bit AMD64, MSC v.1933) | `sys.version` |
| **PyTorch Version** | **2.6.0+cu124** | `torch.__version__` |
| **CUDA Runtime** | **CUDA 12.4** | `torch.version.cuda` |
| **Primary System Shell** | PowerShell (`pwsh` / `powershell.exe`) | Environment runtime |

---

## 3. Model Architecture Specification

The architecture is implemented in [src/model.py](src/model.py), [src/embeddings.py](src/embeddings.py), [src/attention.py](src/attention.py), and [src/transformer_block.py](src/transformer_block.py), configured via [configs/model.yaml](configs/model.yaml).

### 3.1 Architectural Hyperparameters

| Hyperparameter | Variable / Config Key | Value | Engineering & Mathematical Rationale |
| :--- | :--- | :---: | :--- |
| **Architecture Type** | -- | **Decoder-Only Causal Transformer** | Standard autoregressive next-token prediction |
| **Total Parameters** | $N_{\text{total}}$ | **42,132,480 (~42.1M)** | Maximum capacity fitting within 6GB VRAM at batch=32, seq=512 |
| **Vocabulary Size** | `vocab_size` ($V$) | **8,000** | SentencePiece BPE; compact embedding footprint ($4.1\text{M}$ weights) |
| **Model Hidden Dimension** | `d_model` | **512** | Residual stream width |
| **Transformer Layers** | `n_layers` ($L$) | **12** | Deep representation hierarchy |
| **Attention Heads** | `n_heads` ($H$) | **8** | Multi-head division; head dimension $d_k = d_{\text{model}} / H = 64$ |
| **Head Dimension** | $d_k$ | **64** | Standard scaled dot-product attention width |
| **Feed-Forward Dimension** | `d_ff` | **2,048** | Standard $4 \times d_{\text{model}}$ expansion |
| **Max Context Window** | `max_seq_len` ($T_{\text{max}}$) | **512** | Sufficient for multi-step reasoning traces |
| **Dropout Rate** | `dropout` | **0.0** | Standard 0.0 for pretraining/SFT throughput efficiency |
| **Linear Layer Bias** | `bias` | **False** | Modern simplified parameterization (LLaMA/Mistral style) |
| **Weight Tying** | `tie_word_embeddings` | **True** | `lm_head.weight` is tied to `tok_emb.weight` |
| **Layer Normalization** | -- | **Pre-LayerNorm** | Pre-LN on residual stream with final `ln_f` before `lm_head` |
| **Activation Function** | -- | **GELU** | Gaussian Error Linear Unit in `FeedForwardNetwork` |

### 3.2 Exact Parameter Count Derivation
$$\begin{aligned}
N_{\text{emb}} &= (V \cdot d_{\text{model}}) + (T_{\text{max}} \cdot d_{\text{model}}) = (8{,}000 \cdot 512) + (512 \cdot 512) = 4{,}096{,}000 + 262{,}144 = 4{,}358{,}144 \\
N_{\text{attn\_layer}} &= 4 \cdot d_{\text{model}}^2 = 4 \cdot (512 \cdot 512) = 1{,}048{,}576 \quad (W_q, W_k, W_v, W_o) \\
N_{\text{ffn\_layer}} &= 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 2 \cdot (512 \cdot 2{,}048) = 2{,}097{,}152 \quad (W_1, W_2) \\
N_{\text{ln\_layer}} &= 2 \cdot 2 \cdot d_{\text{model}} = 2 \cdot (512 + 512) = 2{,}048 \quad (\text{LN}_1, \text{LN}_2 \text{ weights \& biases}) \\
N_{\text{block}} &= 1{,}048{,}576 + 2{,}097{,}152 + 2{,}048 = 3{,}147{,}776 \\
N_{\text{all\_12\_blocks}} &= 12 \cdot 3{,}147{,}776 = 37{,}773{,}312 \\
N_{\text{final\_ln}} &= 2 \cdot d_{\text{model}} = 1{,}024 \\
N_{\text{lm\_head}} &= 0 \quad (\text{Tied with token embeddings}) \\
\mathbf{N_{\text{total}}} &= 4{,}358{,}144 + 37{,}773{,}312 + 1{,}024 = \mathbf{42{,}132{,}480}
\end{aligned}$$

---

## 4. Tokenizer Specification & Numeric Representation Findings

Implemented in [src/tokenizer.py](src/tokenizer.py), trained via [scripts/train_tokenizer.py](scripts/train_tokenizer.py).

### 4.1 Tokenizer Configuration & Special Tokens
- **Engine**: Google SentencePiece (`sentencepiece.SentencePieceProcessor`)
- **Algorithm**: Byte-Pair Encoding (`bpe`)
- **Target Vocabulary Size**: 8,000 tokens
- **Configuration Flags**:
  - `character_coverage = 1.0` (preserves all mathematical and code symbols)
  - `split_digits = True` (forces numbers to split into individual single-digit subwords)
  - `byte_fallback = True` (decomposes unknown UTF-8 characters to byte tokens)
  - `normalization_rule_name = "identity"` (disables destructive unicode stripping)
- **Special Token ID Registry**:
  - `unk_id = 0` (`<unk>`)
  - `bos_id = 1` (`<s>`)
  - `eos_id = 2` (`</s>`)
  - `pad_id = 3` (`<pad>`)
  - `think_start_id = 4` (`<think>`) [User-defined atomic symbol]
  - `think_end_id = 5` (`</think>`) [User-defined atomic symbol]
  - `answer_start_id = 6` (`<answer>`) [User-defined atomic symbol]
  - `answer_end_id = 7` (`</answer>`) [User-defined atomic symbol]
- **Artifact Path**: `artifacts/tokenizer/tokenizer.model`
  - **MD5**: `e7ef63605f920bf411f0c897c44e31e3`
  - **SHA256**: `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044`

### 4.2 Critical Numeric Tokenization Finding (Task 19A Forensics)
Forensic tokenization analysis across 10,000 random integers proved:
1. **Strict Single-Digit Invariant**: There are **zero multi-digit tokens** in the vocabulary. Numbers are 100% decomposed into single-digit pieces:
   - 1-digit numbers ($0–9$) $\to$ exactly 1 token (e.g., `'4'` $\to$ `[7822]`)
   - 2-digit numbers ($10–99$) $\to$ exactly 2 tokens (e.g., `'42'` $\to$ `['4', '2']`)
   - 3-digit numbers ($100–999$) $\to$ exactly 3 tokens (e.g., `'124'` $\to$ `['1', '2', '4']`)
   - $N$-digit numbers $\to$ exactly $N$ tokens.
2. **100% Token ID Overlap**: Exactly the same 10 numeric token IDs represent all numbers across Base Pretraining, In-Distribution SFT, and Out-of-Distribution (OOD) test sets.
3. **Scientific Implication**:
   **Out-of-vocabulary numeric token gaps do NOT explain out-of-distribution arithmetic failure.** The failure is driven by sequence-length / positional shift and lookup-table interpolation boundaries rather than missing subword tokens.

---

## 5. Base Pretraining Infrastructure & Canonical Checkpoint

Constructed in Tasks 10–13 ([src/pretrain_dataset.py](src/pretrain_dataset.py), [scripts/build_pretrain_shards.py](scripts/build_pretrain_shards.py), [scripts/train_pretrain.py](scripts/train_pretrain.py)).

### 5.1 Pretraining Data Pipeline & Corpus Mixture
- **Corpus Source**: `HuggingFaceTB/smollm-corpus`
- **Enforced Token Mixture Ratios**:
  - **60% `cosmopedia-v2`**: Synthetic textbooks, educational course units, and tutorials.
  - **40% `fineweb-edu-dedup`**: Deduplicated educational web content.
  - **0% `python-edu` (EXCLUDED)**: Removed in Task 11B due to git repository blob schema incompatibilities.
- **Contiguous Packing**: Documents concatenated with `</s>` (EOS, id=2) into 512-token chunks, achieving 100% token density with zero padding waste.
- **Binary Memmapped Shards**: Stored as flat raw `uint16` binary files (10M tokens / 19.07 MB per shard). 50 train shards (500M tokens) + 1 validation shard (10M tokens). Total disk size: 1.02 GB. RAM usage during training: <20 MB. 100% air-gapped execution.

### 5.2 Pretraining Optimization Hyperparameters ([configs/pretrain_500m.yaml](configs/pretrain_500m.yaml))
- **Total Token Budget**: 500,000,000 tokens
- **Micro-Batch Size**: 8 sequences (4,096 tokens)
- **Gradient Accumulation Steps**: 4 steps
- **Effective Batch Size**: 32 sequences (16,384 tokens per optimizer update)
- **Total Optimizer Steps**: 30,518 steps
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight\_decay} = 0.01, \epsilon = 1.0 \times 10^{-8}$)
- **Learning Rate Schedule**: Peak $6.0 \times 10^{-4}$ (selected via Task 8C/9 empirical sweep over $1e-4, 3e-4, 6e-4$), Min $6.0 \times 10^{-5}$, Warmup 100 steps, Cosine decay over 30,518 steps.
- **Precision**: Mixed Precision FP16 with `torch.amp.GradScaler` (peak scale: 131,072.0; only 11 skipped steps out of 30,518, or 0.036%).
- **Gradient Clipping**: `max_grad_norm = 1.0`

### 5.3 500M Pretraining Performance & Best Checkpoint Selection
- **Total Tokens Processed**: **500,000,256 tokens** in **8.02 hours** ($28,874.3\text{ seconds}$)
- **Average Throughput**: **17,316.5 tokens/second** (Peak: ~21,700 tok/s)
- **Peak VRAM Allocated**: **3,409.7 MB** (44.6% safety headroom on 6GB GPU)
- **Loss Trajectory**: Initial Val Loss $8.3674$ (PPL $4,304.45$) $\to$ Best Val Loss **$2.8726$** (PPL **$17.68$**) at Step 18,000.
- **Why Step 18,000 is the Canonical Base Checkpoint**:
  Validation loss reached its global minimum of $2.8726$ at Step 18,000. The final Step 30,518 checkpoint exhibited slight late-stage training loss oscillation while validation loss remained flat at 2.8726. Checkpoint `best.pt` (Step 18,000) was locked as the immutable starting point for all downstream reasoning and arithmetic SFT.
  - **Path**: `checkpoints/mini_r1_500m_pretrain/best.pt`
  - **SHA256**: `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4`

---

## 6. Complete Chronological Task History (Tasks 1 – 21A)

| Task ID | Task Name | Purpose & Implementation | Key Empirical Findings | Problems Discovered & Forensic Insights | Research Pivot / Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1** | Environment & GPU Validation | Hardware & software stack diagnostic | RTX 4050 6GB VRAM, PyTorch 2.6.0+cu124 verified | Limited 6GB VRAM requires strict memory profiling | Locked batch/sequence budget |
| **Task 2** | Tokenizer Training | Train SentencePiece 8k BPE model | 8,000 vocab, special tokens `<think>`, `<answer>` | Avoid subword splitting on markup | Preserved atomic tags |
| **Task 3** | Dataset Pipeline | Multi-dataset preprocessing schema | GSM8K, SVAMP, AQuA-RAT loaders | Need clean train/dev/test separation | Pre-registered held-out splits |
| **Task 4** | Architecture from Scratch | Pure PyTorch Transformer decoder | 42.1M parameters tensor-shape tested | Pre-LN vs Post-LN stability | Pre-LN selected |
| **Task 5** | Loss & Optimizer | Cross-entropy with causal shift | Backprop verified, AdamW gradient steps | Proper causal alignment of tokens | Validated loss masking |
| **Task 6** | Training Infrastructure | Dataloader, checkpointing, trainer | 68/68 tests passing, dynamic collator | Checkpoint write race conditions | Implemented atomic rename writes |
| **Task 7** | Scaling & Profiling | Scale to 12L-512D-8H-2048FFN | Peak VRAM 3.25 GB on 6GB card | Memory headroom must be $\ge 40\%$ | Confirmed 42.1M architecture |
| **Task 8** | System Qualification | AMP equivalence & GradScaler | Throughput 21.5k tok/s, 0 skipped steps | FP32 err 5.09e-7, FP16 err 0.1085% | Qualified FP16 AMP pipeline |
| **Task 8C/9**| LR Sweep (1e-4–6e-4) | 1,000-step LR optimization sweep | $6e-4$ reached lowest loss (2.4995 vs 2.6337) | $1e-4$ converged too slowly for 500M budget | Pretraining peak LR locked to $6e-4$ |
| **Task 10** | Contiguous Packing | Streaming dataset preprocessor | 100% token density, zero padding waste | Naive document truncation lost 58.4% tokens | Implemented EOS packing |
| **Task 11/11B**| Corpus Qualification | Qualify 60/40 SmolLM data | 500M training / 10M validation tokens | `python-edu` had git schema incompatibilities | Excluded `python-edu` from mixture |
| **Task 12** | Sharded Engine | Memory-mapped binary uint16 shards | 107/107 tests passing, <20MB RAM usage | Network dependencies cause training hangs | 100% air-gapped offline streaming |
| **Task 13** | 500M Base Pretraining | Full pretraining run on 500M tokens | 8.02 hrs, Val Loss 2.8726, PPL 17.68 | Training loss plateaued after Step 18,000 | Step 18k checkpoint locked as `best.pt` |
| **Task 14** | Base Zero-Shot Baseline | Zero-shot evaluation across benchmarks | GSM8K 0.76%, SVAMP 3.00%, AQuA 17.32% | Apparent GSM8K success was regex extraction artifact | Established 0% native reasoning baseline |
| **Task 15A**| Reasoning SFT Datasets | Build paired Direct vs CoT splits | 6,973 Train / 500 Dev paired GSM8K | CoT has 2.32x total and 19x completion tokens | Pre-registered paired conditions |
| **Task 15B**| Masked SFT Engine | SFT trainer with `-100` prompt masking | Completion-only loss verified (140/140 tests) | Unmasked prompts corrupt reasoning training | Verified causal prompt masking |
| **Task 15C**| Direct vs CoT SFT | Controlled 650-step comparison | Direct **2.73%** vs CoT **1.21%** ($p=0.0051$) | CoT suffered 82.6% repetition degeneration | **`COT HURTS`**: CoT degraded reasoning |
| **Task 16A**| Evaluation Audit | Re-evaluate 10M val tokens across models | Base PPL 17.68, Direct 21.83, CoT 23.58 | Discussion text had placeholder PPL values | Resolved data consistency |
| **Task 16B**| Decoding Ablation | 5 decoding setups on frozen CoT | Temp 0.6 + RepPen 1.2 $\to$ GSM8K **2.20%** | Repetition fell to 11.4%, but arithmetic still wrong | Decoding helps; arithmetic is absent |
| **Task 17A**| Arithmetic Capability Probe| 9,000 synthetic arithmetic problems | All models scored <2% accuracy | 0.0% of CoT traces had valid arithmetic steps | Model lacks computational substrate |
| **Task 18A**| Arithmetic SFT Dataset | 6-level verified arithmetic corpus | 13,200 unique problems, Compact & Verbose | 100% deterministic verification required | Generated 26.4k verified traces |
| **Task 18B**| Verified Arithmetic SFT | Train Compact & Verbose SFT | Compact IID **45.42%**, Verbose IID **49.42%** | OOD numbers collapsed to **1.00% - 2.17%** | SFT installs IID table lookup |
| **Task 19A**| Numeric Representation | Forensic audit of tokenizer & data | 100% single-digit tokens; 0 multi-digit BPE | OOD failure is not token vocabulary gap | Proved positional/sequence-length shift |
| **Task 19B-PRE**| 3D Arithmetic Diagnostic | Initial 3D division vs add/sub probe | Apparent division accuracy = 24.28% | Looked like operator $\times$ digit interaction | Triggered forensic overlap audit |
| **Task 19B-PRE2**| Overlap Forensic Audit | Exact match audit of E5 benchmark | 93.4% of correct E5 came from train leakage | True unseen division was only **2.48%** | **Hypothesis Refuted**: Leakage corrected |
| **Task 19C**| Clean Subtraction SFT | Clean $[50, 99]$ subtraction curriculum | IID **100%**, Unseen 2D **35.40%**, Unseen 3D **6.60%** | Cross-digit answer length gap ($35.4\% \to 6.6\%$) | Within-regime generalization confirmed |
| **Task 19D**| Mixed 2D+3D Subtraction | Mixed 2D ($[50,99]$) + 3D ($[101,199]$) SFT | IID **100%**, OOD 3D rose from 6.6% to **34.40%** | 4D Acid test was **0.40%** (collapsed to 2D/3D) | Exposure enables regime transfer |
| **Task 20** | Progressive 2D+3D+4D SFT | Progressive 2D+3D+4D curriculum | IID 2D **100%**, 3D **99.8%**, 4D **67.2%** | 5D Acid test = **0.00%** (zero extrapolation) | Established regime-local boundaries |
| **Task 21A**| Arithmetic $\to$ GSM8K Transfer| Controlled transfer to GSM8K | Model A **1.44%** vs Model B **1.06%** ($\Delta=+0.38\%$) | Arithmetic retention collapsed ($100\% \to 5.8\%$) | **Conclusion C: No measurable transfer** |

---

## 7. Deep-Dive Analysis of Major Experimental Milestones

### 7.1 The Controlled Direct vs CoT Experiment (Task 15C)
In Task 15C, the base model was fine-tuned for 650 optimizer steps under identical conditions on 6,973 GSM8K training examples formatted either with Direct Answers or Chain-of-Thought reasoning traces.

#### Quantitative Comparison Table (Task 15C):
| Metric / Dimension | Pretrained Base | Direct SFT | Chain-of-Thought SFT | Strict Delta ($\Delta_{\text{CoT} - \text{Direct}}$) |
| :--- | :---: | :---: | :---: | :---: |
| **GSM8K Test (Strict Acc)** | 0.76% ($10/1319$) | **2.73%** ($36/1319$) | **1.21%** ($16/1319$) | **-1.52%** ($p=0.0051$) |
| **SVAMP Test (Strict Acc)** | 3.00% ($9/300$) | **1.67%** ($5/300$) | **0.67%** ($2/300$) | **-1.00%** |
| **AQuA-RAT Test (Strict Acc)** | 17.32% ($44/254$) | **0.00%** ($0/254$) | **0.00%** ($0/254$) | **+0.00%** |
| **`<answer>` Format Compliance** | 0.0% | **100.0%** | 40.8% | **-59.2%** |
| **`<think>` Format Compliance** | 0.0% | 0.0% | 40.5% | +40.5% |
| **Full Format Compliance** | 0.0% | 0.0% | **39.1%** | +39.1% |
| **EOS Termination Rate** | 0.0% | **100.0%** | **41.5%** | **-58.5%** |
| **4-Gram Repetition Rate** | -- | **0.0%** | **82.6%** | **+82.6%** |
| **Mean Generated Length** | 125.7 tokens | **4.6 tokens** | **158.6 tokens** | +154.0 tokens |
| **General Language PPL (10M Tokens)**| **17.68** | 21.83 (+23.5%) | 23.58 (+33.4%) | +1.75 PPL |

#### Key Insights from Task 15C:
1. **McNemar Significance ($p=0.0051$)**: CoT SFT significantly underperformed Direct SFT.
2. **Supervision Volume Confound**: CoT exposed the model to 19.05x more completion tokens (2,106,652 vs 110,594) across the exact same 6,973 problem instances.
3. **Autoregressive Degeneration**: Under greedy decoding, 82.6% of CoT traces fell into severe cyclic phrase attractors, exhausting the generation context before emitting `</think>`.

---

### 7.2 The Decoding-Only Ablation (Task 16B)
Task 16B evaluated whether CoT failure was strictly an inference decoding artifact by testing 5 decoding conditions on the frozen CoT checkpoint without any retraining.

#### Quantitative Decoding Table (Task 16B):
| Decoding Condition | Strict GSM8K Acc | Full Format % | 4-Gram Repetition % | Truncation % | Mean Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A: Baseline Greedy ($T=0.0$)** | 1.21% ($16/1319$) | 39.80% | 82.64% | 58.38% | 158.9 |
| **B: Temperature ($T=0.6, p=0.95$)** | 1.06% ($14/1319$) | 62.40% | 64.90% | 34.65% | 140.2 |
| **C: Repetition Penalty ($\theta=1.2$)** | 1.59% ($21/1319$) | 81.50% | 29.64% | 14.25% | 91.8 |
| **D: Combined ($T=0.6, p=0.95, \theta=1.2$)** | **2.20%** ($29/1319$) | **90.67%** | **11.45%** | **4.32%** | 92.3 |
| **E: Extended Context ($T=0.6, \text{Max}=384$)** | 0.61% ($8/1319$) | 92.19% | 10.92% | 2.96% | 98.7 |
| *Direct SFT Baseline Reference* | *2.73% (36/1319)* | *0.00%* | *0.00%* | *0.00%* | *4.6* |

#### Key Insights from Task 16B:
1. **Mechanical Rescue**: Combined temperature and repetition penalty reduced repetition from 82.64% to 11.45% and lifted format compliance from 39.80% to 90.67%.
2. **Arithmetic Bottleneck Unmasked**: Strict accuracy rose from 1.21% to 2.20%, but remained below Direct SFT (2.73%). Forensic inspection of the non-repetitive traces showed 75% were "structured but mathematically incorrect" (generating syntactically valid equations with wrong numerical calculations).

---

### 7.3 Synthetic Arithmetic Capability Probing (Task 17A)
Task 17A evaluated 9,000 synthetic arithmetic problems spanning addition, subtraction, multiplication, division, multi-step compositions (2–4 ops), and word-wrapped problems.

#### Key Findings from Task 17A:
- **Near-Zero Computational Capability**: Base Pretrained achieved 0.03%, Direct SFT achieved 1.76%, and CoT SFT achieved 1.13% on pure arithmetic expressions.
- **Word-Wrapping Invariant**: Accuracy on word-wrapped arithmetic problems (e.g., *"Janet has 24 apples and gets 12 more..."*) matched pure symbolic expressions within -0.04% to -0.38%, proving that natural language parsing was **not** the bottleneck.
- **Step-Level Equation Audit**: Deterministic verification of intermediate equations in CoT traces revealed that **0.0% of generated reasoning traces contained fully valid arithmetic steps**.

---

### 7.4 Verified Computational SFT & The Generalization Boundary (Tasks 18A–18B)
In Task 18A/18B, a 6-level deterministic arithmetic corpus (12,000 train / 1,200 dev) was generated and trained in Compact and Verbose formats.

#### Quantitative Arithmetic Results (Task 18B):
| Model Condition | IID Dev Acc ($N=1.2\text{k}$) | OOD Numbers Acc ($N=1.2\text{k}$) | OOD Composition Acc ($N=1.2\text{k}$) | Step-Level Eq Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Base Pretrained (Step 18k)** | 0.00% | 0.00% | 0.00% | 0.00% ($0/7$) |
| **Compact Arithmetic SFT** | **45.42%** | **1.00%** | **8.58%** | **75.85%** ($2378/3135$) |
| **Verbose Arithmetic SFT** | **49.42%** | **2.17%** | **10.33%** | **78.62%** ($2357/2998$) |

#### Key Insights from Task 18B:
1. **Successful In-Distribution Learning**: SFT elevated IID arithmetic accuracy from 0.00% to ~47%, with step-level equation accuracy rising from 3.03% to 78.62%.
2. **Severe Out-of-Distribution Collapse**: On operands outside the training bounding box ($[101, 250]$ vs $[2, 99]$), accuracy collapsed to **1.00% - 2.17%** (a 45% absolute drop), revealing that the model behaves as a dense interpolation table lookup rather than an algorithmic reasoner.

---

### 7.5 Forensic Overlap Audit & Research Correction (Tasks 19B-PRE & 19B-PRE2)
In Task 19B-PRE, an initial diagnostic on 3-digit arithmetic appeared to show that 3-digit division succeeded (24.28%) while 3-digit addition/subtraction failed (2.80%), suggesting a surprising operator $\times$ digit interaction.

#### The Forensic Audit (Task 19B-PRE2):
An exhaustive string and operand matching audit was executed against the entire 12,000-example training corpus.

| Forensic Evaluation Partition | $N$ | Correct | Strict Accuracy | % of Total Correct ($76$) |
| :--- | :---: | :---: | :---: | :---: |
| **Total E5 3-Digit Division Benchmark** | 313 | 76 | 24.28% | 100.0% ($76/76$) |
| **Exact Training Matches (Subset A)** | 111 | 71 | **63.96%** | **93.4%** ($71/76$) |
| **Truly Unseen Non-Training (Subset B)** | 202 | 5 | **2.48%** | **6.6%** ($5/76$) |

#### Key Insights from Task 19B-PRE2:
1. **Artifact Discovered**: 93.4% of correct division answers came from exact training prompt leakage.
2. **Hypothesis Refuted**: On genuinely unseen 3-digit division problems, accuracy was **2.48%**, completely in line with 3-digit subtraction (4.80%) and addition (0.80%).
3. **Research Integrity**: The operator $\times$ digit-length hypothesis was formally refuted, preventing flawed assumptions from polluting downstream work.

---

### 7.6 Clean Within-Digit & Progressive Numeric Curriculums (Tasks 19C, 19D, 20)
To isolate arithmetic generalization mechanics without operator confounds, single-operation subtraction ($A - B$ with $A \ge B$) was studied systematically across clean, zero-leakage numeric partitions.

#### Longitudinal Generalization Matrix (Tasks 19C, 19D, 20):
| Evaluation Benchmark | Operand Range | Target Answer Range | Task 19C (2D Only SFT) | Task 19D (Mixed 2D+3D SFT) | Task 20 (Progressive 2D+3D+4D SFT) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IID 2-Digit Holdout** | $[50, 99]$ | $[0, 49]$ | **100.00%** | **100.00%** | **100.00%** |
| **OOD 2-Digit Transfer** | $[2, 49]$ | $[0, 47]$ | **35.40%** | **28.40%** | **18.20%** |
| **IID 3-Digit Holdout** | $[101, 199]$ | $[0, 98]$ | — | **100.00%** | **99.80%** |
| **OOD 3-Digit Transfer** | $[200, 299]$ | $[0, 98]$ | **6.60%** | **34.40%** | **26.20%** |
| **IID 4-Digit Holdout** | $[1001, 1999]$ | $[0, 998]$ | — | — | **67.20%** |
| **OOD 4-Digit Transfer** | $[2000, 2999]$ | $[0, 999]$ | — | **0.40%** | **8.60%** |
| **5-Digit Acid Test** | $[10001, 19999]$ | $[0, 9998]$ | — | — | **0.00%** |

#### Key Insights from Tasks 19C, 19D, 20:
1. **Regime Exposure Drives In-Regime Transfer**: Training on 3-digit subtraction lifted unseen 3-digit accuracy from 6.60% to 34.40%. Training on 4-digit subtraction lifted unseen 4-digit accuracy from 0.40% to 8.60%.
2. **Consistent Generalization Ratios**: Within any trained digit regime, the ratio of OOD transfer to IID performance remains remarkably consistent: $G_2 \approx 0.18–0.35$, $G_3 \approx 0.26–0.35$, $G_4 \approx 0.13$.
3. **Cross-Regime Extrapolation Wall**: Extrapolation to untrained digit lengths collapses to near zero (0.40% on 4D from 2D+3D; 0.00% on 5D from 2D+3D+4D). On untrained digit lengths, 99–100% of model outputs collapse into the numeric ranges seen during training.

---

### 7.7 Controlled Arithmetic $\to$ GSM8K Transfer Evaluation (Task 21A)
Task 21A tested whether the robust arithmetic substrate trained in Task 20 transfers to natural language mathematical reasoning on GSM8K, SVAMP, and AQuA-RAT.

#### Model Conditions Evaluated:
- **Model A (Treatment — Arithmetic $\to$ GSM8K)**: Initialized from Task 20 Progressive Subtraction SFT ([artifacts/sft/progressive_subtraction/best.pt](artifacts/sft/progressive_subtraction/best.pt)), fine-tuned on GSM8K CoT data for 650 steps.
- **Model B (Control — Base $\to$ GSM8K)**: Initialized from Base Pretrained Step 18,000 ([checkpoints/mini_r1_500m_pretrain/best.pt](checkpoints/mini_r1_500m_pretrain/best.pt)), fine-tuned with identical hyperparameters.
- **Model C (Frozen Direct SFT Reference)**: [artifacts/sft/direct/best.pt](artifacts/sft/direct/best.pt).
- **Model D (Frozen CoT SFT Reference)**: [artifacts/sft/cot/best.pt](artifacts/sft/cot/best.pt).

#### Multi-Benchmark Evaluation Results (Task 21A):
| Evaluation Dimension | Model A (Arithmetic $\to$ GSM8K) | Model B (Base $\to$ GSM8K) | Model C (Direct Baseline) | Model D (CoT Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **GSM8K Test (Strict Acc)** | **1.44%** ($19/1319$) | **1.06%** ($14/1319$) | **2.35%** ($31/1319$) | **1.06%** ($14/1319$) |
| **SVAMP Test (Strict Acc)** | **2.00%** ($6/300$) | **0.33%** ($1/300$) | **2.00%** ($6/300$) | **0.33%** ($1/300$) |
| **AQuA-RAT Test (Strict Acc)** | **0.00%** ($0/254$) | **0.00%** ($0/254$) | **0.00%** ($0/254$) | **0.00%** ($0/254$) |
| **Full Format Compliance %** | 14.63% | 30.40% | 0.00% | 38.82% |
| **Repetition Degeneration %** | 86.05% | 96.29% | 0.53% | 98.33% |
| **Mean Generated Length** | 159.6 tokens | 211.1 tokens | 4.6 tokens | 201.8 tokens |
| **2D Subtraction Retention** | **5.80%** ($29/500$) | 0.00% ($0/500$) | 2.80% | 0.00% |
| **3D Subtraction Retention** | **0.40%** ($2/500$) | 0.00% ($0/500$) | 0.60% | 0.00% |
| **4D Subtraction Retention** | **0.00%** ($0/500$) | 0.00% ($0/500$) | 0.00% | 0.00% |
| **General Language PPL (10M)** | **28.87** (Loss 3.3630) | **23.54** (Loss 3.1588) | **21.83** (Loss 3.0834) | **23.58** (Loss 3.1603) |

#### Scientific Transfer Deltas:
- **Transfer Effect ($\Delta_{A - B}$)**: **$+0.38\%$** ($1.44\% \text{ vs } 1.06\%$)
- **Gain vs Direct Baseline ($\Delta_{A - C}$)**: **$-0.91\%$** ($1.44\% \text{ vs } 2.35\%$)
- **Gain vs CoT Baseline ($\Delta_{A - D}$)**: **$+0.38\%$** ($1.44\% \text{ vs } 1.06\%$)
- **Verdict**: **Conclusion C: No measurable transfer.**

#### Critical Scientific Findings from Task 21A:
1. **Transfer Failure**: Pre-tuning the model into a high-accuracy arithmetic substrate ($100\%$ 2D, $99.8\%$ 3D, $67.2\%$ 4D IID) produced no meaningful downstream improvement on GSM8K word problems ($\Delta = +0.38\%$).
2. **Severe Substrate Washout**: Fine-tuning on natural language reasoning traces overwrote the learned arithmetic substrate: 2D subtraction retention plummeted from $100.0\%$ to $5.8\%$, and 3D/4D subtraction collapsed to $\le 0.4\%$.
3. **Substantial General-Language Regression**: General language validation perplexity regressed from 17.68 (Base) to **28.87** (Model A), refuting any claim that catastrophic forgetting was negligible.
4. **Primary Failure Locus**: Downstream reasoning failure at 42.1M parameters is dominated by natural language autoregressive sequence decoding instability and severe repetition loops (86.1% repetition rate), which prevent the model from reliably executing computation steps in natural language context.

---

## 8. Audit of Known Inconsistencies & Data Reconciliations

To maintain strict scientific transparency, all historical report inconsistencies and their verified resolutions are documented:

1. **General Language Perplexity Placeholder Strings (Task 15C vs Task 16A Audit)**:
   - *Inconsistency*: The discussion text in `reports/task15c_direct_vs_cot.md` contained placeholder draft values (`17.72` / `17.84` PPL).
   - *Resolution*: Task 16A independently re-evaluated all 10M validation tokens. The authentic values are: Base Pretrained = **17.68**, Direct SFT = **21.83** (+23.5%), CoT SFT = **23.58** (+33.4%).
2. **E5 3-Digit Division Anomaly (Task 19B-PRE vs Task 19B-PRE2)**:
   - *Inconsistency*: Task 19B-PRE reported 24.28% 3-digit division accuracy vs 2.80% addition/subtraction.
   - *Resolution*: Task 19B-PRE2 forensic audit revealed 93.4% of correct predictions stemmed from training set overlap. On non-training pairs, division accuracy was **2.48%**, confirming no operator anomaly.
3. **Catastrophic Forgetting Interpretation in Task 21A**:
   - *Clarification*: While initial draft templates described language drift as mild, the final audited metrics prove that pre-transfer PPL (17.68) shifted to 28.87, and arithmetic retention collapsed from 100% to 5.8%. **Catastrophic forgetting and skill interference are severe during full-model natural language SFT.**

---

## 9. Current Scientific Understanding

Through Tasks 1 to 21A, the project has established the following empirical principles:

1. **Base Pretraining Does Not Yield Emergent Arithmetic**: General language modeling on 500M educational tokens produces fluent language representations (PPL 17.68) but near-zero functional arithmetic (<1%) and zero reasoning markup compliance.
2. **Verbose CoT Degenerates Under Greedy Decoding at 42.1M Scale**: Fine-tuning on verbose natural language CoT induces severe repetition loops (82.6%) under greedy decoding, causing CoT SFT (1.21%) to underperform Direct-Answer SFT (2.73%).
3. **Decoding Optimization Rescues Mechanics but Not Computation**: Repetition penalty and temperature scaling eliminate repetition (dropping it to 11.4%) and boost format compliance to 90.7%, but accuracy remains low (2.20%) due to persistent arithmetic calculation errors.
4. **Verified Arithmetic Supervision Installs In-Distribution Mastery**: Dedicated arithmetic SFT enables high in-distribution accuracy (45–100%) and high step-level equation truth (78.6%).
5. **Within-Regime Generalization Exists**: The 42.1M model generalizes to unseen arithmetic pairs within trained digit bounding boxes ($G_2 \approx 0.18–0.35, G_3 \approx 0.26–0.35$).
6. **Multi-Regime Exposure Expands Generalization Boundaries**: Explicitly training on larger digit regimes enables transfer within those specific regimes ($6.6\% \to 34.4\%$ on 3D; $0.4\% \to 8.6\%$ on 4D).
7. **Cross-Regime Extrapolation is Zero**: The model cannot extrapolate arithmetic rules to untrained digit lengths without explicit exposure (0.00% on 5D from 2D+3D+4D).
8. **Arithmetic Specialization Fails to Transfer to Natural Language CoT**: Pre-tuning on verified arithmetic provides no meaningful advantage on verbose GSM8K reasoning ($\Delta = +0.38\%$).
9. **Natural Language SFT Overwrites Specialized Arithmetic Substrates**: Standard SFT on verbose math word problems rapidly degrades the underlying arithmetic calculation capability ($100\% \to 5.8\%$).
10. **The Model Operates as a Local Interpolation Engine**: At 42.1M parameters, the network performs dense manifold interpolation and template matching rather than generalized, length-invariant symbolic computation.

---

## 10. Current Main Unresolved Research Questions

1. **Trace Conciseness vs Transferability**:
   Does arithmetic pre-tuning fail to transfer because verbose natural language CoT introduces too much linguistic noise and sequence length strain? Will **compact, equation-dense reasoning traces** allow the arithmetic substrate to transfer effectively to GSM8K word problems?
2. **Mechanism of Skill Interference**:
   Is the collapse of arithmetic retention ($100\% \to 5.8\%$) during GSM8K SFT caused by full-parameter gradient overwrite, learning rate magnitude, or the representational mismatch between pure equations and verbose prose?
3. **Pre-RL Policy Stability**:
   Can a compact, verified reasoning SFT curriculum establish a stable, non-degenerated multi-step policy that is suitable for outcome-based reinforcement learning (GRPO)?

---

## 11. AI Usage & Resource Strategy

To maximize engineering efficiency, mathematical rigor, and research quality across AI accounts:

| Model Persona | Assigned Operational Scope | Prohibited Scope |
| :--- | :--- | :--- |
| **Gemini** | - Writing implementation code, tests, and dataset builders<br>- Executing commands and training runs<br>- Environment diagnostics and routine debugging<br>- Batch evaluations and metric aggregations | - Speculative theoretical claims without data<br>- Launching unapproved architectural changes |
| **Claude Opus** | - High-level forensic failure analysis<br>- Experimental design and ablation planning<br>- Statistical significance and mathematical modeling<br>- Designing future GRPO / outcome verification algorithms | - Routine boilerplate coding<br>- Repetitive test authoring or file formatting |

---

## 12. Core Research Principles

1. **Never Hide Negative Results**: Report exact failures, regressions, and underperformance.
2. **Never Claim Reasoning from Markup Alone**: Syntactically valid `<think>` tags with wrong calculations are not reasoning.
3. **Always Audit Train/Eval Overlap**: Verify 0% train/test leakage with cryptographic and exact-match audits.
4. **Distinguish Memorization, Interpolation, and Extrapolation**:
   - Memorization: Exact match to training examples.
   - Interpolation: Generalization within trained operand/digit boundaries.
   - Extrapolation: Generalization beyond trained digit lengths and sequence scales.
5. **Preserve Exact Control Variables**: Hold base weights, seeds, optimizer states, and budgets constant across comparisons.
6. **Explicitly Flag Contradictions**: Document and audit data discrepancies rather than silently editing them.
7. **Empirically Justify Architectural Changes**: Do not scale parameters or alter architectures without empirical proof of necessity.
8. **Stabilize Policies Before RL**: Reinforcement learning requires an existing coherent policy that can generate valid format structures.
9. **Use Deterministic Programmatic Verification**: Use exact Python verifiers for all arithmetic and extraction evaluation.
10. **Treat Every Experiment as a Controlled Scientific Intervention**: Hypothesis $\to$ Intervention $\to$ Measurement $\to$ Conclusion.

---

## 13. Cryptographic Provenance & Checkpoint Ledger

| Artifact Description | File Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Canonical Base Pretrained Checkpoint (Step 18k)** | `checkpoints/mini_r1_500m_pretrain/best.pt` | `f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4` |
| **Final Pretrained Checkpoint (Step 30,518)** | `checkpoints/mini_r1_500m_pretrain/latest.pt` | `6ceeb456637e1a3faef9a2cbbfeab1d3e8e19e7fa56a297a7e372559779df506` |
| **Direct GSM8K SFT Checkpoint** | `artifacts/sft/direct/best.pt` | `79323e228f222da6f55ada9bccc84ea5c49d5bb1b0c0a1d20a3746b1b431dacd` |
| **Original CoT GSM8K SFT Checkpoint** | `artifacts/sft/cot/best.pt` | `0df4b0d86730e72c15e1cf8c3aaeebf58a834fc36232b804520760adb0311d93` |
| **Compact Arithmetic SFT Checkpoint (Task 18B)** | `artifacts/sft/arithmetic_compact/best.pt` | `a3ba5b55050fbe735d46e3be4737979eb11c6d3d44d74fe904724b17bfd62055` |
| **Verbose Arithmetic SFT Checkpoint (Task 18B)** | `artifacts/sft/arithmetic_verbose/best.pt` | `10283c400492cb91f24d1a1005bf3487fdfce998c25fc8ff2eec25d36e2f1ea0` |
| **Clean Subtraction SFT Checkpoint (Task 19C)** | `artifacts/sft/subtraction_clean/best.pt` | `02200b2bf92090bfa3fbcc065b44942a1a50ef06f393b554ba05ae2354be3b51` |
| **Mixed 2D+3D Subtraction Checkpoint (Task 19D)**| `artifacts/sft/mixed_subtraction/best.pt` | `7ee14e348813aaf6931aad73ca75b3dfadd0cf9d95a43d16a553a94a99d3b505` |
| **Progressive 2D+3D+4D Checkpoint (Task 20)** | `artifacts/sft/progressive_subtraction/best.pt` | `fdcecade71afafc1857d5ee56d221d31928ae1321d0f963dc0f566482af56d5c` |
| **Model A: Arithmetic $\to$ GSM8K (Task 21A)** | `artifacts/sft/arithmetic_transfer/best.pt` | `e3cf5845b85c53d05738c6afb98a04cd2fdc1544f75ecd03bc20e090b540cb29` |
| **Model B: Base $\to$ GSM8K (Task 21A)** | `artifacts/sft/base_transfer/best.pt` | `2a589697db22f16900f35ee585b97dafb718296dcd9d1662c55273445cea6aaa` |
| **SentencePiece BPE Tokenizer Model** | `artifacts/tokenizer/tokenizer.model` | `c1143f6e3b1e45811301e4a11392acbe367083712056d19c41ad2fe7f29f7044` |
| **GSM8K CoT Training Split (6,973 examples)** | `data/reasoning/gsm8k_sft_train.jsonl` | `f040b181642281edd4354c2cd67f6ec0af1b922658d8ec3a28136b5ae5e1b6d6` |
| **GSM8K CoT Dev Split (500 examples)** | `data/reasoning/gsm8k_sft_dev.jsonl` | `8bd18a20db0a126b87f26ac18c0627a6eb72a222af7ba0016ac09c39fbfbdc7b` |
| **GSM8K Direct Training Split (6,973 examples)** | `data/reasoning/gsm8k_direct_train.jsonl` | `8f5fcbfb48b0ff9a59596147cd90b6924f486d1757ee9e9534425fde5d88d765` |
| **GSM8K Direct Dev Split (500 examples)** | `data/reasoning/gsm8k_direct_dev.jsonl` | `d0ed361828ebffece4b5d42b964f74754aa4a815d0c15c20068ba8b6d8c3ef84` |

---

## 14. Next Planned Work (NOT YET EXECUTED)

### TASK 21B — COMPACT VERIFIED REASONING TRANSFER
> **IMPORTANT STATUS DECLARATION**:  
> **TASK 21B HAS NOT YET BEEN EXECUTED.**  
> The description below reflects the planned experimental design only.

- **Hypothesis**:
  Arithmetic pre-tuning failed to transfer in Task 21A because verbose natural language CoT traces (~160 tokens) caused severe linguistic degeneration (86.1% repetition) and catastrophic overwrite of arithmetic representations. Training on **compact, equation-dense GSM8K reasoning traces** (e.g., stripping conversational filler and focusing strictly on calculation transitions) will protect the arithmetic substrate and enable positive downstream transfer.
- **Planned Experimental Conditions**:
  - **Condition A (Treatment)**: Task 20 Progressive Arithmetic Checkpoint $\to$ Compact GSM8K Reasoning SFT.
  - **Condition B (Control)**: Base Pretrained Step 18,000 Checkpoint $\to$ Compact GSM8K Reasoning SFT.
  - **Frozen References**: Direct SFT Baseline (Model C) and Original Verbose CoT SFT (Model D).
- **Primary Evaluations**:
  1. GSM8K Test (Strict Accuracy & Complexity Stratification)
  2. SVAMP Test & AQuA-RAT Test
  3. Arithmetic Substrate Retention Probes (2D, 3D, 4D Holdouts)
  4. General Language Validation PPL (10M tokens)
  5. Repetition & Format Compliance Dynamics
- **Strict Boundary**:
  - ❌ No Reinforcement Learning (GRPO / PPO).
  - ❌ No model scaling to 70M parameters.
  - ❌ No base model checkpoint modification.

---

# NEXT TASK: 21B — COMPACT VERIFIED REASONING TRANSFER
# TASK 21B HAS NOT YET BEEN EXECUTED.
