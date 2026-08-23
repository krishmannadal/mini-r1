# Mini-R1 Architecture Scaling Analysis

## 1. Current Architecture (Baseline Reference)

Your existing implementation uses:

| Component | Value |
|---|---|
| `vocab_size` (V) | 8,000 |
| `d_model` | 128 |
| `n_layers` (L) | 2 |
| `n_heads` (H) | 4 |
| `d_ff` | 512 (= 4 × d_model) |
| `max_seq_len` (T) | 32 |
| `bias` | False |
| Weight tying | Yes (lm_head shares tok_emb weights) |
| Total params | ~1.42M |

**Key architectural decisions preserved in all candidates below:**
- Pre-LayerNorm Transformer decoder blocks
- GELU activation in FFN
- Separate Q, K, V projections (not fused)
- Learned positional embeddings
- Weight tying between token embedding and lm_head
- No bias in attention or FFN linear layers
- Causal mask registered as buffer

---

## 2. Parameter Counting Formula

Your architecture has these parameter-bearing components:

### Per-Layer Parameters (one Transformer block)

| Component | Shape | Count (no bias) |
|---|---|---|
| `ln1.weight` | (d_model,) | d_model |
| `ln1.bias` | (d_model,) | d_model |
| `q_proj.weight` | (d_model, d_model) | d_model² |
| `k_proj.weight` | (d_model, d_model) | d_model² |
| `v_proj.weight` | (d_model, d_model) | d_model² |
| `out_proj.weight` | (d_model, d_model) | d_model² |
| `ln2.weight` | (d_model,) | d_model |
| `ln2.bias` | (d_model,) | d_model |
| `ffn.fc1.weight` | (d_ff, d_model) | d_model × d_ff |
| `ffn.fc2.weight` | (d_model, d_ff) | d_model × d_ff |

**Per-layer total:**

$$P_{\text{layer}} = 4 \cdot d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} + 4 \cdot d_{\text{model}}$$

Since we use $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ (standard GPT ratio):

$$P_{\text{layer}} = 4d^2 + 2 \cdot d \cdot 4d + 4d = 4d^2 + 8d^2 + 4d = 12d^2 + 4d$$

### Non-Layer Parameters

| Component | Shape | Count |
|---|---|---|
| `tok_emb.weight` | (V, d_model) | V × d_model |
| `pos_emb.weight` | (T, d_model) | T × d_model |
| `ln_f.weight` | (d_model,) | d_model |
| `ln_f.bias` | (d_model,) | d_model |
| `lm_head.weight` | **Tied** to tok_emb | 0 (tied) |

**Non-layer total:**

$$P_{\text{non-layer}} = V \cdot d + T \cdot d + 2d = (V + T + 2) \cdot d$$

### Total Model Parameters

$$\boxed{P_{\text{total}} = L \cdot (12d^2 + 4d) + (V + T + 2) \cdot d}$$

Where:
- $L$ = number of layers
- $d$ = d_model
- $V$ = vocab_size = 8,000
- $T$ = max_seq_len = 512

---

## 3. Candidate Architectures

I searched for configurations where `d_model` is divisible by `n_heads`, `d_ff = 4 × d_model`, and the total parameter count lands near the target. Here are three candidates:

### Architecture A: ~20M Parameters

| Hyperparameter | Value |
|---|---|
| `d_model` | 384 |
| `n_layers` | 8 |
| `n_heads` | 6 |
| `d_k` (per head) | 64 |
| `d_ff` | 1,536 |
| `max_seq_len` | 512 |
| `vocab_size` | 8,000 |

**Parameter count:**

$$P_{\text{layer}} = 12 \times 384^2 + 4 \times 384 = 12 \times 147{,}456 + 1{,}536 = 1{,}769{,}472 + 1{,}536 = 1{,}771{,}008$$

$$P_{\text{layers}} = 8 \times 1{,}771{,}008 = 14{,}168{,}064$$

$$P_{\text{non-layer}} = (8{,}000 + 512 + 2) \times 384 = 8{,}514 \times 384 = 3{,}269{,}376$$

$$\boxed{P_A = 14{,}168{,}064 + 3{,}269{,}376 = 17{,}437{,}440 \approx 17.4\text{M}}$$

---

### Architecture B: ~40M Parameters

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `n_layers` | 12 |
| `n_heads` | 8 |
| `d_k` (per head) | 64 |
| `d_ff` | 2,048 |
| `max_seq_len` | 512 |
| `vocab_size` | 8,000 |

**Parameter count:**

$$P_{\text{layer}} = 12 \times 512^2 + 4 \times 512 = 12 \times 262{,}144 + 2{,}048 = 3{,}145{,}728 + 2{,}048 = 3{,}147{,}776$$

$$P_{\text{layers}} = 12 \times 3{,}147{,}776 = 37{,}773{,}312$$

$$P_{\text{non-layer}} = (8{,}000 + 512 + 2) \times 512 = 8{,}514 \times 512 = 4{,}359{,}168$$

$$\boxed{P_B = 37{,}773{,}312 + 4{,}359{,}168 = 42{,}132{,}480 \approx 42.1\text{M}}$$

---

### Architecture C: ~60M Parameters

| Hyperparameter | Value |
|---|---|
| `d_model` | 640 |
| `n_layers` | 12 |
| `n_heads` | 10 |
| `d_k` (per head) | 64 |
| `d_ff` | 2,560 |
| `max_seq_len` | 512 |
| `vocab_size` | 8,000 |

**Parameter count:**

$$P_{\text{layer}} = 12 \times 640^2 + 4 \times 640 = 12 \times 409{,}600 + 2{,}560 = 4{,}915{,}200 + 2{,}560 = 4{,}917{,}760$$

$$P_{\text{layers}} = 12 \times 4{,}917{,}760 = 59{,}013{,}120$$

$$P_{\text{non-layer}} = (8{,}000 + 512 + 2) \times 640 = 8{,}514 \times 640 = 5{,}448{,}960$$

$$\boxed{P_C = 59{,}013{,}120 + 5{,}448{,}960 = 64{,}462{,}080 \approx 64.5\text{M}}$$

---

## 4. Training Memory Breakdown

During FP32 training with AdamW, GPU memory is consumed by **five components**. Let me explain each one and show the exact byte calculations.

### 4.1 Component Explanations

#### (a) Model Weights

Every trainable parameter is stored as a 32-bit (4-byte) float.

$$M_{\text{weights}} = P \times 4 \text{ bytes}$$

#### (b) Gradients

PyTorch stores one gradient value (FP32) per parameter during backpropagation. Same size as the weights.

$$M_{\text{gradients}} = P \times 4 \text{ bytes}$$

#### (c) Optimizer States (AdamW)

AdamW maintains **two additional FP32 buffers per parameter**:
- **First moment** $m_t$ (exponential moving average of gradients)
- **Second moment** $v_t$ (exponential moving average of squared gradients)

$$M_{\text{optimizer}} = P \times 2 \times 4 = P \times 8 \text{ bytes}$$

#### (d) Activations (Forward Pass Intermediates)

During training, PyTorch's autograd retains intermediate tensors from the forward pass so it can compute gradients during backpropagation. These are the **dominant memory consumer** and scale with both batch size ($B$) and sequence length ($T$).

For each Transformer layer, the retained activations include:

| Activation | Shape | Bytes (FP32) |
|---|---|---|
| Input to LayerNorm1 | (B, T, d) | 4BTd |
| Q, K, V projections | 3 × (B, H, T, d_k) = 3 × (B, T, d) | 12BTd |
| Attention scores | (B, H, T, T) | 4BHT² |
| Attention output (post-softmax × V) | (B, T, d) | 4BTd |
| Input to LayerNorm2 | (B, T, d) | 4BTd |
| FFN intermediate (after fc1, pre-GELU) | (B, T, d_ff) = (B, T, 4d) | 16BTd |
| FFN output | (B, T, d) | 4BTd |

**Per-layer activation memory:**

$$A_{\text{layer}} = 4BT \cdot (d + 3d + d + d + 4d + d) \cdot 4 \text{ bytes}$$

Wait — let me be more careful. Each tensor stored is FP32 (4 bytes per element). Counting elements:

| Tensor | Elements |
|---|---|
| LN1 input residual | BTd |
| Q, K, V (after split) | 3BTd |
| Attention matrix (post-softmax) | BHT² |
| Concat context | BTd |
| LN2 input residual | BTd |
| FFN hidden (fc1 output, for GELU backward) | BT × d_ff = 4BTd |
| FFN output | BTd |

$$\text{Elements per layer} = BTd(1 + 3 + 1 + 1 + 4 + 1) + BHT^2 = 11 \cdot BTd + BHT^2$$

$$A_{\text{layer}} = (11 \cdot BTd + BHT^2) \times 4 \text{ bytes}$$

**Additional non-layer activations:**
- Embedding output: $BTd$ elements → $4BTd$ bytes
- Final LayerNorm input: $BTd$ elements → $4BTd$ bytes
- Logits: $BTV$ elements → $4BTV$ bytes

$$A_{\text{total}} = L \times (11BTd + BHT^2) \times 4 + (2BTd + BTV) \times 4$$

#### (e) Temporary / Workspace Memory

PyTorch and CUDA allocate temporary buffers for:
- cuBLAS workspace for matrix multiplications
- CUDA memory fragmentation overhead
- Python/PyTorch framework overhead

**Rule of thumb:** ~5–15% overhead on top of the above.

---

### 4.2 Full Memory Calculations

Fixed constants: $V = 8{,}000$, $T = 512$, batch size $B = 8$ (reasonable starting point for reasoning training).

#### Architecture A (17.4M, d=384, L=8, H=6)

**Static Memory (does not depend on batch size):**

| Component | Formula | MB |
|---|---|---|
| Weights | 17,437,440 × 4 | 66.5 |
| Gradients | 17,437,440 × 4 | 66.5 |
| Optimizer (m + v) | 17,437,440 × 8 | 133.1 |
| **Static subtotal** | | **266.1 MB** |

**Activation Memory (B=8):**

Per-layer elements: $11 \times 8 \times 512 \times 384 + 8 \times 6 \times 512^2$

$$= 11 \times 1{,}572{,}864 + 8 \times 6 \times 262{,}144$$

$$= 17{,}301{,}504 + 12{,}582{,}912 = 29{,}884{,}416 \text{ elements/layer}$$

Total layer activations: $8 \times 29{,}884{,}416 = 239{,}075{,}328$ elements

Non-layer activations: $(2 \times 8 \times 512 \times 384 + 8 \times 512 \times 8{,}000)$ = $3{,}145{,}728 + 32{,}768{,}000 = 35{,}913{,}728$ elements

Total activation elements: $239{,}075{,}328 + 35{,}913{,}728 = 274{,}989{,}056$

$$M_{\text{activations}} = 274{,}989{,}056 \times 4 = 1{,}099{,}956{,}224 \text{ bytes} \approx 1{,}049 \text{ MB}$$

**Total for Architecture A (B=8):**

| Component | MB |
|---|---|
| Weights | 66.5 |
| Gradients | 66.5 |
| Optimizer states | 133.1 |
| Activations | 1,049 |
| Overhead (~10%) | 131.5 |
| **Total** | **~1,447 MB ≈ 1.41 GB** |

> [!TIP]
> Easily fits. You could increase batch size to B=16 (~2.5 GB) or even B=24 (~3.5 GB).

---

#### Architecture B (42.1M, d=512, L=12, H=8)

**Static Memory:**

| Component | Formula | MB |
|---|---|---|
| Weights | 42,132,480 × 4 | 160.8 |
| Gradients | 42,132,480 × 4 | 160.8 |
| Optimizer (m + v) | 42,132,480 × 8 | 321.6 |
| **Static subtotal** | | **643.2 MB** |

**Activation Memory (B=8):**

Per-layer elements: $11 \times 8 \times 512 \times 512 + 8 \times 8 \times 512^2$

$$= 11 \times 2{,}097{,}152 + 8 \times 8 \times 262{,}144$$

$$= 23{,}068{,}672 + 16{,}777{,}216 = 39{,}845{,}888 \text{ elements/layer}$$

Total layer activations: $12 \times 39{,}845{,}888 = 478{,}150{,}656$ elements

Non-layer: $(2 \times 8 \times 512 \times 512 + 8 \times 512 \times 8{,}000) = 4{,}194{,}304 + 32{,}768{,}000 = 36{,}962{,}304$ elements

Total: $478{,}150{,}656 + 36{,}962{,}304 = 515{,}112{,}960$

$$M_{\text{activations}} = 515{,}112{,}960 \times 4 = 2{,}060{,}451{,}840 \text{ bytes} \approx 1{,}965 \text{ MB}$$

**Total for Architecture B (B=8):**

| Component | MB |
|---|---|
| Weights | 160.8 |
| Gradients | 160.8 |
| Optimizer states | 321.6 |
| Activations | 1,965 |
| Overhead (~10%) | 260.8 |
| **Total** | **~2,869 MB ≈ 2.80 GB** |

> [!TIP]
> Comfortably fits in 6 GB. Room for batch size B=16 (~4.7 GB) or gradient accumulation.

---

#### Architecture C (64.5M, d=640, L=12, H=10)

**Static Memory:**

| Component | Formula | MB |
|---|---|---|
| Weights | 64,462,080 × 4 | 246.0 |
| Gradients | 64,462,080 × 4 | 246.0 |
| Optimizer (m + v) | 64,462,080 × 8 | 492.0 |
| **Static subtotal** | | **984.0 MB** |

**Activation Memory (B=8):**

Per-layer elements: $11 \times 8 \times 512 \times 640 + 8 \times 10 \times 512^2$

$$= 11 \times 2{,}621{,}440 + 8 \times 10 \times 262{,}144$$

$$= 28{,}835{,}840 + 20{,}971{,}520 = 49{,}807{,}360 \text{ elements/layer}$$

Total layer activations: $12 \times 49{,}807{,}360 = 597{,}688{,}320$ elements

Non-layer: $(2 \times 8 \times 512 \times 640 + 8 \times 512 \times 8{,}000) = 5{,}242{,}880 + 32{,}768{,}000 = 38{,}010{,}880$ elements

Total: $597{,}688{,}320 + 38{,}010{,}880 = 635{,}699{,}200$

$$M_{\text{activations}} = 635{,}699{,}200 \times 4 = 2{,}542{,}796{,}800 \text{ bytes} \approx 2{,}425 \text{ MB}$$

**Total for Architecture C (B=8):**

| Component | MB |
|---|---|
| Weights | 246.0 |
| Gradients | 246.0 |
| Optimizer states | 492.0 |
| Activations | 2,425 |
| Overhead (~10%) | 340.9 |
| **Total** | **~3,750 MB ≈ 3.66 GB** |

> [!IMPORTANT]
> Fits at B=8, but leaves only ~2.3 GB headroom. B=16 would push to ~5.5 GB, tight but possible. Mixed precision (FP16 activations + FP32 optimizer) would cut activation memory in half, giving much more room.

---

## 5. Comparison Summary Table

| | **Arch A** | **Arch B** | **Arch C** |
|---|---|---|---|
| d_model | 384 | 512 | 640 |
| n_layers | 8 | 12 | 12 |
| n_heads | 6 | 8 | 10 |
| d_k (per head) | 64 | 64 | 64 |
| d_ff | 1,536 | 2,048 | 2,560 |
| **Parameters** | **17.4M** | **42.1M** | **64.5M** |
| Static memory | 266 MB | 643 MB | 984 MB |
| Activations (B=8) | 1,049 MB | 1,965 MB | 2,425 MB |
| **Total (B=8)** | **~1.4 GB** | **~2.8 GB** | **~3.7 GB** |
| Max B (FP32, 6 GB) | ~32 | ~16 | ~10 |

---

## 6. Recommendation

> [!IMPORTANT]
> **I recommend Architecture B: 42M parameters (d_model=512, n_layers=12, n_heads=8).**

### Reasons

**1. Sweet spot for 6 GB VRAM.**
At B=8 with FP32, it uses only ~2.8 GB — less than half your VRAM. This leaves ample headroom for:
- Increasing batch size to B=16 (~4.7 GB)
- Gradient accumulation for effective batch sizes of 32–64
- Mixed precision (when introduced later) cutting activation memory nearly in half, enabling B=32+

**2. Research-meaningful depth (12 layers).**
Chain-of-thought reasoning requires compositional multi-step processing. Shallow models (8 layers in Arch A) have fewer compositional "thinking steps." Research on reasoning in Transformers (e.g., Merrill & Sabharwal 2023) shows that deeper models solve harder compositional problems. 12 layers gives the model enough depth for multi-step mathematical reasoning while keeping d_model moderate.

**3. Per-head dimension = 64 matches proven scaling recipes.**
GPT-2 (117M), GPT-2-medium (345M), LLaMA-7B, and SmolLM all use $d_k = 64$. This head dimension has empirically optimal query-key resolution for attention patterns. All three architectures share this, but Arch B achieves it with 8 heads at d_model=512, which is a standard and well-studied configuration (identical to GPT-2 Small proportions but with fewer layers).

**4. Arch A (17.4M) is too small for meaningful reasoning.**
With only 8 layers and d_model=384, the model's representational capacity is limited. Chain-of-thought traces in GSM8K contain 4–8 reasoning steps with intermediate calculations. A model this small may memorize patterns but cannot generalize well to unseen multi-step problems.

**5. Arch C (64.5M) is feasible but risky.**
It fits at B=8 but leaves little headroom. Training stability benefits from larger batch sizes (B=16 is often the minimum for stable loss curves with reasoning data). At B=16 in FP32, Arch C would consume ~5.5 GB — right at the edge of your 6 GB limit, risking OOM from CUDA fragmentation or peaks during backward pass. You *could* make it work with mixed precision, but that adds a task dependency.

**6. Arch B enables clean experiments.**
The SFT comparison (direct-answer vs. CoT) requires training two full models under identical conditions. Arch B lets you train both at B=16 with comfortable memory margin, ensuring neither run hits OOM mid-experiment — which would waste GPU hours and complicate comparison.

---

## 7. How Memory Scales with Batch Size (Architecture B)

For quick reference, here is how Architecture B memory scales:

| Batch Size | Activations | Total (with overhead) | Fits in 6 GB? |
|---|---|---|---|
| B=1 | 245 MB | 978 MB | ✅ |
| B=4 | 983 MB | 1,791 MB | ✅ |
| B=8 | 1,965 MB | 2,869 MB | ✅ |
| B=12 | 2,948 MB | 3,951 MB | ✅ |
| B=16 | 3,930 MB | 5,031 MB | ✅ (tight) |
| B=20 | 4,913 MB | 6,112 MB | ⚠️ Edge |
| B=24 | 5,896 MB | 7,193 MB | ❌ OOM |

> [!NOTE]
> With FP16 mixed precision (activations in half precision), activation memory roughly halves. This would let Architecture B train at B=32+ comfortably.

---

## 8. Mathematical Assumptions Made

1. **d_ff = 4 × d_model** — standard GPT/LLaMA FFN expansion ratio.
2. **No bias parameters** — matching your current `bias=False` design. This saves ~0.1% parameters but simplifies counting.
3. **Weight tying** — lm_head shares tok_emb weights (0 additional params for the output projection).
4. **FP32 training** — 4 bytes per parameter/gradient/activation element. Mixed precision analysis deferred per your instruction.
5. **Activation count is an approximation** — PyTorch's autograd may retain slightly more or fewer tensors depending on operation fusion. The counts above are based on your explicit implementation (no `torch.utils.checkpoint`, no fused kernels).
6. **Causal mask buffer** — stored as bool, negligible: $T^2 = 512^2 = 262{,}144$ bytes ≈ 0.25 MB per layer. Not included in the main calculations.
7. **Overhead at 10%** — conservative estimate for CUDA allocator fragmentation, cuBLAS workspace, and PyTorch framework tensors.

---

## 9. Largest Theoretically Trainable Architecture

Working backwards from a hard 6 GB (6,144 MB) limit at B=8:

$$6{,}144 = 1.1 \times \left[\underbrace{16P}_{\text{weights+grads+optim}} + \underbrace{(11 \cdot 8 \cdot 512 \cdot d \cdot L + 8H \cdot 512^2 \cdot L + 2 \cdot 8 \cdot 512 \cdot d + 8 \cdot 512 \cdot 8000) \cdot 4}_{\text{activations}}\right] / (1024^2)$$

With $L = 12$, $H = d/64$, $d_{\text{ff}} = 4d$, this equation yields $d \approx 720$, giving roughly **~80M parameters**. However, this is the absolute ceiling with **zero headroom** — a single spike during backward pass would OOM.

A practical ceiling is **~65M at B=8** or **~42M at B=16**, both leaving sufficient safety margin.
