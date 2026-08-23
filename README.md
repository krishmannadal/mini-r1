# Mini-R1

**A 42.1M-parameter decoder-only SLM research project.**

Built from scratch in pure PyTorch — no HuggingFace model classes, no prebuilt attention — pretrained on 500M tokens on a single 6 GB RTX 4050 laptop GPU, then used to study whether verified arithmetic supervision transfers to natural-language mathematical reasoning.

> **This is a research artifact, not a usable math assistant.** GSM8K accuracy is **2.43%**, statistically indistinguishable from a trivial direct-answer baseline (2.35%). Its value is the controlled experiments and the negative results, not the model's capability. See [Limitations](#15-limitations).

---

## 1. Overview

Mini-R1 asks a question that is hard to answer at large scale: **if you teach a small model arithmetic with deterministically verified supervision, does that competence transfer to word problems?**

At 7B parameters arithmetic is already latent from pretraining, so you cannot separate what you taught from what was already there. At 42M you must teach it — which makes the question clean. And because arithmetic is deterministic, every claim in this project is checkable by Python rather than by another model.

The answer turned out to be **no** for transfer, and **yes** for something else the project found along the way: catastrophic forgetting of a learned computational skill is preventable by replaying the *right* data distribution.

## 2. Research question

> Can a 42.1M-parameter language model, given verified arithmetic supervision, acquire arithmetic competence that transfers to natural-language mathematical reasoning (GSM8K)?

**Secondary**: if it does not transfer, can the failure be localised to a specific, measurable mechanism?

## 3. Architecture

| | |
| :--- | :--- |
| Type | Decoder-only causal Transformer, pre-LayerNorm |
| Parameters | **42,132,480** |
| Layers / model dim / heads / FFN | 12 / 512 / 8 / 2048 |
| Context | 512 tokens |
| Positions | Learned absolute (512 × 512) |
| Linear bias | None |
| Weight tying | **Yes** — `lm_head.weight` *is* `embeddings.tok_emb.weight` |
| Tokenizer | SentencePiece BPE, 8,000 vocab; every digit is its own token; `<think>`/`<answer>` atomic |

Derivation: `12 × (12d² + 4d) + Vd + 512d + 2d` = 37,773,312 + 4,096,000 + 262,144 + 1,024 = **42,132,480**.

## 4. Training

**Pretraining** — 500M tokens, 8.02 hours, avg 17,316 tok/s, peak VRAM 3,409.7 MB of 6,144 MB. AdamW, peak LR 6e-4 (chosen by a 1,000-step sweep: 2.4995 vs 3e-4's 2.6337) cosine → 6e-5, warmup 100, weight decay 0.01, grad clip 1.0, FP16 AMP with GradScaler (0 skipped steps). 16,384 tokens/optimizer step × 30,518 steps.

Best checkpoint at **step 18,000 / 294.9M tokens**, validation loss 2.8726, **PPL 17.68**. Validation loss plateaued at ~295M tokens; the final 41% of the token budget produced no measurable improvement.

**Fine-tuning (final model)** — 650 steps, effective batch 32 (micro 8 × accum 4), AdamW peak LR 1e-4 cosine → 1e-5, warmup 50, seed 42, completion-only loss (prompt tokens masked to −100), all parameters trainable.

## 5. Dataset

**Pretraining**: 500M training / 10M validation tokens, **60% `cosmopedia-v2`** (synthetic textbooks) + **40% `fineweb-edu-dedup`** (deduplicated educational web). `python-edu` excluded for schema incompatibility. EOS-separated contiguous packing achieved **100% token density** (naive truncation wasted 58.4%). Stored as uint16 shards, memory-mapped.

> The 60:40 ratio was a reasoned choice, **not an ablated one**. It is not claimed to be optimal.

**Fine-tuning mixture (final model)** — deterministic global-index schedule, `is_replay = (idx % 4 == 0)`:

| Source | Share | Examples seen |
| :--- | ---: | ---: |
| Compact verified GSM8K reasoning | 75.000% | 15,600 |
| Operator-matched replay arithmetic | 12.506% | 2,600 |
| Task 20 progressive subtraction | 12.494% | 2,600 |

Every arithmetic equation in both replay corpora is verified by deterministic Python evaluation.

## 6. Experiments

| Task | Question | Result |
| :--- | :--- | :--- |
| 13 | Can 42M pretrain on 6 GB? | Yes — PPL 17.68 |
| 14 | Zero-shot reasoning? | GSM8K 0.76%, 0% format compliance |
| 15C | Direct vs CoT supervision? | **CoT hurt**: 2.73% vs 1.21% (p=0.0051) |
| 16B | Is that a decoding artifact? | **No** — repetition 82.6%→11.45%, accuracy still below Direct |
| 17A | Can it do the arithmetic? | No — reasoning-shaped text, wrong numbers |
| 18B | Does verified arithmetic SFT work? | IID 45–49%, OOD numbers 1.00% |
| 19B | Is the arithmetic result real? | **Leakage found** — 24.28% was really **2.48%** |
| 19C/D/20 | Does regime exposure generalise? | Within regime yes; 5-digit **0.00%** |
| 21B | Does arithmetic transfer to GSM8K? | **No** — Δ = **−0.30%** |
| 22 | Does parameter freezing protect it? | **No** — −14.00 pts; tied embeddings froze the output head |
| 22B | Was freezing the cause? | **Yes** — +30.38 pts when unfrozen |
| 23 | Does replay preserve arithmetic? | **Yes** — 2D retention 4.40% → **94.80%** |
| 24 | Final qualification | 8/8 integrity checks passed |

## 7. Key findings

**Established** (matched controls, traceable artifacts):

1. Verified arithmetic SFT produces strong **in-distribution** arithmetic at 42M — 100% on held-out 2-digit subtraction.
2. That competence is **regime-local**: 0.00% at 5 digits with identical tokenization.
3. CoT supervision **hurt** at this scale, and it was not a decoding artifact.
4. Arithmetic competence **does not transfer** to GSM8K (Δ = −0.30%).
5. GSM8K fine-tuning causes **catastrophic forgetting** of arithmetic (100% → ~4%).
6. **Parameter freezing is not a viable protection** here — with tied embeddings it froze the output head, causing non-terminating generation.
7. **Targeted replay preserves arithmetic**: 25% replay containing the probed regime → 94.80% retention.
8. **Replay composition matters at least as much as dose**: 50% *off-distribution* replay preserved nothing (3.80%); 25% *on-distribution* preserved almost everything (94.80%). The arm with the most replay retained the least.

**Not established**: general symbolic reasoning · arbitrary arithmetic extrapolation · any RL effect (none was run) · any scaling effect (one size only) · an optimal replay ratio (ratio and composition are confounded in our series).

## 8. Final model

**Task 23 Model H** · `artifacts/sft/task23_75_25/best.pt`
**SHA256** `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c`

| Benchmark | Result |
| :--- | ---: |
| GSM8K test (1,319), greedy | **2.43%** |
| GSM8K step-level equation validity | **31.64%** |
| SVAMP (300) | 1.67% |
| AQuA (254) | **0.00%** |
| E1 IID 2-digit subtraction | **89.60%** |
| E3 IID 3-digit | **84.60%** |
| E5 IID 4-digit | 38.40% |
| E7 5-digit acid test (never trained) | **0.60%** |
| Retention probe 2D / 3D / 4D | **94.80% / 78.20% / 32.40%** |
| General-language PPL (10M held-out) | **50.74** |

## 9. Reproducibility

Everything except the model weights is in this repository. Weights are distributed via [GitHub Releases](#14-downloading-the-model).

```bash
python scripts/build_pretrain_shards.py                                  # rebuild 973 MB shards
python scripts/build_replay_arithmetic.py                                # rebuild replay corpus
python scripts/train_task22_replay_sft.py --config configs/task23_75_25.yaml   # ~25 min, RTX 4050
python scripts/qualify_final_model.py --batch_size 64                    # final benchmarks
```

Determinism: seed 42 throughout; checkpoints store `torch`/`cuda`/`numpy`/`python` RNG states, the dataset cursor and `tokens_seen`. All datasets and checkpoints carry SHA256s (`release/model_manifest.json`, `submission/HASHES.json`).

## 10. Installation

```bash
git clone https://github.com/<user>/mini-r1.git
cd mini-r1
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                 # 26 test modules
```

Requires Python 3.11+, PyTorch with CUDA (CPU works, slower). ~1 GB VRAM for inference; ~4 GB to train.

## 11. Inference

Download the checkpoint (below), then:

```python
import torch, sys; sys.path.insert(0, ".")
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer

tok = Tokenizer("artifacts/tokenizer/tokenizer.model")
ck = torch.load("mini-r1-final-model-h.pt", map_location="cpu", weights_only=False)
model = MiniR1Transformer(MiniR1Config.from_dict(ck["model_config"]))
model.load_state_dict(ck["model_state_dict"]); model.eval()
```

Prompt format is exact — deviating degrades output sharply:

```
Question: Calculate: 87 - 43\n
```
→
```
<think>
87 - 43 = 44
</think>
<answer>44</answer>
```

Full runnable example: **[`submission/INFERENCE.md`](submission/INFERENCE.md)** · Wrapper API: **[`submission/API.md`](submission/API.md)** · Local demo: `python web/server.py` (stdlib `http.server` on port 8000).

> **Verify every equation.** Step-level validity on GSM8K is 31.64% — roughly two in three emitted equations are wrong. `submission/API.md` provides `solve()` returning `all_steps_verified`; gate on it.

## 12. Evaluation

```bash
python scripts/qualify_final_model.py --batch_size 64      # GSM8K, SVAMP, AQuA, E1-E7, PPL
python scripts/evaluate_task22_replay.py --batch_size 64   # replay-ratio comparison suite
```

Results land in `reports/task24_final_benchmarks.json`. Note the harness left-pads without an attention mask while the model uses learned absolute positions; a batch-size-1 control measured this as worth ~3–14 points of equation validity, so **absolute numbers are conservative**. All arms were measured identically, so comparisons hold.

## 13. Repository structure

```
src/                 Core implementation (model, attention, trainer, tokenizer, eval)
scripts/             114 experiment entry points
tests/               26 test modules
configs/             YAML experiment configs
data/                Research datasets, manifests, eval suites (78 MB)
  pretrain/shards/   NOT in Git - regenerable (973 MB)
artifacts/
  tokenizer/         SentencePiece model (tracked)
  sft/*/config.json  Per-run training configs + hashes (tracked)
  sft/*/*.pt         NOT in Git - see Releases
reports/             56 research reports - the scientific record
submission/          Task 24 package: model card, research summary, inference docs
release/             Model + release manifests
logs/                354 training logs (1.8 MB)
web/                 Local demo server (stdlib http.server)
```

**Key documents**: [`reports/Mini-R1_final_research_summary.md`](reports/Mini-R1_final_research_summary.md) · [`reports/Mini-R1_model_card.md`](reports/Mini-R1_model_card.md) · [`reports/Mini-R1_MASTER_RESEARCH_AND_INTERVIEW_GUIDE.md`](reports/Mini-R1_MASTER_RESEARCH_AND_INTERVIEW_GUIDE.md) · [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)

## 14. Downloading the model

Checkpoints are **not in Git**. The repository is 83 MB; the checkpoints are 54 GB on disk, of which two are published.

Weights are attached to the [GitHub Release](https://github.com/<user>/mini-r1/releases) rather than tracked with Git LFS — GitHub's free LFS tier is 1 GB storage and 1 GB/month bandwidth, which a single 485 MB checkpoint would nearly exhaust. Releases allow 2 GB per file with no quota.

| Asset | Size | SHA256 |
| :--- | ---: | :--- |
| `mini-r1-final-model-h.pt` **(required)** | 485.3 MB | `d630daac…e49c` |
| `mini-r1-model-g.pt` (optional) | 485.3 MB | `fa4ef4e0…924b` |
| `mini-r1-task20-arithmetic.pt` (optional) | 163.8 MB | `fdcecade…6d5c` |
| `mini-r1-base-pretrained.pt` (optional) | 485.3 MB | `f058a558…62a4` |

```bash
gh release download v1.0.0 -p "mini-r1-final-model-h.pt"
sha256sum mini-r1-final-model-h.pt   # PowerShell: Get-FileHash <file> -Algorithm SHA256
```

Full hashes in [`release/model_manifest.json`](release/model_manifest.json). Checkpoints include optimizer state; strip it for inference-only use (509 MB → ~169 MB) — see `submission/INFERENCE.md`.

## 15. Limitations

1. **Not a reasoning model.** GSM8K 2.43% vs direct-answer SFT's 2.35% — one problem out of 1,319. No configuration in the project exceeded ~2.4%.
2. **Arithmetic is regime-local.** 89.60% on trained 2-digit ranges, 20.00% just outside, **0.60% at 5 digits**.
3. **Substantial language regression.** PPL 50.74 vs the 17.68 base — 2.9× worse.
4. **No multiple-choice capability.** AQuA 0.00%, with 0.0% format compliance.
5. **77.3% repetition** on GSM8K under greedy decoding.
6. **Single seed (42)**, one run per arm. No variance estimates anywhere.
7. **The replay-ratio curve is confounded** — arms differ in replay composition as well as share, so no optimal ratio is claimed.
8. **Benchmark numbers are conservative** (harness padding artifact, §12).
9. **No RL was ever run.** GRPO was a stated goal and did not happen.
10. **512-token context.**
11. **No safety tuning, alignment, or content filtering.** It emits fluent, correctly formatted, confidently wrong arithmetic — a more hazardous failure mode than visible breakdown.

**Intended use**: research only. Not for education, tutoring, finance, health, or any decision where a wrong number carries cost.

## 16. Research history

Tasks 1–12 built the infrastructure (from-scratch Transformer, tokenizer, packing, LR sweep, corpus qualification). Task 13 pretrained. Tasks 14–16 established that reasoning was absent and that decoding could not supply it. Tasks 17–20 taught verified arithmetic — and, critically, Task 19B's audit of the project's own favourable result found 35.46% train/eval overlap and revised 24.28% down to 2.48%. Tasks 21–23 tested transfer (null), parameter freezing (backfired, then isolated as the cause), and replay ratio (the positive result). Task 24 froze and qualified the final model.

Four methodological problems were found and corrected by the project itself: data leakage, a confounded manipulation, an evaluation harness artifact, and a denominator-sensitive metric. One earlier conclusion was **retracted** — Task 22 blamed its replay corpus for generation collapse; Task 22B proved the cause was parameter freezing.

Full narrative: [`reports/Mini-R1_final_research_summary.md`](reports/Mini-R1_final_research_summary.md).

---

**Status**: research complete through Task 24. Final model frozen 2026-08-23.
**License**: [MIT](LICENSE)
