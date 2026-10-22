# Mini-R1
### A small Transformer, an arithmetic-retention study, and an auditable experiment trail

Mini-R1 contains a **42,132,480-parameter decoder-only Transformer implemented in PyTorch**, pretraining and fine-tuning scripts, deterministic arithmetic data generators, and reports through Task 24. It helps readers inspect how narrow arithmetic skills survive—or disappear—during word-problem fine-tuning.

**Research artifact · Python / PyTorch / SentencePiece · no public hosted inference demo**

[Inspect final benchmarks](reports/task24_final_benchmarks.md) · [Read the model card](reports/Mini-R1_model_card.md) · [Download published weights](https://github.com/krishmannadal/mini-r1/releases/tag/v1.0.0) · [Inference walkthrough](submission/INFERENCE.md)

> The final model's **recorded GSM8K accuracy is 2.43%**. This is not a dependable math assistant or evidence of general reasoning. The experimental record is the primary deliverable.

## Start with the evidence

The following are **repository-reported results**, checked against the committed reports/configurations during the documentation audit. Training and checkpoint inference were not independently rerun for this refresh.

| Recorded measurement | Result | What it describes |
| --- | ---: | --- |
| GSM8K test, greedy, 1,319 questions | 2.43% | Strict final-answer accuracy |
| E1 IID two-digit subtraction | 89.60% | A specific held-out arithmetic suite |
| Two-digit retention probe | 94.80% | A different probe; not interchangeable with E1 |
| Untrained five-digit suite | 0.60% | Limited extrapolation in this setting |
| General-language perplexity | 50.74 | Regression from recorded base PPL 17.68 |

Sources: [final benchmark report](reports/task24_final_benchmarks.md), [machine-readable results](reports/task24_final_benchmarks.json), [model manifest](release/model_manifest.json).

**Example input format** (illustrative, not a newly observed model prediction):

```text
Question: Calculate: 87 - 43
```

The trained output format uses `<think>...</think>` and `<answer>...</answer>`. Formatting does not establish that the arithmetic or reasoning is correct.

## What is implemented

- Causal multi-head attention, pre-LayerNorm blocks, learned absolute positions and tied input/output embeddings in [src/](src/).
- SentencePiece BPE with 8,000 tokens and digit-aware tokenization.
- Packed pretraining data, mixed-precision training, checkpoint/RNG state handling and evaluation scripts.
- Verified synthetic arithmetic supervision, word-problem fine-tuning, freezing controls and replay arms.
- Reports, data manifests, leakage audits and two published release assets: **Model H** and **Model G**.

The committed architecture/configuration is **12 layers, width 512, 8 heads, FFN 2048, context 512**. Some older comments and the separate portfolio site contain different dimensions; use [src/model.py](src/model.py) and the [release manifest](release/model_manifest.json).

## What the replay experiments establish

| Arm | Total replay share | Replay content | Recorded 2D retention |
| --- | ---: | --- | ---: |
| A | 0% | None | 4.40% |
| G | 10% | Half operator-matched arithmetic, half progressive subtraction | 65.60% |
| H | 25% | Same two replay sources as G | 94.80% |
| F | 50% | Operator-matched arithmetic only | 3.80% |

G and H provide a **same-composition 10%-versus-25% comparison**, with the same configured initialization, optimizer, step budget and seed. At a fixed step budget, increasing replay also reduces primary-data exposure. Each arm has one seed, so the numbers do not estimate run-to-run variation.

**F versus G/H changes both composition and replay ratio.** It cannot isolate composition's causal effect, show that composition matters more than dose, or establish a globally optimal 25% ratio. The supported observation is that the tested mixtures containing progressive subtraction retained much more of the probed subtraction regime. A factorial composition-by-ratio experiment with repeated seeds is still needed.

The separate [Task 22/22B control](reports/task22b_replay_only.md) holds the replay corpus and training settings fixed while changing the trainable parameter set. That supports a freezing-related effect in this run; it does not isolate every frozen tensor's individual mechanism.

## Install

Python 3.11+; run commands from the repository root. CUDA is optional for inference and recommended for training. Install a PyTorch build appropriate for your hardware before the remaining requirements if necessary.

```bash
git clone https://github.com/krishmannadal/mini-r1.git
cd mini-r1
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install numpy
python -m pytest
```

The requirements are not a fully pinned environment. A fresh GPU training environment and the full test suite were not exercised during this README refresh. No API key or `.env` is required for local model inference.

## Run the released model

Download `mini-r1-final-model-h.pt` (508,892,602 bytes) from the release above. With GitHub CLI installed, run from this clone:

```bash
gh release download v1.0.0 --repo krishmannadal/mini-r1 --pattern mini-r1-final-model-h.pt
sha256sum mini-r1-final-model-h.pt
mkdir -p artifacts/sft/task23_75_25
cp mini-r1-final-model-h.pt artifacts/sft/task23_75_25/best.pt
```

Expected SHA-256: `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c`.
PowerShell equivalents: `Get-FileHash`, `New-Item -ItemType Directory -Force`, and `Copy-Item`.

Use the complete runnable Python example in [submission/INFERENCE.md](submission/INFERENCE.md). That example loads the path above, formats the question and generates tokens. The API document contains wrapper/example code, not a deployed service.

**Local web UI caveat:** `python web/server.py` serves a comparison UI at `http://localhost:8000`, but its registry refers to earlier checkpoints and does not include released Model H/G. A clean clone plus the public H release is therefore **not a working UI inference demo**. Use the inference walkthrough for H; do not treat a rendered page as successful model execution.

<details>
<summary>Training, evaluation and artifact prerequisites</summary>

```bash
python scripts/build_pretrain_shards.py
python scripts/build_replay_arithmetic.py
python scripts/train_task22_replay_sft.py --config configs/task23_75_25.yaml
python scripts/qualify_final_model.py --batch_size 64
```

These are research entry points, not a self-contained four-command reproduction. Pretraining needs external corpus access and generated shards. Fine-tuning expects the **Task 20 initialization checkpoint** named and hashed in the configuration; that checkpoint is not one of the two published v1.0.0 assets. Final qualification expects additional checkpoints/datasets and dependencies described in the scripts and reports. Do not substitute Model H as the initialization and call it a reproduction.

The historical evaluator uses left-padding without an attention mask with absolute positional embeddings. This can affect scores and comparisons when prompt lengths differ. A reported batch-size-one check does not prove every benchmark is a conservative lower bound. Re-evaluate with corrected positional handling before making stronger claims.

`src/` contains the model/training stack; `configs/` the experiment settings; `reports/` the scientific record; `data/` datasets/manifests; `release/` artifact identity; `submission/` inference and model documentation.

</details>

## Limitations and next work

- One model size and one seed per arm; no general scaling conclusion or seed-variance estimate.
- No reinforcement-learning stage was run; the name does not imply an R1 reproduction.
- Narrow arithmetic success did not establish useful natural-language mathematical reasoning.
- Earlier favorable results were revised after leakage checks; read the [research summary](reports/Mini-R1_final_research_summary.md) alongside individual reports.
- Next: separate composition and dose with a factorial study, repeat seeds, correct/re-run the evaluation harness, and package an H-compatible inference UI.

## Credits and license

Project by Krishnakumar Mandal. Cite the linked reports and exact checkpoint/configuration when using the work.

[LICENSE](LICENSE) contains the existing MIT source-code license and third-party data notices. Dataset and benchmark terms remain separate. No new license is introduced by this documentation update.
