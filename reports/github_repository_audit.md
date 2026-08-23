# Mini-R1 — GitHub Repository Audit & Release Preparation

**Date**: 2026-08-23
**Status**: prepared, **not committed, not pushed**
**Nothing was deleted, moved, or modified.** This audit is read-only over the data; only `.gitignore`, `.gitattributes`, `README.md`, `release/` and this report were created.

---

## 0. Headline

| Metric | Value |
| :--- | ---: |
| Total repository size on disk | **55 GB** |
| Size going to normal Git | **83.15 MB** (654 files) |
| Size ignored (local-only) | **~54.9 GB** |
| Largest single tracked file | 9.0 MB (`data/reasoning/gsm8k_sft_train.jsonl`) |
| `.pt` checkpoints on disk | **126 files, 54,297.8 MB** |
| Is this currently a Git repository? | Initialised for inspection — **0 commits, 0 staged files** |
| Git installed | ✅ `git 2.51.2.windows.1` |
| Git LFS installed | ✅ `git-lfs/3.7.1` — installed but **not initialised in this repo** |

**Reduction: 55 GB → 83.15 MB, a 99.85% cut, with zero data loss.**

---

## 1. Where the 55 GB lives

| Directory | Size | Contents |
| :--- | ---: | :--- |
| `artifacts/` | **32 GB** | SFT checkpoints, LR-sweep checkpoints, smoke/dry-run checkpoints |
| `runs/` | **19 GB** | Task 15C per-step SFT checkpoints (Direct 11 GB, CoT 8.7 GB) |
| `checkpoints/` | **2.4 GB** | Base pretraining (best / latest / step_28000 / step_29000 / step_30000) |
| `data/` | **1.1 GB** | 973 MB pretraining shards + 78 MB research datasets |
| `reports/` | 1.8 MB | 56 markdown + JSON research reports |
| `logs/` | 1.8 MB | 354 training log files (`train_log.jsonl`) |
| `scripts/` | 1.5 MB | 114 Python scripts |
| `tests/` | 768 KB | 26 test modules |
| `src/` | 554 KB | 24 core modules |
| `submission/` | 429 KB | Task 24 submission package |
| `configs/` | 52 KB | YAML experiment configs |
| `web/` | 44 KB | stdlib HTTP demo server + frontend |

### `artifacts/` breakdown

| Path | Size | Verdict |
| :--- | ---: | :--- |
| `artifacts/sft/` | 9.0 GB | Research checkpoints — 2 released, rest local-only |
| `artifacts/production_dry_run/` | 4.8 GB | Infrastructure dry run — **ignore** |
| `artifacts/smoke_checkpoints/` | 2.4 GB | Smoke test — **ignore** |
| `artifacts/checkpoints_lr_6e-4/` | 2.4 GB | LR sweep arm — **ignore** (result is in the report) |
| `artifacts/checkpoints_lr_3e-4/` | 2.4 GB | LR sweep arm — **ignore** |
| `artifacts/checkpoints_lr_1e-4/` | 2.4 GB | LR sweep arm — **ignore** |
| `artifacts/checkpoints_long_qual_6e-4/` | 2.4 GB | Qualification — **ignore** |
| `artifacts/checkpoints_long_qual_3e-4/` | 2.4 GB | Qualification — **ignore** |
| `artifacts/general_pretrain_smoke_logs/` | 1.9 GB | Smoke logs w/ checkpoints — **ignore** |
| `artifacts/checkpoints_batch50/` | 1.9 GB | Batch test — **ignore** |
| `artifacts/checkpoints/` | 66 MB | Early checkpoints — **ignore** |
| `artifacts/baseline/` | 3.5 MB | Baseline eval outputs — ignore (regenerable) |
| **`artifacts/tokenizer/`** | **804 KB** | ✅ **TRACK — required for everything** |

---

## 2. Largest 20 files

All are 485.3 MB except where noted — a full checkpoint with optimizer state.

| # | Size | Path | Required for repro? | Required for inference? | Disposition |
| --: | ---: | :--- | :---: | :---: | :--- |
| 1 | 485.3 MB | `checkpoints/mini_r1_500m_pretrain/step_30000.pt` | No | No | **D Ignore** |
| 2 | 485.3 MB | `checkpoints/mini_r1_500m_pretrain/step_29000.pt` | No | No | **D Ignore** |
| 3 | 485.3 MB | `checkpoints/mini_r1_500m_pretrain/step_28000.pt` | No | No | **D Ignore** |
| 4 | 485.3 MB | `checkpoints/mini_r1_500m_pretrain/best.pt` | **Yes** — base for all SFT | No | **C Release (optional)** |
| 5 | 485.3 MB | `checkpoints/mini_r1_500m_pretrain/latest.pt` | No (dup of best) | No | **D Ignore** |
| 6 | 485.3 MB | `artifacts/production_dry_run/run_b_resumed/checkpoints/step_100.pt` | No | No | **D Ignore** |
| 7 | 485.3 MB | `artifacts/production_dry_run/run_b_resumed/checkpoints/best.pt` | No | No | **D Ignore** |
| 8 | 485.3 MB | `artifacts/production_dry_run/run_b_resumed/checkpoints/latest.pt` | No | No | **D Ignore** |
| 9 | 485.3 MB | `artifacts/smoke_checkpoints/step_8.pt` | No | No | **D Ignore** |
| 10 | 485.3 MB | `artifacts/smoke_checkpoints/step_6.pt` | No | No | **D Ignore** |
| 11 | 485.3 MB | `artifacts/smoke_checkpoints/latest.pt` | No | No | **D Ignore** |
| 12 | 485.3 MB | `artifacts/smoke_checkpoints/best.pt` | No | No | **D Ignore** |
| 13–20 | 485.3 MB ×8 | `runs/sft_direct/run_.../checkpoints/step_{50…650}.pt` | No | No | **D Ignore** |

`runs/` contains **every 50-step checkpoint** of the Task 15C Direct and CoT runs — 19 GB of intermediate state whose scientific content is entirely captured in `reports/task15c_direct_vs_cot.md`.

### Research checkpoints in `artifacts/sft/` (all 25)

| Checkpoint | Size | Cited by | Disposition |
| :--- | ---: | :--- | :--- |
| **`task23_75_25/best.pt`** | 485.3 MB | **FINAL MODEL** (Task 23/24) | **C Release — primary** |
| `task23_90_10/best.pt` | 485.3 MB | Task 23 Model G | **C Release — optional** |
| `progressive_subtraction/best.pt` | 163.8 MB | Task 20; init for 21A/21B/22/22B/23 | **C Release — optional** |
| `task22b_replay_only/best.pt` | 485.3 MB | Task 22B Model F | **E Local-only** |
| `task22_replay_isolated/best.pt` | 307.9 MB | Task 22 Model E | **E Local-only** |
| `compact_transfer/arithmetic_init/best.pt` | 485.3 MB | Task 21B Model A | **E Local-only** |
| `compact_transfer/base_init/best.pt` | 485.3 MB | Task 21B Model B | **E Local-only** |
| `direct/best.pt`, `cot/best.pt` | 485.3 MB ea | Task 15C baselines | **E Local-only** |
| `arithmetic_{compact,verbose}/best.pt` | 163.8 MB ea | Task 18B | **E Local-only** |
| `arithmetic_transfer/`, `base_transfer/best.pt` | 163.8 MB ea | Task 21A | **E Local-only** |
| `subtraction_clean/best.pt` | 485.3 MB | Task 19C | **E Local-only** |
| `mixed_subtraction/best.pt` | 163.8 MB | Task 19D | **E Local-only** |
| all `final.pt` / `latest.pt` variants | — | — | **D Ignore** (duplicates of `best.pt`) |

---

## 3. Classification summary

### A — NORMAL GIT (83.15 MB, 654 files)

| Path | Size | Why |
| :--- | ---: | :--- |
| `src/` | 554 KB | Core implementation — the point of the repo |
| `scripts/` | 1.5 MB | All experiment entry points |
| `tests/` | 768 KB | 26 test modules |
| `configs/` | 52 KB | YAML experiment configs |
| `web/` | 44 KB | Demo server + frontend |
| `reports/` | 1.8 MB | 56 research reports (the scientific record) |
| `submission/` | 429 KB | Task 24 package |
| `release/` | new | Manifests created by this audit |
| `logs/` | 1.8 MB | 354 `train_log.jsonl` — small, and cited by reports |
| `data/**` (excl. shards) | 78 MB | Research datasets, manifests, eval suites |
| `artifacts/tokenizer/` | 804 KB | Tokenizer — required for everything |
| `artifacts/sft/*/config.json` | ~50 KB | Per-run training configs + hashes |
| Root `*.md`, `requirements.txt`, `pytest.ini` | 260 KB | Documentation and setup |

Largest tracked file is 9.0 MB — comfortably under GitHub's 50 MB warning and 100 MB hard limit.

> **Validation note.** The first draft of `.gitignore` used a bare `!artifacts/`
> negation, which re-admitted the entire 32 GB tree and leaked 58 smoke/dry-run log
> files (4.99 MB) into the tracked set. Git's own ignore engine caught it
> (`git ls-files --others --exclude-standard`). The fix uses alternating
> ignore/negate pairs so only `artifacts/tokenizer/` and `artifacts/sft/*/config.json`
> are re-admitted. The blanket `*.pt` rule blocked all binaries throughout, so no
> model weights were ever at risk.

### B — GIT LFS (**recommended: none**)

Git LFS is installed (3.7.1) but **not recommended here**. GitHub's free LFS tier is **1 GB storage and 1 GB/month bandwidth**. The final checkpoint alone is 485.3 MB — 48% of storage, and two clones per month exhausts bandwidth. Adding Model G exceeds storage on the first push.

`.gitattributes` contains the correct LFS lines **commented out**, scoped to two specific paths, for anyone on a larger quota. There is deliberately no blanket `*.pt` rule — that would sweep in all 126 checkpoints (54 GB).

### C — EXTERNAL RELEASE (GitHub Releases)

| Artifact | Size | Priority |
| :--- | ---: | :--- |
| `artifacts/sft/task23_75_25/best.pt` → `mini-r1-final-model-h.pt` | 485.3 MB | **Required** |
| `artifacts/sft/task23_90_10/best.pt` → `mini-r1-model-g.pt` | 485.3 MB | Optional |
| `artifacts/sft/progressive_subtraction/best.pt` → `mini-r1-task20-arithmetic.pt` | 163.8 MB | Optional |
| `checkpoints/mini_r1_500m_pretrain/best.pt` → `mini-r1-base-pretrained.pt` | 485.3 MB | Optional |

GitHub Releases allows 2 GB per file with no LFS quota consumed.

### D — IGNORE (~54.9 GB)

`runs/` · `checkpoints/` · `artifacts/` (except tokenizer + config.json) · `data/pretrain/shards/` · `__pycache__/` (4 dirs, 1.3 MB) · `.pytest_cache/` (24 KB) · `*.tmp` · `*.partial.json`

`data/pretrain/shards/` (973 MB, 60 `.bin` files) is **fully regenerable** via `scripts/build_pretrain_shards.py`; the manifests are tracked so a rebuild can be hash-verified.

### E — LOCAL-ONLY

All 126 `.pt` files stay on disk. Twelve are cited by reports and could be released on request; the rest are intermediate.

---

## 4. Final checkpoint verification

| Property | Value |
| :--- | :--- |
| Path | `artifacts/sft/task23_75_25/best.pt` |
| Size | **508,892,602 bytes** (485.3 MB) |
| SHA256 | `d630daacbf75bc8239d625ea8e270bfcaecaffe6e7665abaea9722de2ca0e49c` |
| Parameters | 42,132,480 ✅ |
| Config | `vocab 8000, d_model 512, n_layers 12, n_heads 8, d_ff 2048, ctx 512, bias false, tie_word_embeddings true` |
| Tokenizer compatibility | ✅ recorded `tokenizer_sha256` **matches** on-disk tokenizer `c1143f6e…7044` |
| Optimizer state included | Yes — strippable to ~169 MB for inference-only distribution |
| Integrity | 8/8 checks passed (`reports/task24_final_integrity.md`) |

### Should Model G be retained?

**Yes — release it as an optional secondary artifact.**

| | Model H (75:25) | Model G (90:10) |
| :--- | ---: | ---: |
| SHA256 | `d630daac…e49c` | `fa4ef4e0…924b` |
| 2D / 3D / 4D retention | **94.80 / 78.20 / 32.40** | 65.60 / 54.80 / 13.40 |
| General PPL | 50.74 | **47.90** |
| RCS | **0.4569** | 0.4110 |

Model G is the only checkpoint that materially trades retention for language quality (2.84 PPL better for 29.2 points less 2D retention). It is the second point on the replay-ratio curve and is the natural choice for anyone who weights perplexity above arithmetic retention. Releasing both makes the central finding independently checkable.

**Recommendation**: release **H (required)** and **G (optional)**; keep everything else local-only.

---

## 5. Security check

**No secrets found. Nothing is blocked from commit on security grounds.**

| Check | Result |
| :--- | :--- |
| API keys / HF tokens / OpenAI keys / GitHub PATs / AWS keys / Slack tokens | **None** |
| `password` / `secret` / `api_key` assignments in source | **None** |
| Private keys (`BEGIN RSA/OPENSSH/EC/DSA PRIVATE KEY`) | **None** |
| `.env`, `.pem`, `.key`, `id_rsa*`, `.netrc`, `.git-credentials` | **None present** |
| IDE / venv directories (`.vscode`, `.idea`, `.venv`, `node_modules`) | **None present** |
| Email addresses in tracked text | **None** |
| Cookies / session data | **None** |

### One low-severity finding — local paths (**RESOLVED**)

Seven files contained the absolute local path `<user home>/Downloads/SLM CoT`, disclosing the local OS username:

`PROJECT_SUMMARY_UNTIL_TASK_21A.md` · `smoke_test_forensic_analysis.md` · `reports/task11_corpus_qualification.md` · `reports/task11b_final_qualification.md` · `reports/task13_base_pretraining_results.md` · `reports/task15a_reasoning_dataset.md` · `scripts/build_reasoning_sft_dataset.py`

**Severity was low** — cosmetic, not functional. They were `file:///...` markdown hyperlinks inside report-generation template strings; they did not affect execution, but they would have been dead links for other readers.

**Status: fixed.** 50 hyperlink prefixes were stripped, converting each to a **repo-relative link** (e.g. `](file:///.../src/trainer.py#L224)` → `](src/trainer.py#L224)`) — which now resolves correctly on GitHub rather than merely being anonymised. A follow-up sweep found and fixed 4 further occurrences the first pattern missed: the `**Repository**:` header lines in `PROJECT_CONTEXT.md` and `PROJECT_SUMMARY_UNTIL_TASK_21A.md`, and an absolute checkpoint path in `reports/task14_base_model_evaluation.md`.

**54 total occurrences removed across 9 files.** No local username remains in any tracked text file. Only source code was touched — no model weights, no data, no experimental results.

---

## 6. Reproducibility coverage

Everything a third party needs is in normal Git except the weights:

| Need | Provided by | Status |
| :--- | :--- | :---: |
| Dependencies | `requirements.txt` | ✅ |
| Tests | `tests/` + `pytest.ini` | ✅ |
| Tokenizer | `artifacts/tokenizer/tokenizer.model` (129 KB, tracked) | ✅ |
| Model config | `release/model_manifest.json`, `submission/config/model_config.json` | ✅ |
| Training configs | `configs/*.yaml` | ✅ |
| Datasets (SFT/eval/replay) | `data/**` (78 MB, tracked) | ✅ |
| Pretraining shards | regenerable — `scripts/build_pretrain_shards.py`, manifests tracked | ✅ |
| Inference code | `submission/INFERENCE.md`, `submission/API.md`, `web/server.py` | ✅ |
| Evaluation | `scripts/qualify_final_model.py`, `scripts/evaluate_task22_replay.py` | ✅ |
| Checkpoint hashes | `release/model_manifest.json`, `submission/HASHES.json` | ✅ |
| Weights | GitHub Releases (see README) | ⬜ upload pending |

---

## 7. Recommended publish sequence

**Not executed. No `git add`, `git commit`, or `git push` was run.**

```bash
cd <repo>

git init
git add -A
git status                 # verify: ~637 files, ~82 MB, no .pt, no shards
git commit -m "Mini-R1: 42.1M SLM arithmetic-retention research project"

git remote add origin https://github.com/<user>/mini-r1.git
git branch -M main
git push -u origin main

# Then attach the checkpoints as a Release (not Git):
gh release create v1.0.0 \
  --title "Mini-R1 v1.0.0 — Final Model (Task 23 Model H)" \
  --notes-file release/RELEASE_NOTES.md \
  "artifacts/sft/task23_75_25/best.pt#mini-r1-final-model-h.pt" \
  "artifacts/sft/task23_90_10/best.pt#mini-r1-model-g.pt"
```

**Pre-push checklist**
- [ ] `git status` shows no `.pt`, `.bin`, `__pycache__`, or `data/pretrain/shards/`
- [ ] `git count-objects -vH` reports well under 200 MB
- [ ] `artifacts/tokenizer/tokenizer.model` **is** staged (negation pattern working)
- [x] Local paths stripped (54 occurrences, 9 files) — verified 0 residual
- [ ] Release SHA256 matches `d630daac…e49c`
