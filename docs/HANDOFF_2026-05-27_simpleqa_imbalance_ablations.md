# Handoff — SimpleQA imbalance ablations & issue #124 (2026-05-27)

## Context

SimpleQA has a 93.5% hallucination rate for Llama 3.1 8B — by design, the benchmark was curated to contain obscure facts models frequently get wrong. This causes `contrastive_logprob_recon` to collapse: knn_auroc 0.575 vs 0.80+ on well-balanced datasets.

Analysis showed a U-shape relationship between |halluc_rate − 0.5| and contrastive knn_auroc (r = −0.851 excluding mmlu). Datasets near 50/50 get the best scores; extreme imbalance in either direction hurts.

Two ablations were kicked off to investigate:

---

## Ablation 1: Flipped label convention (in progress)

**What:** `contrastive_logprob_recon_b5` — treats hallucinations (93.5% majority) as the inlier cluster instead of correct answers. Config: `ignore_label=0`, `flip_auroc=true`.

**Job:** `05eb21f3ebf5` on alphagpu06. Script: `scripts/dispatch/run_exp_simpleqa_flipped.sh`. Chains Llama then Qwen3.

**Early results** (4/5 Llama seeds complete):
- knn_auroc ~0.651 (+0.076 vs standard 0.575)
- cosine_auroc ~0.624 (+0.22 vs standard 0.401)
- mahal_auroc ~0.631 (+0.23 vs standard 0.399)

Flipping helps substantially but still lags well-balanced datasets (0.80+). The residual gap is the harder signal on SimpleQA (obscure factual recall vs structured QA).

**Once done:** results land in `runs/baseline_comparison_simpleqa_flipped_memmap/` and `runs/baseline_comparison_simpleqa_qwen3_flipped_memmap/`. Run `python scripts/results_table.py` to pick them up.

---

## Ablation 2: Cross-dataset supplementation — issue #124

**Issue:** https://github.com/hyang0129/HalluLens/issues/124

**What:** Merge SimpleQA train + sampled PopQA to get a 50/50 balanced training set at ~2× SimpleQA size. Evaluate on SimpleQA test only.

**Target composition:**
- All 3460 SimpleQA train samples (3236 hallucinated + 224 correct)
- From PopQA train: ~3236 non-hallucinated + ~224 hallucinated
- Combined: ~6920 samples, ~3460 / ~3460 (50/50)

**Both captures available** (same model, same memmap format, same input_dim=4096):
- `shared/icr_capture/simpleqa_train_Llama-3.1-8B-Instruct` — 3460 samples
- `shared/icr_capture/popqa_train_Llama-3.1-8B-Instruct` — 10595 samples

**Implementation plan:**

1. Write `scripts/merge_icr_captures.py` — reads two icr_capture memmap dirs, samples specified hallucinated/non-hallucinated counts from each, writes a merged icr_capture dir in standard format.

   Key things the script must handle:
   - Read `generation.jsonl` from each source to get hallucinated labels and valid sample indices
   - Sample without replacement: from SimpleQA take all 3460; from PopQA take ~3236 non-hallucinated + ~224 hallucinated
   - Concatenate the memmap arrays (activations, attentions, logprobs) and write new `generation.jsonl` + `config.json`
   - The merged dir should be a valid icr_capture that `MemmapActivationParser` can read as-is

2. Add dataset config `configs/datasets/simpleqa_popqa_merged_memmap.json` pointing at the merged dir (train) and existing SimpleQA test dir (test).

3. Add experiment config `configs/experiments/baseline_comparison_simpleqa_popqa_merged_memmap.json` — method `contrastive_logprob_recon`, 5 seeds.

4. Run the merge script on a cluster node (CPU is fine — no GPU needed) to produce the merged dir.

5. Dispatch training via a new `scripts/dispatch/run_exp_simpleqa_popqa_merged.sh`.

**Scientific question:** If knn_auroc on SimpleQA test improves → the "correct answer" activation cluster is domain-agnostic. If not → the model is learning a domain fingerprint (PopQA vs SimpleQA style) rather than a correctness signal. Either result is meaningful.

**Reference icr_capture format** (for merge script):
- `config.json` — metadata (n_samples, layers, etc.)
- `generation.jsonl` — one JSON per sample with at least `hallucinated` bool
- `activations/` — memmap arrays per layer
- `attentions/` — memmap arrays per layer  
- `logprobs.npy` and related

Read `activation_logging/inference_capture_writer.py` (`_init_append`, `write_sample`) for the exact format.

---

## Other active jobs (as of 2026-05-27)

| Job | Node | What | Status |
|---|---|---|---|
| `a5b242d49a32` | alphagpu05 | triviaqa_test Llama capture | ⏳ 9954/14000 |
| `05eb21f3ebf5` | alphagpu06 | simpleqa flipped ablation | 🟢 running |
| alphagpu08, alphagpu10 | — | idle | available |

Once triviaqa_test capture finishes, dispatch `run_exp_triviaqa_llama.sh` to alphagpu08 or alphagpu10.

## SLURM allocations

4 nodes allocated, ~44–49h remaining as of 2026-05-27 13:00 EDT:
- alphagpu05-8879, alphagpu06-8880, alphagpu08-8878, alphagpu10-8877
