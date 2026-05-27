# Handoff — TriviaQA + SimpleQA capture & training (updated 2026-05-27)

## TL;DR (current state)

| Task | Status |
|------|--------|
| `simpleqa_train_Llama-3.1-8B-Instruct` capture | ✅ DONE (3460/3460) |
| `simpleqa_train_Qwen3-8B` capture | ✅ DONE (3460/3460) |
| `triviaqa_train_Llama-3.1-8B-Instruct` capture | ✅ DONE (11000/11000) |
| `triviaqa_test_Llama-3.1-8B-Instruct` capture | ⏳ 9954/14000 — RUNNING on alphagpu05 (job `a5b242d49a32`) |
| SimpleQA baseline training experiment | ✅ DONE (job `ec648541eb1e`) |
| SimpleQA flipped ablation (halluc-as-inlier) | 🟢 RUNNING on alphagpu06 (job `05eb21f3ebf5`) |

---

## Capture history

Original captures launched 2026-05-25 were killed by SLURM wall-time expiry. Subsequent re-dispatches on 2026-05-26 recovered the simpleqa and triviaqa_train captures fully. The triviaqa_test run stalled at 9954.

| Dispatch job | Script | Node | Status | Date |
|---|---|---|---|---|
| `84a2f05591e3` | `run_simpleqa_llama.sh` | alphagpu11 | unknown (timeout) | 2026-05-25 |
| `2aa3bd5ba9a7` | `run_simpleqa_qwen3.sh` | alphagpu17 | unknown (timeout) | 2026-05-25 |
| `0edc15ca1deb` | `run_triviaqa_train_llama.sh` | alphagpu02 | unknown (timeout) | 2026-05-25 |
| `32ddb3744594` | `run_triviaqa_test_llama.sh` | alphagpu15 | unknown (timeout) | 2026-05-25 |
| `2bc052171620` | `run_triviaqa_test_llama.sh` | alphagpu05 | finished | 2026-05-26 |
| `2a2d45950fa1` | `run_triviaqa_train_llama.sh` | alphagpu08 | finished | 2026-05-26 |
| `578476738bf7` | `run_simpleqa_llama.sh` | alphagpu10 | finished | 2026-05-26 |
| `a0bb851e1ebf` | `run_simpleqa_qwen3.sh` | alphagpu06 | finished | 2026-05-26 |

---

## triviaqa_test capture (in progress)

Job `a5b242d49a32` on alphagpu05, resumed in append mode from 9954/14000. ~3.5h remaining.

Monitor:
```bash
ssh empire-ai 'wc -l ~/LLM_research/HalluLens/shared/icr_capture/triviaqa_test_Llama-3.1-8B-Instruct/generation.jsonl'
```

---

## Training: SimpleQA baseline experiment (DONE)

Job `ec648541eb1e` completed 2026-05-27. Results in `runs/baseline_comparison_simpleqa_memmap/` and `runs/baseline_comparison_simpleqa_qwen3_memmap/`. See results_table.py for full numbers.

Key finding: `contrastive_logprob_recon` collapsed on SimpleQA (knn 0.575 Llama / 0.671 Qwen3) due to 93.5% hallucination rate. `llmsknow_probe` was best at 0.763/0.751.

---

## SimpleQA flipped ablation (in progress)

Job `05eb21f3ebf5` on alphagpu06, running `scripts/dispatch/run_exp_simpleqa_flipped.sh`.

Uses `contrastive_logprob_recon_b5` (`ignore_label=0`, `flip_auroc=true`) — treats hallucinations as the inlier cluster. Early results (4/5 Llama seeds): knn ~0.651 (+0.076 vs standard). Chains Llama → Qwen3.

Monitor:
```bash
ssh empire-ai 'grep -E "Completed|Running" ~/LLM_research/HalluLens/shared/logs/05eb21f3ebf5.log | tail -5'
```

---

## Still missing: triviaqa Qwen3 dispatch script

No `run_triviaqa_qwen3.sh` (train or test) has been authored. Configs exist:
- [configs/datasets/triviaqa_qwen3_memmap.json](configs/datasets/triviaqa_qwen3_memmap.json)
- [configs/experiments/baseline_comparison_triviaqa_qwen3_memmap.json](configs/experiments/baseline_comparison_triviaqa_qwen3_memmap.json)

Mirror `run_triviaqa_train_llama.sh` + a test variant, swapping `--model Qwen/Qwen3-8B` and `--out-dir shared/icr_capture/triviaqa_{train,test}_Qwen3-8B`.

---

## Files / locations

- Dispatch scripts: [scripts/dispatch/](scripts/dispatch/)
- Capture output (Empire AI): `~/LLM_research/HalluLens/shared/icr_capture/`
- Logs: `~/LLM_research/HalluLens/shared/logs/<jobid>.log`
- Dispatch manifest: `~/LLM_research/HalluLens/shared/gpu_jobs.json`
- Writer source: [activation_logging/inference_capture_writer.py](activation_logging/inference_capture_writer.py)
- Capture script: [scripts/capture_inference.py](scripts/capture_inference.py)
