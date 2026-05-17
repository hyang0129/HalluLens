# Ablation: LLM-as-judge labels for HotpotQA (faithful ICR Probe reproduction)

## Motivation

Our reproduction of the ICR Probe baseline on HotpotQA + Llama-3.1-8B-Instruct lands at **AUROC ≈ 0.675**, while the ICR paper (Zhang et al., ACL 2025; [arXiv:2507.16488](https://arxiv.org/abs/2507.16488)) reports **0.7982** for Llama-3. Side-by-side on the same data:

| Method | Paper (Llama-3) | Ours (substring, Llama-3.1-Instruct) | Gap |
|---|---|---|---|
| LN-Entropy | 0.6593 | 0.6483 | −0.011 |
| PPL | 0.6721 | 0.6143 | −0.058 |
| SAPLMA | 0.7701 | 0.6935 | −0.077 |
| ICR Probe | 0.7982 | ~0.675 | **−0.123** |

Unsupervised entropy methods match the paper closely; supervised probes drop 8–12 points. The most plausible methodology difference is **labeling**: we use substring match (the LLMsKnow / probing-literature standard), whereas the ICR paper consumes an `output_judge.jsonl` file with a `result_type ∈ {0,1}` field produced by an LLM judge.

This ablation re-labels HotpotQA with an LLM judge and re-runs every method, so we can defensibly answer "did you implement ICR correctly?" with yes.

## Critical finding: the ICR paper does not specify its judge protocol

Investigation of [the ICR release](https://github.com/XavierZhang2002/ICR_Probe) shows the repo contains only 4 source files (`icr_score.py`, `icr_probe.py`, `utils.py`, `config.py`) plus one notebook. **No `judge.py`, no prompts, no `output_judge.jsonl` sample, no OpenAI/grader calls anywhere.** The notebook only *consumes* labels via:

```python
check_result_path = file_path + '/output_judge.jsonl'
check_result.append({'id': obj['id'], 'result': obj['result_type']})
```

Paper Appendix B.1 — the entire labeling-relevant text — says only:

> "The label for each instance is derived from annotated hallucination datasets."

No judge model, no prompt, no decoding settings, no `result_type` derivation rule is documented. **There is nothing to reproduce verbatim.** This is itself a defensible reviewer point: we cannot reproduce what was not released.

### Strongest indirect clue: HaluEval framework reuse

The notebook's directory naming uses suffixes `_KQA` (KnowQA) and `rc.nocontext` that match the [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al. EMNLP 2023) evaluation conventions. Strongly suggests the ICR authors reused HaluEval's QA grading pipeline. **We adopt HaluEval's published judge protocol as our surrogate**, documented as such.

## The judge protocol we will use (HaluEval QA, verbatim)

Source: `RUCAIBox/HaluEval/evaluation/evaluate.py` + `evaluation/qa/qa_evaluation_instruction.txt`.

**Model.** HaluEval originally uses `gpt-3.5-turbo` (unpinned 2023 build). We cannot reproduce that exactly. We will run **two judge configurations**:
1. **Primary (open-weight, reproducible): Llama-3.3-70B-Instruct via vLLM on Empire AI.** Fully self-hosted, deterministic at temperature 0, no API dependency. This is the production result for our ablation table.
2. **Optional (closed-API, fidelity): `gpt-4o-mini`** (cheap, ~$5 for 85K calls). Only if user approves OpenAI spend. Goal: confirm open-weight judge is consistent with closed-API judge.

**Prompt template.** System message (verbatim):
```
You are a huallucination detector. You MUST determine if the provided answer
contains hallucination or not for the question based on the world knowledge.
The answer you provided MUST be "Yes" or "No"
```

User message (verbatim format):
```
{instruction}

#Question#: {question}
#Answer#: {generation}
#Your Judgement#:
```

`{instruction}` is the verbatim contents of `qa_evaluation_instruction.txt` — the HaluEval few-shot prompt with 7 in-context Q/A/judgement examples covering four hallucination types. We commit the file as `data/halueval_qa_judge_prompt.txt` to the repo.

**Decoding.** `temperature=0.0`, `max_tokens=8` (HaluEval leaves it unset; we cap for cost/latency since the response is just "Yes"/"No").

**Output parsing** (verbatim HaluEval logic):
```python
ans = response.replace(".", "")
if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
    label = None     # "failed" — drop sample
elif "Yes" in ans:
    label = 1        # hallucinated
elif "No" in ans:
    label = 0        # faithful
```

**Inputs shown to judge.** Question + generation only. **No gold answer, no context, no supporting facts.** This is reference-free — the judge uses parametric knowledge. This matches HaluEval verbatim. Caveat: ICR may have modified this to inject the gold answer, but we cannot know. We treat reference-free as the principal protocol and run a **reference-conditioned variant** as a robustness check (see §Robustness below).

## Scope

### In scope
- **HotpotQA only.** That is the cell with the published 0.7982 comparison point. The paper also evaluates on HaluEval (which ships with native pos/neg labels — no judge needed for that dataset specifically), SQuAD, and TriviaQA — but we do not pipeline SQuAD/TriviaQA today, and the reviewer-defense argument only needs one cell to land.
- **Llama-3.1-8B-Instruct only.** That is the model with the comparison point.
- **All methods we already have results for, re-evaluated on LLM-judge labels.** Specifically:
  - Re-train supervised probes: `icr_probe`, `saplma`, `linear_probe`, `llmsknow_probe`, `contrastive`, `contrastive_logprob_recon`, `saplma_logprob_recon` (5 seeds each).
  - Re-score unsupervised baselines (no retrain): `logprob_baseline` (PPL family), `se_*`, `selfcheck_*`, `p_true`, `token_entropy`.

### Out of scope
- Other datasets. Single-cell ablation is sufficient.
- Other models (Qwen3-8B). Same reason.
- Changing the primary labeling protocol. Substring stays primary in main tables; LLM-judge is an ablation column.
- Authors-email path (parallel track, see §Parallel).

## Implementation plan

### 1. Vendor the judge prompt

- `data/halueval_qa_judge_prompt.txt` — verbatim from HaluEval repo. Add a header comment citing source URL + commit hash.
- `data/halueval_qa_judge_system.txt` — the system message above.

### 2. Build the judge driver

`scripts/llm_judge_label.py` (new):
- Input: existing generation JSONL (`output/hotpotqa[_train]/Llama-3.1-8B-Instruct/generation.jsonl`).
- Spin up Llama-3.3-70B-Instruct via vLLM (or use a long-running server if already up).
- Apply the HaluEval prompt verbatim per row, with `temperature=0`, `max_tokens=8`.
- Parse with HaluEval's substring rule. Drop "failed" rows. Report failure rate.
- Output: `output/hotpotqa[_train]/Llama-3.1-8B-Instruct/eval_results_llm_judge.json` + `raw_eval_res_llm_judge.jsonl`, mirroring the existing substring-eval schema so downstream training/eval just point to the new file.

Sample count: 7,405 (test) + ~62K (train subset used for current ICR run) ≈ 70K calls. On H200 with Llama-3.3-70B at bs=64, ~3–5 hours.

### 3. Sanity checks (mandatory gate before any retraining)

1. **Failure rate < 2%.** If higher, prompt is broken; fix before proceeding.
2. **Substring vs LLM-judge agreement** on the test set. Expected 80–90%. Report Cohen's κ.
3. **Label class balance** under LLM-judge. Substring is 67% hallucinated; expect LLM-judge to be 50–65%.
4. **Manual spot-check of 50 disagreements** — confirm the judge is doing something sensible, not flipping arbitrarily. If the judge looks broken, fix the prompt and rerun §2.

Document all four checks in `notes/llm_judge_sanity.md` before §4.

### 4. Re-train supervised methods on LLM-judge labels

Add a `--labels-source {substring,llm_judge}` switch to the trainer (or stand up a parallel config `configs/experiments/baseline_comparison_hotpotqa_llmjudge.json`). Re-run:

| Method | Seeds | Approx wall-clock |
|---|---|---|
| icr_probe | 5 | 1.5 h |
| saplma | 5 | 1 h |
| linear_probe | 5 | 0.5 h |
| llmsknow_probe | 5 | 0.5 h |
| contrastive | 5 | 2 h |
| contrastive_logprob_recon | 5 | 2 h |
| saplma_logprob_recon | 5 | 1 h |

Total: ~8.5 h serial, parallelizable across GPUs.

### 5. Re-score unsupervised baselines (no retrain)

For each of `logprob_baseline`, `se_*`, `selfcheck_*`, `p_true`, `token_entropy`: re-compute AUROC using the new labels. This is a 5-minute pandas operation per method.

### 6. Robustness: reference-conditioned variant

Re-run §2 with the prompt modified to include the gold answer, so the judge does answer-equivalence rather than reference-free judgement. Specifically inject before the question:
```
#Reference Answer#: {gold_answer}
```
Re-train ICR Probe only on the reference-conditioned labels. **Purpose:** if ICR Probe under reference-free judge ≠ 0.798 but under reference-conditioned judge = 0.798, that strongly suggests the authors used a gold-conditioned variant. If neither matches, the gap is not labeling.

### 7. Produce the ablation table

| Method | Paper (Llama-3) | Ours, substring | Ours, LLM-judge (ref-free) | Ours, LLM-judge (ref-conditioned) |
|---|---|---|---|---|
| LN-Entropy | 0.6593 | 0.6483 | ? | ? |
| PPL | 0.6721 | 0.6143 | ? | ? |
| SAPLMA | 0.7701 | 0.6935 | ? | ? |
| ICR Probe | 0.7982 | ~0.675 | ? *(target: ≥ 0.778 under one of the two)* | ? |
| contrastive (kNN) | — | 0.8525 | ? *(target: ≥ 0.832, stability check)* | ? |
| linear_probe | — | 0.8218 | ? | ? |
| llmsknow_probe | — | 0.8259 | ? | ? |

Plus the substring-vs-LLM-judge agreement / class-balance / failure-rate stats from §3.

## Success criteria

1. **ICR Probe under at least one LLM-judge variant reproduces the paper to within ±0.02 AUROC** (i.e., ≥ 0.778). Confirms labeling hypothesis.
2. **Our methods (contrastive, linear, llmsknow) are stable to within ±0.02 across labeling regimes.** Pre-empts the "you cherry-picked substring" reviewer attack.
3. **A footnote in the paper** stating: (a) substring is the dominant standard in the short-form QA probing literature (LLMsKnow, P(True), Farquhar et al.); (b) ICR Probe is an outlier in adopting LLM-judge; (c) we reproduce ICR Probe under both regimes for transparency; (d) the ICR paper does not specify its judge protocol and the released code omits the judging script, so we adopt HaluEval's published QA judge as the surrogate, justified by the `_KQA` directory-naming convention in the released notebook.

## Failure mode and recourse

If ICR Probe under both LLM-judge variants still does not reach 0.78:
- Try **Llama-3-8B-Instruct (base 3.0)** instead of 3.1 — the paper says "Llama-3," which most likely means 3.0.
- Try **fixed 80/20 split** instead of 5-fold CV (the paper's protocol).
- Re-audit ICR score computation against the paper §3 and Issue #70's spec.
- Email the authors (parallel track below) — by this point we have data showing where exactly the gap persists, which makes the email focused.

Document the failure path in the same appendix; do not silently drop the ablation.

## Parallel: email the authors

Low-cost: send Zhenliang Zhang + Xiaojun Wan (Peking U.) an email asking for the `output_judge.jsonl` files or judge script. Do not gate this ablation on a reply — but if a reply comes, we can swap our surrogate labels for the authors' actual labels and re-run §4–§7 cheaply.

## Dependencies

- Llama-3.3-70B-Instruct weights accessible on Empire AI (check before starting).
- vLLM serving infra (already in place via `scripts/run_with_server.py`).
- Existing trainer / eval pipelines (already in place via Issue #70 and prior baselines).

## Estimate

| Stage | Wall-clock |
|---|---|
| Vendor prompts + judge driver | 0.5 day |
| Judge inference run (ref-free + ref-conditioned, 70K × 2) | 0.5 day |
| Sanity checks + spot review | 0.5 day |
| Retrain 7 supervised methods × 5 seeds × 2 label sets | 1–2 days |
| Re-score unsupervised + table + writeup | 0.5 day |

**Total: ~3–4 working days, gated on GPU availability.**
