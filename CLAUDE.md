# HalluLens — Claude Code Guidelines

## Paper Writing

The EMNLP 2026 paper draft lives in `paper/`. **When writing the paper, only reference files inside `paper/`, unless the user explicitly names a source outside it.** This applies to drafting, editing, or extending sections, outline, abstract, claims, tables, figures, and bibliography — and covers reading, citing, comparing against, or otherwise consulting external files, not just copying prose from them.

Concretely: do not read `PAPER_ROADMAP.md`, `THEORETICAL_JUSTIFICATION.md`, `docs/planning/`, `results/`, the SOTA tracker, `README.md`, or any other location outside `paper/` when working on paper content. If a fact, number, or framing from outside `paper/` needs to land in the draft, the user will name the source explicitly.

The single source of truth for numbers cited in the paper is `paper/data/*.csv` and `paper/generated/figures/*.numbers.csv`, accessed through the `\result`/`\resdelta`/`\resratio`/`\resultCI`/`\resultPM` macros defined in `paper/macros.tex`. See `paper/README.md` for the build pipeline.

## Project Overview

Research codebase for detecting hallucinations in LLMs via mutual information analysis of intermediate layer activations. Pairs contrastive representation learning with benchmark evaluation (PreciseWikiQA, LLMsKnow).

Paper: [arXiv:2504.17550](https://arxiv.org/abs/2504.17550)
License: CC-BY-NC

## Environment

- Python 3.12 (conda env: `hallulens`)
- Install: `pip install -r requirements.txt`
- Key deps: PyTorch, vLLM, Transformers, FastAPI, LMDB, Zarr

## Supported Models

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | Primary baseline |
| Qwen3 8B | `Qwen/Qwen3-8B` | Thinking mode disabled automatically |

Both models work with all LLMsKnow tasks via the default batch inference path.

## Key Commands

### Inference — capture activations + attentions (icr_capture format)

`scripts/capture_inference.py` runs the model with `output_attentions=True` + `output_hidden_states=True` and writes a memmap `icr_capture` directory (activations, attentions, logprobs, labels). This is the data source for all downstream training.

```bash
python scripts/capture_inference.py \
    --task hotpotqa \
    --split validation \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-dir shared/icr_capture/hotpotqa_Llama-3.1-8B-Instruct \
    --max-prompt-len 512 --max-response-len 64 --r-max 64 --top-k 20
```

Tasks: `hotpotqa`, `mmlu`, `natural_questions`, `popqa`, `sciq`, `searchqa` (`movies` excluded — no train split).
Swap `--model Qwen/Qwen3-8B` as needed. Add `--n-samples 100` for a smoketest.

### Training — config-driven experiment runner

`scripts/run_experiment.py` loads an experiment config and trains + evaluates across seeds. Dataset configs point to `icr_capture` dirs.

```bash
python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa_memmap.json

# Single run without experiment config
python scripts/run_experiment.py \
    --dataset configs/datasets/hotpotqa_memmap.json \
    --method configs/methods/contrastive.json \
    --seed 42

# Quick smoke (1 epoch)
python scripts/run_experiment.py \
    --experiment configs/experiments/baseline_comparison_hotpotqa_memmap.json \
    --max-epochs 1
```

### Data setup
```bash
bash scripts/download_data.sh
```

## Storage Formats

| Format | Use case | Notes |
|--------|----------|-------|
| Zarr   | Large-scale experiments (preferred) | Best compression |
| JSON+NPY | Medium scale, human-readable metadata | ~5-10GB per 100 inferences |
| LMDB   | Maximum storage efficiency | |

## Where Are You Running?

Before running any code, determine the execution context:

**Are you on the Empire AI login node?**
- The login node (`alpha1.empire-ai.org`) is for orchestration only — `git`, `gh`, `gpu_dispatch.py`, file inspection.
- **Never run CPU- or GPU-intensive work there** (no training, no inference, no pytest with real models).
- For quick verification, use `utils/jupyter_exec.py` against an already-allocated GPU node.
- For batch jobs, use `gpu_dispatch.py run` after explicit user approval.

**Are you on a local machine (not on Empire AI)?**
- No GPU is available locally — do not attempt to launch vLLM servers or run training.
- To run GPU work, go through the Empire AI cluster: use `utils/empire_shell.py` for login-node orchestration and `gpu_dispatch.py` to dispatch jobs to GPU nodes.

### MANDATORY: access Empire AI via `utils/empire_shell.py`, never bare ssh

**Do not run `ssh empire-ai 'cmd'` directly.** Every bare SSH call costs ~60s of login overhead and accumulates orphan processes that exhaust `TasksMax=512`, making the node unusable. All shell commands to the Empire AI login node — `git`, `gh`, `squeue`, `gpu_dispatch.py`, file inspection — must go through `utils/empire_shell.py`, which keeps a single warm session:

```bash
python utils/empire_shell.py 'squeue --me'
python utils/empire_shell.py 'gh pr list'
```

Or programmatically:

```python
from utils.empire_shell import run
res = run('squeue --me')
print(res.stdout, res.exit_code)
```

Daemon control: `--status`, `--kill`, `--logs`.

### Claude Code GPU execution (REMOTE_GPU only)

Use `utils/jupyter_exec.py` to run code on the GPU node without user intervention:

```bash
python utils/jupyter_exec.py "import torch; print(torch.cuda.get_device_name(0))"
```

```python
from utils.jupyter_exec import JupyterExecutor

with JupyterExecutor() as jup:
    result = jup.run("import torch; print(torch.cuda.is_available())")
    print(result.stdout)   # stdout text
    print(result.status)   # "ok" | "error" | "timeout"
```

Prefer this over asking the user to run notebook cells manually. Requires `websocket-client`.

## Dataset & Experiment Status

### Canonical results view across all baselines

`scripts/results_table.py` is the **single source of truth** for what experiments have produced numbers. Prefer it over `audit_coverage.py` for any "what do we have" question.

```bash
python scripts/results_table.py                        # → output/results_table/results_table.{json,csv}
python scripts/results_table.py --out-dir /tmp/results
```

### Checking inference/data generation status

**Always run the audit script rather than guessing whether data is present.**

```bash
python scripts/audit_datasets.py --model Qwen/Qwen3-8B
python scripts/audit_datasets.py --model meta-llama/Llama-3.1-8B-Instruct
python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr        # check zarr sample counts
python scripts/audit_datasets.py --model Qwen/Qwen3-8B --zarr-size   # check zarr disk usage
```

Data paths: `output/{dataset}[_train]/{model_name}/generation.jsonl`, `eval_results.json`, `shared/{dataset}[_train]_{model_slug}/activations.zarr/`. Model name = last HuggingFace ID component (e.g. `Qwen3-8B`).

### Checking experiment/training status

```bash
python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_mmlu.json
python scripts/experiment_status.py --experiment configs/experiments/baseline_comparison_mmlu.json --json
```

### Dataset configs convention

- `configs/datasets/{name}_test.json` — test split
- `configs/datasets/{name}_train.json` — train split
- `configs/experiments/baseline_comparison_{name}.json` — experiment config

### Running on GPU nodes

**Never submit or kill SLURM jobs without explicit user approval.** Do not run `sbatch`, `srun`, `scancel`, `gpu_dispatch.py run/kill`, or kill remote processes without asking first:
> "I want to [specific action, e.g. kill PID 12345 on alphagpu22]. Yes/No?"

**Avoid duplicate dispatches** — a timed-out dispatch may have actually launched. Before re-dispatching, check `gpu_dispatch.py jobs --all` and wait at least 2 minutes.

Use `scripts/gpu_dispatch.py` for all GPU job dispatch. **Always run `sync-jupyter` first** — it queries `squeue --me` live and refreshes the node list; the static `configs/nodes.json` goes stale as SLURM allocations come and go.

```bash
python scripts/gpu_dispatch.py sync-jupyter                        # refresh node list from squeue (do this first)
python scripts/gpu_dispatch.py status                              # node availability + GPU util
python scripts/gpu_dispatch.py run bash run_training.sh            # best available node
python scripts/gpu_dispatch.py run --node alphagpu04 bash run_training.sh
python scripts/gpu_dispatch.py run --min-vram 30 bash run_training.sh
python scripts/gpu_dispatch.py jobs --all
python scripts/gpu_dispatch.py kill JOB_ID
```

Job manifest: `shared/gpu_jobs.json`. Always use `resume=True` for long jobs.

