# Probability-Based (Logit/Logprob) Detection — Implementation Plan

This plan adds **probability-based hallucination detection** support (logprobs/uncertainty signals) to HalluLens, complementing the existing **activation-based** logging and probes.

## Why this matters (paper goal)

The primary research goal is to **compare our contrastive representation learning approach** (activation/representation-based) against **SOTA hallucination detection** methods.

Most competitive “intrinsic hallucination detection” baselines in the literature fall into one of two buckets:

1) **Probability / logit uncertainty** (logprobs, entropy, semantic entropy, self-consistency)
2) **Representation / activation signals** (layer-wise probes, geometry/dynamics)

This plan closes the gap for bucket (1) so we can report **fair, apples-to-apples** comparisons alongside our contrastive-representation detectors.

The north star is: for each benchmark sample, we can attribute an uncertainty score to the generated answer and compute metrics like AUROC/AUPRC for hallucination detection, while preserving compatibility with the current OpenAI-protocol inference flow.

## Scope

### In scope

- Capture **token-level log probabilities** (and optionally top-k alternatives) for generated outputs.
- Persist these signals on disk (LMDB and/or JSON logger) alongside existing prompt/response + activation metadata.
- Compute baseline probability/uncertainty detectors:
  - single-pass (no multi-sampling)
  - multi-sample (semantic uncertainty / self-consistency) as an optional extension.
- Keep compatibility with how inference is invoked today via `utils/lm.py` (OpenAI client calling `/v1/chat/completions`).

### Non-goals

- Building a full retrieval/verifier pipeline.
- Claiming SOTA numbers without controlling for access assumptions.
- Editing Meta-copyrighted files (e.g., `utils/lm.py`) unless explicitly requested.

## Current state (gap analysis)

- Activations: already logged via `activation_logging/activations_logger.py` (LMDB/JSON).
- Server: `activation_logging/server.py` runs generation via HuggingFace Transformers and already requests `return_dict_in_generate=True` and `output_hidden_states=True`.
- **Gap**: probability signals are not captured because generation does not enable score outputs (e.g., `output_scores=True`), and no logprob fields are persisted.

## Design principles

- **Do not require client changes**: probability signals should be computed + stored server-side.
- **Minimal logging first**: store only what is needed for robust baselines (token logprobs + alignment info).
- **Stable join keys**: keep a clean link between benchmark item → prompt → response → logprobs/activations → eval label.
- **Comparable evaluation**: store enough metadata to ensure apples-to-apples comparisons (sampling params, temperature, top_p, stop, max_tokens).

## Fair comparison protocol (contrastive reps vs SOTA)

To make the comparison publishable, run both our method and uncertainty baselines under matched conditions:

- Same base model checkpoint (e.g., Mistral-7B-Instruct vs Llama-3.1-8B-Instruct).
- Same prompt formatting and decoding constraints (temperature/top_p/max_tokens/stop).
- Same dataset split and label definition (what counts as hallucinated).
- Same evaluation metric reporting (AUROC/AUPRC + confidence intervals across seeds).
- Separate leaderboards by access assumptions:
  - **Single-pass** (one generation per prompt)
  - **Multi-sample** (K generations per prompt; semantic entropy / self-consistency)
  - **White-box** (activations available) vs **black-box** (text-only)

## Phase 0 — Spec & acceptance criteria

### Acceptance criteria

- For a single request, an on-disk record contains:
  - `prompt`, `response`, `model`
  - `generation_params` (at least `temperature`, `top_p`, `max_tokens`)
  - `generated_token_ids` and `token_logprobs` aligned 1:1
- For a benchmark run (e.g., `tasks/refusal_test/nonsense_mixed_entities.py`):
  - every sample can be mapped to a probability score
  - an evaluation script can compute AUROC/AUPRC for at least 1 baseline uncertainty detector.

## Phase 1 — Add server-side logprob capture (single-sample)

### 1.1 Transformers path (HuggingFace) implementation

Target file(s):
- `activation_logging/server.py`

Changes:
- In `model.generate(...)` enable `output_scores=True` (keep `return_dict_in_generate=True`).
- Compute token-level logprobs for generated tokens.
  - Preferred: `model.compute_transition_scores(...)` if available for the model class.
  - Fallback: for each step score tensor in `outputs.scores`, compute `log_softmax` and pick the generated token id.

Data to compute/store (minimal viable):
- `generated_token_ids: List[int]`
- `token_logprobs: List[float]` (length == number of generated tokens)
- `prompt_token_count: int` and `completion_token_count: int`

Optional (phase 1.5):
- `top_logprobs: List[Dict[int, float]]` per token (top-k alternatives), if we want entropy/margin baselines without storing full logits.

### 1.2 OpenAI-protocol response compatibility

We do **not** need to return logprobs to the client for experiments to work (we can store them on disk only).

Optionally, later:
- support `logprobs` in `/v1/completions` and `/v1/chat/completions` response objects to match OpenAI schema more closely.

## Phase 2 — Extend logging schema (LMDB + JSON)

Target file(s):
- `activation_logging/activations_logger.py`
- `activation_logging/zarr_activations_logger.py` (if used)

Add fields to each stored entry:
- `logprobs` (object)
  - `generated_token_ids`
  - `token_logprobs`
  - `prompt_token_count`
  - `completion_token_count`
  - `avg_token_logprob` (derived, optional)
- `generation_params` (object)
  - `temperature`, `top_p`, `max_tokens`, `stop`

Add/ensure stable IDs:
- `request_id` (UUID) and/or keep existing `prompt_hash`
- `timestamp`
- `sample_idx` (default 0)
- `group_id` (nullable; used for multi-sample methods)

### Compatibility note

Do not break existing readers:
- New fields must be optional.
- Existing activation-only flows should still parse.

## Phase 3 — Parser + dataset alignment

Target file(s):
- `activation_logging/activation_parser.py`

Changes:
- Load logprob payload from LMDB/JSON and expose it in the dataset items.
- Provide helper accessors:
  - `get_token_logprobs(key)`
  - `get_sequence_logprob(key)`

Acceptance checks:
- A small script can iterate a dataset and compute a probability score for each item without touching activations.

## Phase 4 — Implement baseline probability detectors

Create a small baseline module (new file, no Meta code edits), e.g.:
- `activation_research/probability_baselines.py`

Baselines to implement (single-sample):
- **Mean token logprob**: $\frac{1}{T}\sum_t \log p(x_t)$
- **Min token logprob**: $\min_t \log p(x_t)$
- **Sequence logprob**: $\sum_t \log p(x_t)$ (note: length-sensitive)

Optional (if top-k captured):
- token-level entropy
- margin between top-1 and top-2

Outputs:
- Per-example score file (CSV/JSONL) with fields:
  - `sample_id` / `prompt_hash`
  - `score_name`
  - `score_value`
  - `label` (hallucinated vs not)

## Phase 5 — Multi-sample uncertainty (semantic entropy / self-consistency)

Goal: support uncertainty methods that require multiple generations per prompt.

### 5.1 Multi-sample runner

Create a new script (do not edit `utils/lm.py`), e.g.:
- `scripts/run_uncertainty_sampling.py`

Behavior:
- For each prompt, generate `K` samples (temperature > 0) via the existing OpenAI-protocol endpoint.
- Store each sample as its own LMDB/JSON entry with a shared `group_id`.
- Persist a `groups.jsonl` mapping:
  - `group_id`, `prompt_hash`, `sample_keys[]`, `temperature`, `top_p`, `K`

### 5.2 Aggregation & semantic uncertainty

Create an aggregator, e.g.:
- `activation_research/semantic_uncertainty.py`

Start simple:
- Self-consistency disagreement measures (string-level; later semantic clustering).

Optional later:
- semantic clustering using embeddings (requires pinning deps and defining model).

## Phase 6 — Wire into benchmark + eval

Target:
- Ensure `tasks/refusal_test/nonsense_mixed_entities.py` runs end-to-end while producing:
  - logged logprobs on disk
  - an evaluation artifact with uncertainty baseline AUROC/AUPRC

Implementation approach:
- Prefer adding a separate evaluator script that reads the inference/eval outputs + LMDB/JSON logs.
- Avoid modifying Meta-owned utilities.

## Phase 7 — Tests + docs

Tests (minimal):
- Unit-ish test: length alignment between `generated_token_ids` and `token_logprobs`.
- Regression: activation logging still works with logprob capture enabled.

Docs:
- Update `SOTA_TRACKER.md` with a clear statement that we now support the “logit uncertainty” bucket.
- Add a short how-to section:
  - “Run server”
  - “Run benchmark”
  - “Compute uncertainty baselines”

## Risks & mitigations

- **Memory/latency overhead** from storing extra per-token data.
  - Mitigation: store only `token_logprobs` first; gate top-k logging behind a flag.
- **Schema drift** between LMDB/JSON.
  - Mitigation: define a single canonical `logprobs` object and reuse it.
- **OpenAI schema mismatch** (if we later return logprobs).
  - Mitigation: store on disk first; only add API fields once stable.

## Work breakdown (engineering checklist)

- [ ] Add `output_scores=True` and compute token logprobs in `activation_logging/server.py`.
- [ ] Extend `ActivationsLogger` / `JsonActivationsLogger` entry format to include `logprobs`.
- [ ] Extend `ActivationParser` to surface logprob payload.
- [ ] Implement `activation_research/probability_baselines.py` and produce per-example scores.
- [ ] Add optional multi-sample runner + group_id scheme.
- [ ] Validate on `nonsense_mixed_entities.py` end-to-end.
- [ ] Add a small regression test and brief docs.
