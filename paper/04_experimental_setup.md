# Experimental Setup — Outline

Structural skeleton **plus** fully materialized content. By the time a writing agent picks this up, every fact, config pointer, baseline, and number-source that will appear in §4 is named here, with the precise spot our prose draws from. The writing agent should not need to research further; if a claim cannot be supported from the bullets below, surface it to the human rather than inventing one.

**Bib-policy reminder.** Bibkeys in the "References — materialized for §4" section below are already in `paper/references.bib`. Anything marked `[PENDING-APPROVAL]` in the bullets — every **dataset** citation and both **model** citations — is **not** yet in `.bib` and must be human-approved before insertion (per `paper/references.bib` policy header and `CLAUDE.md`). §4 is unusually citation-heavy in the *dataset* slot (six datasets, two models) and every one of those is pending; the writing agent must hedge wording until they land. The candidate entries with everything a human needs to verify are in the "Pending-approval candidates" section at the bottom.

**Structural decision (target length: ~0.5–0.75 page).** This is the smallest budget in the paper. §4 enumerates, it does not argue. The methodological *why* lives in §3; §4 says *what we ran*. Concretely: do not re-state the method, do not re-state the theoretical motivation, do not pre-empt §5 results. Forward-references to §3.3 (training recipe, scorer choice, hyperparameters) and Appendices B / D / E (full hyperparams, dataset details, compute breakdown) carry most of the detail.

**Supersedes** nothing — there is no `experimental_setup_outline.md` predecessor; this is the first materialization of §4.

**Numbers-from-CSV policy.** Per `paper/CLAUDE.md` / `paper/README.md`, every numeric value in the final prose must come from `paper/data/*.csv` (or `paper/generated/figures/*.numbers.csv`) via the `\result` / `\resultCI` / `\resultPM` macros in `macros.tex`. As of 2026-05-19 the only file is `paper/data/baseline_comparison.csv` and it is a placeholder; do not lock §4 prose to specific train/test-split sizes or class-imbalance ratios until the per-dataset CSV(s) ship. Until then, structural facts (which datasets, which models, which baselines, which seeds, which metrics) are safe; per-dataset *sizes* go in Appendix D and are sourced via macros.

---

## 4. Experimental Setup

### 4.1 Models

**Framing sentence.** Two 8B-scale open-weights instruction-tuned models from distinct training pipelines. We choose 8B-scale to keep the full grid (six datasets × multiple baselines × five seeds) tractable on a single 80 GB GPU per cell once activations are cached, and two distinct lineages to foreclose the reviewer concern that results are an artifact of one training recipe.

**Bullets — write as 1–2 sentences in prose:**

- **`meta-llama/Llama-3.1-8B-Instruct`** [PENDING-APPROVAL — Grattafiori et al. 2024, "The Llama 3 Herd of Models," arXiv:2407.21783]. Primary baseline; 32-layer transformer, hidden size 4096, instruction-tuned via SFT + DPO. Used in every cell of the main grid. Activations were cached once per (dataset, split) and re-used across all training methods and seeds (see §3.3 implementation summary and Appendix E).
- **`Qwen/Qwen3-8B`** [PENDING-APPROVAL — Yang et al. 2025, "Qwen3 Technical Report" — verify exact citation before insertion]. Second model family; 32-layer transformer, hidden size 4096, separate pretraining + post-training pipeline from Llama. Thinking mode is disabled (single-pass non-CoT inference) for parity with Llama. Used in every cell of the main grid; per the §2 status table in [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md), 111/132 Qwen cells were complete and 21 outstanding as of 2026-05-15 — the writing agent must check `python scripts/results_table.py` before locking §5 cells but §4 only needs the structural statement.
- **What we did not run.** No 70B-scale models; no closed/API-only models; no third 8B model (SmolLM3 configs exist but are scoped out per [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §5). State this in one sentence and forward to §9 limitations rather than defending here.
- **Why this pair, in 1 sentence of prose.** "We evaluate on two 8B-scale open-weights instruction-tuned models from distinct training pipelines — Llama-3.1-8B-Instruct and Qwen3-8B — to test whether the signal is model-family-specific (an *a priori* live concern) without inflating compute to a third lineage."

**What §4.1 establishes for the rest of the paper.** The two-model frame that §5 main results, §6 transfer, and §7 ablations all live inside. Reader exits §4.1 knowing exactly *what* the model axis of the grid is.

---

### 4.2 Datasets

**Framing sentence.** Six short-form QA / reasoning benchmarks spanning open-domain factoid (NQ, PopQA, HotpotQA, SearchQA), science multiple-choice (SciQ), and broad multi-domain multiple-choice (MMLU). One per-dataset row table in Appendix D carries train/test sizes, evaluator details, and class-imbalance ratios.

**Bullets — write as 1–2 sentences each:**

- **HotpotQA** [PENDING-APPROVAL — Yang et al. 2018, EMNLP, arXiv:1809.09600]. Multi-hop open-domain QA over Wikipedia passages; we use the *distractor* setting train and dev splits as our train/test pair (the test set has hidden answers, as is standard in the probing literature). Substring-match evaluator on the gold answer. Largest cell in the grid by activation-cache footprint; the load-bearing dataset for §7.1 / §7.3 ablations because of seed-0 evidence quality.
- **Natural Questions (NQ)** [PENDING-APPROVAL — Kwiatkowski et al. 2019, TACL]. Open-domain QA from real Google queries with Wikipedia-passage gold answers; we use short-answer-only items. Substring-match evaluator.
- **MMLU** [PENDING-APPROVAL — Hendrycks et al. 2021, ICLR, arXiv:2009.03300]. Multi-domain multiple-choice (57 subjects, four options). Exact-match evaluator on the selected option letter. Included to test whether the signal generalizes from open-ended QA to multi-choice; also the dataset we *exclude* from the sampling-based baselines bundle (NLI clustering degenerates on single-letter answer tokens, per [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §6).
- **PopQA** [PENDING-APPROVAL — Mallen et al. 2023, ACL]. Entity-popularity-stratified factoid QA; spans long-tail entities where retrieval-free LLMs are expected to hallucinate more. Substring-match evaluator. Important for the §5 finding because the *class-imbalance ratio* is materially different from HotpotQA/NQ (more hallucinations).
- **SciQ** [PENDING-APPROVAL — Welbl et al. 2017, W-NUT, arXiv:1707.06209]. Science multiple-choice QA with crowd-authored distractors. Substring-match on the correct option. Cleanest signal in the grid (least noisy gold answers) — useful as a baseline-quality calibration point.
- **SearchQA** [PENDING-APPROVAL — Dunn et al. 2017, arXiv:1704.05179]. Jeopardy! questions paired with web-search snippets; we use questions only, no snippets — strictly closed-book to match the other datasets. Substring-match evaluator. Largest raw split; test capped at 10k items per `PAPER_ROADMAP.md` §6 to bound the sampling-baselines bundle. (The cap applies to §5.3 sampling cells only; the trained-method cells use the full available split — verify with `audit_datasets.py` if locking exact numbers.)
- **Labeling convention.** Substring-match against the gold answer is the binary hallucination label across all six datasets, in line with the dominant standard in the short-form-QA probing literature (`kadavath2022language`, `farquhar2024semantic`, ICR Probe paper). The writing agent must state this *explicitly* — the §4 item 10 methodology footnote in [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) is the longer-form defense; §4 itself owes one sentence and a forward-pointer to that footnote. **Important for the ICR Probe comparison:** our reproduction lands at ~0.675 on Llama-3 vs the paper's 0.7982, and the discrepancy is attributable to labeling (the ICR paper uses an undocumented LLM-judge pipeline whose code is not released — see §4 item 10 footnote). §4 must surface this in one sentence and not at length; the longer treatment lives in [`02_related_work.md`](02_related_work.md) §2.1 ICR Probe bullet and the methodology footnote.
- **Per-dataset table in Appendix D.** Train size, test size, class-imbalance (% hallucinated), evaluator, license. Sourced via `\result{...}` macros from `paper/data/datasets.csv` (file does not yet exist; needs to be generated by `paper/build_numbers.py` — flag for the writing agent / human).

**What §4.2 establishes for the rest of the paper.** The dataset axis of the main grid (§5), the transfer matrix (§6), and the layer-pair / loss-decomposition ablations (§7.1, §7.3). Reader exits §4.2 knowing exactly *what* the six datasets are and *why* MMLU is the one cell missing from the §5.3 sampling-baselines comparison.

---

### 4.3 Baselines

**Framing sentence.** Three classes of baseline, each foreclosing a distinct reviewer concern. Each class is named with the specific implementation we run (config pointer + entrypoint), not just the paper. **Compute axis is load-bearing** for the §5.3 framing: the K=1 cluster shares forward-pass count with our method; the K=10 cluster does not.

**Three subsections (bullets per class — each as 1–2 sentences in prose):**

#### Output-space scalar baselines (K=1; single forward pass; activation-free)

- **Per-token logprob** (`kadavath2022language`). Mean / sequence / perplexity scorers over the response-token logprobs from the *same* greedy generation pass used by our method. Config: [`configs/methods/logprob_baseline.json`](../configs/methods/logprob_baseline.json). Why include: the cheapest possible signal — establishes the *floor*. Deterministic given fixed inference.
- **Token entropy** (`kadavath2022language`). Mean / minimum / next-token entropy over the response tokens. Config: [`configs/methods/token_entropy.json`](../configs/methods/token_entropy.json). Deterministic; same forward pass.
- **P(true)** (`kadavath2022language`). One templated follow-up prompt per generation ("Is the above answer correct? Yes/No"); we read the token probability of "Yes". Cost: one extra short forward pass per example. Implementation: [`tasks/p_true/`](../tasks/p_true/). Forecloses the reviewer concern that the paper omits prompt-based self-evaluation.

#### Activation-space probe baselines (K=1; single forward pass; reads cached activations)

- **Single-layer linear probe.** Logistic regression on the mean-pooled last-token activation at layer 22 (verified against [`configs/methods/linear_probe.json`](../configs/methods/linear_probe.json) — `probe_layer: 22`, `pooling: mean`). This is the canonical read-out floor (`alain2017understanding`); §4 must name this baseline explicitly because the §5 narrative leans on "ours beats the linear probe."
- **LLMsKnow probe** [PENDING-APPROVAL — citation needed; see pending-approval candidates]. Data-adaptive location sweep: sweeps all `(layer, token_position)` pairs in the extraction band `[14, 29]` on a 2k dev subset, selects the best pair by AUROC, then trains a final logistic regression at that location on the full train set. Config: [`configs/methods/llmsknow_probe.json`](../configs/methods/llmsknow_probe.json); implementation at [`activation_research/llmsknow_probe.py`](../activation_research/llmsknow_probe.py). Seeded (5 seeds). Does not use response logprobs. Contrast with the single-layer linear probe (fixed layer 22, mean-pooled) — LLMsKnow chooses *both* layer and token position data-adaptively, so it is a stronger upper-bound on what a linear read-out can achieve at any single `(layer, token)` coordinate.
- **SAPLMA** (`azaria2023internal`). Three-hidden-layer MLP head `4096 → 2048 → 1024 → 512 → 1` with ReLU + Dropout on the last-token activation at layer 22; balanced sampling; LR 1e-3. Config: [`configs/methods/saplma.json`](../configs/methods/saplma.json), model class [`activation_research/model.py:SimpleHaluClassifier`](../activation_research/model.py). ~11M params — the supervised-MLP-on-one-layer reference point that rules out "more capacity is all you need." This is the load-bearing baseline for the §3.4 attribution table.
- **ICR Probe** (`zhang2025icr`). Per-layer JSD-style scalar (one scalar per layer → L-dim vector per generation) fed through an `L → 128 → 64 → 32 → 1` MLP with BatchNorm + LeakyReLU + Dropout, trained on the substring-match label. Config: [`configs/methods/icr_probe.json`](../configs/methods/icr_probe.json); model + trainer at [`activation_research/icr_probe.py`](../activation_research/icr_probe.py) / [`activation_research/icr_trainer.py`](../activation_research/icr_trainer.py). One sentence on the labeling discrepancy (~0.675 reproduction vs. paper's 0.7982 on Llama-3 — attributable to labeling pipeline, see §4 item 10 of `PAPER_ROADMAP.md` and our prose footnote). Phase-2 rollout (5 remaining datasets) is gated on Phase-1 competitive rank per the roadmap; the writing agent must check `python scripts/results_table.py` before locking which ICR cells are in §5 vs. acknowledged as gated-out.
- **ACT-ViT** (`barshalom2025actvit`). ViT over the full `(L × N × D)` activation tensor with adaptive max-pool to `(L_p=8, N_p=100)`, per-LLM `LinearAdapter` to `d_adapter=256`, `patch_h=2 × patch_w=10` patches, a 4-layer / 8-head transformer encoder, BCE head. Config: [`configs/methods/act_vit.json`](../configs/methods/act_vit.json); wired into all 12 `baseline_comparison_*_memmap.json` configs. The architecturally-closest prior method; the reference point that pins the §2.3 novelty claim to "view definition + objective," not "multi-layer probing."
- **Multi-layer linear probe — explicit ablation only.** Concatenation of layers 14–29 fed to a linear head; consistently underperformed the single-layer probe at layer 22 on seed-0 (per [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §2 status table; commit `fabf4de`). Reported once in §7.2 as a motivation row, *not* as a §4 main baseline. State this in one sentence and forward to §7.2; do not defend in §4.

#### Sampling-based baselines (K=10; ten forward passes; output-space)

- **Semantic Entropy (SE)** (`farquhar2024semantic`). Length-normalized SE as the headline scorer; discrete SE reported as a supplementary row. Implementation: [`tasks/sampling_baselines/se.py`](../tasks/sampling_baselines/se.py). K=10 samples per item, NLI-clustering via DeBERTa-v3-large MNLI. Five free-form datasets only (MMLU excluded — NLI clustering degenerates on single-letter answer tokens). SearchQA test capped at 10k items.
- **SelfCheckGPT** (`manakul2023selfcheckgpt`). NLI variant as the headline scorer; BERTScore and n-gram as supplementary rows. Implementation: [`tasks/sampling_baselines/selfcheck.py`](../tasks/sampling_baselines/selfcheck.py). K=10 samples; black-box-compatible (does not require activations).
- **SEP-SE** (`kossen2024semantic`). Ridge probe on last-token activations at the linear-probe layer, target = length-normalized SE score; evaluated as a hallucination detector via AUROC against the binary halu label on test. Implementation: [`tasks/sampling_baselines/sep.py`](../tasks/sampling_baselines/sep.py). Trained on a 5k stratified train subset per (dataset, model). **Bridges the K=1 and K=10 clusters** — single-pass cost at inference, but the *training* signal was derived from a K=10 sampling pass. The closest single-pass white-box competitor to our method; the §5.3 compute-matched panel rests on this comparison being clean. Note explicitly: "**SEP-binary**" was a confabulation propagated from an earlier outline-writer agent and has been purged from all paper-side and roadmap files — it is **not in Kossen et al. 2024** and **not in our codebase** (see [`tasks/sampling_baselines/sep.py:6`](../tasks/sampling_baselines/sep.py#L6)). Do not mention it.

**What we did not run.** Retrieval-augmented baselines (FActScore family, `min2023factscore`); long-form / atomic-fact baselines; closed-model API-only baselines; multilingual baselines. State this in one sentence and forward to §9 limitations.

**What §4.3 establishes for the rest of the paper.** Every baseline §5 names is grounded in a config path + paper citation here; §5.3 compute-matched comparison reads naturally because §4 has already stated the K=1 vs. K=10 partition. The §3.4 2×2 attribution table connects to §4 only via the SAPLMA row — make sure the §4 SAPLMA bullet matches the §3.4 cell description verbatim ("11M-param MLP probe on single layer's last-token activation, layer 22").

---

### 4.4 Training procedure (per-cell)

**Framing sentence.** Per-cell training procedure shared across all trained baselines. Most of the *method-specific* detail lives in §3.3 (compressor architecture, loss form, scorer choice); §4.4 names the *cell-level* protocol — what counts as one (model, dataset, method, seed) cell, where activations come from, how splits are made, and how scores are derived.

**Bullets — write as 1–2 sentences:**

- **One cell = one (model, dataset, method, seed).** Five training seeds per cell; split seed 42. Each seed determines a random train/val split of the training set — the held-out test split is fixed across all methods and seeds within a (model, dataset) cell. The prose states "five random seeds" and reports mean ± 95% CI; the specific seed values are not reported.
- **Activations are cached once.** Per (model, dataset, split), a single greedy inference pass produces the `icr_capture` memmap directory containing per-token hidden states for layers 14–29, attentions, response logprobs (top-20), labels, and metadata. All trained baselines read from the *same* cache — no per-method re-extraction. Pipeline entrypoint: [`scripts/capture_inference.py`](../scripts/capture_inference.py); paths follow the `shared/icr_capture/<dataset>_<split>_<model>/` convention named in the dataset configs.
- **Layer band.** All activation-space methods read from the mid-to-late residual-stream band `[ℓ_lo=14, ℓ_hi=29]` (verified against every `*_memmap.json` method config — `relevant_layers: "14-29"`). Justification lives in §3.2 (truth-direction evidence from `li2023iti`, `marks2024geometry`); §4 does not re-argue it.
- **Training entrypoint.** `python scripts/run_experiment.py --experiment configs/experiments/baseline_comparison_<dataset>_memmap.json` for Llama; `_qwen3_memmap.json` for Qwen3. The runner iterates the methods × seeds product. Resume-safe (`resume=True` by convention).
- **Optimizer & training hyperparameters — forward to §3.3 + Appendix B.** State once that the headline contrastive method runs at `lr=1e-5`, `batch_size=512`, `temperature=0.25`, `recon_lambda=1.0`, `min_total_steps=3000` (from [`configs/methods/contrastive_logprob_recon.json`](../configs/methods/contrastive_logprob_recon.json)). The full hyperparameter table for every baseline lives in Appendix B; §4 prose names the headline-method values only.
- **Scorer choice — forward to §3.3 + §7.4.** KNN at `k=50` (calibrated against `[50, 100, 200, 500, 1000]`) over the truthful-class embedding bank under Euclidean distance is the headline scorer; cosine and Mahalanobis are reported as ablation rows in §7.4 ([`configs/methods/contrastive_logprob_recon.json`](../configs/methods/contrastive_logprob_recon.json) `"metrics": ["cosine", "mds", "knn"]`). §4 states this once and forwards.
- **Cross-dataset transfer — §6 procedure stated cold here.** Per (model, source, target) cell, evaluate the source-trained `(method, seed)` checkpoint on the *target* test split. No retraining, no fine-tuning, no scorer re-calibration beyond what is already done at training time. State this in one sentence here so §6 can open with results rather than method.

**What §4.4 establishes for the rest of the paper.** The mechanical recipe for "one cell" — which is the unit §5, §6, and §7 all report over. Reader exits §4.4 knowing what 5 seeds means, where activations come from, and that the scorer/hyperparameter choices are pinned in §3.3 + Appendix B rather than re-debated in §4.

---

### 4.5 Metrics

**Framing sentence.** AUROC is the headline metric; AUPRC is paired in the main table to surface class-imbalance effects; ECE, FPR@95, and bootstrap 95% CIs across seeds live in the supplementary tables. All metrics evaluated on the held-out test split; no test-set selection across runs.

**Bullets — write as 1 sentence each:**

- **AUROC (headline).** Standard binary-classification AUROC against the substring-match hallucination label. Higher = better. Reported in every cell of the §5 main table and the §6 transfer matrix.
- **AUPRC (paired).** Average precision against the same label. Paired with AUROC in the main table because class imbalance varies materially across datasets (PopQA / SearchQA ≠ SciQ / HotpotQA), and AUROC alone can hide degraded precision on the minority class.
- **Expected Calibration Error (ECE).** Standard binned ECE (15 equal-width bins) on the predicted hallucination score, mapped to [0, 1] via the empirical CDF of training-set scores. Reliability diagrams for one dataset per model in §5.4; ECE numbers in the main table where the column fits.
- **FPR@95 (operational metric).** False-positive rate at 95% true-positive rate — standard operational detection metric (lower = better). Supplementary table only.
- **Bootstrap 95% CIs across seeds.** For every reported AUROC / AUPRC, a 95% CI from 1000-sample bootstrap over the 5 training seeds. Reported as `mean ± half-width` in tables (`\resultCI` macro) and as error bars in figures.
- **Where the numbers come from.** All §5 / §6 / §7 numbers source from `paper/data/*.csv` via the `\result` / `\resultCI` / `\resultPM` macros (per `paper/CLAUDE.md` and `paper/README.md`). §4 states this once and forwards; do not write bare numbers in §4 prose.

**What §4.5 establishes for the rest of the paper.** The metric axis of every results section. Reader exits §4.5 knowing AUROC is the column to scan first, AUPRC is the imbalance check, ECE / FPR@95 / CIs are the rigor check.

---

### 4.6 Compute budget summary

**Framing sentence.** One paragraph naming the total GPU-time budget and the dominant cost driver. Full breakdown in Appendix E.

**Bullets — collapse to one paragraph in prose:**

- **Single 80 GB GPU suffices for one full cell** once activations are cached. Training a single (model, dataset, method, seed) cell is <1 hour for the headline contrastive method; the linear probe and SAPLMA are minutes.
- **The dominant cost is greedy inference + activation caching**, not training. Per [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §7, the longest pole is the per-model inference + activation logging across 6 datasets × 2 splits — ~1 week wall time at single-node throughput. Once cached, every downstream method reuses the same files.
- **Sampling-based baselines** (K=10 sampling pass) account for the bulk of the *non-training* GPU spend: ~25–30 GPU-hours full-scope (both models, 5 free-form datasets), ~12–15 GPU-hours Llama-only fallback.
- **Total GPU-hours table in Appendix E.** §4 prose states the summary; the per-phase breakdown lives in the appendix.

**What §4.6 establishes for the rest of the paper.** A one-paragraph honest accounting of compute. Reader exits §4 knowing the scope of effort represented in §5 / §6 / §7 is order-of-magnitude one-week-wall-time-per-model on a single 80 GB GPU, not a 1000-GPU-day investment.

---

## Open questions for §4

These need explicit decisions before §4 prose finalizes. Most are pending citations or pending verification against live configs / data.

1. **Nine pending-approval citations.** Two model citations (Llama-3.1, Qwen3), six dataset citations (HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA), and one baseline citation (LLMsKnow — paper identity unconfirmed, see pending-approval candidates). All nine must be human-verified before insertion into `.bib`; the writing agent should *not* draft `.bib`-changing prose at full fidelity until all eight land — wording will be hedged. Recommend bundling the verification pass; entries are in the "Pending-approval candidates" section below with arXiv IDs and venues for the human to confirm.
2. **Class-imbalance numbers and per-dataset sizes — defer to Appendix D.** §4.2 bullets state structural facts about each dataset but no concrete class-imbalance percentages, because the numbers must come from `paper/data/datasets.csv` via the `\result` macro and that CSV does not yet exist. Either (a) write `paper/data/datasets.csv` first (preferred — one-time recompute from cached `generation.jsonl` and `eval_results.json`), or (b) defer the imbalance numbers to Appendix D and footnote them out of §4.
3. **`paper/data/datasets.csv` does not exist.** `paper/build_numbers.py` is the pipeline; no script yet generates per-dataset sizes / imbalance ratios. The writing agent owes the human a small companion task: "add a `paper/data/datasets.csv` source step to `build_numbers.py` and populate from cached `generation.jsonl` + `eval_results.json` for each (dataset, model) cell." This is the **smallest blocker** to §4.2 prose finalization.
4. **ICR Probe Phase-2 status — affects how many cells §5 fills.** Per [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §2 status table, ICR Phase-2 rollout (5 remaining datasets × 2 models × 5 seeds) is gated on Phase-1 competitive rank. §4 names the ICR baseline structurally; if Phase-2 does not land by §5 freeze, §4 prose should hedge ("evaluated on HotpotQA across both models; remaining datasets gated on the Phase-1 result"). The writing agent must check `python scripts/results_table.py` before locking the ICR coverage statement.
5. **Qwen seed sweep — 21 cells outstanding (as of 2026-05-15).** [`PAPER_ROADMAP.md`](../PAPER_ROADMAP.md) §2 reports 111/132 Qwen cells complete. If the 21 outstanding cells do not land by §5 freeze, the §4 model statement holds (we still ran Qwen) but §5 must hedge per-cell coverage. Re-run `python scripts/results_table.py` before §4 prose locks.
6. **Whether to fold "what we did not run" into §9 or repeat structurally in §4.3.** Currently §4.3 names three "what we did not run" categories in one sentence each (retrieval, long-form, closed-API, multilingual). §9 limitations gives the longer treatment. Decision: keep both — §4.3 names the scoping choices; §9 explains the why. One sentence in §4 is not duplication. Confirm at internal-review pass.
7. **Page budget.** Target is 0.5–0.75 page. Six subsections at the bullet density above is closer to 0.75 page in prose. If the page budget binds (e.g., §3 expands above its budget), §4.4 is the first to compress — collapse the training-procedure bullets into a single paragraph of three sentences and push the rest to Appendix B. §4.6 is the second to compress — collapse to a single sentence forwarding to Appendix E.

---

## Cross-references this outline expects to call

- §2 (related work) — §4.3 baselines are anchored in §2 citations; §4 cites the *implementations*, §2 cites the *papers*. Avoid duplication.
- §3.2 (information-theoretic argument) — §4.4 forwards layer-band motivation here; §4 does not re-argue.
- §3.3 (architecture + training recipe) — §4.4 forwards optimizer / scorer / hyperparameters here. §4 names the *headline* numerics once; §3.3 names the full specification; Appendix B has the full table.
- §3.4 (2×2 attribution table) — §4.3 SAPLMA bullet must match the §3.4 SAPLMA cell verbatim. Cross-check at draft-time.
- §5 (main results) — every §5 cell maps to a §4 baseline + a §4 metric. §4 is the legend for the §5 table.
- §5.3 (compute-matched comparison) — §4.3's K=1 / K=10 partition and §4.6's compute summary are the structural prerequisites; §5.3 reads them.
- §5.4 (calibration) — §4.5's ECE definition is the prerequisite.
- §6 (transfer) — §4.4 states the transfer procedure cold; §6 opens with results.
- §7.1 (loss decomposition) — §4.3 SAPLMA bullet + §3.4 2×2 are the structural prerequisites.
- §7.2 (multi-layer probe ablation) — §4.3 explicitly punts to §7.2 for the multi-layer probe row.
- §7.3 (layer-pair sensitivity) — §4.4 names the band; §7.3 sweeps within it.
- §7.4 (scorer choice) — §4.4 forwards to §7.4; §7.4 reports the cosine / Mahalanobis / KNN comparison.
- §9 (limitations) — §4.1's "no 70B, no closed models" and §4.2's "Movies excluded, short-form only" both connect to §9.
- Appendix B — full hyperparameter table for every baseline.
- Appendix D — per-dataset details: train/test size, class-imbalance, evaluator prompt, license. *Source CSV does not yet exist (see open question 4).*
- Appendix E — GPU-hours breakdown.

---

## References — materialized for §4

Every entry below is **already in `paper/references.bib`** as of 2026-05-19. Bibkey, citation context, and *role each reference plays in our §4 prose* are stated so the writing agent does not need to re-research. §4 borrows heavily from §2.1 / §2.2 — many citations are repeats of references that anchor §2, used in §4 to ground the *implementation* claim rather than the *prior-art* claim.

### Activation-space probes (§4.3 second class)

- **`alain2017understanding`** — Alain & Bengio, "Understanding intermediate layers using linear classifier probes," ICLR Workshop 2017.
  - **Role in §4 prose.** Provenance for the single-layer linear-probe baseline (one citation when the linear probe is named).
- **`azaria2023internal`** — Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying," Findings of EMNLP 2023.
  - **Role in §4 prose.** Provenance for the SAPLMA baseline. One citation when SAPLMA is named.
- **`zhang2025icr`** — Zhang et al., "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs," ACL 2025.
  - **Role in §4 prose.** Provenance for the ICR Probe baseline. One citation + a forward-pointer to the §4 item 10 methodology footnote in `PAPER_ROADMAP.md` for the labeling-pipeline discrepancy.
- **`barshalom2025actvit`** — Bar-Shalom et al., "Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT," NeurIPS 2025.
  - **Role in §4 prose.** Provenance for the ACT-ViT baseline. One citation when ACT-ViT is named.
### Output-space scalar baselines (§4.3 first class)

- **`kadavath2022language`** — Kadavath et al., "Language Models (Mostly) Know What They Know," arXiv:2207.05221, 2022.
  - **Role in §4 prose.** Single grouped citation for the three output-space scalar baselines (mean logprob, token entropy, P(true)). One citation covers all three.

### Sampling-based baselines (§4.3 third class)

- **`farquhar2024semantic`** — Farquhar et al., "Detecting hallucinations in large language models using semantic entropy," Nature 2024.
  - **Role in §4 prose.** Provenance for the Semantic Entropy baseline (length-normalized headline + discrete supplementary).
- **`manakul2023selfcheckgpt`** — Manakul et al., "SelfCheckGPT," EMNLP 2023.
  - **Role in §4 prose.** Provenance for the SelfCheckGPT baseline (NLI headline + BERTScore + n-gram supplementary).
- **`kossen2024semantic`** — Kossen et al., "Semantic Entropy Probes," arXiv:2406.15927, 2024.
  - **Role in §4 prose.** Provenance for SEP-SE. One citation + a clear, single-sentence disclaimer that "SEP-binary" is a confabulation and we run only SEP-SE.

### Scoped-out references named in §4

- **`min2023factscore`** — Min et al., "FActScore," EMNLP 2023.
  - **Role in §4 prose.** Single citation in the "what we did not run" bullet of §4.3 (retrieval-augmented / long-form). Forward to §9 limitations.

---

## Pending-approval candidates — NOT in references.bib

Live candidates the writing agent needs for §4. **Do not add to `.bib` without human verification** (per `paper/references.bib` policy header and `CLAUDE.md`). Each entry states what the candidate would buy and what the human should verify before approval. **Datasets, models, and LLMsKnow: §4 owes the human nine verification decisions in one batch.**

### Baselines

- **(REQUIRED — activation-space probe in the live grid)** Unknown paper — LLMsKnow. Proposed bibkey: `[UNKNOWN — verify]`.
  - *What it covers.* The LLMsKnow method searches for the most discriminative `(layer, token_position)` pair in the residual stream via a dev-set sweep, then trains a logistic regression probe at that location. This is the approach our `llmsknow_probe` routine implements ([`activation_research/llmsknow_probe.py`](../activation_research/llmsknow_probe.py)).
  - *Would buy.* The §4.3 LLMsKnow bullet; also the §4.2 NQ verify step mentions "LLMsKnow" as a labeling-convention precedent — that forward-reference only works once the citation lands.
  - *Verify before adding to `.bib`.* (a) Identify the paper this baseline corresponds to — the most likely candidate is Slobodkin et al., "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations," NeurIPS 2024 / arXiv:2410.02707, but **this must be verified against the actual paper** (confirm the (layer, token) sweep procedure matches ours). (b) If Slobodkin et al. is confirmed, verify full author list, arXiv ID, and NeurIPS 2024 venue. (c) If another paper is the source, surface it to the human before proceeding.

### Models

- **(REQUIRED — primary model)** Grattafiori et al., "The Llama 3 Herd of Models," arXiv:2407.21783, 2024. Proposed bibkey: `grattafiori2024llama3`.
  - *What it covers.* Pre-training recipe, post-training (SFT + DPO), architecture details, evaluation suites for the Llama 3 family — including the 8B-Instruct variant we use.
  - *Would buy.* The §4.1 Llama citation; otherwise §4.1 is one of the two ungrounded model statements.
  - *Verify before adding to `.bib`.* (a) arXiv ID 2407.21783 + the exact title (often rendered as "The Llama 3 Herd of Models"). (b) **First-author** name and order — Meta papers have very long author lists and the lead changes between v1 / v2; verify against the current arXiv version. Earlier rumors of "Dubey et al." as first author should be cross-checked. (c) Whether to cite the Meta release directly (model card on Hugging Face) in addition — recommend no, the arXiv tech report is the citable form.

- **(REQUIRED — second model)** Yang et al., "Qwen3 Technical Report," 2025 — verify the exact preprint/venue. Proposed bibkey: `yang2025qwen3`.
  - *What it covers.* Qwen3 family pretraining and post-training, including the 8B variant and the thinking-mode mechanism (which we disable).
  - *Would buy.* The §4.1 Qwen3 citation. As of 2026-05-19, the writing agent should **not assume** the canonical Qwen3 citation; the Qwen team historically releases technical reports on arXiv. Verify the current best-citable form.
  - *Verify before adding to `.bib`.* (a) The exact arXiv ID / publication form — confirm against `https://qwenlm.github.io/` or the Hugging Face model card link as of the verification date. (b) Whether the report names "Qwen3-8B" exactly (vs. "Qwen3 8B" or a different formal name). (c) Whether the thinking-mode disable behavior we apply has a documented switch in the report — if yes, the §4.1 prose should cite the report for that fact too.

### Datasets

- **(REQUIRED)** Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," EMNLP 2018, arXiv:1809.09600. Proposed bibkey: `yang2018hotpotqa`.
  - *What it covers.* Multi-hop open-domain QA over Wikipedia; distractor and fullwiki settings.
  - *Would buy.* The §4.2 HotpotQA bullet.
  - *Verify.* (a) arXiv ID. (b) Authors include Zhilin Yang as first author (different from the Qwen3 author Yang — pick the right one). (c) Confirm we use the dev split as our test set (standard probing-literature convention; hidden test).

- **(REQUIRED)** Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering Research," TACL 2019. Proposed bibkey: `kwiatkowski2019nq`.
  - *What it covers.* QA from real Google queries with Wikipedia gold answers.
  - *Would buy.* The §4.2 NQ bullet.
  - *Verify.* (a) TACL volume/issue. (b) Confirm the short-answer-only filter is the standard treatment in the probing literature (LLMsKnow, Kossen, Farquhar all use short-answer NQ — verify against at least one).

- **(REQUIRED)** Hendrycks et al., "Measuring Massive Multitask Language Understanding" (MMLU), ICLR 2021, arXiv:2009.03300. Proposed bibkey: `hendrycks2021mmlu`.
  - *What it covers.* 57-subject multiple-choice benchmark.
  - *Would buy.* The §4.2 MMLU bullet.
  - *Verify.* (a) arXiv ID. (b) Authors: Dan Hendrycks et al. (c) ICLR 2021 venue confirmation.

- **(REQUIRED)** Mallen et al., "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories" (PopQA), ACL 2023. Proposed bibkey: `mallen2023popqa`.
  - *What it covers.* Entity-popularity-stratified factoid QA.
  - *Would buy.* The §4.2 PopQA bullet.
  - *Verify.* (a) ACL Anthology entry (2023.acl-long.546 — confirm). (b) Authors: Alex Mallen et al.

- **(REQUIRED)** Welbl et al., "Crowdsourcing Multiple Choice Science Questions" (SciQ), W-NUT 2017, arXiv:1707.06209. Proposed bibkey: `welbl2017sciq`.
  - *What it covers.* Crowd-authored science MCQ.
  - *Would buy.* The §4.2 SciQ bullet.
  - *Verify.* (a) arXiv ID. (b) W-NUT @ EMNLP 2017 venue. (c) Authors: Johannes Welbl, Nelson F. Liu, Matt Gardner.

- **(REQUIRED)** Dunn et al., "SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine," arXiv:1704.05179, 2017. Proposed bibkey: `dunn2017searchqa`.
  - *What it covers.* Jeopardy! questions paired with web-search snippets.
  - *Would buy.* The §4.2 SearchQA bullet.
  - *Verify.* (a) arXiv ID. (b) Whether the canonical form is arXiv-only or there is a workshop / proceedings version. (c) Confirm our closed-book treatment (questions only, no snippets) is acceptable to cite under the original paper — recommend a footnote noting the deviation.

### Optional (low-priority)

- **(OPTIONAL — only if a reviewer flags the omission)** Welbl, Liu, Gardner & various follow-ups on substring-match labeling conventions. Not currently needed: the §4 item 10 methodology footnote in `PAPER_ROADMAP.md` already covers the substring-match defense by appeal to "literature standard." Add only if a reviewer at rebuttal asks for the specific labeling-convention citation chain.

---

## Drafting order recommendation

For a writing agent starting fresh on §4 prose:

1. **Verify the eight pending citations** (2 models + 6 datasets) and route to human approval in one batch. §4.1 and §4.2 cannot draft cleanly without them.
2. **Generate `paper/data/datasets.csv`** via `paper/build_numbers.py` so §4.2's per-dataset structural statements can be backed by `\result` macros where appropriate. (Smaller blocker than the bib pass; ~1 hour of script work.)
3. **Re-check `python scripts/results_table.py`** to confirm (a) ICR Probe Phase-2 coverage; (b) Qwen seed-sweep completion. §4 wording may need to hedge if either has not landed.
4. **Draft §4.1 (Models) and §4.2 (Datasets) first** — they are the most citation-dense and the most reviewer-visible. Hedge wording until pending citations land.
5. **Draft §4.3 (Baselines) next** — most of the section's *new* content (vs. §2). Pin every baseline to a config path + entrypoint; this is what makes §4 reproducible.
6. **Draft §4.4 (Training procedure), §4.5 (Metrics), §4.6 (Compute)** last — these are the shortest and have the most forward-references; they read cleanly once §3 / §5 are in front of you.
7. **Cross-check the §3.4 SAPLMA cell against the §4.3 SAPLMA bullet** verbatim before locking — the §3.4 attribution narrative depends on this.
8. **Close §4 by writing the §4.1 → §4.6 transitions last.** Each subsection should hand off cleanly; this requires reading the section end-to-end after each subsection lands.
