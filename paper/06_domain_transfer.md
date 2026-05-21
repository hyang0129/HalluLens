# Cross-Dataset Transfer — Outline

Structural skeleton **plus** materialized content. By the time a writing agent picks this up, every comparison, narrative move, and figure/table slot that will appear in §6 is named here, with the precise positioning the prose draws from. The writing agent should not need to research further; if a claim cannot be supported from the bullets below or from a `paper/data/*.csv` cell, surface it to the human rather than inventing one.

**Bib-policy reminder.** §6 introduces no new citations beyond those grounded in §2 and §4. Every method named in §6 has already been cited by [`02_related_work.md`](02_related_work.md) and [`04_experimental_setup.md`](04_experimental_setup.md). §6 prose cites a baseline once on first §6 mention and forwards thereafter; no `[PENDING-APPROVAL]` entries expected. The §6 transfer setup itself is not a new method — it is a re-use of the same trained probes evaluated on a different target split — so no methodological cite is owed.

**Structural decision (target length: ~0.5 page).** §6 is *one specific question* — when our trained probe is moved to a held-out target dataset without retraining, does it carry more transferable hallucination signal than the strongest competing learned probe trained on the same source? The main paper carries the headline answer in a single small table (per-source out-of-domain mean AUROC, ours vs. the best-of-three baselines) and one paragraph of interpretation. The aggregate in-domain / out-of-domain summary corroborates in a second small table. Granular per-cell numbers, per-target breakdowns, and the full 6×6 source-target heatmaps live in Appendix C. §6 explicitly does *not* re-state the method, re-summarize §5, or pre-empt the §7 ablations.

**Supersedes** both:
- the original [`outline.md`](outline.md#L90) §6 bullet (6×6 heatmap as main-paper table — demoted to Appendix C), and
- the 2026-05-21 first draft of this file which led with the in-domain / out-of-domain summary (Option B). That summary is now *supporting* evidence rather than the lead, because a verification pass over `results/transfer_matrix_table.csv` surfaced a stronger and more reviewer-defensible claim at the per-source level. See "Tabling decision" below.

**⚠ Temporary-numbers banner — read before citing any number from this outline.** Every numeric value below was lifted from a 2026-05-21 verification pass over `results/transfer_matrix_table.csv` (counts: 8 wins / 1 tie / 1 loss vs. per-source-best-baseline across 10 reportable model × source cells; NQ-as-source has no data on either side and is excluded from the count). **Those numbers are placeholders, not load-bearing.** They will move when (a) NQ-as-source cells close (see Open question 1 — this is the **critical blocker**), (b) any other outstanding seeds close, and (c) the `paper/data/transfer_per_source.csv` aggregator lands. Until that re-edit happens, do not lift numbers from this outline into prose or figures — pull them from `paper/data/transfer_per_source.csv` (which does not yet exist; see "Data dependencies" below).

**Numbers-from-CSV policy.** Per `paper/README.md`, every numeric value in the final prose must come from `paper/data/*.csv` (or `paper/generated/figures/*.numbers.csv`) via the `\result` / `\resdelta` / `\resultPM` macros in `macros.tex`. As of 2026-05-21, the §6 source aggregator (`results/transfer_matrix_table.csv`) lives outside `paper/` and is *not* yet ingested by `paper/build_numbers.py`. Three new committed CSVs (per the "Data dependencies" table below) must be authored before §6 prose locks.

**Freeze ordering (per [outline.md:82](outline.md#L82)).** §6 prose is written *after* §5 numbers freeze and *after* NQ-as-source completes. The §6 in-domain corroboration table (§6.3) uses the same diagonal cells as §5.1, so §5 / §6 consistency is a hard precondition. Slot reservation is a hard constraint: any prose draft must use the `\result` macros, never bare numerals.

**Tabling decision (locked 2026-05-21, revised same day).** Of the four tabling options the human originally surfaced (per the 2026-05-21 tabling-proposal conversation):
- **Option A** (faceted 6×6 heatmaps per method × model, 8 tiles total) — too dense for the main paper. Reviewers scan but do not read; the headline claim does not fit on a single panel. **→ Appendix C** as the raw-data reference for the per-cell transfer matrix.
- **Option B** (in-domain vs. out-of-domain summary with gap column) — true but bland; the headline "ours wins both columns on both models" leaves the reader without a mechanism. **→ Demoted to §6.3** as *aggregate corroboration* for the §6.2 per-source headline. Still earns its place because the in-domain column is the §5 diagonal restated and the gap column pre-empts the "transfer falls apart" reviewer worry.
- **Option C** (per-target out-of-domain AUROC, mean over 5 source datasets ≠ target) — rigorous per-target view; **→ Appendix C** alongside Option A's heatmaps.
- **Option D** (Δ-heatmap, ours − best baseline per cell) — mixed signs invite cell-counting arguments; **→ Cut.**
- **NEW — Option E** (per-source out-of-domain mean AUROC by training source, ours vs. per-source-best-of-three-baselines, both models) — surfaced in the 2026-05-21 verification pass. **→ Main paper §6.2 headline table.** This is what carries the "no single competing baseline generalizes better than ours" claim and is the strongest single-glance result in §6.

---

## 6. Cross-Dataset Transfer

### 6.1 Setup

**Framing sentence.** Each probe trained on dataset A in §5 is evaluated unchanged on dataset B's test split, for every ordered pair `(A, B)` and both models. The probe is *not* retrained, *not* fine-tuned, and *not* given any signal from B beyond the same activation-extraction pipeline used to compute B's §5 column. Six datasets × six targets × two models × four methods = 144 transfer cells in scope; 5 seeds per cell.

**Bullets — write as 1–2 sentences in prose each:**

- **Methods compared.** Four learned activation-space methods, mirroring the §5.1 activation cluster minus ICR Probe (which lacks the full transfer-matrix coverage to be reportable):
  1. **Ours** — `contrastive_logprob_recon` (provisional method-name flag still open; see [`outline.md`](outline.md#L184) Open Question 7).
  2. **ACT-ViT** — Bar-Shalom et al. 2025; the strongest learned competitor per §5.
  3. **LLMsKnow probe** — Slobodkin et al. 2023; the layer-wise probing baseline.
  4. **SAPLMA** — Azaria & Mitchell; the established single-layer MLP literature baseline.
  Output-space scalars (LogProb, Token Entropy, P(true)) and sampling methods (SE, SelfCheckGPT) are excluded — they have no notion of "transfer" because there is no trained probe to transfer.

- **Per-source out-of-domain definition.** "Per-source out-of-domain mean" = for a fixed (model, method, source) triple, the unweighted mean over the 5 targets ≠ source, each cell itself being a 5-seed mean. Six per-source means per (model, method). This is the unit of §6.2's headline table.

- **Aggregate in-domain / out-of-domain definitions.** "In-domain" = unweighted mean over the 6 diagonal cells (source = target) per (model, method). "Out-of-domain" = unweighted mean over the 30 off-diagonal cells per (model, method). Unweighted is load-bearing — every dataset weighted equally so a dataset-imbalance reviewer cannot argue the mean is dominated by one cell. This is the unit of §6.3's corroboration table.

- **No domain-adaptation step.** State once that no labeled target-domain data, no unsupervised target-domain adaptation, and no target-side calibration is applied. This is a *zero-shot transfer* setup in the probe-on-activations sense.

- **What §6 does *not* claim.** It does not claim our probe is a *universal* hallucination detector — out-of-domain AUROC is materially lower than in-domain on every method on every model (gaps `0.11–0.13 AUROC` on the 2026-05-21 draft). It claims, precisely: across the source datasets where transfer is reportable, no single competing learned baseline matches ours' per-source transfer performance on a majority of (model, source) cells; and ours' aggregate transfer mean exceeds every individual baseline's aggregate transfer mean on both models.

---

### 6.2 Headline table — per-source transfer (Option E, main paper)

**Framing sentence.** Two model-blocks stacked. Within each block, six rows (one per source dataset), with columns showing ours' out-of-domain mean AUROC, the best-of-three-baselines value with the baseline named, and the delta. The headline claim — *no single competing baseline generalizes better than ours when trained on a free-form factual QA source* — lands in one glance.

**Bullets — write as 1–2 sentences in prose each:**

- **Table layout.** Two horizontally stacked subtables, one per model (Llama-3.1-8B-Instruct, Qwen3-8B). Rows = six source datasets, in the §5.1 column order (HotpotQA, NQ, MMLU, PopQA, SciQ, SearchQA). Columns: `Ours (OOD mean)`, `Best baseline (OOD mean)`, `Best baseline name`, `Δ (ours − best baseline)`. The "Best baseline name" column is short — `ACT-ViT` / `LLMsKnow` / `SAPLMA` — and is load-bearing because the best baseline *varies by source*. **Bolding rule:** within each row, bold the cell (ours or best-baseline) whose mean exceeds the other by more than `max(stds)` of the two; otherwise call them tied and bold neither. Mirror the §5.1 bolding rule for consistency.

- **Reporting unit.** Each cell is `mean ± std` across the 5 seeds of the per-source out-of-domain mean. Sourced via `\resultPM{transfer_per_source}{<model>:<method>:<source>:auroc:mean,...:auroc:std}[3]`. The delta column is `\resdelta{transfer_per_source}{<model>:contrastive_logprob_recon:<source>:auroc:mean}{<model>:<best_baseline>:<source>:auroc:mean}[3]`.

- **Draft headline numbers (Llama-3.1-8B-Instruct)** `[TEMP — every number here re-verifies from transfer_per_source.csv]`:

  | Source | Ours (OOD mean) | Best baseline (OOD mean) | Best baseline | Δ |
  |---|---|---|---|---|
  | HotpotQA | **0.748** | 0.707 | ACT-ViT | **+0.041** |
  | NQ | `[PENDING — see Open Q1]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
  | MMLU | 0.671 | 0.675 | ACT-ViT | −0.003 (tied) |
  | PopQA | **0.659** | 0.599 | LLMsKnow | **+0.061** |
  | SciQ | **0.717** | 0.681 | ACT-ViT | **+0.036** |
  | SearchQA | **0.734** | 0.663 | LLMsKnow | **+0.071** |

  **Draft headline numbers (Qwen3-8B)** `[TEMP]`:

  | Source | Ours (OOD mean) | Best baseline (OOD mean) | Best baseline | Δ |
  |---|---|---|---|---|
  | HotpotQA | **0.768** | 0.749 | ACT-ViT | **+0.018** |
  | NQ | `[PENDING — see Open Q1]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
  | MMLU | 0.648 | 0.663 | ACT-ViT | −0.016 (loss) |
  | PopQA | **0.697** | 0.605 | ACT-ViT | **+0.092** |
  | SciQ | **0.734** | 0.650 | LLMsKnow | **+0.085** |
  | SearchQA | **0.748** | 0.696 | LLMsKnow | **+0.051** |

- **What the table is supposed to *show* — three nested claims, prose orders them outermost-first.**
  1. **No single competing baseline matches our per-source transfer.** Across the 10 (model × source) cells where the comparison is reportable on the 2026-05-21 draft, ours wins **8**, ties **1** (Llama-MMLU within ±0.005), and loses **1** (Qwen-MMLU, −0.016) `[TEMP — count re-verifies once NQ closes]`. Both non-wins are MMLU; everything else is a margin-bolded win. This is the cleanest single sentence in §6.
  2. **The best baseline varies by source.** Ours' wins do not come from a single weak competitor — the strongest baseline is ACT-ViT on HotpotQA, MMLU, SciQ-Llama, PopQA-Qwen; LLMsKnow on PopQA-Llama, SearchQA on both models, SciQ-Qwen. Naming the baseline per row pre-empts the "you cherry-picked which baseline to compare against" reviewer move. **This is the load-bearing reviewer-defensibility move in §6.2 — do not collapse the "Best baseline name" column.**
  3. **MMLU as source is a structural exception, not a counterexample.** Both non-wins are MMLU. Plausibly the same mechanism that drives MMLU's outlier behavior elsewhere in the paper — multiple-choice format produces hallucination signal that is so source-format-specific that *no* method extracts cross-domain-transferable features from it. Forward the longer interpretation to §6.4 and §8; in §6.2 say it in one clause.

- **Cell-by-cell narrative — one paragraph after the table.** Fixed template: (1) state the count (claim 1 above); (2) name PopQA, SearchQA, SciQ as the cells where the win is large (>0.05 AUROC, both models or single-model); (3) name MMLU as the one source where transfer is flat for every method, and forward interpretation to §6.4; (4) restate that the best baseline changes per row (claim 2 above); (5) forward NQ once data lands. ~5 sentences; do no more than read the table.

- **Coverage footnote.** Single footnote on the table: per-cell seed counts. On the 2026-05-21 draft, ACT-ViT has one off-diagonal cell missing on Qwen-SearchQA as source (n=24 of 25 expected after subtracting NQ-target); all other off-diagonal cells reportable for ours, LLMsKnow, SAPLMA are 5/5 seeds × 5 targets per source = 25 cells per (model, source). The `transfer_per_source.csv` schema must carry `n_seeds` and `n_target_cells` so the footnote renders via macros.

- **What §6.2 establishes for the rest of the paper.** The §1 contribution-2 verb against domain transfer. The §8 discussion's lead into the MMLU-as-format-outlier story. The §6.3 aggregate table reads as corroboration, not a separate finding. Reader exits §6.2 knowing the headline — *no single competing baseline matches ours' transfer on a majority of reportable sources, and the best baseline changes per source so ours is not lucky-against-one-weak-method*.

---

### 6.3 Aggregate corroboration — in-domain vs. out-of-domain (Option B, supporting)

**Framing sentence.** Demoted from headline to corroboration. Restates the §6.2 finding in aggregate form: the mean across all 30 off-diagonal cells per (model, method) likewise has ours first, beating every individual baseline on both models. Pre-empts the "you sliced this by source" reviewer angle.

**Bullets — write as 1 sentence each:**

- **Table layout.** Two stacked sub-tables (one per model), four rows (one per method), three columns: `In-domain` (6-cell mean), `Out-of-domain` (30-cell mean), `Gap` (= in-domain − out-of-domain).

- **Reporting unit.** `mean ± std` across the 5 seeds of the cell-set mean. Same macro path family as §6.2 but pointing at `transfer_summary.csv`.

- **Draft numbers** `[TEMP — re-verify from transfer_summary.csv; numbers below assume NQ-as-source closes near the all-method mean and aggregates may shift if NQ lands materially off]`:

  | Model | Method | In-domain | Out-of-domain | Gap |
  |---|---|---|---|---|
  | Llama | Ours | **0.812** | **0.694** | +0.117 |
  | Llama | ACT-ViT | 0.768 | 0.657 | +0.111 |
  | Llama | LLMsKnow | 0.779 | 0.632 | +0.146 |
  | Llama | SAPLMA | 0.641 | 0.553 | +0.088 |
  | Qwen | Ours | **0.851** | **0.721** | +0.131 |
  | Qwen | ACT-ViT | 0.807 | 0.681 | +0.126 |
  | Qwen | LLMsKnow | 0.789 | 0.620 | +0.170 |
  | Qwen | SAPLMA | 0.720 | 0.608 | +0.112 |

- **Bolding rule.** Same as §5.1 / §6.2 for `In-domain` and `Out-of-domain` columns. **Do not** bold the `Gap` column — SAPLMA has a small gap by virtue of being uniformly worst, not by being more transfer-robust. State this in the caption.

- **Interpretive footnote.** One sentence: the gap column is reported to anticipate the "does transfer fall apart" question, not to argue smallest-gap is best. The §6.2 per-source view is where the load-bearing transfer-quality claim lives.

- **Coverage footnote.** ACT-ViT off-diagonal coverage on the 2026-05-21 draft is `25/30` (Llama) and `21/30` (Qwen); all other methods are `30/30`. NQ-as-source is a row-shaped hole affecting every method equally; once it closes the totals become `30/30` for ours / LLMsKnow / SAPLMA and `≤30/30` for ACT-ViT pending its NQ cells.

---

### 6.4 Interpretation (one paragraph)

**Framing sentence.** One paragraph after the two tables reading what the result implies about the *signal* the trained probe extracts — not re-stating the §5 method, not pre-empting §7 / §8.

**Bullets — write as 1–2 sentences in prose each:**

- **What this means about ours.** The §6.2 count — no single competing learned baseline matches ours' per-source transfer on a majority of reportable cells, and the best baseline changes per source — is most parsimoniously explained by ours' trained probe extracting a hallucination signal that is *less source-format-specific* than any single competitor's. This connects directly to the §3.2 cross-layer-coherence argument: coherence is a property of the residual stream rather than of the specific QA format, so a representation that targets coherence should transfer further than one that targets surface QA features. §6.4 names the connection in one clause and forwards the full argument to §8.

- **The MMLU exception.** Both non-wins in §6.2 are MMLU-as-source. The MMLU format (multiple-choice, single-letter completion) produces an activation pattern that none of the four methods transfers out of — every method's MMLU-as-source out-of-domain mean is `0.62–0.68` `[TEMP]`, below every method's free-form-QA-as-source mean by `0.03–0.10`. This is plausibly a *signal*-level limitation rather than a *method* asymmetry — there is less cross-domain-transferable hallucination signal to extract from a single-letter completion in the first place. §8 owes the longer interpretation; §6.4 names it in one sentence.

- **Where transfer breaks down on the target side.** On the per-target view (Appendix C, Option C), NQ-as-target is the hardest target across methods (preliminary, pending NQ-source data closing). The §5.1 NQ shortfall re-appears here on the target axis; §8 picks up both.

- **What §6 does *not* establish.** It does not establish that the trained probe is robust to *arbitrary* domain shifts — all six datasets are English short-form factual QA / multi-choice. Long-form (FActScore-style), non-English, or non-QA domains are out of scope; §9 forwards this limitation.

**What §6 establishes for the rest of the paper.** A second axis (beyond §5's in-domain table) on which ours' margin over competing learned baselines is consistent — and on the source-transfer axis, that margin is *not* shared with any single baseline. §1 contribution-2 ("transfers across datasets without retraining" — [outline.md:33](outline.md#L33)) is grounded here, with the verb-strength upgrade that §6.2 supports ("transfers further than every competing baseline on a majority of reportable sources"). The §8 NQ-shortfall + MMLU-format-outlier discussion picks up §6.4. Reader exits §6 with the headline: *ours' learned probe transfers more broadly than every competing baseline when trained on free-form factual QA, and no single baseline closes the gap.*

---

## Open questions for §6

These need explicit decisions before §6 prose finalizes. **§6 prose does not start until Open Q1 closes** (per the 2026-05-21 human directive).

1. **NQ-as-source cells (critical blocker).** On the 2026-05-21 draft, NQ-as-source returns no `status=complete` rows for any method on either model in `results/transfer_matrix_table.csv`. This is a row-shaped coverage hole that affects every method equally and prevents the §6.2 count from being final. Three sub-decisions:
   - **(a) Verify the hole.** Check whether NQ-as-source dispatches were ever submitted or whether they failed silently. Possible explanations: missing in dispatch manifest, failed and not re-dispatched, or aggregator filter dropped them. Run `python scripts/audit_datasets.py` and inspect the `runs/transfer_matrix_memmap/` job manifest before assuming the cells need to be (re-)launched.
   - **(b) Dispatch decision.** If the cells are genuinely missing, this requires explicit user approval per `CLAUDE.md` ("Never submit or kill SLURM jobs without explicit user approval"). Do not pre-empt.
   - **(c) Interim §6 prose freeze.** Until NQ-as-source closes for at least ours + ACT-ViT (the two methods in the §6.2 bold count), §6 prose is **NOT** to be drafted. The 10-cell count becomes 12 once NQ closes, and the headline win/tie/loss count may move (e.g., 10 wins / 1 tie / 1 loss if NQ-as-source is a win for ours on both models — strengthening the headline; or 9 / 1 / 2 if NQ-as-source goes ACT-ViT's way on a model — still a comfortable majority but the framing tightens).

2. **Coverage handling for ACT-ViT (off-diagonal).** ACT-ViT has 25/30 off-diagonal cells on Llama and 21/30 on Qwen on the 2026-05-21 draft. Options:
   - **(a) Report ACT-ViT's mean over the cells it has** (current default). Honest, easy, but invites a "you're cherry-picking" argument if the missing cells systematically include a hard-target column.
   - **(b) Drop ACT-ViT from §6.3** until 30/30 completes. Cleanest but discards the headline competitor.
   - **(c) Mask the cells ACT-ViT is missing from *every* method's mean** so the comparison is over the same cell set. Most rigorous; requires the aggregator to emit two means per method (full and ACT-ViT-aligned).
   - **Recommend (a) for §6.2 / §6.3 with the footnote, plus (c) as a robustness check in Appendix C.**

3. **§6.2 in-domain consistency vs. §5.1.** §6.3's in-domain column = §5.1 diagonal cells, modulo aggregator path. Confirm `transfer_per_source.csv` (or `transfer_summary.csv`) in-domain numbers match `headline_results.csv` (§5) diagonal cells to within rounding before §6 prose ships; a divergence is a §5/§6 reviewer surface.

4. **Headline verb in §1 / abstract.** The §6.2 finding supports a stronger transfer verb than the original [outline.md:33](outline.md#L33) phrasing ("transfers across datasets without retraining"). Candidate upgrades:
   - **(a) "transfers further than any single competing learned baseline on a majority of source datasets"** — most precise; matches the §6.2 count exactly.
   - **(b) "no single competing learned baseline matches our cross-dataset transfer on free-form factual QA"** — punchier; closer to the human's 2026-05-21 framing ("no single competing model can generalize better than we do").
   - **(c) keep the original**, lean on §6.2 to speak for itself.
   - **Recommend (b) for the abstract and §1 contribution-2.** Lock after NQ closes.

5. **`paper/data/transfer_per_source.csv` does not exist.** **Biggest single blocker** to §6.2 prose finalization after NQ. Aggregator companion task:
   - **Input:** `results/transfer_matrix_table.csv`.
   - **Output schema:**
     ```
     # key_schema: model:method:source
     # default_precision: 3
     model,method,source,auroc_mean,auroc_std,n_seeds,n_target_cells
     ```
     One row per (model, method, source) — 2 × 4 × 6 = 48 rows when fully populated. `n_target_cells` = number of target datasets the per-source mean was taken over (should be 5 in nominal coverage).
   - **Compute recipe:** filter `kind=transfer`, `metric_name=auroc`, `status=complete`, `source_dataset ≠ target_dataset`; for each `(model, method, seed)` compute the unweighted mean over the 5 targets ≠ source; then aggregate across the 5 seeds.
   - **Author location:** new `scripts/aggregate_transfer.py` emitting all three §6 CSVs (`transfer_per_source.csv`, `transfer_summary.csv`, `transfer_matrix.csv`) in one pass. Add a `build_numbers.py` step.

6. **Two more CSVs for the appendix.**
   - **`paper/data/transfer_matrix.csv`** for Appendix C Option A (6×6 heatmaps per method × model). Schema `key_schema: model:method:source:target`, columns `auroc_mean`, `auroc_std`, `n_seeds`. Up to 2 × 4 × 6 × 6 = 288 rows.
   - **`paper/data/transfer_per_target.csv`** for Appendix C Option C (per-target out-of-domain). Schema `key_schema: model:method:target`, columns `auroc_mean`, `auroc_std`, `n_cells_used`. 2 × 4 × 6 = 48 rows.
   These are lower priority — they block the appendix, not §6.2.

7. **Method-name consistency.** §6 uses `Contrastive+Recon` as the in-prose label for ours (matching §5.1). The CSV `method` column carries `contrastive_logprob_recon` (matching the source CSV). Pending [`outline.md`](outline.md#L184) Open Question 7 (CLLMR vs. LPCR vs. CL²R), this label may change paper-wide — §6 inherits the §5 / §3 decision.

8. **MMLU-as-source interpretation (cross-section consistency).** §6.4 calls MMLU-as-source a *signal-level* limitation (no method extracts transferable signal from MMLU), not a *method* asymmetry. §5.1's MMLU footnote currently characterizes ACT-ViT's MMLU performance as a ViT-on-letter-token-activations limitation. The two framings are compatible — MMLU's format produces both fewer transferable features *and* a representation that ACT-ViT specifically does not handle well. §6.4 and §5.1 prose must not contradict each other on the MMLU mechanism; recommend one shared sentence in §8 that both forward to.

9. **Heatmap figure for Appendix C (Option A render).** If Option A is kept as 8 heatmap tiles, a new figure renderer is owed: `paper/figures_src/render_transfer_matrix.py`. Alternative: render Option A as a LaTeX table directly from `transfer_matrix.csv` without a figure renderer — cheaper. **Recommend the LaTeX-table route** for the appendix unless the human wants a colored heatmap figure for visual impact.

10. **Page-budget pressure (~0.5 page target).** Subsection sizing under current bullet density: §6.1 ≈ 0.10 page (text only), §6.2 ≈ 0.25 page (table + paragraph), §6.3 ≈ 0.10 page (table only), §6.4 ≈ 0.10 page (paragraph only). Headroom is tight — total ≈ 0.55 page. If §5 expands, the first §6 compression target is §6.3 — collapse to a single sentence forwarding aggregate numbers to a footnote or to Appendix C. §6.2 is load-bearing and cannot compress.

---

## Cross-references this outline expects to call

- §1 (introduction) — §6.2 headline is what §1 contribution-2 ("transfers across datasets without retraining") quotes. §1 verb upgrade (Open Q4) cannot lock before §6.2 / NQ closes.
- §2 (related work) — every method named in §6.2 has its citation grounded in §2; §6 cites on first §6 mention only.
- §3.2 (information-theoretic argument) — §6.4 touches the cross-layer-coherence framing as the *reason* the signal transfers. Does not re-derive.
- §3.3 (architecture + training recipe) — §6.1 names "ours" as `Contrastive+Recon` and forwards architectural detail.
- §4.3 (baselines) — every §6.2 row maps to a §4.3 bullet + config path. §6 does not re-describe baselines.
- §4.4 (training procedure) — §6 cells re-use the §5.1 trained probes; no retraining. Forward.
- §4.5 (metrics) — AUROC ± std definition lives in §4.5; §6 forwards.
- §5.1 (headline table) — §6.3 in-domain column *is* the §5.1 same-dataset diagonal. Consistency check is a hard requirement (Open Q3).
- §5.2 (headline figure) — §6 does not repeat. §6 reader is expected to have §5.2's visual in mind.
- §7 (ablations) — §6 does not pre-empt; if a §7.2 layer-pair sensitivity result implies a transfer interpretation, that lives in §7.2, not §6.4.
- §8 (discussion) — §6.4 forwards the MMLU-as-format-outlier and NQ-as-target shortfall stories to §8; §8 picks them up alongside the §5.1 NQ shortfall (Open Q8).
- §9 (limitations) — §6.4 forwards the "all six datasets are English short-form factual QA" scope to §9.
- Appendix C — Option A 6×6 heatmaps and Option C per-target breakdown live here; §6.2 + §6.3 forward.

---

## References — materialized for §6

§6 introduces **no new bibkeys** beyond those grounded in §2 / §4. Every method named in §6.2 / §6.3 (ours; ACT-ViT — `barshalom2025actvit`; LLMsKnow — pending-approval per §4; SAPLMA — `azaria2023internal`) is already named in §2 / §4 outlines and either present in `references.bib` or in the pending-approval queue.

No §6-specific citation candidates have surfaced. If a reviewer demands a domain-shift / OOD-generalization framing citation (e.g., Hendrycks et al., Recht et al. on robustness), surface to the human; do not insert proactively.

---

## Data dependencies — must exist before §6 prose locks

| File | Status | Companion task |
|---|---|---|
| NQ-as-source cells in `results/transfer_matrix_table.csv` | **MISSING (critical blocker)** | verify hole vs. unsubmitted dispatch; **do not dispatch without explicit user approval**. See Open Q1. |
| `paper/data/transfer_per_source.csv` | does not exist | author `scripts/aggregate_transfer.py`, emit per (model, method, source) out-of-domain mean. Schema in Open Q5. Primary §6.2 source. |
| `paper/data/transfer_summary.csv` | does not exist | same aggregator, emit (model, method) in-domain and out-of-domain means. Schema: `model:method:scope` (scope ∈ `{in_domain, out_of_domain}`), columns `auroc_mean`, `auroc_std`, `n_cells_used`, `n_cells_total`. Primary §6.3 source. |
| `paper/data/transfer_matrix.csv` | does not exist | same aggregator, full per-cell. Schema `model:method:source:target`, columns `auroc_mean`, `auroc_std`, `n_seeds`. Appendix C Option A. |
| `paper/data/transfer_per_target.csv` | does not exist | same aggregator, per-target. Schema `model:method:target`, columns `auroc_mean`, `auroc_std`, `n_cells_used`. Appendix C Option C. |
| `paper/figures_src/render_transfer_matrix.py` | does not exist *(optional)* | only needed if Appendix C Option A is rendered as colored heatmaps. Default plan: skip and use a LaTeX table. |

§6.2 is blocked on NQ + `transfer_per_source.csv`. §6.3 is blocked on `transfer_summary.csv`. Appendix C blocks on `transfer_matrix.csv` and `transfer_per_target.csv`. The heatmap renderer is optional.

**Note on the existing `paper/figures_src/render_transfer.py`.** That script is *not* a transfer-matrix renderer despite its name — it renders a `baseline_comparison.csv` bar chart and is unrelated to §6. Recommend renaming to `render_baseline_bars.py` to avoid confusion; track as a low-priority cleanup item.

---

## Drafting order recommendation

For a writing agent starting fresh on §6 prose:

1. **Block on NQ-as-source closure (Open Q1).** Per the 2026-05-21 human directive, §6 prose does not start until NQ data lands. Verify whether NQ-source cells are unsubmitted, failed, or already running before any dispatch request.
2. **Block on §5 numbers freeze.** Per [`outline.md:82`](outline.md#L82), do not start §6 prose until ACT-ViT 5/5 seeds close and `paper/data/headline_results.csv` lands. §5/§6 in-domain consistency (Open Q3) is a hard precondition.
3. **Author `scripts/aggregate_transfer.py`** (emitting all three §6 CSVs in one pass) and the `build_numbers.py` hook. Spot-check a `\result` call (e.g., Llama PopQA source OOD mean) against `results/transfer_matrix_table.csv` directly.
4. **Resolve Open Q2 (ACT-ViT coverage handling)** before drafting §6.2 prose — the table footnote depends on it.
5. **Resolve Open Q4 (§1 / abstract verb upgrade)** at the same time as §6.2 freezes — the §6.2 count is the verb's anchor.
6. **Draft §6.2 first** (per-source headline table + paragraph). Most reviewer-visible.
7. **Draft §6.1 second** (setup framing, 2–3 sentences). Depends on §6.2 to know what to forward to.
8. **Draft §6.3 third** (aggregate corroboration table). Smallest — table + caption only, no paragraph.
9. **Draft §6.4 last** (interpretation paragraph). Depends on §6.2 to know which sources to call out.
10. **Cross-check §6.3 in-domain column against §5.1 diagonal** as the last step before §6 commits. Any divergence is an aggregator-path bug — fix the aggregator, do not paper over with prose.
11. **Cross-check §1 contribution-2 verb, abstract transfer claim, §6.1 framing, §6.4 closer** for consistent transfer phrasing.
12. **Cross-check §6.4 MMLU framing against §5.1 MMLU footnote and §8 MMLU discussion** for the shared mechanism story (Open Q8).
