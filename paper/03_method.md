# Method — Outline

Structural skeleton **plus** fully materialized content. By the time a writing agent picks this up, every conceptual move that will appear in §3 is named here with claim, supporting argument, and the precise point our prose draws from. The writing agent should not need to research further; if a claim cannot be supported from the bullets below, surface it to the human rather than inventing one.

**Bib-policy reminder.** Bibkeys in the "References — materialized for §3" section below are already in `paper/references.bib`. Any inline cite marked `[PENDING-APPROVAL]` in the bullets below is **not** yet in `.bib` and must be human-approved before insertion (per `paper/references.bib` policy header and `CLAUDE.md`). The §3.2 citations `khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, and `hjelm2019deepinfomax` are **confirmed in `.bib` as of 2026-05-20** (verified against `references.bib`); `[PENDING-APPROVAL]` tags on these five in the bullets below are stale and should be treated as resolved.

**Structural decision (resolves [outline.md:161-163](outline.md#L161-L163) Open Question 1).** Theory is folded into §3 as §3.2, not promoted to a standalone §3 Theory section. Rationale: the load-bearing theoretical move is short (~half the section), it is structurally inseparable from the architecture (you cannot state "supervised contrastive on layer pairs" without first justifying why a contrastive loss on layer pairs is well-posed), and an 8-page EMNLP main paper does not have room for a dedicated theory section. The full information-bound derivation lives in Appendix A; §3.2 carries the argument in compressed prose.

**Target length: ~2 pages.** Slightly above the original §3 budget of 1.5–2 pages in [outline.md:45](outline.md#L45) because we are absorbing the theoretical framing rather than spinning it out. If the page budget tightens, §3.4 (the 2×2 prediction subsection) is the first cut — it can collapse into a single paragraph forwarding to §7 ablations.

**Supersedes** the structural skeletons in [methods_outline.md](methods_outline.md) and [theory_outline.md](theory_outline.md). Those files should be deleted after this outline is reviewed; they are preserved during the transition so the writing agent can sanity-check the merge.

---

## 3. Method

### 3.1 Problem setup and notation

**Framing sentence.** One paragraph stating the detection task, the access regime, and the symbols the rest of §3 will use. Should read fast — no theory yet, no architecture yet.

**Bullets — write each as 1–2 sentences in prose:**

- **Task.** Single-pass white-box hallucination detection: given a frozen language model `M`, a prompt `x`, and `M`'s greedy generation `y`, produce a scalar score `s(x, y) ∈ ℝ` such that higher scores correspond to hallucinated generations. The label `y_label ∈ {0, 1}` is defined operationally: `y_label = 1` when the benchmark's gold-reference evaluator marks the generation incorrect, `y_label = 0` otherwise — matching our codebase convention, which is the inverse of ACT-ViT's (`barshalom2025actvit` uses `y_i = 1` for correct). We follow the operational labeling of `barshalom2025actvit` and decline to commit to the intrinsic-vs-extrinsic taxonomy of `ji2022survey` / `bang2025hallulens`; we treat the taxonomic placement of this detection regime as orthogonal to the architectural argument in §3.2 and revisit it in §9.

- **Access regime.** Read-only access to `M`'s intermediate residual-stream activations during the same forward pass that generated `y`. No retrieval, no external verifier, no resampling — those define the comparison classes in §4 (linking forward to §2.2 / §4 baselines).

- **Notation.** `h_ℓ(x, y) ∈ ℝ^{T_y × d}` = full response-token hidden-state sequence at layer `ℓ` of `M`'s residual stream while reading `(x, y)`, with `L` total layers, `T_y` = number of response tokens, and `d` = hidden dimension (4096 for both supported 8B models). Define the extraction band `[ℓ_lo, ℓ_hi]` (motivated in §3.2 by the residual-stream evidence in §2.1, instantiated in §3.3). State that `h_ℓ` is deterministic given `(x, y)`; `y_label` may have evaluator noise (carried forward to §9 limitations). **Important distinction**: `h_ℓ` is a *sequence over response tokens*, not a single vector — the method consumes the full per-token activation trajectory at each layer, not the last-token activation alone. (This is load-bearing for §3.2's MI argument and §3.3's compressor design; previous drafts of this outline got it wrong.)

- **What we are *not* doing.** No multi-sample inference, no retrieval, no model modification. Single forward pass means the comparison axis with K=10 sampling methods (semantic entropy, SelfCheckGPT) is *compute*, not *signal access* — flagged here so §5.3 reads naturally.

**What §3.1 establishes for the rest of the paper.** The notation that §3.2 and §3.3 use, and the access-regime fact that §4 and §5.3 will repeatedly lean on (single-pass, white-box, no resampling).

---

### 3.2 Why a learned cross-layer compression can carry the signal — the information-theoretic argument

**Framing sentence.** The cleanest argument for why this could work is *not* a geometric story about activations; it is a feasibility argument grounded in mutual information. We can write down a well-posed contrastive objective on layer-pair views because cross-layer mutual information is positive; the empirical fact that the loss is trainable on our data is itself the proof that the argument is non-vacuous.

This subsection makes three coupled claims and one punchline.

**Bullets — write each as 1–3 sentences in prose:**

- **Claim 1 — cross-layer mutual information is positive by construction.** For any two layers `ℓ_a < ℓ_b` in `M`'s residual stream and a fixed response, the per-token activation sequence `h_{ℓ_b} ∈ ℝ^{T_y × d}` is a deterministic function of `h_{ℓ_a}` plus the contributions of intermediate residual blocks, with residual connections explicitly preserving information from earlier layers into later ones. By the data-processing inequality this gives `I(h_{ℓ_a}; h_{ℓ_b}) > 0` for any input distribution that varies non-trivially across `h_{ℓ_a}`. The MI here is between *sequence-valued* objects — full per-token activation tensors at two layers, not single-token vectors — which makes the architectural argument *stronger*, not weaker (more shared information channels, not fewer). **A stronger architectural form is available** (Corollary A.2.1): under the same downward-determinism assumption, the three-way interaction information collapses to `I(h_{ℓ_a}; h_{ℓ_b}; y) = I(h_{ℓ_b}; y) > 0`, i.e., the cross-layer shared information also overlaps with the response variable `y` — the joint is non-orthogonal to what was generated. **Important caveat:** this is still architectural and tells us only that the joint carries *some* shared information about `y` — it says nothing about whether that information is *relevant to hallucination*. `y` is the response tokens, not the hallucination label `y_label`; `h` at any single layer is a *lossy* compression of `(x, y)`, so Corollary A.2.1 upper-bounds `I(h_{ℓ_a}; h_{ℓ_b}; y_label)` but does not lower-bound it. Whether the label-relevant features survive the compression is the empirical claim, owned by Claim 2 and the §3.2 trainability punchline.

- **Claim 2 — InfoNCE is a variational lower bound on mutual information; supervised InfoNCE bounds a *label-conditioned* MI.** State the standard InfoNCE bound `-L_InfoNCE ≤ I(view_a; view_b) - log K` from `oord2018cpc` (already in `.bib`) as the loss family we use, then immediately specialize to the supervised variant — `khosla2020supcon` [PENDING-APPROVAL] — that we actually run. Under SupCon with the asymmetric `ignore_label=1` scheme described in §3.3, the optimized lower bound becomes a one-sided bound on the mutual information between the learned features and the truthful class (`y_label = 0`): the loss pulls truthful-class features into a coherent cluster while hallucinated-class features are only required to be view-consistent with themselves. The cleaner formal version of "InfoNCE bounds mutual information" comes from the tighter Poole et al. analysis (`poole2019variational` [PENDING-APPROVAL]); the alignment/uniformity decomposition (`wang2021understanding` [PENDING-APPROVAL]) gives a complementary lens on what the loss optimizes geometrically. All three citations are pending-approval and the prose hedges accordingly until they land.

- **Claim 3 — the label-relevant subspace of the cross-layer MI is what we need.** Claim 1 establishes that the joint `(h_{ℓ_a}, h_{ℓ_b})` carries shared information; most of it (e.g., the prompt's surface form, the topic, the token-position embedding) is irrelevant to whether `y` is hallucinated. The supervised contrastive objective in Claim 2 selects the *label-relevant* slice of that shared information by making the truthful-class views collapse to a coherent cluster while leaving hallucinated views to repel everything. This is the structural argument for why supervised contrastive on layer pairs is a sensible extractor — not a guarantee of empirical superiority, just a guarantee that the construction targets the right slice of the joint.

- **The punchline — feasibility is established by trainability.** Everything in Claims 1–3 is in principle vacuous if `I(h_{ℓ_a}; h_{ℓ_b})` were zero or if the label-relevant subspace were null. The cleanest empirical confirmation that it is *not* vacuous is the simplest possible one: **our loss descends.** The contrastive loss bounds `I(z_a; z_b)` where `z = f_θ(h)` are the compressed per-layer embeddings, and the data-processing inequality gives `I(z_a; z_b) ≤ I(h_{ℓ_a}; h_{ℓ_b})` — so any positive MI observed at the embedding level is direct evidence of positive MI at the activation level (the DPI direction works in our favor). If `I(view_a; view_b) = 0`, no InfoNCE variant — supervised, unsupervised, or otherwise — could descend, because the lower bound is `-log K`, hit by uniform negatives. The fact that training-loss curves drop and the embeddings cluster meaningfully is the operational proof that the cross-layer MI we are trying to extract is positive and the supervised slice we are trying to isolate is non-empty. **One figure carries this entire argument** — the training-loss curve from a representative full-method run (Figure 1 of §3; sourced from existing logs, no new compute). The prose should land this hard: *the existence of our learned probe is itself the empirical content of the feasibility theorem*.

- **What this argument does NOT claim — and what it does predict.** Three honest negatives plus one confirmed prediction the prose must surface:
  1. It does not claim the *contrastive* subspace beats every other extractor. That is an empirical question §5 answers.
  2. It does not claim cross-layer views are uniquely good (vs., e.g., same-layer + dropout views or augmented input views). The layer-pair sensitivity ablation (§7.3) is what speaks to which layer pairs carry the signal; the choice of "different layers" as views over "augmented inputs" is partly justified by the residual-stream-as-truth-carrier evidence in §2.1 (`li2023iti`, `marks2024geometry`) and partly an empirical bet — we say so.
  3. It does not predict the *magnitude* of the AUROC gain. That is set entirely by the data, not by the bound.
  4. **It does predict the symmetric SupCon failure mode** — and §7.3 Variant 4 confirms it. Corollary A.1.1 (see Theorems section) states: when class 1 has no coherent latent cluster structure, symmetric SupCon introduces gradient noise negatively correlated with the class-0 gradient. The ~10-point AUROC drop from Variant 4 (both classes labeled) is a direct empirical confirmation of this prediction. Present this positively in §3.2 prose: the theory makes a falsifiable prediction, and the ablation confirms it.

**What §3.2 establishes for the rest of the paper.** A short, falsifiable feasibility argument that the method is well-posed (Claims 1–2 architectural + analytical, Punchline empirical-from-trainability), with the geometric story explicitly demoted to "empirical question" so §7 ablations read as confirming/probing rather than rescuing the theory. Reader exits §3.2 knowing exactly what is theoretically guaranteed (positive MI, label-relevant slice well-targeted) and what is empirically owed (which layers, what scorer, how much signal).

---

### 3.3 What we built — supervised contrastive over layer pairs with logprob reconstruction

**Framing sentence.** This subsection turns the feasibility argument into concrete artifacts: the activations we extract, the architecture that compresses them, the loss that pulls the label-relevant signal out, and the scorer that turns a learned embedding into a hallucination score.

**Bullets — each as 1–2 sentences in prose, except the loss bullet which gets a paragraph:**

- **Activation extraction.** From each `(x, y)`, we extract `{h_ℓ(x, y) ∈ ℝ^{T_y × d} : ℓ ∈ [ℓ_lo, ℓ_hi]}` — the **full per-token response activation sequence at every layer in the band**. No last-token reduction at extraction time; the entire trajectory `t = 1..T_y` is retained per layer and per generation. Layer band is mid-to-late residual stream — the band justified in §2.1 by `li2023iti` and `marks2024geometry`. State the exact `[ℓ_lo, ℓ_hi]` and the per-(model, dataset, split) memmap cache once; downstream training reuses cached tensors and no re-extraction across seeds. The decision to keep the full response sequence rather than reducing to a single token is load-bearing — it gives the compressor a sequence to attend over (§3.3) and gives the recon target a per-token logprob signal to regress against (§3.3 recon bullet).

- **The compression network `f_θ`.** A `ProgressiveCompressor` ([`activation_research/model.py:39-103`](../activation_research/model.py#L39-L103)): a **shared transformer-encoder stack with progressive dimensionality reduction**, applied independently to each layer's per-token activation sequence. Concretely: input `(T_y, 4096)` → optional input `LayerNorm` (when activation magnitudes vary across layers, e.g. layer 14 ≈ 50 vs. layer 29 ≈ 200) → input dropout → sinusoidal positional encoding → stack of `TransformerBlock`s (8-head encoder layers with `ff_multiplier=4`) that halve the hidden dim at each block (`4096 → 2048 → 1024 → 512`) → **mean-pool over the response-token dimension `T_y`** → linear projection → output `z ∈ ℝ^{512}` per (generation, layer). One critical structural fact: the compressor is **shared across all layers** (one `f_θ`, not per-layer heads); the same module produces both views of the same generation, ensuring the layer pair is the *only* source of view asymmetry. Lock the exact spec (block count, exact dim schedule, dropout values) in Appendix B. Foreshadow §5: at matched parameter count we beat SAPLMA's 11M-param MLP probe, ruling out "more capacity helps" as the explanation.

- **View construction (layer pairs).** Each training example yields two views by sampling a layer pair `(ℓ_a, ℓ_b)` from the cached band per minibatch ([`activation_research/memmap_contrastive_dataset.py:99-130`](../activation_research/memmap_contrastive_dataset.py#L99-L130) — `num_views=2` typical, `relevant_layers` restricts to the band, `view_sampling_with_replacement` controls duplicates). Each view = one layer's full response activation tensor `(T_y, 4096)` → `f_θ` → one 512-dim embedding. The two views of a generation are therefore two compressed-embedding vectors, each summarizing the full per-token activation sequence at one layer of `M`. State the sampling policy (uniform across the band? fixed early-late pair? scheduled?) once here and defer the sensitivity sweep to §7.3. The structural novelty is exactly the move §2.3 stakes out: the positive views are *different layers* of the same un-augmented forward pass, not data augmentations and not different networks. Note: the dataset supports `num_views > 2` (`_contrastive_collate_kview`), but the headline method runs with two views; K-view fusion is in the legacy ideas pool, not the submission.

- **The supervised contrastive loss with asymmetric `ignore_label` — load-bearing detail, do not gloss.** The loss is supervised contrastive (SupCon, `khosla2020supcon` [PENDING-APPROVAL]) with one critical asymmetry implemented as `use_labels: true, ignore_label: 1` in [`configs/methods/contrastive_logprob_recon.json`](../configs/methods/contrastive_logprob_recon.json) and [`activation_research/training.py:206-272`](../activation_research/training.py#L206-L272):
  - **Truthful class (`y_label = 0`).** All other truthful samples in the batch are positives for the anchor; all hallucinated samples are negatives.
  - **Hallucinated class (`y_label = 1`).** Positives are *only* the anchor's own other layer-views (i.e., views of itself); everything else in the batch — including other hallucinated samples — is a negative.
  - **Why this and not symmetric SupCon.** The key contraindication is not "hallucinations are not a coherent class" (too strong — §7.3 Variant 3 shows the hallucinated side can serve as the one-class anchor nearly as well as the truthful side). The correct framing: **labeling both classes simultaneously forces a second coherent cluster that the data does not support**. Symmetric SupCon's gradient includes a class-1 term pushing hallucinated embeddings toward each other; when class-1 has no latent coherent structure, this gradient is noise that actively fights the class-0 term. The §7.3 Variant 4 ablation confirms this at -10 AUROC points (predicted by Corollary A.1.1). The asymmetric scheme is therefore a *structural* choice — one-class representation learning with the labeled class as the inlier anchor — not an arbitrary implementation preference. Either class can serve as the anchor (Variant 3 ≈ Variant 2 ±1 pt); what is contraindicated is labeling both. The prose must spend ~3–4 sentences on this, not one, because it is the load-bearing methodological choice.
  - **What this is structurally.** A representation-learning form of one-class anomaly detection on truthful samples, with cross-layer view-consistency providing the second source of supervision for both classes. This framing is what we owe the reader; it is also what makes §3.2's "MI between features and the truthful class" specialization the correct theoretical lens.

- **Logprob-reconstruction auxiliary loss.** From the compressed feature `z = f_θ(h_ℓ) ∈ ℝ^{512}`, a **two-layer MLP decoder** `g_φ: ℝ^{512} → ℝ^{R}` with one hidden layer of width 256 and GELU activation (`512 → 256 → 64`, [`activation_research/model.py:172-176`](../activation_research/model.py#L172-L176)) predicts a **resampled fixed-length proxy for the per-token logprob sequence** of `y` under `M`:
  ```
  L_recon = MSE( g_φ(z),   resample( (log p_M(y_t | y_{<t}, x))_{t=1..T_y},  length = R )  )
  ```
  with `R = 64` ([`model.py:153`](../activation_research/model.py#L153)) and *resample* = linear interpolation (`F.interpolate(..., mode="linear")`, [`model.py:233-239`](../activation_research/model.py#L233-L239)). The fixed reconstruction length is an *architectural* choice (decoder output is `R=64`-dim) — the per-token logprob sequence is variable-length and is interpolated to match. **Variance-suppression mechanism** ([`model.py:225-229`](../activation_research/model.py#L225-L229)): when the batch-level variance of the logprob target falls below `1e-4`, the recon term is set to zero for that batch and a diagnostic flag is raised. This prevents the decoder from wasting capacity on near-constant logprob sequences (which would otherwise pull `f_θ` toward features irrelevant to the label). **Why this target.** When `M` is about to hallucinate, the next-token distribution typically broadens — top-token probability drops at the moment of fabrication — so per-token logprobs are causally connected to the hallucination event we want to detect. The recon loss therefore applies regression-style supervision on `f_θ` to retain features that suffice to predict a signal correlated with the label, complementary to (but not redundant with) the classification-style supervision the contrastive loss provides. **What this is *not*.** It is not a sufficient signal on its own — see §3.4 — and it is not the same as SEP (`kossen2024semantic`); SEP regresses a single semantic-entropy *scalar* per generation, we regress a *length-`R` resampled per-token logprob sequence* on a learned representation that already aggregates the full response activation tensor.

- **Total loss.**
  ```
  L_total = L_SupCon-asymm + λ · L_recon
  ```
  State the `λ` value and the temperature `τ` once; full hyperparameter table in Appendix B. Note the prose-level point that `λ` is *not* a knob we tuned heavily — it was set early and left alone — so the §7.1 loss-decomposition ablation is the meaningful sensitivity analysis, not a `λ` sweep.

- **Inference scoring.** Given a trained `f_θ` and a held-out `(x, y)`, compute `z = f_θ(h_ℓ(x, y))` and score via one of three scorers:
  - **KNN (headline scorer).** Distance to the `k`-th nearest neighbor in the training-set truthful-class embedding bank. State `k`, the distance metric, and the bank construction.
  - **Cosine (ablation).** Mean cosine similarity to the training-set truthful centroid.
  - **Mahalanobis (ablation).** Distance to the training-set truthful-class Gaussian under the pooled covariance.

  KNN as headline is an empirical choice from seed-0 evidence (§7.4 reports the comparison); cosine and Mahalanobis are reported as ablation rows. All three are inference-only post-hoc scorers on the same learned `f_θ` — they do not require retraining.

- **Implementation summary.** One paragraph: optimizer, learning rate, batch size, training epochs, `τ`, `λ`, weight decay, seed list (5 seeds {0, 5, 26, 42, 63}, split seed 42). Per-(model, dataset) training: train split only, no cross-dataset training (transfer is reported separately in §6). Reproducibility: cached activations at known paths; configs in `configs/experiments/baseline_comparison_*_memmap.json`; training entry point [`scripts/run_experiment.py`](../scripts/run_experiment.py). Single 80GB GPU suffices for one full cell once activations are cached. Total GPU-hours table in Appendix E.

**What §3.3 establishes for the rest of the paper.** A concrete, reproducible specification of the method that §4 baselines plug into and §5 / §6 / §7 evaluate. The load-bearing methodological detail — *asymmetric SupCon with `ignore_label`* — is named explicitly, so §7.1 (loss decomposition) reads as a planned attribution study rather than a defensive ablation.

---

### 3.4 What the method predicts — the 2×2 attribution table

**Framing sentence.** §3.3 names two distinct supervision channels (supervised contrastive on layer pairs, logprob-recon auxiliary). §3.4 states the 2×2 ablation matrix that attributes the empirical gain to one, the other, or both, and pre-commits to the four outcome-conditional framings. The numbers themselves live in §7.1; this subsection is the conceptual scaffold.

**The 2×2 cells:**

|                           | **No recon**                              | **With recon**                             |
|---------------------------|-------------------------------------------|--------------------------------------------|
| **No SupCon-asymm**       | SAPLMA (`azaria2023internal`)             | **#67** SAPLMA + logprob-recon *(pending)* |
| **With SupCon-asymm**     | **#66** SupCon-asymm only *(pending)*     | **Full method** (headline, in §5)          |

Plus two reference rows the prose must mention:
- **Linear probe** (single layer, no SupCon, no recon, no MLP head) — already in grid; the floor.
- **Unsupervised SimCLR** (`use_labels: false`, no recon) → AUROC ≈ 0.5 — footnote row; demonstrates that *label-free* cross-layer contrastive does not surface the hallucination axis. Important framing point: this rules out *self-supervised* cross-layer contrastive as an extractor, but **does not** by itself rule out supervised-contrastive-alone. That stronger claim is #66's job to test.

**Outcome-conditional framings — the §3.4 prose locks to one once #66 and #67 land:**

- **Outcome 1: #66 < full, #67 < full.** Best case. Both ingredients contribute independently and the combination is irreducible. Frame: "supervised contrastive on layer pairs and logprob recon are independently informative and synergistic."

- **Outcome 2: #66 < full ≈ #67.** Recon is the load-bearing piece; SupCon-asymm contributes modestly on top. Frame honestly: "the logprob-recon auxiliary trick is the primary contribution; the contrastive structure is a useful auxiliary." This would re-position the paper relative to SEP (`kossen2024semantic`) — recon-as-causal-auxiliary becomes the closest neighbor and the contribution narrows to "recon on a deeper architecture with cross-layer structure." See §9 limitations / risk register.

- **Outcome 3: #66 ≈ full > #67.** SupCon-asymm is the load-bearing piece; recon is decorative on top of it. Frame: "the supervised contrastive structure with asymmetric `ignore_label` carries the signal; recon adds little when SupCon-asymm is present." Strongest pitch for the "view definition + objective" novelty claim in §2.3 (broad).

- **Outcome 4: #66 ≈ #67 ≈ full.** Both ingredients are individually sufficient. Frame: "two distinct supervision signals — contrastive cluster formation and per-token logprob regression — each independently recover the signal, and the architecture is robust to which one is supplied." Most informative for the field; arguably the *most* desirable outcome theoretically even if it weakens the "irreducible combination" pitch.

Until #66 and #67 land, the §3.4 prose presents the *menu* with a clear declaration that the headline framing will be selected when data arrives. **The headline claim of §3.4 must not be locked before the ablations resolve** — drafting around all four outcomes is the writing discipline this subsection enforces.

**What §3.4 establishes for the rest of the paper.** A pre-committed attribution structure that constrains §7.1 (loss decomposition) writeup and §5 (headline framing). Reader exits §3 knowing what the empirical contribution *will* be, conditional on the four data-driven scenarios.

---

## Theorems to be added

§3.2 invokes a small, finite set of formal results. Most are off-the-shelf and cite-only; one is genuinely ours and needs a formal statement in Appendix A. This section catalogs both so the writing agent and human reviewer can audit the theoretical surface at a glance.

**Inline in §3.2 body (cite-only, no derivation):**

- **T1 — Data-Processing Inequality (DPI).** Standard, Cover & Thomas. Used twice in §3.2:
  - *Forward direction* (Claim 1): `h_{ℓ_b}` is a deterministic function of `h_{ℓ_a}` plus the contributions of intermediate residual blocks, so `I(h_{ℓ_a}; h_{ℓ_b}) > 0` for any non-degenerate input distribution.
  - *Backward direction* (Punchline): `I(z_a; z_b) ≤ I(h_{ℓ_a}; h_{ℓ_b})` where `z = f_θ(h)`. Positive MI at the embedding level ⇒ positive MI at the activation level.
  - No new statement owed; one inline citation in §3.2 suffices. Lifted to **Lemma A.2** for a clean restatement in our notation so the punchline can cite by number.

- **T2 — InfoNCE variational lower bound on MI.** From `oord2018cpc`, tightened in `poole2019variational`:
  ```
  -L_InfoNCE ≤ I(view_a; view_b) - log K
  ```
  This is the lemma that powers the §3.2 trainability punchline (loss descends below `-log K` ⇒ `I(view_a; view_b) > 0`). Cite `oord2018cpc` for the bound, `poole2019variational` for the tightness analysis. Lifted to **Lemma A.3** in our notation so Proposition A.1 has something to plug into.

- **T3 — (Optional) Alignment / Uniformity decomposition.** `wang2021understanding`. Cite-only, supports Claim 3's geometric lens. No formal restatement needed; drop if §3.2 page budget binds.

**Formal statement in Appendix A (ours, needs proof):**

- **Proposition A.1 — Asymmetric SupCon bounds label-conditioned MI on the truthful class.** *The only result in the paper that is not off-the-shelf.* Combines (a) the SupCon → MI bound from `khosla2020supcon` with (b) the `ignore_label=1` asymmetry implemented in [`activation_research/training.py:206-272`](../activation_research/training.py#L206-L272) and configured in [`configs/methods/contrastive_logprob_recon.json`](../configs/methods/contrastive_logprob_recon.json), and claims:
  - The optimized lower bound under the asymmetric scheme becomes **one-sided**: a bound on `I(z; y_label = 0)` (features ↔ truthful class), not on `I(z; y_label)` symmetrically.
  - Informally: pulling truthful-class features into a coherent cluster while leaving hallucinated views only required to be view-consistent with themselves is exactly what one-class anomaly detection on the truthful inlier class corresponds to in the MI lens.
  - This is the formal counterpart to the §3.3 "asymmetric `ignore_label`" load-bearing detail and the §3.2 Claim 2 specialization.
  - **What the statement must pin down precisely**: (i) batch sampling assumptions, (ii) view-construction protocol (layer-pair sampling from `[ℓ_lo, ℓ_hi]`), (iii) `ignore_label` semantics (anchor with `y_label=1` has positives = own other-layer views only; everything else is negative), (iv) the form of the bound and the constant absorbed into `log K`.
  - **Loss decomposition that the proof hinges on.** Split the total SupCon-asymm loss by anchor class:
    ```
    L_total = L_truth-anchors + L_hall-anchors
    ```
    - `L_truth-anchors` is the SupCon loss restricted to truthful anchors (positives = other truthful samples + own other-layer views).
    - `L_hall-anchors` is **instance-level** InfoNCE for hallucinated anchors (positives = own other-layer views only; every other sample, including other hallucinated samples, is a negative).
    This decomposition is what makes the bound one-sided — the two terms specialize to bounds on *different* quantities.
  - **Proof sketch (three steps).**
    1. **Start from Lemma A.3** (InfoNCE → MI): `-L_InfoNCE ≤ I(view_a; view_b) - log K`.
    2. **Apply Khosla's class-grouping specialization to `L_truth-anchors`.** Grouping positives by class label replaces the InfoNCE numerator's instance-level similarity with a class-averaged similarity; combined with DPI on the class variable, this tightens the bound into a bound on `I(Z; 𝟙[y_label = 0])` — features distinguish "in the truthful cluster" from "everything else." This step is verbatim from Khosla et al., restricted to the truthful subset.
    3. **Show `L_hall-anchors` reverts to instance-level InfoNCE.** Because the positive set for a class-1 anchor is `{own other-layer views}` only, the InfoNCE numerator for these anchors stays instance-level. The bound therefore specializes to `I(z; view-pair-identity)` for hallucinated anchors — i.e., view-consistency for individual hallucinated generations — and contributes **no** bound on `I(Z; 𝟙[y_label = 1])` because no class-grouping occurs on the hallucinated side.
    4. **Sum.** The total loss lower-bounds `I(Z; 𝟙[y_label = 0])` (from step 2) plus an instance-level view-consistency term (from step 3). The class-1 side carries no class-MI term — the bound is one-sided.
  - **What the proof does *not* establish.** The bound is not claimed tight (Poole et al. give the looseness analysis); the bound does not imply sufficiency for hallucination detection (the §3.2 trainability punchline handles sufficiency empirically); the `L_hall-anchors` term's role as a regularizer that prevents hallucinated-class feature collapse is empirical, not bound-implied. Surface all three honestly in the Appendix A discussion paragraph after the proof.

- **Corollary A.1.1 — Symmetric SupCon is contraindicated when one class lacks coherent latent structure.** Follows directly from the Proposition A.1 proof without new machinery. Statement: when the class-1 positive set under symmetric SupCon has near-uniform pairwise similarity (i.e., intra-class InfoNCE numerator terms ≈ inter-class terms), the class-1 gradient approaches zero in expectation but retains a residual component pointing *away* from the truthful cluster (because the denominator includes class-0 terms). The net effect is gradient noise negatively correlated with the class-0 gradient, bounded by the class-1 loss mass times the class-imbalance ratio. **Confirmed empirically:** §7.3 Variant 4 drops ~10 AUROC points vs. Variant 2 (both models, early results 2026-05-20). Proof: ~2 paragraphs following the A.1 loss decomposition, showing that the symmetric scheme does not reduce to the asymmetric one under any hyperparameter choice — the class-1 positive-set construction is what differs, not a scale factor.

- **Lemma A.2 — DPI chain in our notation.** Clean restatement of T1 so §3.2 prose can cite by number. ~3 lines plus a one-sentence proof reference.

- **Corollary A.2.1 — Cross-layer/response co-information under downward determinism.** Follows directly from Lemma A.2 applied to the response variable. Statement: when `h_{ℓ_b}` is a deterministic function of `h_{ℓ_a}` given the prompt (residual-stream determinism), the three-way interaction information satisfies
  ```
  I(h_{ℓ_a} ; h_{ℓ_b} ; y)  :=  I(h_{ℓ_a} ; h_{ℓ_b})  -  I(h_{ℓ_a} ; h_{ℓ_b} | y)  =  I(h_{ℓ_b} ; y),
  ```
  which is strictly positive whenever `h_{ℓ_b}` is a non-constant function of `y` — i.e., for any non-degenerate response distribution. **Proof (two lines).** Downward determinism gives `H(h_{ℓ_b} | h_{ℓ_a}) = H(h_{ℓ_b} | h_{ℓ_a}, y) = 0`, so `I(h_{ℓ_a}; h_{ℓ_b}) = H(h_{ℓ_b})` and `I(h_{ℓ_a}; h_{ℓ_b} | y) = H(h_{ℓ_b} | y)`; substituting into the interaction-information definition gives `H(h_{ℓ_b}) - H(h_{ℓ_b} | y) = I(h_{ℓ_b}; y)`. **What this strengthens.** Claim 1 of §3.2 states only that the cross-layer joint carries shared information; the corollary ties that shared information to the response variable as well — the joint is non-orthogonal to `y`. **Surviving caveat (two sentences).** Under the operational definition `y_label = g_evaluator(x, y)` — deterministic in `(x, y)` for a fixed evaluator — the corollary gives only an *upper bound* on `I(h_{ℓ_a}; h_{ℓ_b}; y_label)`, not a lower bound. Whether the label-relevant features survive the lossy compression that `h` applies to `(x, y)` at any single layer is an empirical question, owned by Claim 2 and the trainability punchline. **Role in §3.2 prose.** Lets Claim 1 cite by number for the three-way statement instead of re-deriving in prose; the §3.2 paragraph stays tight while the formal version lives in the appendix.

- **Lemma A.3 — InfoNCE bound restated.** Poole-tightened bound in our notation. Plugs into Proposition A.1's proof.

**What §3.2 explicitly does NOT claim, and therefore does NOT owe a theorem for** (mirrors the "What this argument does NOT claim" bullet at line 51 of §3.2):

- *Sufficiency.* No claim that the label-relevant subspace is non-empty as a theorem — that is declared **empirical-by-trainability** in the punchline. The trainability of the loss is the operational proof; no separate sufficiency theorem is owed.
- *Uniqueness of layer-pair views.* No claim that different-layer views are theoretically uniquely good vs. augmented-input views or same-layer+dropout views. §7.3 ablation speaks empirically.
- *Magnitude.* No theorem about the size of the AUROC gain. The bound is qualitative (`> 0`), not quantitative.

**Risk: reviewer asks for a sufficiency theorem.** A reviewer may push for a formal statement of "the label-relevant slice of the cross-layer MI is non-empty under assumption X." We do not have such a theorem and the §3.2 strategy is to refuse it by appealing to the trainability punchline. If §3.2 is judged "long on feasibility, short on sufficiency" at internal review, the fallback is to add an *Assumption A.4* (non-degenerate label conditional distribution over activations) and an existence-style corollary — but only as a defensive add, not on first draft.

**Drafting order for Appendix A**: A.2 → A.3 → A.1 (lemmas before the proposition that uses them). Total budget ≤ 1 page; Proposition A.1 is the only one whose proof needs more than a few lines.

---

## Open questions for §3

These need explicit decisions before §3 prose finalizes. Some are pending data, some are pending citations, some are pending an authorial choice.

1. ~~**The three pending-approval citations in §3.2.**~~ **Resolved 2026-05-20** — `khosla2020supcon`, `poole2019variational`, `wang2021understanding`, `tishby2015deep`, and `hjelm2019deepinfomax` are all confirmed present in `references.bib`. The §3.2 MI argument can be drafted at full fidelity. `[PENDING-APPROVAL]` tags on these five in the bullets above are stale.

2. **#66 and #67 outcomes — drives the §3.4 framing.** Tracked in roadmap §2 status table. Until both land, §3.4 prose is locked to the *menu* of four outcomes; the headline framing of §3 (and the abstract) cannot finalize. If the GPU pipeline slips and these don't land by week 3, the fallback is to ship §3.4 as the menu and add a single sentence in §5 saying "the attribution between channels is left to the journal extension" — explicitly acknowledged in §9 limitations. This is a real risk.

3. **Layer-pair sampling policy.** The current implementation samples pairs uniformly across the band (verify); other reasonable choices include fixed early-late pairs, scheduled (curriculum) sampling, or a learned attention over layers. §7.3 sweeps a coarse grid; we do not commit to a sweep over the *sampling policy* itself, only the sampled pair. The prose must state what we do and decline to re-justify it as anything more than an empirical choice; the §7.3 result is what speaks.

4. **Whether the asymmetric `ignore_label` is itself the contribution, distinct from "SupCon on layer pairs."** **Partially resolved by early §7.3 results (2026-05-20).** Early data shows {Variant 2 ≈ Variant 3} >> Variant 4 > Variant 1, meaning: (a) the *direction* of the one-class anchor does not matter — either class works; (b) labeling *both* classes simultaneously hurts badly (-10 AUROC); (c) the contribution is **one-class contrastive learning on layer pairs** (keeping one class as the unlabeled outlier role), not specifically "truthful side as the inlier." The revised positioning answer: the **application of one-class contrastive structure to layer-pair views of a frozen LLM** is the contribution; the **`ignore_label` asymmetry** is the specific implementation of that structure; the **direction of the anchor** (truthful vs. hallucinated) is a secondary empirical choice. §3.3 prose should lead with the structural claim ("one-class contrastive, not symmetric SupCon") and name the direction as a secondary choice. Symmetric SupCon failure (-10 pts) is the stronger claim and is now a confirmed prediction of Corollary A.1.1 — use it as the primary evidence in §3.3's "why this and not symmetric SupCon" paragraph.

5. **The recon target choice — resampled per-token vs. sequence-level.** What we actually regress (verified against [`model.py:225-242`](../activation_research/model.py#L225-L242)) is a **linearly-interpolated resampling of the per-token logprob sequence to fixed length R = 64**, decoded from `z ∈ ℝ^{512}` via a 2-layer MLP. The decoder output dim `R` is an architectural hyperparameter; sequence-level scalar regression (R = 1, closer to `kossen2024semantic`) and unresampled variable-length regression are both reasonable alternatives we have not ablated. Decide: defend R = 64 in §3.3 prose as "a fixed-length proxy that preserves the temporal-broadening signal at the moment of fabrication" — and surface the R-as-hyperparameter and the linear-interpolation choice in Appendix B without claiming either is principled. Add R-sweep as a *future-work* sentence in §9 — not as an ablation we owe. **Additional load-bearing detail surfaced by the code scan:** the variance-suppression mechanism (recon loss → 0 when `var(logprob_target) < 1e-4`) is implementation-correct and worth a sentence in §3.3 because it prevents the auxiliary loss from over-fitting to near-constant sequences. Open whether reviewers will see the suppression as a defensible engineering choice or as a knob hiding a failure mode; on current evidence, it appears to be the former. Verify with the writing agent before locking the framing.

6. **Page budget if §3.4 stays full-fat.** §3.1 + §3.2 + §3.3 + §3.4 at the depth above is likely 2 to 2.25 pages of prose. The §3 target in [outline.md:45](outline.md#L45) is 1.5–2 pages. If the page budget binds, §3.4 collapses first: state the 2×2 in a sentence and a table, forward the four outcomes to §7.1, drop the per-outcome framings from §3 and surface them only in §7.1. Decide after §5 numbers freeze and the rest of the page budget is concrete.

7. **Figure 1 — the training-loss curve as the empirical content of §3.2's feasibility punchline.** The argument leans hard on "trainability is the proof." That argument is much stronger with a visible loss curve in the section than it is in prose alone. The figure exists in training logs (e.g., the seed-0 full-method run on HotpotQA); we owe ~30 min of work to locate it, plot it cleanly, and caption it. Decide whether Figure 1 lives in §3.2 or only in the appendix; recommend in §3.2 because the argument is much less load-bearing without it.

8. **Theory section as §3 vs. its own §3.** This outline resolves it as folded-into-§3 (per the structural-decision header). If a reviewer pushes back at submission time ("the theoretical contribution is buried"), the fallback is to promote §3.2 to its own §3 Theory section before submission — a structural lift that takes ~1 hour. The structural decision header marks this as a soft commit, revisable post-internal-review.

---

## Cross-references this outline expects to call

- §2.1 (activation probing) — §3.2's appeal to "the residual stream carries the truth signal" leans on `li2023iti` and `marks2024geometry` cited there. §3 should not re-cite at length; one back-reference is enough.
- §2.3 (contrastive representation learning) — §3.2's specialization to SupCon and §3.3's view-construction discussion both refer back to the §2.3 framing of the novelty claim.
- §4 (experimental setup) — §3.3 specifies the per-cell training procedure; §4 enumerates the cells. Avoid duplication: §3 cites the *method*, §4 cites the *grid*.
- §5.3 (compute-matched comparison) — §3.1's single-pass framing must seed §5.3 so the compute axis does not appear cold.
- §7.1 (loss decomposition ablation) — §3.4's 2×2 menu is paid off in §7.1. Forward reference is explicit.
- §7.3 (layer-pair sensitivity) — §3.3's layer-pair sampling policy and §3.2's "doesn't claim layer pairs are uniquely good" both punt to §7.3 for the empirical sweep.
- §7.4 (scorer choice) — §3.3's KNN-as-headline-by-empirical-choice forwards to §7.4 for the cosine/Mahalanobis/KNN comparison.
- §9 (limitations) — §3.2's three "does not claim" honest negatives connect to §9; §3.4 outcomes 2 and 3 also have §9 implications.
- Appendix A — §3.2's information-bound argument has its full form here.
- Appendix B — §3.3's hyperparameter table and full `ProgressiveCompressor` spec.

---

## References — materialized for §3

Every entry below is already in `paper/references.bib`. Bibkey, citation context, and *role each reference plays in our §3 prose* are stated so the writing agent does not need to re-research.

- **`oord2018cpc`** — van den Oord, Li, Vinyals, "Representation Learning with Contrastive Predictive Coding," arXiv:1807.03748, 2018.
  - **What they did.** Introduced InfoNCE as a contrastive lower bound on mutual information between two views.
  - **Why we cite in §3.** This is the *loss family* §3.2 invokes for the MI lower-bound argument, and §3.3 instantiates as the supervised variant. One citation in §3.2 (when the bound is first stated), one back-reference in §3.3 (when the loss form is written).
  - **One-sentence role in our prose.** "InfoNCE (`oord2018cpc`) is a variational lower bound on the mutual information between two views; our loss is the supervised specialization of it."

- **`chen2020simclr`** — Chen, Kornblith, Norouzi, Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.
  - **What they did.** Canonical two-augmented-views contrastive recipe for visual representations.
  - **Why we cite in §3.** Reference point for "vanilla SimCLR" in the unsupervised-ablation footnote row in §3.4 / §7.1. The prose contrasts vanilla SimCLR (label-free, augmentation views) against our SupCon-asymm (labels via `ignore_label`, layer-pair views) to motivate the supervised move.
  - **One-sentence role in our prose.** "Vanilla SimCLR (`chen2020simclr`), our label-free baseline, lands at AUROC ≈ 0.5 — establishing that label-free cross-layer contrastive does not surface the hallucination axis."

- **`azaria2023internal`** — Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying," Findings of EMNLP 2023.
  - **What they did.** SAPLMA: feed-forward MLP probe on a single layer's hidden state, trained directly on binary truth labels.
  - **Why we cite in §3.** Anchors the "no SupCon, no recon" cell of the §3.4 2×2 matrix. Architecturally the supervised-MLP-on-single-layer alternative to our method; §3.4 attribution argument depends on the SAPLMA reference point.
  - **One-sentence role in our prose.** "SAPLMA (`azaria2023internal`) is the single-layer supervised-MLP probe corresponding to the {no SupCon, no recon} cell of our attribution matrix."

- **`kossen2024semantic`** — Kossen et al., "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs," arXiv:2406.15927, 2024.
  - **What they did.** Linear probe on a single forward pass's activations to predict the length-normalized semantic-entropy score; single-pass cost, multi-sample signal.
  - **Why we cite in §3.** §3.3's logprob-recon discussion explicitly distinguishes our per-token logprob regression from SEP's sequence-level SE regression. Without the back-reference the reader may conflate the two; with it, the distinction is sharp.
  - **One-sentence role in our prose.** "Our logprob-recon is conceptually adjacent to SEP (`kossen2024semantic`) — both add a regression target causally coupled to hallucination — but ours targets per-token logprobs on a learned representation, not a single SE scalar on a linear probe."

- **`li2023iti`** and **`marks2024geometry`** — cited in §2.1; §3 back-references them once when motivating the mid-to-late residual-stream extraction band in §3.3.
  - **One-sentence role in our prose.** "We extract from a mid-to-late residual-stream band, motivated by causal and geometric evidence that truth-relevant structure concentrates there (`li2023iti`, `marks2024geometry`; see §2.1)."

---

## Pending-approval candidates — NOT in references.bib

Live candidates the writing agent may want to cite in §3. **Do not add to `.bib` without human verification** (per `paper/references.bib` policy header and `CLAUDE.md`). Each entry below states what the candidate would buy us and what the human should verify before approval.

- **(CONFIRMED IN `.bib` 2026-05-20)** Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020 (arXiv:2004.11362). Bibkey: `khosla2020supcon`.
  - *What they did.* Supervised extension of InfoNCE: for an anchor of class `c`, positives are all same-class samples in the batch (across views), negatives are all other-class samples. Showed it improves over cross-entropy and over self-supervised SimCLR on image classification.
  - *Would buy.* The actual loss we run is the asymmetric `ignore_label` variant of SupCon. §3.3 cannot be drafted at full fidelity without naming SupCon as the loss family; §3.2's "label-relevant MI" specialization explicitly invokes the SupCon-MI bound result. Without this citation, the prose has to hand-wave the loss as "a supervised variant of InfoNCE" — possible, but reviewer-bait.
  - *Verify before adding to `.bib`.* (a) arXiv ID 2004.11362 + NeurIPS 2020 venue. (b) Full author list (~10 authors, alphabetical after Khosla — verify exact order). (c) Whether the §3 Wang/Liu-style MI lens we appeal to is in the SupCon paper itself or only in follow-up work.

- **(CONFIRMED IN `.bib` 2026-05-20)** Poole et al., "On Variational Bounds of Mutual Information," ICML 2019 (arXiv:1905.06922). Bibkey: `poole2019variational`.
  - *What they did.* Surveyed and analyzed variational lower bounds on mutual information, including InfoNCE; characterized the tightness of each bound and the failure modes of looser variants.
  - *Would buy.* Lets §3.2 cite the tight bound directly rather than the original CPC paper, which states the bound but does not analyze it in the form the §3 argument needs. A reviewer who wants the "what exactly is InfoNCE bounding" detail will be satisfied by `poole2019variational` and unsatisfied by `oord2018cpc` alone.
  - *Verify before adding to `.bib`.* (a) arXiv ID 2105.06922 vs. 1905.06922 — pull the correct one; the proceedings version is ICML 2019. (b) Full author list (Poole, Ozair, van den Oord, Alemi, Tucker — verify exact order). (c) Whether the specific bound result we invoke is Eq. 5 (InfoNCE-MI) or a different equation — pin down before final draft so the prose cites the right equation.

- **(CONFIRMED IN `.bib` 2026-05-20)** Wang & Liu, "Understanding the Behaviour of Contrastive Loss," CVPR 2021 (arXiv:2012.09740). Bibkey: `wang2021understanding`.
  - *What they did.* Decomposed contrastive losses into alignment (positives close) + uniformity (negatives spread on a hypersphere) terms; showed both are necessary and gave a geometric account of what the loss optimizes.
  - *Would buy.* §3.2 can cite this for the geometric story of *what* the supervised contrastive loss is doing to the learned representation, complementing the MI-bound story. Without it, the prose has to choose: either go fully MI-theoretic (austere; one citation only) or hand-wave geometrically (reviewer-bait).
  - *Verify before adding to `.bib`.* (a) arXiv ID 2012.09740 + CVPR 2021 venue. (b) Authors: Tongzhou Wang, Phillip Isola — but the SupCon-relevant Wang & Liu paper might be a different one (Wang and Liu have multiple contrastive-learning papers in 2020–2022). **Critical to verify the right Wang & Liu paper** before adding to `.bib`; recommend the writing agent flag the candidate by arXiv ID and let the human approve the exact ID.

- **(CONFIRMED IN `.bib` 2026-05-20 — optional; only if §3.2 has room)** Tishby & Zaslavsky, "Deep Learning and the Information Bottleneck Principle," IEEE ITW 2015 (arXiv:1503.02406). Bibkey: `tishby2015deep`.
  - *Would buy.* One-citation anchor for the broader information-bottleneck framing of "what does a neural network preserve across layers." Useful background but not load-bearing.
  - *Verify before adding to `.bib`.* Only worth adding if §3.2 prose has the budget for a one-sentence broader-context citation. Recommend omit on first draft; add only if a reviewer flags §3.2 as "info-theoretic argument without info-theory background."

- **(CONFIRMED IN `.bib` 2026-05-20 — optional; defer until §2.3 framing-level decision is fully resolved)** Hjelm et al., "Learning Deep Representations by Mutual Information Estimation and Maximization" (Deep InfoMax / DIM), ICLR 2019 (arXiv:1808.06670). Bibkey: `hjelm2019deepinfomax`.
  - *Would buy.* Citation for "the broader family of methods that maximize MI between different representations of the same input." Adjacent precedent the §2.3 broad-novelty footnote could also reach for, if the broad claim survives the literature pass.
  - *Verify before adding to `.bib`.* Cross-check with §2.3's adjacent-machinery footnote (CRD / CoDIR / CDS); if Hjelm et al. is judged closer than those, it might want to land in §2.3 rather than §3.2. Recommend defer until §2.3 framing-level decision is fully resolved.

---

## Drafting order recommendation

For a writing agent starting fresh on §3 prose:

1. **Verify the three required pending citations** (`khosla2020supcon`, `poole2019variational`, `wang2021understanding`) and route to human approval. §3.2 cannot draft cleanly without at least the first two.
2. **Locate Figure 1's source data** — the training-loss curve from a representative full-method run. ~30 min of log-grepping.
3. **Draft §3.1** (problem setup) — straightforward, no pending dependencies.
4. **Draft §3.3** (what we built) before §3.2 (information-theoretic argument). The architecture is concrete; drafting it first means §3.2 can reference the concrete loss form when stating the bound, rather than abstractly. This reverses the natural reading order but is the cleaner drafting order.
5. **Draft §3.2** after §3.3, with Figure 1 in hand. The trainability punchline is much sharper when Figure 1 sits below the paragraph that names it.
6. **Defer §3.4** until #66 and #67 land. Until then, §3.4 is the *menu*; pre-write the four outcome-conditional paragraphs but do not select.
7. **Close §3 by writing the §3.1 → §3.2 → §3.3 → §3.4 transitions last.** Each subsection should hand off cleanly to the next; this requires reading the section end-to-end after each subsection lands.
