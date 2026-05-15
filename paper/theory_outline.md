# Theory — Outline

Structural skeleton for the theoretical contribution. Headings + intent notes, not prose. Each subsection states the claim it makes and the evidence the prose will lean on.

**Open structural choice (carried from `outline.md`):** drafted as a standalone §3 Theory section preceding the Method. If folded into Method as §3.1, drop §3.1, §3.2, §3.8 below and compress §3.3–§3.6 into ~3 paragraphs.

**Status note (2026-05-15 — load-bearing for this section):** the contrastive loss in the method is **supervised** contrastive (SupCon, Khosla et al. 2020) with an **asymmetric one-class scheme**, *not* vanilla SimCLR. Specifically [contrastive_logprob_recon.json](../configs/methods/contrastive_logprob_recon.json) sets `use_labels: true, ignore_label: 1`, which in [training.py:206-272](../activation_research/training.py#L206-L272) means: class 0 (truthful answers) form positives with other class-0 samples in the batch; class 1 (hallucinations) are positives only with their own layer-views and repel everything else, including other hallucinations. This is structurally a representation-learning form of one-class anomaly detection, not a self-supervised contrastive method. The theory needs to be written about *this* loss, not about vanilla SimCLR.

---

## 3.1 Setup & Notation

- LLM `M`, prompt `x`, generation `y`. The empirical work uses greedy decoding; the theory makes no assumption on the decoding procedure as long as `(x, y)` is a fixed realized pair.
- Intermediate states `h_ℓ(x, y) ∈ ℝ^d` at layer `ℓ ∈ [1..L]`. Last-token position.
- Hallucination label `y_label ∈ {0, 1}` produced by the benchmark's evaluator over `(x, y)`. **The label is used during training** — it participates in the contrastive loss, not just downstream evaluation.
- Goal: a detector `s: (x, y) → ℝ` correlated with `y_label`.
- Determinism note: `h_ℓ` is deterministic given `(x, y)`; `y_label` may have evaluator noise.

## 3.2 Supervised Contrastive Learning as a Variational Lower Bound on Mutual Information

This subsection establishes what loss we are actually optimizing.

- **The objective.** SupCon with asymmetric `ignore_label`. Positive pairs for a class-0 anchor: all class-0 samples in the batch (across all views) + the anchor's own different views. Positive pairs for a class-1 anchor: only the anchor's own different views. Negatives for both: every other anchor-pair across the batch.
- **The MI bound.** Standard InfoNCE-as-MI result (van den Oord et al. 2018; Poole et al. 2019) gives `-L_InfoNCE ≤ I(view_a; view_b) - log K` for the symmetric case. Under SupCon (Khosla et al. 2020) the optimized lower bound becomes a bound on `I(features; class label)` — see Khosla et al. §3 and Wang & Liu 2021 for the explicit form. For our asymmetric scheme, this further specializes to a one-sided bound: the loss pulls class-0 features into a coherent cluster (maximizing `I(z; y_label = 0)`-style alignment) while leaving class-1 features only constrained to be view-consistent with themselves.
- **What this means in plain terms.** The contrastive loss is doing two things at once: (a) representation learning via cross-layer view alignment, and (b) classifier-like learning via label-driven cluster formation for the truthful class. The label is part of the optimization, not an evaluation afterthought.

## 3.3 Mutual Information Preservation in Residual Networks

Claim: across mid-to-late residual-stream layers, `I(h_{ℓ_a}; h_{ℓ_b}) > 0` for typical inputs. (This claim is *independent* of whether the contrastive loss is supervised or unsupervised — it's an architectural property of the model.)

- **Architectural reason.** Both layers are deterministic functions of the input. Residual connections explicitly preserve information from earlier layers in later ones. MI being positive is structural, not surprising.
- **Empirical reason.** Whether the contrastive loss is supervised or unsupervised, it is trainable on our data — loss curves descend, embeddings cluster meaningfully. *Include training-loss curve as Figure [theory-1].* This trainability is the simplest possible empirical confirmation of cross-layer MI: if `I(h_{ℓ_a}; h_{ℓ_b}) = 0`, no InfoNCE variant could descend.
- **Why it matters for the method.** MI preservation is *necessary* for any cross-layer contrastive scheme to learn at all. If we lacked it, our method would not exist as a viable construction. This is the **feasibility** layer of the argument.

## 3.4 Feasibility vs. Mechanism

Two distinct theoretical claims should not be conflated.

- **Feasibility (§3.3, established).** Cross-layer MI is preserved; contrastive learning across layer pairs is trainable. This is structural and architecture-driven.
- **Mechanism (what actually produces the detector).** The label-relevant subspace of the preserved MI is what carries the hallucination signal. This subspace is selected by **two coupled supervision signals** in our method: (i) the asymmetric label-driven positives/negatives in SupCon, (ii) the logprob-recon auxiliary loss.

- **Empirical observation worth flagging honestly:** vanilla unsupervised SimCLR (no labels, no recon) on the same data → AUROC ≈ 0.5. This is the existing "SimCLR-only" run with `use_labels: false`. It establishes that pure self-supervised cross-layer contrastive — with no label or logprob signal — does not surface the hallucination axis. **It does *not* establish that supervised contrastive alone is insufficient**, because that's a different loss family. The status of supervised-contrastive-alone is being resolved empirically by [#66](https://github.com/hyang0129/HalluLens/issues/66) (pending).

## 3.5 Logprob Reconstruction as a Secondary Supervision Signal

- The recon loss `L_recon = ‖g(z) − logprob_M(y_t | y_{<t}, x)‖²`, summed over token positions `t`, where `z = f_θ(h_ℓ)` is the compressed feature and `g` is a small head ([model.py:172-176](../activation_research/model.py#L172-L176)).
- Why logprobs are a reasonable auxiliary target for *hallucination*:
  - Causal link: when `M` is about to hallucinate, the next-token distribution typically broadens (top-token probability drops) at the moment of fabrication. The logprob signal carries this directly.
  - Therefore `corr(logprob_M(y_t); y_label) > 0`. The training target is causally connected to the quantity we care about.
- The recon loss applies additional training pressure on `f_θ` to retain features that suffice to predict per-token logprobs. By the previous paragraph, those features are correlated with the label.
- **Important caveat for the framing:** under our supervised contrastive setup, the label is *already* part of the primary loss. Recon is therefore not the *primary* subspace-selection mechanism — the labels are. Recon is a complementary regression-based supervision. How much value it adds on top of supervised contrastive is the empirical question that [#66](https://github.com/hyang0129/HalluLens/issues/66) tests.

## 3.6 The Information-Theoretic Argument for the Method's Contribution

The cleanest contribution argument requires the pending ablations to land. This subsection sketches the argument template; the specific numbers and final framing depend on the outcomes of [#66](https://github.com/hyang0129/HalluLens/issues/66) and [#67](https://github.com/hyang0129/HalluLens/issues/67). State up front in the prose that the argument is empirical-pending.

The 2×2 ablation we will present:

|                        | **No recon**                  | **With recon**          |
|------------------------|-------------------------------|-------------------------|
| **No supervised contrastive** | SAPLMA (existing baseline) | **#67 SAPLMA + recon** (pending) |
| **With supervised contrastive** | **#66 sup-contrastive only** (pending) | **Full method** (existing, headline) |

Plus a row for "neither contrastive nor recon, no label-driven structural learning beyond a single linear classifier": linear probe (already in grid). And a footnote row for vanilla SimCLR (`use_labels: false`) → AUROC ≈ 0.5, demonstrating that label-free cross-layer contrastive fails (corroborating §3.4).

**Outcome-conditional framing** — the §3 prose will lock to one of these once #66 and #67 land:

- **Outcome 1: #66 < full ≈ #67.** Both supervision channels useful in isolation; combination is at most modestly synergistic. Frame as: "supervised contrastive and recon each carry signal; the contrastive structure is the architectural innovation we contribute, and recon is a general-purpose auxiliary that benefits any probe of comparable capacity."
- **Outcome 2: #66 ≈ full > #67.** Recon is decorative on top of supervised contrastive; SAPLMA cannot absorb the recon trick. Frame as: "the contrastive structure is the load-bearing innovation; recon is a useful but ultimately redundant auxiliary in the presence of cross-layer supervised contrastive learning."
- **Outcome 3: #66 < full ≈ #67.** *(Same as Outcome 1 but with a stronger interpretation: recon is the load-bearing piece, SAPLMA-with-recon matches us.)* Frame honestly: "the headline contribution is the logprob-recon auxiliary trick, which generalizes across probe architectures. The supervised contrastive structure adds little. SEP becomes our closest comparison."
- **Outcome 4: #66 < full and #67 < full.** Best outcome: both ingredients contribute independently. Frame as: "the combination of supervised contrastive on layer pairs + logprob recon is genuinely irreducible; neither component alone reaches the full method."

The §3 prose can be drafted in skeleton form before the ablations land, but the **headline claim of §3.6 should not be locked until #66 and #67 results are in.** Until then, the section can present the 2×2 setup and the *menu* of conclusions, with one to be selected when data arrives.

## 3.7 What the Theory Predicts and What It Does Not

State predictions with current evidentiary status:

- ✅ **P1 (empirical, confirmed).** Full method > logprob baseline by a margin beyond DPI ceiling slack — i.e., the method clears the ceiling on any signal purely recoverable from logprobs. (Established from main results.)
- ✅ **P2 (empirical, confirmed, but weaker than originally framed).** Vanilla unsupervised SimCLR alone ≈ random AUROC. This rules out *self-supervised* cross-layer contrastive as a hallucination detector but does *not* directly rule out supervised contrastive alone. [#66](https://github.com/hyang0129/HalluLens/issues/66) tests the stronger version.
- 🟡 **P3 (pending #66).** Supervised contrastive only (no recon) is sufficient to clear the linear-probe baseline. If false (#66 ≈ linear probe), the contrastive structure is doing less work than the architecture diagram implies, and recon is doing most of the work.
- 🟡 **P4 (pending #67).** SAPLMA + recon does not match the full method. If false (#67 ≈ full method), the contrastive structure is decorative and the paper's headline framing must pivot.
- 🟡 **P5 (pending layer-pair ablation, roadmap §4 item 8).** Layer-pair sensitivity sweep shows mid-to-late concentration. This is an empirical structural finding, not theory-predicted under the MI-preservation framing.
- 🟡 **P6 (pending transfer table, [#62](https://github.com/hyang0129/HalluLens/issues/62)).** Cross-dataset transfer is better for our method than for a direct linear probe.
- ❌ **Not a claim.** The theory does *not* predict layer-pair-specific advantages over other contrastive view constructions (e.g., same-layer + dropout). Establishing that would require additional ablations we have not run and are not committing to.
- ❌ **Not a claim.** "Cross-layer coherence" as the load-bearing mechanism. Demoted from the original roadmap §1 claim to a falsifiable empirical question subsumed by P3 and P5.

## 3.8 Relation to Prior Theoretical Framings

One paragraph each (not three pages):

- **SAPLMA (Azaria & Mitchell 2023).** Direct BCE supervision on binary labels via an MLP on single-layer activations. No auxiliary target, no contrastive structure. Our SAPLMA+recon ablation (#67) tests how much of our gain over SAPLMA is attributable to the recon trick alone.
- **SEP (Kossen et al. 2024).** Linear probe trained on length-normalized semantic entropy. Same conceptual move as our recon loss — replace direct classification with a regression target that is causally coupled to hallucination — but uses a different signal (SE from K=10 samples) and a simpler architecture. We are a "deep probe with auxiliary causal regression" on a different target (per-token logprobs) with additional supervised contrastive structure layered on.
- **SupCon (Khosla et al. 2020).** The contrastive loss we use, with our novel asymmetric `ignore_label` adaptation that recasts it as one-class anomaly detection (truthful as the coherent class, hallucinations as instances). We contribute the specific application to layer-pair views in residual-stream LLMs, not a new contrastive loss.
- **InfoNCE / SimCLR (van den Oord et al. 2018; Chen et al. 2020).** Theoretical underpinning of the MI bound. Cited for the bound, not as a direct method ancestor — our setting is supervised, not self-supervised.

---

## What this outline commits to

- ✅ **MI preservation as the feasibility argument** — testable from the trainability of any contrastive variant.
- ✅ **Supervised contrastive (with asymmetric `ignore_label`) + logprob recon** as the method, with explicit acknowledgement that this is *supervised* learning, not self-supervised.
- ✅ **A 2×2 ablation structure** (§3.6) that will produce a clean attribution once #66 and #67 land.

## What this outline does NOT commit to

- ❌ A specific geometric picture of the joint representation. No evidence; speculating in §3 invites reviewer objections.
- ❌ The cross-layer-coherence framing from the original roadmap §1. Demoted to P5 (an empirical question to be addressed by the layer-pair sweep).
- ❌ The strong claim from the prior version of this doc that the "2×2 ceiling argument" can be made entirely from existing data without new runs. That claim leaned on the SimCLR-only ablation as a clean falsification, which it is not — vanilla SimCLR is the wrong contrastive flavor relative to the full method.
- ❌ Locking the §3.6 headline framing before #66 and #67 results land.

## Open dependencies before §3 prose can finalize

1. **[#66](https://github.com/hyang0129/HalluLens/issues/66)** — supervised contrastive without recon. Determines P3.
2. **[#67](https://github.com/hyang0129/HalluLens/issues/67)** — SAPLMA + recon auxiliary. Determines P4.
3. **Figure [theory-1]** — contrastive training-loss curve from a representative full-method run. Used as the empirical feasibility evidence in §3.3. No new compute needed; just locate logs from a completed run.

## Forward references this section makes

Every `§[...]` link below must resolve to a real anchor before submission:
- Figure [theory-1]: contrastive training loss curves as evidence of MI preservation.
- §7 ablation: #66, #67, vanilla SimCLR result, layer-pair sensitivity.
- §6 transfer: cross-dataset transfer (P6).
- §5 main results: full method vs. logprob baseline gap (P1).
