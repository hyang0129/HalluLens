# Figure Candidates — EMNLP 2026 Submission

Working list of figure candidates for the 8-page main paper. Not a commitment — the final cut depends on which numbers freeze cleanly in week 3.

Sources: [outline.md](outline.md) §3, §5–§7; roadmap §8 figure list (external).

---

## Recommended cut (3 figures)

These map one-to-one onto the three contribution bullets in [outline.md](outline.md) §1.

### Figure 1 — Method / architecture diagram
- **Section:** §3 (Method). Already reserved as "Figure 1 source" in [outline.md](outline.md) file map.
- **Content:** `ProgressiveCompressor` schematic — layer-pair views (mid-to-late residual-stream layers), asymmetric SupCon over pair embeddings, logprob-recon auxiliary head, KNN scorer at inference.
- **Why:** Comprehension prereq. Readers cannot evaluate the headline result or the layer-pair claim without seeing the architecture.
- **Source assets:** None yet — needs original schematic. Block on this; without it §3 prose has nothing to anchor to.

### Figure 2 — Compute-matched comparison
- **Section:** §5.3 (Main Results).
- **Content:** AUROC vs. forward-pass count. K=1 cluster: ours, linear probe, SAPLMA, SEP-SE, P(true). K=10 cluster: SE (length-normalized), SelfCheckGPT-NLI. One panel per free-form dataset (HotpotQA, NQ, PopQA, SciQ, SearchQA). MMLU shows K=1 cluster only.
- **Why:** Defends contribution 1 — "beats output-space scalars, single-layer linear probe, SAPLMA, and sampling-based methods at matched compute." K=1-vs-K=10 framing is the load-bearing comparison.
- **Subsumes:** §5.2 headline bar chart (same K=1 data). Roll that into a table row instead.
- **Source assets:** Numbers freeze in week 3. Slot reserved, plot script pending.

### Figure 3 — Cross-dataset transfer heatmap
- **Section:** §6 (Cross-Dataset Transfer).
- **Content:** 6×6 source→target AUROC heatmap, one panel per model. Off-diagonal mean + worst-case annotated in body text.
- **Why:** Defends contribution 2 — "transfers across datasets without retraining."
- **Source assets:** Transfer table not yet started (roadmap §2 must-add item 5). Pure analysis — no new GPU.

---

## On the bubble (would be Figure 4 if budget allowed)

### Layer-pair sensitivity heatmap
- **Section:** §7.3 (Ablations) / connects to §3.2 theory.
- **Content:** Sweep early/mid/late layer pairs on 1–2 datasets; heatmap of AUROC over pair index.
- **Why considered:** Only candidate that visualizes contribution 3 — the cross-layer-coherence prediction. Currently no figure in the cut-of-3 defends this bullet.
- **Why not in top 3:** Contribution 3 is the most theoretical of the three claims; the outline already flags §3.4 (attribution table) as the first thing to collapse under page pressure ([outline.md](outline.md) §3 budget note). Layer-pair concentration can ride with the theory in the appendix without losing the claim.
- **Swap criterion:** If transfer (§6) numbers come in weak, promote this in place of Figure 3 and reframe contribution 2 around layer-pair concentration instead.
- **Alt placement:** Shrink into a side panel of Figure 1 if a clean low-real-estate version exists.

---

## Cut (table-only or appendix)

### Headline AUROC bars (outline §5.2)
- Same K=1 data as Figure 2 — redundant. Demote to a row in the main results table.

### Calibration / reliability diagrams (outline §5.4)
- ECE belongs in the main table. Reliability diagrams are informative but lower-priority for an 8-page paper; push to appendix.

### Loss decomposition ablation (outline §7.1)
- Three bars (SimCLR-only, recon-only, full) is a table, not a figure.

### Scorer choice ablation (outline §7.4)
- Three bars (cosine, Mahalanobis, KNN) is a table, not a figure.

---

## Production order

1. **Figure 1 (method diagram)** — block on this; needed for §3 prose. No data dependency.
2. **Figure 3 (transfer heatmap)** — pure CPU recompute on existing checkpoints. Can be produced before §5 numbers freeze.
3. **Figure 2 (compute-matched)** — last; waits on the §5 numbers freeze (roadmap week 3).

---

## Open questions

1. **Two panels per figure or stacked?** Figure 2 has 5–6 dataset panels × 2 models. Decide: 2×3 grid per model (two figures) vs. one wide figure with model as line style. Latter is denser but harder to read in the 8-page format.
2. **Does Figure 2 need error bars or just point estimates?** Roadmap commits to bootstrap 95% CIs in the main table; figure could either show them as shaded bands or defer to the table.
3. **Figure 1 — schematic style.** Box-and-arrow (standard) vs. tensor-flow style (more informative, more space). Decide after §3 prose lands.
