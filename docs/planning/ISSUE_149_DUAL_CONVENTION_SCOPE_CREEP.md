# Issue #149 scope expansion: dual-convention contrastive classifier

This work stays on the issue #149 prefix-view PR by request, but it is a
deliberate expansion beyond the original matched early-detection experiment.
The original scope compared prefix-trained HalluLens and ACT-ViT. The added arm
tests whether the two asymmetric label conventions contain complementary
information that a supervised decision head can combine.

## Architecture

For the same input views, one module runs two independent copies of the current
`LogprobReconProgressiveCompressor`:

- standard branch: 512-d representation, SupCon with `ignore_label=1`;
- mirrored branch: 512-d representation, SupCon with `ignore_label=0`;
- classifier: mean each branch across views, optionally L2-normalize each
  512-d vector, concatenate them, then apply `1024 -> 128 -> 1` with GELU and
  dropout.

The joint loss is

`L = L_std + L_mirror + lambda_recon * (L_recon_std + L_recon_mirror) + lambda_cls * BCE(y_hat, y)`.

All parameters are updated by one optimizer and one backward pass. This is not
the older sequential twin procedure and neither encoder is frozen.

At `input_dim=4096` and `final_dim=512`, the existing encoder has 77,538,112
parameters. The new model has two copies plus 131,329 classifier parameters,
for 155,207,553 parameters total (2.0017x). This arm is therefore intentionally
not parameter-matched to the single-encoder headline method.

## Evaluation and paper interpretation

The primary score is the binary classifier probability, reported with AUROC
and AUPRC. KNN AUROC is also reported independently on the standard and
mirrored 512-d representations, including the issue #149 prefix grid
`k in {1, 4, 8, 16, 32, 48, 64}`. Score-level KNN fusion remains a diagnostic,
not the primary output of this architecture.

For a hypothetical paper, this is best framed as a capacity-rich
complementarity test: does explicitly modeling both contrastive conventions
improve early hallucination detection? Any gain must be shown alongside the
single-standard, single-mirrored, and existing sequential/shared variants, and
described with the approximately doubled parameter and compute budget. A
positive result alone would not establish that the label convention is the
cause unless those ablations agree.

MMLU remains outside the HalluLens benchmark suite. The configured datasets are
HotpotQA, NQ, PopQA, SciQ, and SearchQA.

## Dispatch contract

The full sweep is a filesystem cell queue at
`shared/issue_149_dual_convention_dispatch`, built by
`scripts/dispatch/build_issue_149_dual_convention_cells.py`. Its 25 cells are
the Cartesian product of five datasets and five seeds. Each cell runs one seed
and requires `predictions.csv` as its terminal sentinel. Workers use
`scripts/dispatch/worker_149_dual_convention.sh`, which delegates to the tested
issue #149 claim/heartbeat/recovery worker with the isolated queue root.
