# Token-wise standard-SupCon view × reconstruction factorial

## Question

The existing token-wise results do not cleanly identify whether performance
comes from later-token positive views or from the training objective. The v1
model uses ordinary supervised contrastive learning with a token-zero/later-
token pair, while the earlier token-zero dropout control also changed to the
causal-control loss. This experiment removes that objective confound.

## Predeclared design

All four arms use the original v1 progressive compressor and decoder
(77,538,112 trainable parameters), legacy/standard SupCon, the same optimizer,
dropout, batch size, 3,000-step minimum, validation-KNN checkpoint selection,
and final evaluation. Pair training is restricted to responses with at least
two captured tokens in every arm. Evaluation remains at token zero over all
eligible examples.

| Training views | Full-response reconstruction | Configuration |
|---|---:|---|
| t0 + random same-response tn | yes | Existing `tokenwise_contrastive_first_anchored` baseline |
| t0 + t0, independent model dropout | yes | `tokenwise_supcon_t0_full_recon` |
| t0 + random same-response tn | no | `tokenwise_supcon_tn_no_recon` |
| t0 + t0, independent model dropout | no | `tokenwise_supcon_t0_no_recon` |

The no-reconstruction arms retain the decoder and therefore retain the exact
parameter count; only `recon_lambda` changes from 1 to 0. The t0+t0 arms retain
the v1 input dropout, so their two identical activation inputs receive
independent stochastic model views during training.

## Matrix and analysis

The existing baseline supplies five results. The builder adds exactly 15 new
highest-priority cells: three missing arms × HotpotQA, Natural Questions,
PopQA, SciQ, and SearchQA × training seed 0. MMLU is excluded.

The primary statistic is the unweighted five-dataset macro KNN AUROC, with
per-dataset results retained. Interpret the two factorial effects and their
interaction before doing further recipe tuning:

- later-token view effect at reconstruction=1 and reconstruction=0;
- reconstruction effect for t0+tn and t0+t0 views;
- whether reconstruction and view construction interact.

This is a one-seed confound-resolution experiment, not the final variance
estimate. Any promoted recipe requires a matched multi-seed confirmation.

## Queueing

The builder is idempotent across pending, claimed, done, failed, and cancelled
queue states, and only writes cell files. Existing generic workers remain cell
agnostic and can consume the cells without being restarted.

```bash
python scripts/dispatch/build_tokenwise_supcon_factorial_cells.py \
  --dispatch-root shared/issue_151_knnval_rerun_dispatch
```

Token-wise runs with embedding dumping enabled now emit separate train,
validation, and test artifacts. Each split has embeddings, labels, raw prompt
hashes, deterministic per-row stable IDs, and provenance metadata. Validation
labels are resolved only against the held-out validation rows from the
training capture; test labels are not used to create the validation artifact.
These dumps support a later frozen-scoring study without adding scorers here.
