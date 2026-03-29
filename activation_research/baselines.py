"""Logprob baseline scorers for hallucination detection.

These are non-learned baselines: no training is needed. Per-sample scores are
computed directly from token-level logprobs stored alongside activations.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _safe_auroc(binary_labels, scores):
    """AUROC with safety for single-class edge case."""
    if len(set(binary_labels)) < 2:
        return float('nan')
    return float(roc_auc_score(binary_labels, scores))


def logprob_baseline_scores(records):
    """Compute per-sample logprob baseline scores.

    Parameters
    ----------
    records : list[dict]
        Each record must contain:
        - ``response_token_logprobs`` : Tensor or ndarray of shape (T,)
        - ``response_logprob_mask`` : Tensor or ndarray of shape (T,) bool

    Returns
    -------
    dict
        Keys: ``mean_logprob``, ``seq_logprob``, ``perplexity``
        (each an ndarray of shape (N,)).
    """
    token_logprobs_list = []
    masks_list = []
    for r in records:
        lp = r["response_token_logprobs"]
        if isinstance(lp, torch.Tensor):
            lp = lp.numpy()
        mask = r["response_logprob_mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy().astype(bool)
        token_logprobs_list.append(lp.astype(np.float64))
        masks_list.append(mask)

    token_logprobs = np.stack(token_logprobs_list)  # (N, T)
    masks = np.stack(masks_list)                     # (N, T)

    # Replace padding with 0 for summation
    masked_lp = np.where(masks, token_logprobs, 0.0)
    counts = masks.sum(axis=-1).clip(min=1)

    mean_lp = (masked_lp.sum(axis=-1) / counts).astype(np.float32)
    seq_lp = masked_lp.sum(axis=-1).astype(np.float32)
    perplexity = np.exp(-mean_lp).astype(np.float32)

    return {
        "mean_logprob": mean_lp,
        "seq_logprob": seq_lp,
        "perplexity": perplexity,
    }


def logprob_baseline_auroc(records, outlier_class=1):
    """Compute AUROC for logprob baseline methods.

    Parameters
    ----------
    records : list[dict]
        Each must have ``response_token_logprobs``, ``response_logprob_mask``,
        and ``halu`` fields.
    outlier_class : int
        Which label is the outlier (hallucination).

    Returns
    -------
    dict
        AUROC per scoring method: ``mean_logprob_auroc``,
        ``seq_logprob_auroc``, ``perplexity_auroc``.
    """
    labels = np.array([int(r["halu"]) for r in records])
    binary_labels = (labels == outlier_class).astype(np.int32)

    scores = logprob_baseline_scores(records)

    # Higher mean_logprob = more confident = less hallucination -> negate for OOD scoring
    # Higher seq_logprob = more confident -> negate
    # Higher perplexity = less confident -> use directly as OOD score
    return {
        "mean_logprob_auroc": _safe_auroc(binary_labels, -scores["mean_logprob"]),
        "seq_logprob_auroc": _safe_auroc(binary_labels, -scores["seq_logprob"]),
        "perplexity_auroc": _safe_auroc(binary_labels, scores["perplexity"]),
    }
