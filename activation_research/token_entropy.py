"""Token-entropy baseline for hallucination detection.

This module provides a **non-learned** hallucination detector that scores
each sample using token-level logprob statistics (mean logprob, min logprob,
and approximate Shannon entropy from top-K logprobs).

Usage
-----
The detector works with the same ``ActivationDataset`` /
``PreloadedActivationDataset`` used for contrastive and linear-probe
training — simply enable ``include_response_logprobs=True`` when
constructing the dataset.

    from activation_research.token_entropy import TokenEntropyDetector
    detector = TokenEntropyDetector(outlier_class=1)
    stats = detector.score(test_dataset, batch_size=256)
    print(stats["mean_logprob_auroc"])

The same preloaded dataset can be shared with the ``LinearProbeTrainer``
since the logprob fields are populated alongside activation views.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader
from utils.progress import tqdm

from .metrics import token_entropy_ood_stats


class TokenEntropyDetector:
    """Score a dataset using token-level logprob statistics.

    Parameters
    ----------
    outlier_class : int
        Which label (0 or 1) to treat as the hallucination / OOD class.
    """

    def __init__(self, outlier_class: int = 1):
        self.outlier_class = int(outlier_class)

    def score(
        self,
        dataset,
        *,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """Iterate over *dataset*, collect logprob fields, and compute AUROC.

        Parameters
        ----------
        dataset : Dataset
            Must have ``include_response_logprobs=True``.  Each sample should
            contain ``response_token_logprobs``, ``response_logprob_mask``,
            ``response_topk_logprobs``, and ``halu``.
        batch_size : int
            DataLoader batch size.
        num_workers : int
            DataLoader workers.

        Returns
        -------
        dict
            Output of :func:`token_entropy_ood_stats`.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        records = []
        for batch in tqdm(loader, desc="Token-entropy scoring"):
            B = batch["halu"].shape[0]
            for j in range(B):
                record: Dict[str, Any] = {
                    "halu": int(batch["halu"][j].item()),
                    "response_token_logprobs": batch["response_token_logprobs"][j],
                    "response_logprob_mask": batch["response_logprob_mask"][j],
                }
                if "response_topk_logprobs" in batch:
                    record["response_topk_logprobs"] = batch["response_topk_logprobs"][j]
                records.append(record)

        logger.info(f"Collected {len(records)} samples for token-entropy scoring")
        stats = token_entropy_ood_stats(records, outlier_class=self.outlier_class)
        logger.info(f"Token-entropy results: {stats}")
        return stats
